const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

pub const Embeddings = struct {
    allocator: std.mem.Allocator,
    token_embeddings: Matrix,
    positional_embeddings: Matrix,
    has_cached_input: bool,
    cached_input: Matrix,
    token_optimizer: Adam,
    positional_optimizer: Adam,
    batch_size: usize,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize) !*Embeddings {
        const self = try allocator.create(Embeddings);
        self.* = .{
            .allocator = allocator,
            .token_embeddings = try Matrix.initRandom(allocator, vocab_size, lib.config.embedding_dim, 0.0, 0.02),
            .positional_embeddings = try Matrix.initRandom(allocator, lib.config.max_seq_len, lib.config.embedding_dim, 0.0, 0.02),
            .has_cached_input = false,
            .cached_input = undefined,
            .token_optimizer = try Adam.init(allocator, vocab_size, lib.config.embedding_dim),
            .positional_optimizer = try Adam.init(allocator, lib.config.max_seq_len, lib.config.embedding_dim),
            .batch_size = 1,
        };
        return self;
    }

    pub fn deinit(self: *Embeddings) void {
        self.token_embeddings.deinit();
        self.positional_embeddings.deinit();
        if (self.has_cached_input) self.cached_input.deinit();
        self.token_optimizer.deinit();
        self.positional_optimizer.deinit();
        self.allocator.destroy(self);
    }

    pub fn setBatchSize(self: *Embeddings, batch_size: usize) void {
        self.batch_size = batch_size;
    }

    pub fn forward(self: *Embeddings, input: Matrix) !Matrix {
        if (self.has_cached_input) self.cached_input.deinit();
        self.cached_input = try input.clone();
        self.has_cached_input = true;

        const total_tokens = input.cols;
        if (total_tokens % self.batch_size != 0) return error.InputSizeMismatch;
        const seq_len = total_tokens / self.batch_size;

        var token_embeds = try Matrix.init(self.allocator, total_tokens, lib.config.embedding_dim);
        for (input.data, 0..) |token_id, i| {
            const id: usize = @intFromFloat(token_id);
            const row = self.token_embeddings.data[id * self.token_embeddings.cols .. (id + 1) * self.token_embeddings.cols];
            @memcpy(token_embeds.data[i * token_embeds.cols .. (i + 1) * token_embeds.cols], row);
        }

        var pos_embeds = try Matrix.initZeros(self.allocator, total_tokens, lib.config.embedding_dim);
        const clamped_seq_len = @min(seq_len, lib.config.max_seq_len);
        const pos_data = self.positional_embeddings.data[0 .. clamped_seq_len * lib.config.embedding_dim];

        for (0..self.batch_size) |b| {
            const start_row = b * seq_len;
            const dest_slice = pos_embeds.data[start_row * lib.config.embedding_dim .. (start_row + clamped_seq_len) * lib.config.embedding_dim];
            @memcpy(dest_slice, pos_data);
        }

        const result = try token_embeds.add(&pos_embeds);
        token_embeds.deinit();
        pos_embeds.deinit();
        return result;
    }

    pub fn backward(self: *Embeddings, grads: Matrix, lr: f32) !Matrix {
        var mut_grads = grads;
        defer mut_grads.deinit();
        var token_grads = try Matrix.initZeros(self.allocator, self.token_embeddings.rows, self.token_embeddings.cols);
        defer token_grads.deinit();
        var pos_grads = try Matrix.initZeros(self.allocator, self.positional_embeddings.rows, self.positional_embeddings.cols);
        defer pos_grads.deinit();

        const total_tokens = grads.rows;
        const seq_len = total_tokens / self.batch_size;

        for (self.cached_input.data, 0..) |token_id, i| {
            const id: usize = @intFromFloat(token_id);
            const grad_row = mut_grads.data[i * mut_grads.cols .. (i + 1) * mut_grads.cols];

            const token_grad_row = token_grads.data[id * token_grads.cols .. (id + 1) * token_grads.cols];
            for (token_grad_row, grad_row) |*t, g| t.* += g;

            const seq_index = i % seq_len;
            if (seq_index < self.positional_embeddings.rows) {
                const pos_grad_row = pos_grads.data[seq_index * pos_grads.cols .. (seq_index + 1) * pos_grads.cols];
                for (pos_grad_row, grad_row) |*p, g| p.* += g;
            }
        }

        self.token_optimizer.step(&self.token_embeddings, token_grads, lr);
        self.positional_optimizer.step(&self.positional_embeddings, pos_grads, lr);

        return try Matrix.initZeros(self.allocator, 1, 1);
    }

    pub fn parameters(self: *const Embeddings) usize {
        return self.token_embeddings.data.len + self.positional_embeddings.data.len;
    }

    pub fn toLayer(self: *Embeddings) layer.Layer {
        return layer.toLayer(Embeddings)(self);
    }

    pub fn save(self: *const Embeddings, writer: anytype) !void {
        try self.token_embeddings.save(writer);
        try self.positional_embeddings.save(writer);
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !*Embeddings {
        const self = try allocator.create(Embeddings);
        errdefer allocator.destroy(self);

        var token_embeddings = try Matrix.load(allocator, reader);
        errdefer token_embeddings.deinit();

        var positional_embeddings = try Matrix.load(allocator, reader);
        errdefer positional_embeddings.deinit();

        self.* = .{
            .allocator = allocator,
            .token_embeddings = token_embeddings,
            .positional_embeddings = positional_embeddings,
            .has_cached_input = false,
            .cached_input = undefined,
            .token_optimizer = try Adam.init(allocator, token_embeddings.rows, token_embeddings.cols),
            .positional_optimizer = try Adam.init(allocator, positional_embeddings.rows, positional_embeddings.cols),
            .batch_size = 1,
        };
        return self;
    }
};

test "Embeddings Batching" {
    const allocator = std.testing.allocator;
    const vocab_size = 100;
    var embeddings = try Embeddings.init(allocator, vocab_size);
    defer embeddings.deinit();

    // Set batch size to 2
    embeddings.setBatchSize(2);

    // Create input with 2 sequences of length 50. Total 100 tokens.
    // max_seq_len is 80.
    const seq_len = 50;
    const batch_size = 2;
    const total_tokens = seq_len * batch_size;

    var input = try Matrix.init(allocator, 1, total_tokens);
    defer input.deinit();
    for (input.data) |*val| val.* = 1.0; // Token ID 1

    // Forward
    var output = try embeddings.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(total_tokens, output.rows);
    try std.testing.expectEqual(lib.config.embedding_dim, output.cols);

    // Check positional embeddings
    // Row 0 (Batch 0, Pos 0) should have pos embedding 0
    // Row 50 (Batch 1, Pos 0) should have pos embedding 0
    // They should be equal (assuming token embeddings are same, which they are)

    const row0 = output.data[0..lib.config.embedding_dim];
    const row50 = output.data[50 * lib.config.embedding_dim .. 51 * lib.config.embedding_dim];

    for (row0, row50) |v0, v50| {
        try std.testing.expectApproxEqAbs(v0, v50, 1e-5);
    }

    // Row 1 (Batch 0, Pos 1) should equal Row 51 (Batch 1, Pos 1)
    const row1 = output.data[1 * lib.config.embedding_dim .. 2 * lib.config.embedding_dim];
    const row51 = output.data[51 * lib.config.embedding_dim .. 52 * lib.config.embedding_dim];

    for (row1, row51) |v1, v51| {
        try std.testing.expectApproxEqAbs(v1, v51, 1e-5);
    }

    // Backward
    var grads = try Matrix.initZeros(allocator, total_tokens, lib.config.embedding_dim);
    // defer grads.deinit(); // backward takes ownership
    // Set gradient at Row 50 (Batch 1, Pos 0) to 1.0
    for (grads.data[50 * lib.config.embedding_dim .. 51 * lib.config.embedding_dim]) |*g| g.* = 1.0;

    var grad_input = try embeddings.backward(grads, 0.1);
    grad_input.deinit();
}
