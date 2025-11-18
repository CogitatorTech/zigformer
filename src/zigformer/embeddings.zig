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

    pub fn forward(self: *Embeddings, input: Matrix) !Matrix {
        if (self.has_cached_input) self.cached_input.deinit();
        self.cached_input = try input.clone();
        self.has_cached_input = true;

        const seq_len = input.cols;
        var token_embeds = try Matrix.init(self.allocator, seq_len, lib.config.embedding_dim);
        for (input.data, 0..) |token_id, i| {
            const id: usize = @intFromFloat(token_id);
            const row = self.token_embeddings.data[id * self.token_embeddings.cols .. (id + 1) * self.token_embeddings.cols];
            @memcpy(token_embeds.data[i * token_embeds.cols .. (i + 1) * token_embeds.cols], row);
        }

        var pos_embeds = try Matrix.init(self.allocator, seq_len, lib.config.embedding_dim);
        const clamped_seq_len = @min(seq_len, lib.config.max_seq_len);
        const pos_data = self.positional_embeddings.data[0 .. clamped_seq_len * lib.config.embedding_dim];
        @memcpy(pos_embeds.data, pos_data);

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

        for (self.cached_input.data, 0..) |token_id, i| {
            const id: usize = @intFromFloat(token_id);
            const grad_row = mut_grads.data[i * mut_grads.cols .. (i + 1) * mut_grads.cols];

            const token_grad_row = token_grads.data[id * token_grads.cols .. (id + 1) * token_grads.cols];
            for (token_grad_row, grad_row) |*t, g| t.* += g;

            const pos_grad_row = pos_grads.data[i * pos_grads.cols .. (i + 1) * pos_grads.cols];
            for (pos_grad_row, grad_row) |*p, g| p.* += g;
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
};
