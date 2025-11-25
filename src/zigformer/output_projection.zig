//! Output projection layer.
//!
//! Maps the final hidden states from the transformer to vocabulary logits,
//! producing a probability distribution over the vocabulary for next-token prediction.
//!
//! Mathematical formulation:
//!  - logits = xW^O + b
//!  - probabilities = softmax(logits)
//!
//! where:
//!  - x ∈ ℝ^(batch × d_model) is the final hidden state
//!  - W^O ∈ ℝ^(d_model × vocab_size) is the output weight matrix
//!  - b ∈ ℝ^vocab_size is the bias vector

const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

/// Output projection layer.
///
/// Projects hidden states to vocabulary size for language modeling.
/// The final softmax is typically applied externally during loss computation.
pub const OutputProjection = struct {
    allocator: std.mem.Allocator,
    w_out: Matrix, // Output weight matrix: embedding_dim × vocab_size
    b_out: Matrix, // Output bias: 1 × vocab_size
    optimizer_w: Adam,
    optimizer_b: Adam,
    has_cached_input: bool,
    cached_input: Matrix,

    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize, vocab_size: usize) !*OutputProjection {
        const self = try allocator.create(OutputProjection);
        const std_dev = std.math.sqrt(2.0 / @as(f32, @floatFromInt(embedding_dim)));
        self.* = .{
            .allocator = allocator,
            .w_out = try Matrix.initRandom(allocator, embedding_dim, vocab_size, 0.0, std_dev),
            .b_out = try Matrix.initZeros(allocator, 1, vocab_size),
            .optimizer_w = try Adam.init(allocator, embedding_dim, vocab_size),
            .optimizer_b = try Adam.init(allocator, 1, vocab_size),
            .has_cached_input = false,
            .cached_input = undefined,
        };
        return self;
    }

    pub fn deinit(self: *OutputProjection) void {
        self.w_out.deinit();
        self.b_out.deinit();
        self.optimizer_w.deinit();
        self.optimizer_b.deinit();
        if (self.has_cached_input) self.cached_input.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *OutputProjection, input: Matrix) !Matrix {
        if (self.has_cached_input) self.cached_input.deinit();
        self.cached_input = try input.clone();
        self.has_cached_input = true;

        var output = try self.cached_input.dot(&self.w_out);
        for (0..output.rows) |r| {
            for (0..output.cols) |c| {
                output.data[r * output.cols + c] += self.b_out.data[c];
            }
        }
        return output;
    }

    pub fn backward(self: *OutputProjection, grads: Matrix, lr: f32) !Matrix {
        var mut_grads = grads;
        defer mut_grads.deinit();

        var grad_b_out = try Matrix.initZeros(self.allocator, 1, self.b_out.cols);
        defer grad_b_out.deinit();
        for (0..mut_grads.rows) |r| {
            for (0..mut_grads.cols) |c| {
                grad_b_out.data[c] += mut_grads.at(r, c);
            }
        }

        var transposed_input = try self.cached_input.transpose();
        defer transposed_input.deinit();
        var grad_w_out = try transposed_input.dot(&mut_grads);
        defer grad_w_out.deinit();

        var transposed_w = try self.w_out.transpose();
        defer transposed_w.deinit();
        const grad_input = try mut_grads.dot(&transposed_w);

        self.optimizer_w.step(&self.w_out, grad_w_out, lr);
        self.optimizer_b.step(&self.b_out, grad_b_out, lr);

        return grad_input;
    }

    pub fn parameters(self: *const OutputProjection) usize {
        return self.w_out.data.len + self.b_out.data.len;
    }

    pub fn toLayer(self: *OutputProjection) layer.Layer {
        return layer.toLayer(OutputProjection)(self);
    }

    pub fn save(self: *const OutputProjection, writer: anytype) !void {
        try self.w_out.save(writer);
        try self.b_out.save(writer);
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !*OutputProjection {
        const self = try allocator.create(OutputProjection);
        errdefer allocator.destroy(self);

        var w_out = try Matrix.load(allocator, reader);
        errdefer w_out.deinit();

        var b_out = try Matrix.load(allocator, reader);
        errdefer b_out.deinit();

        self.* = .{
            .allocator = allocator,
            .w_out = w_out,
            .b_out = b_out,
            .optimizer_w = try Adam.init(allocator, w_out.rows, w_out.cols),
            .optimizer_b = try Adam.init(allocator, b_out.rows, b_out.cols),
            .has_cached_input = false,
            .cached_input = undefined,
        };
        return self;
    }
};
test "OutputProjection" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const vocab_size = 10;

    var op = try OutputProjection.init(allocator, embedding_dim, vocab_size);
    defer op.deinit();

    // Test Forward
    var input = try Matrix.initRandom(allocator, 2, embedding_dim, 0.0, 1.0);
    defer input.deinit();

    var output = try op.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(input.rows, output.rows);
    try std.testing.expectEqual(vocab_size, output.cols);

    // Test Backward
    const grads = try Matrix.initRandom(allocator, 2, vocab_size, 0.0, 1.0);
    var grad_input = try op.backward(grads, 0.01);
    defer grad_input.deinit();

    try std.testing.expectEqual(input.rows, grad_input.rows);
    try std.testing.expectEqual(input.cols, grad_input.cols);
}

test "OutputProjection (save and load)" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const vocab_size = 10;

    var op = try OutputProjection.init(allocator, embedding_dim, vocab_size);
    defer op.deinit();

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);
    try op.save(writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    const reader = stream.reader();
    var loaded_op = try OutputProjection.load(allocator, reader);
    defer loaded_op.deinit();

    try std.testing.expectEqual(op.parameters(), loaded_op.parameters());
}
