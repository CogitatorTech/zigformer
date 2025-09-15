const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

pub const OutputProjection = struct {
    allocator: std.mem.Allocator,
    w_out: Matrix,
    b_out: Matrix,
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
};
