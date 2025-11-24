const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

pub const FeedForward = struct {
    allocator: std.mem.Allocator,
    w1: Matrix,
    b1: Matrix,
    w2: Matrix,
    b2: Matrix,

    has_cache: bool,
    cached_input: Matrix,
    cached_hidden_pre: Matrix,
    cached_hidden_post: Matrix,

    optimizer_w1: Adam,
    optimizer_b1: Adam,
    optimizer_w2: Adam,
    optimizer_b2: Adam,

    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize, hidden_dim: usize) !*FeedForward {
        const self = try allocator.create(FeedForward);
        const std_w1 = std.math.sqrt(2.0 / @as(f32, @floatFromInt(embedding_dim)));
        const std_w2 = std.math.sqrt(2.0 / @as(f32, @floatFromInt(hidden_dim)));

        self.* = .{
            .allocator = allocator,
            .w1 = try Matrix.initRandom(allocator, embedding_dim, hidden_dim, 0.0, std_w1),
            .b1 = try Matrix.initZeros(allocator, 1, hidden_dim),
            .w2 = try Matrix.initRandom(allocator, hidden_dim, embedding_dim, 0.0, std_w2),
            .b2 = try Matrix.initZeros(allocator, 1, embedding_dim),
            .has_cache = false,
            .cached_input = undefined,
            .cached_hidden_pre = undefined,
            .cached_hidden_post = undefined,
            .optimizer_w1 = try Adam.init(allocator, embedding_dim, hidden_dim),
            .optimizer_b1 = try Adam.init(allocator, 1, hidden_dim),
            .optimizer_w2 = try Adam.init(allocator, hidden_dim, embedding_dim),
            .optimizer_b2 = try Adam.init(allocator, 1, embedding_dim),
        };
        return self;
    }

    pub fn deinit(self: *FeedForward) void {
        self.w1.deinit();
        self.b1.deinit();
        self.w2.deinit();
        self.b2.deinit();
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_hidden_pre.deinit();
            self.cached_hidden_post.deinit();
        }
        self.optimizer_w1.deinit();
        self.optimizer_b1.deinit();
        self.optimizer_w2.deinit();
        self.optimizer_b2.deinit();
        self.allocator.destroy(self);
    }

    fn relu(mat: *Matrix) void {
        for (mat.data) |*val| val.* = @max(0.0, val.*);
    }

    fn addBias(mat: *Matrix, bias: *const Matrix) void {
        for (0..mat.rows) |r| {
            const row_slice = mat.data[r * mat.cols .. (r + 1) * mat.cols];
            for (row_slice, bias.data) |*val, b| {
                val.* += b;
            }
        }
    }

    pub fn forward(self: *FeedForward, input: Matrix) !Matrix {
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_hidden_pre.deinit();
            self.cached_hidden_post.deinit();
        }
        self.cached_input = try input.clone();
        self.has_cache = true;

        self.cached_hidden_pre = try self.cached_input.dot(&self.w1);
        addBias(&self.cached_hidden_pre, &self.b1);
        self.cached_hidden_post = try self.cached_hidden_pre.clone();
        relu(&self.cached_hidden_post);

        var output = try self.cached_hidden_post.dot(&self.w2);
        addBias(&output, &self.b2);
        const final_output = try output.add(&input);
        output.deinit();
        return final_output;
    }

    pub fn backward(self: *FeedForward, grads: Matrix, lr: f32) !Matrix {
        var mut_grads = grads;
        defer mut_grads.deinit();

        var grad_w2 = try self.cached_hidden_post.transpose();
        var final_grad_w2 = try grad_w2.dot(&mut_grads);
        grad_w2.deinit();
        defer final_grad_w2.deinit();

        var grad_b2 = try Matrix.initZeros(self.allocator, 1, self.b2.cols);
        defer grad_b2.deinit();
        for (0..mut_grads.rows) |r| {
            const row_slice = mut_grads.data[r * mut_grads.cols .. (r + 1) * mut_grads.cols];
            for (grad_b2.data, row_slice) |*gb, g| {
                gb.* += g;
            }
        }

        var w2_t = try self.w2.transpose();
        defer w2_t.deinit();
        var grad_hidden_post = try mut_grads.dot(&w2_t);
        defer grad_hidden_post.deinit();

        for (grad_hidden_post.data, self.cached_hidden_pre.data) |*g, pre| {
            if (pre <= 0) g.* = 0;
        }

        var grad_w1 = try self.cached_input.transpose();
        var final_grad_w1 = try grad_w1.dot(&grad_hidden_post);
        grad_w1.deinit();
        defer final_grad_w1.deinit();

        var grad_b1 = try Matrix.initZeros(self.allocator, 1, self.b1.cols);
        defer grad_b1.deinit();
        for (0..grad_hidden_post.rows) |r| {
            const row_slice = grad_hidden_post.data[r * grad_hidden_post.cols .. (r + 1) * grad_hidden_post.cols];
            for (grad_b1.data, row_slice) |*gb, g| {
                gb.* += g;
            }
        }

        var w1_t = try self.w1.transpose();
        defer w1_t.deinit();
        var grad_input_ff = try grad_hidden_post.dot(&w1_t);
        defer grad_input_ff.deinit();

        const grad_input = try grad_input_ff.add(&mut_grads);

        self.optimizer_w2.step(&self.w2, final_grad_w2, lr);
        self.optimizer_b2.step(&self.b2, grad_b2, lr);
        self.optimizer_w1.step(&self.w1, final_grad_w1, lr);
        self.optimizer_b1.step(&self.b1, grad_b1, lr);

        return grad_input;
    }

    pub fn parameters(self: *const FeedForward) usize {
        return self.w1.data.len + self.b1.data.len + self.w2.data.len + self.b2.data.len;
    }

    pub fn toLayer(self: *FeedForward) layer.Layer {
        return layer.toLayer(FeedForward)(self);
    }

    pub fn save(self: *const FeedForward, writer: anytype) !void {
        try self.w1.save(writer);
        try self.b1.save(writer);
        try self.w2.save(writer);
        try self.b2.save(writer);
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !*FeedForward {
        const self = try allocator.create(FeedForward);
        errdefer allocator.destroy(self);

        var w1 = try Matrix.load(allocator, reader);
        errdefer w1.deinit();

        var b1 = try Matrix.load(allocator, reader);
        errdefer b1.deinit();

        var w2 = try Matrix.load(allocator, reader);
        errdefer w2.deinit();

        var b2 = try Matrix.load(allocator, reader);
        errdefer b2.deinit();

        self.* = .{
            .allocator = allocator,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
            .has_cache = false,
            .cached_input = undefined,
            .cached_hidden_pre = undefined,
            .cached_hidden_post = undefined,
            .optimizer_w1 = try Adam.init(allocator, w1.rows, w1.cols),
            .optimizer_b1 = try Adam.init(allocator, b1.rows, b1.cols),
            .optimizer_w2 = try Adam.init(allocator, w2.rows, w2.cols),
            .optimizer_b2 = try Adam.init(allocator, b2.rows, b2.cols),
        };
        return self;
    }
};
test "FeedForward" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const hidden_dim = 32;

    var ff = try FeedForward.init(allocator, embedding_dim, hidden_dim);
    defer ff.deinit();

    // Test Forward
    var input = try Matrix.initRandom(allocator, 2, embedding_dim, 0.0, 1.0);
    defer input.deinit();

    var output = try ff.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(input.rows, output.rows);
    try std.testing.expectEqual(input.cols, output.cols);

    // Test Backward
    const grads = try Matrix.initRandom(allocator, 2, embedding_dim, 0.0, 1.0);
    // backward takes ownership of grads
    var grad_input = try ff.backward(grads, 0.01);
    defer grad_input.deinit();

    try std.testing.expectEqual(input.rows, grad_input.rows);
    try std.testing.expectEqual(input.cols, grad_input.cols);
}

test "FeedForward (save and load)" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const hidden_dim = 32;

    var ff = try FeedForward.init(allocator, embedding_dim, hidden_dim);
    defer ff.deinit();

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);
    try ff.save(writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    const reader = stream.reader();
    var loaded_ff = try FeedForward.load(allocator, reader);
    defer loaded_ff.deinit();

    try std.testing.expectEqual(ff.parameters(), loaded_ff.parameters());
}
