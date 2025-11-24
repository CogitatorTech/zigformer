const std = @import("std");
const lib = @import("../lib.zig");
const layer = lib.layer;
const SelfAttention = lib.self_attention.SelfAttention;
const FeedForward = lib.feed_forward.FeedForward;
const LayerNorm = lib.layer_norm.LayerNorm;
const Matrix = lib.linalg.Matrix;

pub const TransformerBlock = struct {
    allocator: std.mem.Allocator,
    attention: *SelfAttention,
    feed_forward: *FeedForward,
    norm1: *LayerNorm,
    norm2: *LayerNorm,

    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize, hidden_dim: usize) !*TransformerBlock {
        const self = try allocator.create(TransformerBlock);
        const attention = try allocator.create(SelfAttention);
        attention.* = try SelfAttention.init(allocator, embedding_dim, lib.config.num_heads, 1, lib.config.max_seq_len);
        errdefer allocator.destroy(attention);

        self.* = .{
            .allocator = allocator,
            .attention = attention,
            .feed_forward = try FeedForward.init(allocator, embedding_dim, hidden_dim),
            .norm1 = try LayerNorm.init(allocator, embedding_dim),
            .norm2 = try LayerNorm.init(allocator, embedding_dim),
        };
        return self;
    }

    pub fn deinit(self: *TransformerBlock) void {
        self.attention.deinit();
        self.allocator.destroy(self.attention);
        self.feed_forward.deinit();
        self.norm1.deinit();
        self.norm2.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *TransformerBlock, input: Matrix, use_cache: bool) !Matrix {
        // Self Attention
        var attention_out = try self.attention.forward(input, use_cache);
        defer attention_out.deinit();
        var norm1_out = try self.norm1.forward(attention_out);
        defer norm1_out.deinit();
        var ff_out = try self.feed_forward.forward(norm1_out);
        defer ff_out.deinit();
        const norm2_out = try self.norm2.forward(ff_out);
        return norm2_out;
    }

    pub fn backward(self: *TransformerBlock, grads: lib.linalg.Matrix, lr: f32) !lib.linalg.Matrix {
        const grad_norm2 = try self.norm2.backward(grads, lr);
        const grad_ff = try self.feed_forward.backward(grad_norm2, lr);
        const grad_norm1 = try self.norm1.backward(grad_ff, lr);
        const grad_attn = try self.attention.backward(grad_norm1, lr);
        return grad_attn;
    }

    pub fn parameters(self: *const TransformerBlock) usize {
        return self.attention.parameters() + self.feed_forward.parameters() + self.norm1.parameters() + self.norm2.parameters();
    }

    pub fn toLayer(self: *TransformerBlock) layer.Layer {
        return layer.toLayer(TransformerBlock)(self);
    }

    pub fn setBatchSize(self: *TransformerBlock, batch_size: usize) void {
        self.attention.batch_size = batch_size;
    }

    pub fn resetCache(self: *TransformerBlock) void {
        self.attention.resetCache();
    }

    pub fn save(self: *const TransformerBlock, writer: anytype) !void {
        try self.attention.save(writer);
        try self.feed_forward.save(writer);
        try self.norm1.save(writer);
        try self.norm2.save(writer);
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !*TransformerBlock {
        const self = try allocator.create(TransformerBlock);
        errdefer allocator.destroy(self);

        const attention = try SelfAttention.load(allocator, reader);
        errdefer attention.deinit();

        const feed_forward = try FeedForward.load(allocator, reader);
        errdefer feed_forward.deinit();

        const norm1 = try LayerNorm.load(allocator, reader);
        errdefer norm1.deinit();

        const norm2 = try LayerNorm.load(allocator, reader);
        errdefer norm2.deinit();

        self.* = .{
            .allocator = allocator,
            .attention = attention,
            .feed_forward = feed_forward,
            .norm1 = norm1,
            .norm2 = norm2,
        };
        return self;
    }
};
test "TransformerBlock" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const hidden_dim = 32;

    var tb = try TransformerBlock.init(allocator, embedding_dim, hidden_dim);
    defer tb.deinit();

    // Test Forward
    var input = try Matrix.initRandom(allocator, 2, embedding_dim, 0.0, 1.0);
    defer input.deinit();

    var output = try tb.forward(input, false);
    defer output.deinit();

    try std.testing.expectEqual(input.rows, output.rows);
    try std.testing.expectEqual(input.cols, output.cols);

    // Test Backward
    const grads = try Matrix.initRandom(allocator, 2, embedding_dim, 0.0, 1.0);
    var grad_input = try tb.backward(grads, 0.01);
    defer grad_input.deinit();

    try std.testing.expectEqual(input.rows, grad_input.rows);
    try std.testing.expectEqual(input.cols, grad_input.cols);
}

test "TransformerBlock (save and load)" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const hidden_dim = 32;

    var tb = try TransformerBlock.init(allocator, embedding_dim, hidden_dim);
    defer tb.deinit();

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);
    try tb.save(writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    const reader = stream.reader();
    var loaded_tb = try TransformerBlock.load(allocator, reader);
    defer loaded_tb.deinit();

    try std.testing.expectEqual(tb.parameters(), loaded_tb.parameters());
}
