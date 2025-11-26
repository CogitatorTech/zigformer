//! Transformer block.
//!
//! TransformerBlock consists of:
//!   1. Multi-head self-attention with residual connection and layer norm
//!   2. Position-wise feed-forward network with residual connection and layer norm
//!
//! Architecture (Pre-LN variant):
//!   - x' = x + SelfAttention(LayerNorm(x))
//!   - y = x' + FeedForward(LayerNorm(x'))
//!
//! References:
//!   - "Attention Is All You Need" (Vaswani et al., 2017)

const std = @import("std");
const lib = @import("../lib.zig");
const layer = lib.layer;
const SelfAttention = lib.self_attention.SelfAttention;
const FeedForward = lib.feed_forward.FeedForward;
const LayerNorm = lib.layer_norm.LayerNorm;
const Matrix = lib.linalg.Matrix;

/// Transformer block.
///
/// Processes sequences through attention (for capturing dependencies)
/// and feed-forward layers (for non-linear transformations), with
/// residual connections and layer normalization for training stability.
pub const TransformerBlock = struct {
    allocator: std.mem.Allocator,
    attention: *SelfAttention, // Multi-head self-attention layer
    feed_forward: *FeedForward, // Position-wise FFN
    norm1: *LayerNorm, // Layer norm after attention
    norm2: *LayerNorm, // Layer norm after feed-forward

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
        // Pre-LN Architecture: x' = x + SelfAttention(LayerNorm(x))
        var norm1_out = try self.norm1.forward(input);
        defer norm1_out.deinit();
        var attention_out = try self.attention.forward(norm1_out, use_cache);
        defer attention_out.deinit();
        var residual1 = try input.add(&attention_out);
        defer residual1.deinit();

        // y = x' + FeedForward(LayerNorm(x'))
        var norm2_out = try self.norm2.forward(residual1);
        defer norm2_out.deinit();
        var ff_out = try self.feed_forward.forward(norm2_out);
        defer ff_out.deinit();
        return residual1.add(&ff_out);
    }

    pub fn backward(self: *TransformerBlock, grads: lib.linalg.Matrix, lr: f32) !lib.linalg.Matrix {
        // Backprop through second residual: y = x' + FFN(LN(x'))
        var grad_residual2 = try grads.clone();
        defer grad_residual2.deinit();
        const grads_for_ff = try grads.clone();
        const grad_ff = try self.feed_forward.backward(grads_for_ff, lr); // consumes grads_for_ff
        var grad_norm2 = try self.norm2.backward(grad_ff, lr); // consumes grad_ff
        defer grad_norm2.deinit();
        var grad_after_ff = try grad_norm2.add(&grad_residual2);
        defer grad_after_ff.deinit();

        // Backprop through first residual: x' = x + Attn(LN(x))
        var grad_residual1 = try grad_after_ff.clone();
        defer grad_residual1.deinit();
        const grads_for_attn = try grad_after_ff.clone();
        const grad_attn = try self.attention.backward(grads_for_attn, lr); // consumes grads_for_attn
        var grad_norm1 = try self.norm1.backward(grad_attn, lr); // consumes grad_attn
        defer grad_norm1.deinit();
        return grad_norm1.add(&grad_residual1);
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
    var grads = try Matrix.initRandom(allocator, 2, embedding_dim, 0.0, 1.0);
    defer grads.deinit();
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

test "TransformerBlock Pre-LN architecture" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const hidden_dim = 32;

    var tb = try TransformerBlock.init(allocator, embedding_dim, hidden_dim);
    defer tb.deinit();

    var input = try Matrix.initRandom(allocator, 2, embedding_dim, 0.0, 1.0);
    defer input.deinit();

    // Store input values for residual verification
    var input_copy = try input.clone();
    defer input_copy.deinit();

    var output = try tb.forward(input, false);
    defer output.deinit();

    // Output should have same shape as input
    try std.testing.expectEqual(input.rows, output.rows);
    try std.testing.expectEqual(input.cols, output.cols);

    // Output should be different from input (due to transformations)
    var all_same = true;
    for (input_copy.data, output.data) |in_val, out_val| {
        if (@abs(in_val - out_val) > 1e-6) {
            all_same = false;
            break;
        }
    }
    try std.testing.expect(!all_same);
}

test "TransformerBlock residual connections" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const hidden_dim = 32;

    var tb = try TransformerBlock.init(allocator, embedding_dim, hidden_dim);
    defer tb.deinit();

    // Create input with known pattern
    var input = try Matrix.init(allocator, 2, embedding_dim);
    defer input.deinit();
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 10));
    }

    var output = try tb.forward(input, false);
    defer output.deinit();

    // The residual connections should ensure output magnitude is related to input
    // This is a basic sanity check that residuals are working
    var input_norm: f32 = 0;
    var output_norm: f32 = 0;
    for (input.data) |val| input_norm += val * val;
    for (output.data) |val| output_norm += val * val;

    // Output should have non-zero magnitude if residuals work
    try std.testing.expect(output_norm > 0);
}
