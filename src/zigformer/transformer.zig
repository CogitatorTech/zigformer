const std = @import("std");
const lib = @import("../lib.zig");
const layer = lib.layer;
const SelfAttention = lib.self_attention.SelfAttention;
const FeedForward = lib.feed_forward.FeedForward;
const LayerNorm = lib.layer_norm.LayerNorm;

pub const TransformerBlock = struct {
    allocator: std.mem.Allocator,
    attention: *SelfAttention,
    feed_forward: *FeedForward,
    norm1: *LayerNorm,
    norm2: *LayerNorm,

    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize, hidden_dim: usize) !*TransformerBlock {
        const self = try allocator.create(TransformerBlock);
        self.* = .{
            .allocator = allocator,
            .attention = try SelfAttention.init(allocator, embedding_dim),
            .feed_forward = try FeedForward.init(allocator, embedding_dim, hidden_dim),
            .norm1 = try LayerNorm.init(allocator, embedding_dim),
            .norm2 = try LayerNorm.init(allocator, embedding_dim),
        };
        return self;
    }

    pub fn deinit(self: *TransformerBlock) void {
        self.attention.deinit();
        self.feed_forward.deinit();
        self.norm1.deinit();
        self.norm2.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *TransformerBlock, input: lib.linalg.Matrix) !lib.linalg.Matrix {
        var attention_out = try self.attention.forward(input);
        var norm1_out = try self.norm1.forward(attention_out);
        attention_out.deinit();
        var ff_out = try self.feed_forward.forward(norm1_out);
        norm1_out.deinit();
        const norm2_out = try self.norm2.forward(ff_out);
        ff_out.deinit();
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
};
