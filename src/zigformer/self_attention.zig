const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

pub const SelfAttention = struct {
    allocator: std.mem.Allocator,
    embedding_dim: usize,
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    has_cache: bool,
    cached_input: Matrix,
    cached_q: Matrix,
    cached_k: Matrix,
    cached_v: Matrix,
    cached_attention_scores: Matrix,
    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,

    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize) !*SelfAttention {
        const self = try allocator.create(SelfAttention);
        const std_dev = std.math.sqrt(2.0 / @as(f32, @floatFromInt(embedding_dim)));
        self.* = .{
            .allocator = allocator,
            .embedding_dim = embedding_dim,
            .w_q = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_k = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_v = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .has_cache = false,
            .cached_input = undefined,
            .cached_q = undefined,
            .cached_k = undefined,
            .cached_v = undefined,
            .cached_attention_scores = undefined,
            .optimizer_w_q = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_k = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_v = try Adam.init(allocator, embedding_dim, embedding_dim),
        };
        return self;
    }

    pub fn deinit(self: *SelfAttention) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_q.deinit();
            self.cached_k.deinit();
            self.cached_v.deinit();
            self.cached_attention_scores.deinit();
        }
        self.optimizer_w_q.deinit();
        self.optimizer_w_k.deinit();
        self.optimizer_w_v.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *SelfAttention, input: Matrix) !Matrix {
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_q.deinit();
            self.cached_k.deinit();
            self.cached_v.deinit();
            self.cached_attention_scores.deinit();
        }
        self.cached_input = try input.clone();
        self.has_cache = true;

        self.cached_q = try self.cached_input.dot(&self.w_q);
        self.cached_k = try self.cached_input.dot(&self.w_k);
        self.cached_v = try self.cached_input.dot(&self.w_v);

        const dk_sqrt = std.math.sqrt(@as(f32, @floatFromInt(self.embedding_dim)));
        var k_t = try self.cached_k.transpose();
        defer k_t.deinit();
        var scores = try self.cached_q.dot(&k_t);

        for (scores.data) |*s| s.* /= dk_sqrt;

        const seq_len = scores.rows;
        for (0..seq_len) |i| {
            for (i + 1..seq_len) |j| {
                scores.set(i, j, -std.math.inf(f32));
            }
        }

        for (0..scores.rows) |r| {
            var max_val: f32 = -std.math.inf(f32);
            const row_slice = scores.data[r * scores.cols .. (r + 1) * scores.cols];
            for (row_slice) |s| max_val = @max(max_val, s);

            var sum_exp: f32 = 0.0;
            for (row_slice) |*s| {
                s.* = std.math.exp(s.* - max_val);
                sum_exp += s.*;
            }
            if (sum_exp > 0) {
                for (row_slice) |*s| s.* /= sum_exp;
            }
        }
        self.cached_attention_scores = scores;

        var attention = try self.cached_attention_scores.dot(&self.cached_v);
        const result = try attention.add(&self.cached_input);
        attention.deinit();
        return result;
    }

    pub fn backward(self: *SelfAttention, grads: Matrix, lr: f32) !Matrix {
        var mut_grads = grads;
        defer mut_grads.deinit();

        var grad_attention = try mut_grads.clone();
        defer grad_attention.deinit();
        var grad_input_residual = try mut_grads.clone();
        defer grad_input_residual.deinit();

        var scores_t = try self.cached_attention_scores.transpose();
        defer scores_t.deinit();
        var grad_v = try scores_t.dot(&grad_attention);
        defer grad_v.deinit();

        var v_t = try self.cached_v.transpose();
        defer v_t.deinit();
        var grad_scores = try grad_attention.dot(&v_t);
        defer grad_scores.deinit();

        const dk_sqrt = std.math.sqrt(@as(f32, @floatFromInt(self.embedding_dim)));

        for (0..self.cached_attention_scores.rows) |r| {
            const score_row = self.cached_attention_scores.data[r * self.cached_attention_scores.cols .. (r + 1) * self.cached_attention_scores.cols];
            const grad_row = grad_scores.data[r * grad_scores.cols .. (r + 1) * grad_scores.cols];
            var dot_product: f32 = 0;
            for (score_row, grad_row) |s, g| dot_product += s * g;
            for (score_row, grad_row) |s, *g| g.* = s * (g.* - dot_product);
            for (grad_row) |*g| g.* /= dk_sqrt;
        }

        var grad_q = try grad_scores.dot(&self.cached_k);
        defer grad_q.deinit();
        var grad_scores_t = try grad_scores.transpose();
        defer grad_scores_t.deinit();
        var grad_k = try grad_scores_t.dot(&self.cached_q);
        defer grad_k.deinit();

        var cached_input_t = try self.cached_input.transpose();
        defer cached_input_t.deinit();

        var grad_w_q = try cached_input_t.dot(&grad_q);
        defer grad_w_q.deinit();
        var grad_w_k = try cached_input_t.dot(&grad_k);
        defer grad_w_k.deinit();
        var grad_w_v = try cached_input_t.dot(&grad_v);
        defer grad_w_v.deinit();

        self.optimizer_w_q.step(&self.w_q, grad_w_q, lr);
        self.optimizer_w_k.step(&self.w_k, grad_w_k, lr);
        self.optimizer_w_v.step(&self.w_v, grad_w_v, lr);

        var w_q_t = try self.w_q.transpose();
        defer w_q_t.deinit();
        var grad_input_q = try grad_q.dot(&w_q_t);
        defer grad_input_q.deinit();

        var w_k_t = try self.w_k.transpose();
        defer w_k_t.deinit();
        var grad_input_k = try grad_k.dot(&w_k_t);
        defer grad_input_k.deinit();

        var w_v_t = try self.w_v.transpose();
        defer w_v_t.deinit();
        var grad_input_v = try grad_v.dot(&w_v_t);
        defer grad_input_v.deinit();

        var grad_input = try grad_input_q.add(&grad_input_k);
        defer grad_input.deinit();
        var grad_input_sum = try grad_input.add(&grad_input_v);
        defer grad_input_sum.deinit();

        return grad_input_sum.add(&grad_input_residual);
    }

    pub fn parameters(self: *const SelfAttention) usize {
        return self.w_q.data.len + self.w_k.data.len + self.w_v.data.len;
    }

    pub fn toLayer(self: *SelfAttention) layer.Layer {
        return layer.toLayer(SelfAttention)(self);
    }
};
