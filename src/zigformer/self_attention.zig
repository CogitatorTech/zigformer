const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

pub const SelfAttention = struct {
    allocator: std.mem.Allocator,
    embedding_dim: usize,
    num_heads: usize,
    head_dim: usize,

    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix, // Output projection

    has_cache: bool,
    cached_input: Matrix,
    cached_q: Matrix,
    cached_k: Matrix,
    cached_v: Matrix,
    cached_attention_scores: Matrix, // Stores scores for all heads
    cached_context: Matrix, // Stores concatenated context before output projection

    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
    optimizer_w_o: Adam,

    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize) !*SelfAttention {
        const num_heads = lib.config.num_heads;
        if (embedding_dim % num_heads != 0) {
            return error.EmbeddingDimNotDivisibleByNumHeads;
        }
        const head_dim = embedding_dim / num_heads;

        const self = try allocator.create(SelfAttention);
        const std_dev = std.math.sqrt(2.0 / @as(f32, @floatFromInt(embedding_dim)));

        self.* = .{
            .allocator = allocator,
            .embedding_dim = embedding_dim,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .w_q = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_k = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_v = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_o = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .has_cache = false,
            .cached_input = undefined,
            .cached_q = undefined,
            .cached_k = undefined,
            .cached_v = undefined,
            .cached_attention_scores = undefined,
            .cached_context = undefined,
            .optimizer_w_q = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_k = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_v = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_o = try Adam.init(allocator, embedding_dim, embedding_dim),
        };
        return self;
    }

    pub fn deinit(self: *SelfAttention) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_q.deinit();
            self.cached_k.deinit();
            self.cached_v.deinit();
            self.cached_attention_scores.deinit();
            self.cached_context.deinit();
        }
        self.optimizer_w_q.deinit();
        self.optimizer_w_k.deinit();
        self.optimizer_w_v.deinit();
        self.optimizer_w_o.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *SelfAttention, input: Matrix) !Matrix {
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_q.deinit();
            self.cached_k.deinit();
            self.cached_v.deinit();
            self.cached_attention_scores.deinit();
            self.cached_context.deinit();
        }
        self.cached_input = try input.clone();
        self.has_cache = true;

        // Linear projections
        self.cached_q = try self.cached_input.dot(&self.w_q);
        self.cached_k = try self.cached_input.dot(&self.w_k);
        self.cached_v = try self.cached_input.dot(&self.w_v);

        const seq_len = input.rows;
        const dk_sqrt = std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));

        // Initialize attention scores storage
        // We store scores as (num_heads * seq_len, seq_len) for simpler matrix ops if possible,
        // or just handle logic manually.
        // Let's do manual loop for clarity and correctness with MHA.
        self.cached_attention_scores = try Matrix.init(self.allocator, self.num_heads * seq_len, seq_len);
        self.cached_context = try Matrix.init(self.allocator, seq_len, self.embedding_dim);

        for (0..self.num_heads) |h| {
            // Extract Q, K, V for this head
            // Q_h: (seq_len, head_dim)
            var q_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
            defer q_h.deinit();
            var k_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
            defer k_h.deinit();
            var v_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
            defer v_h.deinit();

            for (0..seq_len) |r| {
                const start = h * self.head_dim;
                const end = start + self.head_dim;
                @memcpy(q_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_q.data[r * self.embedding_dim + start .. r * self.embedding_dim + end]);
                @memcpy(k_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_k.data[r * self.embedding_dim + start .. r * self.embedding_dim + end]);
                @memcpy(v_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_v.data[r * self.embedding_dim + start .. r * self.embedding_dim + end]);
            }

            // Attention scores = softmax(Q K^T / sqrt(dk))
            var k_h_t = try k_h.transpose();
            defer k_h_t.deinit();
            var scores = try q_h.dot(&k_h_t);
            // Don't defer scores yet, we need to copy it to cached_attention_scores

            // Scale
            for (scores.data) |*s| s.* /= dk_sqrt;

            // Masking (causal)
            for (0..seq_len) |i| {
                for (i + 1..seq_len) |j| {
                    scores.set(i, j, -std.math.inf(f32));
                }
            }

            // Softmax
            for (0..seq_len) |r| {
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

            // Store scores
            for (0..seq_len) |r| {
                const src_row = scores.data[r * seq_len .. (r + 1) * seq_len];
                const dst_row = self.cached_attention_scores.data[(h * seq_len + r) * seq_len .. (h * seq_len + r + 1) * seq_len];
                @memcpy(dst_row, src_row);
            }

            // Compute context for this head: scores * V
            var context_h = try scores.dot(&v_h);
            defer context_h.deinit();
            scores.deinit(); // Now we can free scores

            // Concatenate into cached_context
            for (0..seq_len) |r| {
                const src_row = context_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                const dst_start = r * self.embedding_dim + h * self.head_dim;
                @memcpy(self.cached_context.data[dst_start .. dst_start + self.head_dim], src_row);
            }
        }

        // Output projection
        var output = try self.cached_context.dot(&self.w_o);

        // Residual connection
        const result = try output.add(&self.cached_input);
        output.deinit();
        return result;
    }

    pub fn backward(self: *SelfAttention, grads: Matrix, lr: f32) !Matrix {
        var mut_grads = grads;
        defer mut_grads.deinit();

        // Gradients for Residual connection
        var grad_input_residual = try mut_grads.clone();
        defer grad_input_residual.deinit();

        // Gradients through Output Projection
        var w_o_t = try self.w_o.transpose();
        defer w_o_t.deinit();
        var grad_context = try mut_grads.dot(&w_o_t);
        defer grad_context.deinit();

        var context_t = try self.cached_context.transpose();
        defer context_t.deinit();
        var grad_w_o = try context_t.dot(&mut_grads);
        defer grad_w_o.deinit();

        // Gradients for Q, K, V (accumulated from heads)
        var grad_q = try Matrix.initZeros(self.allocator, self.cached_q.rows, self.cached_q.cols);
        defer grad_q.deinit();
        var grad_k = try Matrix.initZeros(self.allocator, self.cached_k.rows, self.cached_k.cols);
        defer grad_k.deinit();
        var grad_v = try Matrix.initZeros(self.allocator, self.cached_v.rows, self.cached_v.cols);
        defer grad_v.deinit();

        const seq_len = self.cached_input.rows;
        const dk_sqrt = std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));

        for (0..self.num_heads) |h| {
            // Reconstruct Q_h, K_h, V_h, Scores_h for backward pass
            var q_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
            defer q_h.deinit();
            var k_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
            defer k_h.deinit();
            var v_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
            defer v_h.deinit();
            var scores_h = try Matrix.init(self.allocator, seq_len, seq_len);
            defer scores_h.deinit();

            for (0..seq_len) |r| {
                const start = h * self.head_dim;
                const end = start + self.head_dim;
                @memcpy(q_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_q.data[r * self.embedding_dim + start .. r * self.embedding_dim + end]);
                @memcpy(k_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_k.data[r * self.embedding_dim + start .. r * self.embedding_dim + end]);
                @memcpy(v_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_v.data[r * self.embedding_dim + start .. r * self.embedding_dim + end]);

                const score_src = self.cached_attention_scores.data[(h * seq_len + r) * seq_len .. (h * seq_len + r + 1) * seq_len];
                @memcpy(scores_h.data[r * seq_len .. (r + 1) * seq_len], score_src);
            }

            // Extract grad_context for this head
            var grad_context_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
            defer grad_context_h.deinit();
            for (0..seq_len) |r| {
                const start = h * self.head_dim;
                const end = start + self.head_dim;
                @memcpy(grad_context_h.data[r * self.head_dim .. (r + 1) * self.head_dim], grad_context.data[r * self.embedding_dim + start .. r * self.embedding_dim + end]);
            }

            // dV = Scores^T * dContext
            var scores_h_t = try scores_h.transpose();
            defer scores_h_t.deinit();
            var grad_v_h = try scores_h_t.dot(&grad_context_h);
            defer grad_v_h.deinit();

            // dScores = dContext * V^T
            var v_h_t = try v_h.transpose();
            defer v_h_t.deinit();
            var grad_scores_h = try grad_context_h.dot(&v_h_t);
            defer grad_scores_h.deinit();

            // Backprop through Softmax
            for (0..seq_len) |r| {
                const score_row = scores_h.data[r * seq_len .. (r + 1) * seq_len];
                const grad_row = grad_scores_h.data[r * seq_len .. (r + 1) * seq_len];
                var dot_product: f32 = 0;
                for (score_row, grad_row) |s, g| dot_product += s * g;
                for (score_row, grad_row) |s, *g| g.* = s * (g.* - dot_product);
                // Backprop through scaling
                for (grad_row) |*g| g.* /= dk_sqrt;
            }

            // dQ = dScores * K
            var grad_q_h = try grad_scores_h.dot(&k_h);
            defer grad_q_h.deinit();

            // dK = dScores^T * Q
            var grad_scores_h_t = try grad_scores_h.transpose();
            defer grad_scores_h_t.deinit();
            var grad_k_h = try grad_scores_h_t.dot(&q_h);
            defer grad_k_h.deinit();

            // Accumulate into global grad_q, grad_k, grad_v
            for (0..seq_len) |r| {
                const start = h * self.head_dim;
                const end = start + self.head_dim;
                const dst_q = grad_q.data[r * self.embedding_dim + start .. r * self.embedding_dim + end];
                const src_q = grad_q_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                for (dst_q, src_q) |*d, s| d.* += s;

                const dst_k = grad_k.data[r * self.embedding_dim + start .. r * self.embedding_dim + end];
                const src_k = grad_k_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                for (dst_k, src_k) |*d, s| d.* += s;

                const dst_v = grad_v.data[r * self.embedding_dim + start .. r * self.embedding_dim + end];
                const src_v = grad_v_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                for (dst_v, src_v) |*d, s| d.* += s;
            }
        }

        // Gradients for W_Q, W_K, W_V
        var cached_input_t = try self.cached_input.transpose();
        defer cached_input_t.deinit();

        var grad_w_q = try cached_input_t.dot(&grad_q);
        defer grad_w_q.deinit();
        var grad_w_k = try cached_input_t.dot(&grad_k);
        defer grad_w_k.deinit();
        var grad_w_v = try cached_input_t.dot(&grad_v);
        defer grad_w_v.deinit();

        // Update weights
        self.optimizer_w_q.step(&self.w_q, grad_w_q, lr);
        self.optimizer_w_k.step(&self.w_k, grad_w_k, lr);
        self.optimizer_w_v.step(&self.w_v, grad_w_v, lr);
        self.optimizer_w_o.step(&self.w_o, grad_w_o, lr);

        // Compute gradient w.r.t input
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
        return self.w_q.data.len + self.w_k.data.len + self.w_v.data.len + self.w_o.data.len;
    }

    pub fn toLayer(self: *SelfAttention) layer.Layer {
        return layer.toLayer(SelfAttention)(self);
    }
};

test "SelfAttention Forward/Backward" {
    const allocator = std.testing.allocator;
    // Assuming default config: embedding_dim=128, num_heads=4.
    // We can't easily check config values here, but we can check if init succeeds.

    const embedding_dim = lib.config.embedding_dim;
    var attn = try SelfAttention.init(allocator, embedding_dim);
    defer attn.deinit();

    const seq_len = 10;
    var input = try Matrix.initRandom(allocator, seq_len, embedding_dim, 0.0, 1.0);
    defer input.deinit();

    // Forward
    var output = try attn.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(seq_len, output.rows);
    try std.testing.expectEqual(embedding_dim, output.cols);

    // Backward
    const grads = try Matrix.initRandom(allocator, seq_len, embedding_dim, 0.0, 1.0);
    // defer grads.deinit(); // backward takes ownership

    var grad_input = try attn.backward(grads, 0.01);
    defer grad_input.deinit();

    try std.testing.expectEqual(seq_len, grad_input.rows);
    try std.testing.expectEqual(embedding_dim, grad_input.cols);
}
