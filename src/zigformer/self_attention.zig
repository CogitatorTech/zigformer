//! Multi-head self-attention.
//!
//! Implements scaled dot-product attention with multiple heads.
//!
//! Mathematical formulation:
//!  - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
//!  - MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
//!
//! References:
//!  - "Attention Is All You Need" (Vaswani et al., 2017)

const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

/// Multi-head self-attention layer.
///
/// Implements the attention mechanism that allows the model to focus on different
/// parts of the input sequence. Each attention head learns to attend to different
/// aspects of the representation.
///
/// Key concepts:
/// - Query (Q): "What am I looking for?"
/// - Key (K): "What do I contain?"
/// - Value (V): "What information do I carry?"
/// - Attention scores: Similarity between queries and keys
/// - Context: Weighted sum of values based on attention scores
pub const SelfAttention = struct {
    allocator: std.mem.Allocator,
    embedding_dim: usize, // Total dimension (must be divisible by num_heads)
    num_heads: usize, // Number of parallel attention heads
    head_dim: usize, // Dimension per head (embedding_dim / num_heads)

    // Learnable weight matrices for Q, K, V projections and output
    // In math notation: W^Q, W^K, W^V ∈ ℝ^(d_model × d_model)
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix, // Output projection matrix W^O

    batch_size: usize,

    // Cached values for backward pass
    has_cache: bool,
    cached_input: Matrix,
    cached_q: Matrix,
    cached_k: Matrix,
    cached_v: Matrix,
    cached_attention_scores: Matrix, // Stores scores for all heads
    cached_context: Matrix, // Stores concatenated context before output projection

    // KV Cache for efficient autoregressive generation
    k_cache: Matrix,
    v_cache: Matrix,
    cache_len: usize,
    max_seq_len: usize,

    // Optimizers for each weight matrix
    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
    optimizer_w_o: Adam,

    /// Initialize a multi-head self-attention layer.
    ///
    /// Parameters:
    ///   allocator: Memory allocator
    ///   embedding_dim: Dimension of input embeddings (d_model)
    ///   num_heads: Number of parallel attention heads (h)
    ///   batch_size: Batch size for training/inference
    ///   max_seq_len: Maximum sequence length for KV caching
    ///
    /// Returns:
    ///   Initialized SelfAttention struct
    ///
    /// Errors:
    ///   error.EmbeddingDimNotDivisibleByHeads: if embedding_dim % num_heads != 0
    pub fn init(allocator: std.mem.Allocator, embedding_dim: usize, num_heads: usize, batch_size: usize, max_seq_len: usize) !SelfAttention {
        if (embedding_dim % num_heads != 0) {
            return error.EmbeddingDimNotDivisibleByHeads;
        }
        const head_dim = embedding_dim / num_heads;

        // Xavier/He initialization: std_dev = sqrt(2 / fan_in)
        const std_dev = std.math.sqrt(2.0 / @as(f32, @floatFromInt(embedding_dim)));

        return .{
            .allocator = allocator,
            .embedding_dim = embedding_dim,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .w_q = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_k = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_v = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .w_o = try Matrix.initRandom(allocator, embedding_dim, embedding_dim, 0.0, std_dev),
            .batch_size = batch_size,
            .has_cache = false,
            .cached_input = try Matrix.init(allocator, 0, 0),
            .cached_q = try Matrix.init(allocator, 0, 0),
            .cached_k = try Matrix.init(allocator, 0, 0),
            .cached_v = try Matrix.init(allocator, 0, 0),
            .cached_attention_scores = try Matrix.init(allocator, 0, 0),
            .cached_context = try Matrix.init(allocator, 0, 0),
            .k_cache = try Matrix.initZeros(allocator, batch_size * max_seq_len, embedding_dim),
            .v_cache = try Matrix.initZeros(allocator, batch_size * max_seq_len, embedding_dim),
            .cache_len = 0,
            .max_seq_len = max_seq_len,
            .optimizer_w_q = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_k = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_v = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_o = try Adam.init(allocator, embedding_dim, embedding_dim),
        };
    }

    /// Deallocate all resources used by this attention layer.
    pub fn deinit(self: *SelfAttention) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
        self.cached_input.deinit();
        self.cached_q.deinit();
        self.cached_k.deinit();
        self.cached_v.deinit();
        self.cached_attention_scores.deinit();
        self.cached_context.deinit();
        self.k_cache.deinit();
        self.v_cache.deinit();
        self.optimizer_w_q.deinit();
        self.optimizer_w_k.deinit();
        self.optimizer_w_v.deinit();
        self.optimizer_w_o.deinit();
    }

    /// Reset the KV cache for autoregressive generation.
    ///
    /// This should be called between different sequences or when starting
    /// a new generation task. It clears cached activations and resets cache length.
    pub fn resetCache(self: *SelfAttention) void {
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_q.deinit();
            self.cached_k.deinit();
            self.cached_v.deinit();
            self.cached_attention_scores.deinit();
            self.cached_context.deinit();
            self.has_cache = false;
        }
        self.cache_len = 0;
    }

    /// Compute multi-head self-attention forward pass.
    ///
    /// Mathematical formulation:
    ///   1. Linear projections: Q = XW^Q, K = XW^K, V = XW^V
    ///   2. Split into heads: Q, K, V ∈ ℝ^(batch × seq_len × d_model) → h heads of ℝ^(batch × seq_len × d_k)
    ///   3. Scaled dot-product attention: scores = QK^T / sqrt(d_k)
    ///   4. Apply causal mask (for autoregressive models): scores[i,j] = -∞ if j > i
    ///   5. Softmax: attention_weights = softmax(scores)
    ///   6. Context: context = attention_weights × V
    ///   7. Concatenate heads and project: output = Concat(head_1, ..., head_h)W^O
    ///
    /// Parameters:
    ///   input: Input tensor of shape (seq_len, embedding_dim) or (batch_size * seq_len, embedding_dim)
    ///   use_cache: Whether to use KV caching for efficient autoregressive generation
    ///
    /// Returns:
    ///   Output tensor of same shape as input
    pub fn forward(self: *SelfAttention, input: Matrix, use_cache: bool) !Matrix {
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
        var k_new = try self.cached_input.dot(&self.w_k);
        defer k_new.deinit();
        var v_new = try self.cached_input.dot(&self.w_v);
        defer v_new.deinit();

        // Update Cache if enabled
        if (use_cache) {
            // Input must be 1 token per batch (rows == batch_size)
            if (input.rows % self.batch_size != 0) {
                return error.InputRowsNotDivisibleByBatchSize;
            }

            const seq_len_in = input.rows / self.batch_size;

            for (0..self.batch_size) |b| {
                const cache_start = b * self.max_seq_len + self.cache_len;
                const input_start = b * seq_len_in;

                // Copy new K/V to cache
                for (0..seq_len_in) |i| {
                    @memcpy(self.k_cache.data[(cache_start + i) * self.embedding_dim .. (cache_start + i + 1) * self.embedding_dim], k_new.data[(input_start + i) * self.embedding_dim .. (input_start + i + 1) * self.embedding_dim]);
                    @memcpy(self.v_cache.data[(cache_start + i) * self.embedding_dim .. (cache_start + i + 1) * self.embedding_dim], v_new.data[(input_start + i) * self.embedding_dim .. (input_start + i + 1) * self.embedding_dim]);
                }
            }
            self.cache_len += seq_len_in;

            // Create matrices for the cached K/V (copy the active portion of the cache)
            const cache_rows = self.batch_size * self.cache_len;
            self.cached_k = try Matrix.init(self.allocator, cache_rows, self.embedding_dim);
            self.cached_v = try Matrix.init(self.allocator, cache_rows, self.embedding_dim);
            @memcpy(self.cached_k.data, self.k_cache.data[0 .. cache_rows * self.embedding_dim]);
            @memcpy(self.cached_v.data, self.v_cache.data[0 .. cache_rows * self.embedding_dim]);
        } else {
            // Training mode: transfer ownership of k_new and v_new to cached_k and cached_v
            self.cached_k = k_new;
            self.cached_v = v_new;
            // Prevent defer from freeing these since we've transferred ownership
            k_new = try Matrix.init(self.allocator, 0, 0);
            v_new = try Matrix.init(self.allocator, 0, 0);
        }

        const total_rows = input.rows;
        const seq_len = total_rows / self.batch_size;
        const dk_sqrt = std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));

        // Output rows = input rows
        self.cached_attention_scores = try Matrix.init(self.allocator, self.num_heads * total_rows, if (use_cache) self.cache_len else seq_len);
        self.cached_context = try Matrix.init(self.allocator, total_rows, self.embedding_dim);

        for (0..self.batch_size) |b| {
            const batch_offset = b * seq_len;
            const cache_offset = b * self.max_seq_len;
            const current_cache_len = if (use_cache) self.cache_len else seq_len;

            for (0..self.num_heads) |h| {
                // Extract Q for this head and batch
                var q_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
                defer q_h.deinit();

                for (0..seq_len) |r| {
                    const global_r = batch_offset + r;
                    const start = h * self.head_dim;
                    const end = start + self.head_dim;
                    @memcpy(q_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_q.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
                }

                // Extract K, V for this head and batch (from cache or current)
                var k_h = try Matrix.init(self.allocator, current_cache_len, self.head_dim);
                defer k_h.deinit();
                var v_h = try Matrix.init(self.allocator, current_cache_len, self.head_dim);
                defer v_h.deinit();

                if (use_cache) {
                    for (0..current_cache_len) |r| {
                        const global_r = cache_offset + r;
                        const start = h * self.head_dim;
                        const end = start + self.head_dim;
                        @memcpy(k_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.k_cache.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
                        @memcpy(v_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.v_cache.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
                    }
                } else {
                    for (0..seq_len) |r| {
                        const global_r = batch_offset + r;
                        const start = h * self.head_dim;
                        const end = start + self.head_dim;
                        @memcpy(k_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_k.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
                        @memcpy(v_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_v.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
                    }
                }

                // Attention scores = softmax(Q K^T / sqrt(dk))
                var k_h_t = try k_h.transpose();
                defer k_h_t.deinit();
                var scores = try q_h.dot(&k_h_t);
                // Don't defer scores yet, we need to copy it to cached_attention_scores

                // Scale
                for (scores.data) |*s| s.* /= dk_sqrt;

                // Masking (causal)
                // If use_cache, we are attending to [0..cache_len].
                // The current query is at position `cache_len - 1` (if single token).
                // Or `cache_len - seq_len + i`.
                // If use_cache is true, we assume we are generating, so we attend to everything in the past.
                // So NO masking needed for the past tokens.
                // But if we are processing a prompt (seq_len > 1) with use_cache=true (filling cache),
                // we DO need masking for the prompt tokens against themselves.

                // Masking (causal)
                // If use_cache, we are attending to [0..cache_len].
                // The current query is at position `cache_len - seq_len + i` (where i is 0..seq_len).
                // We need to mask keys that are in the future relative to the query.
                // Keys are at indices 0..cache_len.
                // Mask key j if j > query_pos.

                const current_cache_len_val = if (use_cache) self.cache_len else seq_len;
                const old_cache_len = if (use_cache) self.cache_len - seq_len else 0;

                for (0..seq_len) |i| {
                    const query_pos = old_cache_len + i;
                    // Mask keys j where j > query_pos
                    const start_mask = query_pos + 1;
                    if (start_mask < current_cache_len_val) {
                        for (start_mask..current_cache_len_val) |j| {
                            scores.set(i, j, -std.math.inf(f32));
                        }
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
                    const global_r = batch_offset + r;
                    const src_row = scores.data[r * seq_len .. (r + 1) * seq_len];
                    const dst_row = self.cached_attention_scores.data[(h * total_rows + global_r) * seq_len .. (h * total_rows + global_r + 1) * seq_len];
                    @memcpy(dst_row, src_row);
                }

                // Compute context for this head: scores * V
                var context_h = try scores.dot(&v_h);
                defer context_h.deinit();
                scores.deinit(); // Now we can free scores

                // Concatenate into cached_context
                for (0..seq_len) |r| {
                    const global_r = batch_offset + r;
                    const src_row = context_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                    const dst_start = global_r * self.embedding_dim + h * self.head_dim;
                    @memcpy(self.cached_context.data[dst_start .. dst_start + self.head_dim], src_row);
                }
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

        const total_rows = self.cached_input.rows;
        const seq_len = total_rows / self.batch_size;
        const dk_sqrt = std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));

        for (0..self.batch_size) |b| {
            const batch_offset = b * seq_len;

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
                    const global_r = batch_offset + r;
                    const start = h * self.head_dim;
                    const end = start + self.head_dim;
                    @memcpy(q_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_q.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
                    @memcpy(k_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_k.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
                    @memcpy(v_h.data[r * self.head_dim .. (r + 1) * self.head_dim], self.cached_v.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);

                    const score_src = self.cached_attention_scores.data[(h * total_rows + global_r) * seq_len .. (h * total_rows + global_r + 1) * seq_len];
                    @memcpy(scores_h.data[r * seq_len .. (r + 1) * seq_len], score_src);
                }

                // Extract grad_context for this head
                var grad_context_h = try Matrix.init(self.allocator, seq_len, self.head_dim);
                defer grad_context_h.deinit();
                for (0..seq_len) |r| {
                    const global_r = batch_offset + r;
                    const start = h * self.head_dim;
                    const end = start + self.head_dim;
                    @memcpy(grad_context_h.data[r * self.head_dim .. (r + 1) * self.head_dim], grad_context.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end]);
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
                    const global_r = batch_offset + r;
                    const start = h * self.head_dim;
                    const end = start + self.head_dim;
                    const dst_q = grad_q.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end];
                    const src_q = grad_q_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                    for (dst_q, src_q) |*d, s| d.* += s;

                    const dst_k = grad_k.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end];
                    const src_k = grad_k_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                    for (dst_k, src_k) |*d, s| d.* += s;

                    const dst_v = grad_v.data[global_r * self.embedding_dim + start .. global_r * self.embedding_dim + end];
                    const src_v = grad_v_h.data[r * self.head_dim .. (r + 1) * self.head_dim];
                    for (dst_v, src_v) |*d, s| d.* += s;
                }
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

    pub fn save(self: *const SelfAttention, writer: anytype) !void {
        try writer.writeInt(usize, self.embedding_dim, .little);
        try writer.writeInt(usize, self.num_heads, .little);
        try writer.writeInt(usize, self.head_dim, .little);

        try self.w_q.save(writer);
        try self.w_k.save(writer);
        try self.w_v.save(writer);
        try self.w_o.save(writer);
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !*SelfAttention {
        const embedding_dim = try reader.readInt(usize, .little);
        const num_heads = try reader.readInt(usize, .little);
        const head_dim = try reader.readInt(usize, .little);

        const self = try allocator.create(SelfAttention);
        errdefer allocator.destroy(self);

        var w_q = try Matrix.load(allocator, reader);
        errdefer w_q.deinit();

        var w_k = try Matrix.load(allocator, reader);
        errdefer w_k.deinit();

        var w_v = try Matrix.load(allocator, reader);
        errdefer w_v.deinit();

        var w_o = try Matrix.load(allocator, reader);
        errdefer w_o.deinit();

        self.* = .{
            .allocator = allocator,
            .embedding_dim = embedding_dim,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .batch_size = 1,
            .has_cache = false,
            .cached_input = try Matrix.init(allocator, 0, 0),
            .cached_q = try Matrix.init(allocator, 0, 0),
            .cached_k = try Matrix.init(allocator, 0, 0),
            .cached_v = try Matrix.init(allocator, 0, 0),
            .cached_attention_scores = try Matrix.init(allocator, 0, 0),
            .cached_context = try Matrix.init(allocator, 0, 0),
            .k_cache = try Matrix.initZeros(allocator, 1 * lib.config.max_seq_len, embedding_dim),
            .v_cache = try Matrix.initZeros(allocator, 1 * lib.config.max_seq_len, embedding_dim),
            .cache_len = 0,
            .max_seq_len = lib.config.max_seq_len,
            .optimizer_w_q = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_k = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_v = try Adam.init(allocator, embedding_dim, embedding_dim),
            .optimizer_w_o = try Adam.init(allocator, embedding_dim, embedding_dim),
        };
        return self;
    }
};

test "SelfAttention Forward/Backward" {
    const allocator = std.testing.allocator;
    // Assuming default config: embedding_dim=128, num_heads=4.
    // We can't easily check config values here, but we can check if init succeeds.

    const embedding_dim = lib.config.embedding_dim;
    var attn = try SelfAttention.init(allocator, embedding_dim, lib.config.num_heads, 1, lib.config.max_seq_len);
    defer attn.deinit();

    const seq_len = 10;
    var input = try Matrix.initRandom(allocator, seq_len, embedding_dim, 0.0, 1.0);
    defer input.deinit();

    // Forward
    var output = try attn.forward(input, false);
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

test "SelfAttention Causal Masking with Cache" {
    const allocator = std.testing.allocator;
    const embedding_dim = 16;
    const num_heads = 2;
    const seq_len = 4;
    var attn = try SelfAttention.init(allocator, embedding_dim, num_heads, 1, 10);
    defer attn.deinit();

    var input = try Matrix.initRandom(allocator, seq_len, embedding_dim, 0.0, 1.0);
    defer input.deinit();

    // Forward with use_cache=true
    // This simulates processing a prompt of length 4.
    var output = try attn.forward(input, true);
    defer output.deinit();

    // Check cached_attention_scores
    // Should be (num_heads * seq_len) x seq_len
    // For each head, the 4x4 score matrix should be lower triangular (masked).
    // scores[i, j] should be -inf (or very small after softmax, but we check pre-softmax scores if possible?
    // Wait, cached_attention_scores stores POST-softmax scores.
    // So masked values should be 0.0.

    const scores = attn.cached_attention_scores;
    // Rows: num_heads * seq_len
    // Cols: seq_len (since cache_len = seq_len after first pass)

    for (0..num_heads) |h| {
        for (0..seq_len) |r| {
            for (r + 1..seq_len) |c| {
                const global_r = h * seq_len + r;
                const score = scores.at(global_r, c);
                // Expect 0.0 (masked)
                try std.testing.expectEqual(@as(f32, 0.0), score);
            }
        }
    }
}
