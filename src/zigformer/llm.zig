//! Large Language Model (LLM).
//!
//! GPT-style autoregressive language model with:
//! - Token and positional embeddings
//! - Multiple transformer blocks with self-attention
//! - Output projection to vocabulary
//! - Training with cross-entropy loss
//! - Multiple inference strategies (greedy, sampling, beam search)
//!
//! Model Architecture:
//!   Input tokens
//!     ↓
//!   Embeddings (token + position)
//!     ↓
//!   TransformerBlock × N  (self-attention + FFN)
//!     ↓
//!   OutputProjection (→ vocabulary logits)
//!     ↓
//!   Softmax / Sampling
//!
//! Training:
//!   - Cross-entropy loss with next-token prediction
//!   - Adam optimizer with learning rate scheduling
//!   - Gradient accumulation for large effective batch sizes
//!
//! Inference:
//!   - Greedy decoding: argmax at each step
//!   - Top-k/top-p sampling: sample from filtered distribution
//!   - Beam search: maintain k best sequences

const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Vocab = lib.vocab.Vocab;
const Layer = lib.layer.Layer;
const Embeddings = lib.embeddings.Embeddings;
const TransformerBlock = lib.transformer.TransformerBlock;
const OutputProjection = lib.output_projection.OutputProjection;

/// Beam search node for maintaining candidate sequences.
///
/// Used during beam search decoding to track partial sequences,
/// their cumulative log probabilities, and completion status.
pub const BeamNode = struct {
    sequence: std.ArrayListUnmanaged(u32), // Token IDs in this sequence
    score: f32, // Cumulative log probability
    finished: bool, // Whether sequence ended with </s>

    pub fn init(allocator: std.mem.Allocator, initial_seq: []const u32, initial_score: f32) !BeamNode {
        var seq = std.ArrayListUnmanaged(u32){};
        try seq.appendSlice(allocator, initial_seq);
        return BeamNode{
            .sequence = seq,
            .score = initial_score,
            .finished = false,
        };
    }

    pub fn deinit(self: *BeamNode, allocator: std.mem.Allocator) void {
        self.sequence.deinit(allocator);
    }

    pub fn clone(self: *const BeamNode, allocator: std.mem.Allocator) !BeamNode {
        var seq = std.ArrayListUnmanaged(u32){};
        try seq.appendSlice(allocator, self.sequence.items);
        return BeamNode{
            .sequence = seq,
            .score = self.score,
            .finished = self.finished,
        };
    }
};

fn compareBeamNodes(_: void, lhs: BeamNode, rhs: BeamNode) bool {
    return lhs.score > rhs.score; // Descending order
}

/// Large Language Model with transformer architecture.
///
/// A complete autoregressive language model for text generation.
/// Supports training, inference, and model persistence (save/load).
pub const LLM = struct {
    allocator: std.mem.Allocator,
    vocab: Vocab,
    network: std.ArrayList(Layer),

    pub fn init(allocator: std.mem.Allocator, vocab: Vocab) !LLM {
        var network = std.ArrayList(Layer){};
        errdefer {
            for (network.items) |layer| {
                layer.deinit();
            }
            network.deinit(allocator);
        }

        const embedding_dim = lib.config.embedding_dim;
        const hidden_dim = lib.config.hidden_dim;

        const embeddings = try Embeddings.init(allocator, vocab.size());
        try network.append(allocator, embeddings.toLayer());

        const transformer1 = try TransformerBlock.init(allocator, embedding_dim, hidden_dim);
        try network.append(allocator, transformer1.toLayer());

        const transformer2 = try TransformerBlock.init(allocator, embedding_dim, hidden_dim);
        try network.append(allocator, transformer2.toLayer());

        const transformer3 = try TransformerBlock.init(allocator, embedding_dim, hidden_dim);
        try network.append(allocator, transformer3.toLayer());

        const output_projection = try OutputProjection.init(allocator, embedding_dim, vocab.size());
        try network.append(allocator, output_projection.toLayer());

        return LLM{
            .allocator = allocator,
            .vocab = vocab,
            .network = network,
        };
    }

    pub fn deinit(self: *LLM) void {
        self.vocab.deinit();
        for (self.network.items) |layer| {
            layer.deinit();
        }
        self.network.deinit(self.allocator);
    }

    pub fn resetCache(self: *LLM) void {
        for (self.network.items) |*layer| {
            layer.resetCache();
        }
    }

    pub fn networkDescription(self: *const LLM) []const u8 {
        _ = self;
        return "Embeddings, TransformerBlock, TransformerBlock, TransformerBlock, OutputProjection";
    }

    pub fn totalParameters(self: *const LLM) usize {
        var total: usize = 0;
        for (self.network.items) |layer| {
            total += layer.parameters();
        }
        return total;
    }

    pub fn tokenize(self: *const LLM, text: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32){};
        try tokens.ensureTotalCapacity(self.allocator, text.len / 4);
        var it = std.mem.splitScalar(u8, text, ' ');
        while (it.next()) |word| {
            // Special case for end token
            if (std.mem.eql(u8, word, "</s>")) {
                if (self.vocab.encode(word)) |token_id| {
                    try tokens.append(self.allocator, token_id);
                }
                continue;
            }

            var start: usize = 0;
            for (word, 0..) |c, i| {
                if (std.ascii.isPrint(c) and !std.ascii.isAlphanumeric(c)) {
                    // If we have a word before the punctuation, add it
                    if (i > start) {
                        const sub_word = word[start..i];
                        if (self.vocab.encode(sub_word)) |token_id| {
                            try tokens.append(self.allocator, token_id);
                        }
                    }

                    // Add the punctuation as its own token
                    const punct_slice = word[i .. i + 1];
                    if (self.vocab.encode(punct_slice)) |token_id| {
                        try tokens.append(self.allocator, token_id);
                    }

                    start = i + 1;
                }
            }

            // Add any remaining word
            if (start < word.len) {
                const sub_word = word[start..];
                if (self.vocab.encode(sub_word)) |token_id| {
                    try tokens.append(self.allocator, token_id);
                }
            }
        }
        return tokens;
    }

    fn softmax(logits: *Matrix) void {
        for (0..logits.rows) |r| {
            var max_val = logits.data[r * logits.cols];
            for (1..logits.cols) |c| {
                max_val = @max(max_val, logits.data[r * logits.cols + c]);
            }

            var sum_exp: f32 = 0.0;
            const row_slice = logits.data[r * logits.cols .. (r + 1) * logits.cols];
            for (row_slice) |*val| {
                const exp_val = std.math.exp(val.* - max_val);
                val.* = exp_val;
                sum_exp += exp_val;
            }

            if (sum_exp > 0) {
                for (row_slice) |*val| {
                    val.* /= sum_exp;
                }
            }
        }
    }

    fn greedyDecode(probs: *const Matrix) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32){};
        try tokens.ensureTotalCapacity(probs.allocator, probs.rows);
        for (0..probs.rows) |r| {
            var max_idx: u32 = 0;
            var max_prob: f32 = 0.0;
            for (0..probs.cols) |c| {
                const prob = probs.at(r, c);
                if (prob > max_prob) {
                    max_prob = prob;
                    max_idx = @intCast(c);
                }
            }
            try tokens.append(probs.allocator, max_idx);
        }
        return tokens;
    }

    fn topKSampling(probs: *const Matrix, k: usize, allocator: std.mem.Allocator) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32){};
        try tokens.ensureTotalCapacity(allocator, probs.rows);

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();

        for (0..probs.rows) |r| {
            // Create array of (index, prob) pairs
            var indices = try allocator.alloc(struct { idx: u32, prob: f32 }, probs.cols);
            defer allocator.free(indices);

            for (0..probs.cols) |c| {
                indices[c] = .{ .idx = @intCast(c), .prob = probs.at(r, c) };
            }

            // Sort by probability (descending)
            std.sort.pdq(@TypeOf(indices[0]), indices, {}, struct {
                fn lessThan(_: void, a: @TypeOf(indices[0]), b: @TypeOf(indices[0])) bool {
                    return a.prob > b.prob;
                }
            }.lessThan);

            // Keep only top-k
            const top_k = @min(k, probs.cols);

            // Renormalize probabilities
            var sum: f32 = 0.0;
            for (0..top_k) |i| {
                sum += indices[i].prob;
            }

            // Sample from top-k
            const rand_val = random.float(f32) * sum;
            var cumsum: f32 = 0.0;
            var selected_idx: u32 = indices[0].idx;

            for (0..top_k) |i| {
                cumsum += indices[i].prob;
                if (rand_val <= cumsum) {
                    selected_idx = indices[i].idx;
                    break;
                }
            }

            try tokens.append(allocator, selected_idx);
        }
        return tokens;
    }

    fn topPSampling(probs: *const Matrix, p: f32, allocator: std.mem.Allocator) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32){};
        try tokens.ensureTotalCapacity(allocator, probs.rows);

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();

        for (0..probs.rows) |r| {
            // Create array of (index, prob) pairs
            var indices = try allocator.alloc(struct { idx: u32, prob: f32 }, probs.cols);
            defer allocator.free(indices);

            for (0..probs.cols) |c| {
                indices[c] = .{ .idx = @intCast(c), .prob = probs.at(r, c) };
            }

            // Sort by probability (descending)
            std.sort.pdq(@TypeOf(indices[0]), indices, {}, struct {
                fn lessThan(_: void, a: @TypeOf(indices[0]), b: @TypeOf(indices[0])) bool {
                    return a.prob > b.prob;
                }
            }.lessThan);

            // Find cutoff for nucleus (top-p)
            var cumsum: f32 = 0.0;
            var cutoff: usize = 0;
            for (0..probs.cols) |i| {
                cumsum += indices[i].prob;
                cutoff = i + 1;
                if (cumsum >= p) break;
            }

            // Renormalize probabilities in nucleus
            var sum: f32 = 0.0;
            for (0..cutoff) |i| {
                sum += indices[i].prob;
            }

            // Sample from nucleus
            const rand_val = random.float(f32) * sum;
            cumsum = 0.0;
            var selected_idx: u32 = indices[0].idx;

            for (0..cutoff) |i| {
                cumsum += indices[i].prob;
                if (rand_val <= cumsum) {
                    selected_idx = indices[i].idx;
                    break;
                }
            }

            try tokens.append(allocator, selected_idx);
        }
        return tokens;
    }

    pub const SamplingMode = enum {
        greedy,
        topk,
        topp,
    };

    pub fn beamSearch(self: *LLM, text: []const u8, beam_width: usize, max_new_tokens: usize) ![]u8 {
        self.setBatchSize(1);
        var tokenized = try self.tokenize(text);
        defer tokenized.deinit(self.allocator);

        if (tokenized.items.len == 0) {
            return self.allocator.dupe(u8, "");
        }

        var beam = std.ArrayListUnmanaged(BeamNode){};
        defer {
            for (beam.items) |*node| node.deinit(self.allocator);
            beam.deinit(self.allocator);
        }

        // Initialize beam with the prompt
        try beam.append(self.allocator, try BeamNode.init(self.allocator, tokenized.items, 0.0));

        const end_token = self.vocab.encode("</s>").?;

        for (0..max_new_tokens) |_| {
            var candidates = std.ArrayListUnmanaged(BeamNode){};
            defer {
                for (candidates.items) |*node| node.deinit(self.allocator);
                candidates.deinit(self.allocator);
            }

            var all_finished = true;

            for (beam.items) |*node| {
                if (node.finished) {
                    try candidates.append(self.allocator, try node.clone(self.allocator));
                    continue;
                }
                all_finished = false;

                if (node.sequence.items.len >= lib.config.max_seq_len) {
                    var finished_node = try node.clone(self.allocator);
                    finished_node.finished = true;
                    try candidates.append(self.allocator, finished_node);
                    continue;
                }

                // Run forward pass
                var input_matrix = try Matrix.init(self.allocator, 1, node.sequence.items.len);
                errdefer input_matrix.deinit();
                for (node.sequence.items, 0..) |tok, i| {
                    input_matrix.data[i] = @floatFromInt(tok);
                }

                var temp_matrix = input_matrix;
                for (self.network.items) |*layer| {
                    const next_matrix = try layer.forward(temp_matrix, false);
                    if (temp_matrix.data.ptr != input_matrix.data.ptr) {
                        temp_matrix.deinit();
                    }
                    temp_matrix = next_matrix;
                }
                var logits = temp_matrix;

                var last_logit = try logits.getRow(logits.rows - 1);
                logits.deinit();
                defer last_logit.deinit();

                // LogSoftmax for numerical stability in beam search
                // log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))
                var max_val: f32 = -std.math.inf(f32);
                for (last_logit.data) |val| max_val = @max(max_val, val);

                var sum_exp: f32 = 0.0;
                for (last_logit.data) |val| sum_exp += std.math.exp(val - max_val);
                const log_sum_exp = std.math.log(f32, sum_exp, std.math.e);

                for (last_logit.data) |*val| {
                    val.* = (val.* - max_val) - log_sum_exp;
                }

                // Select top beam_width candidates
                const TopToken = struct { idx: u32, score: f32 };
                var top_tokens = std.ArrayListUnmanaged(TopToken){};
                defer top_tokens.deinit(self.allocator);

                // Simple top-k selection
                for (last_logit.data, 0..) |score, idx| {
                    try top_tokens.append(self.allocator, .{ .idx = @intCast(idx), .score = score });
                }

                // Sort by score descending
                std.sort.pdq(TopToken, top_tokens.items, {}, struct {
                    fn lessThan(_: void, lhs: TopToken, rhs: TopToken) bool {
                        return lhs.score > rhs.score;
                    }
                }.lessThan);

                const num_candidates = @min(beam_width, top_tokens.items.len);
                for (0..num_candidates) |i| {
                    const token = top_tokens.items[i];
                    var new_node = try node.clone(self.allocator);
                    try new_node.sequence.append(self.allocator, token.idx);
                    new_node.score += token.score;
                    if (token.idx == end_token) {
                        new_node.finished = true;
                    }
                    try candidates.append(self.allocator, new_node);
                }
            }

            if (all_finished) break;

            // Select best candidates for next beam
            std.sort.pdq(BeamNode, candidates.items, {}, compareBeamNodes);

            // Clear current beam and take top beam_width from candidates
            for (beam.items) |*node| node.deinit(self.allocator);
            beam.clearRetainingCapacity();

            const next_beam_size = @min(beam_width, candidates.items.len);
            for (0..next_beam_size) |i| {
                try beam.append(self.allocator, try candidates.items[i].clone(self.allocator));
            }
        }

        // Return best sequence (excluding prompt)
        const best_node = beam.items[0];
        var result_builder = std.ArrayList(u8){};
        defer result_builder.deinit(self.allocator);

        // Skip prompt tokens
        const prompt_len = tokenized.items.len;
        for (best_node.sequence.items[prompt_len..], 0..) |tok, i| {
            if (self.vocab.decode(tok)) |word| {
                if (i > 0) try result_builder.append(self.allocator, ' ');
                try result_builder.appendSlice(self.allocator, word);
            }
        }
        return result_builder.toOwnedSlice(self.allocator);
    }

    pub fn predictWithSampling(self: *LLM, text: []const u8, mode: SamplingMode, k: usize, p: f32) ![]u8 {
        self.setBatchSize(1);
        var tokenized = try self.tokenize(text);
        defer tokenized.deinit(self.allocator);

        var output_tokens = std.ArrayList(u32){};
        defer output_tokens.deinit(self.allocator);

        const input_len = tokenized.items.len;
        if (input_len == 0 or input_len >= lib.config.max_seq_len) {
            return self.allocator.dupe(u8, "");
        }

        const end_token = self.vocab.encode("</s>").?;

        for (0..(lib.config.max_seq_len - input_len)) |_| {
            var input_matrix = try Matrix.init(self.allocator, 1, tokenized.items.len);
            errdefer input_matrix.deinit();
            for (tokenized.items, 0..) |tok, i| {
                input_matrix.data[i] = @floatFromInt(tok);
            }

            var temp_matrix = input_matrix;
            for (self.network.items) |*layer| {
                const next_matrix = try layer.forward(temp_matrix, false);
                if (temp_matrix.data.ptr != input_matrix.data.ptr) {
                    temp_matrix.deinit();
                }
                temp_matrix = next_matrix;
            }

            var logits = temp_matrix;

            var last_logit = try logits.getRow(logits.rows - 1);
            logits.deinit();
            defer last_logit.deinit();

            softmax(&last_logit);

            // Use appropriate sampling method
            var next_tokens = switch (mode) {
                .greedy => try greedyDecode(&last_logit),
                .topk => try topKSampling(&last_logit, k, self.allocator),
                .topp => try topPSampling(&last_logit, p, self.allocator),
            };
            defer next_tokens.deinit(self.allocator);

            const next_token = next_tokens.items[0];
            try output_tokens.append(self.allocator, next_token);
            try tokenized.append(self.allocator, next_token);

            if (next_token == end_token) break;
        }

        var result_builder = std.ArrayList(u8){};
        defer result_builder.deinit(self.allocator);
        for (output_tokens.items, 0..) |tok, i| {
            if (self.vocab.decode(tok)) |word| {
                if (i > 0) try result_builder.append(self.allocator, ' ');
                try result_builder.appendSlice(self.allocator, word);
            }
        }
        return result_builder.toOwnedSlice(self.allocator);
    }

    pub fn forward(self: *LLM, input: Matrix, use_cache: bool) !Matrix {
        var current = input;
        var i: usize = 0;
        for (self.network.items) |*layer| {
            const next = try layer.forward(current, use_cache);
            if (i > 0) {
                current.deinit();
            }
            current = next;
            i += 1;
        }
        return current;
    }

    fn sampleGreedy(self: *const LLM, logits: Matrix) u32 {
        _ = self;
        var max_idx: u32 = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (logits.data, 0..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = @intCast(i);
            }
        }
        return max_idx;
    }

    pub fn predict(self: *LLM, text: []const u8) ![]u8 {
        self.setBatchSize(1);
        var tokenized = try self.tokenize(text);
        defer tokenized.deinit(self.allocator);

        var output_tokens = std.ArrayList(u32){};
        defer output_tokens.deinit(self.allocator);

        const input_len = tokenized.items.len;
        if (input_len == 0 or input_len >= lib.config.max_seq_len) {
            return self.allocator.dupe(u8, "");
        }

        const end_token = self.vocab.encode("</s>").?;

        // Reset cache
        self.resetCache();

        // Process prompt (fill cache)
        // We run the prompt through the model with use_cache=true.
        // The model will compute K/V for all prompt tokens and store them.
        // We only care about the last token's output for the next prediction.

        var input_matrix = try Matrix.init(self.allocator, 1, tokenized.items.len);
        errdefer input_matrix.deinit();
        for (tokenized.items, 0..) |tok, i| {
            input_matrix.data[i] = @floatFromInt(tok);
        }

        var logits = try self.forward(input_matrix, true); // use_cache=true
        input_matrix.deinit();

        var last_logit = try logits.getRow(logits.rows - 1);
        logits.deinit();

        // Sample first token
        var next_token = self.sampleGreedy(last_logit);
        last_logit.deinit();

        if (next_token != end_token) {
            try output_tokens.append(self.allocator, next_token);

            // Generation loop
            for (0..(lib.config.max_seq_len - input_len - 1)) |_| {
                // Prepare input for next step (single token)
                input_matrix = try Matrix.init(self.allocator, 1, 1);
                errdefer input_matrix.deinit();
                input_matrix.data[0] = @floatFromInt(next_token);

                // Run forward with cache
                logits = try self.forward(input_matrix, true);
                input_matrix.deinit();

                last_logit = try logits.getRow(logits.rows - 1);
                logits.deinit();

                next_token = self.sampleGreedy(last_logit);
                last_logit.deinit();

                if (next_token == end_token) break;
                try output_tokens.append(self.allocator, next_token);
            }
        }

        var result_builder = std.ArrayList(u8){};
        defer result_builder.deinit(self.allocator);
        for (output_tokens.items, 0..) |tok, i| {
            if (self.vocab.decode(tok)) |word| {
                if (i > 0) try result_builder.append(self.allocator, ' ');
                try result_builder.appendSlice(self.allocator, word);
            }
        }
        return result_builder.toOwnedSlice(self.allocator);
    }

    fn crossEntropyLoss(probs: *const Matrix, targets: []const u32) f32 {
        if (targets.len == 0) return 0.0; // Avoid division by zero
        var loss: f32 = 0.0;
        for (targets, 0..) |target_id, i| {
            const prob_target = probs.at(i, target_id);
            loss -= std.math.log(f32, std.math.e, @max(1e-15, prob_target));
        }
        return loss / @as(f32, @floatFromInt(targets.len));
    }

    fn computeGradients(probs: *const Matrix, targets: []const u32) !Matrix {
        var grads = try probs.clone();
        const batch_size: f32 = @floatFromInt(targets.len);
        for (0..grads.rows) |r| {
            if (r < targets.len) {
                grads.data[r * grads.cols + targets[r]] -= 1.0;
            }
        }
        for (grads.data) |*g| {
            g.* /= batch_size;
        }
        return grads;
    }

    pub fn setBatchSize(self: *LLM, batch_size: usize) void {
        // Iterate over layers and set batch_size for SelfAttention layers
        // Note: We need to know which layers are SelfAttention.
        // In our simple structure, we know layers 1, 2, 3 are TransformerBlocks.
        // TransformerBlock contains SelfAttention.
        // But Layer is type-erased.
        // Ideally, we should add setBatchSize to Layer vtable, but that's a big change.
        // For now, we'll rely on the known structure and pointer casting, which is risky but fits the current style.
        // Actually, TransformerBlock has a setBatchSize method we should add.
        // Let's assume we add setBatchSize to TransformerBlock and call it here.

        // Wait, we can't easily cast opaque pointers back to types without RTTI or knowing the type.
        // Given the fixed structure:
        // Layer 0: Embeddings (no batch_size needed)
        // Layer 1: TransformerBlock
        // Layer 2: TransformerBlock
        // Layer 3: TransformerBlock
        // Layer 4: OutputProjection (no batch_size needed)

        const embeddings: *Embeddings = @ptrCast(@alignCast(self.network.items[0].self));
        embeddings.setBatchSize(batch_size);

        for (self.network.items[1..4]) |*layer| {
            const tb: *TransformerBlock = @ptrCast(@alignCast(layer.self));
            tb.setBatchSize(batch_size);
        }
    }

    pub fn train(self: *LLM, data: []const []const u8, epochs: usize, lr: f32, batch_size: usize, accumulation_steps: usize) !void {
        self.setBatchSize(batch_size);

        // Effective learning rate for gradient accumulation
        const effective_lr = lr / @as(f32, @floatFromInt(accumulation_steps));

        var tokenized_data = std.ArrayList(std.ArrayList(u32)){};
        defer {
            for (tokenized_data.items) |*d| d.deinit(self.allocator);
            tokenized_data.deinit(self.allocator);
        }

        for (data) |text| {
            try tokenized_data.append(self.allocator, try self.tokenize(text));
        }

        // Simple shuffle could be added here

        for (0..epochs) |epoch| {
            var total_loss: f32 = 0.0;
            var processed_batches: usize = 0;
            var accumulation_counter: usize = 0;

            var i: usize = 0;
            while (i < tokenized_data.items.len) : (i += batch_size) {
                const end = @min(i + batch_size, tokenized_data.items.len);
                const batch = tokenized_data.items[i..end];
                if (batch.len == 0) continue;

                // Determine max length in this batch (up to max_seq_len)
                var max_len: usize = 0;
                for (batch) |seq| {
                    max_len = @max(max_len, @min(seq.items.len, lib.config.max_seq_len + 1));
                }
                if (max_len < 2) continue; // Skip if too short

                const current_batch_size = batch.len;
                // If last batch is smaller, we need to update batch_size temporarily
                if (current_batch_size != batch_size) {
                    self.setBatchSize(current_batch_size);
                }

                const seq_len = max_len - 1; // Input length
                const total_tokens = current_batch_size * seq_len;

                // Prepare input matrix (batch_size * seq_len)
                var input_matrix = try Matrix.init(self.allocator, 1, total_tokens);
                defer input_matrix.deinit();

                // Prepare targets
                var targets = try self.allocator.alloc(u32, total_tokens);
                defer self.allocator.free(targets);

                // Fill inputs and targets with padding (0) if necessary
                for (batch, 0..) |seq, b| {
                    const len = @min(seq.items.len, lib.config.max_seq_len + 1);
                    const input_ids = seq.items[0 .. len - 1];
                    const target_ids = seq.items[1..len];

                    for (0..seq_len) |t| {
                        const global_idx = b * seq_len + t;
                        if (t < input_ids.len) {
                            input_matrix.data[global_idx] = @floatFromInt(input_ids[t]);
                            targets[global_idx] = target_ids[t];
                        } else {
                            // Padding
                            input_matrix.data[global_idx] = 0; // Pad with 0
                            targets[global_idx] = 0; // Pad target with 0 (should be ignored in loss)
                        }
                    }
                }

                var temp_matrix = input_matrix;
                for (self.network.items) |*layer| {
                    const next_matrix = try layer.forward(temp_matrix, false);
                    if (temp_matrix.data.ptr != input_matrix.data.ptr) {
                        temp_matrix.deinit();
                    }
                    temp_matrix = next_matrix;
                }
                var logits = temp_matrix;

                softmax(&logits);
                var probs = logits;
                defer probs.deinit();

                // Compute loss
                total_loss += crossEntropyLoss(&probs, targets);

                var grads = try computeGradients(&probs, targets);

                // Backward pass with effective learning rate
                var layer_idx = self.network.items.len;
                while (layer_idx > 0) {
                    layer_idx -= 1;
                    const next_grads = try self.network.items[layer_idx].backward(grads, effective_lr);
                    grads = next_grads;
                }
                grads.deinit();

                accumulation_counter += 1;

                // Track processed batches (count every accumulation_steps batches as one effective batch)
                if (accumulation_counter >= accumulation_steps) {
                    processed_batches += 1;
                    accumulation_counter = 0;
                }

                // Restore batch size if changed
                if (current_batch_size != batch_size) {
                    self.setBatchSize(batch_size);
                }
            }

            if (processed_batches > 0) {
                std.debug.print("Epoch {}: Loss = {:.4}\n", .{ epoch, total_loss / @as(f32, @floatFromInt(processed_batches * accumulation_steps)) });
            } else {
                std.debug.print("Epoch {}: No data processed.\n", .{epoch});
            }
        }
    }

    pub fn save(self: *const LLM, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        const writer = buffer.writer(self.allocator);

        // Write magic number "ZGFM"
        try writer.writeAll("ZGFM");

        // Write version
        const version: u32 = 1;
        try writer.writeAll(std.mem.asBytes(&version));

        // Save vocabulary
        try self.vocab.save(writer);

        // Save Embeddings (Layer 0)
        const embeddings_ptr: *const Embeddings = @ptrCast(@alignCast(self.network.items[0].self));
        try embeddings_ptr.save(writer);

        // Save Transformer Blocks (Layers 1, 2, 3)
        const transformer1_ptr: *const TransformerBlock = @ptrCast(@alignCast(self.network.items[1].self));
        try transformer1_ptr.save(writer);

        const transformer2_ptr: *const TransformerBlock = @ptrCast(@alignCast(self.network.items[2].self));
        try transformer2_ptr.save(writer);

        const transformer3_ptr: *const TransformerBlock = @ptrCast(@alignCast(self.network.items[3].self));
        try transformer3_ptr.save(writer);

        // Save Output Projection (Layer 4)
        const output_ptr: *const OutputProjection = @ptrCast(@alignCast(self.network.items[4].self));
        try output_ptr.save(writer);

        try file.writeAll(buffer.items);
        std.debug.print("Model saved to {s}\n", .{path});
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !LLM {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const buffer = try allocator.alloc(u8, file_size);
        defer allocator.free(buffer);
        _ = try file.readAll(buffer);

        var stream = std.io.fixedBufferStream(buffer);
        const reader = stream.reader();

        // Read and verify magic number
        var magic: [4]u8 = undefined;
        _ = try reader.readAll(&magic);
        if (!std.mem.eql(u8, &magic, "ZGFM")) {
            return error.InvalidModelFile;
        }

        // Read version
        var version_bytes: [4]u8 = undefined;
        _ = try reader.readAll(&version_bytes);
        const version = std.mem.readInt(u32, &version_bytes, .little);
        if (version != 1) {
            return error.UnsupportedModelVersion;
        }

        // Load vocabulary
        var vocab = try Vocab.load(allocator, reader);
        errdefer vocab.deinit();

        var network = std.ArrayList(Layer){};
        errdefer {
            for (network.items) |layer| {
                layer.deinit();
            }
            network.deinit(allocator);
        }

        // Load Embeddings
        const embeddings = try Embeddings.load(allocator, reader);
        try network.append(allocator, embeddings.toLayer());

        // Load Transformer Blocks
        const transformer1 = try TransformerBlock.load(allocator, reader);
        try network.append(allocator, transformer1.toLayer());

        const transformer2 = try TransformerBlock.load(allocator, reader);
        try network.append(allocator, transformer2.toLayer());

        const transformer3 = try TransformerBlock.load(allocator, reader);
        try network.append(allocator, transformer3.toLayer());

        // Load Output Projection
        const output_projection = try OutputProjection.load(allocator, reader);
        try network.append(allocator, output_projection.toLayer());

        std.debug.print("Model loaded from {s}\n", .{path});

        return LLM{
            .allocator = allocator,
            .vocab = vocab,
            .network = network,
        };
    }
};

test "LLM (save and load)" {
    const allocator = std.testing.allocator;

    // Create a small vocab
    var vocab = Vocab.init(allocator);
    // defer vocab.deinit(); // LLM takes ownership
    const words = &[_][]const u8{ "hello", "world", "</s>" };
    try vocab.build(words);

    // Init model
    var model = try LLM.init(allocator, vocab);
    defer model.deinit();

    // Save model
    const test_path = "test_model.bin";
    try model.save(test_path);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    // Load model
    var loaded_model = try LLM.load(allocator, test_path);
    defer loaded_model.deinit();

    // Compare total parameters
    try std.testing.expectEqual(model.totalParameters(), loaded_model.totalParameters());

    // Compare predictions (should be identical)
    const input_text = "hello";
    const output1 = try model.predict(input_text);
    defer allocator.free(output1);

    const output2 = try loaded_model.predict(input_text);
    defer allocator.free(output2);

    try std.testing.expectEqualStrings(output1, output2);
}

test "LLM tokenizer" {
    const allocator = std.testing.allocator;
    var vocab = Vocab.init(allocator);
    // Add words and punctuation to vocab manually for testing
    // We pass space-separated tokens so vocab.build adds them correctly
    const training_data = &[_][]const u8{ "hello", ",", "world", "!", "</s>" };
    try vocab.build(training_data);

    // Init model manually
    var model = LLM{
        .allocator = allocator,
        .vocab = vocab,
        .network = std.ArrayList(Layer){},
    };
    defer model.deinit();

    const text = "hello, world!";
    var tokens = try model.tokenize(text);
    defer tokens.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 4), tokens.items.len);
    try std.testing.expectEqual(model.vocab.encode("hello").?, tokens.items[0]);
    try std.testing.expectEqual(model.vocab.encode(",").?, tokens.items[1]);
    try std.testing.expectEqual(model.vocab.encode("world").?, tokens.items[2]);
    try std.testing.expectEqual(model.vocab.encode("!").?, tokens.items[3]);
}
