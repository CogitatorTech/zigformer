const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Vocab = lib.vocab.Vocab;
const Layer = lib.layer.Layer;
const Embeddings = lib.embeddings.Embeddings;
const TransformerBlock = lib.transformer.TransformerBlock;
const OutputProjection = lib.output_projection.OutputProjection;

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
        for (self.network.items) |layer| {
            layer.deinit();
        }
        self.network.deinit(self.allocator);
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

    fn tokenize(self: *const LLM, text: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32){};
        try tokens.ensureTotalCapacity(self.allocator, text.len / 4);
        var it = std.mem.splitScalar(u8, text, ' ');
        while (it.next()) |word| {
            if (self.vocab.encode(word)) |token_id| {
                try tokens.append(self.allocator, token_id);
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

    pub fn predict(self: *LLM, text: []const u8) ![]u8 {
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
            defer input_matrix.deinit();
            for (tokenized.items, 0..) |tok, i| {
                input_matrix.data[i] = @floatFromInt(tok);
            }

            var temp_matrix = input_matrix;
            for (self.network.items) |*layer| {
                const next_matrix = try layer.forward(temp_matrix);
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
            var next_tokens = try greedyDecode(&last_logit);
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

    fn crossEntropyLoss(probs: *const Matrix, targets: []const u32) f32 {
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

    pub fn train(self: *LLM, data: []const []const u8, epochs: usize, lr: f32) !void {
        var tokenized_data = std.ArrayList(std.ArrayList(u32)){};
        defer {
            for (tokenized_data.items) |*d| d.deinit(self.allocator);
            tokenized_data.deinit(self.allocator);
        }

        for (data) |text| {
            try tokenized_data.append(self.allocator, try self.tokenize(text));
        }

        for (0..epochs) |epoch| {
            var total_loss: f32 = 0.0;
            var processed_count: usize = 0;
            for (tokenized_data.items) |training_row| {
                if (training_row.items.len < 2) continue;
                processed_count += 1;

                const len = @min(training_row.items.len, lib.config.max_seq_len + 1);
                const input_ids = training_row.items[0 .. len - 1];
                const target_ids = training_row.items[1..len];

                var input_matrix = try Matrix.init(self.allocator, 1, input_ids.len);
                defer input_matrix.deinit();
                for (input_ids, 0..) |tok, i| input_matrix.data[i] = @floatFromInt(tok);

                var temp_matrix = input_matrix;
                for (self.network.items) |*layer| {
                    const next_matrix = try layer.forward(temp_matrix);
                    if (temp_matrix.data.ptr != input_matrix.data.ptr) {
                        temp_matrix.deinit();
                    }
                    temp_matrix = next_matrix;
                }
                var logits = temp_matrix;

                softmax(&logits);
                var probs = logits;
                defer probs.deinit();

                total_loss += crossEntropyLoss(&probs, target_ids);

                var grads = try computeGradients(&probs, target_ids);

                var i = self.network.items.len;
                while (i > 0) {
                    i -= 1;
                    const next_grads = try self.network.items[i].backward(grads, lr);
                    grads = next_grads;
                }
                grads.deinit();
            }
            if (processed_count > 0) {
                std.debug.print("Epoch {}: Loss = {:.4}\n", .{ epoch, total_loss / @as(f32, @floatFromInt(processed_count)) });
            } else {
                std.debug.print("Epoch {}: No data processed.\n", .{epoch});
            }
        }
    }
};
