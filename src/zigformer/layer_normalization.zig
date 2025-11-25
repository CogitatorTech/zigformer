//! Layer Normalization.
//!
//! Layer normalization normalizes inputs across the feature dimension,
//! computing mean and variance for each example in a mini-batch.
//!
//! Mathematical formulation:
//!  - μ = (1/d) Σ x_i            (mean)
//!  - σ² = (1/d) Σ (x_i - μ)²    (variance)
//!  - x̂ = (x - μ) / sqrt(σ² + ε) (normalize)
//!  - y = γ ⊙ x̂ + β              (scale and shift)
//!
//! where:
//!  - x ∈ ℝ^d is the input
//!  - γ, β ∈ ℝ^d are learnable parameters (scale and shift)
//!  - ε is a small constant for numerical stability
//!  - ⊙ denotes element-wise multiplication
//!
//! References:
//!  - "Layer Normalization" (Ba et al., 2016)

const std = @import("std");
const lib = @import("../lib.zig");
const linalg = lib.linalg;
const Matrix = linalg.Matrix;
const Adam = lib.optimizer.Adam;
const layer = lib.layer;

/// Layer Normalization module.
///
/// Normalizes activations across the feature dimension (last dimension),
/// which helps with training stability and convergence speed. Unlike
///  batch normalization, layer norm is independent across batches.
pub const LayerNorm = struct {
    allocator: std.mem.Allocator,
    gamma: Matrix, // Scale parameter (γ), initialized to 1
    beta: Matrix, // Shift parameter (β), initialized to 0
    optimizer_gamma: Adam,
    optimizer_beta: Adam,
    has_cache: bool,
    cached_input: Matrix,
    cached_mean: Matrix, // Mean for each sample
    cached_inv_std_dev: Matrix, // Inverse standard deviation for each sample
    epsilon: f32 = 1e-5, // Small constant for numerical stability

    /// Initialize a layer normalization module.
    ///
    /// Parameters:
    ///   allocator: Memory allocator
    ///   feature_dim: Dimension of features to normalize
    ///
    /// Returns:
    ///   Initialized LayerNorm struct
    pub fn init(allocator: std.mem.Allocator, feature_dim: usize) !*LayerNorm {
        const self = try allocator.create(LayerNorm);
        const gamma = try Matrix.init(allocator, 1, feature_dim);
        for (gamma.data) |*g| g.* = 1.0; // Initialize scale to 1

        self.* = .{
            .allocator = allocator,
            .gamma = gamma,
            .beta = try Matrix.initZeros(allocator, 1, feature_dim), // Initialize shift to 0
            .optimizer_gamma = try Adam.init(allocator, 1, feature_dim),
            .optimizer_beta = try Adam.init(allocator, 1, feature_dim),
            .has_cache = false,
            .cached_input = undefined,
            .cached_mean = undefined,
            .cached_inv_std_dev = undefined,
        };
        return self;
    }

    /// Deallocate all resources used by this layer norm module.
    pub fn deinit(self: *LayerNorm) void {
        self.gamma.deinit();
        self.beta.deinit();
        self.optimizer_gamma.deinit();
        self.optimizer_beta.deinit();
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_mean.deinit();
            self.cached_inv_std_dev.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Compute layer normalization forward pass.
    ///
    /// For each row (sample) in the input:
    ///   1. Compute mean: μ = (1/d) Σ x_i
    ///   2. Compute variance: σ² = (1/d) Σ (x_i - μ)²
    ///   3. Normalize: x̂_i = (x_i - μ) / sqrt(σ² + ε)
    ///   4. Scale and shift: y_i = γ_i * x̂_i + β_i
    ///
    /// Parameters:
    ///   input: Input tensor of shape (batch, features)
    ///
    /// Returns:
    ///   Normalized output of same shape as input
    pub fn forward(self: *LayerNorm, input: Matrix) !Matrix {
        if (self.has_cache) {
            self.cached_input.deinit();
            self.cached_mean.deinit();
            self.cached_inv_std_dev.deinit();
        }
        self.cached_input = try input.clone();
        self.has_cache = true;

        self.cached_mean = try Matrix.init(self.allocator, input.rows, 1);
        self.cached_inv_std_dev = try Matrix.init(self.allocator, input.rows, 1);
        var output = try Matrix.init(self.allocator, input.rows, input.cols);

        const feature_dim_f32: f32 = @floatFromInt(input.cols);

        for (0..input.rows) |r| {
            var sum: f32 = 0.0;
            const row_slice = input.data[r * input.cols .. (r + 1) * input.cols];
            for (row_slice) |val| {
                sum += val;
            }
            const mean = sum / feature_dim_f32;
            self.cached_mean.data[r] = mean;

            var var_sum: f32 = 0.0;
            for (row_slice) |val| {
                const diff = val - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / feature_dim_f32;
            const inv_std_dev = 1.0 / std.math.sqrt(variance + self.epsilon);
            self.cached_inv_std_dev.data[r] = inv_std_dev;

            const out_row_slice = output.data[r * output.cols .. (r + 1) * output.cols];
            for (row_slice, out_row_slice, 0..) |in_val, *out_val, c| {
                const norm = (in_val - mean) * inv_std_dev;
                out_val.* = norm * self.gamma.data[c] + self.beta.data[c];
            }
        }
        return output;
    }

    pub fn backward(self: *LayerNorm, grads: Matrix, lr: f32) !Matrix {
        var mut_grads = grads;
        defer mut_grads.deinit();

        var grad_gamma = try Matrix.initZeros(self.allocator, 1, self.gamma.cols);
        defer grad_gamma.deinit();
        var grad_beta = try Matrix.initZeros(self.allocator, 1, self.beta.cols);
        defer grad_beta.deinit();
        var grad_input = try Matrix.init(self.allocator, mut_grads.rows, mut_grads.cols);

        const feature_dim_f32: f32 = @floatFromInt(mut_grads.cols);

        for (0..mut_grads.rows) |r| {
            const mean = self.cached_mean.data[r];
            const inv_std_dev = self.cached_inv_std_dev.data[r];
            const in_row = self.cached_input.data[r * mut_grads.cols .. (r + 1) * mut_grads.cols];
            const grad_row = mut_grads.data[r * mut_grads.cols .. (r + 1) * mut_grads.cols];
            const grad_in_row = grad_input.data[r * mut_grads.cols .. (r + 1) * mut_grads.cols];

            var dnorm_sum: f32 = 0;
            var dnorm_dot_norm_sum: f32 = 0;

            for (in_row, grad_row, 0..) |x, grad_out, c| {
                const norm = (x - mean) * inv_std_dev;
                grad_gamma.data[c] += grad_out * norm;
                grad_beta.data[c] += grad_out;

                const dnorm = grad_out * self.gamma.data[c];
                dnorm_sum += dnorm;
                dnorm_dot_norm_sum += dnorm * norm;
            }

            for (in_row, 0..) |x, c| {
                const norm = (x - mean) * inv_std_dev;
                const dnorm = mut_grads.at(r, c) * self.gamma.data[c];
                grad_in_row[c] = (1.0 / feature_dim_f32) * inv_std_dev * (feature_dim_f32 * dnorm - dnorm_sum - norm * dnorm_dot_norm_sum);
            }
        }

        self.optimizer_gamma.step(&self.gamma, grad_gamma, lr);
        self.optimizer_beta.step(&self.beta, grad_beta, lr);

        return grad_input;
    }

    pub fn parameters(self: *const LayerNorm) usize {
        return self.gamma.data.len + self.beta.data.len;
    }

    pub fn toLayer(self: *LayerNorm) layer.Layer {
        return layer.toLayer(LayerNorm)(self);
    }

    pub fn save(self: *const LayerNorm, writer: anytype) !void {
        try self.gamma.save(writer);
        try self.beta.save(writer);
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !*LayerNorm {
        const self = try allocator.create(LayerNorm);
        errdefer allocator.destroy(self);

        var gamma = try Matrix.load(allocator, reader);
        errdefer gamma.deinit();

        var beta = try Matrix.load(allocator, reader);
        errdefer beta.deinit();

        self.* = .{
            .allocator = allocator,
            .gamma = gamma,
            .beta = beta,
            .optimizer_gamma = try Adam.init(allocator, gamma.rows, gamma.cols),
            .optimizer_beta = try Adam.init(allocator, beta.rows, beta.cols),
            .has_cache = false,
            .cached_input = undefined,
            .cached_mean = undefined,
            .cached_inv_std_dev = undefined,
        };
        return self;
    }
};
test "LayerNorm" {
    const allocator = std.testing.allocator;
    const feature_dim = 16;

    var ln = try LayerNorm.init(allocator, feature_dim);
    defer ln.deinit();

    // Test Forward
    var input = try Matrix.initRandom(allocator, 2, feature_dim, 0.0, 1.0);
    defer input.deinit();

    var output = try ln.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(input.rows, output.rows);
    try std.testing.expectEqual(input.cols, output.cols);

    // Test Backward
    const grads = try Matrix.initRandom(allocator, 2, feature_dim, 0.0, 1.0);
    var grad_input = try ln.backward(grads, 0.01);
    defer grad_input.deinit();

    try std.testing.expectEqual(input.rows, grad_input.rows);
    try std.testing.expectEqual(input.cols, grad_input.cols);
}

test "LayerNorm (save and load)" {
    const allocator = std.testing.allocator;
    const feature_dim = 16;

    var ln = try LayerNorm.init(allocator, feature_dim);
    defer ln.deinit();

    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);
    try ln.save(writer);

    var stream = std.io.fixedBufferStream(buffer.items);
    const reader = stream.reader();
    var loaded_ln = try LayerNorm.load(allocator, reader);
    defer loaded_ln.deinit();

    try std.testing.expectEqual(ln.parameters(), loaded_ln.parameters());
}
