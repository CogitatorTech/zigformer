//! Optimization algorithms.
//!
//! Implements Adam (Adaptive Moment Estimation) optimizer.
//!
//! Adam update rules:
//!  - m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           (first moment)
//!  - v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²          (second moment)
//!  - m̂_t = m_t / (1 - β₁^t)                        (bias correction)
//!  - v̂_t = v_t / (1 - β₂^t)                        (bias correction)
//!  - θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)    (parameter update)
//!
//! where:
//!  - g_t is the gradient at time t
//!  - α is the learning rate
//!  - β₁, β₂ are exponential decay rates (typically 0.9, 0.999)
//!  - ε is a small constant for numerical stability (1e-8)
//!
//! References:
//!  - "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015)

const std = @import("std");
const lib = @import("../lib.zig");
const Matrix = lib.linalg.Matrix;

/// Adam optimizer.
///
/// Maintains running averages of gradients and squared gradients for each parameter.
/// Adapts learning rates per-parameter based on gradient statistics.
pub const Adam = struct {
    allocator: std.mem.Allocator,
    /// First moment estimate (momentum)
    m: Matrix,
    /// Second moment estimate (RMSprop)
    v: Matrix,
    /// Time step counter for bias correction
    timestep: usize = 0,
    /// Exponential decay rate for first moment
    beta1: f32 = 0.9,
    /// Exponential decay rate for second moment
    beta2: f32 = 0.999,
    /// Small constant for numerical stability
    epsilon: f32 = 1e-8,

    clip_threshold: f32 = 1.0,

    // Gradient accumulation support
    grad_accumulator: Matrix,
    accumulation_counter: usize = 0,

    /// Initialize Adam optimizer for a parameter matrix.
    ///
    /// Parameters:
    ///   allocator: Memory allocator
    ///   rows: Number of rows in parameter matrix
    ///   cols: Number of columns in parameter matrix
    ///
    /// Returns:
    ///   Initialized Adam optimizer with zero moments
    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Adam {
        return Adam{
            .m = try Matrix.initZeros(allocator, rows, cols),
            .v = try Matrix.initZeros(allocator, rows, cols),
            .grad_accumulator = try Matrix.initZeros(allocator, rows, cols),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Adam) void {
        self.m.deinit();
        self.v.deinit();
        self.grad_accumulator.deinit();
    }

    pub fn step(self: *Adam, params: *Matrix, grads: Matrix, lr: f32) void {
        self.timestep += 1;

        // Gradient Clipping
        var sum_sq: f32 = 0.0;
        for (grads.data) |g| {
            sum_sq += g * g;
        }
        const norm = std.math.sqrt(sum_sq);
        if (norm > self.clip_threshold) {
            const scale = self.clip_threshold / (norm + 1e-6);
            for (grads.data) |*g| {
                g.* *= scale;
            }
        }

        for (self.m.data, grads.data) |*m_val, g_val| {
            m_val.* = self.beta1 * m_val.* + (1.0 - self.beta1) * g_val;
        }

        for (self.v.data, grads.data) |*v_val, g_val| {
            v_val.* = self.beta2 * v_val.* + (1.0 - self.beta2) * (g_val * g_val);
        }

        const beta1_t = std.math.pow(f32, self.beta1, @floatFromInt(self.timestep));
        const beta2_t = std.math.pow(f32, self.beta2, @floatFromInt(self.timestep));

        const m_hat_denom = 1.0 - beta1_t;
        const v_hat_denom = 1.0 - beta2_t;

        for (params.data, self.m.data, self.v.data) |*p, m_val, v_val| {
            const m_hat = m_val / m_hat_denom;
            const v_hat = v_val / v_hat_denom;
            const update = lr * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
            p.* -= update;
        }
    }

    /// Accumulate gradients without updating parameters
    pub fn accumulateGradients(self: *Adam, grads: Matrix, accumulation_steps: usize) void {
        const scale = 1.0 / @as(f32, @floatFromInt(accumulation_steps));
        for (self.grad_accumulator.data, grads.data) |*acc, g| {
            acc.* += g * scale;
        }
        self.accumulation_counter += 1;
    }

    /// Apply accumulated gradients and reset accumulator
    pub fn applyAccumulated(self: *Adam, params: *Matrix, lr: f32) void {
        if (self.accumulation_counter == 0) return;

        self.timestep += 1;

        // Gradient Clipping on accumulated gradients
        var sum_sq: f32 = 0.0;
        for (self.grad_accumulator.data) |g| {
            sum_sq += g * g;
        }
        const norm = std.math.sqrt(sum_sq);
        if (norm > self.clip_threshold) {
            const scale = self.clip_threshold / (norm + 1e-6);
            for (self.grad_accumulator.data) |*g| {
                g.* *= scale;
            }
        }

        for (self.m.data, self.grad_accumulator.data) |*m_val, g_val| {
            m_val.* = self.beta1 * m_val.* + (1.0 - self.beta1) * g_val;
        }

        for (self.v.data, self.grad_accumulator.data) |*v_val, g_val| {
            v_val.* = self.beta2 * v_val.* + (1.0 - self.beta2) * (g_val * g_val);
        }

        const beta1_t = std.math.pow(f32, self.beta1, @floatFromInt(self.timestep));
        const beta2_t = std.math.pow(f32, self.beta2, @floatFromInt(self.timestep));

        const m_hat_denom = 1.0 - beta1_t;
        const v_hat_denom = 1.0 - beta2_t;

        for (params.data, self.m.data, self.v.data) |*p, m_val, v_val| {
            const m_hat = m_val / m_hat_denom;
            const v_hat = v_val / v_hat_denom;
            const update = lr * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
            p.* -= update;
        }

        // Reset accumulator
        for (self.grad_accumulator.data) |*acc| {
            acc.* = 0.0;
        }
        self.accumulation_counter = 0;
    }
};

test "Adam step" {
    const allocator = std.testing.allocator;
    var adam = try Adam.init(allocator, 1, 4);
    defer adam.deinit();

    var params = try Matrix.init(allocator, 1, 4);
    defer params.deinit();
    params.data[0] = 0.1;
    params.data[1] = 0.2;
    params.data[2] = 0.3;
    params.data[3] = 0.4;

    var grads = try Matrix.init(allocator, 1, 4);
    defer grads.deinit();
    grads.data[0] = 0.01;
    grads.data[1] = 0.01;
    grads.data[2] = 0.01;
    grads.data[3] = 0.01;

    const initial_params = try allocator.alloc(f32, 4);
    defer allocator.free(initial_params);
    @memcpy(initial_params, params.data);

    adam.step(&params, grads, 0.001);

    for (params.data, initial_params) |p, initial_p| {
        try std.testing.expect(p < initial_p);
    }
}
