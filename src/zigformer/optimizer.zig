const std = @import("std");
const linalg = @import("linear_algebra.zig");
const Matrix = linalg.Matrix;

pub const Adam = struct {
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    timestep: u32 = 0,
    m: Matrix,
    v: Matrix,
    allocator: std.mem.Allocator,

    clip_threshold: f32 = 1.0,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Adam {
        return Adam{
            .m = try Matrix.initZeros(allocator, rows, cols),
            .v = try Matrix.initZeros(allocator, rows, cols),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Adam) void {
        self.m.deinit();
        self.v.deinit();
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
