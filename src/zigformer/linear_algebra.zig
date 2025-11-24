const std = @import("std");

pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        return Matrix{
            .rows = rows,
            .cols = cols,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn initZeros(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const mat = try init(allocator, rows, cols);
        @memset(mat.data, 0.0);
        return mat;
    }

    pub fn initRandom(
        allocator: std.mem.Allocator,
        rows: usize,
        cols: usize,
        mean: f32,
        std_dev: f32,
    ) !Matrix {
        const mat = try init(allocator, rows, cols);
        const seed = @as(u64, @intCast(std.time.nanoTimestamp()));
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        for (mat.data) |*val| {
            val.* = rand.float(f32) * std_dev + mean;
        }
        return mat;
    }

    pub fn deinit(self: *Matrix) void {
        self.allocator.free(self.data);
    }

    pub fn clone(self: *const Matrix) !Matrix {
        const new_data = try self.allocator.dupe(f32, self.data);
        return .{
            .rows = self.rows,
            .cols = self.cols,
            .data = new_data,
            .allocator = self.allocator,
        };
    }

    pub fn at(self: Matrix, row: usize, col: usize) f32 {
        std.debug.assert(row < self.rows);
        std.debug.assert(col < self.cols);
        return self.data[row * self.cols + col];
    }

    pub fn set(self: *Matrix, row: usize, col: usize, value: f32) void {
        std.debug.assert(row < self.rows);
        std.debug.assert(col < self.cols);
        self.data[row * self.cols + col] = value;
    }

    pub fn getRow(self: *const Matrix, row: usize) !Matrix {
        const mat = try Matrix.init(self.allocator, 1, self.cols);
        const start = row * self.cols;
        @memcpy(mat.data, self.data[start .. start + self.cols]);
        return mat;
    }

    pub fn add(self: *const Matrix, other: *const Matrix) !Matrix {
        std.debug.assert(self.rows == other.rows and self.cols == other.cols);
        const result = try self.clone();
        for (result.data, other.data) |*r, o| {
            r.* += o;
        }
        return result;
    }

    pub fn dot(self: *const Matrix, other: *const Matrix) !Matrix {
        std.debug.assert(self.cols == other.rows);
        var result = try Matrix.init(self.allocator, self.rows, other.cols);
        @memset(result.data, 0.0);
        for (0..self.rows) |i| {
            for (0..other.cols) |j| {
                var sum: f32 = 0.0;
                for (0..self.cols) |k| {
                    sum += self.at(i, k) * other.at(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    pub fn transpose(self: *const Matrix) !Matrix {
        var result = try Matrix.init(self.allocator, self.cols, self.rows);
        for (0..self.rows) |r| {
            for (0..self.cols) |c| {
                result.set(c, r, self.at(r, c));
            }
        }
        return result;
    }

    // SIMD-optimized matrix addition
    pub fn addSIMD(self: *const Matrix, other: *const Matrix) !Matrix {
        std.debug.assert(self.rows == other.rows and self.cols == other.cols);
        const result = try self.clone();

        const Vec4 = @Vector(4, f32);
        const total_len = result.data.len;
        const vec_len = total_len / 4;

        // Process 4 elements at a time with SIMD
        for (0..vec_len) |i| {
            const idx = i * 4;
            const va: Vec4 = result.data[idx..][0..4].*;
            const vb: Vec4 = other.data[idx..][0..4].*;
            const vr = va + vb;
            result.data[idx..][0..4].* = vr;
        }

        // Handle remainder with scalar operations
        for (vec_len * 4..total_len) |i| {
            result.data[i] += other.data[i];
        }

        return result;
    }

    // SIMD-optimized dot product for inner loop
    pub fn dotSIMD(self: *const Matrix, other: *const Matrix) !Matrix {
        std.debug.assert(self.cols == other.rows);
        var result = try Matrix.init(self.allocator, self.rows, other.cols);
        @memset(result.data, 0.0);

        const Vec4 = @Vector(4, f32);
        const vec_len = self.cols / 4;

        for (0..self.rows) |i| {
            for (0..other.cols) |j| {
                var sum: f32 = 0.0;

                // SIMD inner loop
                var vec_sum = @as(Vec4, @splat(0.0));
                for (0..vec_len) |k_vec| {
                    const k = k_vec * 4;
                    const self_vec: Vec4 = self.data[i * self.cols + k ..][0..4].*;
                    var other_vec: Vec4 = undefined;
                    for (0..4) |offset| {
                        other_vec[offset] = other.at(k + offset, j);
                    }
                    vec_sum += self_vec * other_vec;
                }

                // Reduce vector sum
                sum += @reduce(.Add, vec_sum);

                // Handle remainder with scalar
                for (vec_len * 4..self.cols) |k| {
                    sum += self.at(i, k) * other.at(k, j);
                }

                result.set(i, j, sum);
            }
        }

        return result;
    }

    pub fn save(self: *const Matrix, writer: anytype) !void {
        try writer.writeInt(usize, self.rows, .little);
        try writer.writeInt(usize, self.cols, .little);
        for (self.data) |val| {
            try writer.writeInt(u32, @as(u32, @bitCast(val)), .little);
        }
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !Matrix {
        const rows = try reader.readInt(usize, .little);
        const cols = try reader.readInt(usize, .little);
        const matrix = try Matrix.init(allocator, rows, cols);
        for (matrix.data) |*val| {
            const bits = try reader.readInt(u32, .little);
            val.* = @as(f32, @bitCast(bits));
        }
        return matrix;
    }
};
test "SIMD add correctness" {
    const allocator = std.testing.allocator;

    // Test with size divisible by 4
    var a = try Matrix.init(allocator, 2, 4);
    defer a.deinit();
    var b = try Matrix.init(allocator, 2, 4);
    defer b.deinit();

    for (0..8) |i| {
        a.data[i] = @floatFromInt(i);
        b.data[i] = @floatFromInt(i * 2);
    }

    var result_scalar = try a.add(&b);
    defer result_scalar.deinit();
    var result_simd = try a.addSIMD(&b);
    defer result_simd.deinit();

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(result_scalar.data[i], result_simd.data[i], 1e-6);
    }
}

test "SIMD add with non-aligned size" {
    const allocator = std.testing.allocator;

    // Test with size NOT divisible by 4 (remainder case)
    var a = try Matrix.init(allocator, 1, 7);
    defer a.deinit();
    var b = try Matrix.init(allocator, 1, 7);
    defer b.deinit();

    for (0..7) |i| {
        a.data[i] = @floatFromInt(i);
        b.data[i] = 1.0;
    }

    var result_scalar = try a.add(&b);
    defer result_scalar.deinit();
    var result_simd = try a.addSIMD(&b);
    defer result_simd.deinit();

    for (0..7) |i| {
        try std.testing.expectApproxEqAbs(result_scalar.data[i], result_simd.data[i], 1e-6);
    }
}

test "SIMD dot correctness" {
    const allocator = std.testing.allocator;

    // Test with size divisible by 4
    var a = try Matrix.init(allocator, 2, 4);
    defer a.deinit();
    var b = try Matrix.init(allocator, 4, 3);
    defer b.deinit();

    for (0..8) |i| {
        a.data[i] = @floatFromInt(i + 1);
    }
    for (0..12) |i| {
        b.data[i] = @floatFromInt(i + 1);
    }

    var result_scalar = try a.dot(&b);
    defer result_scalar.deinit();
    var result_simd = try a.dotSIMD(&b);
    defer result_simd.deinit();

    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(result_scalar.data[i], result_simd.data[i], 1e-4);
    }
}

test "SIMD dot with non-aligned size" {
    const allocator = std.testing.allocator;

    // Test with size NOT divisible by 4
    var a = try Matrix.init(allocator, 2, 5);
    defer a.deinit();
    var b = try Matrix.init(allocator, 5, 2);
    defer b.deinit();

    for (0..10) |i| {
        a.data[i] = @floatFromInt(i + 1);
        b.data[i] = @floatFromInt(i + 1);
    }

    var result_scalar = try a.dot(&b);
    defer result_scalar.deinit();
    var result_simd = try a.dotSIMD(&b);
    defer result_simd.deinit();

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(result_scalar.data[i], result_simd.data[i], 1e-4);
    }
}
