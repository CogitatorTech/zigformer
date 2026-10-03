//! Linear algebra operations for neural network computations.
//!
//! Provides a Matrix type and operations:
//! - Matrix multiplication
//! - Element-wise operations
//! - Transpose
//! - SIMD-optimized variants
//!
//! Mathematical notation:
//! - Matrix multiplication: C = AB where C[i,j] = Σ_k A[i,k] * B[k,j]
//! - Transpose: A^T where A^T[i,j] = A[j,i]
//! - Element-wise add: C = A + B where C[i,j] = A[i,j] + B[i,j]

const std = @import("std");
const builtin = @import("builtin");

const can_thread = !builtin.single_threaded and builtin.target.os.tag != .freestanding;
const work_per_thread = 4_000_000;

/// Dense matrix representation.
///
/// Stores data in row-major format: A[i,j] is accessed as data[i * cols + j]
/// All operations allocate new matrices and return them; the caller owns the memory.
pub const Matrix = struct {
    rows: usize, // Number of rows
    cols: usize, // Number of columns
    data: []f32, // Contiguous array in row-major order
    allocator: std.mem.Allocator,

    /// Initialize a matrix with uninitialized data.
    ///
    /// Parameters:
    ///   allocator: Memory allocator
    ///   rows: Number of rows
    ///   cols: Number of columns
    ///
    /// Returns:
    ///   Initialized matrix with allocated but uninitialized data
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
        var seed: u64 = undefined;
        std.Options.debug_io.random(std.mem.asBytes(&seed));
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
        if (comptime !can_thread) return self.dotWithThreads(other, 1);
        const work = self.rows * self.cols * other.cols;
        if (work < 2 * work_per_thread) return self.dotWithThreads(other, 1);
        return self.dotWithThreads(other, work / work_per_thread);
    }

    pub fn dotWithThreads(self: *const Matrix, other: *const Matrix, num_threads: usize) !Matrix {
        std.debug.assert(self.cols == other.rows);
        var result = try Matrix.init(self.allocator, self.rows, other.cols);
        errdefer result.deinit();

        if (result.data.len == 0) return result;

        if (comptime !can_thread) {
            computeDotRows(self, other, &result, 0, self.rows);
            return result;
        } else if (num_threads <= 1 or self.rows <= 1) {
            computeDotRows(self, other, &result, 0, self.rows);
            return result;
        }

        const thread_count = @min(num_threads, self.rows, std.Thread.getCpuCount() catch 1);
        if (thread_count <= 1) {
            computeDotRows(self, other, &result, 0, self.rows);
            return result;
        }

        // ponytail: workers are created per large matrix multiply; use a persistent pool if profiling shows spawn overhead matters.
        const threads = self.allocator.alloc(std.Thread, thread_count) catch {
            computeDotRows(self, other, &result, 0, self.rows);
            return result;
        };
        defer self.allocator.free(threads);

        var spawned: usize = 0;
        for (0..thread_count) |worker_id| {
            const start_row = self.rows / thread_count * worker_id + @min(worker_id, self.rows % thread_count);
            const end_row = self.rows / thread_count * (worker_id + 1) + @min(worker_id + 1, self.rows % thread_count);
            threads[spawned] = std.Thread.spawn(.{}, computeDotRows, .{ self, other, &result, start_row, end_row }) catch {
                computeDotRows(self, other, &result, start_row, end_row);
                for (worker_id + 1..thread_count) |remaining_id| {
                    const remaining_start = self.rows / thread_count * remaining_id + @min(remaining_id, self.rows % thread_count);
                    const remaining_end = self.rows / thread_count * (remaining_id + 1) + @min(remaining_id + 1, self.rows % thread_count);
                    computeDotRows(self, other, &result, remaining_start, remaining_end);
                }
                for (threads[0..spawned]) |thread| thread.join();
                return result;
            };
            spawned += 1;
        }

        for (threads[0..spawned]) |thread| thread.join();
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

    // SIMD-optimized dot product.
    pub fn dotSIMD(self: *const Matrix, other: *const Matrix) !Matrix {
        return self.dotWithThreads(other, 1);
    }

    pub fn save(self: *const Matrix, writer: anytype) !void {
        try writer.writeInt(usize, self.rows, .little);
        try writer.writeInt(usize, self.cols, .little);
        for (self.data) |val| {
            try writer.writeInt(u32, @as(u32, @bitCast(val)), .little);
        }
    }

    pub fn load(allocator: std.mem.Allocator, reader: *std.Io.Reader) !Matrix {
        const rows = try reader.takeInt(usize, .little);
        const cols = try reader.takeInt(usize, .little);
        const matrix = try Matrix.init(allocator, rows, cols);
        for (matrix.data) |*val| {
            const bits = try reader.takeInt(u32, .little);
            val.* = @as(f32, @bitCast(bits));
        }
        return matrix;
    }
};

fn computeDotRows(a: *const Matrix, b: *const Matrix, result: *Matrix, start_row: usize, end_row: usize) void {
    const Vec4 = @Vector(4, f32);
    const vector_cols = b.cols - b.cols % 4;

    for (start_row..end_row) |i| {
        const a_row = a.data[i * a.cols ..][0..a.cols];
        const result_row = result.data[i * b.cols ..][0..b.cols];

        var j: usize = 0;
        while (j < vector_cols) : (j += 4) {
            var sums = @as(Vec4, @splat(0));
            for (a_row, 0..) |a_value, k| {
                const b_values: Vec4 = b.data[k * b.cols + j ..][0..4].*;
                sums += @as(Vec4, @splat(a_value)) * b_values;
            }
            result_row[j..][0..4].* = sums;
        }

        for (vector_cols..b.cols) |col| {
            var sum: f32 = 0;
            for (a_row, 0..) |a_value, k| {
                sum += a_value * b.data[k * b.cols + col];
            }
            result_row[col] = sum;
        }
    }
}

fn dotReference(a: *const Matrix, b: *const Matrix) !Matrix {
    std.debug.assert(a.cols == b.rows);
    var result = try Matrix.init(a.allocator, a.rows, b.cols);
    errdefer result.deinit();
    for (0..a.rows) |i| {
        for (0..b.cols) |j| {
            var sum: f32 = 0;
            for (0..a.cols) |k| sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            result.data[i * b.cols + j] = sum;
        }
    }
    return result;
}
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

    // Vectorizes output columns while reducing across the inner dimension.
    var a = try Matrix.init(allocator, 2, 4);
    defer a.deinit();
    var b = try Matrix.init(allocator, 4, 8);
    defer b.deinit();

    for (a.data, 0..) |*value, i| value.* = @floatFromInt(i + 1);
    for (b.data, 0..) |*value, i| value.* = @floatFromInt(i + 1);

    var result_scalar = try dotReference(&a, &b);
    defer result_scalar.deinit();
    var result_simd = try a.dotSIMD(&b);
    defer result_simd.deinit();

    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(result_scalar.data[i], result_simd.data[i], 1e-4);
    }
}

test "SIMD dot with non-aligned size" {
    const allocator = std.testing.allocator;

    // Test vectorized columns, a scalar column tail, and a non-aligned inner dimension.
    var a = try Matrix.init(allocator, 3, 5);
    defer a.deinit();
    var b = try Matrix.init(allocator, 5, 7);
    defer b.deinit();

    for (a.data, 0..) |*value, i| value.* = @floatFromInt(i + 1);
    for (b.data, 0..) |*value, i| value.* = @floatFromInt(i + 1);

    var result_scalar = try dotReference(&a, &b);
    defer result_scalar.deinit();
    var result_simd = try a.dotSIMD(&b);
    defer result_simd.deinit();

    for (0..21) |i| {
        try std.testing.expectApproxEqAbs(result_scalar.data[i], result_simd.data[i], 1e-4);
    }
}

test "automatic parallel dot matches scalar result" {
    const allocator = std.testing.allocator;
    var a = try Matrix.init(allocator, 256, 128);
    defer a.deinit();
    var b = try Matrix.init(allocator, 128, 256);
    defer b.deinit();

    for (a.data, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 7)) / 7;
    for (b.data, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 11)) / 11;

    var expected = try dotReference(&a, &b);
    defer expected.deinit();
    var actual = try a.dot(&b);
    defer actual.deinit();

    for (expected.data, actual.data) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 1e-3);
    }
}
