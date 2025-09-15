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
        var prng = std.Random.DefaultPrng.init(0);
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
};
