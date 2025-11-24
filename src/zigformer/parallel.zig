const std = @import("std");
const linalg = @import("linear_algebra.zig");
const Matrix = linalg.Matrix;

/// Parallel matrix multiplication using thread pool
pub fn dotParallel(a: *const Matrix, b: *const Matrix, num_threads: usize) !Matrix {
    std.debug.assert(a.cols == b.rows);
    var result = try Matrix.init(a.allocator, a.rows, b.cols);
    @memset(result.data, 0.0);

    if (num_threads <= 1 or a.rows < num_threads) {
        // Fall back to sequential computation
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f32 = 0.0;
                for (0..a.cols) |k| {
                    sum += a.at(i, k) * b.at(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = a.allocator, .n_jobs = num_threads });
    defer pool.deinit();

    const rows_per_thread = a.rows / num_threads;
    var wg: std.Thread.WaitGroup = .{};

    for (0..num_threads) |thread_id| {
        const start_row = thread_id * rows_per_thread;
        const end_row = if (thread_id == num_threads - 1) a.rows else (thread_id + 1) * rows_per_thread;

        pool.spawnWg(&wg, computeRows, .{ a, b, &result, start_row, end_row });
    }

    pool.waitAndWork(&wg);
    return result;
}

fn computeRows(a: *const Matrix, b: *const Matrix, result: *Matrix, start_row: usize, end_row: usize) void {
    for (start_row..end_row) |i| {
        for (0..b.cols) |j| {
            var sum: f32 = 0.0;
            for (0..a.cols) |k| {
                sum += a.at(i, k) * b.at(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

test "parallel dot correctness" {
    const allocator = std.testing.allocator;

    var a = try Matrix.init(allocator, 8, 6);
    defer a.deinit();
    var b = try Matrix.init(allocator, 6, 4);
    defer b.deinit();

    // Initialize with test data
    for (0..48) |i| {
        a.data[i] = @floatFromInt(i + 1);
    }
    for (0..24) |i| {
        b.data[i] = @floatFromInt(i + 1);
    }

    var result_sequential = try a.dot(&b);
    defer result_sequential.deinit();

    var result_parallel = try dotParallel(&a, &b, 4);
    defer result_parallel.deinit();

    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(result_sequential.data[i], result_parallel.data[i], 1e-4);
    }
}

test "parallel dot with small matrix" {
    const allocator = std.testing.allocator;

    // Small matrix should fall back to sequential
    var a = try Matrix.init(allocator, 2, 3);
    defer a.deinit();
    var b = try Matrix.init(allocator, 3, 2);
    defer b.deinit();

    for (0..6) |i| {
        a.data[i] = @floatFromInt(i + 1);
        b.data[i] = @floatFromInt(i + 1);
    }

    var result_sequential = try a.dot(&b);
    defer result_sequential.deinit();

    var result_parallel = try dotParallel(&a, &b, 4);
    defer result_parallel.deinit();

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(result_sequential.data[i], result_parallel.data[i], 1e-4);
    }
}
