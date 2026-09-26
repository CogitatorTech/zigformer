//! Parallel computation utilities.
//!
//! Provides threaded parallel execution for computationally intensive
//! operations like matrix multiplication.

const std = @import("std");
const linalg = @import("linear_algebra.zig");
const Matrix = linalg.Matrix;

/// Parallel matrix multiplication using worker threads.
///
/// Splits the output rows across up to `num_threads` worker threads.
pub fn dotParallel(a: *const Matrix, b: *const Matrix, num_threads: usize) !Matrix {
    return a.dotWithThreads(b, num_threads);
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

test "parallel dot caps workers to row count" {
    const allocator = std.testing.allocator;

    // More requested workers than output rows must still produce correct results.
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
