const std = @import("std");
const linalg = @import("linear_algebra.zig");
const Matrix = linalg.Matrix;

pub const Layer = struct {
    self: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        forward: *const fn (self: *anyopaque, input: Matrix) anyerror!Matrix,
        backward: *const fn (self: *anyopaque, grads: Matrix, lr: f32) anyerror!Matrix,
        parameters: *const fn (self: *const anyopaque) usize,
        deinit: *const fn (self: *anyopaque) void,
    };

    pub fn forward(l: *Layer, input: Matrix) !Matrix {
        return l.vtable.forward(l.self, input);
    }

    pub fn backward(l: *Layer, grads: Matrix, lr: f32) !Matrix {
        return l.vtable.backward(l.self, grads, lr);
    }

    pub fn parameters(l: *const Layer) usize {
        return l.vtable.parameters(l.self);
    }

    pub fn deinit(l: *const Layer) void {
        return l.vtable.deinit(l.self);
    }
};

pub fn toLayer(comptime T: type) fn (ptr: *T) Layer {
    return struct {
        fn toLayerImpl(ptr: *T) Layer {
            return .{
                .self = ptr,
                .vtable = &.{
                    .forward = forward,
                    .backward = backward,
                    .parameters = parameters,
                    .deinit = deinit,
                },
            };
        }
        fn forward(self: *anyopaque, input: Matrix) anyerror!Matrix {
            const ptr: *T = @ptrCast(@alignCast(self));
            return ptr.forward(input);
        }
        fn backward(self: *anyopaque, grads: Matrix, lr: f32) anyerror!Matrix {
            const ptr: *T = @ptrCast(@alignCast(self));
            return ptr.backward(grads, lr);
        }
        fn parameters(self: *const anyopaque) usize {
            const ptr: *const T = @ptrCast(@alignCast(self));
            return ptr.parameters();
        }
        fn deinit(self: *anyopaque) void {
            const ptr: *T = @ptrCast(@alignCast(self));
            ptr.deinit();
        }
    }.toLayerImpl;
}
