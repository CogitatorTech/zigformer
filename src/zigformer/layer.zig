const std = @import("std");
const linalg = @import("linear_algebra.zig");
const Matrix = linalg.Matrix;

pub const Layer = struct {
    self: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        forward: *const fn (self: *anyopaque, input: Matrix, use_cache: bool) anyerror!Matrix,
        backward: *const fn (self: *anyopaque, grads: Matrix, lr: f32) anyerror!Matrix,
        parameters: *const fn (self: *const anyopaque) usize,
        deinit: *const fn (self: *anyopaque) void,
        resetCache: *const fn (self: *anyopaque) void,
    };

    pub fn forward(l: *Layer, input: Matrix, use_cache: bool) !Matrix {
        return l.vtable.forward(l.self, input, use_cache);
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

    pub fn resetCache(l: *Layer) void {
        l.vtable.resetCache(l.self);
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
                    .resetCache = resetCache,
                },
            };
        }
        fn forward(self: *anyopaque, input: Matrix, use_cache: bool) anyerror!Matrix {
            const ptr: *T = @ptrCast(@alignCast(self));
            // Check if T has forward with use_cache, otherwise call without
            // Zig doesn't support reflection on function args easily at runtime, but we can use comptime check.
            if (@hasDecl(T, "forward")) {
                const ForwardType = @TypeOf(T.forward);
                const type_info = @typeInfo(ForwardType);
                // Check if it accepts 3 arguments (self, input, use_cache)
                // Note: self is implicit in struct methods if called as method, but here we check the function type.
                // T.forward is a function.
                if (switch (type_info) {
                    .@"fn" => |f| f.params.len == 3,
                    else => false,
                }) {
                    return ptr.forward(input, use_cache);
                } else {
                    return ptr.forward(input);
                }
            }
            // Fallback (should not happen if T is a valid Layer)
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
        fn resetCache(self: *anyopaque) void {
            const ptr: *T = @ptrCast(@alignCast(self));
            if (@hasDecl(T, "resetCache")) {
                ptr.resetCache();
            }
        }
    }.toLayerImpl;
}
