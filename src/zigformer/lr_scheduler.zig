const std = @import("std");

pub const SchedulerType = enum {
    constant,
    linear_decay,
    cosine_annealing,
};

pub const LRScheduler = struct {
    scheduler_type: SchedulerType,
    initial_lr: f32,
    min_lr: f32,
    total_steps: usize,
    current_step: usize,

    pub fn init(scheduler_type: SchedulerType, initial_lr: f32, total_steps: usize) LRScheduler {
        return LRScheduler{
            .scheduler_type = scheduler_type,
            .initial_lr = initial_lr,
            .min_lr = initial_lr * 0.01, // 1% of initial LR
            .total_steps = total_steps,
            .current_step = 0,
        };
    }

    pub fn step(self: *LRScheduler) void {
        self.current_step += 1;
    }

    pub fn getLR(self: *const LRScheduler) f32 {
        return switch (self.scheduler_type) {
            .constant => self.initial_lr,
            .linear_decay => self.linearDecay(),
            .cosine_annealing => self.cosineAnnealing(),
        };
    }

    fn linearDecay(self: *const LRScheduler) f32 {
        if (self.total_steps == 0 or self.current_step >= self.total_steps) {
            return self.min_lr;
        }
        const progress: f32 = @as(f32, @floatFromInt(self.current_step)) / @as(f32, @floatFromInt(self.total_steps));
        return self.initial_lr * (1.0 - progress) + self.min_lr * progress;
    }

    fn cosineAnnealing(self: *const LRScheduler) f32 {
        if (self.total_steps == 0 or self.current_step >= self.total_steps) {
            return self.min_lr;
        }
        const progress: f32 = @as(f32, @floatFromInt(self.current_step)) / @as(f32, @floatFromInt(self.total_steps));
        const cosine_val = std.math.cos(progress * std.math.pi);
        return self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1.0 + cosine_val);
    }
};

test "LRScheduler constant" {
    var scheduler = LRScheduler.init(.constant, 0.001, 100);
    try std.testing.expectApproxEqAbs(0.001, scheduler.getLR(), 1e-6);
    scheduler.step();
    try std.testing.expectApproxEqAbs(0.001, scheduler.getLR(), 1e-6);
}

test "LRScheduler linear decay" {
    var scheduler = LRScheduler.init(.linear_decay, 0.001, 100);
    const initial_lr = scheduler.getLR();
    try std.testing.expectApproxEqAbs(0.001, initial_lr, 1e-6);

    // After 50 steps, should be halfway
    for (0..50) |_| {
        scheduler.step();
    }
    const mid_lr = scheduler.getLR();
    try std.testing.expect(mid_lr < initial_lr);
    try std.testing.expect(mid_lr > scheduler.min_lr);
}

test "LRScheduler cosine annealing" {
    var scheduler = LRScheduler.init(.cosine_annealing, 0.001, 100);
    const initial_lr = scheduler.getLR();
    try std.testing.expectApproxEqAbs(0.001, initial_lr, 1e-6);

    for (0..50) |_| {
        scheduler.step();
    }
    const mid_lr = scheduler.getLR();
    try std.testing.expect(mid_lr < initial_lr);
}
