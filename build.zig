// file: build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create a module for the core library.
    const zigformer_mod = b.addModule("zigformer", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create an options artifact for compile-time configuration.
    const config_options = b.addOptions();
    config_options.addOption(usize, "embedding_dim", 128);
    config_options.addOption(usize, "hidden_dim", 256);
    config_options.addOption(usize, "max_seq_len", 80);

    // Add the options to the library module under the name "config".
    zigformer_mod.addOptions("config", config_options);

    // Create the main command-line executable's module.
    const cli_module = b.addModule("cli", .{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_module.addImport("zigformer", zigformer_mod);

    // Wire Chilli dependency into the CLI module
    const chilli_dep = b.dependency("chilli", .{});
    const chilli_module = chilli_dep.module("chilli");
    cli_module.addImport("chilli", chilli_module);

    // Create the executable artifact from the module.
    const exe = b.addExecutable(.{
        .name = "zigformer-cli",
        .root_module = cli_module,
    });

    b.installArtifact(exe);

    // Create a "run" step to execute the application.
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    // Create the test executable.
    const test_exe = b.addExecutable(.{
        .name = "zigformer-tests",
        .root_module = zigformer_mod, // The tests are part of the library module.
    });
    test_exe.kind = .@"test";

    // Create a "test" step to run the unit tests.
    const test_run_cmd = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&test_run_cmd.step);
}
