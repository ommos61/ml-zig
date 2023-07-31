const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const twice = b.addExecutable("twice", "src/twice.zig");
    twice.setTarget(target);
    twice.setBuildMode(mode);
    twice.install();

    const gates = b.addExecutable("gates", "src/gates.zig");
    gates.setTarget(target);
    gates.setBuildMode(mode);
    gates.install();

    const xor = b.addExecutable("xor", "src/xor.zig");
    xor.setTarget(target);
    xor.setBuildMode(mode);
    xor.install();

    const nn_xor = b.addExecutable("nn_xor", "src/nn_xor.zig");
    nn_xor.setTarget(target);
    nn_xor.setBuildMode(mode);
    nn_xor.install();

    const nn_adder = b.addExecutable("nn_adder", "src/nn_adder.zig");
    nn_adder.setTarget(target);
    nn_adder.setBuildMode(mode);
    nn_adder.install();

    const twice_step = b.step("twice", "Run the twice app");
    if (b.args) |args| twice.run().addArgs(args);
    twice_step.dependOn(&twice.run().step);

    const gates_step = b.step("gates", "Run the gates app");
    gates_step.dependOn(&gates.run().step);

    const xor_step = b.step("xor", "Run the xor app");
    xor_step.dependOn(&xor.run().step);

    const nn_xor_step = b.step("nn_xor", "Run the nn xor app");
    nn_xor_step.dependOn(&nn_xor.run().step);

    const nn_adder_step = b.step("nn_adder", "Run the nn adder app");
    nn_adder_step.dependOn(&nn_adder.run().step);
}
