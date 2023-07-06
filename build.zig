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

    const twice_step = b.step("twice", "Run the twice app");
    if (b.args) |args| twice.run().addArgs(args);
    twice_step.dependOn(&twice.run().step);

    const gates_step = b.step("gates", "Run the gates app");
    gates_step.dependOn(&gates.run().step);

    const xor_step = b.step("xor", "Run the xor app");
    xor_step.dependOn(&xor.run().step);
}
