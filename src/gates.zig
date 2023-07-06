const std = @import("std");

// OR gate
const or_train = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 1, 1, 1 },
};
// AND gate
const and_train = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 0, 1, 0 },
    [_]f32{ 1, 0, 0 },
    [_]f32{ 1, 1, 1 },
};
// NAND gate
const nand_train = [_][3]f32{
    [_]f32{ 0, 0, 1 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 1, 1, 0 },
};
const train = nand_train;
const RndGen = std.rand.DefaultPrng;
var rnd: std.rand.DefaultPrng = undefined;

fn rand_float() f32 {
    return rnd.random().float(f32);
}

fn sigmoidf(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

fn cost(w1: f32, w2: f32, b: f32) f32 {
    // y = x * w
    var result: f32 = 0;
    for (train) |el| {
        var x1 = el[0];
        var x2 = el[1];
        var y = sigmoidf(x1 * w1 + x2 * w2 + b);
        var d = (y - el[2]);
        result += d * d;
    }
    result /= @as(f32, train.len);
    return result;
}

pub fn main() !void {
    // Create the stdout
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    // Initialize the random number generator
    const seed = @truncate(u64, @bitCast(u128, std.time.nanoTimestamp()));
    //const seed = 69;
    rnd = RndGen.init(seed);

    var w1: f32 = rand_float();
    var w2: f32 = rand_float();
    var b: f32 = rand_float();
    try stdout.print("model is {d}, {d}.\n", .{ w1, w2 });
    try bw.flush();

    const eps: f32 = 1e-3;
    const rate: f32 = 1e-1;

    try stdout.print("cost: {d}, w1 = {d}, w2 = {d}, b = {d}\n", .{ cost(w1, w2, b), w1, w2, b });
    try bw.flush();
    const iter_count: usize = 100 * 1000;
    var iter: usize = 0;
    while (iter < iter_count) : (iter += 1) {
        var c = cost(w1, w2, b);
        var dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        var dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        var db = (cost(w1, w2, b + eps) - c) / eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
        //try stdout.print("cost: {d}, w1 = {d}, w2 = {d}\n", .{ cost(w1, w2), w1, w2 });
        //try bw.flush();
    }
    try stdout.print("--------------------\n", .{});
    try stdout.print("iter_count: {d}\n", .{iter_count});
    try stdout.print("cost: {d}, w1 = {d}, w2 = {d}, b = {d}\n", .{ cost(w1, w2, b), w1, w2, b });
    try bw.flush();

    const input = [_]u8{ 0, 1 };
    for (input) |_, i| {
        for (input) |_, j| {
            try stdout.print("{d} | {d}: {d}\n", .{ i, j, sigmoidf(@intToFloat(f32, i) * w1 + @intToFloat(f32, j) * w2 + b) });
        }
    }
    try bw.flush();
}
