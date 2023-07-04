const std = @import("std");

const train = [_][2]f32{
    [_]f32{ 0, 0 },
    [_]f32{ 1, 2 },
    [_]f32{ 2, 4 },
    [_]f32{ 3, 6 },
    [_]f32{ 4, 8 },
};
const RndGen = std.rand.DefaultPrng;
var rnd: std.rand.DefaultPrng = undefined;

fn rand_float() f32 {
    return rnd.random().float(f32);
}

fn cost(w: f32, b: f32) f32 {
    // y = x * w
    var result: f32 = 0;
    for (train) |el| {
        var x = el[0];
        var y = x * w + b;
        var d = y - el[1];
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

    var w: f32 = rand_float() * 10.0;
    var b: f32 = rand_float() * 5.0;
    try stdout.print("random float is {d}.\n", .{w});
    try bw.flush();

    const eps: f32 = 1e-3;
    const rate: f32 = 1e-3;
    var iter: usize = 0;
    try stdout.print("cost: {d}\n", .{cost(w, b)});
    try bw.flush();
    while (iter < 500) : (iter += 1) {
        var c = cost(w, b);
        var dw = (cost(w + eps, b) - c) / eps;
        var db = (cost(w, b + eps) - c) / eps;
        w -= rate * dw;
        b -= rate * db;
        try stdout.print("cost: {d}, w = {d}, b = {d}\n", .{ cost(w, b), w, b });
        try bw.flush();
    }
    try stdout.print("--------------------\n", .{});
    try stdout.print("w: {d}, b: {d}\n", .{ w, b });
    try bw.flush();
}
