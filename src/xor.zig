const std = @import("std");

// Create the stdout
const stdout_file = std.io.getStdOut().writer();
var bw = std.io.bufferedWriter(stdout_file);
const stdout = bw.writer();

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
// XOR gate
const xor_train = [_][3]f32{
    [_]f32{ 0, 0, 0 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 1, 1, 0 },
};
const train = xor_train;
const RndGen = std.rand.DefaultPrng;
var rnd: std.rand.DefaultPrng = undefined;

const xor_model = struct {
    or_w1: f32,
    or_w2: f32,
    or_b: f32,
    nand_w1: f32,
    nand_w2: f32,
    nand_b: f32,
    and_w1: f32,
    and_w2: f32,
    and_b: f32,
};

fn rand_float() f32 {
    return rnd.random().float(f32);
}

fn sigmoidf(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

fn forward(m: xor_model, x1: f32, x2: f32) f32 {
    var or_y = sigmoidf(x1 * m.or_w1 + x2 * m.or_w2 + m.or_b);
    var nand_y = sigmoidf(x1 * m.nand_w1 + x2 * m.nand_w2 + m.nand_b);
    return sigmoidf(or_y * m.and_w1 + nand_y * m.and_w2 + m.and_b);
}

fn finite_diff(model: xor_model, eps: f32) xor_model {
    var m: xor_model = model;
    var gradient: xor_model = init_model();
    const c = cost(m);

    var saved: f32 = undefined;
    saved = m.or_w1;
    m.or_w1 += eps;
    gradient.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;
    saved = m.or_w2;
    m.or_w2 += eps;
    gradient.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;
    saved = m.or_b;
    m.or_b += eps;
    gradient.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    gradient.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;
    saved = m.nand_w2;
    m.nand_w2 += eps;
    gradient.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;
    saved = m.nand_b;
    m.nand_b += eps;
    gradient.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    gradient.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;
    saved = m.and_w2;
    m.and_w2 += eps;
    gradient.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;
    saved = m.and_b;
    m.and_b += eps;
    gradient.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    return gradient;
}

fn learn(model: xor_model, gradient: xor_model, rate: f32) xor_model {
    var m = model;

    m.or_w1 -= rate * gradient.or_w1;
    m.or_w2 -= rate * gradient.or_w2;
    m.or_b -= rate * gradient.or_b;
    m.nand_w1 -= rate * gradient.nand_w1;
    m.nand_w2 -= rate * gradient.nand_w2;
    m.nand_b -= rate * gradient.nand_b;
    m.and_w1 -= rate * gradient.and_w1;
    m.and_w2 -= rate * gradient.and_w2;
    m.and_b -= rate * gradient.and_b;

    return m;
}

fn cost(m: xor_model) f32 {
    var result: f32 = 0;
    for (train) |el| {
        var x1 = el[0];
        var x2 = el[1];
        var y = forward(m, x1, x2);
        var d = (y - el[2]);
        result += d * d;
    }
    result /= @as(f32, train.len);
    return result;
}

fn init_model() xor_model {
    var m = xor_model{
        .or_w1 = rand_float(),
        .or_w2 = rand_float(),
        .or_b = rand_float(),
        .nand_w1 = rand_float(),
        .nand_w2 = rand_float(),
        .nand_b = rand_float(),
        .and_w1 = rand_float(),
        .and_w2 = rand_float(),
        .and_b = rand_float(),
    };
    return m;
}

fn dump_xor_model(m: xor_model) !void {
    try stdout.print("or:   w1 = {d}, w2 = {d}, b = {d}\n", .{ m.or_w1, m.or_w2, m.or_b });
    try stdout.print("nand: w1 = {d}, w2 = {d}, b = {d}\n", .{ m.nand_w1, m.nand_w2, m.nand_b });
    try stdout.print("and:  w1 = {d}, w2 = {d}, b = {d}\n", .{ m.and_w1, m.and_w2, m.and_b });
    try bw.flush();
}

pub fn main() !void {
    // Initialize the random number generator
    const seed = @truncate(u64, @bitCast(u128, std.time.nanoTimestamp()));
    //const seed = 69;
    rnd = RndGen.init(seed);

    var model: xor_model = init_model();
    try dump_xor_model(model);

    const eps: f32 = 1e-1;
    const rate: f32 = 1e-1;

    try stdout.print("cost: {d}\n", .{cost(model)});
    try bw.flush();
    try dump_xor_model(finite_diff(model, eps));
    const iter_count: usize = 100 * 1000;
    var iter: usize = 0;
    while (iter < iter_count) : (iter += 1) {
        const gradient = finite_diff(model, eps);
        model = learn(model, gradient, rate);
        //var c = cost(model);
        //try stdout.print("cost: {d}\n", .{c});
        //try bw.flush();
    }
    try stdout.print("--------------------\n", .{});
    try stdout.print("iter_count: {d}\n", .{iter_count});
    try stdout.print("cost: {d}\n", .{cost(model)});
    try bw.flush();

    const input = [_]u8{ 0, 1 };
    for (input) |_, i| {
        for (input) |_, j| {
            try stdout.print("{d} ^ {d}: {d}\n", .{ i, j, forward(model, @intToFloat(f32, i), @intToFloat(f32, j)) });
        }
    }
    try stdout.print("-OR-------------------\n", .{});
    for ([_]u8{ 0, 1 }) |_, i| {
        for ([_]u8{ 0, 1 }) |_, j| {
            try stdout.print("{d} | {d}: {d}\n", .{ i, j, sigmoidf(@intToFloat(f32, i) * model.or_w1 + @intToFloat(f32, j) * model.or_w2 + model.or_b) });
        }
    }
    try stdout.print("-NAND-----------------\n", .{});
    for ([_]u8{ 0, 1 }) |_, i| {
        for ([_]u8{ 0, 1 }) |_, j| {
            try stdout.print("~({d} & {d}): {d}\n", .{ i, j, sigmoidf(@intToFloat(f32, i) * model.nand_w1 + @intToFloat(f32, j) * model.nand_w2 + model.nand_b) });
        }
    }
    try stdout.print("-AND------------------\n", .{});
    for ([_]u8{ 0, 1 }) |_, i| {
        for ([_]u8{ 0, 1 }) |_, j| {
            try stdout.print("{d} & {d}: {d}\n", .{ i, j, sigmoidf(@intToFloat(f32, i) * model.and_w1 + @intToFloat(f32, j) * model.and_w2 + model.and_b) });
        }
    }
    try bw.flush();
}
