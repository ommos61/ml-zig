const std = @import("std");
const print = std.debug.print;

const nn = @import("nn.zig");
const Mat = nn.Mat;
const mat_put = nn.mat_put;
const mat_get = nn.mat_get;
const NN = nn.NN;
const nn_init = nn.nn_init;
const nn_alloc = nn.nn_alloc;
const nn_rand = nn.nn_rand;
const nn_cost = nn.nn_cost;
const nn_finite_diff = nn.nn_finite_diff;
const nn_learn = nn.nn_learn;
const nn_forward = nn.nn_forward;
const nn_input = nn.nn_input;
const nn_output = nn.nn_output;

// training data
var train_xor = [_]f32{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
var train_or = [_]f32{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

pub fn main() !void {
    nn_init();

    const train_mat = Mat{ .rows = 4, .cols = 3, .stride = 3, .es = &train_xor };
    const ti = Mat{ .rows = 4, .cols = 2, .stride = 3, .es = train_mat.es };
    const to = Mat{ .rows = 4, .cols = 1, .stride = 3, .es = train_mat.es[2..] };

    // setup the network architecture
    var arch = [_]u32{ 2, 2, 1 };
    var model: NN = try nn_alloc(&arch, arch.len);
    var gradient: NN = try nn_alloc(&arch, arch.len);
    nn_rand(model, 0, 1);

    const eps: f32 = 1e-1;
    const rate: f32 = 1e-1;

    std.debug.print("--------------------------\n", .{});
    //   nn_print(model, "xor_model");
    const iter_count: usize = 100 * 1000;
    std.debug.print("Running {d} iterations...\n", .{iter_count});
    std.debug.print("Cost = {d:.6}\n", .{nn_cost(model, ti, to)});
    var i: usize = 0;
    while (i < iter_count) : (i += 1) {
        nn_finite_diff(model, gradient, eps, ti, to);
        nn_learn(model, gradient, rate);
        //        nn_print(model, "xor_model");
    }
    //    nn_print(model, "xor_model");
    std.debug.print("Cost = {d:.6}\n", .{nn_cost(model, ti, to)});

    std.debug.print("--------------------------\n", .{});
    i = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            mat_put(nn_input(model), 0, 0, @intToFloat(f32, i));
            mat_put(nn_input(model), 0, 1, @intToFloat(f32, j));
            nn_forward(model);
            std.debug.print("{d} ^ {d} = {d:.6}\n", .{ i, j, mat_get(nn_output(model), 0, 0) });
        }
    }
}
