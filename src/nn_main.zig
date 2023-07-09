const std = @import("std");

const nn = @import("nn.zig");
const Mat = nn.Mat;
const mat_alloc = nn.mat_alloc;
const mat_dot = nn.mat_dot;
const mat_sum = nn.mat_sum;
const mat_get = nn.mat_get;
const mat_put = nn.mat_put;
const mat_sigmoid = nn.mat_sigmoid;
const mat_row = nn.mat_row;
const mat_copy = nn.mat_copy;
const mat_rand = nn.mat_rand;
const mat_print = nn.mat_print;

var xor_train = [_]f32{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
const train_xor_mat = Mat{ .rows = 4, .cols = 3, .stride = 3, .es = &xor_train };

var train_mat_in = Mat{ .rows = 4, .cols = 2, .stride = 3, .es = train_xor_mat.es };
var train_mat_out = Mat{ .rows = 4, .cols = 1, .stride = 3, .es = xor_train[2..] };

const Xor = struct {
    a0: Mat,
    w1: Mat,
    b1: Mat,
    a1: Mat,
    w2: Mat,
    b2: Mat,
    a2: Mat,
};

fn xor_alloc() !Xor {
    var model: Xor = undefined;
    model.a0 = try mat_alloc(1, 2);
    // layer 1
    model.w1 = try mat_alloc(2, 2);
    model.b1 = try mat_alloc(1, 2);
    model.a1 = try mat_alloc(1, 2);
    // layer 2
    model.w2 = try mat_alloc(2, 1);
    model.b2 = try mat_alloc(1, 1);
    model.a2 = try mat_alloc(1, 1);

    return model;
}

fn forward(model: Xor) void {
    // a1 = sigmoid(x * w1 + b)
    mat_dot(model.a1, model.a0, model.w1);
    mat_sum(model.a1, model.b1);
    mat_sigmoid(model.a1);

    // a2 = sigmoid(a1 * w2 + b2)
    mat_dot(model.a2, model.a1, model.w2);
    mat_sum(model.a2, model.b2);
    mat_sigmoid(model.a2);
}

fn cost(model: Xor, ti: Mat, to: Mat) f32 {
    std.debug.assert(ti.rows == to.rows);
    std.debug.assert(to.cols == model.a2.cols);

    var c: f32 = 0;
    var i: usize = 0;
    while (i < ti.rows) : (i += 1) {
        var x = mat_row(ti, i);
        var y = mat_row(to, i);

        mat_copy(model.a0, x);
        //mat_print(x, "x");
        //mat_print(model.a0, "a0");
        forward(model);

        var j: usize = 0;
        while (j < to.cols) : (j += 1) {
            var diff = mat_get(model.a2, 0, j) - mat_get(y, 0, j);
            c += diff * diff;
        }
    }
    return c / @intToFloat(f32, ti.rows);
}

fn finite_diff(model: Xor, gradient: Xor, eps: f32, ti: Mat, to: Mat) void {
    var c = cost(model, ti, to);
    var saved: f32 = undefined;

    var i: usize = 0;
    while (i < model.w1.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.w1.cols) : (j += 1) {
            saved = mat_get(model.w1, i, j);
            mat_put(model.w1, i, j, saved + eps);
            mat_put(gradient.w1, i, j, (cost(model, ti, to) - c) / eps);
            mat_put(model.w1, i, j, saved);
        }
    }
    i = 0;
    while (i < model.b1.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.b1.cols) : (j += 1) {
            saved = mat_get(model.b1, i, j);
            mat_put(model.b1, i, j, saved + eps);
            mat_put(gradient.b1, i, j, (cost(model, ti, to) - c) / eps);
            mat_put(model.b1, i, j, saved);
        }
    }
    i = 0;
    while (i < model.w2.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.w2.cols) : (j += 1) {
            saved = mat_get(model.w2, i, j);
            mat_put(model.w2, i, j, saved + eps);
            mat_put(gradient.w2, i, j, (cost(model, ti, to) - c) / eps);
            mat_put(model.w2, i, j, saved);
        }
    }
    i = 0;
    while (i < model.b2.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.b2.cols) : (j += 1) {
            saved = mat_get(model.b2, i, j);
            mat_put(model.b2, i, j, saved + eps);
            mat_put(gradient.b2, i, j, (cost(model, ti, to) - c) / eps);
            mat_put(model.b2, i, j, saved);
        }
    }
}

fn xor_learn(model: Xor, gradient: Xor, rate: f32) void {
    var i: usize = 0;
    while (i < model.w1.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.w1.cols) : (j += 1) {
            var value = mat_get(model.w1, i, j) - rate * mat_get(gradient.w1, i, j);
            mat_put(model.w1, i, j, value);
        }
    }
    i = 0;
    while (i < model.b1.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.b1.cols) : (j += 1) {
            var value = mat_get(model.b1, i, j) - rate * mat_get(gradient.b1, i, j);
            mat_put(model.b1, i, j, value);
        }
    }
    i = 0;
    while (i < model.w2.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.w2.cols) : (j += 1) {
            var value = mat_get(model.w2, i, j) - rate * mat_get(gradient.w2, i, j);
            mat_put(model.w2, i, j, value);
        }
    }
    i = 0;
    while (i < model.b2.rows) : (i += 1) {
        var j: usize = 0;
        while (j < model.b2.cols) : (j += 1) {
            var value = mat_get(model.b2, i, j) - rate * mat_get(gradient.b2, i, j);
            mat_put(model.b2, i, j, value);
        }
    }
}

pub fn main() !void {
    var model: Xor = try xor_alloc();
    var gradient: Xor = try xor_alloc();

    mat_rand(model.w1, 0, 1);
    mat_rand(model.b1, 0, 1);
    mat_rand(model.w2, 0, 1);
    mat_rand(model.b2, 0, 1);

    const eps: f32 = 1e-1;
    const rate: f32 = 1e-1;

    std.debug.print("----- training data ------\n", .{});
    mat_print(train_mat_in, "ti");
    mat_print(train_mat_out, "t0");

    std.debug.print("--------------------------\n", .{});
    std.debug.print("cost = {d:.6}\n", .{cost(model, train_mat_in, train_mat_out)});
    var n: usize = 0;
    const iter_count: usize = 10 * 1000;
    while (n < iter_count) : (n += 1) {
        finite_diff(model, gradient, eps, train_mat_in, train_mat_out);
        xor_learn(model, gradient, rate);

        std.debug.print("{d:6}: cost = {d:.6}\n", .{ n, cost(model, train_mat_in, train_mat_out) });
    }

    std.debug.print("--------------------------\n", .{});
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            mat_put(model.a0, 0, 0, @intToFloat(f32, i));
            mat_put(model.a0, 0, 1, @intToFloat(f32, j));
            forward(model);
            var y: f32 = mat_get(model.a2, 0, 0);

            std.debug.print("{d} ^ {d} = {d:.6}\n", .{ i, j, y });
        }
    }
}
