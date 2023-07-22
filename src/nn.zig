const std = @import("std");
const print = std.debug.print;

// Setup random number generator
const RndGen = std.rand.DefaultPrng;
var rnd: std.rand.DefaultPrng = undefined;

// General purpose allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

pub const Mat = struct {
    rows: usize,
    cols: usize,
    stride: usize,
    es: []f32,
};

fn rand_float() f32 {
    return rnd.random().float(f32);
}

fn sigmoidf(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

pub fn nn_init() void {
    const seed = @truncate(u64, @bitCast(u128, std.time.nanoTimestamp()));
    //const seed = 69;
    rnd = RndGen.init(seed);
}

pub fn mat_alloc(rows: usize, cols: usize) !Mat {
    var mat = Mat{ .rows = rows, .cols = cols, .stride = cols, .es = undefined };
    mat.es = try std.heap.page_allocator.alloc(@TypeOf(mat.es[0]), rows * cols);

    return mat;
}

pub fn mat_get(mat: Mat, i: usize, j: usize) f32 {
    return mat.es[i * mat.stride + j];
}

pub fn mat_put(mat: Mat, i: usize, j: usize, v: f32) void {
    mat.es[i * mat.stride + j] = v;
}

pub fn mat_fill(dst: Mat, v: f32) void {
    var i: usize = 0;
    while (i < dst.rows) : (i += 1) {
        var j: usize = 0;
        while (j < dst.cols) : (j += 1) {
            mat_put(dst, i, j, v);
        }
    }
}

pub fn mat_rand(dst: Mat, low: f32, high: f32) void {
    std.debug.assert(dst.rows != 0);
    std.debug.assert(dst.cols != 0);
    var i: usize = 0;
    while (i < dst.rows) : (i += 1) {
        var j: usize = 0;
        while (j < dst.cols) : (j += 1) {
            mat_put(dst, i, j, rand_float() * (high - low) + low);
        }
    }
}

pub fn mat_row(mat: Mat, index: usize) Mat {
    const offset: usize = index * mat.stride;
    var row = Mat{ .rows = 1, .cols = mat.cols, .stride = mat.stride, .es = mat.es[offset .. offset + mat.cols] };

    return row;
}

pub fn mat_copy(dst: Mat, src: Mat) void {
    std.debug.assert(src.rows == dst.rows);
    std.debug.assert(src.cols == dst.cols);

    var i: usize = 0;
    while (i < dst.rows) : (i += 1) {
        var j: usize = 0;
        while (j < dst.cols) : (j += 1) {
            mat_put(dst, i, j, mat_get(src, i, j));
        }
    }
}

pub fn mat_dot(dst: Mat, a: Mat, b: Mat) void {
    std.debug.assert(a.cols == b.rows);
    std.debug.assert(dst.rows == a.rows);
    std.debug.assert(dst.cols == b.cols);
    var i: usize = 0;
    while (i < dst.rows) : (i += 1) {
        var j: usize = 0;
        while (j < dst.cols) : (j += 1) {
            mat_put(dst, i, j, 0);
            var k: usize = 0;
            while (k < a.cols) : (k += 1) {
                mat_put(dst, i, j, mat_get(dst, i, j) + mat_get(a, i, k) * mat_get(b, k, j));
            }
        }
    }
}

pub fn mat_sum(dst: Mat, a: Mat) void {
    std.debug.assert(dst.rows == a.rows);
    std.debug.assert(dst.cols == a.cols);
    var i: usize = 0;
    while (i < dst.rows) : (i += 1) {
        var j: usize = 0;
        while (j < dst.cols) : (j += 1) {
            mat_put(dst, i, j, mat_get(dst, i, j) + mat_get(a, i, j));
        }
    }
}

pub fn mat_sigmoid(mat: Mat) void {
    var i: usize = 0;
    while (i < mat.rows) : (i += 1) {
        var j: usize = 0;
        while (j < mat.cols) : (j += 1) {
            mat_put(mat, i, j, sigmoidf(mat_get(mat, i, j)));
        }
    }
}

pub fn mat_print(mat: Mat, name: [*:0]const u8) void {
    std.debug.assert(mat.rows != 0);
    std.debug.assert(mat.cols != 0);
    std.debug.print("{s} [{d}x{d}] = [\n", .{ name, mat.rows, mat.cols });
    var i: usize = 0;
    while (i < mat.rows) : (i += 1) {
        var j: usize = 0;
        std.debug.print("    ", .{});
        while (j < mat.cols) : (j += 1) {
            std.debug.print("{d:.6} ", .{mat_get(mat, i, j)});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("]\n", .{});
}

pub const NN = struct {
    count: usize,
    ws: []Mat,
    bs: []Mat,
    as: []Mat, // The amount of activations is count+1
};

pub fn nn_alloc(arch: [*]u32, arch_count: usize) !NN {
    std.debug.assert(arch_count > 0);
    var model = NN{ .count = arch_count - 1, .ws = undefined, .bs = undefined, .as = undefined };
    model.as = try allocator.alloc(Mat, arch_count);
    model.ws = try allocator.alloc(Mat, arch_count - 1);
    model.bs = try allocator.alloc(Mat, arch_count - 1);
    model.as[0] = try mat_alloc(1, arch[0]);
    var i: usize = 1;
    while (i < arch_count) : (i += 1) {
        model.ws[i - 1] = try mat_alloc(model.as[i - 1].cols, arch[i]);
        model.bs[i - 1] = try mat_alloc(1, arch[i]);
        model.as[i] = try mat_alloc(1, arch[i]);
    }

    return model;
}

pub fn nn_input(model: NN) Mat {
    return model.as[0];
}

pub fn nn_output(model: NN) Mat {
    return model.as[model.count];
}

pub fn nn_rand(model: NN, low: f32, high: f32) void {
    var i: usize = 0;
    while (i < model.count) : (i += 1) {
        mat_rand(model.ws[i], low, high);
        mat_rand(model.bs[i], low, high);
    }
}
pub fn nn_forward(model: NN) void {
    var i: usize = 0;
    while (i < model.count) : (i += 1) {
        mat_dot(model.as[i + 1], model.as[i], model.ws[i]);
        mat_sum(model.as[i + 1], model.bs[i]);
        mat_sigmoid(model.as[i + 1]);
    }
    //nn_print(model, "forward");
}

pub fn nn_cost(model: NN, ti: Mat, to: Mat) f32 {
    std.debug.assert(ti.rows == to.rows);
    std.debug.assert(to.cols == nn_output(model).cols);

    var cost: f32 = 0;
    var i: usize = 0;
    while (i < ti.rows) : (i += 1) {
        var x = nn_input(model);
        var y = nn_output(model);

        mat_copy(x, mat_row(ti, i));
        var expected = mat_row(to, i);
        nn_forward(model);
        var j: usize = 0;
        while (j < to.cols) : (j += 1) {
            var d: f32 = mat_get(y, 0, j) - mat_get(expected, 0, j);
            cost += d * d;
        }
    }
    return cost / @intToFloat(f32, ti.rows);
}

pub fn nn_finite_diff(model: NN, gradient: NN, eps: f32, ti: Mat, to: Mat) void {
    var saved: f32 = undefined;
    var cost: f32 = nn_cost(model, ti, to);

    var k: usize = 0;
    while (k < model.count) : (k += 1) {
        var i: usize = 0;
        while (i < model.ws[k].rows) : (i += 1) {
            var j: usize = 0;
            while (j < model.ws[k].cols) : (j += 1) {
                saved = mat_get(model.ws[k], i, j);
                mat_put(model.ws[k], i, j, saved + eps);
                mat_put(gradient.ws[k], i, j, (nn_cost(model, ti, to) - cost) / eps);
                mat_put(model.ws[k], i, j, saved);
            }
        }
        i = 0;
        while (i < model.bs[k].rows) : (i += 1) {
            var j: usize = 0;
            while (j < model.bs[k].cols) : (j += 1) {
                saved = mat_get(model.bs[k], i, j);
                mat_put(model.bs[k], i, j, saved + eps);
                mat_put(gradient.bs[k], i, j, (nn_cost(model, ti, to) - cost) / eps);
                mat_put(model.bs[k], i, j, saved);
            }
        }
    }
}

pub fn nn_learn(model: NN, gradient: NN, rate: f32) void {
    var k: usize = 0;
    while (k < model.count) : (k += 1) {
        var i: usize = 0;
        while (i < model.ws[k].rows) : (i += 1) {
            var j: usize = 0;
            while (j < model.ws[k].cols) : (j += 1) {
                mat_put(model.ws[k], i, j, mat_get(model.ws[k], i, j) - rate * mat_get(gradient.ws[k], i, j));
            }
        }
        i = 0;
        while (i < model.bs[k].rows) : (i += 1) {
            var j: usize = 0;
            while (j < model.bs[k].cols) : (j += 1) {
                mat_put(model.bs[k], i, j, mat_get(model.bs[k], i, j) - rate * mat_get(gradient.bs[k], i, j));
            }
        }
    }
}

pub fn nn_print(model: NN, name: [*:0]const u8) void {
    print("{s} = [\n", .{name});
    print("  as[{d:3}]: {d} x {d}, [", .{ 0, model.as[0].rows, model.as[0].cols });
    var j: usize = 0;
    while (j < model.as[0].rows * model.as[0].cols) : (j += 1) {
        print("{d:.6} ", .{model.as[0].es[j]});
    }
    print("],\n", .{});
    var i: usize = 0;
    while (i < model.count) : (i += 1) {
        print("  ws[{d:3}]: {d} x {d}, [", .{ i, model.ws[i].rows, model.ws[i].cols });
        j = 0;
        while (j < model.ws[i].rows * model.ws[i].cols) : (j += 1) {
            print("{d:.6} ", .{model.ws[i].es[j]});
        }
        print("],\n", .{});
        print("  bs[{d:3}]: {d} x {d}, [", .{ i, model.bs[i].rows, model.bs[i].cols });
        j = 0;
        while (j < model.bs[i].rows * model.bs[i].cols) : (j += 1) {
            print("{d:.6} ", .{model.bs[i].es[j]});
        }
        print("],\n", .{});
        print("  as[{d:3}]: {d} x {d}, [", .{ i + 1, model.as[i + 1].rows, model.as[i + 1].cols });
        j = 0;
        while (j < model.as[i + 1].rows * model.as[i + 1].cols) : (j += 1) {
            print("{d:.6} ", .{model.as[i + 1].es[j]});
        }
        print("],\n", .{});
    }
    print("]\n", .{});
}
