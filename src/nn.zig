const std = @import("std");

// Setup random number generator
const RndGen = std.rand.DefaultPrng;
var rnd: std.rand.DefaultPrng = undefined;

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

pub fn mat_get(mat: Mat, i: usize, j: usize) @TypeOf(mat.es[0]) {
    return mat.es[i * mat.stride + j];
}

pub fn mat_put(mat: Mat, i: usize, j: usize, v: @TypeOf(mat.es[0])) void {
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
    const offset: usize = index * mat.cols;
    var row = Mat{ .rows = 1, .cols = mat.cols, .stride = mat.cols, .es = mat.es[offset .. offset + mat.cols] };

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
    std.debug.print("{s} = [\n", .{name});
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
