pub struct FeatureLayout {
    pub offset: usize,
    pub dims: usize,
}

macro_rules! point_layout {
    ($($feature:ident: $dims:expr,)*) => {
        point_layout!(@munch, 0 $(, ($feature, $dims))*);
    };

    (@munch, $offset:expr, ($feature:ident, $dims:expr) $($tail:tt)*) => {
        pub const $feature: FeatureLayout = FeatureLayout {
            offset: $offset,
            dims: $dims,
        };

        point_layout!(@munch, $offset + $dims $($tail)*);
    };

    (@munch, $offset:expr) => {
        pub const POINT_DIMS: usize = $offset;
    };
}

point_layout! {
    T_SIN_LAYOUT: 13,
    T_COS_LAYOUT: 13,
    Y_SIN_LAYOUT: 7,
    Y_COS_LAYOUT: 7,
    X_SIN_LAYOUT: 8,
    X_COS_LAYOUT: 8,
}

pub const EMBEDDINGS: usize = 820;

pub fn encode_point(input: &mut [f32; POINT_DIMS], t: f32, y: f32, x: f32) {
    let range = |layout: FeatureLayout| layout.offset..layout.offset + layout.dims;
    encode_sin(&mut input[range(T_SIN_LAYOUT)], t);
    encode_cos(&mut input[range(T_COS_LAYOUT)], t);
    encode_sin(&mut input[range(Y_SIN_LAYOUT)], y);
    encode_cos(&mut input[range(Y_COS_LAYOUT)], y);
    encode_sin(&mut input[range(X_SIN_LAYOUT)], x);
    encode_cos(&mut input[range(X_COS_LAYOUT)], x);
}

pub fn encode_embedding(i: f32) -> f32 {
    (i * EMBEDDINGS as f32).min((EMBEDDINGS - 1) as f32)
}

pub fn encode_frame(points: &mut [[f32; POINT_DIMS]], width: usize, height: usize, time: f32) {
    assert_eq!(points.len(), width * height);
    for y in 0..height {
        for x in 0..width {
            let yr = y as f32 / (height - 1) as f32;
            let xr = x as f32 / (width - 1) as f32;
            encode_point(&mut points[y * width + x], time, yr, xr);
        }
    }
}

pub fn encode_sin(input: &mut [f32], n: f32) {
    for (i, x) in input.iter_mut().enumerate() {
        *x = sin_pi_approx((n - 0.5) * (1 << i) as f32);
    }
}

pub fn encode_cos(input: &mut [f32], n: f32) {
    for (i, x) in input.iter_mut().enumerate() {
        *x = cos_pi_approx((n - 0.5) * (1 << i) as f32);
    }
}

fn sin_pi_approx(x: f32) -> f32 {
    cos_pi_approx(x - 0.5)
}

fn cos_pi_approx(x: f32) -> f32 {
    let x = (x.rem_euclid(2.0) - 1.0).abs();
    -4.0 * x * x * x + 6.0 * x * x - 1.0
}
