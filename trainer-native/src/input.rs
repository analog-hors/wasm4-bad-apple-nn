use std::f32::consts::PI;

pub const POINT_DIMS: usize = 80;

pub fn encode_sample(input: &mut [f32; POINT_DIMS], i: f32, y: f32, x: f32) {
    encode_sin(&mut input[ 0..20], i);
    encode_cos(&mut input[20..40], i);
    encode_sin(&mut input[40..50], y);
    encode_cos(&mut input[50..60], y);
    encode_sin(&mut input[60..70], x);
    encode_cos(&mut input[70..80], x);
}

fn encode_sin(input: &mut [f32], n: f32) {
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((n - 0.5) * PI * (1 << i) as f32).sin();
    }
}

fn encode_cos(input: &mut [f32], n: f32) {
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((n - 0.5) * PI * (1 << i) as f32).cos();
    }
}
