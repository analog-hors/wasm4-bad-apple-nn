use std::f32::consts::PI;

pub const POINT_DIMS: usize = (13 + 7 + 8) * 2;
pub const EMBEDDINGS: usize = 820;

pub fn encode_point(input: &mut [f32; POINT_DIMS], i: f32, y: f32, x: f32) {
    encode_sin(&mut input[ 0..13], i);
    encode_cos(&mut input[13..26], i);
    encode_sin(&mut input[26..32], y);
    encode_cos(&mut input[32..38], y);
    encode_sin(&mut input[38..45], x);
    encode_cos(&mut input[45..52], x);
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
