use bytemuck::Pod;

fn view_flat_mut<T: Pod, U: Pod>(t: &mut T) -> &mut [U] {
    bytemuck::cast_slice_mut(std::slice::from_mut(t))
}

fn activation<T: Pod>(f: impl Fn(f32) -> f32, t: &mut T) {
    for n in view_flat_mut::<_, f32>(t) {
        *n = f(*n);
    }
}

fn softplus(n: f32) -> f32 {
    (1.0 + n.exp()).ln()
}

fn mish(n: f32) -> f32 {
    n * softplus(n).tanh()
}

fn sigmoid(n: f32) -> f32 {
    1.0 / (1.0 + (-n).exp())
}

pub struct Embedding<const I: usize, const O: usize> {
    weight: [[i8; O]; I],
}

impl<const I: usize, const O: usize> Embedding<I, O> {
    fn forward(&self, input: usize, output: &mut [f32; O]) {
        for o in 0..O {
            output[o] = self.weight[input][o] as f32 / WEIGHT_QUANT_RANGE * WEIGHT_CLIP_RANGE;
        }
    }
}

pub struct Linear<const I: usize, const O: usize> {
    weight: [[i8; I]; O],
    bias: [i8; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    fn forward(&self, input: &[f32; I], output: &mut [f32; O]) {
        for o in 0..O {
            output[o] = self.bias[o] as f32 / BIAS_QUANT_RANGE * BIAS_CLIP_RANGE;
            for i in 0..I {
                let w = self.weight[o][i] as f32 / WEIGHT_QUANT_RANGE * WEIGHT_CLIP_RANGE;
                output[o] += w * input[i];
            }
        }
    }
}

include!("../../model.rs");

pub const DECODER_BUFFER_SIZE: usize = std::mem::size_of::<[f32; 1000]>();

struct LayerBuffer<'b, I: Pod> {
    buffer: &'b mut [u8],
    _phantom: std::marker::PhantomData<I>,
}

impl<'b, I: Pod> LayerBuffer<'b, I> {
    fn new(buffer: &'b mut [u8], input: &I) -> Self {
        let input = bytemuck::cast_slice(std::slice::from_ref(input));
        buffer[..input.len()].copy_from_slice(input);
        Self { buffer, _phantom: std::marker::PhantomData }
    }

    fn layer<O: Pod>(self, f: impl FnOnce(&mut I, &mut O)) -> LayerBuffer<'b, O> {
        let input_size = std::mem::size_of::<I>();
        let output_size = std::mem::size_of::<O>();

        let (input, output) = self.buffer[..input_size + output_size].split_at_mut(input_size);
        f(&mut bytemuck::cast_slice_mut(input)[0], &mut bytemuck::cast_slice_mut(output)[0]);
        
        self.buffer.copy_within(input_size..input_size + output_size, 0);
        LayerBuffer { buffer: self.buffer, _phantom: std::marker::PhantomData }
    }

    fn finish(self) -> &'b I {
        let input_size = std::mem::size_of::<I>();
        let input = &mut self.buffer[..input_size];
        &bytemuck::cast_slice(input)[0]
    }
}

type L0Output = [f32; 128];
type L1Output = [f32; 128];
type L2Output = [f32; 1];

pub fn model(buffer: &mut [u8; DECODER_BUFFER_SIZE], i: f32, y: f32, x : f32) -> f32 {
    let em1i = ((i * EM.weight.len() as f32) as usize).min(EM.weight.len() - 1);
    let em2i = (em1i + 1).min(EM.weight.len() - 1);
    let res = (i * EM.weight.len() as f32).fract();

    let mut em1 = [0.0; 32];
    EM.forward(em1i, &mut em1);

    let mut em2 = [0.0; 32];
    EM.forward(em2i, &mut em2);

    let output = LayerBuffer::<[f32; 1]>::new(buffer, &[1.0])
        .layer(|_, output: &mut [f32; 56 + 32]| {
            encode_point(output, i, y, x);
            for i in 0..32 {
                output[56 + i] = em1[i] * (1.0 - res) + em2[i] * res;
            }
        })
        .layer(|input, output: &mut L0Output| {
            L0.forward(input, output);
            activation(mish, output);
        })
        .layer(|input, output: &mut L1Output| {
            L1.forward(input, output);
            activation(mish, output);        
        })
        .layer(|input, output: &mut L2Output| {
            L2.forward(input, output);
            activation(sigmoid, output);
        })
        .finish();
    output[0]
}

pub fn decoder_size() -> usize {
    std::mem::size_of_val(&EM)
        + std::mem::size_of_val(&L0)
        + std::mem::size_of_val(&L1)
        + std::mem::size_of_val(&L2)
}

fn encode_point(input: &mut [f32], i: f32, y: f32, x: f32) {
    encode_sin(&mut input[ 0..13], i);
    encode_cos(&mut input[13..26], i);
    encode_sin(&mut input[26..32], y);
    encode_cos(&mut input[32..38], y);
    encode_sin(&mut input[38..45], x);
    encode_cos(&mut input[45..52], x);
    input[52..56].fill(0.0);
}

fn encode_sin(input: &mut [f32], n: f32) {
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((n - 0.5) * std::f32::consts::PI * (1 << i) as f32).sin();
    }
}

fn encode_cos(input: &mut [f32], n: f32) {
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((n - 0.5) * std::f32::consts::PI * (1 << i) as f32).cos();
    }
}

