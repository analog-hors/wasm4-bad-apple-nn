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

pub fn model(i: f32, y: f32, x : f32) -> f32 {
    let mut input = [0.0; 56 + 32];
    encode_input(i, y, x, &mut input);

    let mut l0_output = [0.0; 128];
    L0.forward(&mut input, &mut l0_output);
    activation(mish, &mut l0_output);

    let mut l1_output = [0.0; 128];
    L1.forward(&mut l0_output, &mut l1_output);
    activation(mish, &mut l1_output);

    let mut l2_output = [0.0; 1];
    L2.forward(&mut l1_output, &mut l2_output);
    activation(sigmoid, &mut l2_output);

    l2_output[0]
}

fn encode_input(i: f32, y: f32, x : f32, output: &mut [f32; 56 + 32]) {
    let em1i = ((i * EM.weight.len() as f32) as usize).min(EM.weight.len() - 1);
    let em2i = (em1i + 1).min(EM.weight.len() - 1);
    let res = (i * EM.weight.len() as f32).fract();

    let mut em1 = [0.0; 32];
    EM.forward(em1i, &mut em1);

    let mut em2 = [0.0; 32];
    EM.forward(em2i, &mut em2);

    let point = &mut bytemuck::cast_slice_mut::<_, [f32; 56]>(&mut output[..56])[0];
    input_encoding::encode_point(point, i, y, x);
    for i in 0..32 {
        output[56 + i] = em1[i] * (1.0 - res) + em2[i] * res;
    }
}

pub fn decoder_size() -> usize {
    std::mem::size_of_val(&EM)
        + std::mem::size_of_val(&L0)
        + std::mem::size_of_val(&L1)
        + std::mem::size_of_val(&L2)
}
