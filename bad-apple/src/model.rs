fn mish<const LEN: usize>(input: &[i32; LEN], output: &mut [i8; LEN]) {
    for (o, i) in output.iter_mut().zip(input) {
        *o = (*i / WEIGHT_SCALE as i32).clamp(0, INPUT_SCALE as i32) as i8;
    }
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
            output[o] = self.weight[input][o] as f32 / WEIGHT_SCALE as f32;
        }
    }
}

pub struct Linear<const I: usize, const O: usize> {
    weight: [[i8; I]; O],
    bias: [i32; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    fn forward(&self, input: &[i8; I], output: &mut [i32; O]) {
        *output = self.bias;
        for o in 0..O {
            for i in 0..I {
                output[o] += input[i] as i32 * self.weight[o][i] as i32;
            }
        }
    }
}

include!("../../model.rs");

pub fn model(i: f32, y: f32, x : f32) -> f32 {
    let mut input = [0.0; 56 + 32];
    encode_input(i, y, x, &mut input);

    let mut input_c = [0; 56 + 32];
    for (o, i) in input_c.iter_mut().zip(&input) {
        *o = ((i * INPUT_SCALE as f32).round() as i32).clamp(-127, 127) as i8;
    }

    let mut l0_output = [0; 128];
    L0.forward(&input_c, &mut l0_output);

    let mut l0_output_c = [0; 128];
    mish(&l0_output, &mut l0_output_c);

    let mut l1_output = [0; 96];
    L1.forward(&l0_output_c, &mut l1_output);
    
    let mut l1_output_c = [0; 96];
    mish(&l1_output, &mut l1_output_c);

    let mut l2_output = [0; 1];
    L2.forward(&l1_output_c, &mut l2_output);

    sigmoid(l2_output[0] as f32 / (INPUT_SCALE as i32 * WEIGHT_SCALE as i32) as f32)
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
