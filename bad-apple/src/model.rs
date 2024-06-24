use input_encoding::*;

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
    weight: [[i8; O]; I],
    bias: [i32; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    fn forward(&self, input: &[i8; I], output: &mut [i32; O]) {
        *output = self.bias;
        for i in 0..I {
            for o in 0..O {
                output[o] += input[i] as i32 * self.weight[i][o] as i32;
            }
        }
    }

    fn init_accumulator(&self, output: &mut [i32; O]) {
        *output = self.bias;
    }

    #[inline(never)]
    fn partial_forward(&self, offset: usize, input: &[i8], output: &mut [i32; O]) {
        for i in 0..input.len() {
            for o in 0..O {
                output[o] += input[i] as i32 * self.weight[offset + i][o] as i32;
            }
        }
    }
}

include!("../../model.rs");

pub type Accumulator = [i32; 128];

pub fn init_accumulator(acc: &mut Accumulator) {
    L0.init_accumulator(acc);
}

pub fn add_time_features(t: f32, acc: &mut Accumulator) {
    add_sin_feature::<{T_SIN_LAYOUT.offset}, {T_SIN_LAYOUT.dims}>(t, acc);
    add_cos_feature::<{T_COS_LAYOUT.offset}, {T_COS_LAYOUT.dims}>(t, acc);
    add_embedding_0(t, acc);
    add_embedding_1(t, acc);
}

pub fn add_y_features(y: f32, acc: &mut Accumulator) {
    add_sin_feature::<{Y_SIN_LAYOUT.offset}, {Y_SIN_LAYOUT.dims}>(y, acc);
    add_cos_feature::<{Y_COS_LAYOUT.offset}, {Y_COS_LAYOUT.dims}>(y, acc);
}

pub fn add_x_features(x: f32, acc: &mut Accumulator) {
    add_sin_feature::<{X_SIN_LAYOUT.offset}, {X_SIN_LAYOUT.dims}>(x, acc);
    add_cos_feature::<{X_COS_LAYOUT.offset}, {X_COS_LAYOUT.dims}>(x, acc);
}

fn add_embedding_0(t: f32, acc: &mut Accumulator) {
    let em1_index = ((t * EM0.weight.len() as f32) as usize).min(EM0.weight.len() - 1);
    let em2_index = (em1_index + 1).min(EM0.weight.len() - 1);
    let progress = (t * EM0.weight.len() as f32).fract();

    let mut em1 = [0.0; 20];
    EM0.forward(em1_index, &mut em1);

    let mut em2 = [0.0; 20];
    EM0.forward(em2_index, &mut em2);

    let mut em = [0.0; 20];
    for i in 0..em.len() {
        em[i] = em1[i] * (1.0 - progress) + em2[i] * progress;
    }

    let mut em_scaled = [0; 20];
    scale_input(&em, &mut em_scaled);

    L0.partial_forward(POINT_DIMS, &em_scaled, acc);
}

fn add_embedding_1(t: f32, acc: &mut Accumulator) {
    let em1_index = ((t * EM1.weight.len() as f32) as usize).min(EM1.weight.len() - 1);
    let em2_index = (em1_index + 1).min(EM1.weight.len() - 1);
    let progress = (t * EM1.weight.len() as f32).fract();

    let mut em1 = [0.0; 44];
    EM1.forward(em1_index, &mut em1);

    let mut em2 = [0.0; 44];
    EM1.forward(em2_index, &mut em2);

    let mut em = [0.0; 44];
    for i in 0..em.len() {
        em[i] = em1[i] * (1.0 - progress) + em2[i] * progress;
    }

    let mut em_scaled = [0; 44];
    scale_input(&em, &mut em_scaled);

    L0.partial_forward(POINT_DIMS + 20, &em_scaled, acc);
}

pub fn decode(accumulator: &Accumulator) -> f32 {
    let mut l0_output_c = [0; 128];
    mish(accumulator, &mut l0_output_c);

    let mut l1_output = [0; 96];
    L1.forward(&l0_output_c, &mut l1_output);
    
    let mut l1_output_c = [0; 96];
    mish(&l1_output, &mut l1_output_c);

    let mut l2_output = [0; 1];
    L2.forward(&l1_output_c, &mut l2_output);

    sigmoid(l2_output[0] as f32 / (INPUT_SCALE as i32 * WEIGHT_SCALE as i32) as f32)
}

fn add_sin_feature<const OFFSET: usize, const DIMS: usize>(n: f32, acc: &mut Accumulator) {
    let mut input: [f32; DIMS] = [0.0; DIMS];
    encode_sin(&mut input, n);

    let mut input_scaled = [0; DIMS];
    scale_input(&input, &mut input_scaled);

    L0.partial_forward(OFFSET, &input_scaled, acc);
}

fn add_cos_feature<const OFFSET: usize, const DIMS: usize>(n: f32, acc: &mut Accumulator) {
    let mut input: [f32; DIMS] = [0.0; DIMS];
    encode_cos(&mut input, n);

    let mut input_scaled = [0; DIMS];
    scale_input(&input, &mut input_scaled);

    L0.partial_forward(OFFSET, &input_scaled, acc);
}

fn scale_input<const LEN: usize>(input: &[f32; LEN], output: &mut [i8; LEN]) {
    for (o, i) in output.iter_mut().zip(input) {
        *o = ((i * INPUT_SCALE as f32).round() as i32).clamp(-127, 127) as i8;
    }
}

pub fn decoder_size() -> usize {
    std::mem::size_of_val(&EM0)
        + std::mem::size_of_val(&EM1)
        + std::mem::size_of_val(&L0)
        + std::mem::size_of_val(&L1)
        + std::mem::size_of_val(&L2)
}
