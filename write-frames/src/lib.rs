use std::io::Cursor;
use image::{GrayImage, ImageFormat};

const FRAMES: usize = 6572;
const WIDTH: u32 = 160;
const HEIGHT: u32 = 120;

#[no_mangle]
pub extern "C" fn main() {
    log!("decoder size: {}", bad_apple::decoder_size());
    let mut image = GrayImage::new(WIDTH, HEIGHT);
    for i in 0..FRAMES {
        let start = now();

        let mut acc = [0; 128];
        bad_apple::init_accumulator(&mut acc);
        bad_apple::add_time_features(i as f32 / (FRAMES - 1) as f32, &mut acc);
        for y in 0..image.height() {
            let mut acc = acc;
            bad_apple::add_y_features(y as f32 / (image.height() - 1) as f32, &mut acc);
            for x in 0..image.width() {
                let mut acc = acc;
                bad_apple::add_x_features(x as f32 / (image.width() - 1) as f32, &mut acc);
                let p = bad_apple::decode(&acc);
                image.get_pixel_mut(x, y).0 = [(p * 255.0).round() as u8];
            }
        }
        let elapsed = now() - start;
        log!("{:.2}", elapsed);
        write_image(&image, &format!("decoded/{:04}.png", i));
    }
}

fn write_image(image: &GrayImage, path: &str) {
    let mut buffer = Vec::new();
    image.write_to(&mut Cursor::new(&mut buffer), ImageFormat::Png).unwrap();

    unsafe {
        write_file(
            path.as_ptr(),
            path.len() as u32,
            buffer.as_ptr(),
            buffer.len() as u32,
        );
    }
}

fn now() -> f64 {
    unsafe { time() }
}

#[macro_export]
macro_rules! log {
    ($($body:tt)*) => {{
        let message = format!($($body)*);
        unsafe {
            print(message.as_ptr(), message.len() as u32);
        }
    }};
}

extern "C" {
    fn time() -> f64;

    fn print(message: *const u8, size: u32);

    fn write_file(path: *const u8, path_size: u32, data: *const u8, data_size: u32);
}
