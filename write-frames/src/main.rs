fn main() {
    println!("decoder size: {}", bad_apple::decoder_size());
    let mut buffer = [0; bad_apple::DECODER_BUFFER_SIZE];
    let mut image = image::GrayImage::new(160, 120);
    for i in 0..6572 {
        let now = std::time::Instant::now();
        for y in 0..image.height() {
            for x in 0..image.width() {
                let ir = i as f32 / (6572 - 1) as f32;
                let yr = y as f32 / (image.height() - 1) as f32;
                let xr = x as f32 / (image.width() - 1) as f32;
                let o = bad_apple::model(&mut buffer, ir, yr, xr);
                image.get_pixel_mut(x, y).0 = [(o * 255.0) as u8];
            }
        }
        println!("{:?}", now.elapsed());
        image.save(format!("decoded/{:04}.png", i)).unwrap();
    }
}
