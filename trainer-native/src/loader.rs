use std::path::Path;
use image::GrayImage;
use image::io::Reader as ImageReader;
use rand::{Rng, SeedableRng};
use rand_pcg::Mcg128Xsl64;
use thiserror::Error;

use super::input;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("image decode error: {0}")]
    ImageDecodeError(#[from] image::ImageError),
}

pub struct Loader {
    images: Vec<GrayImage>,
    rng: Mcg128Xsl64,
}

impl Loader {
    pub fn new(path: impl AsRef<Path>, seed: u64) -> Result<Self, LoaderError> {
        let mut images = Vec::new();
        let rng = Mcg128Xsl64::seed_from_u64(seed);

        let mut files = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.metadata()?.is_file() {
                files.push(entry.path());
            }
        }

        files.sort();
        for file in files {
            let reader = ImageReader::open(file)?;
            images.push(reader.decode()?.to_luma8());
        }

        Ok(Loader { images, rng })
    }

    pub fn fill_batch(&mut self, points: &mut [[f32; input::POINT_DIMS]], targets: &mut [f32]) {
        for (point, target) in points.iter_mut().zip(targets) {
            let i = self.rng.gen_range(0..self.images.len());
            let y = self.rng.gen_range(0..self.images[i].height());
            let x = self.rng.gen_range(0..self.images[i].width());

            let ir = i as f32 / (self.images.len() - 1) as f32;
            let yr = y as f32 / (self.images[i].height() - 1) as f32;
            let xr = x as f32 / (self.images[i].width() - 1) as f32;
            input::encode_sample(point, ir, yr, xr);

            let pixel = self.images[i].get_pixel(x, y).0[0];
            *target = pixel as f32 / 255.0;
        }
    }
}
