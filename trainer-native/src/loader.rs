use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
use rayon::prelude::*;
use image::GrayImage;
use image::io::Reader as ImageReader;
use rand::{Rng, SeedableRng};
use rand_pcg::Mcg128Xsl64;
use thiserror::Error;

pub struct Batch {
    pub points: Vec<[f32; input_encoding::POINT_DIMS]>,
    pub embeddings: Vec<f32>,
    pub targets: Vec<f32>,
}

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("image decode error: {0}")]
    ImageDecodeError(#[from] image::ImageError),
}

pub struct Loader {
    recv: Receiver<Batch>,
}

impl Loader {
    pub fn new(path: impl AsRef<Path>, batch_size: usize, seed: u64) -> Result<Self, LoaderError> {
        let mut images = Vec::new();
        let mut rng = Mcg128Xsl64::seed_from_u64(seed);

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
            images.push(reader.decode()?.into_luma8());
        }

        let (send, recv) = sync_channel(0);
        std::thread::spawn(move || loop {
            let batch = make_batch(&images, &mut rng, batch_size);
            if send.send(batch).is_err() {
                break;
            }
        });

        Ok(Loader { recv })
    }

    pub fn next_batch(&mut self) -> Batch {
        self.recv.recv().unwrap()
    }
}

fn make_batch(images: &[GrayImage], rng: &mut Mcg128Xsl64, batch_size: usize) -> Batch {
    let mut points = vec![[0.0; input_encoding::POINT_DIMS]; batch_size];
    let mut embeddings = vec![0.0; batch_size];
    let mut targets = vec![0.0; batch_size];

    let mut jobs = Vec::with_capacity(batch_size);
    for ((point, embedding), target) in points.iter_mut().zip(&mut embeddings).zip(&mut targets) {
        let t = rng.gen_range(0..images.len());
        let y = rng.gen_range(0..images[t].height());
        let x = rng.gen_range(0..images[t].width());
        jobs.push(((t, y, x), (point, embedding, target)));
    }
    jobs.into_par_iter().for_each(|((t, y, x), (point, embedding, target))| {
        let tr = t as f32 / (images.len() - 1) as f32;
        let yr = y as f32 / (images[t].height() - 1) as f32;
        let xr = x as f32 / (images[t].width() - 1) as f32;
        input_encoding::encode_point(point, tr, yr, xr);
        *embedding = input_encoding::encode_embedding(tr);

        let pixel = images[t].get_pixel(x, y).0[0];
        *target = pixel as f32 / 255.0;
    });

    Batch { points, embeddings, targets }
}
