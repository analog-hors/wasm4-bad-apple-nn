use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
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

        let (send, recv) = sync_channel(1);
        std::thread::spawn(move || loop {
            let batch = make_batch(&images, &mut rng, batch_size);
            send.send(batch).unwrap();
        });

        Ok(Loader { recv })
    }

    pub fn next_batch(&mut self) -> Batch {
        self.recv.recv().unwrap()
    }
}

fn make_batch(images: &[GrayImage], rng: &mut impl Rng, batch_size: usize) -> Batch {
    let mut batch = Batch {
        points: Vec::with_capacity(batch_size),
        embeddings: Vec::with_capacity(batch_size),
        targets: Vec::with_capacity(batch_size),
    };

    for _ in 0..batch_size {
        let i = rng.gen_range(0..images.len());
        let y = rng.gen_range(0..images[i].height());
        let x = rng.gen_range(0..images[i].width());

        let mut point = [0.0; input_encoding::POINT_DIMS];
        let ir = i as f32 / (images.len() - 1) as f32;
        let yr = y as f32 / (images[i].height() - 1) as f32;
        let xr = x as f32 / (images[i].width() - 1) as f32;
        input_encoding::encode_point(&mut point, ir, yr, xr);

        let embedding = input_encoding::encode_embedding(ir);

        let pixel = images[i].get_pixel(x, y).0[0];
        let target = pixel as f32 / 255.0;

        batch.points.push(point);
        batch.embeddings.push(embedding);
        batch.targets.push(target);
    }
    
    batch
}
