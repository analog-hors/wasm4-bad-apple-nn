use std::ffi::{CStr, c_char};

mod loader;

use loader::Loader;

#[no_mangle]
pub unsafe extern "C" fn point_dims() -> u64 {
    input_encoding::POINT_DIMS as u64
}

#[no_mangle]
pub unsafe extern "C" fn embeddings() -> u64 {
    input_encoding::EMBEDDINGS as u64
}

#[no_mangle]
pub unsafe extern "C" fn encode_frame_points(
    points: *mut [f32; input_encoding::POINT_DIMS],
    width: u32,
    height: u32,
    frame: f32,
) {
    let points = std::slice::from_raw_parts_mut(points, width as usize * height as usize);
    input_encoding::encode_frame(points, width as usize, height as usize, frame);
}

#[no_mangle]
pub unsafe extern "C" fn encode_frame_embedding(frame: f32) -> f32 {
    input_encoding::encode_embedding(frame)
}

#[no_mangle]
pub unsafe extern "C" fn loader_new(path: *const c_char, seed: u64) -> *mut Loader {
    let path = CStr::from_ptr(path);
    let Ok(path) = path.to_str() else {
        return std::ptr::null_mut();
    };
    let Ok(loader) = Loader::new(path, seed) else {
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(loader))
}

#[no_mangle]
pub unsafe extern "C" fn loader_fill_batch(
    loader: *mut Loader,
    points: *mut [f32; input_encoding::POINT_DIMS],
    embeddings: *mut f32,
    targets: *mut f32,
    batch_size: u64,
) {
    let points = std::slice::from_raw_parts_mut(points, batch_size as usize);
    let embeddings = std::slice::from_raw_parts_mut(embeddings, batch_size as usize);
    let targets = std::slice::from_raw_parts_mut(targets, batch_size as usize);
    (*loader).fill_batch(points, embeddings, targets);
}

#[no_mangle]
pub unsafe extern "C" fn loader_drop(loader: *mut Loader) {
    drop(Box::from_raw(loader));
}
