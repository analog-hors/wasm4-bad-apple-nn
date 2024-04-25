use std::ffi::{CStr, c_char};

mod input;
mod loader;

use loader::Loader;

#[no_mangle]
pub unsafe extern "C" fn point_dims() -> u64 {
    input::POINT_DIMS as u64
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
    points: *mut [f32; input::POINT_DIMS],
    targets: *mut f32,
    batch_size: u64,
) {
    let points = std::slice::from_raw_parts_mut(points, batch_size as usize);
    let targets = std::slice::from_raw_parts_mut(targets, batch_size as usize);
    (*loader).fill_batch(points, targets);
}

#[no_mangle]
pub unsafe extern "C" fn loader_drop(loader: *mut Loader) {
    drop(Box::from_raw(loader));
}
