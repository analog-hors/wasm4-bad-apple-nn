const PALETTE: *mut [u32; 4] = 0x04 as _;
const SYSTEM_FLAGS: *mut u8 = 0x001f as _;
const FRAMEBUFFER: *mut [u8; 6400] = 0xa0 as _;

const FRAME_COUNT: usize = 6572 * 2;

static mut FRAME: usize = 0;

#[no_mangle]
unsafe fn update() {
    if FRAME == 0 {
        *PALETTE = [0x000000, 0x555555, 0xAAAAAA, 0xFFFFFF];
        *SYSTEM_FLAGS = 0b11;
        (*FRAMEBUFFER).fill(0);
    }

    let (sx, sy) = match FRAME % 8 {
        0 => (0, 0),
        1 => (2, 0),
        2 => (3, 1),
        3 => (1, 1),
        4 => (1, 0),
        5 => (3, 0),
        6 => (2, 1),
        7 => (0, 1),
        _ => unreachable!(),
    };
    for fby in (sy..120).step_by(2) {
        for fbx in (sx..160).step_by(4) {
            let index = (fby + 20) * 160 + fbx;
            let pixel = bad_apple::model(
                FRAME as f32 / FRAME_COUNT as f32,
                fby as f32 / 120.0,
                fbx as f32 / 160.0,
            );
            let pixel = if pixel < 0.25 {
                0
            } else if pixel < 0.50 {
                1
            } else if pixel < 0.75 {
                2
            } else {
                3
            };
            (*FRAMEBUFFER)[index / 4] &= !(0b11 << index % 4 * 2);
            (*FRAMEBUFFER)[index / 4] |= pixel << index % 4 * 2;
        }
    }

    FRAME = (FRAME + 1) % FRAME_COUNT;
}
