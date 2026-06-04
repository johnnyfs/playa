use crate::config::ZPrimeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenViewport {
    pub width_pixels: u32,
    pub height_pixels: u32,
    pub width_tiles: u32,
    pub height_tiles: u32,
}

pub fn screen_viewport(config: &ZPrimeConfig) -> ScreenViewport {
    ScreenViewport {
        width_pixels: config.screen_width_pixels,
        height_pixels: config.screen_height_pixels,
        width_tiles: config.screen_width_pixels / config.tile_size_pixels,
        height_tiles: config.screen_height_pixels / config.tile_size_pixels,
    }
}
