use crate::config::ZPrimeConfig;
use crate::sim::components::WorldRegion;
use crate::sim::resources::WorldMap;

pub fn screen_camera_translation(world_map: &WorldMap, region: WorldRegion) -> Option<(f32, f32)> {
    let (screen_x, screen_y) = region.overworld_screen()?;
    Some(screen_camera_translation_for_config(
        &world_map.config,
        screen_x,
        screen_y,
    ))
}

pub fn screen_camera_translation_for_screen(
    world_map: &WorldMap,
    screen_x: i32,
    screen_y: i32,
) -> (f32, f32) {
    screen_camera_translation_for_config(&world_map.config, screen_x, screen_y)
}

pub fn screen_camera_translation_for_config(
    config: &ZPrimeConfig,
    screen_x: i32,
    screen_y: i32,
) -> (f32, f32) {
    let world_width_px = config.world_width as f32 * config.screen_width_pixels as f32;
    let world_height_px = config.world_height as f32 * config.screen_height_pixels as f32;
    let screen_origin_x = screen_x as f32 * config.screen_width_pixels as f32;
    let screen_origin_y = screen_y as f32 * config.screen_height_pixels as f32;
    (
        -world_width_px / 2.0 + screen_origin_x + config.screen_width_pixels as f32 / 2.0,
        world_height_px / 2.0 - (screen_origin_y + config.screen_height_pixels as f32 / 2.0),
    )
}
