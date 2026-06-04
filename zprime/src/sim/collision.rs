use crate::sim::components::{Hitbox, LocalPosition};
use crate::sim::resources::Screen;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

pub fn hitbox_rect(position: LocalPosition, hitbox: Hitbox) -> Rect {
    Rect {
        min_x: position.x + hitbox.offset_x,
        min_y: position.y + hitbox.offset_y,
        max_x: position.x + hitbox.offset_x + hitbox.width,
        max_y: position.y + hitbox.offset_y + hitbox.height,
    }
}

pub fn collides_with_solid(
    screen: &Screen,
    position: LocalPosition,
    hitbox: Hitbox,
    tile_size: u32,
) -> bool {
    let rect = hitbox_rect(position, hitbox);
    let screen_width = (screen.width_tiles * tile_size) as f32;
    let screen_height = (screen.height_tiles * tile_size) as f32;

    if rect.min_x < 0.0
        || rect.min_y < 0.0
        || rect.max_x > screen_width
        || rect.max_y > screen_height
    {
        return true;
    }

    let tile_size = tile_size as f32;
    let min_tile_x = (rect.min_x / tile_size).floor() as u32;
    let min_tile_y = (rect.min_y / tile_size).floor() as u32;
    let max_tile_x = ((rect.max_x - 0.001) / tile_size).floor() as u32;
    let max_tile_y = ((rect.max_y - 0.001) / tile_size).floor() as u32;

    for tile_y in min_tile_y..=max_tile_y {
        for tile_x in min_tile_x..=max_tile_x {
            if !screen.terrain_at(tile_x, tile_y).is_walkable() {
                return true;
            }
        }
    }

    false
}
