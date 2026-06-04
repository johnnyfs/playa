use bevy::prelude::*;

use crate::action::Action;
use crate::sim::collision::collides_with_solid;
use crate::sim::components::{
    Direction, FacingDirection, Hitbox, LocalPosition, OverworldPosition, Player, Velocity,
    WorldRegion,
};
use crate::sim::resources::{CurrentAction, WorldMap};

pub fn movement_system(
    action: Res<CurrentAction>,
    world_map: Res<WorldMap>,
    mut players: Query<
        (
            &mut WorldRegion,
            &mut LocalPosition,
            &mut Velocity,
            &mut FacingDirection,
            &Hitbox,
        ),
        With<Player>,
    >,
) {
    let Ok((mut region, mut position, mut velocity, mut facing, hitbox)) = players.single_mut()
    else {
        return;
    };

    let (direction, dx, dy) = match action.0 {
        Action::MoveUp => (
            Direction::Up,
            0.0,
            -world_map.config.player_speed_pixels_per_tick,
        ),
        Action::MoveDown => (
            Direction::Down,
            0.0,
            world_map.config.player_speed_pixels_per_tick,
        ),
        Action::MoveLeft => (
            Direction::Left,
            -world_map.config.player_speed_pixels_per_tick,
            0.0,
        ),
        Action::MoveRight => (
            Direction::Right,
            world_map.config.player_speed_pixels_per_tick,
            0.0,
        ),
        _ => {
            velocity.x = 0.0;
            velocity.y = 0.0;
            return;
        }
    };

    facing.0 = direction;
    velocity.x = dx;
    velocity.y = dy;

    let WorldRegion::Overworld(current_overworld) = *region else {
        return;
    };

    let mut next_region = current_overworld;
    let mut next_position = LocalPosition {
        x: position.x + dx,
        y: position.y + dy,
    };

    let screen_width = world_map.config.screen_width_pixels as f32;
    let screen_height = world_map.config.screen_height_pixels as f32;
    let min_x = next_position.x + hitbox.offset_x;
    let max_x = min_x + hitbox.width;
    let min_y = next_position.y + hitbox.offset_y;
    let max_y = min_y + hitbox.height;

    if min_x < 0.0 {
        if current_overworld.screen_x == 0 {
            velocity.x = 0.0;
            return;
        }
        next_region.screen_x -= 1;
        next_position.x = screen_width - hitbox.offset_x - hitbox.width;
    } else if max_x > screen_width {
        if current_overworld.screen_x >= world_map.config.world_width - 1 {
            velocity.x = 0.0;
            return;
        }
        next_region.screen_x += 1;
        next_position.x = -hitbox.offset_x;
    }

    if min_y < 0.0 {
        if current_overworld.screen_y == 0 {
            velocity.y = 0.0;
            return;
        }
        next_region.screen_y -= 1;
        next_position.y = screen_height - hitbox.offset_y - hitbox.height;
    } else if max_y > screen_height {
        if current_overworld.screen_y >= world_map.config.world_height - 1 {
            velocity.y = 0.0;
            return;
        }
        next_region.screen_y += 1;
        next_position.y = -hitbox.offset_y;
    }

    if !world_map.in_overworld_bounds(next_region.screen_x, next_region.screen_y) {
        return;
    }

    let Some(next_screen) = world_map.screen(next_region.screen_x, next_region.screen_y) else {
        return;
    };

    if collides_with_solid(
        next_screen,
        next_position,
        *hitbox,
        world_map.config.tile_size_pixels,
    ) {
        velocity.x = 0.0;
        velocity.y = 0.0;
        return;
    }

    *region = WorldRegion::Overworld(OverworldPosition {
        screen_x: next_region.screen_x,
        screen_y: next_region.screen_y,
        indoors: false,
    });
    *position = next_position;
}
