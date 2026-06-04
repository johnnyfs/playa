use bevy::prelude::*;

use crate::action::Action;
use crate::config::ZPrimeConfig;
use crate::sim::components::{
    Direction, FacingDirection, Health, Hitbox, LocalPosition, OverworldPosition, Player, StableId,
    SwordState, Velocity, WorldRegion,
};
use crate::sim::resources::{
    CurrentAction, NextStableId, Screen, SimulationTick, TerrainType, WorldMap,
};

pub fn insert_world(app: &mut App, config: ZPrimeConfig) {
    let width_tiles = config.screen_width_pixels / config.tile_size_pixels;
    let height_tiles = config.screen_height_pixels / config.tile_size_pixels;

    let mut screens = Vec::new();
    for screen_y in 0..config.world_height {
        for screen_x in 0..config.world_width {
            let mut screen = Screen {
                screen_x,
                screen_y,
                width_tiles,
                height_tiles,
                terrain: vec![TerrainType::Floor; (width_tiles * height_tiles) as usize],
            };
            add_outer_boundary_walls(&mut screen, &config);
            add_room_obstructions(&mut screen);
            screens.push(screen);
        }
    }

    app.insert_resource(WorldMap {
        config: config.clone(),
        screens,
    })
    .insert_resource(CurrentAction(Action::Noop))
    .insert_resource(SimulationTick(0))
    .insert_resource(NextStableId(2));

    app.world_mut().spawn((
        StableId(1),
        Player,
        WorldRegion::Overworld(OverworldPosition {
            screen_x: 0,
            screen_y: 0,
            indoors: false,
        }),
        LocalPosition { x: 32.0, y: 32.0 },
        Velocity { x: 0.0, y: 0.0 },
        FacingDirection(Direction::Down),
        Hitbox {
            offset_x: 4.0,
            offset_y: 8.0,
            width: 8.0,
            height: 8.0,
        },
        Health { hp: 6 },
        SwordState::default(),
    ));
}

fn add_outer_boundary_walls(screen: &mut Screen, config: &ZPrimeConfig) {
    for x in 0..screen.width_tiles {
        if screen.screen_y == 0 {
            screen.set_terrain(x, 0, TerrainType::Wall);
        }
        if screen.screen_y == config.world_height - 1 {
            screen.set_terrain(x, screen.height_tiles - 1, TerrainType::Wall);
        }
    }

    for y in 0..screen.height_tiles {
        if screen.screen_x == 0 {
            screen.set_terrain(0, y, TerrainType::Wall);
        }
        if screen.screen_x == config.world_width - 1 {
            screen.set_terrain(screen.width_tiles - 1, y, TerrainType::Wall);
        }
    }
}

fn add_room_obstructions(screen: &mut Screen) {
    if screen.width_tiles < 12 || screen.height_tiles < 9 {
        return;
    }

    match (screen.screen_x, screen.screen_y) {
        (0, 0) => {
            set_wall_run(screen, &[(5, 4), (6, 4), (7, 4), (5, 5)]);
        }
        (1, 0) => {
            set_wall_run(screen, &[(4, 3), (4, 4), (4, 5), (9, 6), (10, 6)]);
        }
        (0, 1) => {
            set_wall_run(screen, &[(6, 2), (7, 2), (8, 2), (8, 3), (8, 4)]);
        }
        (1, 1) => {
            set_wall_run(screen, &[(5, 5), (6, 5), (7, 5), (7, 4), (10, 3)]);
        }
        _ => {}
    }
}

fn set_wall_run(screen: &mut Screen, tiles: &[(u32, u32)]) {
    for &(x, y) in tiles {
        if x < screen.width_tiles && y < screen.height_tiles {
            screen.set_terrain(x, y, TerrainType::Wall);
        }
    }
}
