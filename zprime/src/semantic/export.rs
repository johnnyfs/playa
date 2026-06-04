use std::collections::HashMap;

use bevy::prelude::*;

use crate::semantic::state::{
    SemanticEntity, SemanticState, SemanticTerrain, SemanticTile, SemanticValue,
};
use crate::sim::collision::hitbox_rect;
use crate::sim::components::{
    FacingDirection, Health, Hitbox, LocalPosition, Player, StableId, SwordHitbox, SwordPhase,
    SwordState, WorldRegion,
};
use crate::sim::resources::{SimulationTick, WorldMap};

pub fn semantic_state(world: &mut World) -> SemanticState {
    let tick = world.resource::<SimulationTick>().0;
    let world_map = world.resource::<WorldMap>().clone();

    let (player_id, region, player_position, facing, player_hitbox, health, sword_state) = {
        let mut player_query = world.query_filtered::<(
            &StableId,
            &WorldRegion,
            &LocalPosition,
            &FacingDirection,
            &Hitbox,
            &Health,
            &SwordState,
        ), With<Player>>();
        let (player_id, region, player_position, facing, player_hitbox, health, sword_state) =
            player_query
                .single(world)
                .expect("semantic_state requires one player");
        (
            *player_id,
            *region,
            *player_position,
            *facing,
            *player_hitbox,
            *health,
            *sword_state,
        )
    };
    let (screen_x, screen_y) = region
        .overworld_screen()
        .expect("phase 0 semantic_state only supports overworld");

    let mut player_attrs = hitbox_attrs(player_position, player_hitbox);
    player_attrs.insert(
        "sword_active".to_string(),
        SemanticValue::Bool(matches!(sword_state.phase, SwordPhase::Swinging { .. })),
    );
    if let SwordPhase::Swinging { ticks_remaining } = sword_state.phase {
        player_attrs.insert(
            "sword_ticks_remaining".to_string(),
            SemanticValue::Int(ticks_remaining as i64),
        );
    }

    let player = SemanticEntity {
        id: player_id.0,
        kind: "player".to_string(),
        type_name: "Player".to_string(),
        screen_x,
        screen_y,
        local_x: player_position.x,
        local_y: player_position.y,
        facing: Some(facing.0),
        hp: Some(health.hp),
        alive: health.hp > 0,
        attrs: player_attrs,
    };

    let mut entities = Vec::new();
    {
        let mut sword_query = world.query_filtered::<(
            &StableId,
            &WorldRegion,
            &LocalPosition,
            &Hitbox,
        ), With<SwordHitbox>>();
        for (id, sword_region, sword_position, sword_hitbox) in sword_query.iter(world) {
            let Some((sword_screen_x, sword_screen_y)) = sword_region.overworld_screen() else {
                continue;
            };
            entities.push(SemanticEntity {
                id: id.0,
                kind: "sword".to_string(),
                type_name: "SwordHitbox".to_string(),
                screen_x: sword_screen_x,
                screen_y: sword_screen_y,
                local_x: sword_position.x,
                local_y: sword_position.y,
                facing: Some(facing.0),
                hp: None,
                alive: true,
                attrs: hitbox_attrs(*sword_position, *sword_hitbox),
            });
        }
    }
    entities.sort_by_key(|entity| entity.id);

    let screen = world_map
        .screen(screen_x, screen_y)
        .expect("player screen must exist in world map");
    let mut tiles = Vec::with_capacity(screen.terrain.len());
    for y in 0..screen.height_tiles {
        for x in 0..screen.width_tiles {
            let terrain = screen.terrain_at(x, y);
            tiles.push(SemanticTile {
                x,
                y,
                terrain_type: terrain.as_str().to_string(),
                walkable: terrain.is_walkable(),
            });
        }
    }

    let mut attrs = HashMap::new();
    attrs.insert(
        "fixed_timestep_hz".to_string(),
        SemanticValue::Int(world_map.config.fixed_timestep_hz as i64),
    );

    SemanticState {
        tick,
        region,
        player,
        entities,
        terrain: SemanticTerrain {
            screen_x,
            screen_y,
            width_tiles: screen.width_tiles,
            height_tiles: screen.height_tiles,
            tiles,
        },
        attrs,
    }
}

fn hitbox_attrs(position: LocalPosition, hitbox: Hitbox) -> HashMap<String, SemanticValue> {
    let rect = hitbox_rect(position, hitbox);
    HashMap::from([
        (
            "hitbox_offset_x".to_string(),
            SemanticValue::Float(hitbox.offset_x as f64),
        ),
        (
            "hitbox_offset_y".to_string(),
            SemanticValue::Float(hitbox.offset_y as f64),
        ),
        (
            "hitbox_width".to_string(),
            SemanticValue::Float(hitbox.width as f64),
        ),
        (
            "hitbox_height".to_string(),
            SemanticValue::Float(hitbox.height as f64),
        ),
        (
            "hitbox_min_x".to_string(),
            SemanticValue::Float(rect.min_x as f64),
        ),
        (
            "hitbox_min_y".to_string(),
            SemanticValue::Float(rect.min_y as f64),
        ),
        (
            "hitbox_max_x".to_string(),
            SemanticValue::Float(rect.max_x as f64),
        ),
        (
            "hitbox_max_y".to_string(),
            SemanticValue::Float(rect.max_y as f64),
        ),
    ])
}
