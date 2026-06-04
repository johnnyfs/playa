use bevy::prelude::*;

use crate::action::Action;
use crate::sim::components::{
    Direction, FacingDirection, Hitbox, LocalPosition, Player, StableId, SwordHitbox, SwordPhase,
    SwordState, WorldRegion,
};
use crate::sim::resources::{CurrentAction, NextStableId, WorldMap};

const SWORD_LENGTH: f32 = 14.0;
const SWORD_THICKNESS: f32 = 8.0;

pub fn sword_state_system(
    action: Res<CurrentAction>,
    world_map: Res<WorldMap>,
    mut players: Query<&mut SwordState, With<Player>>,
) {
    let Ok(mut sword_state) = players.single_mut() else {
        return;
    };

    match sword_state.phase {
        SwordPhase::Idle => {
            if action.0 == Action::Attack {
                sword_state.phase = SwordPhase::Swinging {
                    ticks_remaining: world_map.config.sword_swing_ticks,
                };
            }
        }
        SwordPhase::Swinging { ticks_remaining } => {
            sword_state.phase = if ticks_remaining > 1 {
                SwordPhase::Swinging {
                    ticks_remaining: ticks_remaining - 1,
                }
            } else {
                SwordPhase::Idle
            };
        }
    }
}

pub fn sword_hitbox_for(
    position: LocalPosition,
    player_hitbox: Hitbox,
    facing: Direction,
) -> (LocalPosition, Hitbox) {
    let player_min_x = position.x + player_hitbox.offset_x;
    let player_min_y = position.y + player_hitbox.offset_y;
    let player_center_x = player_min_x + player_hitbox.width / 2.0;
    let player_center_y = player_min_y + player_hitbox.height / 2.0;

    match facing {
        Direction::Up => (
            LocalPosition {
                x: player_center_x - SWORD_THICKNESS / 2.0,
                y: player_min_y - SWORD_LENGTH,
            },
            Hitbox {
                offset_x: 0.0,
                offset_y: 0.0,
                width: SWORD_THICKNESS,
                height: SWORD_LENGTH,
            },
        ),
        Direction::Down => (
            LocalPosition {
                x: player_center_x - SWORD_THICKNESS / 2.0,
                y: player_min_y + player_hitbox.height,
            },
            Hitbox {
                offset_x: 0.0,
                offset_y: 0.0,
                width: SWORD_THICKNESS,
                height: SWORD_LENGTH,
            },
        ),
        Direction::Left => (
            LocalPosition {
                x: player_min_x - SWORD_LENGTH,
                y: player_center_y - SWORD_THICKNESS / 2.0,
            },
            Hitbox {
                offset_x: 0.0,
                offset_y: 0.0,
                width: SWORD_LENGTH,
                height: SWORD_THICKNESS,
            },
        ),
        Direction::Right => (
            LocalPosition {
                x: player_min_x + player_hitbox.width,
                y: player_center_y - SWORD_THICKNESS / 2.0,
            },
            Hitbox {
                offset_x: 0.0,
                offset_y: 0.0,
                width: SWORD_LENGTH,
                height: SWORD_THICKNESS,
            },
        ),
    }
}

pub fn refresh_sword_hitbox_system(
    mut commands: Commands,
    mut next_id: ResMut<NextStableId>,
    player: Query<
        (
            Entity,
            &WorldRegion,
            &LocalPosition,
            &FacingDirection,
            &Hitbox,
            &SwordState,
        ),
        With<Player>,
    >,
    mut swords: Query<
        (Entity, &mut WorldRegion, &mut LocalPosition, &mut Hitbox),
        (With<SwordHitbox>, Without<Player>),
    >,
) {
    let Ok((player_entity, region, position, facing, player_hitbox, sword_state)) = player.single()
    else {
        return;
    };

    let SwordPhase::Swinging { .. } = sword_state.phase else {
        for (entity, _, _, _) in &mut swords {
            commands.entity(entity).despawn();
        }
        return;
    };

    let (sword_position, sword_hitbox) = sword_hitbox_for(*position, *player_hitbox, facing.0);
    let mut existing = None;
    for (entity, mut sword_region, mut local_position, mut hitbox) in &mut swords {
        *sword_region = *region;
        *local_position = sword_position;
        *hitbox = sword_hitbox;
        existing = Some(entity);
    }

    if existing.is_none() {
        let id = next_id.0;
        next_id.0 += 1;
        commands.spawn((
            StableId(id),
            SwordHitbox {
                owner: player_entity,
            },
            *region,
            sword_position,
            sword_hitbox,
        ));
    }
}
