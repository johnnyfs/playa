use bevy::camera::ScalingMode;
use bevy::prelude::*;

use crate::sim::camera::screen_camera_translation;
use crate::sim::components::{Hitbox, LocalPosition, Player, SwordHitbox, WorldRegion};
use crate::sim::resources::{TerrainType, WorldMap};

#[derive(Component)]
pub struct PlayerVisual;

#[derive(Component)]
pub struct SwordVisual;

#[derive(Component)]
pub struct MainCamera;

const CAMERA_SCROLL_SPEED: f32 = 900.0;

pub fn setup_rendering(mut commands: Commands, world_map: Res<WorldMap>) {
    let (camera_x, camera_y) = screen_camera_translation(
        &world_map,
        crate::sim::components::WorldRegion::Overworld(crate::sim::components::OverworldPosition {
            screen_x: 0,
            screen_y: 0,
            indoors: false,
        }),
    )
    .expect("overworld camera center");
    commands.spawn((
        MainCamera,
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: world_map.config.screen_width_pixels as f32,
                height: world_map.config.screen_height_pixels as f32,
            },
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(camera_x, camera_y, 999.0),
    ));

    let tile_size = world_map.config.tile_size_pixels as f32;
    let world_width_px =
        world_map.config.world_width as f32 * world_map.config.screen_width_pixels as f32;
    let world_height_px =
        world_map.config.world_height as f32 * world_map.config.screen_height_pixels as f32;
    let world_origin_x = -world_width_px / 2.0;
    let world_origin_y = world_height_px / 2.0;

    for screen in &world_map.screens {
        let origin_x = screen.screen_x as f32 * world_map.config.screen_width_pixels as f32;
        let origin_y = screen.screen_y as f32 * world_map.config.screen_height_pixels as f32;
        for y in 0..screen.height_tiles {
            for x in 0..screen.width_tiles {
                let terrain = screen.terrain_at(x, y);
                let color = match terrain {
                    TerrainType::Floor => Color::srgb(0.20, 0.45, 0.22),
                    TerrainType::Wall => Color::srgb(0.23, 0.20, 0.18),
                };
                commands.spawn((
                    Sprite::from_color(color, Vec2::splat(tile_size - 1.0)),
                    Transform::from_xyz(
                        world_origin_x + origin_x + x as f32 * tile_size + tile_size / 2.0,
                        world_origin_y - (origin_y + y as f32 * tile_size + tile_size / 2.0),
                        0.0,
                    ),
                ));
            }
        }
    }

    commands.spawn((
        PlayerVisual,
        Sprite::from_color(Color::srgb(0.1, 0.25, 0.85), Vec2::new(16.0, 16.0)),
        Transform::from_xyz(0.0, 0.0, 2.0),
    ));
    commands.spawn((
        SwordVisual,
        Sprite::from_color(Color::srgb(0.9, 0.85, 0.25), Vec2::new(8.0, 14.0)),
        Transform::from_xyz(0.0, 0.0, 3.0),
        Visibility::Hidden,
    ));
}

pub fn sync_rendering_system(
    time: Res<Time>,
    world_map: Res<WorldMap>,
    player: Query<(&WorldRegion, &LocalPosition, &Hitbox), With<Player>>,
    sword: Query<(&WorldRegion, &LocalPosition, &Hitbox), With<SwordHitbox>>,
    mut camera: Query<
        &mut Transform,
        (
            With<MainCamera>,
            Without<PlayerVisual>,
            Without<SwordVisual>,
        ),
    >,
    mut player_visual: Query<
        (&mut Transform, &mut Sprite),
        (
            With<PlayerVisual>,
            Without<MainCamera>,
            Without<SwordVisual>,
        ),
    >,
    mut sword_visual: Query<
        (&mut Transform, &mut Sprite, &mut Visibility),
        (
            With<SwordVisual>,
            Without<MainCamera>,
            Without<PlayerVisual>,
        ),
    >,
) {
    if let (Ok((region, position, hitbox)), Ok((mut transform, mut sprite))) =
        (player.single(), player_visual.single_mut())
    {
        set_rect_transform(
            &world_map,
            *region,
            *position,
            *hitbox,
            &mut transform,
            &mut sprite,
        );
    }

    if let (Ok((region, _, _)), Ok(mut camera_transform)) = (player.single(), camera.single_mut()) {
        if let Some((camera_x, camera_y)) = screen_camera_translation(&world_map, *region) {
            let current = camera_transform.translation.truncate();
            let target = Vec2::new(camera_x, camera_y);
            let delta = target - current;
            let max_step = CAMERA_SCROLL_SPEED * time.delta_secs();
            let next = if delta.length() <= max_step {
                target
            } else {
                current + delta.normalize() * max_step
            };
            camera_transform.translation.x = next.x;
            camera_transform.translation.y = next.y;
        }
    }

    let Ok((mut transform, mut sprite, mut visibility)) = sword_visual.single_mut() else {
        return;
    };

    if let Ok((region, position, hitbox)) = sword.single() {
        set_rect_transform(
            &world_map,
            *region,
            *position,
            *hitbox,
            &mut transform,
            &mut sprite,
        );
        *visibility = Visibility::Visible;
    } else {
        *visibility = Visibility::Hidden;
    }
}

fn set_rect_transform(
    world_map: &WorldMap,
    region: WorldRegion,
    position: LocalPosition,
    hitbox: Hitbox,
    transform: &mut Transform,
    sprite: &mut Sprite,
) {
    let Some((screen_x, screen_y)) = region.overworld_screen() else {
        return;
    };
    let global_x = screen_x as f32 * world_map.config.screen_width_pixels as f32
        + position.x
        + hitbox.offset_x;
    let global_y = screen_y as f32 * world_map.config.screen_height_pixels as f32
        + position.y
        + hitbox.offset_y;
    let world_width_px =
        world_map.config.world_width as f32 * world_map.config.screen_width_pixels as f32;
    let world_height_px =
        world_map.config.world_height as f32 * world_map.config.screen_height_pixels as f32;
    sprite.custom_size = Some(Vec2::new(hitbox.width, hitbox.height));
    transform.translation.x = global_x + hitbox.width / 2.0 - world_width_px / 2.0;
    transform.translation.y = world_height_px / 2.0 - (global_y + hitbox.height / 2.0);
}
