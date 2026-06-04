use zprime::sim::camera::screen_camera_translation_for_config;
#[cfg(feature = "visual")]
use zprime::visual::screen::screen_viewport;
use zprime::{Action, Direction, SemanticValue, WorldRegion, ZPrimeConfig, ZPrimeEnv};

fn default_env() -> ZPrimeEnv {
    ZPrimeEnv::new(ZPrimeConfig::default())
}

fn sword_active(env: &mut ZPrimeEnv) -> bool {
    matches!(
        env.state().player.attrs.get("sword_active"),
        Some(SemanticValue::Bool(true))
    )
}

#[test]
fn reset_creates_player_on_screen_zero_zero() {
    let mut env = default_env();
    let state = env.reset(123);

    assert_eq!(state.tick, 0);
    assert_eq!(state.player.screen_x, 0);
    assert_eq!(state.player.screen_y, 0);
    assert_eq!(state.player.facing, Some(Direction::Down));
    assert!(matches!(
        state.region,
        WorldRegion::Overworld(position)
            if position.screen_x == 0 && position.screen_y == 0 && !position.indoors
    ));
}

#[test]
fn player_moves_right_when_stepping_move_right() {
    let mut env = default_env();
    let before = env.reset(0);
    let after = env.step(Action::MoveRight).state;

    assert!(after.player.local_x > before.player.local_x);
    assert_eq!(after.player.local_y, before.player.local_y);
    assert_eq!(after.player.facing, Some(Direction::Right));
}

#[test]
fn player_cannot_move_beyond_outer_world_boundary() {
    let mut env = default_env();
    env.reset(0);

    for _ in 0..32 {
        env.step(Action::MoveLeft);
    }

    let state = env.state();
    assert_eq!(state.player.screen_x, 0);
    assert!(state.player.local_x >= -4.0);
}

#[test]
fn player_transitions_from_screen_zero_zero_to_one_zero_crossing_right_edge() {
    let mut env = default_env();
    env.reset(0);

    let mut transitioned = false;
    for _ in 0..80 {
        let state = env.step(Action::MoveRight).state;
        if state.player.screen_x == 1 {
            assert_eq!(state.player.screen_y, 0);
            assert!(state.player.local_x <= 0.0);
            transitioned = true;
            break;
        }
    }

    assert!(transitioned, "player did not transition to screen (1, 0)");
}

#[test]
fn overworld_perimeter_is_walled_and_internal_edges_are_open() {
    let mut env = default_env();
    let state = env.reset(0);

    assert_tile_walkable(&state.terrain, 0, 5, false);
    assert_tile_walkable(&state.terrain, 5, 0, false);
    assert_tile_walkable(&state.terrain, 15, 5, true);
    assert_tile_walkable(&state.terrain, 5, 11, true);

    for _ in 0..80 {
        let state = env.step(Action::MoveRight).state;
        if state.player.screen_x == 1 {
            break;
        }
    }
    let east_state = env.state();
    assert_eq!(east_state.terrain.screen_x, 1);
    assert_eq!(east_state.terrain.screen_y, 0);
    assert_tile_walkable(&east_state.terrain, 0, 5, true);
    assert_tile_walkable(&east_state.terrain, 15, 5, false);
    assert_tile_walkable(&east_state.terrain, 5, 0, false);
    assert_tile_walkable(&east_state.terrain, 5, 11, true);

    let mut env = default_env();
    env.reset(0);
    for _ in 0..40 {
        let state = env.step(Action::MoveDown).state;
        if state.player.screen_y == 1 {
            break;
        }
    }
    let south_state = env.state();
    assert_eq!(south_state.terrain.screen_x, 0);
    assert_eq!(south_state.terrain.screen_y, 1);
    assert_tile_walkable(&south_state.terrain, 0, 5, false);
    assert_tile_walkable(&south_state.terrain, 5, 0, true);
    assert_tile_walkable(&south_state.terrain, 5, 11, false);
}

#[test]
fn each_screen_has_distinct_interior_obstructions() {
    let mut env = default_env();
    let state = env.reset(0);
    assert_tile_walkable(&state.terrain, 5, 4, false);

    move_until_screen(&mut env, 1, 0, Action::MoveRight, 80);
    let east_state = env.state();
    assert_tile_walkable(&east_state.terrain, 4, 3, false);
    assert_tile_walkable(&east_state.terrain, 5, 4, true);

    move_until_screen(&mut env, 1, 1, Action::MoveDown, 40);
    let southeast_state = env.state();
    assert_tile_walkable(&southeast_state.terrain, 7, 5, false);
    assert_tile_walkable(&southeast_state.terrain, 4, 3, true);

    let mut env = default_env();
    env.reset(0);
    move_until_screen(&mut env, 0, 1, Action::MoveDown, 40);
    let south_state = env.state();
    assert_tile_walkable(&south_state.terrain, 8, 2, false);
    assert_tile_walkable(&south_state.terrain, 7, 5, true);
}

#[test]
fn edge_crossing_is_core_simulation_state_not_visual_only() {
    let config = ZPrimeConfig {
        screen_width_pixels: 64,
        screen_height_pixels: 64,
        tile_size_pixels: 16,
        ..Default::default()
    };

    let mut env = ZPrimeEnv::new(config);
    env.reset(0);

    for _ in 0..6 {
        env.step(Action::MoveRight);
    }
    let right_state = env.state();
    assert_eq!(right_state.player.screen_x, 1);
    assert_eq!(right_state.player.screen_y, 0);
    assert_eq!(right_state.player.local_x, -4.0);
    assert!(matches!(
        right_state.region,
        WorldRegion::Overworld(position)
            if position.screen_x == 1 && position.screen_y == 0 && !position.indoors
    ));

    env.step(Action::MoveLeft);
    let left_state = env.state();
    assert_eq!(left_state.player.screen_x, 0);
    assert_eq!(left_state.player.screen_y, 0);
    assert_eq!(left_state.player.local_x, 52.0);

    let mut env = ZPrimeEnv::new(ZPrimeConfig {
        screen_width_pixels: 64,
        screen_height_pixels: 64,
        tile_size_pixels: 16,
        ..Default::default()
    });
    env.reset(0);

    for _ in 0..5 {
        env.step(Action::MoveDown);
    }
    let down_state = env.state();
    assert_eq!(down_state.player.screen_x, 0);
    assert_eq!(down_state.player.screen_y, 1);
    assert_eq!(down_state.player.local_y, -8.0);

    env.step(Action::MoveUp);
    let up_state = env.state();
    assert_eq!(up_state.player.screen_x, 0);
    assert_eq!(up_state.player.screen_y, 0);
    assert_eq!(up_state.player.local_y, 48.0);
}

#[test]
fn player_cannot_transition_beyond_screen_one_zero_to_the_right() {
    let mut env = default_env();
    env.reset(0);

    for _ in 0..160 {
        env.step(Action::MoveRight);
    }

    let state = env.state();
    assert_eq!(state.player.screen_x, 1);
    assert_eq!(state.player.screen_y, 0);
    assert!(state.player.local_x <= 244.0);
}

#[test]
fn wall_tiles_block_player_movement() {
    let mut env = default_env();
    env.reset(0);

    for _ in 0..6 {
        env.step(Action::MoveDown);
    }
    let before_wall = env.state();
    assert_eq!(before_wall.player.local_y, 56.0);

    for _ in 0..16 {
        env.step(Action::MoveRight);
    }

    let state = env.state();
    assert_eq!(state.player.screen_x, 0);
    assert_eq!(state.player.screen_y, 0);
    assert_eq!(state.player.local_x, 68.0);
}

#[test]
fn attack_creates_active_sword_state() {
    let mut env = default_env();
    env.reset(0);

    let state = env.step(Action::Attack).state;

    assert!(sword_active(&mut env));
    assert!(matches!(
        state.player.attrs.get("sword_active"),
        Some(SemanticValue::Bool(true))
    ));
    assert_eq!(state.entities.len(), 1);
    assert_eq!(state.entities[0].kind, "sword");
}

#[test]
fn sword_swing_expires_after_configured_ticks() {
    let mut config = ZPrimeConfig::default();
    config.sword_swing_ticks = 3;
    let mut env = ZPrimeEnv::new(config);
    env.reset(0);

    env.step(Action::Attack);
    assert!(sword_active(&mut env));

    env.step(Action::Noop);
    assert!(sword_active(&mut env));

    env.step(Action::Noop);
    assert!(sword_active(&mut env));

    env.step(Action::Noop);
    let state = env.state();
    assert!(!sword_active(&mut env));
    assert!(state.entities.is_empty());
}

#[test]
fn semantic_state_reports_current_screen_and_player_local_position() {
    let mut env = default_env();
    env.reset(0);
    env.step(Action::MoveRight);
    let state = env.state();

    assert_eq!(state.terrain.screen_x, state.player.screen_x);
    assert_eq!(state.terrain.screen_y, state.player.screen_y);
    assert_eq!(state.player.local_x, 36.0);
    assert_eq!(state.player.local_y, 32.0);
    assert_eq!(state.terrain.width_tiles, 16);
    assert_eq!(state.terrain.height_tiles, 12);
}

#[test]
fn headless_stepping_works_without_visual_feature() {
    let mut env = default_env();
    env.reset(0);

    for action in [Action::Noop, Action::MoveDown, Action::Attack, Action::Noop] {
        let result = env.step(action);
        assert!(!result.done);
        assert_eq!(result.reward, 0.0);
    }
}

#[test]
fn camera_target_changes_when_screen_changes() {
    let config = ZPrimeConfig::default();
    let origin = screen_camera_translation_for_config(&config, 0, 0);
    let east = screen_camera_translation_for_config(&config, 1, 0);

    assert_ne!(origin, east);
    assert!(east.0 > origin.0);
    assert_eq!(origin.1, east.1);
}

#[cfg(feature = "visual")]
#[test]
fn visual_viewport_matches_exact_screen_tile_span() {
    let config = ZPrimeConfig {
        screen_width_pixels: 16 * 16,
        screen_height_pixels: 12 * 16,
        tile_size_pixels: 16,
        ..Default::default()
    };
    let viewport = screen_viewport(&config);

    assert_eq!(viewport.width_pixels, 256);
    assert_eq!(viewport.height_pixels, 192);
    assert_eq!(viewport.width_tiles, 16);
    assert_eq!(viewport.height_tiles, 12);
}

fn assert_tile_walkable(
    terrain: &zprime::SemanticTerrain,
    x: u32,
    y: u32,
    expected_walkable: bool,
) {
    let tile = terrain
        .tiles
        .iter()
        .find(|tile| tile.x == x && tile.y == y)
        .expect("semantic tile exists");
    assert_eq!(
        tile.walkable, expected_walkable,
        "tile ({x}, {y}) on screen ({}, {})",
        terrain.screen_x, terrain.screen_y
    );
}

fn move_until_screen(
    env: &mut ZPrimeEnv,
    screen_x: i32,
    screen_y: i32,
    action: Action,
    max_steps: usize,
) {
    for _ in 0..max_steps {
        let state = env.step(action).state;
        if state.player.screen_x == screen_x && state.player.screen_y == screen_y {
            return;
        }
    }
    panic!("player did not reach screen ({screen_x}, {screen_y})");
}
