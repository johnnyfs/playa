use bevy::prelude::*;

use crate::action::Action;
use crate::sim::resources::CurrentAction;

pub fn keyboard_input_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut action: ResMut<CurrentAction>,
) {
    action.0 = if keyboard.just_pressed(KeyCode::Space) {
        Action::Attack
    } else if keyboard.pressed(KeyCode::ArrowUp) || keyboard.pressed(KeyCode::KeyW) {
        Action::MoveUp
    } else if keyboard.pressed(KeyCode::ArrowDown) || keyboard.pressed(KeyCode::KeyS) {
        Action::MoveDown
    } else if keyboard.pressed(KeyCode::ArrowLeft) || keyboard.pressed(KeyCode::KeyA) {
        Action::MoveLeft
    } else if keyboard.pressed(KeyCode::ArrowRight) || keyboard.pressed(KeyCode::KeyD) {
        Action::MoveRight
    } else {
        Action::Noop
    };
}
