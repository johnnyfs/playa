use bevy::prelude::*;

use crate::sim::movement::movement_system;
use crate::sim::resources::SimulationTick;
use crate::sim::sword::{refresh_sword_hitbox_system, sword_state_system};

pub fn add_simulation_systems(app: &mut App) {
    app.add_systems(
        Update,
        (
            movement_system,
            sword_state_system,
            refresh_sword_hitbox_system,
            advance_tick_system,
        )
            .chain(),
    );
}

fn advance_tick_system(mut tick: ResMut<SimulationTick>) {
    tick.0 += 1;
}
