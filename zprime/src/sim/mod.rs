pub mod camera;
pub mod collision;
pub mod components;
pub mod env;
pub mod movement;
pub mod resources;
pub mod sword;
pub mod systems;
pub mod worldgen;

use bevy::prelude::*;

use crate::config::ZPrimeConfig;

pub fn build_app(config: ZPrimeConfig) -> App {
    let mut app = App::new();
    worldgen::insert_world(&mut app, config);
    systems::add_simulation_systems(&mut app);
    app
}
