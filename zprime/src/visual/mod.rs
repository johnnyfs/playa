#[cfg(feature = "visual")]
pub mod input;
#[cfg(feature = "visual")]
pub mod rendering;
#[cfg(feature = "visual")]
pub mod screen;

#[cfg(feature = "visual")]
pub fn run() {
    use bevy::prelude::*;
    use bevy::window::{Window, WindowPlugin, WindowResizeConstraints, WindowResolution};

    use crate::config::ZPrimeConfig;
    use crate::sim::{systems, worldgen};
    use crate::visual::screen::screen_viewport;

    let config = ZPrimeConfig::default();
    let viewport = screen_viewport(&config);
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Z' visual debug".to_string(),
                resolution: WindowResolution::new(viewport.width_pixels, viewport.height_pixels)
                    .with_scale_factor_override(1.0),
                resize_constraints: WindowResizeConstraints {
                    min_width: viewport.width_pixels as f32,
                    min_height: viewport.height_pixels as f32,
                    max_width: viewport.width_pixels as f32,
                    max_height: viewport.height_pixels as f32,
                },
                resizable: false,
                ..default()
            }),
            ..default()
        }),
    );
    worldgen::insert_world(&mut app, config);
    app.add_systems(Startup, rendering::setup_rendering);
    app.add_systems(Update, input::keyboard_input_system);
    systems::add_simulation_systems(&mut app);
    app.add_systems(Update, rendering::sync_rendering_system);
    app.run();
}
