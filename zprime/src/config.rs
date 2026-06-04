#[derive(Debug, Clone)]
pub struct ZPrimeConfig {
    pub world_width: i32,
    pub world_height: i32,
    pub screen_width_pixels: u32,
    pub screen_height_pixels: u32,
    pub tile_size_pixels: u32,
    pub fixed_timestep_hz: u32,
    pub player_speed_pixels_per_tick: f32,
    pub sword_swing_ticks: u32,
}

impl Default for ZPrimeConfig {
    fn default() -> Self {
        Self {
            world_width: 2,
            world_height: 2,
            screen_width_pixels: 256,
            screen_height_pixels: 192,
            tile_size_pixels: 16,
            fixed_timestep_hz: 60,
            player_speed_pixels_per_tick: 4.0,
            sword_swing_ticks: 8,
        }
    }
}
