use bevy::prelude::*;

use crate::action::Action;
use crate::config::ZPrimeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainType {
    Floor,
    Wall,
}

impl TerrainType {
    pub fn is_walkable(self) -> bool {
        matches!(self, Self::Floor)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Wall => "wall",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Screen {
    pub screen_x: i32,
    pub screen_y: i32,
    pub width_tiles: u32,
    pub height_tiles: u32,
    pub terrain: Vec<TerrainType>,
}

impl Screen {
    pub fn terrain_at(&self, x: u32, y: u32) -> TerrainType {
        let index = (y * self.width_tiles + x) as usize;
        self.terrain[index]
    }

    pub fn set_terrain(&mut self, x: u32, y: u32, terrain: TerrainType) {
        let index = (y * self.width_tiles + x) as usize;
        self.terrain[index] = terrain;
    }
}

#[derive(Debug, Clone, Resource)]
pub struct WorldMap {
    pub config: ZPrimeConfig,
    pub screens: Vec<Screen>,
}

impl WorldMap {
    pub fn screen(&self, screen_x: i32, screen_y: i32) -> Option<&Screen> {
        self.screens
            .iter()
            .find(|screen| screen.screen_x == screen_x && screen.screen_y == screen_y)
    }

    pub fn in_overworld_bounds(&self, screen_x: i32, screen_y: i32) -> bool {
        screen_x >= 0
            && screen_y >= 0
            && screen_x < self.config.world_width
            && screen_y < self.config.world_height
    }
}

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct CurrentAction(pub Action);

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct SimulationTick(pub u64);

#[derive(Debug, Clone, Copy, Resource)]
pub struct NextStableId(pub u64);
