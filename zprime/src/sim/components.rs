use std::fmt;

use bevy::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Up => f.write_str("up"),
            Self::Down => f.write_str("down"),
            Self::Left => f.write_str("left"),
            Self::Right => f.write_str("right"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverworldPosition {
    pub screen_x: i32,
    pub screen_y: i32,
    pub indoors: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnderworldPosition {
    pub dungeon_id: i32,
    pub room_x: i32,
    pub room_y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub enum WorldRegion {
    Overworld(OverworldPosition),
    Underworld(UnderworldPosition),
}

impl WorldRegion {
    pub fn overworld_screen(self) -> Option<(i32, i32)> {
        match self {
            Self::Overworld(pos) => Some((pos.screen_x, pos.screen_y)),
            Self::Underworld(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct LocalPosition {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Hitbox {
    pub offset_x: f32,
    pub offset_y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct Health {
    pub hp: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwordPhase {
    Idle,
    Swinging { ticks_remaining: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct SwordState {
    pub phase: SwordPhase,
}

impl Default for SwordState {
    fn default() -> Self {
        Self {
            phase: SwordPhase::Idle,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct Player;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct FacingDirection(pub Direction);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct StableId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct SwordHitbox {
    pub owner: Entity,
}
