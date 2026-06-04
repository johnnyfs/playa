use std::collections::HashMap;

use crate::sim::components::{Direction, WorldRegion};

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticState {
    pub tick: u64,
    pub region: WorldRegion,
    pub player: SemanticEntity,
    pub entities: Vec<SemanticEntity>,
    pub terrain: SemanticTerrain,
    pub attrs: HashMap<String, SemanticValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticEntity {
    pub id: u64,
    pub kind: String,
    pub type_name: String,
    pub screen_x: i32,
    pub screen_y: i32,
    pub local_x: f32,
    pub local_y: f32,
    pub facing: Option<Direction>,
    pub hp: Option<i32>,
    pub alive: bool,
    pub attrs: HashMap<String, SemanticValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticTerrain {
    pub screen_x: i32,
    pub screen_y: i32,
    pub width_tiles: u32,
    pub height_tiles: u32,
    pub tiles: Vec<SemanticTile>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticTile {
    pub x: u32,
    pub y: u32,
    pub terrain_type: String,
    pub walkable: bool,
}
