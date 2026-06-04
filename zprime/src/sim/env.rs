use std::collections::HashMap;

use bevy::prelude::*;

use crate::action::Action;
use crate::config::ZPrimeConfig;
use crate::semantic::export::semantic_state;
use crate::semantic::state::{SemanticState, SemanticValue};
use crate::sim::build_app;
use crate::sim::resources::CurrentAction;

pub struct ZPrimeEnv {
    app: App,
    config: ZPrimeConfig,
}

impl ZPrimeEnv {
    pub fn new(config: ZPrimeConfig) -> Self {
        Self {
            app: build_app(config.clone()),
            config,
        }
    }

    pub fn reset(&mut self, seed: u64) -> SemanticState {
        self.app = build_app(self.config.clone());
        self.app.world_mut().insert_resource(ResetSeed(seed));
        self.state()
    }

    pub fn step(&mut self, action: Action) -> StepResult {
        self.app.world_mut().resource_mut::<CurrentAction>().0 = action;
        self.app.update();
        self.app.world_mut().resource_mut::<CurrentAction>().0 = Action::Noop;

        StepResult {
            state: self.state(),
            reward: 0.0,
            done: false,
            info: StepInfo::default(),
        }
    }

    pub fn state(&mut self) -> SemanticState {
        semantic_state(self.app.world_mut())
    }
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub state: SemanticState,
    pub reward: f32,
    pub done: bool,
    pub info: StepInfo,
}

#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    pub attrs: HashMap<String, SemanticValue>,
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct ResetSeed(pub u64);
