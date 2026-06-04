# Z'

Minimal Rust/Bevy simulation substrate for Zelda-like RL/HRL experiments.

## Commands

```bash
cargo test
cargo run --no-default-features --features headless
cargo run --features visual
```

## Phase 0 Behavior

- Headless mode steps one fixed simulation tick per `ZPrimeEnv::step(Action)`.
- The default world is a 2x2 overworld of 256x176 pixel screens with 16 pixel tiles.
- Movement is continuous pixel movement, cardinal-only, and blocked by walls or the outer world boundary.
- Shared screen edges transition between adjacent overworld screens.
- `Action::Attack` starts a sword swing if the sword is idle. The swing lasts `sword_swing_ticks`; because there is one action per tick, the attack tick itself is not also a movement tick, but later movement actions are allowed while the sword remains active.
- `ZPrimeEnv::state()` exports semantic state independent of rendering, including player state, current terrain, sword activity, and sword hitbox entities.
