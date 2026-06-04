#[cfg(feature = "visual")]
fn main() {
    zprime::visual::run();
}

#[cfg(not(feature = "visual"))]
fn main() {
    let mut env = zprime::ZPrimeEnv::new(zprime::ZPrimeConfig::default());
    let state = env.reset(0);
    println!(
        "Z' headless smoke: tick={} screen=({}, {}) player=({:.1}, {:.1})",
        state.tick,
        state.player.screen_x,
        state.player.screen_y,
        state.player.local_x,
        state.player.local_y
    );
}
