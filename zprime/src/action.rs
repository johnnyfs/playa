#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Action {
    #[default]
    Noop,
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    Attack,
}
