//! Mandelbrot explorer library: everything except the window and CLI, which
//! live in `main.rs`.
//!
//! - [`app`]: platform-independent explorer state (navigation, iteration
//!   policy, reference orbit lifecycle) producing [`app::FrameParams`]
//! - [`progressive`]: schedules GPU work into budget-sized, reusable tiles
//! - [`render`]: the backend-agnostic [`render::Renderer`] interface and the
//!   Metal backend
//! - [`bookmarks`], [`flight`]: saved places and smooth zoom-and-pan flights
//!   between them
//! - [`perturb`], [`bigfixed`]: deep-zoom math (reference orbits, series
//!   approximation, arbitrary-precision fixed point)
//!
//! Unit tests live next to the code they test; tests that combine components
//! are in `tests/`.

pub mod app;
pub mod bigfixed;
pub mod bookmarks;
pub mod flight;
pub mod perturb;
pub mod progressive;
pub mod render;
