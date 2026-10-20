//! Mandelbrot explorer library: everything except the window and CLI, which
//! live in `main.rs`.
//!
//! - [`app`]: platform-independent explorer state (navigation, reference
//!   orbit lifecycle) producing [`app::FrameParams`]
//! - [`progressive`]: schedules GPU work into budget-sized, reusable tiles
//! - [`policy`]: decides that work and its iteration limits
//! - [`render`]: the backend-agnostic [`render::Renderer`] interface and the
//!   Metal backend
//! - [`bookmarks`], [`flight`]: saved places and smooth zoom-and-pan flights
//!   between them
//! - [`harness`]: scripted scenarios for measuring and comparing the
//!   progressive renderer's behavior
//! - [`perturb`], [`bigfixed`]: deep-zoom math (reference orbits, series
//!   approximation, arbitrary-precision fixed point)
//!
//! Unit tests live next to the code they test; tests that combine components
//! are in `tests/`.

pub mod app;
pub mod bigfixed;
pub mod bookmarks;
pub mod flight;
pub mod harness;
pub mod perturb;
pub mod policy;
pub mod progressive;
pub mod render;
