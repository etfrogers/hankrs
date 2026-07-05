//! A library for performing quasi-discrete Hankel transforms.
//!
//! This crate provides tools for efficiently computing Hankel transforms
//! of various orders using a quasi-discrete approximation.

#![warn(missing_docs)]

mod hankel;

/// Convenience functions for performing one-off Hankel transforms.
pub mod one_shot;

/// The primary struct used for computing Hankel transforms.
pub use crate::hankel::HankelTransform;
