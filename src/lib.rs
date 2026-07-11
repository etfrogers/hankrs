//! A library for performing quasi-discrete Hankel transforms.
//!
//! This crate provides tools for efficiently computing Hankel transforms
//! of various orders using a quasi-discrete approximation. It operates seamlessly
//! on `ndarray` arrays and natively supports both real (`f64`) and complex
//! (`num_complex::Complex64`) numbers via the [`HankelScalar`] trait.
//!
//! # Examples and Guides
//! Comprehensive examples and theoretical demonstrations can be found in the
//! [online `hankrs` book](https://etfrogers.github.io/hankrs/).
//!
//! ## Quick Start
//! ```rust
//! use hankrs::HankelTransform;
//! use ndarray::{Array1, Axis};
//!
//! // 1. Create a transformer for order 0 up to radius 10.0 with 256 points
//! let transformer = HankelTransform::new(0, 10.0, 256);
//!
//! // 2. Define a function on the generated radial grid `transformer.radius()`
//! let r = transformer.radius();
//! let f = r.mapv(|rad| (-rad * rad).exp()); // Gaussian function
//!
//! // 3. Compute the quasi-discrete Hankel transform (QDHT)
//! let ht = transformer.qdht(&f, Axis(0));
//!
//! // The transformed values are evaluated at `transformer.kr()` (or `transformer.frequency()`)
//! ```

#![warn(missing_docs)]

mod hankel;

/// Convenience functions for performing one-off Hankel transforms.
pub mod one_shot;

/// The primary struct used for computing Hankel transforms.
pub use crate::hankel::{HankelScalar, HankelTransform};
