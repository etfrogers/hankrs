# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- Support for Spherical Hankel transforms (`new_spherical`, etc.).
- Documentation and examples for spherical transforms.
- Criterion benchmarks for transform matrix creation and QDHT operations.

### Changed
- Shifted API from `&Array` to `ArrayView` to offer more flexible and ergonomic array handling.
- Overhauled error handling to use structured `Error` types rather than strings (using `thiserror`).
- Removed static dimension constraints to improve compatibility with Python bindings.
- Migrated to a faster version of `bessel-zeros` and integrated `real-bessel` where applicable to boost speed.

### Performance
- Integrated `blas-src` to drastically speed up matrix multiplications.
- Parallelized transform operations (QDHT and IQDHT) using `rayon`.
- Parallelized the generation of the transform matrix.
- Optimized transform matrix creation by exploiting its symmetry.

### Fixed
- Resolved various `clippy` warnings across the codebase and tests.
- CI pipeline now includes `cargo test --release` to verify optimized builds.

## [0.1.0] - 2024-03-XX
- Initial release featuring core Quasi-Discrete Hankel Transforms (QDHT).
