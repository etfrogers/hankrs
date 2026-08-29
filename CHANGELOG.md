# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-30

### Added
- First stable `v1.0.0` release.
- Added `qdht_spherical` and `iqdht_spherical` convenience functions in `one_shot` module.
- New `HankelError` enum (`EmptyGrid`, `InvalidRadius`, `Interpolation`, `InvalidOrder`) replacing `InterpError`.
- `HankelTransform` now derives `Clone` and `PartialEq`.
- `HankelError` now derives `PartialEq` and `Eq`.
- Dimension mismatch assertions and `# Panics` doc sections for `qdht` and `iqdht`.
- Packaging verification steps in CI (`cargo package --verbose` with and without `blas`).
- Dedicated error and panic unit tests in `tests/test_errors.rs`.

### Performance & Optimization
- Optimized performance using parallelization and switching to faster Bessel calculation and root computation.
- Made BLAS hardware acceleration an **optional feature** (`[features] blas = ["ndarray/blas"]`), allowing pure-Rust builds without external C/Fortran system dependencies while retaining optional BLAS speedups.
- Added multi-dimensional Criterion benchmarks (`2D` and `3D` transforms) in `benches/hankel_benchmark.rs`.
- Added commit benchmarking script (`bench_commits.sh`) for performance tracking.
- Cached $k$-space transformers (`TRANSFORMERS_K` `LazyLock`) in the test suite to eliminate redundant Bessel root and matrix computations.

### Changed
- **Breaking**: `HankelTransform` constructors (`new`, `new_from_r_grid`, `new_from_k_grid`, `new_spherical`, `new_spherical_from_r_grid`,
`new_spherical_from_k_grid`) now return `Result<HankelTransform, HankelError>` instead of panicking on invalid or empty grids.
- **Breaking**: `one_shot::qdht` and `one_shot::iqdht` now return `Result<(Array1<f64>, Array<T, D>), HankelError>`.
- Replaced monolithic `num` dependency with `num-traits` and removed unused runtime dependencies (`conv`, `csv`, `rand`, `rstest_reuse`).
- Corrected method docstrings for `max_kr`, `max_frequency`, `to_transform_k`, and `to_original_k`.

## [0.2.1] - 2026-07-21

### Fixed
- Fixed an approximation error in the spherical transform.

## [0.2.0] - 2026-07-20

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
