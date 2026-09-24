# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-09-24

### Security & Dependencies
- Updated `amos-bessel-rs` to `1.0.1`, removing the unmaintained `paste` transitive dependency.
- Resolved security and soundness advisories in lockfile (`rustls` TLS 1.3 handshake fix `RUSTSEC-2026-0285`, `rand` soundness fix `RUSTSEC-2026-0097`).
- Added `deny.toml` and GitHub Actions `Security & License Audit` workflow (`security.yml`) running automated `cargo-deny` and `cargo-audit` scans on push, PR, and weekly schedule.

### Benchmark Maintenance
- Cleaned up `benches/hankel_benchmark.rs`: renamed 1D QDHT group to `qdht1d`, removed obsolete manual 3-step scaling baseline (`direct_1d`), and removed temporary scratch baseline in batch beam propagation suite.

## [1.2.0] - 2026-09-23

### Added
- Added pre-scaled QDHT matrix $M_{\text{qdht}}[i, j] = T_{ij} \frac{J_V[i]}{J_R[j]}$ computed at initialization, enabling direct dense matrix multiplication without per-transform vector scaling sweeps.
- Added `HankelTransform::qdht_matrix(&self) -> ArrayView2<'_, f64>` accessor returning a view of the pre-scaled forward QDHT matrix.
- Added `HankelTransform::iqdht_matrix(&self) -> Array2<f64>` accessor returning the pre-scaled inverse IQDHT matrix.
- Added `HankelTransform::iqdht_scale(&self) -> f64` accessor returning the scalar factor $(v_{\max} / r_{\max})^2$ relating IQDHT to QDHT.
- Added `std::ops::MulAssign<f64>` supertrait bound to `HankelScalar`.

### Performance & Optimization
- **Dense 2D GEMM Fast Path**: Added dedicated matrix-matrix multiplication fast path for 2D arrays along axis 0 and axis 1, eliminating the memory-bandwidth bottleneck of iterating 1D lanes and achieving an **11.5× speedup** over `1.1.0` (and **2.2× faster than NumPy** with BLAS).
- **1D GEMV Fast Path**: Direct sequential matrix-vector multiplication fast path for 1D arrays without Rayon thread dispatch overhead.
- **Complex GEMM Decomposition**: For complex arrays (`Complex<f64>`), real and imaginary parts are separated and multiplied concurrently via `rayon::join`. Because the transform matrix is purely real, this requires only two real GEMMs rather than four, halving total floating-point arithmetic compared to generic complex matrix multiplication.
- **Memory Optimization**: Used scalar conversion factor `iqdht_scale` for IQDHT instead of storing a redundant third matrix, saving 33.5 MB of RAM at $N=2048$.
- Added `beam_batch_2048x128` benchmark in `benches/hankel_benchmark.rs` to track 2D optical propagation batch performance.

### Deprecated
- `HankelScalar::div_real_array`: Deprecated in favor of the pre-scaled transform matrix.
- `HankelScalar::mul_real_array_assign`: Deprecated in favor of the pre-scaled transform matrix.

## [1.1.0] - 2026-08-31

### Added
- Added public `TransformType` enum (`Polar`, `Spherical`) representing the coordinate symmetry of the transform.
- Added `HankelTransform::transform_type(&self) -> TransformType` getter method for runtime transform type introspection.
- `TransformType` derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, and `Hash`.
- `HankelTransform`'s `Debug` format now includes the `transform_type` field.

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
