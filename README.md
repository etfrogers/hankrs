hankrs - Quasi-Discrete Hankel Transforms for Rust
====================================================

##### Edward Rogers

[![Build Status](https://github.com/etfrogers/hankrs/actions/workflows/ci.yml/badge.svg)](https://github.com/etfrogers/hankrs/actions)
[![Coverage](https://codecov.io/gh/etfrogers/hankrs/branch/main/graph/badge.svg)](https://codecov.io/gh/etfrogers/hankrs)
[![Crates.io](https://img.shields.io/crates/v/hankrs.svg)](https://crates.io/crates/hankrs)
[![Docs.rs](https://docs.rs/hankrs/badge.svg)](https://docs.rs/hankrs)

`hankrs` is a Rust implementation of the quasi-discrete Hankel transform as developed by Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega:

> *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"*
  Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
  J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

It was designed for use primarily in cases where a discrete Hankel transform is required, similar to the FFT for a Fourier transform. It operates on functions stored in `ndarray` arrays.

I have used this code extensively (originally in Python) for beam-propagation-method calculations of radially-symmetric beams. In the radially symmetric case, the 2D FFT over x and y that would be used in a non-symmetric system is replaced by a 1D QDHT over r, making the computational load much lighter and allowing bigger simulations.

`hankrs` is a Rust port of my Python library, [PyHank](https://github.com/etfrogers/pyhank) (which itself was inspired by Adam Wyatt's [Matlab version](https://uk.mathworks.com/matlabcentral/fileexchange/15623-hankel-transform)). It aims to simplify the interface and utilizes the `ndarray` crate for array operations.

It provides both a simple single-shot interface (`one_shot` module), and a more advanced approach (via the `HankelTransform` struct) that speeds up computation significantly if making multiple transforms on the same grid.

Contributions and comments are welcome using Github at:
http://github.com/etfrogers/hankrs

Installation
------------

You can add `hankrs` to your Rust project via `cargo`:

```bash
cargo add hankrs
```

For development and running the tests, just use standard Cargo commands:

```bash
cargo build
cargo test
```

**Note for Linux/Windows Developers:** When compiling tests or benchmarks, this crate links against the system's OpenBLAS library for acceleration. You must install the OpenBLAS development headers (e.g., `sudo apt-get install libopenblas-dev` on Ubuntu) prior to running `cargo test`, otherwise the build will fail to link.

Bugs & Contribution
-------------------

Please use Github to report bugs, feature requests and submit your code:
http://github.com/etfrogers/hankrs

Documentation
-------------

The API documentation for `hankrs` can be found at [docs.rs](https://docs.rs/hankrs), or you can build it locally with:

```bash
cargo doc --open
```

Comprehensive examples and mathematical background are provided in the [online `hankrs` book](https://etfrogers.github.io/hankrs/).

Usage
-----

Please refer to the [online examples](https://etfrogers.github.io/hankrs/) for extensive walkthroughs and mathematical comparisons. You can also view the tests (e.g. `tests/test_transformer.rs` and `tests/test_one_shot.rs`) for usage examples.
