# Known Transforms — Spherical Hankel Transform

The *spherical* Hankel transform (SQDHT) is a variant of the QDHT designed for
functions with spherical (3-D radial) symmetry. It computes

$$ H_\text{sph}^{(n)}\{f\}(k) = \int_0^\infty f(r)\\; j_n(kr) \\; r^2 \\; dr $$

where \\(j_n(x)\\) is the nth-order spherical Bessel function of the first kind, defined as 

$$ j_n(x) = \sqrt{\frac{\pi}{2x}} \\; J_{n+\frac{1}{2}}(x) $$ 

Where \\(J_n(x)\\) is the Bessel function of the first kind. For n = 0, this reduces to \\(j_0(x) = \sin(x)/x\\) and hence

$$ H_\text{sph}^{(0)}\{f\}(k) = \int_0^\infty f(r) \\, \frac{\sin(kr)}{kr} \\, r^2 \\, dr $$

While the zeroth order transform is demonstrated below, the `HankelTransform` object can be used to compute 
transforms of any integer order.

A `HankelTransform` configured for spherical symmetry can be created with
`HankelTransform::new_spherical(order, max_radius, n_points)`.

Below we verify two well-known transform pairs against their analytical results.

```rust
{{#include ../../examples/spherical_known_transforms.rs}}
```

## Gaussian Function

The order-0 spherical Hankel transform of the Gaussian \\(f(r) = e^{-ar^2}\\) is

$$H_\text{sph}\{e^{-ar^2}\}(k) = \frac{\sqrt{\pi}}{4\\, a^{3/2}}\\, e^{-k^2 / 4a}$$

<!-- cmdrun cargo run --example spherical_known_transforms -->
![Gaussian function and its spherical Hankel transform](images/spherical_known_transforms_gaussian.png)

The SQDHT (circles) closely follows the analytical result (solid line) across the
entire wavenumber range.

## Top-Hat Function

The order-0 spherical Hankel transform of the top-hat \\(f(r) = 1\\) for \\(r < a\\), \\(0\\) otherwise, is

$$H_\text{sph}\{f\}(k) = \frac{\sin(ka) - ka\cos(ka)}{k^3}$$

This is the 3-D analogue of the familiar jinc function that arises in the
standard (2-D radially-symmetric) Hankel transform.

![Top-hat function and its spherical Hankel transform](images/spherical_known_transforms_tophat.png)

The oscillatory decay of the transform reflects the sharp edge of the top-hat,
again in excellent agreement with the analytical formula.
