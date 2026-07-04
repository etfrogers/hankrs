# Known Transforms

Below we demonstrate a range of known Hankel transform pairs from various sources.

First we demonstrate the Gaussian function from Pissens [1] and its inverse transform.

Then we check the "generalised top-hat" and "generalised jinc" functions from Guizar-Sicairos and Guitierrez-Vega [2].

Finally, we look at the function \\(f(r) = \frac{1}{r^2 + a^2}\\), the Hankel transform of which is \\(K_0(av)\\), where \\(K_0\\) is the modified Bessel function of the second kind of order 0. [1]

> [1] *“Chapter 9: The Hankel Transform.”* Piessens, R. in The Transforms and Applications Handbook: Second Edition. Ed. Alexander D. Poularikas. Boca Raton: CRC Press LLC, 2000
> [2] *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"* Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega. J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

```rust
{{#include ../../examples/known_transforms.rs}}
```

## Gaussian Function
<!-- cmdrun cargo run --example known_transforms -->
![Gaussian Function HT](images/known_transforms_gaussian.png)

![Inverse Gaussian Function HT](images/known_transforms_inv_gaussian.png)

## Generalised Jinc to Top-Hat
![Top-Hat Order 0](images/known_transforms_tophat_0.png)
![Top-Hat Order 1](images/known_transforms_tophat_1.png)
![Top-Hat Order 4](images/known_transforms_tophat_4.png)

## Generalised Top-Hat to Jinc
![Jinc Order 0](images/known_transforms_jinc_0.png)
![Jinc Order 1](images/known_transforms_jinc_1.png)
![Jinc Order 4](images/known_transforms_jinc_4.png)

## K0 Function
![K0 HT](images/known_transforms_k0.png)
