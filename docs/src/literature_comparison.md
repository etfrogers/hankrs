# Literature Comparison

This example is a comparison of `hankrs` against results from the original publication of the quasi-discrete Hankel transform method:

> *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"*
> Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
> J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

```rust
{{#include ../../examples/literature_comparison.rs}}
```

## Figure 1: Hankel Transform of a Sinc Function

We reproduce figure 1 of Guizar-Sicairos & Guitierrez-Vega, showing the Hankel transform of a sinc function and the dynamical error, for orders 1 and 4.


### Order 1
![HT Order 1](images/lit_comp_sinc_ht_1.png)
![Error Order 1](images/lit_comp_sinc_error_1.png)

### Order 4
![HT Order 4](images/lit_comp_sinc_ht_4.png)
![Error Order 4](images/lit_comp_sinc_error_4.png)

## Figure 3: Round-Trip Transformation

We reproduce figure 3, applying a forward transform and an inverse transform (round-trip), demonstrating high fidelity.

![HT of Top-Hat](images/lit_comp_fig3_ht.png)
![Round Trip](images/lit_comp_fig3_retrieved.png)

## Table 1: Error Calculations
The calculated errors for the Hankel transform and the reconstructed function match the values given in Table 1 of the Guizar-Sicairos & Guitierrez-Vega paper.

<!-- cmdrun cargo run --example literature_comparison -->
