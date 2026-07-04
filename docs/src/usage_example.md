# Complex Beam Propagation

To demonstrate the full power of `HankelTransform` with complex values, we will give an example of propagating a radially-symmetric beam using the beam propagation method.

In this case, it will be a simple Gaussian beam propagating away from focus and diverging. We'll utilize the `<T: HankelScalar>` generic trait to perform the QDHT directly on a `Complex64` array.

First we will use a loop over the \\(z\\) position, and then we will demonstrate that the `HankelTransform` methods can perfectly vectorize the \\(z\\)-axis propagation by processing multiple columns natively.

Here is the source code:

```rust
{{#include ../../examples/usage_example.rs}}
```

**Initial Electric Field Distribution:**

<!-- cmdrun cargo run --example usage_example -->
![Initial Field](images/usage_example_initial_field.png)

![K Vector](images/usage_example_k_vector.png)

**Radial Field Intensity as a Function of Propagation:**
![Field Intensity](images/usage_example_irz.png)

Because the intensity drops as the beam expands, it might be difficult to clearly see the beam growing in \\(r\\). To show that better, let's plot the intensity normalised such that the peak intensity at each \\(z\\) coordinate is identical.

**Normalised Radial Field Intensity:**

![Normalised Field Intensity](images/usage_example_irz_norm.png)

**Vectorised Approach Match:**

We can also verify that propagating it using the vectorised array matches the loop exactly.

![Vectorised Intensity](images/usage_example_irz_vec.png)
