# Simple Example

In this example (as in the [One-Shot Example](one_shot_example.md)) we will check the band limit of a jinc function: \\(f(r) = \frac{J_1(r)}{r}\\).
The (0 order) Hankel transform of this should be the top hat function.

Here we create a `HankelTransform` object and use its `qdht` method.
In this simple case, the simpler, single shot functions used in the One-Shot Example may be simpler to use. It should be noted, however, that they are not well suited for multiple transforms on the same grid and the approach taken here is recommended.

Here is the source code:

```rust
{{#include ../../examples/simple_example.rs}}
```

**Original Function:**

<!-- cmdrun cargo run --example simple_example -->
![Original Function](images/simple_example_f.png)

**Hankel Transform (QDHT):**

![Hankel Transform](images/simple_example_ht.png)
