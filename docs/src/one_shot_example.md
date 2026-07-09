# One-Shot Example

In this example (as in the [Simple Example](simple_example.md)), we will check the band limit of a jinc function: \\(f(r) = \frac{J_1(r)}{r}\\).
The (0 order) Hankel transform of this should be the top hat function. We'll also do the reverse: transforming a Top-Hat function to get a Jinc function.

In this case, we will use the simple, single shot functions. It should be noted though, this simplicity comes at an increased overhead, and for multiple transforms on the same grid, the approach in the Simple Example is recommended.

Here is the source code:

```rust
{{#include ../../examples/one_shot_example.rs}}
```

**Original Function (Jinc):**

<!-- cmdrun cargo run --example one_shot_example -->
![Original Function](images/one_shot_example_f.png)

**Hankel Transform (Top Hat):**

![Hankel Transform](images/one_shot_example_ht.png)

**Original Function (Top Hat):**

![Top Hat Function](images/one_shot_example_tophat.png)

**Hankel Transform (Jinc):**

![Hankel Transform of Top Hat](images/one_shot_example_jinc.png)
