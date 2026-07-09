# Speed Comparison: Single-Shot vs Reusable Objects

For a simple case, there are convenience functions (`hankrs::one_shot::qdht` and `iqdht`) which can be used to calculate the transform of a function without needing to manually manage a state object.

However, these functions come with **heavy internal overhead** because they completely re-calculate all of the Bessel roots and transformation matrices on every single call. To demonstrate this overhead, we will use the exact same beam-propagation application as in our [Complex Beam Propagation](usage_example.md) example, and time the difference between generating a state machine once vs relying on `one_shot`.

Here is the source code:

```rust
{{#include ../../examples/speed_usage_example.rs}}
```

### The Results

If we compile and run this in `release` mode via `cargo run --release --example speed_usage_example`, the performance gap is monumental:

<!-- cmdrun cargo run --release --example speed_usage_example -->

The struct-based approach completes in a fraction of a second, yielding roughly a **~50x speedup** just by holding onto the `HankelTransform` struct. When performing multiple identical QDHT operations, it is significantly faster to maintain and reuse `HankelTransform` objects.

To verify they output exactly the same data, we can plot the resulting beam propagation for both approaches:

![Single Shot Output](images/speed_usage_single_shot.png)
![Object Output](images/speed_usage_object.png)
