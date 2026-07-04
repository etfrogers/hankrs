mod helper;
use hankrs::one_shot::qdht;
use ndarray::{Array1, Axis};
use amos_bessel_rs::bessel_j;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- PART 1: Jinc to Top-Hat ---
    // Create a grid for r points and calculate the jinc function.
    let r = Array1::linspace(0.0, 100.0, 1024);
    
    // The calculation fails at r = 0, so we have to set that manually to the limit of 1/2.
    let f = r.mapv(|rad| {
        if rad == 0.0 {
            0.5
        } else {
            bessel_j(1.0, rad).unwrap() / rad
        }
    });

    helper::plot_1d(
        "one_shot_example_f.png",
        &r,
        &f,
        "Jinc Function",
        "Radius /m",
        "Amplitude",
        0.0..100.0,
        -0.2..0.6,
    )?;

    // Now take the Hankel transform using `qdht`:
    let (kr, ht) = qdht(r.clone(), &f, 0, Axis(0));

    // As expected, this is a top-hat function bandlimited to k<1, except for numerical error.
    helper::plot_1d(
        "one_shot_example_ht.png",
        &kr,
        &ht,
        "Hankel Transform (Top Hat)",
        "Radial wavevector /m^-1",
        "Amplitude",
        0.0..5.0,
        -0.2..1.2,
    )?;

    // --- PART 2: Top-Hat to Jinc ---
    // Create a simple top-hat function
    let r2 = Array1::linspace(0.0, 5.0, 200);
    let f2 = r2.mapv(|rad| if rad < 1.0 { 1.0 } else { 0.0 });
    
    helper::plot_1d(
        "one_shot_example_tophat.png",
        &r2,
        &f2,
        "Top Hat Function",
        "Radius /m",
        "Amplitude",
        0.0..5.0,
        -0.2..1.5,
    )?;

    // Transform
    let (kr2, ht2) = qdht(r2.clone(), &f2, 0, Axis(0));

    helper::plot_1d(
        "one_shot_example_jinc.png",
        &kr2,
        &ht2,
        "QDHT of a Top Hat Function",
        "k_r",
        "Amplitude",
        0.0..30.0,
        -0.5..1.5,
    )?;

    Ok(())
}
