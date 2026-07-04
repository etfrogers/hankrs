mod helper;
use hankrs::HankelTransform;
use amos_bessel_rs::bessel_j;
use ndarray::Axis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a HankelTransform object which holds the grid for r and kr points.
    let transformer = HankelTransform::new(0, 100.0, 1024);
    let radius = transformer.radius();
    
    // Note that although the calculation fails at r = 0, transformer.radius() does not include r=0.
    let f = radius.mapv(|rad| bessel_j(1.0, rad).unwrap() / rad);

    helper::plot_1d(
        "simple_example_f.png",
        radius,
        &f,
        "Jinc Function",
        "Radius /m",
        "Amplitude",
        0.0..100.0,
        -0.2..0.6,
    )?;

    // Now take the Hankel transform using transformer.qdht
    let ht = transformer.qdht(&f, Axis(0));

    // As expected, this is a top-hat function bandlimited to k<1, except for numerical error.
    helper::plot_1d(
        "simple_example_ht.png",
        transformer.kr(),
        &ht,
        "Hankel Transform",
        "Radial wavevector /m^-1",
        "Amplitude",
        0.0..5.0,
        -0.2..1.2,
    )?;

    Ok(())
}
