mod helper;

use hankrs::HankelTransform;
use helper::plot_1d_original_and_transform;
use ndarray::{Array1, Axis};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Spherical Gaussian transform
    //
    // The order-0 spherical Hankel transform of a Gaussian f(r) = exp(-a*r^2) has the
    // analytical result:
    //
    //   H_sph{ f(r) }(k) = (sqrt(pi) / (4 * a^(3/2))) * exp(-k^2 / (4a))
    //
    // We demonstrate this for a = 2.0.
    let a = 2.0_f64;
    let r_max = 20.0;
    let n_points = 250;

    let transformer = HankelTransform::new_spherical(0, r_max, n_points);

    let function: Array1<f64> = transformer.radius().mapv(|r| (-a * r.powi(2)).exp());

    let actual_transform = transformer.qdht(&function, Axis(0));
    let kr = transformer.kr();

    // Dense analytical curve for plotting
    let kr_linear = Array1::linspace(0.0, *kr.last().unwrap(), 1024);
    let expected_transform = kr_linear.mapv(|k| {
        let prefactor = PI.sqrt() / (4.0 * a.powf(1.5));
        prefactor * (-(k.powi(2)) / (4.0 * a)).exp()
    });

    // Dense original function for plotting
    let r_linear = Array1::linspace(0.0, r_max * 0.5, 1024);
    let f_linear = r_linear.mapv(|r| (-a * r.powi(2)).exp());

    plot_1d_original_and_transform(
        "spherical_known_transforms_gaussian.png",
        r_linear.view(),
        f_linear.view(),
        "Gaussian function: exp(-r²)",
        "Radius r",
        "Amplitude",
        0.0..4.0,
        -0.05..1.05,
        kr_linear.view(),
        expected_transform.view(),
        kr.view(),
        actual_transform.view(),
        "Analytical",
        "SQDHT",
        "Spherical Hankel transform - Gaussian",
        "Wavenumber k",
        "Amplitude",
        0.0..10.0,
        -0.01..0.2,
    )?;

    // 2. Spherical top-hat transform
    //
    // The order-0 spherical Hankel transform of a top-hat function f(r) = 1 for r < a,
    // 0 otherwise, has the analytical result:
    //
    //   H_sph{ f(r) }(k) = (sin(ka) - ka*cos(ka)) / k^3
    //
    // We demonstrate this for a = 0.5
    let a_hat = 0.5_f64;
    let r_max_hat = 20.0;
    let n_points_hat = 1000;

    let transformer_hat = HankelTransform::new_spherical(0, r_max_hat, n_points_hat);

    // Build top-hat on the QDHT grid; snap `a` to the nearest grid point
    let function_hat: Array1<f64> = transformer_hat
        .radius()
        .mapv(|r| if r < a_hat { 1.0 } else { 0.0 });

    // Snap a to the nearest grid sample to remove discretisation error in the comparison
    let actual_a = transformer_hat
        .radius()
        .iter()
        .rposition(|&r| r < a_hat)
        .map(|idx| transformer_hat.radius()[idx])
        .unwrap_or(a_hat);

    let actual_transform_hat = transformer_hat.qdht(&function_hat, Axis(0));
    let kr_hat = transformer_hat.kr();

    // Dense analytical curve for plotting
    let kr_hat_linear = Array1::linspace(1e-6, *kr_hat.last().unwrap(), 1024);
    let expected_transform_hat = kr_hat_linear
        .mapv(|k| { (k * actual_a).sin() - k * actual_a * (k * actual_a).cos() } / k.powi(3));

    // Dense original function for plotting
    let r_hat_linear = Array1::linspace(0.0, 3.0, 1024);
    let f_hat_linear = r_hat_linear.mapv(|r| if r < a_hat { 1.0 } else { 0.0 });

    plot_1d_original_and_transform(
        "spherical_known_transforms_tophat.png",
        r_hat_linear.view(),
        f_hat_linear.view(),
        "Top-hat function: 1 for r < a",
        "Radius r",
        "Amplitude",
        0.0..2.0,
        -0.1..1.2,
        kr_hat_linear.view(),
        expected_transform_hat.view(),
        kr_hat.view(),
        actual_transform_hat.view(),
        "Analytical",
        "SQDHT",
        "Spherical Hankel transform - Top-Hat",
        "Wavenumber k",
        "Amplitude",
        0.0..30.0,
        -0.02..0.05,
    )?;

    Ok(())
}
