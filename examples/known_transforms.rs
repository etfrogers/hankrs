mod helper;

use amos_bessel_rs::{bessel_j, bessel_k};
use hankrs::HankelTransform;
use hankrs::one_shot::{iqdht, qdht};
use helper::plot_1d_original_and_transform;
use ndarray::{Array1, ArrayView1, Axis};
use std::f64::consts::PI;

fn generalised_top_hat(r: ArrayView1<f64>, a: f64, p: i32) -> Array1<f64> {
    r.mapv(|rad| if rad <= a { rad.powi(p) } else { 0.0 })
}

fn generalised_jinc(v: ArrayView1<f64>, a: f64, p: i32) -> Array1<f64> {
    v.mapv(|val| {
        if val != 0.0 {
            a.powi(p + 1) * bessel_j(p as f64 + 1.0, 2.0 * PI * a * val).unwrap() / val
        } else {
            if p == -1 {
                f64::INFINITY
            } else if p == -2 {
                -PI
            } else if p == 0 {
                PI * a * a
            } else {
                0.0
            }
        }
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Gaussian function
    // First we try a Gaussian function, the Hankel transform of which should also be Gaussian.
    //
    // Note the definition in Guizar-Sicairos varies from that used by
    // Pissens by a factor of 2\pi in both scaling of the argument (so we use
    // HankelTransform.kr rather than HankelTransform.v) and also scaling of the magnitude.
    let a = 3.0;
    let radius = Array1::linspace(0.0, 3.0, 1024);
    let f = radius.mapv(|r: f64| (-a * a * r * r).exp());
    let (kr, actual_ht) = qdht(radius.clone(), &f, 0, Axis(0));

    let kr_linear = Array1::linspace(0.0, 50.0, 1024);
    let expected_ht =
        kr_linear.mapv(|k: f64| 2.0 * PI * (1.0 / (2.0 * a * a)) * (-k * k / (4.0 * a * a)).exp());

    plot_1d_original_and_transform(
        "known_transforms_gaussian.png",
        radius.view(),
        f.view(),
        "Gaussian function",
        "Radius /r",
        "Amplitude",
        0.0..3.0,
        0.0..1.05,
        kr_linear.view(),
        expected_ht.view(),
        kr.view(),
        actual_ht.view(),
        "Analytical",
        "QDHT",
        "Hankel transform - Gaussian",
        "Frequency /v",
        "Amplitude",
        0.0..50.0,
        0.0..0.4,
    )?;

    // 2. Inverse Gaussian function
    // Now we repeat for the inverse transform
    let kr2 = Array1::linspace(0.0, 50.0, 1024);
    let ht = kr2.mapv(|k: f64| 2.0 * PI * (1.0 / (2.0 * a * a)) * (-k * k / (4.0 * a * a)).exp());
    let (r, actual_f) = iqdht(kr2.clone(), &ht, 0, Axis(0));

    let r_linear = Array1::linspace(0.0, 1.0, 1024);
    let expected_f = r_linear.mapv(|rad: f64| (-a * a * rad * rad).exp());

    plot_1d_original_and_transform(
        "known_transforms_inv_gaussian.png",
        kr2.view(),
        ht.view(),
        "Hankel transform - Gaussian function",
        "Frequency /k",
        "Amplitude",
        0.0..50.0,
        0.0..0.4,
        r_linear.view(),
        expected_f.view(),
        r.view(),
        actual_f.view(),
        "Analytical",
        "QDHT",
        "Original function after IQDHT",
        "Radius /r",
        "Amplitude",
        0.0..1.2,
        0.0..1.2,
    )?;

    // 3. Generalised jinc and top-hat
    // Next we define functions to calculate the generalised top-hat and jinc
    // functions, as defined by Guizar-Sicairos and Guitierrez-Vega.
    //
    // Note that for p=0 these become a standard top-hat and
    // jinc(r) = J_1(r)/r functions.
    let ylims3 = [
        (-0.2..0.8, -0.1..1.1),
        (-0.05..0.15, -0.1..0.6),
        (-0.003..0.007, -0.01..0.07),
    ];
    // For demonstration, we choose a = 0.5 and run the code for
    // orders 0, 1 and 4 plotting and checking the mean absolute error each time.
    // First check that the Hankel transform of the generalised jinc is calculated
    // correctly.
    let a_val = 0.5;
    let r3 = Array1::linspace(0.0, 30.0, 1024);
    for (i, &order) in [0, 1, 4].iter().enumerate() {
        let f = generalised_jinc(r3.view(), a_val, order);
        let (kr, actual_ht) = qdht(r3.clone(), &f, order, Axis(0));
        let v = kr.mapv(|k| k / (2.0 * PI));

        let v_linear = Array1::linspace(0.0, 1.5, 1024);
        let expected_ht = generalised_top_hat(v_linear.view(), a_val, order);

        let title_orig = format!("Generalised jinc function, order = {}", order);
        let title_trans = format!("Hankel transform - generalised top-hat, order = {}", order);
        plot_1d_original_and_transform(
            &format!("known_transforms_tophat_{}.png", order),
            r3.view(),
            f.view(),
            &title_orig,
            "Radius /r",
            "Amplitude",
            0.0..30.0,
            ylims3[i].0.clone(),
            v_linear.view(),
            expected_ht.view(),
            v.view(),
            actual_ht.view(),
            "Analytical",
            "QDHT",
            &title_trans,
            "Frequency /v",
            "Amplitude",
            0.0..1.5,
            ylims3[i].1.clone(),
        )?;
    }

    // 4. Generalised top-hat to jinc
    // Now we repeat but the other way round: the Hankel transform of the top-hat
    // function should be the jinc function.
    let r4 = Array1::linspace(0.0, 2.0, 1024);
    let ylims4 = [
        (-0.05..1.05, -0.2..0.8),
        (-0.05..0.5, -0.05..0.15),
        (-0.05..0.07, -0.003..0.007),
    ];

    for (i, &order) in [0, 1, 4].iter().enumerate() {
        let transformer = HankelTransform::new(order, 2.0, 1024);
        let f = generalised_top_hat(transformer.radius(), a_val, order);
        let actual_ht = transformer.qdht(&f, Axis(0));
        let v = transformer.frequency();

        let v_linear = Array1::linspace(0.0, transformer.max_frequency(), 1024);
        let expected_ht = generalised_jinc(v_linear.view(), a_val, order);

        let f_orig_linear = generalised_top_hat(r4.view(), a_val, order);
        let title_orig = format!("Generalised top-hat function, order = {}", order);
        let title_trans = format!("Hankel transform - generalised jinc, order = {}", order);
        plot_1d_original_and_transform(
            &format!("known_transforms_jinc_{}.png", order),
            r4.view(),
            f_orig_linear.view(),
            &title_orig,
            "Radius /r",
            "Amplitude",
            0.0..2.0,
            ylims4[i].0.clone(),
            v_linear.view(),
            expected_ht.view(),
            v.view(),
            actual_ht.view(),
            "Analytical",
            "QDHT",
            &title_trans,
            "Frequency /v",
            "Amplitude",
            0.0..20.0,
            ylims4[i].1.clone(),
        )?;
    }

    // 5. 1/(r^2 + a^2)
    // Now we investigate the function f(r) = 1/(r^2 + a^2),
    // the Hankel transform of which is K_0(av).
    //
    // Note again the scaling factor of 2\pi.
    let a5 = 1.0;
    let transformer = HankelTransform::new(0, 50.0, 1024);
    let f5 = transformer.radius().mapv(|r| 1.0 / (r * r + a5 * a5));
    let actual_ht = transformer.qdht(&f5, Axis(0));
    let kr_grid = transformer.kr();

    let kr_linear = Array1::linspace(1e-6, 8.0, 1024);
    let expected_ht = kr_linear.mapv(|k| 2.0 * PI * bessel_k(0.0, a5 * k).unwrap());

    let r_linear = Array1::linspace(0.0, 20.0, 1024);
    let f5_linear = r_linear.mapv(|r| 1.0 / (r * r + a5 * a5));

    plot_1d_original_and_transform(
        "known_transforms_k0.png",
        r_linear.view(),
        f5_linear.view(),
        "1/(r^2 + a^2)",
        "Radius /r",
        "Amplitude",
        0.0..20.0,
        -0.05..1.05,
        kr_linear.view(),
        expected_ht.view(),
        kr_grid.view(),
        actual_ht.view(),
        "Analytical",
        "QDHT",
        "Hankel transform - 2pi K0(ak)",
        "Frequency /v",
        "Amplitude",
        0.0..8.0,
        -0.5..20.0,
    )?;

    Ok(())
}
