mod helper;
use hankrs::one_shot::{iqdht, qdht};
use hankrs::{HankelScalar, HankelTransform};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

use crate::helper::imagesc;

// Speed of single-shot vs reuse of a HankelTransform object
//
// For a simple case there are two simple forward and inverse functions which can be used to calculate the
// Hankel transform of a function sampled at an arbitrary set of points in radius / wave-number space.
// Here we will use the same example application as the usage_example:
// a beam-propagation method propagation of a radially-symmetric Gaussian beam.
fn propagate_using_object(
    r: ArrayView1<f64>,
    field: ArrayView1<Complex64>,
    nr: usize,
    nz: usize,
    z: ArrayView1<f64>,
    k0: f64,
) -> Array2<f64> {
    let transformer = HankelTransform::new_from_r_grid(0, r.to_owned());
    let field_for_transform = transformer.to_transform_r(&field).unwrap(); // Resampled field
    let hankel_transform = transformer.qdht(&field_for_transform, Axis(0));

    let mut propagated_field = Array2::<Complex64>::zeros((nr, nz));
    let kz = transformer
        .kr()
        .mapv(|kr_val| (k0 * k0 - kr_val * kr_val).sqrt());

    for (n, &z_loop) in z.iter().enumerate() {
        let phi_z = kz.mapv(|kz_val| kz_val * z_loop); // Propagation phase
        let hankel_transform_at_z = ndarray::Zip::from(&hankel_transform)
            .and(&phi_z)
            .map_collect(|&ekr, &phi| ekr * Complex64::new(0.0, phi).exp()); // Apply propagation
        let field_at_z = transformer.iqdht(&hankel_transform_at_z, Axis(0)); // iQDHT
        let field_slice = transformer.to_original_r(&field_at_z).unwrap(); // Interpolate output
        propagated_field.column_mut(n).assign(&field_slice);
    }
    propagated_field.mapv(|c| c.norm_sqr())
}

fn propagate_using_single_shot(
    r: ArrayView1<f64>,
    field: ArrayView1<Complex64>,
    nr: usize,
    nz: usize,
    z: ArrayView1<f64>,
    k0: f64,
) -> Array2<f64> {
    let (kr, hankel_transform) = qdht(r.to_owned(), &field, 0, Axis(0));

    let mut propagated_field = Array2::<Complex64>::zeros((nr, nz));
    let kz = kr.mapv(|kr_val| (k0 * k0 - kr_val * kr_val).sqrt());

    for (n, &z_loop) in z.iter().enumerate() {
        let phi_z = kz.mapv(|kz_val| kz_val * z_loop);
        let hankel_transform_at_z = ndarray::Zip::from(&hankel_transform)
            .and(&phi_z)
            .map_collect(|&ekr, &phi| ekr * Complex64::new(0.0, phi).exp());
        let (r_transform, field_at_z) = iqdht(kr.clone(), &hankel_transform_at_z, 0, Axis(0));

        let field_slice = Complex64::spline(r_transform.view(), field_at_z.view(), r, Axis(0));
        propagated_field.column_mut(n).assign(&field_slice);
    }
    propagated_field.mapv(|c| c.norm_sqr())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nr = 1024;
    let r_max = 5e-3;
    let r = Array1::linspace(0.0, r_max, nr);

    let nz = 100;
    let z_max = 0.1;
    let z = Array1::linspace(0.0, z_max, nz);

    let dr = 100e-6;
    let lambda_ = 488e-9;
    let k0 = 2.0 * PI / lambda_;

    let field_real = helper::gauss1d(r.view(), 0.0, dr); // Initial field
    let field: Array1<Complex64> = field_real.mapv(|v| Complex64::new(v, 0.0));

    // Now we need two functions that propagate the beam in two ways (giving the same answer).
    // The first will use single shot, the second will use a HankelTransform object.
    // Below we will run each of them in turn and compare the speed.

    // Now run and time the two functions:
    let start = Instant::now();
    let single_shot_intensity =
        propagate_using_single_shot(r.view(), field.view(), nr, nz, z.view(), k0);
    let duration = start.elapsed();
    println!(
        "Single shot propagation took {:.2?} s",
        duration.as_secs_f64()
    );

    let start = Instant::now();
    let object_intensity = propagate_using_object(r.view(), field.view(), nr, nz, z.view(), k0);
    let duration = start.elapsed();
    println!("Object propagation took {:.2?} s", duration.as_secs_f64());

    // The single shot approach takes a *lot* longer!
    // Plot the two results to check they are the same:

    let z_mm = z.mapv(|val| val * 1e3);
    let r_mm = r.mapv(|val| val * 1e3);

    imagesc(
        "speed_usage_single_shot.png",
        z_mm.view(),
        r_mm.view(),
        single_shot_intensity.view(),
        "Single Shot Output",
        "Propagation distance (z) /mm",
        "Radial position (r) /mm",
        0.0..1.0,
    )?;

    imagesc(
        "speed_usage_object.png",
        z_mm.view(),
        r_mm.view(),
        object_intensity.view(),
        "Object Output",
        "Propagation distance (z) /mm",
        "Radial position (r) /mm",
        0.0..1.0,
    )?;

    Ok(())
}
