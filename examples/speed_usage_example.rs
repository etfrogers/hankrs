mod helper;
use hankrs::one_shot::{iqdht, qdht};
use hankrs::{HankelScalar, HankelTransform};
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

fn propagate_using_object(
    r: &Array1<f64>,
    field: &Array1<Complex64>,
    nr: usize,
    nz: usize,
    z: &Array1<f64>,
    k0: f64,
) -> Array2<f64> {
    let transformer = HankelTransform::new_from_r_grid(0, r.clone());
    let field_for_transform = transformer.to_transform_r(field).unwrap();
    let hankel_transform = transformer.qdht(&field_for_transform, Axis(0));

    let mut propagated_field = Array2::<Complex64>::zeros((nr, nz));
    let kz = transformer
        .kr()
        .mapv(|kr_val| (k0 * k0 - kr_val * kr_val).sqrt());

    for (n, &z_loop) in z.iter().enumerate() {
        let phi_z = kz.mapv(|kz_val| kz_val * z_loop);
        let hankel_transform_at_z = ndarray::Zip::from(&hankel_transform)
            .and(&phi_z)
            .map_collect(|&ekr, &phi| ekr * Complex64::new(0.0, phi).exp());
        let field_at_z = transformer.iqdht(&hankel_transform_at_z, Axis(0));
        let field_slice = transformer.to_original_r(&field_at_z).unwrap();
        propagated_field.column_mut(n).assign(&field_slice);
    }
    propagated_field.mapv(|c| c.norm_sqr())
}

fn propagate_using_single_shot(
    r: &Array1<f64>,
    field: &Array1<Complex64>,
    nr: usize,
    nz: usize,
    z: &Array1<f64>,
    k0: f64,
) -> Array2<f64> {
    let (kr, hankel_transform) = qdht(r.clone(), field, 0, Axis(0));

    let mut propagated_field = Array2::<Complex64>::zeros((nr, nz));
    let kz = kr.mapv(|kr_val| (k0 * k0 - kr_val * kr_val).sqrt());

    for (n, &z_loop) in z.iter().enumerate() {
        let phi_z = kz.mapv(|kz_val| kz_val * z_loop);
        let hankel_transform_at_z = ndarray::Zip::from(&hankel_transform)
            .and(&phi_z)
            .map_collect(|&ekr, &phi| ekr * Complex64::new(0.0, phi).exp());
        let (r_transform, field_at_z) = iqdht(kr.clone(), &hankel_transform_at_z, 0, Axis(0));

        let field_slice = Complex64::spline(&r_transform, &field_at_z, r, Axis(0));
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

    let field_real = helper::gauss1d(&r, 0.0, dr);
    let field: Array1<Complex64> = field_real.mapv(|v| Complex64::new(v, 0.0));

    let start = Instant::now();
    let _single_shot_intensity = propagate_using_single_shot(&r, &field, nr, nz, &z, k0);
    let duration = start.elapsed();
    println!(
        "Single shot propagation took {:.2?} s",
        duration.as_secs_f64()
    );

    let start = Instant::now();
    let _object_intensity = propagate_using_object(&r, &field, nr, nz, &z, k0);
    let duration = start.elapsed();
    println!("Object propagation took {:.2?} s", duration.as_secs_f64());

    Ok(())
}
