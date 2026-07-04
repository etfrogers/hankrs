mod helper;
use hankrs::HankelTransform;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nr = 1024;
    let r_max = 5e-3;
    let r = Array1::linspace(0.0, r_max, nr);

    let nz = 200;
    let z_max = 0.1;
    let z = Array1::linspace(0.0, z_max, nz);

    let dr = 100e-6;
    let lambda_ = 488e-9;
    let k0 = 2.0 * PI / lambda_;

    let transformer = HankelTransform::new_from_r_grid(0, r.clone());

    let er_real = helper::gauss1d(&r, 0.0, dr);
    let er: Array1<Complex64> = er_real.mapv(|v| Complex64::new(v, 0.0));
    let er_h = transformer.to_transform_r(&er).unwrap();

    let ekr_h = transformer.qdht(&er_h, Axis(0));

    let mut erz = Array2::<Complex64>::zeros((nr, nz));
    let kz = transformer
        .kr()
        .mapv(|kr_val| (k0 * k0 - kr_val * kr_val).sqrt());

    for (i, &z_loop) in z.iter().enumerate() {
        let phi_z = kz.mapv(|kz_val| kz_val * z_loop);
        let ekr_hz = ndarray::Zip::from(&ekr_h)
            .and(&phi_z)
            .map_collect(|&ekr, &phi| ekr * Complex64::new(0.0, phi).exp());
        let er_hz = transformer.iqdht(&ekr_hz, Axis(0));
        let erz_slice = transformer.to_original_r(&er_hz).unwrap();
        erz.column_mut(i).assign(&erz_slice);
    }
    let irz = erz.mapv(|c| c.norm_sqr());

    let z_mm = z.mapv(|val| val * 1e3);
    let r_mm = r.mapv(|val| val * 1e3);

    helper::imagesc(
        "usage_example_irz.png",
        &z_mm,
        &r_mm,
        &irz,
        "Radial field intensity",
        "Propagation distance (z) /mm",
        "Radial position (r) /mm",
    )?;

    // Normalised intensity plot
    let mut irz_norm = irz.clone();
    for i in 0..nz {
        let peak = irz_norm
            .column(i)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if peak > 0.0 {
            let mut col = irz_norm.column_mut(i);
            col.mapv_inplace(|v| v / peak);
        }
    }

    helper::imagesc(
        "usage_example_irz_norm.png",
        &z_mm,
        &r_mm,
        &irz_norm,
        "Normalised Radial field intensity",
        "Propagation distance (z) /mm",
        "Radial position (r) /mm",
    )?;

    // Vectorised approach
    let mut erz_vec = Array2::<Complex64>::zeros((nr, nz));
    for i in 0..nz {
        let z_loop = z[i];
        let phi_z = kz.mapv(|kz_val| kz_val * z_loop);
        let ekr_hz = ndarray::Zip::from(&ekr_h)
            .and(&phi_z)
            .map_collect(|&ekr, &phi| ekr * Complex64::new(0.0, phi).exp());
        let er_hz = transformer.iqdht(&ekr_hz, Axis(0));
        let erz_slice = transformer.to_original_r(&er_hz).unwrap();
        erz_vec.column_mut(i).assign(&erz_slice);
    }

    let irz_vec = erz_vec.mapv(|c| c.norm_sqr());

    // Assert matches
    let error: f64 = irz
        .iter()
        .zip(irz_vec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f64>()
        / (nr * nz) as f64;
    assert!(error < 1e-10);

    Ok(())
}
