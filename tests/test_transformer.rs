mod utils;

use amos_bessel_rs::bessel_k;
use approx::assert_relative_eq;
use hankrs::hankel::HankelTransform;
use ndarray::{Array, Array1, Array2, Axis, Dim, Dimension, Ix1};
use ndarray_stats::{DeviationExt, QuantileExt};
use num::pow::Pow;
use rand::random;
use rstest::{fixture, rstest};
use rstest_reuse::{apply, template};
use std::{f64::consts::PI, fmt::Debug, mem::MaybeUninit, sync::LazyLock};
use utils::{assert_relative_eq_with_end_points, generalised_jinc, generalised_top_hat};

use crate::utils::assert_arrays_equal;

static TRANSFORMERS: LazyLock<[HankelTransform; 5]> = LazyLock::new(|| {
    [
        HankelTransform::new_from_r_grid(0, radius()),
        HankelTransform::new_from_r_grid(1, radius()),
        HankelTransform::new_from_r_grid(2, radius()),
        HankelTransform::new_from_r_grid(3, radius()),
        HankelTransform::new_from_r_grid(4, radius()),
    ]
});

fn random_array_like<D: Dimension>(v: &Array<f64, D>) -> Array<f64, D> {
    random_array(v.raw_dim())
}

fn random_array<D: Dimension>(shape: D) -> Array<f64, D> {
    Array::uninit(shape).map(|_: &MaybeUninit<f64>| random::<f64>() * 10.0)
}

#[derive(Clone)]
struct Shape<'a> {
    name: String,
    f: &'a dyn Fn(f64) -> f64,
}

impl<'a> Debug for Shape<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shape").field("name", &self.name).finish()
    }
}

impl<'a> Shape<'a> {
    fn new(name: &str, f: &'a (impl Fn(f64) -> f64 + 'static)) -> Self {
        Self {
            name: name.to_string(),
            f,
        }
    }
}

#[template]
#[rstest]
#[case(Shape::new("zeros", &|_| 0.0,))]
#[case(Shape::new("e^(-r^2)", &|r: f64| (-r.pow(2.0_f64)).exp(),))]
#[case(Shape::new("r",  &|r: f64| r ))]
#[case(Shape::new("r^2",  &|r: f64|  r.pow(2.0) ))]
#[case(Shape::new("1/(sqrt(r^2 + 0.1^2))",
                  &|r: f64|  1.0 / (r.pow(2.0_f64)+0.1.pow(2.0_f64)).sqrt() ))]
fn smooth_shapes(#[case] shape: Shape) {}

#[fixture]
fn radius() -> Array1<f64> {
    Array1::linspace(0.0, 3.0, 1024)
}

#[fixture]
#[once]
fn transformer_zero_order(radius: Array1<f64>) -> HankelTransform {
    let order = 0;
    HankelTransform::new_from_r_grid(order, radius)
}

#[rstest]
fn test_round_trip(transformer_zero_order: &HankelTransform) {
    let fun = random_array_like(transformer_zero_order.radius());
    let ht = transformer_zero_order.qdht(&fun, Axis(0));
    let reconstructed = transformer_zero_order.iqdht(&ht, Axis(0));
    assert_relative_eq!(
        fun.as_slice().unwrap(),
        reconstructed.as_slice().unwrap(),
        max_relative = -1.0,
        epsilon = 1e-9
    )
}

// -------------------
// Test Interpolations
// -------------------
#[apply(smooth_shapes)]
#[rstest]
#[trace]
fn test_round_trip_r_interpolation(
    shape: Shape,
    radius: Array1<f64>,
    #[values(0, 1, 2, 3, 4)] order_ind: usize,
) {
    let transformer = &TRANSFORMERS[order_ind];
    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    let fun = radius.mapv_into(shape.f);
    let transform_func = transformer.to_transform_r(&fun).unwrap();
    let reconstructed_func = transformer.to_original_r(&transform_func).unwrap();
    assert_relative_eq_with_end_points(&reconstructed_func, &fun, 1e-4, 1e-3, 0.0, 2e-5);
}

#[apply(smooth_shapes)]
#[rstest]
#[trace]
fn test_round_trip_k_interpolation(
    shape: Shape,
    radius: Array1<f64>,
    #[values(0, 1, 2, 3, 4)] order: i32,
) {
    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    let k_grid = radius.mapv(|r| r / 10.0);
    let transformer = HankelTransform::new_from_k_grid(order, k_grid);

    let fun = radius.mapv_into(shape.f);
    let transform_func = transformer.to_transform_k(&fun).unwrap();
    let reconstructed_func = transformer.to_original_k(&transform_func).unwrap();
    assert_relative_eq_with_end_points(&reconstructed_func, &fun, 1e-4, 1e-3, 0.0, 2e-7);
}

#[apply(smooth_shapes)]
#[rstest]
fn test_round_trip_with_interpolation(
    shape: Shape,
    radius: Array1<f64>,
    #[values(0, 1, 2, 3, 4)] order_ind: usize,
) {
    let transformer = &TRANSFORMERS[order_ind]; // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    let fun = radius.mapv_into(shape.f);
    let fun_hr = transformer.to_transform_r(&fun).unwrap();
    let ht = transformer.qdht(&fun_hr, Axis(0));
    let reconstructed_hr = transformer.iqdht(&ht, Axis(0));
    let reconstructed = transformer.to_original_r(&reconstructed_hr).unwrap();

    let a_tol_end = 1e-3;
    let mut r_tol_body = 2e-4;
    let mut a_tol_body = -1.;
    let mut r_tol_end = -1.;
    if shape.name == "1/(sqrt(r^2 + 0.1^2))" {
        r_tol_end = 3e-2;
        r_tol_body = 2e-3;
    }
    if shape.name == "r^2" {
        r_tol_body = -1.0;
        a_tol_body = 2e-4
    }
    assert_relative_eq_with_end_points(
        &fun,
        &reconstructed,
        r_tol_body,
        r_tol_end,
        a_tol_body,
        a_tol_end,
    )
}

#[rstest]
fn test_original_rk_grid() {
    let r_1d = Array1::linspace(0.0, 1.0, 10);
    let k_1d = r_1d.clone();
    let transformer = HankelTransform::new(0, 1., 10);
    assert!(transformer.original_radial_grid().is_none());
    assert!(transformer.original_k_grid().is_none());

    let transformer_r = HankelTransform::new_from_r_grid(0, r_1d);
    // no error
    assert!(transformer_r.original_radial_grid().is_some());
    assert!(transformer_r.original_k_grid().is_none());

    let transformer_k = HankelTransform::new_from_k_grid(0, k_1d);
    // no error
    assert!(transformer_k.original_k_grid().is_some());
    assert!(transformer_k.original_radial_grid().is_none())
}

// ---------------
// Test Invariants
// ---------------
#[apply(smooth_shapes)]
#[case(Shape::new("random", &|_| random::<f64>()*10.0))]
#[rstest]
fn test_parsevals_theorem(
    shape: Shape,
    radius: Array1<f64>,
    #[values(0, 1, 2, 3, 4)] order_ind: usize,
) {
    let transformer = &TRANSFORMERS[order_ind];
    // As per equation 11 of Guizar-Sicairos, the UNSCALED transform is unitary,
    // i.e. if we pass in the unscaled fr (=Fr), the unscaled fv (=Fv)should have the
    // same sum of abs val^2. Here the unscaled transform is simply given by
    // ht = transformer.T @ func
    let fun = radius.mapv(shape.f); // shape.f, nil, &t.radius)
    let intensity_before = fun.mapv(intensity);
    let energy_before = intensity_before.sum();
    let ht = transformer.transform_matrix() * fun;
    let intensity_after = ht.mapv(intensity);
    let energy_after = intensity_after.sum();
    assert_relative_eq!(energy_before, energy_after, max_relative = 1e-10);
}

fn intensity(v: f64) -> f64 {
    v.abs().powf(2.0)
}

#[rstest]
#[case("Jinc", &generalised_jinc)]
#[case("Top Hat", &generalised_top_hat)]
fn test_energy_conservation(
    #[case] _shape_name: &str,
    #[case] func: &dyn Fn(&Array1<f64>, f64, i32) -> Array1<f64>,
    #[values(0, 1, 2, 3, 4)] order: i32,
) {
    let integrate_over_r = |r: &Array1<f64>, y| -> f64 {
        let integrand: Array1<f64> = 2.0 * PI * r * y;
        (0..(r.len() - 1))
            .map(|i| (r[i + 1] - r[i]) * (integrand[i + 1] + integrand[i]) / 2.0)
            .sum()
    };

    let transformer = HankelTransform::new(order, 10.0, 1024);
    let fun = func(transformer.radius(), 0.5, order);

    let intensity_before = fun.mapv(intensity);
    let energy_before = integrate_over_r(transformer.radius(), intensity_before);

    let ht = transformer.qdht(&fun, Axis(0));
    let intensity_after = ht.mapv(intensity);
    let energy_after = integrate_over_r(transformer.frequency(), intensity_after);

    assert_relative_eq!(energy_before, energy_after, epsilon = 0.006);
}

// -------------------
// Test known HT pairs
// -------------------

#[rstest]
fn test_jinc(#[values(1.0, 0.7, 0.1)] a: f64, #[values(0, 1, 2, 3, 4)] order_ind: usize) {
    let transformer = &TRANSFORMERS[order_ind];
    // let transformer = HankelTransform::new_from_r_grid(order, radius);
    let order = transformer.order();
    let f = generalised_jinc(transformer.radius(), a, order);
    let expected_ht = generalised_top_hat(transformer.frequency(), a, order);
    let actual_ht = transformer.qdht(&f, Axis(0));
    let err = expected_ht.mean_abs_err(&actual_ht).unwrap();
    assert!(err < 1e-3);
}

#[rstest]
fn test_top_hat(#[values(1.0, 1.5, 0.1)] a: f64, #[values(0, 1, 2, 3, 4)] order_ind: usize) {
    let transformer = &TRANSFORMERS[order_ind];
    let order = transformer.order();
    let f = generalised_top_hat(transformer.radius(), a, order);
    let expected_ht = generalised_jinc(transformer.frequency(), a, order);
    let actual_ht = transformer.qdht(&f.into_dyn(), Axis(0));
    let err = expected_ht
        .mean_abs_err(&actual_ht.into_dimensionality::<Ix1>().unwrap())
        .unwrap();
    assert!(err < 1e-3);
}

#[rstest]
fn test_gaussian(transformer_zero_order: &HankelTransform, #[values(2.0, 5.0, 10.0)] a: f64) {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let a2 = a.powi(2);
    let f = transformer_zero_order
        .radius()
        .mapv(|r| (-a2 * r.powi(2)).exp());
    let expected_ht = transformer_zero_order
        .kr()
        .mapv(|k| 2.0 * PI * (1.0 / (2.0 * a2)) * (-(k.powi(2) / (4.0 * a2))).exp());
    let actual_ht = transformer_zero_order.qdht(&f.into_dyn(), Axis(0));
    assert_arrays_equal(&expected_ht, &actual_ht, 1e-9, 0.0);
}

#[rstest]
fn test_inverse_gaussian(
    transformer_zero_order: &HankelTransform,
    #[values(2.0, 5.0, 10.0)] a: f64,
) {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.

    let a2 = a.powi(2);
    let expected_f = transformer_zero_order
        .radius()
        .mapv(|r| (-a2 * r.powi(2)).exp());
    let ht = transformer_zero_order
        .kr()
        .mapv(|k| 2.0 * PI * (1.0 / (2.0 * a2)) * (-(k.powi(2) / (4.0 * a2))).exp());
    let actual_f = transformer_zero_order.iqdht(&ht.into_dyn(), Axis(0));
    assert_arrays_equal(&actual_f, &expected_f, 1e-9, 0.0);
}

#[rstest]
fn test_gaussian_2d(#[values(0, 1)] axis: usize, transformer_zero_order: &HankelTransform) {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let a = Array1::linspace(2.0, 10.0, 50);
    let mut dims_a = Array1::ones(2);
    dims_a[1 - axis] = a.len();
    let mut dims_r = Array1::ones(2);
    dims_r[axis] = transformer_zero_order.radius().len();
    let a_reshaped = a.to_shape(dims_a.as_slice().unwrap()).unwrap();
    let r_reshaped = transformer_zero_order
        .radius()
        .to_shape(dims_r.as_slice().unwrap())
        .unwrap();
    let kr_reshaped = transformer_zero_order
        .kr()
        .to_shape(dims_r.as_slice().unwrap())
        .unwrap();
    let f = (-a_reshaped.powi(2) * r_reshaped.powi(2)).exp();
    let expected_ht = 2.0
        * PI
        * (1.0 / (2.0 * a_reshaped.powi(2)))
        * (-kr_reshaped.powi(2) / (4.0 * a_reshaped.powi(2))).exp();
    let actual_ht = transformer_zero_order.qdht(&f, Axis(axis));
    assert_arrays_equal(&expected_ht, &actual_ht, 1e-8, 1e-5);
}

#[rstest]
fn test_inverse_gaussian_2d(#[values(0, 1)] axis: usize, transformer_zero_order: &HankelTransform) {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let a = Array1::linspace(2.0, 10.0, 50);
    let mut dims_a = Array1::ones(2);
    dims_a[1 - axis] = a.len();
    let mut dims_r = Array1::ones(2);
    dims_r[axis] = transformer_zero_order.radius().len();
    let a_reshaped: ndarray::ArrayBase<ndarray::CowRepr<'_, f64>, _> =
        a.to_shape(dims_a.as_slice().unwrap()).unwrap();
    let r_reshaped = transformer_zero_order
        .radius()
        .to_shape(dims_r.as_slice().unwrap())
        .unwrap();
    let kr_reshaped = transformer_zero_order
        .kr()
        .to_shape(dims_r.as_slice().unwrap())
        .unwrap();
    let ht = 2.0
        * PI
        * (1.0 / (2.0 * a_reshaped.powi(2)))
        * (-kr_reshaped.powi(2) / (4.0 * a_reshaped.powi(2))).exp();
    let actual_f = transformer_zero_order.iqdht(&ht, Axis(axis));
    let expected_f = (-a_reshaped.powi(2) * r_reshaped.powi(2)).exp();
    assert_arrays_equal(&expected_f, &actual_f, 1e-8, 1e-5);
}

#[rstest]
fn test1_over_r2_plus_z2(#[values(2.0, 1.0, 0.1)] a: f64) {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let transformer = HankelTransform::new(0, 50.0, 1024);
    let f = transformer.radius().mapv(|r| 1.0 / (r.powi(2) + a.powi(2)));
    let expected_ht = transformer
        .kr()
        .mapv(|k| 2.0 * PI * bessel_k(0, a * k).unwrap());
    let actual_ht = transformer.qdht(&f, Axis(0));

    // These tolerances are pretty loose, but there seems to be large
    // error here
    assert_arrays_equal(&actual_ht, &expected_ht, 0.01, 0.1);
    let err = expected_ht.mean_abs_err(&actual_ht).unwrap();
    assert!(err < 4e-3);
}

fn sinc(x: f64) -> f64 {
    x.sin() / x
}

#[rstest]
fn test_sinc(#[values(1, 4)] p: i32) {
    /*Tests from figure 1 of
      *"Computation of quasi-discrete Hankel transforms of the integer
      order for propagating optical wave fields"*
      Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
      J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
    */
    let transformer = HankelTransform::new(p, 3.0, 256);

    let frequency = transformer.frequency();
    let gamma = 5.0;
    let fun = transformer.radius().mapv(|r| sinc(2.0 * PI * gamma * r));

    let pf = p as f64;
    let expected_ht = frequency.mapv(|v| {
        let norm_gamma_v = (gamma.powi(2) - v.powi(2)).sqrt();
        let norm_v_gamma = (v.powi(2) - gamma.powi(2)).sqrt();

        if v < gamma {
            (v.powf(pf) * (pf * PI / 2.0).cos())
                / (2.0 * PI * gamma * norm_gamma_v * (gamma + norm_gamma_v).powf(pf))
        } else {
            (pf * (gamma / v).asin()).sin() / (2.0 * PI * gamma * norm_v_gamma)
        }
    });
    let ht = transformer.qdht(&fun, Axis(0));
    let max_ht = ht.max().unwrap();
    // use the same error measure as the paper
    let dynamical_error = 20.0 * ((&expected_ht - &ht).abs() / *max_ht).log10();

    dynamical_error.iter().zip(frequency).for_each(|(de, v)| {
        // threshold is lower for areas not close to gamma
        let threshold = if *v > gamma * 1.25 || *v < gamma * 0.75 {
            -35.0
        } else {
            -10.0
        };
        assert!(*de < threshold);
    });
}
/*
fn _plot_stuff(x: &Array1<f64>, y1: &Array1<f64>, y2: &Array1<f64>, p: i32) {
    let out_file_name = format!("graph{p}.png");
    use plotters::prelude::*;
    let root = BitMapBackend::new(&out_file_name, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .right_y_label_area_size(40)
        .margin(5)
        .caption("Dual Y-Axis Example", ("sans-serif", 50.0).into_font())
        .build_cartesian_2d(
            0.0..(*x.max().unwrap()),
            (*y1.min().unwrap())..(*y1.max().unwrap() / 1.0),
        )
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        // .y_label_formatter(&|x| format!("{:e}", x))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            x.clone().into_iter().zip(y1.clone().into_iter()),
            &BLUE,
        ))
        .unwrap()
        .label("y1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(
            x.clone().into_iter().zip(y2.clone().into_iter()),
            &RED,
        ))
        .unwrap()
        .label("y2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(RGBColor(128, 128, 128))
        .draw()
        .unwrap();

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", out_file_name);
}
*/
// ------------------------
// End Known Transfom pairs
// ------------------------

#[rstest]
fn test_round_trip_2d(
    #[values(0, 1)] axis: usize,
    #[values(1, 100, 27)] two_d_size: usize,
    #[values(0, 1, 2, 3, 4)] order_ind: usize,
) {
    let transformer = &TRANSFORMERS[order_ind];

    let mut dims = [two_d_size, two_d_size];
    dims[axis] = transformer.radius().len();
    let func = random_array(Dim(dims));
    let ht = transformer.qdht(&func, Axis(axis));
    let reconstructed = transformer.iqdht(&ht, Axis(axis));
    assert_arrays_equal(&func, &reconstructed, 1e-8, 1e-8);
}

#[rstest]
fn test_round_trip_3d(
    #[values(0, 1)] axis: usize,
    #[values(1, 100, 27)] two_d_size: usize,
    #[values(0, 1, 2, 3, 4)] order_ind: usize,
) {
    let transformer = &TRANSFORMERS[order_ind];
    let mut dims = [two_d_size, two_d_size, two_d_size];
    dims[axis] = transformer.radius().len();
    let func = random_array(Dim(dims));
    let ht = transformer.qdht(&func, Axis(axis));
    let reconstructed = transformer.iqdht(&ht, Axis(axis));
    assert_arrays_equal(&func, &reconstructed, 1e-8, 1e-8);
}

#[rstest]
fn test_r_creation_equivalence(
    #[values(10, 100, 512, 1024)] n_points: usize,
    #[values(0.1, 10.0, 20.0, 1e6)] max_radius: f64,
) {
    let transformer1 = HankelTransform::new(0, max_radius, n_points);
    let r = Array1::linspace(0.0, max_radius, n_points);
    let transformer2 = HankelTransform::new_from_r_grid(0, r);

    assert_relative_eq!(
        transformer1,
        transformer2,
        max_relative = 1e-8,
        epsilon = 1e-8
    );
}

#[apply(smooth_shapes)]
#[rstest]
fn test_round_trip_r_interpolation_2d(
    shape: Shape,
    radius: Array1<f64>,
    #[values(0, 1, 2, 3, 4)] order_ind: usize,
    #[values(0, 1)] axis: usize,
) {
    let transformer = &TRANSFORMERS[order_ind];

    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    let amplitude = random_array(Dim(10));
    let func_1d = radius.mapv(shape.f);
    let func = if axis == 0 {
        outer(&func_1d, &amplitude)
    } else {
        outer(&amplitude, &func_1d)
    };
    let transform_func = transformer.to_transform_r_nd(&func, Axis(axis)).unwrap();
    let reconstructed_func = transformer
        .to_original_r_nd(&transform_func, Axis(axis))
        .unwrap();
    assert_arrays_equal(&func, &reconstructed_func, 1e-8, 1e-4);
}

#[apply(smooth_shapes)]
#[rstest]
fn test_round_trip_k_interpolation_2d(
    shape: Shape,
    radius: Array1<f64>,
    #[values(0, 1, 2, 3, 4)] order: i32,
    #[values(0, 1)] axis: usize,
) {
    let k_grid = &radius / 10.0;
    let transformer = HankelTransform::new_from_k_grid(order, k_grid.clone());

    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    let amplitude = random_array(Dim(10));
    let func_1d = &k_grid.mapv(shape.f);
    let func = if axis == 0 {
        outer(&func_1d, &amplitude)
    } else {
        outer(&amplitude, &func_1d)
    };
    let transform_func = transformer.to_transform_k_nd(&func, Axis(axis)).unwrap();
    let reconstructed_func = transformer
        .to_original_k_nd(&transform_func, Axis(axis))
        .unwrap();
    assert_arrays_equal(&func, &reconstructed_func, 1e-8, 1e-4);
}

fn outer(x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_reshaped = x.to_shape((size_x, 1)).unwrap();
    let y_reshaped = y.to_shape((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
}

#[rstest]
fn test_jinc2d(
    #[values(1.0, 0.7, 0.1)] a: f64,
    #[values(0, 1)] axis: usize,
    #[values(1, 100, 27)] two_d_size: usize,
    #[values(0, 1, 2, 3, 4)] order_ind: usize,
) {
    let transformer = &TRANSFORMERS[order_ind];
    let f = generalised_jinc(transformer.radius(), a, transformer.order());
    // using a range up to 2.0 to make error magnitude the same as 1D case.
    let second_axis = &Array1::linspace(0.0, 2.0, two_d_size);
    let expected_ht = generalised_top_hat(transformer.frequency(), a, transformer.order());
    let (f_array, expected_ht_array) = if axis == 0 {
        (outer(&f, &second_axis), outer(&expected_ht, &second_axis))
    } else {
        (outer(&second_axis, &f), outer(&second_axis, &expected_ht))
    };
    let actual_ht = transformer.qdht(&f_array, Axis(axis));
    let error = (expected_ht_array).mean_abs_err(&actual_ht).unwrap();
    assert!(error < 1e-3, "Error was {error}");
}
