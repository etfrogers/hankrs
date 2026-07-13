mod utils;

use std::f64::consts::PI;

use amos_bessel_rs::bessel_k;
use hankrs::{
    HankelTransform,
    one_shot::{iqdht, qdht},
};
use ndarray::{Array1, Array2, Axis, CowArray, Dim, s};
use ndarray_stats::DeviationExt;
use rstest::rstest;
use utils::{generalised_jinc, generalised_top_hat, outer, radius};

use crate::utils::assert_arrays_equal;

#[rstest]
fn test_jinc_oneshot(
    radius: Array1<f64>,
    #[values(1.0, 0.7, 0.1)] a: f64,
    #[values(0, 1, 2, 3, 4)] order: i32,
) {
    let f = generalised_jinc(radius.view(), a, order);
    let (kr, actual_ht) = qdht(radius, &f, order, Axis(0));
    let v = kr / (2.0 * PI);
    let expected_ht = generalised_top_hat(v.view(), a, order);
    let error = expected_ht.mean_abs_err(&actual_ht).unwrap();
    assert!(error < 1e-3);
}

#[rstest]
fn test_jinc2d_oneshot(
    radius: Array1<f64>,
    #[values(1.0, 0.7, 0.1)] a: f64,
    #[values(0, 1, 2, 3, 4)] order: i32,
    #[values(0, 1)] axis: usize,
    #[values(1, 35, 27)] two_d_size: usize,
) {
    let f = generalised_jinc(radius.view(), a, order);
    let second_axis = Array1::linspace(0.0, 6.0, two_d_size);
    let f_array = if axis == 0 {
        outer(f.view(), second_axis.view())
    } else {
        outer(second_axis.view(), f.view())
    };
    let (kr, actual_ht) = qdht(radius, &f_array, order, Axis(axis));
    let v = kr / (2.0 * PI);
    let expected_ht = generalised_top_hat(v.view(), a, order);
    let expected_ht_array = if axis == 0 {
        outer(expected_ht.view(), second_axis.view())
    } else {
        outer(second_axis.view(), expected_ht.view())
    };
    let error = expected_ht_array.mean_abs_err(&actual_ht).unwrap();
    // multiply tolerance to allow for the larger values caused;
    // by second_axis having values greater than 1;
    assert!(error < 4e-3);
}

#[rstest]
fn test_top_hat(
    radius: Array1<f64>,
    #[values(1.0, 1.5, 0.1)] a: f64,
    #[values(0, 1, 2, 3, 4)] order: i32,
) {
    let f = generalised_top_hat(radius.view(), a, order);
    let (kr, actual_ht) = qdht(radius, &f, order, Axis(0));
    let v = kr / (2.0 * PI);
    let expected_ht = generalised_jinc(v.view(), a, order);
    let error = expected_ht.mean_abs_err(&actual_ht).unwrap();
    assert!(error < 1e-3);
}

#[rstest]
fn test_gaussian(#[values(2.0, 5.0, 10.0)] a: f64, radius: Array1<f64>) {
    // Note the definition in Guizar-Sicairos varies by 2.0 * PI in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let f = (-a.powi(2) * radius.powi(2)).exp();
    let (kr, actual_ht) = qdht(radius, &f, 0, Axis(0));
    let expected_ht =
        2.0 * PI * (1.0 / (2.0 * a.powi(2))) * (-kr.powi(2) / (4.0 * a.powi(2))).exp();
    assert_arrays_equal(&expected_ht, &actual_ht, 1e-8, 1e-5);
}

#[rstest]
fn test_inverse_gaussian(#[values(2.0, 5.0, 10.0)] a: f64) {
    // Note the definition in Guizar-Sicairos varies by 2.0 * PI in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let kr = Array1::linspace(0.0, 200.0, 1024);
    let ht = 2.0 * PI * (1.0 / (2.0 * a.powi(2))) * (-kr.powi(2) / (4.0 * a.powi(2))).exp();
    let (r, actual_f) = iqdht(kr, &ht, 0, Axis(0));
    let expected_f = (-a.powi(2) * r.powi(2)).exp();
    assert_arrays_equal(&expected_f, &actual_f, 1e-8, 1e-5);
}

#[rstest]
fn test_gaussian_2d(#[values(0, 1)] axis: usize, radius: Array1<f64>) {
    // Note the definition in Guizar-Sicairos varies by 2.0 * PI in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let a = Array1::linspace(2.0, 10.0, 50);
    let mut dims_a = [1, 1];
    dims_a[1 - axis] = a.len();
    let mut dims_r = [1, 1];
    dims_r[axis] = radius.len();
    let a_reshaped = a.to_shape(dims_a).unwrap();
    let r_reshaped = radius.to_shape(dims_r).unwrap();
    let f = (-a_reshaped.powi(2) * r_reshaped.powi(2)).exp();
    let (kr, actual_ht) = qdht(radius, &f, 0, Axis(axis));
    let kr_reshaped = kr.to_shape(dims_r).unwrap();
    let expected_ht = 2.0
        * PI
        * (1.0 / (2.0 * a_reshaped.powi(2)))
        * (-kr_reshaped.powi(2) / (4.0 * a_reshaped.powi(2))).exp();
    assert_arrays_equal(&expected_ht, &actual_ht, 1e-8, 1e-5);
}

#[rstest]
fn test_inverse_gaussian_2d(#[values(0, 1)] axis: usize) {
    // Note the definition in Guizar-Sicairos varies by 2.0 * PI in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let kr = Array1::linspace(0.0, 200.0, 1024);
    let a = Array1::linspace(2.0, 10.0, 50);
    let mut dims_a = [1, 1];
    dims_a[1 - axis] = a.len();
    let mut dims_r = [1, 1];
    dims_r[axis] = kr.len();
    let a_reshaped = a.to_shape(dims_a).unwrap();
    let kr_reshaped: CowArray<f64, Dim<[usize; 2]>> = kr.to_shape(dims_r).unwrap();
    let term: Array2<f64> = -kr_reshaped.powi(2) / (4.0 * a_reshaped.powi(2));
    let ht: Array2<f64> = 2.0 * PI * (1.0 / (2.0 * a_reshaped.powi(2))) * term.exp();
    let (r, actual_f) = iqdht(kr, &ht, 0, Axis(axis));
    let r_reshaped = r.to_shape(dims_r).unwrap();
    let expected_f = (-a_reshaped.powi(2) * r_reshaped.powi(2)).exp();
    assert_arrays_equal(&expected_f, &actual_f, 1e-8, 1e-5);
}

#[rstest]
fn test_1_over_r2_plus_z2(#[values(2.0, 1.0, 0.5)] a: f64) {
    // Note the definition in Guizar-Sicairos varies by 2.0 * PI in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let r = Array1::linspace(0.0, 50.0, 1024);
    let f = 1.0 / (r.powi(2) + a.powi(2));
    let (kr, actual_ht) = qdht(r, &f, 0, Axis(0));
    let expected_ht = 2.0 * PI * (a * kr).mapv_into(|v| bessel_k(0, v).unwrap());
    // as this diverges at zero, the first few entries have higher errors, so ignore them
    let expected_ht = expected_ht.slice(s![10..]);
    let actual_ht = actual_ht.slice(s![10..]);
    let error = expected_ht.mean_abs_err(&actual_ht).unwrap();
    assert!(error < 1e-3);
}

// -------------------
// Test equivalence of one-shot and standard
// -------------------
#[rstest]
fn test_jinc_equivalence(
    #[values(1.0, 0.7, 0.1)] a: f64,
    #[values(0, 1, 2, 3, 4)] order: i32,
    radius: Array1<f64>,
) {
    let f = generalised_jinc(radius.view(), a, order);
    let (_, one_shot_ht) = qdht(radius.clone(), &f, order, Axis(0));

    let transformer = HankelTransform::new_from_r_grid(order, radius);
    let f_t = generalised_jinc(transformer.radius(), a, order);
    let standard_ht = transformer.qdht(&f_t, Axis(0));
    assert_arrays_equal(&one_shot_ht, &standard_ht, 1e-8, 1e-5);
}

#[should_panic] // generalised_top_hat has discontinuities, so deals badly with interpolation
#[rstest]
fn test_top_hat_equivalence(
    #[values(1.0, 0.7, 0.1)] a: f64,
    #[values(0, 1, 2, 3, 4)] order: i32,
    radius: Array1<f64>,
) {
    let f = generalised_top_hat(radius.view(), a, order);
    let (_, one_shot_ht) = qdht(radius.clone(), &f, order, Axis(0));
    let transformer = HankelTransform::new_from_r_grid(order, radius);
    let f_t = generalised_top_hat(transformer.radius(), a, order);
    let standard_ht = transformer.qdht(&f_t, Axis(0));
    assert_arrays_equal(&one_shot_ht, &standard_ht, 1e-8, 1e-5);
}

#[rstest]
fn test_gaussian_equivalence(#[values(2.0, 5.0, 10.0)] a: f64, radius: Array1<f64>) {
    // Note the definition in Guizar-Sicairos varies by 2.0 * PI in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    let f = (-a.powi(2) * radius.powi(2)).exp();
    let (_, one_shot_ht) = qdht(radius.clone(), &f, 0, Axis(0));
    let transformer = HankelTransform::new_from_r_grid(0, radius);
    let f_t = (-a.powi(2) * transformer.radius().powi(2)).exp();
    let standard_ht = transformer.qdht(&f_t, Axis(0));
    assert_arrays_equal(&one_shot_ht, &standard_ht, 1e-4, 1e-3);
}

#[rstest]
fn test_1_over_r2_plus_z2_equivalence(#[values(2.0, 1.0, 0.1)] a: f64) {
    let r = Array1::linspace(0.0, 50.0, 1024);
    let f = 1.0 / (r.powi(2) + a.powi(2));

    let transformer = HankelTransform::new_from_r_grid(0, r.clone());

    let f_transformer = 1.0 / (transformer.radius().powi(2) + a.powi(2));
    assert_arrays_equal(
        &transformer.to_transform_r(&f).unwrap(),
        &f_transformer,
        1e-6,
        1e-2,
    );
    let (kr, one_shot_ht) = qdht(r, &f, 0, Axis(0));
    assert_arrays_equal(&kr, transformer.kr(), 1e-8, 1e-5);
    let standard_ht = transformer.qdht(&f_transformer, Axis(0));
    assert_arrays_equal(&one_shot_ht, &standard_ht, 1e-2, 1e-3);
}
