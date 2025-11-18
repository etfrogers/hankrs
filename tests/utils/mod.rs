use std::f64::{INFINITY, consts::PI};

use amos_bessel_rs::bessel_j;
use approx::{AbsDiffEq, RelativeEq, assert_relative_eq, relative_eq};
use ndarray::{Array1, s};

// ----------------
// HELPER FUNCTIONS
// ----------------

#[derive(Debug, PartialEq)]
pub struct _Array1Comp(pub Array1<f64>);

impl AbsDiffEq for _Array1Comp {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            if !a.abs_diff_eq(b, epsilon) {
                return false;
            }
        }
        true
    }
}

impl RelativeEq for _Array1Comp {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            if !a.relative_eq(b, epsilon, max_relative) {
                return false;
            }
        }
        true
    }
}

pub fn assert_relative_eq_with_end_points(
    expected: Array1<f64>,
    actual: Array1<f64>,
    max_rel_body: f64,
    max_rel_end: f64,
    eps_body: f64,
    eps_end: f64,
) {
    let n = expected.len();
    assert_relative_eq!(
        expected[0],
        actual[0],
        epsilon = eps_end,
        max_relative = max_rel_end
    );

    assert_relative_eq!(
        expected[n - 1],
        actual[n - 1],
        epsilon = eps_end,
        max_relative = max_rel_end
    );

    assert_arrays_equal(
        expected.slice(s![1..n - 2]).as_slice().unwrap(),
        actual.slice(s![1..n - 2]).as_slice().unwrap(),
        eps_body,
        max_rel_body,
    );
}

fn abs_rel_errors(a: f64, b: f64) -> (f64, f64) {
    let abs_e = (a - b).abs();
    let rel_e = abs_e / a.abs().max(b.abs());
    (abs_e, rel_e)
}

pub(crate) fn assert_arrays_equal(actual: &[f64], expected: &[f64], eps: f64, max_rel: f64) {
    let actual = actual.to_vec();
    let expected = expected.to_vec();
    // let reference = reference.clone().into_vec();
    // let exp_error = MACHINE_CONSTANTS.abs_error_tolerance;

    for (i, (&act, exp)) in actual.iter().zip(expected).enumerate() {
        // let ref_val = reference.get(i);
        // let tolerances = Tolerances::new(act, exp, ref_val, exp_error);
        if !relative_eq!(act, exp, epsilon = eps, max_relative = max_rel) {
            let (actual_error, relative_error) = abs_rel_errors(act, exp);
            panic!(
                "Failed on matching values at index {i}\n\
                Actual: {act:e}\n\
                Expected: {exp:e}\n\
                \n\
                Absolute tolerance: {eps:e}\n\
                Relative tolerance: {max_rel:e}\n\
                Absolute error: {actual_error:e}\n\
                Relative error: {relative_error:e}\n\
                "
            );
        };
    }
}

// ---------------
// MATHS FUNCTIONS
// ----------------
pub fn generalised_top_hat(r: &Array1<f64>, a: f64, p: i32) -> Array1<f64> {
    r.mapv(|r| generalised_top_hat_f(r, a, p))
    // f := utils.ApplyVec(func(val f64) f64 { return generalisedTopHatF(val, a, p) }, nil, r)
    // return f
}

fn generalised_top_hat_f(r: f64, a: f64, p: i32) -> f64 {
    // var val f64
    if r <= a {
        r.powi(p) //math.Pow(r, f64(p))
    } else {
        0.0
    }
    // othwerise 0

    // return val
}

pub fn generalised_jinc(v: &Array1<f64>, a: f64, p: i32) -> Array1<f64> {
    v.mapv(|v| generalised_jinc_f(v, a, p))

    // f := utils.ApplyVec(func(val f64) f64 { return generalisedJincF(val, a, p) }, nil, v)
    // return f
}

fn generalised_jinc_f(v: f64, a: f64, p: i32) -> f64 {
    // var val f64
    if v == 0. {
        match p {
            -1 => INFINITY,
            -2 => -PI,
            0 => PI * a.powi(2),
            _ => 0.0,
        }
    } else {
        let prefactor = a.powi(p + 1);
        let x = 2.0 * PI * a * v;
        let j = bessel_j(p + 1, x).unwrap();
        prefactor * j / v
    }
}
