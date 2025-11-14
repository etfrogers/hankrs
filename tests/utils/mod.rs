use approx::{AbsDiffEq, RelativeEq, assert_relative_eq};
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

    assert_relative_eq!(
        expected.slice(s![1..n - 2]).as_slice().unwrap(),
        actual.slice(s![1..n - 2]).as_slice().unwrap(),
        epsilon = eps_body,
        max_relative = max_rel_body
    );
}
/*
// ---------------
// MATHS FUNCTIONS
// ----------------
func GeneralisedTopHat(r mat.Vector, a float64, p int) mat.Vector {
    f := utils.ApplyVec(func(val float64) float64 { return generalisedTopHatF(val, a, p) }, nil, r)
    return f
}

func generalisedTopHatF(r float64, a float64, p int) float64 {
    var val float64
    if r <= a {
        val = math.Pow(r, float64(p))
    }
    // othwerise 0

    return val
}

func GeneralisedJinc(v mat.Vector, a float64, p int) mat.Vector {
    f := utils.ApplyVec(func(val float64) float64 { return generalisedJincF(val, a, p) }, nil, v)
    return f
}

func generalisedJincF(v float64, a float64, p int) float64 {

    var val float64
    if v == 0. {
        switch {
        case p == -1:
            val = math.Inf(1)
        case p == -2:
            val = -math.Pi
        case p == 0:
            val = math.Pi * math.Pow(a, 2)
        default:
            val = 0
        }
    } else {
        prefactor := math.Pow(a, float64(p+1))
        x := 2 * math.Pi * a * v
        j := math.Jn(p+1, x)
        val = prefactor * j / v
    }

    return val
}
*/
