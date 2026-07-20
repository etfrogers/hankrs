use std::f64::consts::PI;

use super::{spherical_jn, spherical_jn_zeros};
use approx::assert_relative_eq;
use ndarray::{Array1, array};
use rstest::rstest;

#[rstest]
#[allow(clippy::approx_constant)]
fn test_jn_spherical_zeros() {
    let order = 0;
    // zeros are n*pi for n = 0
    let zs = spherical_jn_zeros(order, 10);
    assert_relative_eq!(zs, PI * Array1::range(1.0, 11.0, 1.0));

    // https://www.researchgate.net/figure/Zeros-of-the-spherical-Bessel-functions_tbl1_348819348
    // n, j0​(x),   j1​(x),   j2​(x),   j3​(x),   j4​(x)
    // 1, 3.14159, 4.49341, 5.76346, 6.98793, 8.18256
    // 2, 6.28319, 7.72525, 9.09501, 10.4171, 11.7049
    // 3, 9.42478, 10.9041, 12.3229, 13.6980, 15.0397
    // 4, 12.5664, 14.0662, 15.5146, 16.9236, 18.3013
    // 5, 15.7080, 17.2208, 18.6890, 20.1218, 21.5254

    let expected_zeros = array![
        [3.14159, 4.49341, 5.76346, 6.98793, 8.18256],
        [6.28319, 7.72525, 9.09501, 10.4171, 11.7049],
        [9.42478, 10.9041, 12.3229, 13.6980, 15.0397],
        [12.5664, 14.0662, 15.5146, 16.9236, 18.3013],
        [15.7080, 17.2208, 18.6890, 20.1218, 21.5254],
    ];

    for order in 0..5i32 {
        let zs = spherical_jn_zeros(order, 5);
        assert_relative_eq!(zs, expected_zeros.column(order as usize), epsilon = 1e-4);
    }

    // test that the zeros are actually zeros of the spherical Bessel function
    for order in 0..10 {
        let zs = spherical_jn_zeros(order, 10);
        for z in zs.iter() {
            assert_relative_eq!(spherical_jn(order as f64, *z), 0.0, epsilon = 2e-12);
        }
    }
}
