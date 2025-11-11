use std::f64::consts::PI;

use amos_bessel_rs::{BesselResult, bessel_j, bessel_y};
use conv::ConvUtil;
use ndarray::Array1;
use num_complex::{Complex64, ComplexFloat};

#[cfg(test)]
mod test;

#[derive(Debug, PartialEq)]
pub enum BesselFunType {
    J,
    Y,
    JP,
    YP,
}

impl BesselFunType {
    fn is_non_derivative(&self) -> bool {
        return *self == BesselFunType::J || *self == BesselFunType::Y;
    }
}

//BESSEL_ZEROS: Finds the first n zeros of a bessel function
//
//	z = bessel_zeros(d, a, n, e)
//
//	z	=	zeros of the bessel function
//	d	=	Bessel function type:
//			1:	Ja
//			2:	Ya
//			3:	Ja'
//			4:	Ya'
//	a	=	Bessel order (a>=0)
//	n	=	Number of zeros to find
//	e	=	Relative error in root
//
//	This function uses the routine described in:
//		"An Algorithm with ALGOL 60 Program for the Computation of the
//		zeros of the Ordinary Bessel Functions and those of their
//		Derivatives".
//		N. M. Temme
//		Journal of Computational Physics, 32, 270-279 (1979)
//
// Translated from Adam Wyatt's Matlab version
pub fn bessel_zeros(
    func_type: &BesselFunType,
    order: i32,
    n_zeros: usize,
    precision: f64,
) -> Array1<f64> {
    let a: f64 = order.into();
    let mut z = Array1::zeros(n_zeros);

    let aa = a.powf(2.0);
    let mu = 4.0 * aa;
    let mu2 = mu.powf(2.0);
    let mu3 = mu.powf(3.0);
    let mu4 = mu.powf(4.0);

    let mut p: f64;
    let p0: f64;
    let p1: f64;
    let q1: f64;
    if func_type.is_non_derivative() {
        p = 7.0 * mu - 31.0;
        p0 = mu - 1.0;

        if (1.0 + p) == p {
            p1 = 0.0;
            q1 = 0.0;
        } else {
            p1 = 4.0 * (253.0 * mu2 - 3722.0 * mu + 17869.0) * p0 / (15.0 * p);
            q1 = 1.6 * (83.0 * mu2 - 982.0 * mu + 3779.0) / p;
        }
    } else {
        p = 7.0 * mu2 + 82.0 * mu - 9.0;
        p0 = mu + 3.0;
        if (p + 1.0) == 1.0 {
            p1 = 0.0;
            q1 = 0.0;
        } else {
            p1 = (4048.0 * mu4 + 131264.0 * mu3 - 221984.0 * mu2 - 417600.0 * mu + 1012176.0)
                / (60.0 * p);
            q1 = 1.6 * (83.0 * mu3 + 2075.0 * mu2 - 3039.0 * mu + 3537.0) / p;
        }
    }

    let t = if (*func_type == BesselFunType::J) || (*func_type == BesselFunType::YP) {
        0.25
    } else {
        0.75
    };

    let tt = 4.0 * t;

    let (pp1, qq1) = if func_type.is_non_derivative() {
        (5. / 48., -5. / 36.)
    } else {
        (-7. / 48., 35. / 288.)
    };

    let y = 0.375 * PI;
    let bb = if a >= 3.0 { a.powf(-2.0 / 3.0) } else { 1.0 };

    let a1 = 3 * order - 8;
    // psi = (.5*a + .25)*PI;

    for s in 1..=n_zeros {
        let sf: f64 = s.value_as().unwrap();
        let mut x: f64;
        let mut w: f64 = 0.0;
        let mut j: i32;
        if (order == 0) && (s == 1) && (*func_type == BesselFunType::JP) {
            x = 0.0;
            j = 0;
        } else {
            if TryInto::<i32>::try_into(s).unwrap() >= a1 {
                let b = (sf + 0.5 * a - t) * PI;
                let c = 0.015625 / (b.powf(2.0));
                x = b - 0.125 * (p0 - p1 * c) / (b * (1.0 - q1 * c));
            } else {
                if s == 1 {
                    x = match func_type {
                        BesselFunType::J => -2.33811,
                        BesselFunType::Y => -1.17371,
                        BesselFunType::JP => -1.01879,
                        BesselFunType::YP => -2.29444,
                    };
                } else {
                    x = y * (4.0 * sf - tt);
                    let v = x.powf(-2.0);
                    x = -(x.powf(2.0 / 3.0)) * (1.0 + v * (pp1 + qq1 * v));
                }
                let u = x * bb;
                let v = fi(2.0 / 3.0 * (-u).powf(1.5));
                w = 1.0 / v.cos();
                let xx = 1.0 - w.powf(2.0);
                let c = (u / xx).sqrt();
                x = if func_type.is_non_derivative() {
                    w * (a + c * (-5.0 / u - c * (6.0 - 10.0 / xx)) / (48.0 * a * u))
                } else {
                    w * (a + c * (7.0 / u + c * (18.0 - 14.0 / xx)) / (48.0 * a * u))
                }
            }

            j = 0;
            while (j == 0) || ((j < 5) && ((w / x).abs() > precision)) {
                let xx = x.powf(2.0);
                let x4 = x.powf(4.0);
                let a2 = aa - xx;
                let r0 = bessr(&func_type, order, x);
                j = j + 1;
                let q: f64;
                let u: f64;
                if func_type.is_non_derivative() {
                    u = r0;
                    w = 6.0 * x * (2.0 * a + 1.0);
                    p = (1.0 - 4.0 * a2) / w;
                    q = (4.0 * (xx - mu) - 2.0 - 12.0 * a) / w;
                } else {
                    u = -xx * r0 / a2;
                    let v = 2.0 * x * a2 / (3.0 * (aa + xx));
                    w = 64.0 * a2.powf(3.0);
                    q = 2.0 * v * (1.0 + mu2 + 32.0 * mu * xx + 48.0 * x4) / w;
                    p = v * (1.0 + (40.0 * mu * xx + 48.0 * x4 - mu2) / w);
                }
                w = u * (1.0 + p * r0) / (1.0 + q * r0);
                x = x + w;
            }
            z[s - 1] = x;
        }
    }
    return z;
}

fn fi(y: f64) -> f64 {
    let c1 = 1.570796;
    if y == 0.0 {
        0.0
    } else if y > 1e5 {
        c1
    } else {
        let mut p: f64;
        if y < 1.0 {
            p = (3.0 * y).powf(1.0 / 3.0);
            let pp = p.powf(2.0);
            p *= 1.0 + pp * (pp * (27.0 - 2.0 * pp) - 210.0) / 1575.0;
        } else {
            p = 1.0 / (y + c1);
            let pp = p.powf(2.0);
            p = c1
                - p * (1.0
                    + pp * (2310.0 + pp * (3003.0 + pp * (4818.0 + pp * (8591.0 + pp * 16328.0))))
                        / 3465.0);
        }
        let pp = (y + p).powf(2.0);
        let r = (p - (p + y).atan()) / pp;
        p - (1.0 + pp) * r * (1.0 + r / (p + y))
    }
}

fn bessr(fun_type: &BesselFunType, order: i32, x: f64) -> BesselResult<f64> {
    let z = Complex64::new(x, 0.0);
    let order_f: f64 = order.into();
    match fun_type {
        BesselFunType::J => {
            let a = bessel_j(order_f, x)?;
            let b = bessel_j(order_f + 1.0, x)?;
            a / b
        }
        BesselFunType::Y => (bessel_y(order_f, z) / bessel_y(order_f + 1.0, z)).re(),
        BesselFunType::JP => a / x - bessel_j(order_f + 1.0, x)? / bessel_j(order_f, x),
        BesselFunType::YP => (a / x - bessel_y(order_f + 1.0, z)? / bessel_y(order_f, z)?).re(),
    }
}
