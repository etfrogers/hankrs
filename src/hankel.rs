use approx::{AbsDiffEq, RelativeEq};
use ndarray::Zip;
use ndarray::parallel::prelude::*;
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView, ArrayView2, Axis, Data, Dim, DimAdd, Dimension,
    Ix1, IxDyn, RemoveAxis, s,
};
use ndarray_interp::interp1d::{Interp1DBuilder, cubic_spline::CubicSpline};
use ndarray_stats::QuantileExt;
use rayon::prelude::*;
use roots::{SimpleConvergency, find_root_brent};
use std::f64::consts::FRAC_PI_2;
use std::fmt::Display;
use std::{f64::consts::PI, fmt::Debug};
use thiserror::Error;

use amos_bessel_rs::bessel_j;
use bessel_zeros::{BesselFunType, fast::bessel_zeros};
use ndarray::ArrayView1;
use num::Zero;
use num_complex::Complex;
use real_bessel::jn as bessel_j_real;

/// A trait for scalar types that can be processed by the Hankel transform.
/// It abstracts over basic array arithmetic and matrix multiplications.
pub trait HankelScalar: Clone + Zero + Send + Sync {
    /// Multiplies a purely real transform matrix with a vector of this scalar type.
    fn dot_real_matrix(matrix: ArrayView2<f64>, vector: ArrayView1<Self>) -> Array1<Self>;

    /// Divides a vector of this scalar type by a purely real vector.
    fn div_real_array(vector: ArrayView1<Self>, scale: ArrayView1<f64>) -> Array1<Self>;

    /// Multiplies a mutable vector of this scalar type in-place by a purely real vector.
    fn mul_real_array_assign(vector: &mut Array1<Self>, scale: ArrayView1<f64>);

    /// Interpolates the array along the specified axis using a cubic spline.
    fn spline<D>(
        x0: ArrayView1<f64>,
        y0: ArrayView<Self, D>,
        x: ArrayView1<f64>,
        axis: Axis,
    ) -> Array<Self, D>
    where
        D: Dimension + RemoveAxis,
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>;
}

/// Implementation of [`HankelScalar`] for real, 64-bit floating point numbers (`f64`).
/// This allows `HankelTransform` methods to be called directly on purely real arrays.
impl HankelScalar for f64 {
    fn dot_real_matrix(matrix: ArrayView2<f64>, vector: ArrayView1<f64>) -> Array1<f64> {
        matrix.dot(&vector)
    }
    fn div_real_array(vector: ArrayView1<f64>, scale: ArrayView1<f64>) -> Array1<f64> {
        vector.to_owned() / scale
    }
    fn mul_real_array_assign(vector: &mut Array1<f64>, scale: ArrayView1<f64>) {
        *vector *= &scale;
    }
    fn spline<D>(
        x0: ArrayView1<f64>,
        y0: ArrayView<f64, D>,
        x: ArrayView1<f64>,
        axis: Axis,
    ) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    {
        spline_f64(x0, y0, x, axis)
    }
}

/// Implementation of [`HankelScalar`] for complex, 64-bit floating point numbers (`Complex<f64>`).
/// This allows `HankelTransform` methods to be called directly on complex arrays. Real and
/// imaginary parts are processed through the underlying transform operations seamlessly.
impl HankelScalar for Complex<f64> {
    fn dot_real_matrix(
        matrix: ArrayView2<f64>,
        vector: ArrayView1<Complex<f64>>,
    ) -> Array1<Complex<f64>> {
        let real_part = matrix.dot(&vector.mapv(|c| c.re));
        let imag_part = matrix.dot(&vector.mapv(|c| c.im));
        ndarray::Zip::from(&real_part)
            .and(&imag_part)
            .map_collect(|&r, &i| Complex::new(r, i))
    }
    fn div_real_array(
        vector: ArrayView1<Complex<f64>>,
        scale: ArrayView1<f64>,
    ) -> Array1<Complex<f64>> {
        ndarray::Zip::from(vector)
            .and(scale)
            .map_collect(|&v, &s| v / s)
    }
    fn mul_real_array_assign(vector: &mut Array1<Complex<f64>>, scale: ArrayView1<f64>) {
        ndarray::Zip::from(vector)
            .and(scale)
            .for_each(|v, &s| *v *= s);
    }
    fn spline<D>(
        x0: ArrayView1<f64>,
        y0: ArrayView<Complex<f64>, D>,
        x: ArrayView1<f64>,
        axis: Axis,
    ) -> Array<Complex<f64>, D>
    where
        D: Dimension + RemoveAxis,
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    {
        let real_part = spline_f64(x0, y0.mapv(|c| c.re).view(), x, axis);
        let imag_part = spline_f64(x0, y0.mapv(|c| c.im).view(), x, axis);
        ndarray::Zip::from(&real_part)
            .and(&imag_part)
            .map_collect(|&r, &i| Complex::new(r, i))
    }
}

/// The main struct for performing Hankel Transforms
///
/// This struct computes the quasi-discrete approximation of the continuous Hankel transform of order p:
///
/// `H_p{ f(r) } = Integral[ f(r) * J_p(k * r) * r dr ]` from `r = 0` to `infinity`
///
/// For the QDHT to work, the function must be sampled at specific points, which this struct generates
/// and stores in `self.r`. Any transform on this grid will be sampled at points
/// `self.v` (frequency space) or equivalently `self.kr`
/// (angular frequency or wavenumber space).
///
/// ## Native Complex Support
/// Because `HankelTransform` accepts arrays parameterized by `T: HankelScalar`, you can seamlessly
/// transform complex-valued arrays (e.g. `Array1<Complex64>`) without splitting them into real
/// and imaginary parts manually.
///
/// ## Examples
/// ```rust
/// # extern crate blas_src;
/// use hankrs::HankelTransform;
/// use ndarray::{Array1, Axis};
///
/// let transformer = HankelTransform::new(0, 10.0, 256);
/// let r = transformer.radius();
/// let f = r.mapv(|rad| (-rad * rad).exp());
///
/// // Perform the quasi-discrete Hankel transform
/// let ht = transformer.qdht(&f, Axis(0));
/// ```
///
/// The constructor has one required argument (`order`). The remaining arguments offer
/// three different ways of specifying the radial (and therefore implicitly the frequency) points:
///
/// 1. Supply both a maximum radius `max_radius` and number of transform points `n_points`
/// 2. Supply the original (often equally spaced) `radial_grid` on which you currently
///    have sample points. This approach allows easy conversion from the original grid using
///    [`HankelTransform::to_transform_r`]. `t = HankelTransform::new_from_r_grid(order, r)`
///    is effectively equivalent to `t = HankelTransform::new(order, max_radius, n_points)`
///    except for the fact that the original radial grid is stored in the [`HankelTransform`]
///    object for use in [`HankelTransform::to_transform_r`] and
///    [`HankelTransform::to_original_r`].
/// 3. Supply the original (often equally spaced) `k`-space grid on which you
///    currently have sample points. This is most useful if you intend to do inverse
///    transforms. It allows easy conversion to and from the original grid using
///    [`HankelTransform::to_original_k`] and [`HankelTransform::to_transform_k`].
///    As in option 2, `n_points` is determined by `k_grid.len()`.
///    `max_radius` is determined in a more complex way from the maximum of `k_grid`.
///
/// The algorithm used is that from:
///
/// > *"Computation of quasi-discrete Hankel transforms of the integer
/// > order for propagating optical wave fields"*
/// > Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
/// > J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
///
/// The algorithm also uses root finding to calculate the roots of the bessel function.
#[derive(PartialEq)]
pub struct HankelTransform {
    /// Transform order `p`
    order: i32,
    /// Number of sample points `N`
    n_points: usize,
    /// Radial extent of transform `r_max`
    max_radius: f64,
    /// Frequency extent of transform
    max_v: f64,
    /// wavenumber extent of transform
    max_kr: f64,
    /// Original radial grid on which sample points were provided
    original_radial_grid: Option<Array1<f64>>,
    /// Original `k`-space grid on which sample points were provided
    original_k_grid: Option<Array1<f64>>,
    /// Radial co-ordinate vector
    r: Array1<f64>,
    /// Radial wave number co-ordinate vector
    kr: Array1<f64>,
    /// Frequency co-ordinate vector
    v: Array1<f64>,
    /// Transform matrix
    t: Array2<f64>,
    /// Frequency transform vector `J_V = J_{p+1}(\alpha) / v_{max}`
    jv: Array1<f64>,
    /// Radius transform vector `J_R = J_{p+1}(\alpha) / r_max`
    jr: Array1<f64>,
}

impl Debug for HankelTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HankelTransform")
            .field("order", &self.order)
            .field("n_points", &self.n_points)
            .field("max_radius", &self.max_radius)
            .finish()
    }
}

impl AbsDiffEq for HankelTransform {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.relative_eq(other, epsilon, Self::default_max_relative())
    }
}

impl RelativeEq for HankelTransform {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.order == other.order
            && self.n_points == other.n_points
            && self
                .max_radius
                .relative_eq(&other.max_radius, epsilon, max_relative)
            && self.r.relative_eq(&other.r, epsilon, max_relative)
            && self.kr.relative_eq(&other.kr, epsilon, max_relative)
            && self.v.relative_eq(&other.v, epsilon, max_relative)
            && self.t.relative_eq(&other.t, epsilon, max_relative)
            && self.jr.relative_eq(&other.jr, epsilon, max_relative)
            && self.jv.relative_eq(&other.jv, epsilon, max_relative)
    }
}

impl HankelTransform {
    /// Create a new `HankelTransform` by explicitly specifying the maximum radius and number of points.
    ///
    /// # Arguments
    /// * `order` - Transform order `p`.
    /// * `max_radius` - Radial extent of the transform `r_max`.
    /// * `n_points` - Number of sample points `N`.
    pub fn new(order: i32, max_radius: f64, n_points: usize) -> Self {
        Self::build(
            order,
            n_points,
            Some(max_radius),
            None,
            None,
            TransformType::Polar,
        )
    }

    /// Create a new `HankelTransform` from an existing radial grid.
    ///
    /// This uses the length of the grid as the number of points, and the maximum value
    /// of the grid as the maximum radius.
    ///
    /// # Arguments
    /// * `order` - Transform order `p`.
    /// * `radial_grid` - The radial grid that will be used to sample input functions.
    pub fn new_from_r_grid(order: i32, radial_grid: Array1<f64>) -> HankelTransform {
        Self::build(
            order,
            radial_grid.len(),
            None,
            Some(radial_grid),
            None,
            TransformType::Polar,
        )
    }

    /// Create a new `HankelTransform` from an existing `k`-space grid.
    ///
    /// This uses the length of the grid as the number of points. The maximum radius
    /// is determined dynamically from the maximum value of the `k`-grid.
    ///
    /// # Arguments
    /// * `order` - Transform order `p`.
    /// * `k_grid` - The `k`-space grid that will be used to sample input functions.
    pub fn new_from_k_grid(order: i32, k_grid: Array1<f64>) -> Self {
        Self::build(
            order,
            k_grid.len(),
            None,
            None,
            Some(k_grid),
            TransformType::Polar,
        )
    }

    /// Create a new spherical `HankelTransform` by explicitly specifying the maximum radius and number of points.
    ///
    /// # Arguments
    /// * `order` - Transform order `p`.
    /// * `max_radius` - Radial extent of the transform `r_max`.
    /// * `n_points` - Number of sample points `N`.
    ///
    /// # See Also
    /// The [online `hankrs` book](https://etfrogers.github.io/hankrs/spherical_known_transforms.html)
    /// gives details of and demonstrates verified transform pairs (Gaussian and top-hat) for the spherical QDHT.
    pub fn new_spherical(order: i32, max_radius: f64, n_points: usize) -> Self {
        Self::build(
            order,
            n_points,
            Some(max_radius),
            None,
            None,
            TransformType::Spherical,
        )
    }

    /// Create a new spherical `HankelTransform` from an existing radial grid.
    ///
    /// This uses the length of the grid as the number of points, and the maximum value
    /// of the grid as the maximum radius.
    ///
    /// # Arguments
    /// * `order` - Transform order `p`.
    /// * `radial_grid` - The radial grid that will be used to sample input functions.
    ///
    /// # See Also
    /// The [online `hankrs` book](https://etfrogers.github.io/hankrs/spherical_known_transforms.html)
    /// gives details of and demonstrates verified transform pairs (Gaussian and top-hat) for the spherical QDHT.
    pub fn new_spherical_from_r_grid(order: i32, radial_grid: Array1<f64>) -> HankelTransform {
        Self::build(
            order,
            radial_grid.len(),
            None,
            Some(radial_grid),
            None,
            TransformType::Spherical,
        )
    }

    /// Create a new spherical `HankelTransform` from an existing `k`-space grid.
    ///
    /// This uses the length of the grid as the number of points. The maximum radius
    /// is determined dynamically from the maximum value of the `k`-grid.
    ///
    /// # Arguments
    /// * `order` - Transform order `p`.
    /// * `k_grid` - The `k`-space grid that will be used to sample input functions.
    ///
    /// # See Also
    /// The [online `hankrs` book](https://etfrogers.github.io/hankrs/spherical_known_transforms.html)
    /// gives details of and demonstrates verified transform pairs (Gaussian and top-hat) for the spherical QDHT.
    pub fn new_spherical_from_k_grid(order: i32, k_grid: Array1<f64>) -> Self {
        Self::build(
            order,
            k_grid.len(),
            None,
            None,
            Some(k_grid),
            TransformType::Spherical,
        )
    }

    fn build(
        order: i32,
        n_points: usize,
        max_radius: Option<f64>,
        original_radial_grid: Option<Array1<f64>>,
        original_k_grid: Option<Array1<f64>>,
        transform_type: TransformType,
    ) -> Self {
        let zero_fun: fn(i32, usize) -> Array1<f64> = match transform_type {
            TransformType::Polar => |order: i32, n_points| {
                Array1::from_vec(bessel_zeros(BesselFunType::J, order, n_points, 1e-6))
            },
            TransformType::Spherical => {
                |order: i32, n_points: usize| spherical_jn_zeros(order, n_points)
            }
        };
        // Calculate N+1 roots must be calculated before max_radius can be derived from k_grid
        let alpha = zero_fun(order, n_points + 1);

        let alpha_n1 = alpha[n_points];
        let alpha = alpha.slice(s![0..n_points]).to_owned();

        let max_radius = match (max_radius, &original_radial_grid, &original_k_grid) {
            (Some(mr), None, None) => mr,
            (None, Some(rg), None) => *rg.max().unwrap(),
            (None, None, Some(kg)) => {
                let v_max = kg.max().unwrap() / (2.0 * PI);
                alpha_n1 / (2.0 * PI * v_max)
            }
            _ => unreachable!(
                "Invaritant violated: exactly one of max_radius, original_radial_grid, or original_k_grid must be supplied"
            ),
        };
        // Calculate co-ordinate vectors
        let r = alpha.clone() * max_radius / alpha_n1;
        let v = alpha.clone() / (2.0 * PI * max_radius);
        let kr = 2.0 * PI * v.clone();
        let max_v = alpha_n1 / (2.0 * PI * max_radius);
        let max_kr = 2.0 * PI * max_v;
        let s = alpha_n1;

        let (jp1, jr, jv): (Array1<_>, Array1<_>, Array1<_>) = match transform_type {
            TransformType::Polar => {
                let jp1 = alpha.map(|a| bessel_j_real(order + 1, *a).abs());
                (jp1.clone(), jp1.clone() / max_radius, jp1 / max_v)
            }
            TransformType::Spherical => {
                let jp1 = alpha.map(|a| spherical_jn((order + 1) as f64, *a).abs());
                (
                    jp1.clone(),
                    jp1.clone() / max_radius,
                    jp1 * (max_radius.powi(2) * (PI / (2.0 * s.powi(3))).sqrt()),
                )
            }
        };

        let mut t = Array2::<f64>::zeros((n_points, n_points));
        Zip::indexed(&mut t).par_for_each(|(i, j), t_val| {
            // Only evaluate the expensive Bessel function for the upper triangle and diagonal
            if i <= j {
                let a_i = alpha[i];
                let a_j = alpha[j];
                let jp1_i = jp1[i];
                let jp1_j = jp1[j];

                let jp_val = match transform_type {
                    TransformType::Polar => bessel_j_real(order, (a_i * a_j) / s),
                    TransformType::Spherical => {
                        spherical_jn(order as f64, (a_i * a_j) / s) / (2.0 * n_points as f64).sqrt()
                    }
                };
                // Write directly into the final T matrix
                *t_val = 2.0 * jp_val / (jp1_i * jp1_j * s);
            }
        });
        // MIRROR to the lower triangle sequentially
        // Memory copying is instantaneous compared to evaluating the AMOS library
        for i in 0..n_points {
            for j in 0..i {
                t[[i, j]] = t[[j, i]];
            }
        }

        Self {
            order,
            n_points,
            max_radius,
            max_v,
            max_kr,
            original_radial_grid,
            original_k_grid,
            r,
            kr,
            v,
            t,
            jr,
            jv,
        }
    }

    /// Returns the order `p` of the transform.
    pub fn order(&self) -> i32 {
        self.order
    }

    /// Returns the maximum radius `r_max` of the transform.
    pub fn max_radius(&self) -> f64 {
        self.max_radius
    }

    /// Returns the maximum radius `r_max` of the transform.
    pub fn max_kr(&self) -> f64 {
        self.max_kr
    }

    /// Returns the maximum radius `r_max` of the transform.
    pub fn max_frequency(&self) -> f64 {
        self.max_v
    }

    /// Returns the number of sample points `N`.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Return the original radial grid used to construct the object, or `None`
    /// if the constructor was not called specifying a `radial_grid` parameter.
    ///
    /// # Returns
    /// An `Option` containing a view of the original radial grid used to construct the object.
    pub fn original_radial_grid<'a>(&'a self) -> Option<ArrayView1<'a, f64>> {
        self.original_radial_grid.as_ref().map(|a| a.view())
    }

    /// Return the original k grid used to construct the object, or `None`
    /// if the constructor was not called specifying a `k_grid` parameter.
    ///
    /// # Returns
    /// An `Option` containing a view of the original k grid used to construct the object.
    pub fn original_k_grid<'a>(&'a self) -> Option<ArrayView1<'a, f64>> {
        self.original_k_grid.as_ref().map(|a| a.view())
    }

    /// Interpolate a function, assumed to have been given at the original radial
    /// grid points used to construct the [`HankelTransform`] object onto the grid required
    /// for use in the QDHT algorithm.
    ///
    /// If the [`HankelTransform`] object was constructed with a (say) equally-spaced
    /// grid in radius, then it needs the function to transform to be sampled at a specific
    /// grid before it can be passed to [`HankelTransform::qdht`]. This method provides
    /// a convenient way of doing this.
    ///
    /// # Arguments
    /// * `function` - The function to be interpolated. Specified at the radial points
    ///   from [`HankelTransform::original_radial_grid`].
    /// * `axis` - Axis representing the radial dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function suitable for passing to
    /// [`HankelTransform::qdht`] (sampled at `self.r`).
    pub fn to_transform_r<T: HankelScalar, S: Data<Elem = T>>(
        &self,
        function: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<T>, InterpError> {
        self.to_transform_r_nd(function, Axis(0))
    }

    /// Multi-dimensional equivalent of [`HankelTransform::to_transform_r`].
    ///
    /// Interpolates an N-dimensional function, assumed to have been given at the original radial
    /// grid points, onto the grid required for use in the QDHT algorithm along a specified axis.
    ///
    /// # Arguments
    /// * `function` - The N-dimensional function to be interpolated.
    /// * `axis` - Axis representing the radial dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function suitable for passing to [`HankelTransform::qdht`].
    pub fn to_transform_r_nd<T: HankelScalar, D: Dimension + RemoveAxis, S: Data<Elem = T>>(
        &self,
        function: &ArrayBase<S, D>,
        axis: Axis,
    ) -> Result<Array<T, D>, InterpError>
    where
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    {
        if let Some(r_grid) = self.original_radial_grid() {
            Ok(T::spline(r_grid, function.view(), self.r.view(), axis))
        } else {
            Err(InterpError {
                message: "Attempted to interpolate onto transform radial grid on HankelTransform \
                    object that was not constructed with a radial grid"
                    .to_string(),
            })
        }
    }

    /// Interpolate a function, assumed to have been given at the Hankel transform points
    /// `self.r` (as returned by [`HankelTransform::iqdht`]) back onto the original grid
    /// used to construct the [`HankelTransform`] object.
    ///
    /// If the [`HankelTransform`] object was constructed with a (say) equally-spaced
    /// grid in radius, it may be useful to convert back to this grid after an IQDHT.
    /// This method provides a convenient way of doing this.
    ///
    /// # Arguments
    /// * `function` - The function to be interpolated. Specified at the radial points
    ///   `self.r`.
    /// * `axis` - Axis representing the radial dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function at the points held in [`HankelTransform::original_radial_grid`].
    pub fn to_original_r<T: HankelScalar, S: Data<Elem = T>>(
        &self,
        function: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<T>, InterpError> {
        self.to_original_r_nd(function, Axis(0))
    }

    /// Multi-dimensional equivalent of [`HankelTransform::to_original_r`].
    ///
    /// Interpolates an N-dimensional function, assumed to have been given at the Hankel transform points,
    /// back onto the original radial grid used to construct the object along a specified axis.
    ///
    /// # Arguments
    /// * `function` - The N-dimensional function to be interpolated.
    /// * `axis` - Axis representing the radial dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function at the points held in [`HankelTransform::original_radial_grid`].
    pub fn to_original_r_nd<T: HankelScalar, S, D>(
        &self,
        function: &ArrayBase<S, D>,
        axis: Axis,
    ) -> Result<Array<T, D>, InterpError>
    where
        D: Dimension + RemoveAxis,
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
        S: Data<Elem = T>,
    {
        if let Some(r_grid) = self.original_radial_grid() {
            Ok(T::spline(self.r.view(), function.view(), r_grid, axis))
        } else {
            Err(InterpError {
                message: "Attempted to interpolate onto original_radial_grid on HankelTransform \
                    object that was not constructed with a r_grid"
                    .to_string(),
            })
        }
    }

    /// Interpolate a function, assumed to have been given at the original k
    /// grid points used to construct the [`HankelTransform`] object onto the grid required
    /// for use in the IQDHT algorithm.
    ///
    /// If the [`HankelTransform`] object was constructed with a (say) equally-spaced
    /// grid in `k`, then it needs the function to transform to be sampled at a specific
    /// grid before it can be passed to [`HankelTransform::iqdht`]. This method provides
    /// a convenient way of doing this.
    ///
    /// # Arguments
    /// * `function` - The function to be interpolated. Specified at the k points
    ///   from [`HankelTransform::original_k_grid`].
    /// * `axis` - Axis representing the frequency dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function suitable for passing to
    /// [`HankelTransform::qdht`] (sampled at `self.kr`).
    pub fn to_transform_k<T: HankelScalar, S: Data<Elem = T>>(
        &self,
        function: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<T>, InterpError> {
        self.to_transform_k_nd(function, Axis(0))
    }

    /// Multi-dimensional equivalent of [`HankelTransform::to_transform_k`].
    ///
    /// Interpolates an N-dimensional function, assumed to have been given at the original `k`-space
    /// grid points, onto the grid required for use in the IQDHT algorithm along a specified axis.
    ///
    /// # Arguments
    /// * `function` - The N-dimensional function to be interpolated.
    /// * `axis` - Axis representing the frequency dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function suitable for passing to [`HankelTransform::iqdht`].
    pub fn to_transform_k_nd<T: HankelScalar, S, D>(
        &self,
        function: &ArrayBase<S, D>,
        axis: Axis,
    ) -> Result<Array<T, D>, InterpError>
    where
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
        S: Data<Elem = T>,
        D: Dimension + RemoveAxis,
    {
        if let Some(k_grid) = self.original_k_grid() {
            Ok(T::spline(k_grid, function.view(), self.kr.view(), axis))
        } else {
            Err(InterpError {
                message: "Attempted to interpolate onto transform k grid on HankelTransform \
                    object that was not constructed with a k_grid"
                    .to_string(),
            })
        }
    }

    /// Interpolate a function, assumed to have been given at the Hankel transform points
    /// `self.k` (as returned by [`HankelTransform::qdht`]) back onto the original grid
    /// used to construct the [`HankelTransform`] object.
    ///
    /// If the [`HankelTransform`] object was constructed with a (say) equally-spaced
    /// grid in `k`, it may be useful to convert back to this grid after a QDHT.
    /// This method provides a convenient way of doing this.
    ///
    /// # Arguments
    /// * `function` - The function to be interpolated. Specified at the radial points
    ///   `self.k`.
    /// * `axis` - Axis representing the frequency dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function at the points held in [`HankelTransform::original_k_grid`].
    pub fn to_original_k<T: HankelScalar, S: Data<Elem = T>>(
        &self,
        function: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<T>, InterpError> {
        self.to_original_k_nd(function, Axis(0))
    }

    /// Multi-dimensional equivalent of [`HankelTransform::to_original_k`].
    ///
    /// Interpolates an N-dimensional function, assumed to have been given at the Hankel transform points,
    /// back onto the original `k`-space grid used to construct the object along a specified axis.
    ///
    /// # Arguments
    /// * `function` - The N-dimensional function to be interpolated.
    /// * `axis` - Axis representing the frequency dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function at the points held in [`HankelTransform::original_k_grid`].
    pub fn to_original_k_nd<T: HankelScalar, S, D>(
        &self,
        function: &ArrayBase<S, D>,
        axis: Axis,
    ) -> Result<Array<T, D>, InterpError>
    where
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
        D: Dimension + RemoveAxis,
        S: Data<Elem = T>,
    {
        if let Some(k_grid) = self.original_k_grid() {
            Ok(T::spline(self.kr.view(), function.view(), k_grid, axis))
        } else {
            Err(InterpError {
                message: "Attempted to interpolate onto original_k_grid on HankelTransform \
                    object that was not constructed with a k grid"
                    .to_string(),
            })
        }
    }

    /// QDHT: Quasi Discrete Hankel Transform
    ///
    /// Performs the Hankel transform of a function of radius, returning
    /// a function of frequency.
    ///
    /// Mathematically, it computes `F(v) = H{ f(r) }`.
    /// In terms of the discrete matrix operations, it evaluates:
    ///
    /// `F = J_V * ( T * (f / J_R) )`
    ///
    /// where `T` is the symmetric transform matrix, `J_V` and `J_R` are the scale factors.
    /// The division by `J_R` and multiplication by `J_V` are element-wise operations.
    ///
    /// # Warning
    /// The input function must be sampled at the points `self.r`, and the output
    /// will be sampled at the points `self.v` (or equivalently `self.kr`).
    ///
    /// # Arguments
    /// * `fr` - Function in real space as a function of radius (sampled at `self.r`).
    /// * `axis` - Axis over which to compute the Hankel transform.
    ///
    /// # Returns
    /// Function in frequency space (sampled at `self.v`).
    pub fn qdht<T: HankelScalar, D: Dimension + RemoveAxis, S>(
        &self,
        fr: &ArrayBase<S, D>,
        axis: Axis,
    ) -> Array<T, D>
    where
        S: Data<Elem = T>,
    {
        let scale_factor_input = self.jr.view();
        let scale_factor_output = self.jv.view();
        self.transform_by_lines(fr, axis, scale_factor_input, scale_factor_output)
    }

    /// IQDHT: Inverse Quasi Discrete Hankel Transform
    ///
    /// Performs the inverse Hankel transform of a function of frequency, returning
    /// a function of radius.
    ///
    /// Mathematically, it computes `f(r) = H^{-1}{ F(v) }`.
    /// Because the QDHT transform matrix `T` is symmetric and its own inverse, the discrete matrix operation is identical to the forward transform, but with the role of the scale factors `J_R` and `J_V` reversed:
    ///
    /// `f = J_R * ( T * (F / J_V) )`
    ///
    /// # Arguments
    /// * `fv` - Function in frequency space (sampled at `self.v`).
    /// * `axis` - Axis over which to compute the Hankel transform.
    ///
    /// # Returns
    /// Radial function (sampled at `self.r`) = IHT(fv).
    pub fn iqdht<T: HankelScalar, D: Dimension, S>(
        &self,
        fv: &ArrayBase<S, D>,
        axis: Axis,
    ) -> Array<T, D>
    where
        S: Data<Elem = T>,
    {
        self.transform_by_lines(fv, axis, self.jv.view(), self.jr.view())
    }

    fn transform_by_lines<T: HankelScalar, D: Dimension, S>(
        &self,
        f: &ArrayBase<S, D>,
        axis: Axis,
        scale_factor_input: ArrayView1<f64>,
        scale_factor_output: ArrayView1<f64>,
    ) -> Array<T, D>
    where
        S: Data<Elem = T>,
    {
        let mut transform = Array::zeros(f.dim());

        // 1. Swap into_iter() for into_par_iter() on both lanes
        // 2. Swap the for-loop for .for_each()
        transform
            .lanes_mut(axis)
            .into_iter() // 1. Start as a normal sequential iterator
            .zip(f.lanes(axis)) // 2. Zip them sequentially
            .par_bridge() // 3. MAGIC: Hand the sequential pipeline over to Rayon's thread pool
            .for_each(|(mut transform_line, fr_line)| {
                let scaled_line = T::div_real_array(fr_line, scale_factor_input);
                let mut transformed = T::dot_real_matrix(self.t.view(), scaled_line.view());
                T::mul_real_array_assign(&mut transformed, scale_factor_output);
                transform_line.assign(&transformed);
            });
        transform
    }

    /// Returns a view of the transform matrix.
    pub fn transform_matrix<'a>(&'a self) -> ArrayView2<'a, f64> {
        self.t.view()
    }

    /// Returns a view of the radial coordinate vector `r`.
    pub fn radius<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.r.view()
    }

    /// Returns a view of the frequency coordinate vector `v`.
    pub fn frequency<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.v.view()
    }

    /// Returns a view of the radial wave number coordinate vector `kr`.
    pub fn kr<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.kr.view()
    }

    /// Consumes the transform and returns the radial coordinate vector `r`.
    pub(crate) fn into_radius(self) -> Array1<f64> {
        self.r
    }

    /// Consumes the transform and returns the radial wave number coordinate vector `kr`.
    pub(crate) fn into_kr(self) -> Array1<f64> {
        self.kr
    }
}

fn perms<D: Dimension>(axis: Axis, ndim: usize) -> (D, D) {
    let mut forward_perm: Vec<usize> = (0..ndim).collect();
    forward_perm.remove(axis.index());
    forward_perm.insert(0, axis.index());

    let mut backward_perm: Vec<usize> = (0..ndim).collect();
    backward_perm.remove(0);
    backward_perm.insert(axis.index(), 0);

    let forward_dim = IxDyn(&forward_perm);
    let backward_dim = IxDyn(&backward_perm);
    let forward_d: D = D::from_dimension(&forward_dim).expect("Dimension conversion failed");
    let backward_d: D = D::from_dimension(&backward_dim).expect("Dimension conversion failed");
    (forward_d, backward_d)
}

fn spline_f64<D>(
    x0: ArrayView1<f64>,
    y0: ArrayView<f64, D>,
    x: ArrayView1<f64>,
    axis: Axis,
) -> Array<f64, D>
where
    D: Dimension + RemoveAxis,
    Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
{
    let (y0, inverse_perms) = if axis != Axis(0) {
        let (forward_perms, inverse_perms) = perms::<D>(axis, y0.ndim());
        let y0_ = y0.permuted_axes(forward_perms);

        (y0_, Some(inverse_perms))
    } else {
        (y0, None)
    };
    let interpolator = Interp1DBuilder::new(y0.into_dyn())
        .x(x0)
        .strategy(CubicSpline::new().extrapolate(true))
        .build()
        .unwrap();
    let mut result = interpolator
        .interp_array(&x)
        .unwrap()
        .into_dimensionality::<D>()
        .unwrap();
    if axis != Axis(0) {
        result = result.permuted_axes(inverse_perms.unwrap()).to_owned();
    }
    result
}

/// An error that occurs when trying to interpolate onto a transform grid (k or r)
/// on a HankelTransform object that was not constructed with a such a grid.
#[derive(Debug, Clone, Error, Default)]
pub struct InterpError {
    message: String,
}

impl Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error trying to interpolate: {}", self.message)
    }
}

enum TransformType {
    Polar,
    Spherical,
}

// adapted from SciPy Cookbook https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html
pub(super) fn spherical_jn_zeros(order: i32, n_points: usize) -> Array1<f64> {
    let mut zerosj = Array2::<f64>::zeros((order as usize + 1, n_points));
    for (i, v) in zerosj.row_mut(0).iter_mut().enumerate() {
        *v = (i as f64 + 1.0) * PI;
    }
    if order == 0 {
        return zerosj.row(0).to_owned();
    }

    let mut points = Array1::from_shape_fn(n_points + order as usize, |j| (j as f64 + 1.0) * PI);

    // Setup the exact same convergence tolerances as SciPy's default brentq
    // SciPy defaults: xtol=2e-12, maxiter=100
    let mut convergency = SimpleConvergency {
        eps: 2e-12,
        max_iter: 100,
    };

    for i in 1..=order as usize {
        for j in 0..(n_points + (order as usize) - i) {
            let a = points[j];
            let b = points[j + 1];

            // find_root_brent requires opposite signs at `a` and `b` (a valid bracket).
            let root = find_root_brent(a, b, |r| spherical_jn(i as f64, r), &mut convergency)
                .expect("Failed to converge or bracket the root");

            // Update in-place! Because we read `points[j+1]` on the next loop,
            // overwriting `points[j]` here is perfectly safe and zero-allocation.
            points[j] = root;
        }
        zerosj
            .row_mut(i)
            .slice_mut(s![..n_points])
            .assign(&points.slice(s![..n_points]));
    }
    zerosj.row(order as usize).to_owned()
}

pub(super) fn spherical_jn(order: f64, z: f64) -> f64 {
    (FRAC_PI_2 / z).sqrt() * bessel_j(order + 0.5, z).unwrap()
}
