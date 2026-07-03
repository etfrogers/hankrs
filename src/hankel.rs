use approx::{AbsDiffEq, RelativeEq};
use ndarray::{Array, Array1, Array2, Axis, Dim, DimAdd, Dimension, IxDyn, NewAxis, RemoveAxis, s};
use ndarray_interp::interp1d::{Interp1DBuilder, cubic_spline::CubicSpline};
use ndarray_stats::QuantileExt;
use std::{f64::consts::PI, fmt::Debug};

use amos_bessel_rs::bessel_j;
use bessel_zeros::{BesselFunType, bessel_zeros};

/// The main struct for performing Hankel Transforms
///
/// For the QDHT to work, the function must be sampled at specific points, which this struct generates
/// and stores in `self.r`. Any transform on this grid will be sampled at points
/// `self.v` (frequency space) or equivalently `self.kr`
/// (angular frequency or wavenumber space).
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
/// 3. Supply the original (often equally spaced) $k$-space grid on which you
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
    /// Transform order $p$
    order: i32,
    /// Number of sample points $N$
    n_points: usize,
    /// Radial extent of transform $r_{max}$
    max_radius: f64,
    /// Original radial grid on which sample points were provided
    original_radial_grid: Option<Array1<f64>>,
    /// Original $k$-space grid on which sample points were provided
    original_k_grid: Option<Array1<f64>>,
    /// Radial co-ordinate vector
    r: Array1<f64>,
    /// Radial wave number co-ordinate vector
    kr: Array1<f64>,
    /// Frequency co-ordinate vector
    v: Array1<f64>,
    /// Transform matrix
    t: Array2<f64>,
    /// Frequency transform vector $J_V = J_{p+1}(\alpha) / v_{max}$
    jv: Array1<f64>,
    /// Radius transform vector $J_R = J_{p+1}(\alpha) / r_{max}$
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
    pub fn new(order: i32, max_radius: f64, n_points: usize) -> Self {
        Self::build(n_points, Some(max_radius), order, None, None)
    }

    pub fn new_from_r_grid(order: i32, radial_grid: Array1<f64>) -> HankelTransform {
        Self::build(
            radial_grid.len(),
            Some(*radial_grid.max().unwrap()),
            order,
            Some(radial_grid),
            None,
        )
    }

    pub fn new_from_k_grid(order: i32, k_grid: Array1<f64>) -> Self {
        Self::build(k_grid.len(), None, order, None, Some(k_grid))
    }

    fn build(
        n_points: usize,
        max_radius: Option<f64>,
        order: i32,
        original_radial_grid: Option<Array1<f64>>,
        original_k_grid: Option<Array1<f64>>,
    ) -> Self {
        // Calculate N+1 roots must be calculated before max_radius can be derived from k_grid
        let alpha = Array1::from_vec(bessel_zeros(&BesselFunType::J, order, n_points + 1, 1e-6));

        let alpha_n1 = alpha[n_points];
        let alpha = alpha.slice(s![0..n_points]).to_owned();

        let max_radius = match max_radius {
            Some(mr) => mr,
            None => match original_k_grid {
                Some(ref kg) => {
                    let v_max = kg.max().unwrap() / (2.0 * PI);
                    alpha_n1 / (2.0 * PI * v_max)
                }
                None => panic!("Either k_grid or radial parameters must be supplied"),
            },
        };
        // Calculate co-ordinate vectors
        let r = alpha.clone() * max_radius / alpha_n1;
        let v = alpha.clone() / (2.0 * PI * max_radius);
        let kr = 2.0 * PI * v.clone();
        let v_max = alpha_n1 / (2.0 * PI * max_radius);
        let s = alpha_n1;

        // Calculate hankel matrix and vectors
        let alpha_row = alpha.to_shape((n_points, 1)).unwrap();
        let alpha_col: Array2<_> = alpha.slice(s![NewAxis, ..]).to_owned();
        let alpha_matrix = alpha_row.dot(&alpha_col);
        let jp: Array2<_> = alpha_matrix.map(|a| bessel_j(order, a / s).unwrap());
        let jp1: Array1<_> = alpha.map(|a| bessel_j(order + 1, *a).unwrap().abs());

        let jp1_row: Array2<_> = jp1.slice(s![.., NewAxis]).to_owned();
        let jp1_col: Array2<_> = jp1.slice(s![NewAxis, ..]).to_owned();

        let t: Array2<_> = 2.0 * jp / (jp1_row.dot(&jp1_col) * s);
        let jr: Array1<_> = jp1.clone() / max_radius;
        let jv: Array1<_> = jp1 / v_max;

        Self {
            order,
            n_points,
            max_radius,
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

    pub fn order(&self) -> i32 {
        self.order
    }

    pub fn max_radius(&self) -> f64 {
        self.max_radius
    }

    pub fn n_points(self) -> usize {
        self.n_points
    }

    /// Return the original radial grid used to construct the object, or `None`
    /// if the constructor was not called specifying a `radial_grid` parameter.
    ///
    /// # Returns
    /// An `Option` containing the original radial grid used to construct the object.
    pub fn original_radial_grid(&self) -> Option<&Array1<f64>> {
        self.original_radial_grid.as_ref()
    }

    /// Return the original k grid used to construct the object, or `None`
    /// if the constructor was not called specifying a `k_grid` parameter.
    ///
    /// # Returns
    /// An `Option` containing the original k grid used to construct the object.
    pub fn original_k_grid(&self) -> Option<&Array1<f64>> {
        self.original_k_grid.as_ref()
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

    pub fn to_transform_r(&self, function: &Array1<f64>) -> Result<Array1<f64>, &str> {
        self.to_transform_r_nd(function, Axis(0))
    }

    pub fn to_transform_r_nd<D: Dimension + RemoveAxis>(
        &self,
        function: &Array<f64, D>,
        axis: Axis,
    ) -> Result<Array<f64, D>, &str>
    where
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    {
        if let Some(r_grid) = self.original_radial_grid() {
            Ok(spline(r_grid, function, &self.r, axis))
        } else {
            Err(
                "Attempted to interpolate onto transform radial grid on HankelTransform \
                object that was not constructed with a radial grid",
            )
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
    pub fn to_original_r(&self, function: &Array1<f64>) -> Result<Array1<f64>, &str> {
        self.to_original_r_nd(function, Axis(0))
    }

    pub fn to_original_r_nd<D>(
        &self,
        function: &Array<f64, D>,
        axis: Axis,
    ) -> Result<Array<f64, D>, &str>
    where
        D: Dimension + RemoveAxis,
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    {
        if let Some(r_grid) = self.original_radial_grid() {
            Ok(spline(&self.r, function, r_grid, axis))
        } else {
            Err(
                "Attempted to interpolate onto original_radial_grid on HankelTransform \
                object that was not constructed with a r_grid",
            )
        }
    }

    /// Interpolate a function, assumed to have been given at the original k
    /// grid points used to construct the [`HankelTransform`] object onto the grid required
    /// for use in the IQDHT algorithm.
    ///
    /// If the [`HankelTransform`] object was constructed with a (say) equally-spaced
    /// grid in $k$, then it needs the function to transform to be sampled at a specific
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
    pub fn to_transform_k(&self, function: &Array1<f64>) -> Result<Array1<f64>, &str> {
        self.to_transform_k_nd(function, Axis(0))
    }

    pub fn to_transform_k_nd<D: Dimension + RemoveAxis>(
        &self,
        function: &Array<f64, D>,
        axis: Axis,
    ) -> Result<Array<f64, D>, &str>
    where
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    {
        if let Some(k_grid) = self.original_k_grid() {
            Ok(spline(k_grid, function, &self.kr, axis))
        } else {
            Err(
                "Attempted to interpolate onto transform k grid on HankelTransform \
                object that was not constructed with a k_grid",
            )
        }
    }

    /// Interpolate a function, assumed to have been given at the Hankel transform points
    /// `self.k` (as returned by [`HankelTransform::qdht`]) back onto the original grid
    /// used to construct the [`HankelTransform`] object.
    ///
    /// If the [`HankelTransform`] object was constructed with a (say) equally-spaced
    /// grid in $k$, it may be useful to convert back to this grid after a QDHT.
    /// This method provides a convenient way of doing this.
    ///
    /// # Arguments
    /// * `function` - The function to be interpolated. Specified at the radial points
    ///   `self.k`.
    /// * `axis` - Axis representing the frequency dependence of `function`.
    ///
    /// # Returns
    /// Interpolated function at the points held in [`HankelTransform::original_k_grid`].
    pub fn to_original_k(&self, function: &Array1<f64>) -> Result<Array1<f64>, &str> {
        self.to_original_k_nd(function, Axis(0))
    }

    pub fn to_original_k_nd<D: Dimension + RemoveAxis>(
        &self,
        function: &Array<f64, D>,
        axis: Axis,
    ) -> Result<Array<f64, D>, &str>
    where
        Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    {
        if let Some(k_grid) = self.original_k_grid() {
            Ok(spline(&self.kr, function, k_grid, axis))
        } else {
            Err(
                "Attempted to interpolate onto original_k_grid on HankelTransform \
                object that was not constructed with a k grid",
            )
        }
    }

    /// QDHT: Quasi Discrete Hankel Transform
    ///
    /// Performs the Hankel transform of a function of radius, returning
    /// a function of frequency.
    ///
    /// $f_v(v) = \mathcal{H}^{-1}\{f_r(r)\}$
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
    pub fn qdht<D: Dimension + RemoveAxis>(&self, fr: &Array<f64, D>, axis: Axis) -> Array<f64, D> {
        let scale_factor_input = &self.jr;
        let scale_factor_output = &self.jv;
        self.transform_by_lines(fr, axis, scale_factor_input, scale_factor_output)
    }

    /// IQDHT: Inverse Quasi Discrete Hankel Transform
    ///
    /// Performs the inverse Hankel transform of a function of frequency, returning
    /// a function of radius.
    ///
    /// $f_r(r) = \mathcal{H}^{-1}\{f_v(v)\}$
    ///
    /// # Arguments
    /// * `fv` - Function in frequency space (sampled at `self.v`).
    /// * `axis` - Axis over which to compute the Hankel transform.
    ///
    /// # Returns
    /// Radial function (sampled at `self.r`) = IHT(fv).
    pub fn iqdht<D: Dimension>(&self, fv: &Array<f64, D>, axis: Axis) -> Array<f64, D> {
        self.transform_by_lines(fv, axis, &self.jv, &self.jr)
    }

    fn transform_by_lines<D: Dimension>(
        &self,
        f: &Array<f64, D>,
        axis: Axis,
        scale_factor_input: &Array1<f64>,
        scale_factor_output: &Array1<f64>,
    ) -> Array<f64, D> {
        let mut transform = Array::zeros(f.dim());

        for (mut transform_line, fr_line) in
            transform.lanes_mut(axis).into_iter().zip(f.lanes(axis))
        {
            let scaled_line = fr_line.to_owned() / scale_factor_input;
            let mut transformed = self.t.dot(&scaled_line);
            transformed *= scale_factor_output;
            transform_line.assign(&transformed);
        }
        transform
    }

    pub fn transform_matrix(&self) -> &Array2<f64> {
        &self.t
    }

    pub fn radius(&self) -> &Array1<f64> {
        &self.r
    }

    pub fn frequency(&self) -> &Array1<f64> {
        &self.v
    }

    pub fn kr(&self) -> &Array1<f64> {
        &self.kr
    }

    pub(crate) fn into_radius(self) -> Array1<f64> {
        self.r
    }

    pub(crate) fn into_kr(self) -> Array1<f64> {
        self.kr
    }
}

pub fn perms<D: Dimension>(axis: Axis) -> (D, D) {
    let ndim = D::NDIM.expect("Dimension must be fixed");
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

fn spline<D>(x0: &Array1<f64>, y0: &Array<f64, D>, x: &Array1<f64>, axis: Axis) -> Array<f64, D>
where
    D: Dimension + RemoveAxis,
    Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
{
    let (y0, inverse_perms) = if axis != Axis(0) {
        let (forward_perms, inverse_perms) = perms::<D>(axis);
        let mut y0_ = y0.view();
        y0_ = y0_.permuted_axes(forward_perms);

        (y0_.to_owned(), Some(inverse_perms))
    } else {
        (y0.clone(), None)
    };
    let interpolator = Interp1DBuilder::new(y0.into_dyn())
        .x(x0.clone())
        .strategy(CubicSpline::new().extrapolate(true))
        .build()
        .unwrap();
    let mut result = interpolator
        .interp_array(x)
        .unwrap()
        .into_dimensionality::<D>()
        .unwrap();
    if axis != Axis(0) {
        result = result.permuted_axes(inverse_perms.unwrap()).to_owned();
    }
    result
}
