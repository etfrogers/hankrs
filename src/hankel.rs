use ndarray::{Array1, Array2, Axis, NewAxis, s};
use ndarray_interp::interp1d::{Interp1DBuilder, cubic_spline::CubicSpline};
use ndarray_stats::QuantileExt;
use std::{f64::consts::PI, fmt::Debug};

use amos_bessel_rs::{BesselFunType, bessel_j, bessel_zeros};

// The main class for performing Hankel Transforms
//
// For the QDHT to work, the function must be sampled a specific points, which this class generates
// and stores in :attr:`HankelTransform.r`. Any transform on this grid will be sampled at points
// :attr:`.HankelTransform.v` (frequency space) or equivalently :attr:`.HankelTransform.kr`
// (angular frequency or wavenumber space).

// The constructor has one required argument (``order``). The remaining four arguments offer
// three different ways of specifying the radial (and therefore implicitly the frequency) points:

// 1. Supply both a maximum radius ``r_max`` and number of transform points ``n_points``
// 2. Supply the original (often equally spaced) ``radial_grid`` on which you have currently
//     have sample points. This approach allows easy conversion from the original grid using
//     :meth:`.HankelTransform.to_transform_r()`. ``t = HankelTransform(order, radial_grid=r)``
//     is effectively equivalent to ``t = HankelTransform(order, n_points=r.size, r_max=np.max(r))``
//     except for the fact the the original radial grid is stored in the :class:`.HankelTransform`
//     object for use in :meth:`~.HankelTransform.to_transform_r` and
//     :meth:`~.HankelTransform.to_original_r`.
// 3. Supply the original (often equally spaced) :math:`k`-space grid on which you
//     have currently have sample points. This is most use if you intend to do inverse
//     transforms. It allows easy conversion to and from the original grid using
//     :meth:`~.HankelTransform.to_original_k()` and :meth:`~.HankelTransform.to_transform_k()`.
//     As in option 2, :attr:`.HankelTransform.n_points` is determined by ``k_grid.size``.
//     :attr:`HankelTransform.r_max` is determined in a more complex way from ``np.max(k_grid)``.

// :parameter order: Transform order :math:`p`
// :type order: :class:`int`
// :parameter max_radius: (Optional) Radial extent of transform :math:`r_\textrm{max}`
// :type max_radius: :class:`float`
// :parameter n_points: (Optional) Number of sample points :math:`N`
// :type n_points: :class:`int`
// :parameter radial_grid: (Optional) The radial grid that will be used to sample input functions
//     it is used to set `N` and :math:`r_\textrm{max}` by ``n_points = radial_grid.size`` and
//     ``r_max = np.max(radial_grid)``
// :type radial_grid: :class:`numpy.ndarray`
// :parameter k_grid: (Optional) Number of sample points :math:`N`
// :type k_grid: :class:`numpy.ndarray`

// :ivar alpha: The first :math:`N` Roots of the :math:`p` th order Bessel function.
// :ivar alpha_n1: (N+1)th root :math:`\alpha_{N1}`
// :ivar r: Radial co-ordinate vector
// :ivar v: frequency co-ordinate vector
// :ivar kr: Radial wave number co-ordinate vector
// :ivar v_max: Limiting frequency :math:`v_\textrm{max} = \alpha_{N1}/(2 \pi R)`
// :ivar S: RV product :math:`2\pi r_\textrm{max} v_max`
// :ivar T: Transform matrix
// :ivar JR: Radius transform vector :math:`J_R = J_{p+1}(\alpha) / r_\textrm{max}`
// :ivar JV: Frequency transform vector :math:`J_V = J_{p+1}(\alpha) / v_\textrm{max}`

// The algorithm used is that from:

//     *"Computation of quasi-discrete Hankel transforms of the integer
//     order for propagating optical wave fields"*
//     Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
//     J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

// The algorithm also calls the function :func:`scipy.special.jn_zeros` to calculate
// the roots of the bessel function.
pub struct HankelTransform {
    order: i32,
    n_points: usize,
    max_radius: f64,
    original_radial_grid: Option<Array1<f64>>,
    original_k_grid: Option<Array1<f64>>,
    alpha: Array1<f64>,
    alpha_n1: f64,
    r: Array1<f64>,
    kr: Array1<f64>,
    v_max: f64,
    s: f64,
    t: Array2<f64>,
    jv: Array1<f64>,
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

/*
impl HankelTransform{
    fn new_from_r_grid(order: i32,
                 radial_grid: Array1<f64>) -> Self{
        // """Constructor"""

        // usage = "Either radial_grid or k_grid or both max_radius and n_points must be supplied"
        // if radial_grid is None and k_grid is None:
        //     if max_radius is None or n_points is None:
        //         raise ValueError(usage)
        // elif k_grid is not None:
        //     if max_radius is not None or n_points is not None or radial_grid is not None:
        //         raise ValueError(usage)
        //     assert k_grid.ndim == 1, "k grid must be a 1d array"
            // n_points = k_grid.size
        // elif radial_grid is not None:
        //     if max_radius is not None or n_points is not None:
        //         raise ValueError(usage)
        //     assert radial_grid.ndim == 1, "Radial grid must be a 1d array"
            let max_radius = *radial_grid.max().unwrap();
            let n_points = radial_grid.len();


        // Calculate N+1 roots must be calculated before max_radius can be derived from k_grid
        let alpha = bessel_zeros(order, n_points + 1);
        let alpha_n1 = alpha.last().unwrap();
        let alpha = alpha.slice(s![0..n_points]);

        // if k_grid is not None:
        //     v_max = np.max(k_grid) / (2 * np.pi)
        //     max_radius = self.alpha_n1 / (2 * np.pi * v_max)
        // self._max_radius = max_radius

        // Calculate co-ordinate vectors
        let r = alpha * max_radius / alpha_n1;
        let v = alpha / (2.0 * pi * max_radius);
        let kr = 2.0 * pi * v;
        let v_max = alpha_n1 / (2.0 * pi * max_radius);
        let S = alpha_n1;

        // Calculate hankel matrix and vectors
        let jp = scipy_bessel.jv(order, (alpha[:, np.newaxis] @ alpha[np.newaxis, :]) / S);
        let jp1 = (scipy_bessel.jv(order + 1, alpha)).abs();
        let T = 2 * jp / ((jp1[:, np.newaxis] @ jp1[np.newaxis, :]) * S);
        let JR = jp1 / max_radius;
        let JV = jp1 / v_max;

        Self{
            order,
            n_points,
           original_radial_grid: Some(radial_grid),
           max_radius,
           original_k_grid: None,
           alpha: alpha.slice()
           };
    }

}
 */
impl HankelTransform {
    pub fn new(order: i32, max_radius: f64, n_points: usize) -> Self {
        Self::build(n_points, Some(max_radius), order, None, None)
    }

    pub fn new_from_r_grid(order: i32, radial_grid: Array1<f64>) -> HankelTransform {
        /*
           usage = 'Either radial_grid or k_grid or both max_radius and n_points must be supplied'
           if radial_grid is None and k_grid is None:
               if max_radius is None or n_points is None:
                   raise ValueError(usage)
           elif k_grid is not None:
               if max_radius is not None or n_points is not None or radial_grid is not None:
                   raise ValueError(usage)
               assert k_grid.ndim == 1, 'k grid must be a 1d array'
               n_points = k_grid.size
           elif radial_grid is not None:
               if max_radius is not None or n_points is not None:
                   raise ValueError(usage)
               assert radial_grid.ndim == 1, 'Radial grid must be a 1d array'
               max_radius = np.max(radial_grid)
               n_points = radial_grid.size
           else:
               raise ValueError(usage)  # pragma: no cover - backup case: cannot currently be reached
        */
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
        println!("{}", alpha);
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
        let kr = 2.0 * PI * v;
        let v_max = alpha_n1 / (2.0 * PI * max_radius);
        let s = alpha_n1;

        // Calculate hankel matrix and vectors
        // jp = scipy_bessel.jv(order, (self.alpha[:, np.newaxis] @ self.alpha[np.newaxis, :]) / self.S)
        // jp1 = np.abs(scipy_bessel.jv(order + 1, self.alpha))
        // self.T = 2 * jp / ((jp1[:, np.newaxis] @ jp1[np.newaxis, :]) * self.S)
        // self.JR = jp1 / self.max_radius
        // self.JV = jp1 / self.v_max
        let alpha_row: Array2<_> = alpha.clone().into_shape((n_points, 1)).unwrap();
        // let alpha_row: Array2<_> = alpha.slice(s![.., NewAxis]).to_owned();
        let alpha_col: Array2<_> = alpha.slice(s![NewAxis, ..]).to_owned();
        let alpha_matrix = alpha_row.dot(&alpha_col);
        let jp: Array2<_> = alpha_matrix.map(|a| bessel_j(order, a / s).unwrap());
        let jp1: Array1<_> = alpha.map(|a| bessel_j(order + 1, *a).unwrap().abs());

        let jp1_row: Array2<_> = jp1.slice(s![.., NewAxis]).to_owned();
        let jp1_col: Array2<_> = jp1.slice(s![NewAxis, ..]).to_owned();

        let t: Array2<_> = 2.0 * jp / (jp1_row.dot(&jp1_col) * s);
        let jr: Array1<_> = jp1.clone() / max_radius;
        let jv: Array1<_> = jp1 / v_max;

        // jp := mat.NewDense(h.nPoints, h.nPoints, nil)
        // jp.Outer(1/h.S, h.alpha, h.alpha)
        // jp.Apply(func(i, j int, v float64) float64 { return math.Jn(h.order, v) }, jp)

        // jp1 := mat.NewVecDense(h.nPoints, nil)
        // utils.ApplyVec(func(v float64) float64 { return math.Abs(math.Jn(h.order+1, v)) }, jp1, h.alpha)
        // jp1Mat := mat.NewDense(h.nPoints, h.nPoints, nil)
        // jp1Mat.Outer(h.S, jp1, jp1)
        // h.T = *mat.NewDense(h.nPoints, h.nPoints, nil)
        // h.T.DivElem(jp, jp1Mat)
        // h.T.Scale(2, &(h.T))
        // h.JR = *mat.NewVecDense(h.nPoints, nil)
        // h.JR.ScaleVec(1/h.maxRadius, jp1)
        // h.JV = *mat.NewVecDense(h.nPoints, nil)
        // h.JV.ScaleVec(1/h.vMax, jp1)
        Self {
            order,
            n_points,
            max_radius,
            original_radial_grid,
            original_k_grid,
            alpha,
            alpha_n1,
            r,
            kr,
            v_max,
            s,
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

    /// Return the original radial grid used to construct the object, or raise a :class:`ValueError`
    /// if the constructor was not called specifying a ``radial_grid`` parameter.
    ///
    /// :return: The original radial grid used to construct the object.
    /// :rtype: :class:`numpy.ndarray`
    pub fn original_radial_grid(&self) -> Option<&Array1<f64>> {
        self.original_radial_grid.as_ref()
    }

    /// Return the original k grid used to construct the object, or raise a :class:`ValueError`
    /// if the constructor was not called specifying a ``k_grid`` parameter.
    /// :return: The original k grid used to construct the object.
    /// :rtype: :class:`numpy.ndarray`
    pub fn original_k_grid(&self) -> Option<&Array1<f64>> {
        self.original_k_grid.as_ref()
    }

    /// Interpolate a function, assumed to have been given at the original radial
    // grid points used to construct the ``HankelTransform`` object onto the grid required
    // of use in the QDHT algorithm.

    // If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
    // grid in radius, then it needs the function to transform to be sampled at a specific
    // grid before it can be passed to :meth:`.HankelTransform.qdht`. This method provides
    // a convenient way of doing this.

    // :parameter function: The function to be interpolated. Specified at the radial points
    //     :attr:`~.HankelTransform.original_radial_grid`.
    // :type function: :class:`numpy.ndarray`
    // :parameter axis: Axis representing the radial dependence of `function`.
    // :type axis: :class:`int`

    // :return: Interpolated function suitable to passing to
    //     :meth:`HankelTransform.qdht` (sampled at ``self.r``)
    // :rtype: :class:`numpy.ndarray`

    pub fn to_transform_r(
        &self,
        function: &Array1<f64>, /*,  axis: Axis*/
    ) -> Result<Array1<f64>, &str> {
        //         let data =     array![0.0,  0.5, 1.0 ];
        // let x =        array![0.0,  1.0, 2.0 ];
        // let query =    array![0.5,  1.0, 1.5 ];
        // let expected = array![0.25, 0.5, 0.75];
        if let Some(r_grid) = self.original_radial_grid() {
            Ok(spline(r_grid, function, &self.r, Axis(0)))
        } else {
            Err(
                "Attempted to interpolate onto transform radial grid on HankelTransform \
                object that was not constructed with a radial grid",
            )
        }
    }

    // Interpolate a function, assumed to have been given at the Hankel transform points
    // ``self.r`` (as returned by :meth:`HankelTransform.iqdht`) back onto the original grid
    // used to construct the ``HankelTransform`` object.

    // If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
    // grid in radius, it may be useful to convert back to this grid after a IQDHT.
    // This method provides a convenient way of doing this.

    // :parameter function: The function to be interpolated. Specified at the radial points
    //     ``self.r``.
    // :type function: :class:`numpy.ndarray`
    // :parameter axis: Axis representing the radial dependence of `function`.
    // :type axis: :class:`int`

    // :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_radial_grid`.
    // :rtype: :class:`numpy.ndarray`

    pub fn to_original_r(
        &self,
        function: &Array1<f64>, /* , axis: Axis*/
    ) -> Result<Array1<f64>, &str> {
        if let Some(r_grid) = self.original_radial_grid() {
            Ok(spline(&self.r, function, r_grid, Axis(0)))
        } else {
            Err(
                "Attempted to interpolate onto original_radial_grid on HankelTransform \
                object that was not constructed with a r_grid",
            )
        }
    }

    // Interpolate a function, assumed to have been given at the original k
    // grid points used to construct the ``HankelTransform`` object onto the grid required
    // of use in the IQDHT algorithm.

    // If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
    // grid in :math:`k`, then it needs the function to transform to be sampled at a specific
    // grid before it can be passed to :meth:`.HankelTransform.iqdht`. This method provides
    // a convenient way of doing this.

    // :parameter function: The function to be interpolated. Specified at the k points
    //     :attr:`~.HankelTransform.original_k_grid`.
    // :type function: :class:`numpy.ndarray`
    // :parameter axis: Axis representing the frequency dependence of `function`.
    // :type axis: :class:`int`

    // :return: Interpolated function suitable to passing to
    //     :meth:`HankelTransform.qdht` (sampled at ``self.kr``)
    // :rtype: :class:`numpy.ndarray`
    pub fn to_transform_k(&self, function: &Array1<f64>) -> Result<Array1<f64>, &str> {
        if let Some(k_grid) = self.original_k_grid() {
            Ok(spline(k_grid, function, &self.kr, Axis(0)))
        } else {
            Err(
                "Attempted to interpolate onto transform k grid on HankelTransform \
                object that was not constructed with a k_grid",
            )
        }
    }

    /// Interpolate a function, assumed to have been given at the Hankel transform points
    // ``self.k`` (as returned by :meth:`HankelTransform.qdht`) back onto the original grid
    // used to construct the ``HankelTransform`` object.

    // If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
    // grid in :math:`k`, it may be useful to convert back to this grid after a QDHT.
    // This method provides a convenient way of doing this.

    // :parameter function: The function to be interpolated. Specified at the radial points
    //     ``self.k``.
    // :type function: :class:`numpy.ndarray`
    // :parameter axis: Axis representing the frequency dependence of `function`.
    // :type axis: :class:`int`

    // :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_k_grid`.
    // :rtype: :class:`numpy.ndarray`

    pub fn to_original_k(&self, function: &Array1<f64>) -> Result<Array1<f64>, &str> {
        if let Some(k_grid) = self.original_k_grid() {
            Ok(spline(&self.kr, function, k_grid, Axis(0)))
        } else {
            Err(
                "Attempted to interpolate onto original_k_grid on HankelTransform \
                object that was not constructed with a k grid",
            )
        }
    }

    /// QDHT: Quasi Discrete Hankel Transform

    // Performs the Hankel transform of a function of radius, returning
    // a function of frequency.

    // .. math::
    //     f_v(v) = \mathcal{H}^{-1}\{f_r(r)\}

    // .. warning:
    //     The input function must be sampled at the points ``self.r``, and the output
    //     will be sampled at the points ``self.v`` (or equivalently ``self.kr``)

    // :parameter fr: Function in real space as a function of radius (sampled at ``self.r``)
    // :type fr: :class:`numpy.ndarray`
    // :parameter axis: Axis over which to compute the Hankel transform.
    // :type axis: :class:`int`

    // :return: Function in frequency space (sampled at ``self.v``)
    // :rtype: :class:`numpy.ndarray`

    // pub fn qdht_1d(&self, fr: &Array1<f64>) -> Array1<f64> {}

    pub fn qdht(&self, fr: &Array1<f64>, axis: Axis) -> Array1<f64> {
        if (fr.ndim() == 1) || (axis == Axis(fr.ndim() - 2)) {
            let (jr, jv) = self.get_scaling_factors(fr);

            let fv = jv * self.t.dot(&(fr / jr));
            return fv;
        } else {
            todo!();
            // _fr = np.core.swapaxes(fr, axis, -2);
            // (jr, jv) = self._get_scaling_factors(_fr);
            // fv = jv * np.matmul(self.T, (_fr / jr));
            // return np.core.swapaxes(fv, axis, -2);
        }
    }

    // IQDHT: Inverse Quasi Discrete Hankel Transform

    // Performs the inverse Hankel transform of a function of frequency, returning
    // a function of radius.

    // .. math::
    //     f_r(r) = \mathcal{H}^{-1}\{f_v(v)\}

    // :parameter fv: Function in frequency space (sampled at self.v)
    // :type fv: :class:`numpy.ndarray`
    // :parameter axis: Axis over which to compute the Hankel transform.
    // :type axis: :class:`int`

    // :return: Radial function (sampled at self.r) = IHT(fv)
    // :rtype: :class:`numpy.ndarray`
    pub fn iqdht(&self, fv: &Array1<f64>, axis: Axis) -> Array1<f64> {
        if (fv.ndim() == 1) || (axis == Axis(fv.ndim() - 2)) {
            let (jr, jv) = self.get_scaling_factors(fv);
            jr * self.t.dot(&(fv / jv))
        } else {
            todo!()
            // _fv = np.core.swapaxes(fv, axis, -2);
            // (jr, jv) = self._get_scaling_factors(_fv);
            // fr = jr * np.matmul(self.T, (_fv / jv));
            // np.core.swapaxes(fr, axis, -2)
        }
    }

    fn get_scaling_factors(&self, f: &Array1<f64>) -> (&Array1<f64>, &Array1<f64>) {
        if f.ndim() > 1 {
            todo!();
            // n2 = list(f.shape);
            // n2[-2] = 1;
            // _shape = np.ones_like(n2);
            // _shape[-2] = len(self.JR);
            // (
            //     np.reshape(self.JR, _shape) * np.ones(n2),
            //     np.reshape(self.JV, _shape) * np.ones(n2),
            // )
        } else {
            (&self.jr, &self.jv)
        }
    }
}

fn spline(x0: &Array1<f64>, y0: &Array1<f64>, x: &Array1<f64>, _axis: Axis) -> Array1<f64> {
    // f = interpolate.interp1d(x0, y0, axis=axis, fill_value="extrapolate", kind="cubic")
    // return f(x)
    let interpolator = Interp1DBuilder::new(y0.clone())
        .x(x0.clone())
        .strategy(CubicSpline::new().extrapolate(true))
        .build()
        .unwrap();
    interpolator.interp_array(x).unwrap()
}
