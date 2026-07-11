use ndarray::{Array, Array1, ArrayBase, Axis, Data, Dim, DimAdd, Dimension, RemoveAxis};

use crate::hankel::{HankelScalar, HankelTransform};

/// Perform a quasi-discrete Hankel transform of the function `f` (sampled at points
/// `r`) and return the transformed function and its sample points in `k`-space.
///
/// If you require the transform on a frequency axis (as opposed to the `k`-axis), the
/// frequency axis `v` can be calculated using `v = k / (2 * pi)`.
///
/// # Warning
/// This method is a convenience wrapper for [`HankelTransform::qdht`], but incurs a
/// significant overhead in calculating the [`HankelTransform`] object. If you
/// are performing multiple transforms on the same grid, it will be much quicker to
/// construct a single [`HankelTransform`] object and call
/// [`HankelTransform::qdht`] multiple times.
///
/// # Arguments
/// * `r` - The radial coordinates at which the function is sampled.
/// * `f` - The value of the function to be transformed.
/// * `order` - The order of the Hankel Transform to perform. Defaults to 0.
/// * `axis` - Axis over which to compute the Hankel transform.
///
/// # Returns
/// A tuple containing the `k` coordinates of the transformed function and its values.
pub fn qdht<T: HankelScalar, D, S>(
    r: Array1<f64>,
    f: &ArrayBase<S, D>,
    order: i32,
    axis: Axis,
) -> (Array1<f64>, Array<T, D>)
where
    D: Dimension + RemoveAxis,
    Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    S: Data<Elem = T>,
{
    let transformer = HankelTransform::new_from_r_grid(order, r);
    let f_transform = transformer.to_transform_r_nd(f, axis).unwrap();
    let ht = transformer.qdht(&f_transform, axis);
    (transformer.into_kr(), ht)
}

/// Perform an inverse quasi-discrete Hankel transform of the function `f` (sampled at points
/// `k`) and return the transformed function and its sample points in radial space.
///
/// If you have the transform on a frequency axis (as opposed to a `k`-axis), the
/// `k`-axis can be calculated using `k = 2 * pi * f`.
///
/// # Warning
/// This method is a convenience wrapper for [`HankelTransform::iqdht`], but incurs a
/// significant overhead in calculating the [`HankelTransform`] object. If you
/// are performing multiple transforms on the same grid, it will be much quicker to
/// construct a single [`HankelTransform`] object and call
/// [`HankelTransform::iqdht`] multiple times.
///
/// # Arguments
/// * `k` - The `k` coordinates at which the function is sampled.
/// * `f` - The value of the function to be transformed.
/// * `order` - The order of the Hankel Transform to perform. Defaults to 0.
/// * `axis` - Axis over which to compute the Hankel transform.
///
/// # Returns
/// A tuple containing the radial coordinates of the transformed function and its values.
pub fn iqdht<T: HankelScalar, D, S>(
    k: Array1<f64>,
    f: &ArrayBase<S, D>,
    order: i32,
    axis: Axis,
) -> (Array1<f64>, Array<T, D>)
where
    D: Dimension + RemoveAxis,
    Dim<[usize; 1]>: DimAdd<<D as Dimension>::Smaller>,
    S: Data<Elem = T>,
{
    let transformer = HankelTransform::new_from_k_grid(order, k);
    let f_transform = transformer.to_transform_k_nd(f, axis).unwrap();
    let ht = transformer.iqdht(&f_transform, axis);
    (transformer.into_radius(), ht)
}
