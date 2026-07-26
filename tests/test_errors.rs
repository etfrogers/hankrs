use hankrs::{HankelError, HankelTransform};
use ndarray::array;

#[test]
fn test_error_empty_grid() {
    // Zero points should return EmptyGrid
    let result = HankelTransform::new(0, 10.0, 0);
    assert!(matches!(result, Err(HankelError::EmptyGrid)));

    // Empty original r grid should also return EmptyGrid
    let empty_grid = array![];
    let result2 = HankelTransform::new_from_r_grid(0, empty_grid);
    assert!(matches!(result2, Err(HankelError::EmptyGrid)));
}

#[test]
fn test_error_invalid_radius() {
    // Zero radius should return InvalidRadius
    let result = HankelTransform::new(0, 0.0, 256);
    assert!(matches!(result, Err(HankelError::InvalidRadius)));

    // Negative radius should return InvalidRadius
    let result2 = HankelTransform::new(0, -10.0, 256);
    assert!(matches!(result2, Err(HankelError::InvalidRadius)));

    // Negative radius should return InvalidRadius
    let result2 = HankelTransform::new(0, f64::NAN, 256);
    assert!(matches!(result2, Err(HankelError::InvalidRadius)));
}

#[test]
fn test_error_interpolation_missing_grid() {
    // Construct without providing original_k_grid
    let transformer = HankelTransform::new(0, 10.0, 256).unwrap();

    let some_data = array![1.0, 2.0, 3.0];

    // Attempting to interpolate onto transform k grid should fail
    // because no original_k_grid was provided during construction
    let result = transformer.to_transform_k(&some_data);
    assert!(matches!(result, Err(HankelError::Interpolation(_))));

    // Attempting to interpolate onto original r grid should fail
    // because no original_r_grid was provided during construction
    let result2 = transformer.to_original_r(&some_data);
    assert!(matches!(result2, Err(HankelError::Interpolation(_))));
}
