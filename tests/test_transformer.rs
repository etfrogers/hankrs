mod utils;

use approx::assert_relative_eq;
use hankrs::hankel::HankelTransform;
use ndarray::{Array1, Axis};
use num::pow::Pow;
use rand::random;
use rstest::{fixture, rstest};
use rstest_reuse::{apply, template};
use std::{f64::consts::PI, mem::MaybeUninit};
use utils::assert_relative_eq_with_end_points;
// const pi: f64 = math.PI;
const MAX_ORDER: i32 = 4;

// type RadialSuite struct {
// 	suite.Suite
// 	radius mat.VecDense
// }

// type HankelTestSuite struct {
// 	RadialSuite
// 	transformer gohank.HankelTransform
// 	order       int
// }

// -------------
// SUITE RUNNERS
// -------------

// func TestSuite(t *testing.T) {
// 	for order := 0; order <= maxOrder; order++ {
// 		hs := new(HankelTestSuite)
// 		hs.order = order
// 		suite.Run(t, hs)
// 	}
// }

// func TestRadialSuite(t *testing.T) {
// 	s := new(RadialSuite)
// 	suite.Run(t, s)
// }

// func (suite *HankelTestSuite) SetupTest() {
// 	suite.radius = *utils.Linspace(0, 3, 1024)
// 	suite.transformer = gohank.NewTransformFromRadius(suite.order, &suite.radius)
// }

// func (suite *RadialSuite) SetupTest() {
// 	suite.radius = *utils.Linspace(0, 3, 1024)
// }

fn random_vec_like(v: &Array1<f64>) -> Array1<f64> {
    let shape = v.dim();
    Array1::uninit(shape).map(|_: &MaybeUninit<f64>| random::<f64>() * 10.0)
}

#[derive(Clone)]
struct Shape<'a> {
    name: String,
    f: &'a dyn Fn(f64) -> f64,
}

impl<'a> Shape<'a> {
    fn new(name: &str, f: &'a (impl Fn(f64) -> f64 + 'static)) -> Self {
        Self {
            name: name.to_string(),
            f,
        }
    }
}

// static SMOOTH_SHAPES: LazyLock<Vec<Shape>> = LazyLock::new(|| {
#[template]
#[rstest]
#[case(Shape {
            name: "zeros".to_string(),
            f: &|_| 0.0,
        },)]
#[case( Shape {
            name: "e^(-r^2)".to_string(),
            f: &|r: f64| (-r.pow(2.0_f64)).exp(),
        })]
#[case( Shape {
        name:"r".to_string(), f: &|r: f64| r })]
#[case( Shape {
        name:"r^2".to_string(), f: &|r: f64|  r.pow(2.0) })]
#[case( Shape {
 name:"1/(sqrt(r^2 + 0.1^2))".to_string(), f: &|r: f64|  1.0 / (r.pow(2.0_f64)+0.1.pow(2.0_f64)).sqrt() })]
fn smooth_shapes(#[case] shape: Shape) {}
// fn smooth_shapes<'a>() -> Vec<Shape<'a>> {
//     vec![
//         Shape {
//             name: "zeros".to_string(),
//             f: &|_| 0.0,
//         },
//         Shape {
//             name: "e^(-r^2)".to_string(),
//             f: &|r: f64| (-r.pow(2.0_f64)).exp(),
//         },
//         // {"r", func(r float64) float64 { return r }},
//         // {"r^2", func(r float64) float64 { return math.Pow(r, 2) }},
//         // {"1/(sqrt(r^2 + 0.1^2))", func(r float64) float64 { return 1 / math.Sqrt(math.Pow(r, 2)+math.Pow(0.1, 2)) }},
//     ]
// }

// #[fixture]
// fn all_shapes<'a>(smooth_shapes: Vec<Shape<'a>>) -> Vec<Shape<'a>> {
//     let mut v = smooth_shapes.clone();
//     v.push(Shape::new("random", &|_| random::<f64>() * 10.0));
//     v
// }

#[fixture]
fn radius() -> Array1<f64> {
    Array1::linspace(0.0, 3.0, 1024)
}

#[fixture]
#[once]
fn transformer(radius: Array1<f64>) -> HankelTransform {
    let order = 0;
    HankelTransform::new_from_r_grid(order, radius)
}

#[rstest]
fn test_round_trip(transformer: &HankelTransform) {
    let fun = random_vec_like(transformer.original_radial_grid());
    let ht = transformer.qdht(&fun, Axis(0));
    let reconstructed = transformer.iqdht(&ht, Axis(0));
    assert_relative_eq!(
        fun.as_slice().unwrap(),
        reconstructed.as_slice().unwrap(),
        max_relative = -1.0,
        epsilon = 1e-9
    )
}

// -------------------
// Test Interpolations
// -------------------
#[apply(smooth_shapes)]
#[rstest]
fn test_round_trip_r_interpolation(
    shape: Shape,
    radius: Array1<f64>,
    transformer: &HankelTransform,
) {
    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    let fun = radius.mapv_into(shape.f);
    let transform_func = transformer.to_transform_r(&fun);
    let reconstructed_func = transformer.to_original_r(&transform_func);
    assert_relative_eq_with_end_points(reconstructed_func, fun, 1e-4, 1e-3, -1.0, 2e-5);
}

#[apply(smooth_shapes)]
#[rstest]
fn test_round_trip_k_interpolation(shape: Shape, radius: Array1<f64>) {
    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    let order = 0;
    let k_grid = radius.mapv(|r| r / 10.0);
    let transformer = HankelTransform::new_from_k_grid(order, k_grid);

    let fun = radius.mapv_into(shape.f);
    let transform_func = transformer.to_transform_k(&fun);
    let reconstructed_func = transformer.to_original_k(&transform_func);
    assert_relative_eq_with_end_points(reconstructed_func, fun, 1e-4, 1e-3, 0.0, 2e-7);
}
// func (suite *RadialSuite) TestRoundTripKInterpolation() {
//     for _, shape := range smoothShapes {
//         order := 0
//         suite.Run(fmt.Sprintf("%v, %v", shape.name, order), func() {

//             kGrid := utils.ApplyVec(func(r float64) float64 { return r / 10 }, nil, &suite.radius)
//             transformer := gohank.NewTransformFromKGrid(order, kGrid)

//             // the function must be smoothish for interpolation
//             // to work. Random every point doesn't work
//             fun := utils.ApplyVec(shape.f, nil, kGrid)
//             transform_func := transformer.ToTransformK(fun)
//             reconstructed_func := transformer.ToOriginalK(transform_func)
//             testutils.AssertInDeltaVecWithEndPoints(suite.T(), fun, reconstructed_func, 1e-4, 1e-3, -1, 2e-7)
//         })
//     }
// }
/*
func (t *HankelTestSuite) TestRoundTripWithInterpolation() {
    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    for _, shape := range smoothShapes {
        t.Run(fmt.Sprintf("%v, %v", shape.name, t.order), func() {
            fun := utils.ApplyVec(shape.f, nil, &t.radius)
            fun_hr := t.transformer.ToTransformR(fun)
            ht := t.transformer.QDHT(fun_hr)
            reconstructed_hr := t.transformer.IQDHT(ht)
            reconstructed := t.transformer.ToOriginalR(reconstructed_hr)

            aTolEnd := 1e-3
            rTolBody := 2e-4
            aTolBody := -1.
            rTolEnd := -1.
            if shape.name == "1/(sqrt(r^2 + 0.1^2))" {
                rTolEnd = 3e-2
                rTolBody = 2e-3
            }
            if shape.name == "r^2" {
                rTolBody = -1.
                aTolBody = 2e-4
            }
            testutils.AssertInDeltaVecWithEndPoints(t.T(), fun, reconstructed, rTolBody, rTolEnd, aTolBody, aTolEnd)
        })
    }
}

func TestOriginalRKGrid(t *testing.T) {
    r_1d := utils.Linspace(0, 1, 10)
    var k_1d mat.VecDense
    k_1d.CloneFromVec(r_1d)
    transformer := gohank.NewTransform(0, 1., 10)
    assert.Panics(t, func() { transformer.OriginalRadialGrid() })
    assert.Panics(t, func() { transformer.OriginalKGrid() })

    transformer = gohank.NewTransformFromRadius(0, r_1d)
    // no error
    _ = transformer.OriginalRadialGrid()
    assert.Panics(t, func() { transformer.OriginalKGrid() })

    transformer = gohank.NewTransformFromKGrid(0, &k_1d)
    // no error
    _ = transformer.OriginalKGrid()
    assert.Panics(t, func() { transformer.OriginalRadialGrid() })
}

// ---------------
// Test Invariants
// ---------------
func (t *HankelTestSuite) TestParsevalsTheorem() {
    // As per equation 11 of Guizar-Sicairos, the UNSCALED transform is unitary,
    // i.e. if we pass in the unscaled fr (=Fr), the unscaled fv (=Fv)should have the
    // same sum of abs val^2. Here the unscaled transform is simply given by
    // ht = transformer.T @ func
    for _, shape := range all_shapes {
        t.Run(fmt.Sprintf("%v, %v", shape.name, t.order), func() {
            fun := utils.ApplyVec(shape.f, nil, &t.radius)
            intensityBefore := utils.ApplyVec(intensity, nil, fun)
            energyBefore := mat.Sum(intensityBefore)
            ht := mat.NewVecDense(fun.Len(), nil)
            ht.MulVec(&t.transformer.T, fun)
            intensityAfter := utils.ApplyVec(intensity, nil, ht)
            energyAfter := mat.Sum(intensityAfter)
            assert.InDelta(t.T(), energyBefore, energyAfter, 1e-8)
        })
    }
}

func intensity(v float64) float64 {
    return math.Pow(math.Abs(v), 2)
}

func (t *HankelTestSuite) TestEnergyConservation() {
    shapes := []struct {
        name string
        f    func(mat.Vector, float64, int) mat.Vector
    }{{"Jinc", testutils.GeneralisedJinc},
        {"Top Hat", testutils.GeneralisedTopHat}}

    integrateOverR := func(r, y *mat.VecDense) float64 {
        integrand := mat.NewVecDense(y.Len(), nil)
        for i := 0; i < y.Len(); i++ {
            integrand.SetVec(i, 2*pi*r.AtVec(i)*y.AtVec(i))
        }
        return integrate.Trapezoidal(r.RawVector().Data, integrand.RawVector().Data)
    }

    for _, shape := range shapes {
        t.Run(fmt.Sprintf("%v, %v", shape.name, t.order), func() {
            transformer := gohank.NewTransform(t.transformer.Order(), 10, t.transformer.NPoints())
            fun := shape.f(transformer.Radius(), 0.5, transformer.Order())
            intensityBefore := utils.ApplyVec(intensity, nil, fun).(*mat.VecDense)
            energyBefore := integrateOverR(transformer.Radius().(*mat.VecDense), intensityBefore)

            ht := transformer.QDHT(fun)
            intensityAfter := utils.ApplyVec(intensity, nil, ht).(*mat.VecDense)
            energyAfter := integrateOverR(transformer.V().(*mat.VecDense), intensityAfter)

            assert.InDelta(t.T(), energyBefore, energyAfter, 0.006)
        })
    }
}

// -------------------
// Test known HT pairs
// -------------------

func (t *HankelTestSuite) TestJinc() {
    for _, a := range []float64{1, 0.7, 0.1} {
        t.Run(fmt.Sprint(a), func() {
            f := testutils.GeneralisedJinc(t.transformer.Radius(), a, t.order)
            expected_ht := testutils.GeneralisedTopHat(t.transformer.V(), a, t.order)
            actual_ht := t.transformer.QDHT(f)
            err := testutils.MeanAbsError(expected_ht, actual_ht)
            assert.Less(t.T(), err, 1e-3)
        })
    }
}

func (t *HankelTestSuite) TestTopHat() {
    for _, a := range []float64{1, 1.5, 0.1} {
        t.Run(fmt.Sprint(a), func() {
            f := testutils.GeneralisedTopHat(t.transformer.Radius(), a, t.order)
            expected_ht := testutils.GeneralisedJinc(t.transformer.V(), a, t.order)
            actual_ht := t.transformer.QDHT(f)
            assert.Less(t.T(), testutils.MeanAbsError(expected_ht, actual_ht), 1e-3)
        })
    }
}

func (t *RadialSuite) TestGaussian() {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    transformer := gohank.NewTransformFromRadius(0, &t.radius)
    for _, a := range []float64{2, 5, 10} {
        t.Run(fmt.Sprint(a), func() {
            f := mat.NewVecDense(transformer.Radius().Len(), nil)
            a2 := math.Pow(a, 2)
            utils.ApplyVec(func(r float64) float64 { return math.Exp(-a2 * math.Pow(r, 2)) }, f, transformer.Radius())
            expected_ht := mat.NewVecDense(transformer.Radius().Len(), nil)
            utils.ApplyVec(func(kr float64) float64 { return 2 * pi * (1 / (2 * a2)) * math.Exp(-math.Pow(kr, 2)/(4*a2)) },
                expected_ht, transformer.Kr())
            actual_ht := transformer.QDHT(f)
            testutils.AssertInDeltaVec(t.T(), expected_ht, actual_ht, -1, 1e-9)
        })
    }
}

func (t *RadialSuite) TestInverseGaussian() {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    transformer := gohank.NewTransformFromRadius(0, &t.radius)
    for _, a := range []float64{2, 5, 10} {
        t.Run(fmt.Sprint(a), func() {
            ht := mat.NewVecDense(transformer.Radius().Len(), nil)
            a2 := math.Pow(a, 2)
            utils.ApplyVec(func(kr float64) float64 { return 2 * pi * (1 / (2 * a2)) * math.Exp(-math.Pow(kr, 2)/(4*a2)) }, ht, transformer.Kr())
            // ht = 2 * nppi * (1 / (2 * a * *2)) * np.exp(-transformer.kr**2/(4*a**2))
            actual_f := transformer.IQDHT(ht)
            expected_f := mat.NewVecDense(transformer.Radius().Len(), nil)
            utils.ApplyVec(func(r float64) float64 { return math.Exp(-a2 * math.Pow(r, 2)) }, expected_f, transformer.Radius())
            // expected_f = np.exp(-a * *2 * transformer.r * *2)
            testutils.AssertInDeltaVec(t.T(), expected_f, actual_f, -1, 1e-9)
        })
    }
}

// @pytest.mark.parametrize('axis', [0, 1])
// func (t *HankelTestSuit) test_gaussian_2d(axis int, radius, np.ndarray){
//     // Note the definition in Guizar-Sicairos varies by 2*pi in
//     // both scaling of the argument (so use kr rather than v) and
//     // scaling of the magnitude.
//     transformer = HankelTransform(order=0, radial_grid=radius)
//     a = np.linspace(2, 10)
//     dims_a = np.ones(2, np.int)
//     dims_a[1-axis] = len(a)
//     dims_r = np.ones(2, np.int)
//     dims_r[axis] = len(transformer.r)
//     a_reshaped = np.reshape(a, dims_a)
//     r_reshaped = np.reshape(transformer.r, dims_r)
//     kr_reshaped = np.reshape(transformer.kr, dims_r)
//     f = np.exp(-a_reshaped**2 * r_reshaped**2)
//     expected_ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
//     actual_ht = transformer.qdht(f, axis=axis)
//     assert np.allclose(expected_ht, actual_ht)

// @pytest.mark.parametrize('axis', [0, 1])
// func (t *HankelTestSuit) test_inverse_gaussian_2d(axis int, radius, np.ndarray){
//     // Note the definition in Guizar-Sicairos varies by 2*pi in
//     // both scaling of the argument (so use kr rather than v) and
//     // scaling of the magnitude.
//     transformer = HankelTransform(order=0, radial_grid=radius)
//     a = np.linspace(2, 10)
//     dims_a = np.ones(2, np.int)
//     dims_a[1-axis] = len(a)
//     dims_r = np.ones(2, np.int)
//     dims_r[axis] = len(transformer.r)
//     a_reshaped = np.reshape(a, dims_a)
//     r_reshaped = np.reshape(transformer.r, dims_r)
//     kr_reshaped = np.reshape(transformer.kr, dims_r)
//     ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
//     actual_f = transformer.iqdht(ht, axis=axis)
//     expected_f = np.exp(-a_reshaped ** 2 * r_reshaped ** 2)
//     assert np.allclose(expected_f, actual_f)

func (t *RadialSuite) Test1OverR2plusZ2() {
    // Note the definition in Guizar-Sicairos varies by 2*pi in
    // both scaling of the argument (so use kr rather than v) and
    // scaling of the magnitude.
    t.T().Skip("skipping as it requires a modifed bessel function of the second kind")
    transformer := gohank.NewTransform(0, 50, 1024)
    for _, a := range []float64{2, 5, 10} {
        t.Run(fmt.Sprint(a), func() {
            f := mat.NewVecDense(t.radius.Len(), nil)
            utils.ApplyVec(func(r float64) float64 { return 1 / (math.Pow(r, 2) + math.Pow(a, 2)) }, f, &t.radius)
            // f = 1 / (transformer.r**2 + a**2)
            // kn cannot handle complex arguments, so a must be real
            expected_ht := mat.NewVecDense(t.radius.Len(), nil)
            utils.ApplyVec(func(k float64) float64 { return 2 * pi * math.Y0(a*k) }, expected_ht, &t.radius)
            //2 * np.pi * scipy_bessel.kn(0, a*transformer.kr)
            actual_ht := transformer.QDHT(f)
            // These tolerances are pretty loose, but there seems to be large
            // error here
            testutils.AssertInDeltaVec(t.T(), expected_ht, actual_ht, -1, 0.01)
            err := testutils.MeanAbsError(expected_ht, actual_ht)
            assert.Less(t.T(), err, 4e-3)
        })
    }

}

func sinc(x float64) float64 {
    return math.Sin(x) / x
}

func TestSinc(t *testing.T) {
    /*Tests from figure 1 of
      *"Computation of quasi-discrete Hankel transforms of the integer
      order for propagating optical wave fields"*
      Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
      J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
    */
    for _, p := range []int{1, 4} {
        t.Run(fmt.Sprint(p), func(t *testing.T) {
            transformer := gohank.NewTransform(p, 3, 256)
            v := transformer.V()
            gamma := 5.
            fun := mat.NewVecDense(v.Len(), nil)
            utils.ApplyVec(func(r float64) float64 { return sinc(2.0 * pi * gamma * r) }, fun, transformer.Radius())
            expected_ht := mat.NewVecDense(fun.Len(), nil)

            utils.ApplyVec(func(v_ float64) float64 {
                pf := float64(p)
                if v_ < gamma {
                    return (math.Pow(v_, pf) * math.Cos(pf*pi/2) /
                        (2 * pi * gamma * math.Sqrt(math.Pow(gamma, 2)-math.Pow(v_, 2)) *
                            math.Pow(gamma+math.Sqrt(math.Pow(gamma, 2)-math.Pow(v_, 2)), pf)))
                } else {
                    return (math.Sin(pf*math.Asin(gamma/v_)) /
                        (2 * pi * gamma * math.Sqrt(math.Pow(v_, 2)-math.Pow(gamma, 2))))
                }
            }, expected_ht, v)
            ht := transformer.QDHT(fun)
            maxHT := slices.Max(ht.(*mat.VecDense).RawVector().Data)
            for i := 0; i < expected_ht.Len(); i++ {
                // use the same error measure as the paper
                dynamical_error := 20 * math.Log10(math.Abs(expected_ht.AtVec(i)-ht.AtVec(i))/maxHT)

                threshold := -10.
                if v.AtVec(i) > gamma*1.25 || v.AtVec(i) < gamma*0.75 {
                    // threshold is lower for areas not close to gamma
                    threshold = -35
                }
                assert.Less(t, dynamical_error, threshold)

            }
        })
    }
}

// ------------------------
// End Known Transfom pairs
// ------------------------

// Internal test of generalised jinc func
func TestGeneralisedJincZero(t *testing.T) {
    for _, a := range []float64{1, 0.7, 0.1, 136., 1e-6} {
        for p := -10; p < 10; p++ {
            t.Run(fmt.Sprintf("%f, %d", a, p), func(t *testing.T) {
                if p == -1 {
                    t.Skip("Skipping test for p=-1 as 1/eps does not go to inf correctly")
                }
                eps := 1e-200
                if p == -2 {
                    eps = 1e-5 / a
                }
                v := mat.NewVecDense(2, []float64{0, eps})
                val := testutils.GeneralisedJinc(v, a, p)

                tolerance := 2e-9
                assert.InDelta(t, val.AtVec(0), val.AtVec(1), tolerance)
            })
        }
    }
}
*/
/*

@pytest.mark.parametrize('two_d_size', [1, 100, 27])
@pytest.mark.parametrize('axis', [0, 1])
def test_round_trip_2d(two_d_size: int, axis: int, radius: np.ndarray, transformer: HankelTransform):
    dims = np.ones(2, np.int) * two_d_size
    dims[axis] = radius.size
    func = np.random.random(dims)
    ht = transformer.qdht(func, axis=axis)
    reconstructed = transformer.iqdht(ht, axis=axis)
    assert np.allclose(func, reconstructed)


@pytest.mark.parametrize('two_d_size', [1, 100, 27])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_round_trip_3d(two_d_size: int, axis: int, radius: np.ndarray, transformer: HankelTransform):
    dims = np.ones(3, np.int) * two_d_size
    dims[axis] = radius.size
    func = np.random.random(dims)
    ht = transformer.qdht(func, axis=axis)
    reconstructed = transformer.iqdht(ht, axis=axis)
    assert np.allclose(func, reconstructed)





def test_initialisation_errors():
    r_1d = np.linspace(0, 1, 10)
    k_1d = r_1d.copy()
    r_2d = np.repeat(r_1d[:, np.newaxis], repeats=5, axis=1)
    k_2d = r_2d.copy()
    with pytest.raises(ValueError):
        // missing any radius or k info
        HankelTransform(order=0)
    with pytest.raises(ValueError):
        // missing n_points
        HankelTransform(order=0, max_radius=1)
    with pytest.raises(ValueError):
        // missing max_radius
        HankelTransform(order=0, n_points=10)
    with pytest.raises(ValueError):
        // radial_grid and n_points
        HankelTransform(order=0, radial_grid=r_1d, n_points=10)
    with pytest.raises(ValueError):
        // radial_grid and max_radius
        HankelTransform(order=0, radial_grid=r_1d, max_radius=1)

    with pytest.raises(ValueError):
        // k_grid and n_points
        HankelTransform(order=0, k_grid=k_1d, n_points=10)
    with pytest.raises(ValueError):
        // k_grid and max_radius
        HankelTransform(order=0, k_grid=k_1d, max_radius=1)
    with pytest.raises(ValueError):
        // k_grid and r_grid
        HankelTransform(order=0, k_grid=k_1d, radial_grid=r_1d)

    with pytest.raises(AssertionError):
        HankelTransform(order=0, radial_grid=r_2d)
    with pytest.raises(AssertionError):
        HankelTransform(order=0, radial_grid=k_2d)

    // no error
    _ = HankelTransform(order=0, max_radius=1, n_points=10)
    _ = HankelTransform(order=0, radial_grid=r_1d)
    _ = HankelTransform(order=0, k_grid=k_1d)


@pytest.mark.parametrize('n', [10, 100, 512, 1024])
@pytest.mark.parametrize('max_radius', [0.1, 10, 20, 1e6])
func (t *HankelTestSuit) test_r_creation_equivalence(n int, max_radius, float){
    transformer1 = HankelTransform(order=0, n_points=1024, max_radius=50)
    r = np.linspace(0, 50, 1024)
    transformer2 = HankelTransform(order=0, radial_grid=r)

    for key, val in transformer1.__dict__.items():
        if key == '_original_radial_grid':
            continue
        val2 = getattr(transformer2, key)
        if val is None:
            assert val2 is None
        else:
            assert np.allclose(val, val2)




@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('axis', [0, 1])
def test_round_trip_r_interpolation_2d(radius: np.ndarray, order: int, shape: Callable, axis: int):
    transformer = HankelTransform(order=order, radial_grid=radius)

    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    dims_amplitude = np.ones(2, np.int)
    dims_amplitude[1-axis] = 10
    amplitude = np.random.random(dims_amplitude)
    dims_radius = np.ones(2, np.int)
    dims_radius[axis] = len(radius)
    func = np.reshape(shape(radius), dims_radius) * np.reshape(amplitude, dims_amplitude)
    transform_func = transformer.to_transform_r(func, axis=axis)
    reconstructed_func = transformer.to_original_r(transform_func, axis=axis)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)


@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('axis', [0, 1])
def test_round_trip_k_interpolation_2d(radius: np.ndarray, order: int, shape: Callable, axis: int):
    k_grid = radius/10
    transformer = HankelTransform(order=order, k_grid=k_grid)

    // the function must be smoothish for interpolation
    // to work. Random every point doesn't work
    dims_amplitude = np.ones(2, np.int)
    dims_amplitude[1-axis] = 10
    amplitude = np.random.random(dims_amplitude)
    dims_k = np.ones(2, np.int)
    dims_k[axis] = len(radius)
    func = np.reshape(shape(k_grid), dims_k) * np.reshape(amplitude, dims_amplitude)
    transform_func = transformer.to_transform_k(func, axis=axis)
    reconstructed_func = transformer.to_original_k(transform_func, axis=axis)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)





@pytest.mark.parametrize('two_d_size', [1, 100, 27])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('a', [1, 0.7, 0.1])
def test_jinc2d(transformer: HankelTransform, a: float, axis: int, two_d_size: int):
    f = generalised_jinc(transformer.r, a, transformer.order)
    second_axis = np.outer(np.linspace(0, 6, two_d_size), f)
    expected_ht = generalised_top_hat(transformer.v, a, transformer.order)
    if axis == 0:
        f_array = np.outer(f, second_axis)
        expected_ht_array = np.outer(expected_ht, second_axis)
    else:
        f_array = np.outer(second_axis, f)
        expected_ht_array = np.outer(second_axis, expected_ht)
    actual_ht = transformer.qdht(f_array, axis=axis)
    error = np.mean(np.abs(expected_ht_array-actual_ht))
    assert error < 1e-3
*/
