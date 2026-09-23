#[cfg(feature = "blas")]
extern crate blas_src;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use hankrs::HankelTransform;
use ndarray::Axis;

fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("creation");

    for size in [256, 1024].iter() {
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            size,
            |b, &n| {
                b.iter(|| {
                    HankelTransform::new(black_box(0), black_box(10.0), black_box(n)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_transforming(c: &mut Criterion) {
    let mut group = c.benchmark_group("qdht");

    for size in [256, 1024].iter() {
        let transformer = HankelTransform::new(0, 10.0, *size).unwrap();
        let r = transformer.radius();
        let f = r.mapv(|rad| (-rad * rad).exp());

        group.bench_with_input(
            criterion::BenchmarkId::new("by_lines", size),
            size,
            |b, &_n| b.iter(|| transformer.qdht(black_box(&f), Axis(0))),
        );

        let t = transformer.transform_matrix();
        let scale_in = transformer.radius();
        let scale_out = transformer.frequency();
        group.bench_with_input(
            criterion::BenchmarkId::new("direct_1d", size),
            size,
            |b, &_n| {
                b.iter(|| {
                    let scaled = &f / &scale_in;
                    let mut out = t.dot(&scaled);
                    out *= &scale_out;
                    out
                })
            },
        );
    }
    group.finish();
}

fn bench_transforming2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("qdht2d");

    for size in [256, 1024].iter() {
        let transformer = HankelTransform::new(0, 10.0, *size).unwrap();
        let r = transformer.radius();
        let mut f = ndarray::Array2::<f64>::zeros((100, *size));
        for mut row in f.rows_mut() {
            row.assign(&r.mapv(|rad| (-rad * rad).exp()));
        }

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            size,
            |b, &_n| b.iter(|| transformer.qdht(&f, Axis(1))),
        );
    }
    group.finish();
}

fn bench_transforming3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("qdht3d");

    for size in [256].iter() {
        let transformer = HankelTransform::new(0, 10.0, *size).unwrap();
        let r = transformer.radius();
        let mut f = ndarray::Array3::<f64>::zeros((10, 10, *size));
        // Fill all rows along the last axis with the same function
        for mut row in f.rows_mut() {
            row.assign(&r.mapv(|rad| (-rad * rad).exp()));
        }

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            size,
            |b, &_n| b.iter(|| transformer.qdht(&f, Axis(2))),
        );
    }
    group.finish();
}

fn bench_beam_batch(c: &mut Criterion) {
    use num_complex::Complex;

    let mut group = c.benchmark_group("beam_batch_2048x128");
    let transformer = HankelTransform::new(0, 10.0, 2048).unwrap();
    let r = transformer.radius();
    let mut f = ndarray::Array2::<Complex<f64>>::zeros((2048, 128));
    for mut col in f.columns_mut() {
        col.assign(&r.mapv(|rad| Complex::new((-rad * rad).exp(), 0.1)));
    }

    group.bench_function("transform_by_lines", |b| {
        b.iter(|| transformer.iqdht(black_box(&f), Axis(0)))
    });

    let t = transformer.transform_matrix();
    group.bench_function("direct_2d_gemm", |b| {
        b.iter(|| {
            let f_re = f.mapv(|c| c.re);
            let f_im = f.mapv(|c| c.im);
            let res_re = t.dot(&f_re);
            let res_im = t.dot(&f_im);
            ndarray::Zip::from(&res_re)
                .and(&res_im)
                .map_collect(|&re, &im| Complex::new(re, im))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_creation,
    bench_transforming,
    bench_transforming2d,
    bench_transforming3d,
    bench_beam_batch
);
criterion_main!(benches);

