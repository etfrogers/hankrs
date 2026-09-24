#[cfg(feature = "blas")]
extern crate blas_src;

use criterion::{Criterion, criterion_group, criterion_main};
use hankrs::HankelTransform;
use ndarray::Axis;
use std::hint::black_box;

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
    let mut group = c.benchmark_group("qdht1d");

    for size in [256, 1024].iter() {
        let transformer = HankelTransform::new(0, 10.0, *size).unwrap();
        let r = transformer.radius();
        let f = r.mapv(|rad| (-rad * rad).exp());

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            size,
            |b, &_n| b.iter(|| transformer.qdht(black_box(&f), Axis(0))),
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
            |b, &_n| b.iter(|| transformer.qdht(black_box(&f), Axis(1))),
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
            |b, &_n| b.iter(|| transformer.qdht(black_box(&f), Axis(2))),
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

    group.bench_function("iqdht_2d", |b| {
        b.iter(|| transformer.iqdht(black_box(&f), Axis(0)))
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
