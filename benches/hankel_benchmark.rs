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
            |b, &n| b.iter(|| HankelTransform::new(black_box(0), black_box(10.0), black_box(n))),
        );
    }
    group.finish();
}

fn bench_transforming(c: &mut Criterion) {
    let mut group = c.benchmark_group("qdht");

    for size in [256, 1024].iter() {
        let transformer = HankelTransform::new(0, 10.0, *size);
        let r = transformer.radius();
        let f = r.mapv(|rad| (-rad * rad).exp());

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            size,
            |b, &_n| b.iter(|| transformer.qdht(&f, Axis(0))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_creation, bench_transforming);
criterion_main!(benches);
