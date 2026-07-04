use hankrs::one_shot::qdht;
use ndarray::{Array1, Axis};
use plotters::prelude::*;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Determine the project root using CARGO_MANIFEST_DIR
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let image_dir = format!("{}/docs/src/images", manifest_dir);
    let image_path = format!("{}/simple_plot.png", image_dir);

    // 2. Create the images directory if it doesn't exist
    fs::create_dir_all(&image_dir)?;

    // 3. Create a simple top-hat function
    let radius = Array1::linspace(0.0, 5.0, 200);
    let f = radius.mapv(|r| if r < 1.0 { 1.0 } else { 0.0 });
    
    // 4. Transform
    let (kr, ht) = qdht(radius.clone(), &f, 0, Axis(0));

    // 5. Plot the result
    let root = BitMapBackend::new(&image_path, (800, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("QDHT of a Top Hat Function", ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..30f64, -0.5f64..1.5f64)?;

    chart.configure_mesh()
        .x_desc("k_r")
        .y_desc("Amplitude")
        .draw()?;

    let points: Vec<(f64, f64)> = kr.into_iter().zip(ht.into_iter()).map(|(x, y)| (x, y)).collect();

    chart.draw_series(LineSeries::new(points, &RED))?;

    root.present()?;

    Ok(())
}
