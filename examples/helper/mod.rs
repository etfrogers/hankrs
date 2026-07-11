#![allow(dead_code)]
use ndarray::{Array1, ArrayView1, ArrayView2};
use plotters::prelude::*;
use std::{fs, ops::Range};

pub fn get_image_path(filename: &str) -> String {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let image_dir = format!("{}/docs/src/images", manifest_dir);
    fs::create_dir_all(&image_dir).unwrap();
    format!("{}/{}", image_dir, filename)
}

pub fn gauss1d(x: ArrayView1<f64>, x0: f64, fwhm: f64) -> Array1<f64> {
    let factor = -2.0 * 2.0f64.ln() / (fwhm * fwhm);
    x.mapv(|val| (factor * (val - x0).powi(2)).exp())
}

pub fn plot_1d(
    filename: &str,
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    title: &str,
    xlabel: &str,
    ylabel: &str,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_image_path(filename);
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart
        .configure_mesh()
        .x_desc(xlabel)
        .y_desc(ylabel)
        .draw()?;

    let points: Vec<(f64, f64)> = x
        .into_iter()
        .zip(y.into_iter())
        .map(|(a, b)| (*a, *b))
        .collect();
    chart.draw_series(LineSeries::new(points, &BLUE))?;

    root.present()?;
    Ok(())
}

pub fn plot_1d_compare(
    filename: &str,
    x1: ArrayView1<f64>,
    y1: ArrayView1<f64>,
    x2: ArrayView1<f64>,
    y2: ArrayView1<f64>,
    label1: &str,
    label2: &str,
    title: &str,
    xlabel: &str,
    ylabel: &str,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_image_path(filename);
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart
        .configure_mesh()
        .x_desc(xlabel)
        .y_desc(ylabel)
        .draw()?;

    let p1: Vec<(f64, f64)> = x1
        .into_iter()
        .zip(y1.into_iter())
        .map(|(a, b)| (*a, *b))
        .collect();
    chart
        .draw_series(LineSeries::new(p1, &BLUE))?
        .label(label1)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    let p2: Vec<(f64, f64)> = x2
        .into_iter()
        .zip(y2.into_iter())
        .map(|(a, b)| (*a, *b))
        .collect();
    chart
        .draw_series(PointSeries::of_element(p2, 3, &RED, &|c, s, st| {
            return EmptyElement::at(c) + Cross::new((0, 0), s, st.filled());
        }))?
        .label(label2)
        .legend(|(x, y)| Cross::new((x, y), 3, &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

pub fn plot_1d_original_and_transform(
    filename: &str,
    x_orig: ArrayView1<f64>,
    y_orig: ArrayView1<f64>,
    title_orig: &str,
    xlabel_orig: &str,
    ylabel_orig: &str,
    x_range_orig: std::ops::Range<f64>,
    y_range_orig: std::ops::Range<f64>,
    x1_trans: ArrayView1<f64>,
    y1_trans: ArrayView1<f64>,
    x2_trans: ArrayView1<f64>,
    y2_trans: ArrayView1<f64>,
    label1: &str,
    label2: &str,
    title_trans: &str,
    xlabel_trans: &str,
    ylabel_trans: &str,
    x_range_trans: std::ops::Range<f64>,
    y_range_trans: std::ops::Range<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_image_path(filename);
    let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(300);

    let mut chart_orig = ChartBuilder::on(&upper)
        .caption(title_orig, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range_orig, y_range_orig)?;

    chart_orig
        .configure_mesh()
        .x_desc(xlabel_orig)
        .y_desc(ylabel_orig)
        .draw()?;

    let p_orig: Vec<(f64, f64)> = x_orig
        .into_iter()
        .zip(y_orig.into_iter())
        .map(|(a, b)| (*a, *b))
        .collect();
    chart_orig.draw_series(LineSeries::new(p_orig, &BLUE))?;

    let mut chart_trans = ChartBuilder::on(&lower)
        .caption(title_trans, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range_trans, y_range_trans)?;

    chart_trans
        .configure_mesh()
        .x_desc(xlabel_trans)
        .y_desc(ylabel_trans)
        .draw()?;

    let p1: Vec<(f64, f64)> = x1_trans
        .into_iter()
        .zip(y1_trans.into_iter())
        .map(|(a, b)| (*a, *b))
        .collect();
    chart_trans
        .draw_series(LineSeries::new(p1, &BLUE))?
        .label(label1)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    let p2: Vec<(f64, f64)> = x2_trans
        .into_iter()
        .zip(y2_trans.into_iter())
        .map(|(a, b)| (*a, *b))
        .collect();
    chart_trans
        .draw_series(PointSeries::of_element(p2, 3, &RED, &|c, s, st| {
            return EmptyElement::at(c) + Cross::new((0, 0), s, st.filled());
        }))?
        .label(label2)
        .legend(|(x, y)| Cross::new((x, y), 3, &RED));

    chart_trans
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn imagesc(
    filename: &str,
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    z: ArrayView2<f64>,
    title: &str,
    xlabel: &str,
    ylabel: &str,
    y_range: Range<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_image_path(filename);
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = *x.first().unwrap_or(&0.0);
    let x_max = *x.last().unwrap_or(&1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_range.clone())?;

    chart
        .configure_mesh()
        .x_desc(xlabel)
        .y_desc(ylabel)
        .draw()?;

    let y_min = *y.first().unwrap_or(&0.0);
    let y_max = *y.last().unwrap_or(&1.0);
    let (nx, ny) = (x.len(), y.len());
    let dx = (x_max - x_min) / (nx as f64);
    let dy = (y_max - y_min) / (ny as f64);

    let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
    let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_range = z_max - z_min;

    // Use a color map (Viridis-like or simple heatmap)
    chart.draw_series(x.iter().enumerate().flat_map(|(i, &xi)| {
        y.iter().enumerate().map(move |(j, &yi)| {
            // array2 is typically accessed by (row, col) which corresponds to (y, x) or (x, y) depending on conventions.
            // python example: z = Irz, dimensions: z.shape == (nr, Nz), so (y, x).
            let val = z[[j, i]];
            let norm = if z_range == 0.0 {
                0.0
            } else {
                (val - z_min) / z_range
            };

            // HSL interpolation for heatmap (blue -> red)
            let hue = (1.0 - norm) * 240.0; // 240 is blue, 0 is red
            let color = HSLColor(hue / 360.0, 1.0, 0.5);

            Rectangle::new([(xi, yi), (xi + dx, yi + dy)], color.filled())
        })
    }))?;

    root.present()?;
    Ok(())
}
