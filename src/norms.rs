use ndarray::{Array2, ArrayView2, ArrayView1};
use rayon::prelude::*;

pub fn rms_layernorm(
    input: &ArrayView2<f32>,
    weight: &ArrayView1<f32>,
    eps: f32,
) -> Array2<f32> {
    let result: Vec<f32> = input.outer_iter()
        .par_bridge()
        .flat_map(|row| {
            let variance = row.mapv(|x| x.powi(2)).mean().unwrap() + eps;
            let inv_rms = 1.0 / variance.sqrt();
            (&row * inv_rms * weight).to_vec()
        })
        .collect();

    Array2::from_shape_vec(input.dim(), result)
        .expect("Failed to create output array")
}