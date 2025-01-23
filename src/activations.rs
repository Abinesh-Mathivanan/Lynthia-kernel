// src/activations.rs
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

pub fn swiglu(e: &ArrayView2<f32>, g: &ArrayView2<f32>) -> Array2<f32> {
    assert_eq!(e.shape(), g.shape(), "Input arrays must have the same shape");
    
    let shape = e.shape();
    let (rows, cols) = (shape[0], shape[1]);
    
    let result: Vec<f32> = e
        .iter()
        .zip(g.iter())
        .par_bridge()  // Convert to parallel iterator
        .map(|(e_val, g_val)| {
            let sigmoid = 1.0 / (1.0 + (-*e_val).exp());
            (e_val * sigmoid) * g_val
        })
        .collect();

    Array2::from_shape_vec((rows, cols), result)
        .expect("Failed to create array from collected vector")
}