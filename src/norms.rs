// src/norms.rs
use ndarray::{Array1, Array2};

pub struct RmsLayernorm {
    weight: Array1<f32>,
    eps: f32,
}

impl RmsLayernorm {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            weight: Array1::ones(hidden_size),
            eps: 1e-6,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let variance = x.mapv(|v| v.powi(2)).mean().unwrap() + self.eps;
        let inv_rms = 1.0 / variance.sqrt();
        x * inv_rms * &self.weight
    }
}