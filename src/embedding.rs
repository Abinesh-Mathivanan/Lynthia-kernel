// src/embedding.rs
use ndarray::Array2;

pub struct Embedding {
    weight: Array2<f32>,
}

impl Embedding {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            weight: Array2::zeros((hidden_size, hidden_size)),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.weight)
    }
}