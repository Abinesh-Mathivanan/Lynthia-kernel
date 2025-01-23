// src/lora.rs
use ndarray::Array2;

pub struct LoraLayer {
    base_weight: Array2<f32>,
    lora_a: Array2<f32>,
    lora_b: Array2<f32>,
    scaling: f32,
}

impl LoraLayer {
    pub fn new(input_dim: usize, output_dim: usize, rank: usize, scaling: f32) -> Self {
        Self {
            base_weight: Array2::zeros((input_dim, output_dim)),
            lora_a: Array2::zeros((input_dim, rank)),
            lora_b: Array2::zeros((rank, output_dim)),
            scaling,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let base = x.dot(&self.base_weight);
        let lora = x.dot(&self.lora_a).dot(&self.lora_b) * self.scaling;
        base + lora
    }
}