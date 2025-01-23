// src/lora.rs
use ndarray::{Array2, ArrayView2, linalg::general_mat_mul};

#[derive(Debug, Clone)]
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

    pub fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut base = Array2::zeros((x.shape()[0], self.base_weight.shape()[1]));
        general_mat_mul(1.0, x, &self.base_weight, 0.0, &mut base);

        let mut lora = Array2::zeros((x.shape()[0], self.lora_a.shape()[1]));
        general_mat_mul(1.0, x, &self.lora_a, 0.0, &mut lora);
        general_mat_mul(self.scaling, &lora, &self.lora_b, 0.0, &mut base);

        base
    }
}