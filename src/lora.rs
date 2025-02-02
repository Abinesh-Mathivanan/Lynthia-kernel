use ndarray::{Array2, ArrayView2, linalg::general_mat_mul};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct LoraLayer {
    base_weight: Array2<f32>,
    lora_a: Array2<f32>,
    lora_b: Array2<f32>,
    scaling: f32,
    learning_rate: f32,

    grad_a: Option<Array2<f32>>,
    grad_b: Option<Array2<f32>>,
}


impl LoraLayer {
    pub fn new(input_dim: usize, output_dim: usize, rank: usize, learning_rate: f32) -> Self {
        let scaling = 1.0 / rank as f32;
        let mut rng = rand::thread_rng();
        
        Self {
            base_weight: Array2::zeros((input_dim, output_dim)),
            lora_a: Array2::from_shape_fn((input_dim, rank), |_| rng.gen_range(-0.01..0.01)),
            lora_b: Array2::from_shape_fn((rank, output_dim), |_| rng.gen_range(-0.01..0.01)),
            scaling,
            learning_rate,
            grad_a: None,
            grad_b: None,
        }
    }

    pub fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut base = Array2::zeros((x.shape()[0], self.base_weight.shape()[1]));
        general_mat_mul(1.0, x, &self.base_weight, 0.0, &mut base);

        let mut lora = Array2::zeros((x.shape()[0], self.lora_a.shape()[1]));
        general_mat_mul(1.0, x, &self.lora_a, 0.0, &mut lora);
        general_mat_mul(self.scaling, &lora, &self.lora_b, 1.0, &mut base);

        base
    }

    pub fn backward(&mut self, x: &ArrayView2<f32>, grad_output: &ArrayView2<f32>) -> Array2<f32> {
        let grad_b = self.scaling * x.t().dot(grad_output);
        let grad_a = self.scaling * x.dot(&grad_output.dot(&self.lora_b.t()));
        
        self.grad_a = Some(grad_a);
        self.grad_b = Some(grad_b);

        grad_output.dot(&self.lora_b.t()) * self.scaling
    }

    pub fn update(&mut self) {
        return;
    }
}