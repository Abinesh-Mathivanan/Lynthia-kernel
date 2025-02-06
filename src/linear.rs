use ndarray::Array2;

pub struct Linear {
    weight: Array2<f32>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weight: Array2::zeros((input_dim, output_dim)),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.weight)
    }
}