use ndarray::Array2;

pub struct FeedForward {
    linear1: Array2<f32>,
    linear2: Array2<f32>,
}

impl FeedForward {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            linear1: Array2::zeros((hidden_size, hidden_size)),
            linear2: Array2::zeros((hidden_size, hidden_size)),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = x.dot(&self.linear1);
        h.dot(&self.linear2)
    }
}