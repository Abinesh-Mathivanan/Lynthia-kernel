use lynthia_kernel::{lora::LoraLayer, losses::cross_entropy_loss};
use ndarray::{Array2, Axis};
use rand::Rng;

struct MultiLayerNetwork {
    lora1: LoraLayer,
    lora2: LoraLayer,
}

impl MultiLayerNetwork {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, rank: usize, learning_rate: f32) -> Self {
        let lora1 = LoraLayer::new(input_dim, hidden_dim, rank, learning_rate);
        let lora2 = LoraLayer::new(hidden_dim, output_dim, rank, learning_rate);
        Self { lora1, lora2 }
    }

    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let hidden = self.lora1.forward(&input.view());
        self.lora2.forward(&hidden.view())
    }

    fn backward(&mut self, input: &Array2<f32>, label: &Array2<i32>) {
        // Convert to views before passing to forward/backward
        let input_view = input.view();
        let hidden = self.lora1.forward(&input_view);
        let output = self.lora2.forward(&hidden.view());
    
        // Calculate gradients with proper view handling
        let grad_output = output - label.mapv(|x| x as f32);
        let grad_hidden = self.lora2.backward(&hidden.view(), &grad_output.view());
        self.lora1.backward(&input_view, &grad_hidden.view());
    }

    fn update(&mut self) {
        self.lora1.update();
        self.lora2.update();
    }
}

fn main() {
    let input_dim = 4;
    let hidden_dim = 8;
    let output_dim = 4;
    let rank = 2;
    let learning_rate = 0.01;
    let mut network = MultiLayerNetwork::new(input_dim, hidden_dim, output_dim, rank, learning_rate);

    let num_samples = 100;
    let (inputs, labels) = generate_dataset(num_samples, input_dim, output_dim);

    let num_epochs = 10;
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        for (input, label) in inputs.axis_iter(Axis(0)).zip(labels.axis_iter(Axis(0))) {
            let input = input.to_owned().insert_axis(Axis(0));
            let label = label.to_owned().insert_axis(Axis(0));
            
            let output = network.forward(&input);
            let loss = cross_entropy_loss(&output, &label.view());
            total_loss += loss;
            
            network.backward(&input, &label);
            network.update();
        }

        println!("Epoch {}: Loss = {}", epoch, total_loss / num_samples as f32);
    }
}

fn generate_dataset(num_samples: usize, input_dim: usize, output_dim: usize) -> (Array2<f32>, Array2<i32>) {
    let mut rng = rand::thread_rng();
    let mut inputs = Array2::zeros((num_samples, input_dim));
    let mut labels = Array2::zeros((num_samples, 1));

    for i in 0..num_samples {
        for j in 0..input_dim {
            inputs[[i, j]] = rng.gen_range(-1.0..1.0);
        }
        labels[[i, 0]] = rng.gen_range(0..output_dim as i32);
    }

    (inputs, labels)
}