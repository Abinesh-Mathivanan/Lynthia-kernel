use lynthia_kernel::{lora::LoraLayer, losses::cross_entropy_loss};
use ndarray::{s, Array2, Axis};
use rand::Rng;

fn main() {
    let input_dim = 4;
    let output_dim = 4;
    let rank = 2;
    let learning_rate = 0.01;
    let batch_size = 10;
    let mut lora = LoraLayer::new(input_dim, output_dim, rank, learning_rate);

    let num_samples = 100;
    let (inputs, labels) = generate_dataset(num_samples, input_dim, output_dim);

    let num_epochs = 10;
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let input_batch = inputs.slice(s![batch_start..batch_end, ..]);
            let label_batch = labels.slice(s![batch_start..batch_end, ..]);

            // Forward pass
            let output = lora.forward(&input_batch.view());
            let loss = cross_entropy_loss(&output, &label_batch.view());
            total_loss += loss;

            // Backward pass - calculate gradient manually
            let grad_output = output - label_batch.mapv(|x| x as f32);
            lora.backward(&input_batch.view(), &grad_output.view());
            lora.update();
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