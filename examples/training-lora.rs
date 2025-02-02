use lynthia_kernel::{lora::LoraLayer, losses::cross_entropy_loss};
use ndarray::{Array2, Axis};
use rand::Rng;

fn main() {
    let input_dim = 4;
    let output_dim = 4;
    let rank = 2;
    let learning_rate = 0.01;
    let mut lora = LoraLayer::new(input_dim, output_dim, rank, learning_rate);

    let num_samples = 100;
    let (inputs, labels) = generate_dataset(num_samples, input_dim, output_dim);

    let num_epochs = 10;
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        for (input, label) in inputs.axis_iter(Axis(0)).zip(labels.axis_iter(Axis(0))) {
            let input = input.to_owned().insert_axis(Axis(0));
            let label = label.to_owned().insert_axis(Axis(0));
            
            let output = lora.forward(&input.view());
            let loss = cross_entropy_loss(&output, &label.view());
            total_loss += loss;

            let grad_output = output - label.mapv(|x| x as f32);
            lora.backward(&input.view(), &grad_output.view());
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