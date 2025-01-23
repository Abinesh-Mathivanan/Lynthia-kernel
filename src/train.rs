// src/train.rs
use crate::{model::LlamaModel, optimizers::AdamW};

pub fn fine_tune(
    model: &mut LlamaModel,
    dataset: &Dataset,
    optimizer: &mut AdamW,
    epochs: usize,
) {
    for epoch in 0..epochs {
        for (inputs, targets) in dataset.iter() {
            // Forward pass
            let outputs = model.forward(&inputs);
            let loss = cross_entropy_loss(&outputs, &targets);

            // Backward pass
            let grads = model.backward(&outputs, &targets);

            // Update parameters
            optimizer.step(&mut model.params(), &grads);
        }
    }
}