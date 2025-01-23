// examples/simple.rs
use lynthia_kernel::{lora::LoraLayer, losses::cross_entropy_loss};
use ndarray::{array, Array2};

fn main() {
    let lora = LoraLayer::new(4, 4, 2, 0.1);
    let input = Array2::zeros((2, 4));
    let output = lora.forward(&input.view());
    println!("LoRA output:\n{:?}", output);

    // Logits: 2 samples, 3 classes
    let logits = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // Shape [2, 3]
    
    // Labels: class indices (0-based), shape [2, 1]
    let labels = array![[2i32], [1i32]];  // Class indices for 3 classes
    
    let loss = cross_entropy_loss(&logits, &labels.view());
    println!("Cross entropy loss: {}", loss);
}