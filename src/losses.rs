// src/losses.rs
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

pub fn cross_entropy_loss(logits: &Array2<f32>, labels: &ArrayView2<i32>) -> f32 {
    // Check for compatible batch sizes and label dimensions
    assert_eq!(
        logits.shape()[0], 
        labels.shape()[0], 
        "Batch size mismatch: logits has {} samples, labels has {}", 
        logits.shape()[0], 
        labels.shape()[0]
    );
    assert_eq!(
        labels.shape()[1], 
        1, 
        "Labels must be class indices with shape [batch_size, 1]"
    );

    logits.axis_iter(Axis(0))
        .zip(labels.axis_iter(Axis(0)))
        .par_bridge()
        .map(|(logit_row, label_row)| {
            let max_logit = logit_row.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let logsumexp = logit_row.mapv(|x| (x - max_logit).exp()).sum().ln() + max_logit;
            
            let label_idx = label_row[0] as usize;  // Extract class index from [1] to [num_classes]
            if label_idx == 0 {  // Handle padding index if needed
                0.0
            } else {
                logsumexp - logit_row[label_idx]
            }
        })
        .sum::<f32>() / logits.shape()[0] as f32
}