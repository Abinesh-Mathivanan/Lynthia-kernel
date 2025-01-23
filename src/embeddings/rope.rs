use ndarray::{Array3, ArrayView2, s};
use rayon::prelude::*;

pub fn apply_rope(
    q: &mut Array3<f32>,
    k: &mut Array3<f32>,
    cos: &ArrayView2<f32>,
    sin: &ArrayView2<f32>,
) {
    let (_batch_size, seq_len, n_heads_head_dim) = q.dim();
    let head_dim = n_heads_head_dim / cos.shape()[1]; // Calculate head_dim from cos/sin dims

    q.outer_iter_mut()
        .zip(k.outer_iter_mut())
        .par_bridge()
        .for_each(|(mut q_head, mut k_head)| {
            for pos in 0..seq_len {
                for h in 0..cos.shape()[1] {
                    let cos_val = cos[[pos % cos.shape()[0], h]];
                    let sin_val = sin[[pos % sin.shape()[0], h]];
                    
                    let start = h * head_dim;
                    let end = start + head_dim;
                    
                    let mut q_slice = q_head.slice_mut(s![pos, start..end]);
                    let mut k_slice = k_head.slice_mut(s![pos, start..end]);
                    
                    for i in 0..(head_dim / 2) {
                        let q0 = q_slice[i];
                        let q1 = q_slice[i + head_dim/2];
                        q_slice[i] = q0 * cos_val - q1 * sin_val;
                        q_slice[i + head_dim/2] = q1 * cos_val + q0 * sin_val;

                        let k0 = k_slice[i];
                        let k1 = k_slice[i + head_dim/2];
                        k_slice[i] = k0 * cos_val - k1 * sin_val;
                        k_slice[i + head_dim/2] = k1 * cos_val + k0 * sin_val;
                    }
                }
            }
        });
}