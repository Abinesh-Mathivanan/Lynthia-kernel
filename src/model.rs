use ndarray::Array2;
use crate::{attention::MultiHeadAttention, ffn::FeedForward, norms::RmsLayernorm, embedding::Embedding, linear::Linear};

pub struct LlamaLayer {
    attention: MultiHeadAttention,
    mlp: FeedForward,
    norm: RmsLayernorm,
}

pub struct LlamaModel {
    layers: Vec<LlamaLayer>,
    embedding: Embedding,
    output: Linear,
}

impl LlamaModel {
    pub fn new(num_layers: usize, hidden_size: usize, num_heads: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LlamaLayer {
                attention: MultiHeadAttention::new(hidden_size, num_heads),
                mlp: FeedForward::new(hidden_size),
                norm: RmsLayernorm::new(hidden_size),
            })
            .collect();

        Self {
            layers,
            embedding: Embedding::new(hidden_size),
            output: Linear::new(hidden_size, hidden_size),
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut x = self.embedding.forward(input);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.output.forward(&x)
    }
}

impl LlamaLayer {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let attention_output = self.attention.forward(x);
        let mlp_output = self.mlp.forward(&attention_output);
        self.norm.forward(&mlp_output)
    }
}