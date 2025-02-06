use ndarray::Array2;

pub fn quantize_4bit(weights: &Array2<f32>) -> (Array2<u8>, f32, f32) {
    let min = weights.fold(f32::INFINITY, |acc, &x| acc.min(x));
    let max = weights.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let scale = (max - min) / 15.0; // 4-bit range: 0-15

    let quantized = weights.mapv(|x| ((x - min) / scale).round() as u8);
    (quantized, min, scale)
}

pub fn dequantize_4bit(quantized: &Array2<u8>, min: f32, scale: f32) -> Array2<f32> {
    quantized.mapv(|x| x as f32 * scale + min)
}