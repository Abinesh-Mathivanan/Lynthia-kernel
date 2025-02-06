use ndarray::Array2;

pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Array2<f32>>, 
    v: Vec<Array2<f32>>, 
    t: usize,            
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    pub fn step(&mut self, params: &mut [Array2<f32>], grads: &[Array2<f32>]) {
        self.t += 1;
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            self.m[i] = self.beta1 * &self.m[i] + (1.0 - self.beta1) * grad;
            self.v[i] = self.beta2 * &self.v[i] + (1.0 - self.beta2) * grad.mapv(|x| x.powi(2));

            let m_hat = &self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = &self.v[i] / (1.0 - self.beta2.powi(self.t as i32));

            *param -= self.lr * m_hat / (v_hat.mapv(|x| x.sqrt()) + self.eps);
        }
    }
}