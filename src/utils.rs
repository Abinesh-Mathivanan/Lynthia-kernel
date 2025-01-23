use ndarray::ArrayView2;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UnslothError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
}

pub fn validate_dims(
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
) -> Result<(), UnslothError> {
    if a.shape() != b.shape() {
        return Err(UnslothError::DimensionMismatch {
            expected: format!("{:?}", a.shape()),
            actual: format!("{:?}", b.shape()),
        });
    }
    Ok(())
}