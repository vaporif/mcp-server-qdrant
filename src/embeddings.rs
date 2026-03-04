use async_trait::async_trait;

use crate::errors::{Error, Result};

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text).await
    }
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

#[cfg(feature = "candle")]
mod candle_backend;
#[cfg(feature = "candle")]
pub use candle_backend::CandleEmbedder;

#[cfg(any(feature = "onnx", feature = "onnx-fetch"))]
mod onnx_backend;
#[cfg(any(feature = "onnx", feature = "onnx-fetch"))]
pub use onnx_backend::OnnxEmbedder;

pub async fn create_embedding_provider(model_name: &str) -> Result<Box<dyn EmbeddingProvider>> {
    #[cfg(feature = "candle")]
    {
        return Ok(Box::new(CandleEmbedder::new(model_name).await?));
    }

    #[cfg(all(any(feature = "onnx", feature = "onnx-fetch"), not(feature = "candle")))]
    {
        return Ok(Box::new(OnnxEmbedder::new(model_name).await?));
    }

    #[allow(unreachable_code)]
    Err(Error::Config(
        "no embedding backend enabled — compile with 'candle' or 'onnx' feature".into(),
    ))
}
