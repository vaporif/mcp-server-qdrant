use async_trait::async_trait;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
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

pub fn create_embedding_provider(model_name: &str) -> anyhow::Result<Box<dyn EmbeddingProvider>> {
    #[cfg(feature = "candle")]
    {
        return Ok(Box::new(CandleEmbedder::new(model_name)?));
    }

    #[cfg(all(any(feature = "onnx", feature = "onnx-fetch"), not(feature = "candle")))]
    {
        return Ok(Box::new(OnnxEmbedder::new(model_name)?));
    }

    #[allow(unreachable_code)]
    Err(anyhow::anyhow!(
        "no embedding backend enabled — compile with 'candle' or 'onnx' feature"
    ))
}
