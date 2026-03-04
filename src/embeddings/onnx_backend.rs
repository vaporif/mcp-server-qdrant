use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use hf_hub::api::sync::Api;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::EmbeddingProvider;
use crate::errors::{Error, Result};

pub struct OnnxEmbedder {
    inner: Arc<OnnxInner>,
}

struct OnnxInner {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimension: usize,
}

// Safety: Session is guarded by Mutex, Tokenizer is thread-safe.
unsafe impl Send for OnnxInner {}
unsafe impl Sync for OnnxInner {}

impl OnnxEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let repo = Api::new()
            .map_err(|e| Error::ModelDownload(e.to_string()))?
            .model(model_name.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let onnx_path = repo
            .get("onnx/model.onnx")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;

        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| Error::ModelDownload(e.to_string()))?,
        )
        .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let dimension = config["hidden_size"]
            .as_u64()
            .ok_or_else(|| Error::Config("missing hidden_size in config.json".into()))?;
        let dimension = usize::try_from(dimension)
            .map_err(|_| Error::Config("hidden_size too large".into()))?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| Error::Tokenizer(e.to_string()))?;

        let session = Session::builder()
            .map_err(|e| Error::Embedding(e.to_string()))?
            .commit_from_file(&onnx_path)
            .map_err(|e| Error::Embedding(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(OnnxInner {
                session: Mutex::new(session),
                tokenizer,
                dimension,
            }),
        })
    }
}

impl OnnxInner {
    #[allow(clippy::cast_precision_loss)]
    fn embed_sync(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &id) in encoding.get_ids().iter().enumerate() {
                input_ids[[i, j]] = i64::from(id);
            }
            for (j, &mask) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[[i, j]] = i64::from(mask);
            }
            for (j, &type_id) in encoding.get_type_ids().iter().enumerate() {
                token_type_ids[[i, j]] = i64::from(type_id);
            }
        }

        let input_ids_tensor =
            Tensor::from_array(input_ids.clone()).map_err(|e| Error::Embedding(e.to_string()))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask.clone())
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let token_type_ids_tensor =
            Tensor::from_array(token_type_ids).map_err(|e| Error::Embedding(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Embedding(format!("session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let (shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let hidden_size = shape
            .get(2)
            .and_then(|&d| usize::try_from(d).ok())
            .unwrap_or(self.dimension);

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let seq_len = max_len;
            let mut pooled = vec![0f32; hidden_size];
            let mut mask_sum = 0f32;

            for j in 0..seq_len {
                let mask_val = attention_mask[[i, j]] as f32;
                if mask_val > 0.0 {
                    let offset = (i * seq_len + j) * hidden_size;
                    for (k, p) in pooled.iter_mut().enumerate() {
                        *p += output_data[offset + k] * mask_val;
                    }
                    mask_sum += mask_val;
                }
            }

            if mask_sum > 0.0 {
                for p in &mut pooled {
                    *p /= mask_sum;
                }
            }

            // L2 normalization
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for p in &mut pooled {
                *p /= norm;
            }

            results.push(pooled);
        }

        Ok(results)
    }
}

#[async_trait]
impl EmbeddingProvider for OnnxEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let inner = self.inner.clone();
        let text = text.to_string();

        tokio::task::spawn_blocking(move || {
            let texts = vec![text];
            let mut results = inner.embed_sync(&texts)?;
            results
                .pop()
                .ok_or_else(|| Error::Embedding("empty embedding result".into()))
        })
        .await
        .map_err(|e| Error::Embedding(format!("spawn_blocking join: {e}")))?
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        let texts = texts.to_vec();

        tokio::task::spawn_blocking(move || inner.embed_sync(&texts))
            .await
            .map_err(|e| Error::Embedding(format!("spawn_blocking join: {e}")))?
    }

    fn dimension(&self) -> usize {
        self.inner.dimension
    }
}
