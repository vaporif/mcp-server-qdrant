use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use super::EmbeddingProvider;
use crate::errors::{Error, Result};

pub struct CandleEmbedder {
    inner: Arc<CandleInner>,
}

struct CandleInner {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
}

impl CandleEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let device = Device::Cpu;

        let repo = Api::new()
            .map_err(|e| Error::ModelDownload(e.to_string()))?
            .model(model_name.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| Error::ModelDownload(e.to_string()))?,
        )
        .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let dimension = config.hidden_size;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| Error::Tokenizer(e.to_string()))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| Error::Embedding(e.to_string()))?
        };
        let model = BertModel::load(vb, &config).map_err(|e| Error::Embedding(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(CandleInner {
                model,
                tokenizer,
                device,
                dimension,
            }),
        })
    }
}

impl CandleInner {
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

        let mut input_ids = vec![0u32; batch_size * max_len];
        let mut type_ids = vec![0u32; batch_size * max_len];
        let mut attention_mask = vec![0u32; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let types = encoding.get_type_ids();
            let mask = encoding.get_attention_mask();
            let seq_len = ids.len();
            let offset = i * max_len;
            input_ids[offset..offset + seq_len].copy_from_slice(ids);
            type_ids[offset..offset + seq_len].copy_from_slice(types);
            attention_mask[offset..offset + seq_len].copy_from_slice(mask);
        }

        let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let type_ids = Tensor::from_vec(type_ids, (batch_size, max_len), &self.device)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let attention_mask_tensor =
            Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)
                .map_err(|e| Error::Embedding(e.to_string()))?;

        let output = self
            .model
            .forward(&input_ids, &type_ids, Some(&attention_mask_tensor))
            .map_err(|e| Error::Embedding(e.to_string()))?;

        // Mean pooling with attention mask
        let mask_f32 = attention_mask_tensor
            .to_dtype(DType::F32)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let mask_expanded = mask_f32
            .unsqueeze(2)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let masked_output = output
            .broadcast_mul(&mask_expanded)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let summed = masked_output
            .sum(1)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let mask_sum = mask_expanded
            .sum(1)
            .and_then(|t| t.clamp(1e-9, f64::MAX))
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let mean_pooled = summed
            .broadcast_div(&mask_sum)
            .map_err(|e| Error::Embedding(e.to_string()))?;

        // L2 normalization
        let norms = mean_pooled
            .sqr()
            .and_then(|t| t.sum_keepdim(1))
            .and_then(|t| t.sqrt())
            .and_then(|t| t.clamp(1e-12, f64::MAX))
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let normalized = mean_pooled
            .broadcast_div(&norms)
            .map_err(|e| Error::Embedding(e.to_string()))?;

        normalized
            .to_vec2::<f32>()
            .map_err(|e| Error::Embedding(e.to_string()))
    }
}

// Safety: BertModel uses CPU tensors only (no GPU pointers), Tokenizer is thread-safe.
unsafe impl Send for CandleInner {}
unsafe impl Sync for CandleInner {}

#[async_trait]
impl EmbeddingProvider for CandleEmbedder {
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
