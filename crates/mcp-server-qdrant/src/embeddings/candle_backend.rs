use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use super::EmbeddingProvider;

pub struct CandleEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
}

impl CandleEmbedder {
    pub fn new(model_name: &str) -> anyhow::Result<Self> {
        let device = Device::Cpu;

        let repo = Api::new()?.model(model_name.to_string());

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let dimension = config.hidden_size;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dimension,
        })
    }

    fn embed_sync(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

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

        let input_ids =
            Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)?;
        let type_ids =
            Tensor::from_vec(type_ids, (batch_size, max_len), &self.device)?;
        let attention_mask_tensor =
            Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)?;

        let output = self
            .model
            .forward(&input_ids, &type_ids, Some(&attention_mask_tensor))?;

        // Mean pooling with attention mask
        let mask_f32 = attention_mask_tensor.to_dtype(DType::F32)?;
        let mask_expanded = mask_f32.unsqueeze(2)?; // (batch, seq, 1)
        let masked_output = output.broadcast_mul(&mask_expanded)?;
        let summed = masked_output.sum(1)?; // (batch, hidden)
        let mask_sum = mask_expanded.sum(1)?.clamp(1e-9, f64::MAX)?; // (batch, 1)
        let mean_pooled = summed.broadcast_div(&mask_sum)?; // (batch, hidden)

        // L2 normalization
        let norms = mean_pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        let normalized = mean_pooled.broadcast_div(&norms)?;

        let normalized = normalized.to_vec2::<f32>()?;
        Ok(normalized)
    }
}

#[async_trait]
impl EmbeddingProvider for CandleEmbedder {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let mut results = self.embed_sync(&texts)?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
    }

    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_sync(texts)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
