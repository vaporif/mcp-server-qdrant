use std::sync::Mutex;

use async_trait::async_trait;
use hf_hub::api::sync::Api;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::EmbeddingProvider;

pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimension: usize,
}

impl OnnxEmbedder {
    pub fn new(model_name: &str) -> anyhow::Result<Self> {
        let repo = Api::new()?.model(model_name.to_string());

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let onnx_path = repo.get("onnx/model.onnx")?;

        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let dimension = config["hidden_size"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing hidden_size in config.json"))?;
        let dimension =
            usize::try_from(dimension).map_err(|_| anyhow::anyhow!("hidden_size too large"))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let session = Session::builder()?.commit_from_file(&onnx_path)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimension,
        })
    }

    #[allow(clippy::cast_precision_loss)]
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

        let input_ids_tensor = Tensor::from_array(input_ids.clone())?;
        let attention_mask_tensor = Tensor::from_array(attention_mask.clone())?;
        let token_type_ids_tensor = Tensor::from_array(token_type_ids)?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock poisoned: {e}"))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;

        let (shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;
        // Shape derefs to &[i64], dims should be [batch_size, seq_len, hidden_size]
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
