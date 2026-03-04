use std::collections::HashMap;
use std::sync::Arc;

use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance,
    FieldType as QdrantFieldType, Filter, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder, value::Kind,
};
use qdrant_client::{Payload, Qdrant};

use crate::config::{QdrantConfig, QdrantLocation};
use crate::embeddings::EmbeddingProvider;
use crate::errors::{Error, Result};
use crate::filters::FieldType;

pub type Metadata = HashMap<String, serde_json::Value>;

pub struct Entry {
    pub content: String,
    pub metadata: Option<Metadata>,
}

pub struct QdrantConnector {
    client: Qdrant,
    embedding: Arc<dyn EmbeddingProvider>,
    indexes: Vec<crate::filters::FilterableField>,
}

impl QdrantConnector {
    #[allow(clippy::result_large_err)]
    pub fn new(config: &QdrantConfig, embedding: Arc<dyn EmbeddingProvider>) -> Result<Self> {
        let client = match &config.location {
            QdrantLocation::Remote { url, api_key } => {
                let mut builder = Qdrant::from_url(url);
                if let Some(key) = api_key {
                    builder = builder.api_key(key.as_str());
                }
                builder.build().map_err(Error::Qdrant)?
            }
            QdrantLocation::Local { path } => {
                let url = "http://localhost:6334";
                tracing::warn!(
                    "Local path {path:?} specified but qdrant-client only supports remote. Connecting to {url}"
                );
                Qdrant::from_url(url).build().map_err(Error::Qdrant)?
            }
        };

        Ok(Self {
            client,
            embedding,
            indexes: config.filterable_fields.clone(),
        })
    }

    pub async fn ensure_collection(&self, collection_name: &str) -> Result<()> {
        if self.client.collection_exists(collection_name).await? {
            return Ok(());
        }

        let dim = self.embedding.dimension() as u64;
        self.client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
            )
            .await?;

        for field in &self.indexes {
            let qdrant_type = match field.field_type {
                FieldType::Keyword => QdrantFieldType::Keyword,
                FieldType::Integer => QdrantFieldType::Integer,
                FieldType::Float => QdrantFieldType::Float,
                FieldType::Boolean => QdrantFieldType::Bool,
            };
            self.client
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    collection_name,
                    &field.name,
                    qdrant_type,
                ))
                .await?;
        }

        Ok(())
    }

    pub async fn store(&self, entry: Entry, collection_name: &str) -> Result<()> {
        self.ensure_collection(collection_name).await?;

        let vector = self
            .embedding
            .embed(&entry.content)
            .await
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let mut payload = Payload::new();
        payload.insert("document", entry.content.as_str());
        if let Some(metadata) = &entry.metadata {
            let mut map = serde_json::Map::new();
            for (key, value) in metadata {
                map.insert(key.clone(), value.clone());
            }
            if let Ok(p) = Payload::try_from(serde_json::Value::Object(map)) {
                for (k, v) in HashMap::<String, qdrant_client::qdrant::Value>::from(p) {
                    payload.insert(k, v);
                }
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let point = PointStruct::new(id, vector, payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(collection_name, vec![point]).wait(true))
            .await?;

        Ok(())
    }

    pub async fn search(
        &self,
        query: &str,
        collection_name: &str,
        limit: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<Entry>> {
        self.ensure_collection(collection_name).await?;

        let vector = self
            .embedding
            .embed(query)
            .await
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let mut request = QueryPointsBuilder::new(collection_name)
            .query(qdrant_client::qdrant::Query::new_nearest(vector))
            .limit(limit as u64)
            .with_payload(true);

        if let Some(f) = filter {
            request = request.filter(f);
        }

        let response = self.client.query(request).await?;

        let entries = response
            .result
            .into_iter()
            .filter_map(|point| {
                let content = extract_string(&point.payload, "document")?;

                let metadata: Metadata = point
                    .payload
                    .iter()
                    .filter(|(k, _)| k.as_str() != "document")
                    .filter_map(|(k, v)| qdrant_value_to_json(v).map(|jv| (k.clone(), jv)))
                    .collect();

                let metadata = if metadata.is_empty() {
                    None
                } else {
                    Some(metadata)
                };

                Some(Entry { content, metadata })
            })
            .collect();

        Ok(entries)
    }
}

/// Convert a JSON value to a Qdrant Filter.
///
/// Expects the Qdrant REST API filter format, e.g.:
/// ```json
/// { "must": [{ "key": "city", "match": { "value": "London" } }] }
/// ```
/// Convert a JSON value to a Qdrant Filter.
///
/// Expects the Qdrant REST API filter format, e.g.:
/// ```json
/// { "must": [{ "key": "city", "match": { "value": "London" } }] }
/// ```
#[allow(clippy::result_large_err)]
pub fn json_to_qdrant_filter(value: &serde_json::Value) -> Result<Filter> {
    use qdrant_client::qdrant::Condition;

    #[allow(clippy::result_large_err)]
    fn parse_condition(c: &serde_json::Value) -> Result<Condition> {
        let key = c
            .get("key")
            .and_then(|k| k.as_str())
            .ok_or(Error::Config("filter condition missing 'key'".into()))?;

        if let Some(match_obj) = c.get("match") {
            if let Some(s) = match_obj.get("value").and_then(serde_json::Value::as_str) {
                return Ok(Condition::matches(key, s.to_string()));
            }
            if let Some(n) = match_obj.get("value").and_then(serde_json::Value::as_i64) {
                return Ok(Condition::matches(key, n));
            }
            if let Some(b) = match_obj.get("value").and_then(serde_json::Value::as_bool) {
                return Ok(Condition::matches(key, b));
            }
            return Err(Error::Config("unsupported match value type".into()));
        }

        Err(Error::Config(format!(
            "unsupported filter condition for key '{key}'"
        )))
    }

    #[allow(clippy::result_large_err)]
    fn parse_conditions(arr: &[serde_json::Value]) -> Result<Vec<Condition>> {
        arr.iter().map(parse_condition).collect()
    }

    let mut filter = Filter::default();

    if let Some(must) = value.get("must").and_then(|v| v.as_array()) {
        filter.must = parse_conditions(must)?;
    }
    if let Some(should) = value.get("should").and_then(|v| v.as_array()) {
        filter.should = parse_conditions(should)?;
    }
    if let Some(must_not) = value.get("must_not").and_then(|v| v.as_array()) {
        filter.must_not = parse_conditions(must_not)?;
    }

    Ok(filter)
}

fn extract_string(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> Option<String> {
    payload.get(key).and_then(|v| {
        if let Some(Kind::StringValue(s)) = &v.kind {
            Some(s.clone())
        } else {
            None
        }
    })
}

fn qdrant_value_to_json(value: &qdrant_client::qdrant::Value) -> Option<serde_json::Value> {
    value.kind.as_ref().map(|k| match k {
        Kind::NullValue(_) => serde_json::Value::Null,
        Kind::DoubleValue(v) => serde_json::json!(*v),
        Kind::IntegerValue(v) => serde_json::json!(*v),
        Kind::StringValue(v) => serde_json::Value::String(v.clone()),
        Kind::BoolValue(v) => serde_json::Value::Bool(*v),
        Kind::StructValue(s) => {
            let map: serde_json::Map<String, serde_json::Value> = s
                .fields
                .iter()
                .filter_map(|(k, v)| qdrant_value_to_json(v).map(|jv| (k.clone(), jv)))
                .collect();
            serde_json::Value::Object(map)
        }
        Kind::ListValue(l) => {
            let arr: Vec<serde_json::Value> =
                l.values.iter().filter_map(qdrant_value_to_json).collect();
            serde_json::Value::Array(arr)
        }
    })
}
