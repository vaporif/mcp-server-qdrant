use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("qdrant: {0}")]
    Qdrant(Box<qdrant_client::QdrantError>),

    #[error("embedding: {0}")]
    Embedding(String),

    #[error("config: {0}")]
    Config(String),

    #[error("no collection name provided and no default configured")]
    NoCollection,

    #[error("store not available in read-only mode")]
    ReadOnly,

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("model download: {0}")]
    ModelDownload(String),
}

impl From<qdrant_client::QdrantError> for Error {
    fn from(e: qdrant_client::QdrantError) -> Self {
        Self::Qdrant(Box::new(e))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
