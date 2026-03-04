use std::borrow::Cow;
use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ErrorData as McpError, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::config::Config;
use crate::qdrant::{Entry, QdrantConnector};

const DEFAULT_STORE_DESCRIPTION: &str =
    "Keep the memory for later use, when you are asked to remember something.";
const DEFAULT_FIND_DESCRIPTION: &str = "Look up memories in Qdrant. Use this tool when you need to: \n - Find memories by their content \n - Access memories for further analysis \n - Get some personal information about the user";

#[derive(Deserialize, JsonSchema)]
pub struct StoreParams {
    /// The information to store
    pub information: String,
    /// Optional metadata as JSON object
    #[serde(default)]
    pub metadata: Option<Value>,
    /// Collection name (uses default if not provided)
    #[serde(default)]
    pub collection_name: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct FindParams {
    /// The search query
    pub query: String,
    /// Collection name (uses default if not provided)
    #[serde(default)]
    pub collection_name: Option<String>,
    /// Optional Qdrant filter as JSON
    #[serde(default)]
    pub query_filter: Option<Value>,
}

#[derive(Clone)]
pub struct QdrantMcpServer {
    connector: Arc<QdrantConnector>,
    config: Arc<Config>,
    tool_router: ToolRouter<Self>,
}

impl QdrantMcpServer {
    pub fn new(connector: Arc<QdrantConnector>, config: Arc<Config>) -> Self {
        let mut tool_router = Self::tool_router();

        // Override tool descriptions from config or defaults
        let store_desc: Cow<'static, str> = config
            .tools
            .store_description
            .clone()
            .map_or(Cow::Borrowed(DEFAULT_STORE_DESCRIPTION), Cow::Owned);
        let find_desc: Cow<'static, str> = config
            .tools
            .find_description
            .clone()
            .map_or(Cow::Borrowed(DEFAULT_FIND_DESCRIPTION), Cow::Owned);

        if let Some(route) = tool_router.map.get_mut("qdrant-store") {
            route.attr.description = Some(store_desc);
        }
        if let Some(route) = tool_router.map.get_mut("qdrant-find") {
            route.attr.description = Some(find_desc);
        }

        Self {
            connector,
            config,
            tool_router,
        }
    }

    fn resolve_collection(&self, param: Option<&String>) -> Result<String, McpError> {
        param
            .or(self.config.qdrant.collection_name.as_ref())
            .cloned()
            .ok_or_else(|| {
                McpError::invalid_params(
                    "no collection name provided and no default configured",
                    None,
                )
            })
    }
}

#[tool_router]
impl QdrantMcpServer {
    #[tool(
        name = "qdrant-store",
        description = "Store information in the Qdrant vector database"
    )]
    async fn store(
        &self,
        Parameters(params): Parameters<StoreParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.config.qdrant.read_only {
            return Err(McpError::invalid_request(
                "store not available in read-only mode",
                None,
            ));
        }

        let collection = self.resolve_collection(params.collection_name.as_ref())?;

        let metadata = params.metadata.and_then(|v| {
            if let Value::Object(map) = v {
                Some(
                    map.into_iter()
                        .collect::<std::collections::HashMap<String, Value>>(),
                )
            } else {
                None
            }
        });

        let entry = Entry {
            content: params.information,
            metadata,
        };

        self.connector
            .store(entry, &collection)
            .await
            .map_err(|e| McpError::internal_error(format!("store failed: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "stored in collection '{collection}'"
        ))]))
    }

    #[tool(
        name = "qdrant-find",
        description = "Find information in the Qdrant vector database using semantic search"
    )]
    async fn find(
        &self,
        Parameters(params): Parameters<FindParams>,
    ) -> Result<CallToolResult, McpError> {
        let collection = self.resolve_collection(params.collection_name.as_ref())?;

        if params.query_filter.is_some() && !self.config.qdrant.allow_arbitrary_filter {
            return Err(McpError::invalid_params(
                "arbitrary filters are disabled — set QDRANT_ALLOW_ARBITRARY_FILTER=true to enable",
                None,
            ));
        }

        let filter = params
            .query_filter
            .map(|f| crate::qdrant::json_to_qdrant_filter(&f))
            .transpose()
            .map_err(|e| McpError::invalid_params(format!("invalid filter: {e}"), None))?;

        let entries = self
            .connector
            .search(
                &params.query,
                &collection,
                self.config.qdrant.search_limit,
                filter,
            )
            .await
            .map_err(|e| McpError::internal_error(format!("search failed: {e}"), None))?;

        if entries.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "no results found",
            )]));
        }

        let text = format_entries(&entries);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for QdrantMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            Implementation::new("mcp-server-qdrant", env!("CARGO_PKG_VERSION")),
        )
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn format_entries(entries: &[crate::qdrant::Entry]) -> String {
    let mut out = String::new();
    for entry in entries {
        out.push_str("<entry>");
        out.push_str("<content>");
        out.push_str(&xml_escape(&entry.content));
        out.push_str("</content>");
        if let Some(meta) = &entry.metadata {
            out.push_str("<metadata>");
            if let Ok(json) = serde_json::to_string(meta) {
                out.push_str(&xml_escape(&json));
            }
            out.push_str("</metadata>");
        }
        out.push_str("</entry>");
    }
    out
}
