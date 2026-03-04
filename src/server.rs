use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ErrorData as McpError, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::config::Config;
use crate::filters::{FieldType, FilterCondition};
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
    /// Dynamic filter fields from `filterable_fields` config
    #[serde(flatten)]
    #[schemars(skip)]
    pub extra_fields: HashMap<String, Value>,
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

        // Hide collection_name from both tools when a default is configured
        if config.qdrant.collection_name.is_some() {
            for tool_name in ["qdrant-store", "qdrant-find"] {
                if let Some(route) = tool_router.map.get_mut(tool_name) {
                    let mut schema = (*route.attr.input_schema).clone();
                    remove_schema_property(&mut schema, "collection_name");
                    route.attr.input_schema = Arc::new(schema);
                }
            }
        }

        // Manipulate qdrant-find schema based on filter configuration
        if let Some(route) = tool_router.map.get_mut("qdrant-find") {
            let mut schema = (*route.attr.input_schema).clone();

            if !config.qdrant.filterable_fields.is_empty() {
                // Mode 1: Replace query_filter with typed per-field params
                remove_schema_property(&mut schema, "query_filter");

                for field in &config.qdrant.filterable_fields {
                    let json_type = match (&field.field_type, &field.condition) {
                        (_, Some(FilterCondition::Any | FilterCondition::Except)) => {
                            let item_type = match field.field_type {
                                FieldType::Integer => "integer",
                                _ => "string",
                            };
                            serde_json::json!({
                                "type": "array",
                                "items": {"type": item_type},
                                "description": field.description
                            })
                        }
                        (FieldType::Keyword, _) => serde_json::json!({
                            "type": "string",
                            "description": field.description
                        }),
                        (FieldType::Integer, _) => serde_json::json!({
                            "type": "integer",
                            "description": field.description
                        }),
                        (FieldType::Float, _) => serde_json::json!({
                            "type": "number",
                            "description": field.description
                        }),
                        (FieldType::Boolean, _) => serde_json::json!({
                            "type": "boolean",
                            "description": field.description
                        }),
                    };

                    add_schema_property(&mut schema, &field.name, json_type, field.required);
                }
            } else if !config.qdrant.allow_arbitrary_filter {
                // Mode 3: No filterable fields + arbitrary filter disabled → hide query_filter
                remove_schema_property(&mut schema, "query_filter");
            }
            // Mode 2: No filterable fields + arbitrary filter enabled → keep query_filter as-is

            route.attr.input_schema = Arc::new(schema);
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

        // Build filter from either typed fields or arbitrary query_filter
        let filter = if !self.config.qdrant.filterable_fields.is_empty()
            && !params.extra_fields.is_empty()
        {
            let f = crate::filters::make_filter(
                &self.config.qdrant.filterable_fields,
                &params.extra_fields,
            )
            .map_err(|e| McpError::invalid_params(format!("invalid filter: {e}"), None))?;
            Some(f)
        } else if let Some(raw_filter) = params.query_filter {
            if !self.config.qdrant.allow_arbitrary_filter {
                return Err(McpError::invalid_params(
                    "arbitrary filters are disabled — set QDRANT_ALLOW_ARBITRARY_FILTER=true to enable",
                    None,
                ));
            }
            Some(
                crate::qdrant::json_to_qdrant_filter(&raw_filter)
                    .map_err(|e| McpError::invalid_params(format!("invalid filter: {e}"), None))?,
            )
        } else {
            None
        };

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

fn remove_schema_property(schema: &mut serde_json::Map<String, Value>, property: &str) {
    if let Some(Value::Object(props)) = schema.get_mut("properties") {
        props.remove(property);
    }
    if let Some(Value::Array(required)) = schema.get_mut("required") {
        required.retain(|v| v.as_str() != Some(property));
    }
}

fn add_schema_property(
    schema: &mut serde_json::Map<String, Value>,
    name: &str,
    property_schema: Value,
    required: bool,
) {
    if let Some(Value::Object(props)) = schema.get_mut("properties") {
        props.insert(name.to_string(), property_schema);
    }
    if required && let Some(Value::Array(req)) = schema.get_mut("required") {
        req.push(Value::String(name.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_schema_property_removes_from_properties_and_required() {
        let schema_json = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "collection_name": {"type": "string"}
            },
            "required": ["query", "collection_name"]
        });
        let mut schema: serde_json::Map<String, Value> =
            serde_json::from_value(schema_json).unwrap();

        remove_schema_property(&mut schema, "collection_name");

        let props = schema["properties"].as_object().unwrap();
        assert!(!props.contains_key("collection_name"));
        assert!(props.contains_key("query"));

        let required = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(!required.contains(&"collection_name"));
        assert!(required.contains(&"query"));
    }

    #[test]
    fn remove_schema_property_handles_optional_field() {
        let schema_json = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "collection_name": {"type": "string"}
            },
            "required": ["query"]
        });
        let mut schema: serde_json::Map<String, Value> =
            serde_json::from_value(schema_json).unwrap();

        remove_schema_property(&mut schema, "collection_name");

        let props = schema["properties"].as_object().unwrap();
        assert!(!props.contains_key("collection_name"));
    }

    #[test]
    fn add_schema_property_adds_to_properties() {
        let schema_json = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let mut schema: serde_json::Map<String, Value> =
            serde_json::from_value(schema_json).unwrap();

        add_schema_property(
            &mut schema,
            "city",
            serde_json::json!({"type": "string", "description": "City name"}),
            false,
        );

        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("city"));
        assert_eq!(props["city"]["type"], "string");

        assert!(
            !schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str() == Some("city"))
        );
    }

    #[test]
    fn add_schema_property_required() {
        let schema_json = serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let mut schema: serde_json::Map<String, Value> =
            serde_json::from_value(schema_json).unwrap();

        add_schema_property(
            &mut schema,
            "category",
            serde_json::json!({"type": "string", "description": "Category"}),
            true,
        );

        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_str() == Some("category"))
        );
    }
}
