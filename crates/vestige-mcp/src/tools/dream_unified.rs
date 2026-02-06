//! Unified Dream Tool
//!
//! Exposes dream cycle results (connections, clusters, insights) via MCP.
//! Designed for LLM synthesis: returns raw memory content so the calling
//! Claude instance can generate meaningful insights from clusters.

use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::Storage;

/// Input schema for unified dream tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["explore", "clusters", "insights"],
                "description": "Action: 'explore' (walk connection graph from a memory), 'clusters' (find groups of related memories for synthesis), 'insights' (get stored insights)"
            },
            "memory_id": {
                "type": "string",
                "description": "Memory ID to explore connections from (required for 'explore' action)"
            },
            "depth": {
                "type": "integer",
                "description": "Graph walk depth for 'explore' (default: 2, max: 3)",
                "default": 2,
                "minimum": 1,
                "maximum": 3
            },
            "max_cluster_size": {
                "type": "integer",
                "description": "Maximum memories per cluster to return (default: 20)",
                "default": 20,
                "minimum": 5,
                "maximum": 50
            },
            "max_clusters": {
                "type": "integer",
                "description": "Maximum number of clusters to return (default: 10)",
                "default": 10,
                "minimum": 1,
                "maximum": 25
            },
            "min_strength": {
                "type": "number",
                "description": "Minimum connection strength for clustering (default: 0.95)",
                "default": 0.95,
                "minimum": 0.5,
                "maximum": 1.0
            },
            "limit": {
                "type": "integer",
                "description": "Max insights to return for 'insights' action (default: 20)",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            }
        },
        "required": ["action"]
    })
}

#[derive(Debug, Deserialize)]
struct DreamArgs {
    action: String,
    memory_id: Option<String>,
    depth: Option<usize>,
    max_cluster_size: Option<usize>,
    max_clusters: Option<usize>,
    min_strength: Option<f64>,
    limit: Option<i32>,
}

/// Execute unified dream tool
pub async fn execute(
    storage: &Arc<Mutex<Storage>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: DreamArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    match args.action.as_str() {
        "explore" => execute_explore(storage, &args).await,
        "clusters" => execute_clusters(storage, &args).await,
        "insights" => execute_insights(storage, &args).await,
        other => Err(format!(
            "Unknown action '{}'. Use: explore, clusters, insights",
            other
        )),
    }
}

/// Walk the connection graph from a memory, returning related memories with content
async fn execute_explore(
    storage: &Arc<Mutex<Storage>>,
    args: &DreamArgs,
) -> Result<Value, String> {
    let memory_id = args
        .memory_id
        .as_ref()
        .ok_or("memory_id is required for 'explore' action")?;
    let max_depth = args.depth.unwrap_or(2).clamp(1, 3);

    let storage = storage.lock().await;

    // BFS walk from the given memory
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut results: Vec<Value> = Vec::new();
    let mut edges: Vec<Value> = Vec::new();

    visited.insert(memory_id.clone());
    queue.push_back((memory_id.clone(), 0));

    while let Some((current_id, depth)) = queue.pop_front() {
        // Get the memory content
        if let Ok(Some(node)) = storage.get_node(&current_id) {
            results.push(serde_json::json!({
                "id": node.id,
                "content": node.content,
                "nodeType": node.node_type,
                "tags": node.tags,
                "depth": depth,
                "retentionStrength": node.retention_strength,
            }));
        }

        // Don't explore beyond max depth
        if depth >= max_depth {
            continue;
        }

        // Get connections
        if let Ok(connections) = storage.get_connections_for_memory(&current_id) {
            for conn in connections {
                let neighbor_id = if conn.source_id == current_id {
                    &conn.target_id
                } else {
                    &conn.source_id
                };

                edges.push(serde_json::json!({
                    "from": conn.source_id,
                    "to": conn.target_id,
                    "strength": conn.strength,
                    "type": conn.link_type,
                }));

                if !visited.contains(neighbor_id) {
                    visited.insert(neighbor_id.clone());
                    queue.push_back((neighbor_id.clone(), depth + 1));
                }
            }
        }
    }

    Ok(serde_json::json!({
        "action": "explore",
        "root": memory_id,
        "depth": max_depth,
        "totalMemories": results.len(),
        "totalEdges": edges.len(),
        "memories": results,
        "edges": edges,
    }))
}

/// Build clusters from persisted connections using union-find,
/// then return member memory content for LLM synthesis
async fn execute_clusters(
    storage: &Arc<Mutex<Storage>>,
    args: &DreamArgs,
) -> Result<Value, String> {
    let max_cluster_size = args.max_cluster_size.unwrap_or(20).clamp(5, 50);
    let max_clusters = args.max_clusters.unwrap_or(10).clamp(1, 25);
    let min_strength = args.min_strength.unwrap_or(0.95).clamp(0.5, 1.0);

    let storage = storage.lock().await;

    // Load all connections above the strength threshold
    let all_connections = storage.get_all_connections().map_err(|e| e.to_string())?;
    let strong_connections: Vec<_> = all_connections
        .iter()
        .filter(|c| c.strength >= min_strength)
        .collect();

    if strong_connections.is_empty() {
        return Ok(serde_json::json!({
            "action": "clusters",
            "minStrength": min_strength,
            "totalConnections": 0,
            "totalClusters": 0,
            "clusters": [],
            "hint": "No connections above the strength threshold. Try lowering min_strength or run 'vestige consolidate --deep' first."
        }));
    }

    // Union-Find to build clusters
    let mut parent: HashMap<String, String> = HashMap::new();

    fn find(parent: &mut HashMap<String, String>, x: &str) -> String {
        if !parent.contains_key(x) {
            parent.insert(x.to_string(), x.to_string());
            return x.to_string();
        }
        let p = parent.get(x).unwrap().clone();
        if p == x {
            return x.to_string();
        }
        let root = find(parent, &p);
        parent.insert(x.to_string(), root.clone());
        root
    }

    fn union(parent: &mut HashMap<String, String>, a: &str, b: &str) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent.insert(ra, rb.to_string());
        }
    }

    for conn in &strong_connections {
        union(&mut parent, &conn.source_id, &conn.target_id);
    }

    // Group by root
    let all_ids: Vec<String> = parent.keys().cloned().collect();
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for id in &all_ids {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push(id.clone());
    }

    // Sort clusters by size (largest first), filter singletons
    let mut clusters: Vec<Vec<String>> = groups
        .into_values()
        .filter(|g| g.len() >= 2)
        .collect();
    clusters.sort_by(|a, b| b.len().cmp(&a.len()));
    clusters.truncate(max_clusters);

    // Load memory content for each cluster
    let mut cluster_results: Vec<Value> = Vec::new();
    for (i, member_ids) in clusters.iter().enumerate() {
        let mut members: Vec<Value> = Vec::new();
        let mut tags_freq: HashMap<String, usize> = HashMap::new();

        for id in member_ids.iter().take(max_cluster_size) {
            if let Ok(Some(node)) = storage.get_node(id) {
                for tag in &node.tags {
                    *tags_freq.entry(tag.clone()).or_default() += 1;
                }
                members.push(serde_json::json!({
                    "id": node.id,
                    "content": node.content,
                    "nodeType": node.node_type,
                    "tags": node.tags,
                }));
            }
        }

        // Top tags for this cluster (hints for Claude about what the cluster is about)
        let mut top_tags: Vec<(String, usize)> = tags_freq.into_iter().collect();
        top_tags.sort_by(|a, b| b.1.cmp(&a.1));
        let top_tags: Vec<String> = top_tags.into_iter().take(5).map(|(t, _)| t).collect();

        cluster_results.push(serde_json::json!({
            "clusterIndex": i,
            "totalMembers": member_ids.len(),
            "returnedMembers": members.len(),
            "topTags": top_tags,
            "members": members,
        }));
    }

    Ok(serde_json::json!({
        "action": "clusters",
        "minStrength": min_strength,
        "totalConnections": strong_connections.len(),
        "totalClusters": cluster_results.len(),
        "clusters": cluster_results,
        "synthesisPrompt": "Review each cluster's member content and synthesize a concise insight about what pattern, theme, or relationship connects these memories. Focus on actionable understanding, not just describing the connection."
    }))
}

/// Return stored insights from the dream cycle
async fn execute_insights(
    storage: &Arc<Mutex<Storage>>,
    args: &DreamArgs,
) -> Result<Value, String> {
    let limit = args.limit.unwrap_or(20).clamp(1, 100);

    let storage = storage.lock().await;
    let insights = storage.get_insights(limit).map_err(|e| e.to_string())?;

    let formatted: Vec<Value> = insights
        .iter()
        .map(|i| {
            serde_json::json!({
                "id": i.id,
                "insight": i.insight,
                "sourceMemories": i.source_memories,
                "confidence": i.confidence,
                "noveltyScore": i.novelty_score,
                "insightType": i.insight_type,
                "generatedAt": i.generated_at.to_rfc3339(),
                "tags": i.tags,
                "feedback": i.feedback,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "action": "insights",
        "total": formatted.len(),
        "insights": formatted,
    }))
}
