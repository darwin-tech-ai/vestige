//! Vestige CLI
//!
//! Command-line interface for managing cognitive memory system.

use std::path::PathBuf;

use chrono::Utc;
use clap::{Parser, Subcommand};
use colored::Colorize;
use vestige_core::{
    ConnectionRecord, DreamConfig, DreamMemory, IngestInput, InsightRecord, MemoryDreamer, Storage,
};

/// Vestige - Cognitive Memory System CLI
#[derive(Parser)]
#[command(name = "vestige")]
#[command(author = "samvallad33")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "CLI for the Vestige cognitive memory system")]
#[command(
    long_about = "Vestige is a cognitive memory system based on 130 years of memory research.\n\nIt implements FSRS-6, spreading activation, synaptic tagging, and more."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show memory statistics
    Stats {
        /// Show tagging/retention distribution
        #[arg(long)]
        tagging: bool,

        /// Show cognitive state distribution
        #[arg(long)]
        states: bool,
    },

    /// Run health check with warnings and recommendations
    Health,

    /// Run memory consolidation cycle
    Consolidate {
        /// Run deep consolidation: dream cycle with connection discovery and insight generation
        #[arg(long)]
        deep: bool,

        /// Minimum similarity threshold for connection discovery (0.0-1.0, default: 0.85)
        #[arg(long, default_value = "0.85")]
        min_similarity: f64,
    },

    /// Restore memories from backup file
    Restore {
        /// Path to backup JSON file
        file: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Stats { tagging, states } => run_stats(tagging, states),
        Commands::Health => run_health(),
        Commands::Consolidate {
            deep,
            min_similarity,
        } => run_consolidate(deep, min_similarity),
        Commands::Restore { file } => run_restore(file),
    }
}

/// Run stats command
fn run_stats(show_tagging: bool, show_states: bool) -> anyhow::Result<()> {
    let storage = Storage::new(None)?;
    let stats = storage.get_stats()?;

    println!("{}", "=== Vestige Memory Statistics ===".cyan().bold());
    println!();

    // Basic stats
    println!("{}: {}", "Total Memories".white().bold(), stats.total_nodes);
    println!(
        "{}: {}",
        "Due for Review".white().bold(),
        stats.nodes_due_for_review
    );
    println!(
        "{}: {:.1}%",
        "Average Retention".white().bold(),
        stats.average_retention * 100.0
    );
    println!(
        "{}: {:.2}",
        "Average Storage Strength".white().bold(),
        stats.average_storage_strength
    );
    println!(
        "{}: {:.2}",
        "Average Retrieval Strength".white().bold(),
        stats.average_retrieval_strength
    );
    println!(
        "{}: {}",
        "With Embeddings".white().bold(),
        stats.nodes_with_embeddings
    );

    if let Some(model) = &stats.embedding_model {
        println!("{}: {}", "Embedding Model".white().bold(), model);
    }

    if let Some(oldest) = stats.oldest_memory {
        println!(
            "{}: {}",
            "Oldest Memory".white().bold(),
            oldest.format("%Y-%m-%d %H:%M:%S")
        );
    }
    if let Some(newest) = stats.newest_memory {
        println!(
            "{}: {}",
            "Newest Memory".white().bold(),
            newest.format("%Y-%m-%d %H:%M:%S")
        );
    }

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{}: {:.1}%",
        "Embedding Coverage".white().bold(),
        embedding_coverage
    );

    // Tagging distribution (retention levels)
    if show_tagging {
        println!();
        println!("{}", "=== Retention Distribution ===".yellow().bold());

        let memories = storage.get_all_nodes(500, 0)?;
        let total = memories.len();

        if total > 0 {
            let high = memories
                .iter()
                .filter(|m| m.retention_strength >= 0.7)
                .count();
            let medium = memories
                .iter()
                .filter(|m| m.retention_strength >= 0.4 && m.retention_strength < 0.7)
                .count();
            let low = memories
                .iter()
                .filter(|m| m.retention_strength < 0.4)
                .count();

            print_distribution_bar("High (>=70%)", high, total, "green");
            print_distribution_bar("Medium (40-70%)", medium, total, "yellow");
            print_distribution_bar("Low (<40%)", low, total, "red");
        } else {
            println!("{}", "No memories found.".dimmed());
        }
    }

    // State distribution
    if show_states {
        println!();
        println!(
            "{}",
            "=== Cognitive State Distribution ===".magenta().bold()
        );

        let memories = storage.get_all_nodes(500, 0)?;
        let total = memories.len();

        if total > 0 {
            let (active, dormant, silent, unavailable) = compute_state_distribution(&memories);

            print_distribution_bar("Active", active, total, "green");
            print_distribution_bar("Dormant", dormant, total, "yellow");
            print_distribution_bar("Silent", silent, total, "red");
            print_distribution_bar("Unavailable", unavailable, total, "magenta");

            println!();
            println!("{}", "State Thresholds:".dimmed());
            println!("  {} >= 0.70 accessibility", "Active".green());
            println!("  {} >= 0.40 accessibility", "Dormant".yellow());
            println!("  {} >= 0.10 accessibility", "Silent".red());
            println!("  {} < 0.10 accessibility", "Unavailable".magenta());
        } else {
            println!("{}", "No memories found.".dimmed());
        }
    }

    Ok(())
}

/// Compute cognitive state distribution for memories
fn compute_state_distribution(
    memories: &[vestige_core::KnowledgeNode],
) -> (usize, usize, usize, usize) {
    let mut active = 0;
    let mut dormant = 0;
    let mut silent = 0;
    let mut unavailable = 0;

    for memory in memories {
        // Accessibility = 0.5*retention + 0.3*retrieval + 0.2*storage
        let accessibility = memory.retention_strength * 0.5
            + memory.retrieval_strength * 0.3
            + memory.storage_strength * 0.2;

        if accessibility >= 0.7 {
            active += 1;
        } else if accessibility >= 0.4 {
            dormant += 1;
        } else if accessibility >= 0.1 {
            silent += 1;
        } else {
            unavailable += 1;
        }
    }

    (active, dormant, silent, unavailable)
}

/// Print a distribution bar
fn print_distribution_bar(label: &str, count: usize, total: usize, color: &str) {
    let percentage = if total > 0 {
        (count as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let bar_width: usize = 30;
    let filled = ((percentage / 100.0) * bar_width as f64) as usize;
    let empty = bar_width.saturating_sub(filled);

    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(empty));
    let colored_bar = match color {
        "green" => bar.green(),
        "yellow" => bar.yellow(),
        "red" => bar.red(),
        "magenta" => bar.magenta(),
        _ => bar.white(),
    };

    println!(
        "  {:15} [{:30}] {:>4} ({:>5.1}%)",
        label, colored_bar, count, percentage
    );
}

/// Run health check
fn run_health() -> anyhow::Result<()> {
    let storage = Storage::new(None)?;
    let stats = storage.get_stats()?;

    println!("{}", "=== Vestige Health Check ===".cyan().bold());
    println!();

    // Determine health status
    let (status, status_color) = if stats.total_nodes == 0 {
        ("EMPTY", "white")
    } else if stats.average_retention < 0.3 {
        ("CRITICAL", "red")
    } else if stats.average_retention < 0.5 {
        ("DEGRADED", "yellow")
    } else {
        ("HEALTHY", "green")
    };

    let colored_status = match status_color {
        "green" => status.green().bold(),
        "yellow" => status.yellow().bold(),
        "red" => status.red().bold(),
        _ => status.white().bold(),
    };

    println!("{}: {}", "Status".white().bold(), colored_status);
    println!("{}: {}", "Total Memories".white(), stats.total_nodes);
    println!(
        "{}: {}",
        "Due for Review".white(),
        stats.nodes_due_for_review
    );
    println!(
        "{}: {:.1}%",
        "Average Retention".white(),
        stats.average_retention * 100.0
    );

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{}: {:.1}%",
        "Embedding Coverage".white(),
        embedding_coverage
    );
    println!(
        "{}: {}",
        "Embedding Service".white(),
        if storage.is_embedding_ready() {
            "Ready".green()
        } else {
            "Not Ready".red()
        }
    );

    // Warnings
    let mut warnings = Vec::new();

    if stats.average_retention < 0.5 && stats.total_nodes > 0 {
        warnings
            .push("Low average retention - consider running consolidation or reviewing memories");
    }

    if stats.nodes_due_for_review > 10 {
        warnings.push("Many memories are due for review");
    }

    if stats.total_nodes > 0 && stats.nodes_with_embeddings == 0 {
        warnings.push("No embeddings generated - semantic search unavailable");
    }

    if embedding_coverage < 50.0 && stats.total_nodes > 10 {
        warnings.push("Low embedding coverage - run consolidation to improve semantic search");
    }

    if !warnings.is_empty() {
        println!();
        println!("{}", "Warnings:".yellow().bold());
        for warning in &warnings {
            println!("  {} {}", "!".yellow().bold(), warning.yellow());
        }
    }

    // Recommendations
    let mut recommendations = Vec::new();

    if status == "CRITICAL" {
        recommendations
            .push("CRITICAL: Many memories have very low retention. Review important memories.");
    }

    if stats.nodes_due_for_review > 5 {
        recommendations.push("Review due memories to strengthen retention.");
    }

    if stats.nodes_with_embeddings < stats.total_nodes {
        recommendations
            .push("Run 'vestige consolidate' to generate embeddings for better semantic search.");
    }

    if stats.total_nodes > 100 && stats.average_retention < 0.7 {
        recommendations.push("Consider running periodic consolidation to maintain memory health.");
    }

    if recommendations.is_empty() && status == "HEALTHY" {
        recommendations.push("Memory system is healthy!");
    }

    println!();
    println!("{}", "Recommendations:".cyan().bold());
    for rec in &recommendations {
        let icon = if rec.starts_with("CRITICAL") {
            "!".red().bold()
        } else {
            ">".cyan()
        };
        let text = if rec.starts_with("CRITICAL") {
            rec.red().to_string()
        } else {
            rec.to_string()
        };
        println!("  {} {}", icon, text);
    }

    Ok(())
}

/// Run consolidation cycle
fn run_consolidate(deep: bool, min_similarity: f64) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Consolidation ===".cyan().bold());
    println!();
    println!("Running memory consolidation cycle...");
    println!();

    let mut storage = Storage::new(None)?;
    let result = storage.run_consolidation()?;

    println!(
        "{}: {}",
        "Nodes Processed".white().bold(),
        result.nodes_processed
    );
    println!(
        "{}: {}",
        "Nodes Promoted".white().bold(),
        result.nodes_promoted
    );
    println!("{}: {}", "Nodes Pruned".white().bold(), result.nodes_pruned);
    println!(
        "{}: {}",
        "Decay Applied".white().bold(),
        result.decay_applied
    );
    println!(
        "{}: {}",
        "Embeddings Generated".white().bold(),
        result.embeddings_generated
    );
    println!("{}: {}ms", "Duration".white().bold(), result.duration_ms);

    if deep {
        println!();
        println!(
            "{}",
            "=== Deep Consolidation (Dream Cycle) ===".magenta().bold()
        );
        println!();
        println!("Loading memories for dream cycle...");

        // Load all nodes from storage
        let nodes = storage.get_all_nodes(5000, 0)?;
        println!(
            "  {} memories loaded",
            nodes.len().to_string().white().bold()
        );

        // Convert KnowledgeNode -> DreamMemory with embeddings for semantic similarity
        println!("  Loading embeddings for semantic similarity...");
        let mut embeddings_loaded = 0usize;
        let dream_memories: Vec<DreamMemory> = nodes
            .iter()
            .map(|n| {
                let embedding = storage.get_node_embedding(&n.id).ok().flatten();
                if embedding.is_some() {
                    embeddings_loaded += 1;
                }
                DreamMemory {
                    id: n.id.clone(),
                    content: n.content.clone(),
                    embedding,
                    tags: n.tags.clone(),
                    created_at: n.created_at,
                    access_count: n.reps.max(0) as u32,
                }
            })
            .collect();

        println!(
            "  {} embeddings loaded ({:.0}% coverage)",
            embeddings_loaded.to_string().white().bold(),
            if nodes.is_empty() {
                0.0
            } else {
                embeddings_loaded as f64 / nodes.len() as f64 * 100.0
            }
        );
        println!("  Running dream cycle (semantic pairwise similarity)...");

        // Configure dream cycle with custom similarity threshold
        let config = DreamConfig {
            min_similarity,
            ..DreamConfig::default()
        };
        println!(
            "  Similarity threshold: {}",
            format!("{:.2}", min_similarity).white().bold()
        );

        // MemoryDreamer::dream() is async, so we need a tokio runtime
        let rt = tokio::runtime::Runtime::new()?;
        let dreamer = MemoryDreamer::with_config(config);
        let dream_result = rt.block_on(dreamer.dream(&dream_memories));

        // Get discovered connections from the dreamer
        let discovered_connections = dreamer.get_connections();

        println!();
        println!("{}", "--- Dream Results ---".yellow().bold());
        println!(
            "{}: {}",
            "Memories Analyzed".white().bold(),
            dream_result.stats.memories_analyzed
        );
        println!(
            "{}: {}",
            "Connections Evaluated".white().bold(),
            dream_result.stats.connections_evaluated
        );
        println!(
            "{}: {}",
            "Connections Discovered".white().bold(),
            dream_result.new_connections_found
        );
        println!(
            "{}: {}",
            "Clusters Found".white().bold(),
            dream_result.stats.clusters_found
        );
        println!(
            "{}: {}",
            "Insights Generated".white().bold(),
            dream_result.insights_generated.len()
        );
        println!(
            "{}: {}ms",
            "Dream Duration".white().bold(),
            dream_result.duration_ms
        );

        // Persist discovered connections to the database
        let mut connections_saved = 0;
        for conn in &discovered_connections {
            let now = Utc::now();
            let record = ConnectionRecord {
                source_id: conn.from_id.clone(),
                target_id: conn.to_id.clone(),
                strength: conn.similarity,
                link_type: format!("{:?}", conn.connection_type),
                created_at: now,
                last_activated: now,
                activation_count: 1,
            };
            if storage.save_connection(&record).is_ok() {
                connections_saved += 1;
            }
        }

        // Persist generated insights to the database
        let mut insights_saved = 0;
        for insight in &dream_result.insights_generated {
            let record = InsightRecord {
                id: insight.id.clone(),
                insight: insight.insight.clone(),
                source_memories: insight.source_memories.clone(),
                confidence: insight.confidence,
                novelty_score: insight.novelty_score,
                insight_type: format!("{:?}", insight.insight_type),
                generated_at: insight.generated_at,
                tags: insight.tags.clone(),
                feedback: None,
                applied_count: 0,
            };
            if storage.save_insight(&record).is_ok() {
                insights_saved += 1;
            }
        }

        println!();
        println!("{}", "--- Persisted ---".green().bold());
        println!(
            "{}: {}",
            "Connections Saved".green().bold(),
            connections_saved
        );
        println!("{}: {}", "Insights Saved".green().bold(), insights_saved);

        // Display generated insights
        if !dream_result.insights_generated.is_empty() {
            println!();
            println!("{}", "--- Insights ---".yellow().bold());
            for (i, insight) in dream_result.insights_generated.iter().enumerate() {
                println!(
                    "  {}. {} {}",
                    i + 1,
                    insight.insight,
                    format!(
                        "(confidence: {:.2}, novelty: {:.2})",
                        insight.confidence, insight.novelty_score
                    )
                    .dimmed()
                );
            }
        }
    }

    println!();
    println!(
        "{}",
        format!(
            "Consolidation complete: {} nodes processed, {} embeddings generated in {}ms",
            result.nodes_processed, result.embeddings_generated, result.duration_ms
        )
        .green()
    );

    Ok(())
}

/// Run restore from backup
fn run_restore(backup_path: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Restore ===".cyan().bold());
    println!();
    println!("Loading backup from: {}", backup_path.display());

    // Read and parse backup
    let backup_content = std::fs::read_to_string(&backup_path)?;

    #[derive(serde::Deserialize)]
    struct BackupWrapper {
        #[serde(rename = "type")]
        _type: String,
        text: String,
    }

    #[derive(serde::Deserialize)]
    struct RecallResult {
        results: Vec<MemoryBackup>,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MemoryBackup {
        content: String,
        node_type: Option<String>,
        tags: Option<Vec<String>>,
        source: Option<String>,
    }

    let wrapper: Vec<BackupWrapper> = serde_json::from_str(&backup_content)?;
    let recall_result: RecallResult = serde_json::from_str(&wrapper[0].text)?;
    let memories = recall_result.results;

    println!("Found {} memories to restore", memories.len());
    println!();

    // Initialize storage
    println!("Initializing storage...");
    let mut storage = Storage::new(None)?;

    println!("Generating embeddings and ingesting memories...");
    println!();

    let total = memories.len();
    let mut success_count = 0;

    for (i, memory) in memories.into_iter().enumerate() {
        let input = IngestInput {
            content: memory.content.clone(),
            node_type: memory.node_type.unwrap_or_else(|| "fact".to_string()),
            source: memory.source,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: memory.tags.unwrap_or_default(),
            valid_from: None,
            valid_until: None,
        };

        match storage.ingest(input) {
            Ok(_node) => {
                success_count += 1;
                println!(
                    "[{}/{}] {} {}",
                    i + 1,
                    total,
                    "OK".green(),
                    truncate(&memory.content, 60)
                );
            }
            Err(e) => {
                println!("[{}/{}] {} {}", i + 1, total, "FAIL".red(), e);
            }
        }
    }

    println!();
    println!(
        "Restore complete: {}/{} memories restored",
        success_count.to_string().green().bold(),
        total
    );

    // Show stats
    let stats = storage.get_stats()?;
    println!();
    println!("{}: {}", "Total Nodes".white(), stats.total_nodes);
    println!(
        "{}: {}",
        "With Embeddings".white(),
        stats.nodes_with_embeddings
    );

    Ok(())
}

/// Truncate a string for display (UTF-8 safe)
fn truncate(s: &str, max_chars: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max_chars {
        s
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}
