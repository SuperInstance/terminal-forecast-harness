//! # Resource Predictor
//!
//! Forecasts resource usage (memory, CPU, duration) for terminal commands
//! using exponential moving averages (EMA) with variance estimates.
//!
//! "When you run `cargo build`, memory usage increases by ~200MB."
//! "This command typically needs 2GB RAM."
//!
//! Predicts resource needs *before* execution and warns if predicted
//! requirements exceed available capacity.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single resource usage observation for a command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceObservation {
    /// Memory usage in bytes.
    pub memory_bytes: u64,
    /// CPU time in milliseconds.
    pub cpu_time_ms: u64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Aggregated resource statistics for a command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    /// Command name.
    pub command: String,
    /// Number of observations.
    pub observation_count: u64,
    /// Exponential moving average of memory usage (bytes).
    pub memory_ema: f64,
    /// Exponential moving average of CPU time (ms).
    pub cpu_ema: f64,
    /// Exponential moving average of duration (ms).
    pub duration_ema: f64,
    /// Memory variance estimate (using EMA of squared deviations).
    pub memory_var: f64,
    /// Minimum observed memory.
    pub memory_min: u64,
    /// Maximum observed memory.
    pub memory_max: u64,
}

/// A resource prediction for an upcoming command.
#[derive(Debug, Clone)]
pub struct ResourcePrediction {
    /// The command being predicted.
    pub command: String,
    /// Predicted memory usage in bytes (EMA + 1σ).
    pub predicted_memory_bytes: u64,
    /// Predicted CPU time in ms.
    pub predicted_cpu_ms: u64,
    /// Predicted wall-clock duration in ms.
    pub predicted_duration_ms: u64,
    /// Confidence: number of past observations.
    pub observation_count: u64,
    /// Whether the prediction is a warning (exceeds available).
    pub is_warning: bool,
    /// Human-readable warning message, if any.
    pub warning_message: Option<String>,
}

impl std::fmt::Display for ResourcePrediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_gb = self.predicted_memory_bytes as f64 / 1e9;
        write!(f, "`{}` typically needs {:.1} GB RAM", self.command, mem_gb)?;
        if self.observation_count > 0 {
            write!(f, " (based on {} observations)", self.observation_count)?;
        }
        if let Some(ref msg) = self.warning_message {
            write!(f, "\n⚠️ {}", msg)?;
        }
        Ok(())
    }
}

/// Current system resource availability.
#[derive(Debug, Clone, Default)]
pub struct ResourceAvailability {
    /// Available memory in bytes (e.g., from `/proc/meminfo` `MemAvailable`).
    pub free_memory_bytes: u64,
    /// Total physical memory in bytes.
    pub total_memory_bytes: u64,
    /// Number of logical CPU cores.
    pub cpu_cores: usize,
}

/// Predicts command resource usage using EMA with variance.
///
/// ## Example
///
/// ```
/// use terminal_forecast_harness::resource_predictor::*;
///
/// let mut rp = ResourcePredictor::new();
/// rp.observe("cargo build", ResourceObservation {
///     memory_bytes: 2_000_000_000,
///     cpu_time_ms: 30000,
///     duration_ms: 45000,
/// });
/// rp.observe("cargo build", ResourceObservation {
///     memory_bytes: 2_100_000_000,
///     cpu_time_ms: 28000,
///     duration_ms: 42000,
/// });
///
/// let pred = rp.predict("cargo build", None).unwrap();
/// assert_eq!(pred.observation_count, 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePredictor {
    /// Per-command resource statistics.
    stats: HashMap<String, ResourceStats>,
    /// EMA smoothing factor (0.0 = no update, 1.0 = latest only).
    alpha: f64,
}

impl ResourcePredictor {
    /// Create a new predictor with default smoothing factor α = 0.3.
    ///
    /// α = 0.3 gives ~70% weight to recent observations over a ~5-sample
    /// window, balancing responsiveness with noise reduction.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            alpha: 0.3,
        }
    }

    /// Create a predictor with a custom smoothing factor in [0, 1].
    ///
    /// Higher α = more weight to recent observations (faster adaptation,
    /// noisier). Lower α = more smoothing (slower to react, more stable).
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is outside the range [0, 1].
    pub fn with_alpha(alpha: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1]"
        );
        Self {
            stats: HashMap::new(),
            alpha,
        }
    }

    /// Number of commands with resource data.
    pub fn num_commands(&self) -> usize {
        self.stats.len()
    }

    /// Get the EMA smoothing factor.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Record a resource observation for a command.
    ///
    /// Updates the EMA for memory, CPU, and duration, plus an EMA-based
    /// variance estimate for memory. Tracks min/max for summary statistics.
    ///
    /// ```
    /// use terminal_forecast_harness::resource_predictor::*;
    ///
    /// let mut rp = ResourcePredictor::new();
    /// rp.observe("ls", ResourceObservation {
    ///     memory_bytes: 10_000_000,
    ///     cpu_time_ms: 50,
    ///     duration_ms: 100,
    /// });
    /// assert_eq!(rp.num_commands(), 1);
    /// ```
    pub fn observe(&mut self, command: &str, obs: ResourceObservation) {
        let alpha = self.alpha;
        let entry = self.stats.entry(command.to_string()).or_insert_with(|| {
            ResourceStats {
                command: command.to_string(),
                observation_count: 0,
                memory_ema: obs.memory_bytes as f64,
                cpu_ema: obs.cpu_time_ms as f64,
                duration_ema: obs.duration_ms as f64,
                memory_var: 0.0,
                memory_min: obs.memory_bytes,
                memory_max: obs.memory_bytes,
            }
        });

        if entry.observation_count == 0 {
            // First observation: initialize EMAs to exact values
            entry.memory_ema = obs.memory_bytes as f64;
            entry.cpu_ema = obs.cpu_time_ms as f64;
            entry.duration_ema = obs.duration_ms as f64;
            entry.memory_var = 0.0;
        } else {
            // Standard EMA update
            entry.memory_ema = alpha * obs.memory_bytes as f64 + (1.0 - alpha) * entry.memory_ema;
            entry.cpu_ema = alpha * obs.cpu_time_ms as f64 + (1.0 - alpha) * entry.cpu_ema;
            entry.duration_ema = alpha * obs.duration_ms as f64 + (1.0 - alpha) * entry.duration_ema;
            // EMA-based variance (centered around current EMA)
            let diff = obs.memory_bytes as f64 - entry.memory_ema;
            entry.memory_var = alpha * diff * diff + (1.0 - alpha) * entry.memory_var;
        }

        entry.observation_count += 1;
        entry.memory_min = entry.memory_min.min(obs.memory_bytes);
        entry.memory_max = entry.memory_max.max(obs.memory_bytes);
    }

    /// Record a batch of observations for a command.
    ///
    /// ```
    /// use terminal_forecast_harness::resource_predictor::*;
    ///
    /// let mut rp = ResourcePredictor::new();
    /// rp.observe_batch("npm test", &[
    ///     ResourceObservation { memory_bytes: 500_000_000, cpu_time_ms: 10000, duration_ms: 15000 },
    ///     ResourceObservation { memory_bytes: 600_000_000, cpu_time_ms: 12000, duration_ms: 18000 },
    /// ]);
    /// assert_eq!(rp.stats_for("npm test").unwrap().observation_count, 2);
    /// ```
    pub fn observe_batch(&mut self, command: &str, observations: &[ResourceObservation]) {
        for obs in observations {
            self.observe(command, obs.clone());
        }
    }

    /// Predict resource usage for a command.
    ///
    /// The predicted memory is EMA + 1 standard deviation (a conservative
    /// upper bound that covers ~84% of expected cases under normality).
    ///
    /// Returns `None` if the command has no observations.
    ///
    /// If `available` is provided, checks whether predicted usage exceeds
    /// available free memory and sets `is_warning` accordingly.
    ///
    /// ```
    /// use terminal_forecast_harness::resource_predictor::*;
    ///
    /// let mut rp = ResourcePredictor::new();
    /// rp.observe("build", ResourceObservation {
    ///     memory_bytes: 2_000_000_000,
    ///     cpu_time_ms: 30000,
    ///     duration_ms: 45000,
    /// });
    ///
    /// let pred = rp.predict("build", None).unwrap();
    /// assert!(!pred.is_warning);
    ///
    /// let avail = ResourceAvailability {
    ///     free_memory_bytes: 500_000_000,
    ///     total_memory_bytes: 4_000_000_000,
    ///     cpu_cores: 4,
    /// };
    /// let pred_warn = rp.predict("build", Some(&avail)).unwrap();
    /// assert!(pred_warn.is_warning);
    /// ```
    pub fn predict(
        &self,
        command: &str,
        available: Option<&ResourceAvailability>,
    ) -> Option<ResourcePrediction> {
        let stats = self.stats.get(command)?;
        let std_dev = stats.memory_var.sqrt();
        let predicted_memory = (stats.memory_ema + std_dev) as u64;

        let (is_warning, warning_message) = match available {
            Some(avail) if predicted_memory > avail.free_memory_bytes => {
                let pred_gb = predicted_memory as f64 / 1e9;
                let free_gb = avail.free_memory_bytes as f64 / 1e9;
                (
                    true,
                    Some(format!(
                        "This build typically uses {:.1} GB. You have {:.1} GB free.",
                        pred_gb, free_gb
                    )),
                )
            }
            _ => (false, None),
        };

        Some(ResourcePrediction {
            command: command.to_string(),
            predicted_memory_bytes: predicted_memory,
            predicted_cpu_ms: stats.cpu_ema as u64,
            predicted_duration_ms: stats.duration_ema as u64,
            observation_count: stats.observation_count,
            is_warning,
            warning_message,
        })
    }

    /// Get the raw statistics for a command.
    pub fn stats_for(&self, command: &str) -> Option<&ResourceStats> {
        self.stats.get(command)
    }

    /// Generate a human-readable description of the memory usage pattern.
    ///
    /// ```
    /// use terminal_forecast_harness::resource_predictor::*;
    ///
    /// let mut rp = ResourcePredictor::new();
    /// rp.observe("cargo build", ResourceObservation {
    ///     memory_bytes: 2_000_000_000,
    ///     cpu_time_ms: 30000,
    ///     duration_ms: 45000,
    /// });
    /// let desc = rp.memory_delta_description("cargo build").unwrap();
    /// assert!(desc.contains("cargo build"));
    /// assert!(desc.contains("MB"));
    /// ```
    pub fn memory_delta_description(&self, command: &str) -> Option<String> {
        let stats = self.stats.get(command)?;
        let delta_mb = stats.memory_ema / 1_000_000.0;
        Some(format!(
            "When you run `{}`, memory usage increases by ~{:.0} MB",
            command, delta_mb
        ))
    }

    /// Compute the coefficient of variation for memory (std / mean).
    /// Higher values indicate more variable resource usage.
    pub fn memory_cv(&self, command: &str) -> Option<f64> {
        let stats = self.stats.get(command)?;
        if stats.memory_ema <= 0.0 || stats.observation_count < 2 {
            return None;
        }
        let std_dev = stats.memory_var.sqrt();
        Some(std_dev / stats.memory_ema)
    }

    /// Serialize to JSON for persistence across sessions.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Get all command names with resource data.
    pub fn commands(&self) -> Vec<String> {
        self.stats.keys().cloned().collect()
    }

    /// Get the total memory variance for all commands with >1 observation.
    pub fn total_variance(&self) -> f64 {
        self.stats
            .values()
            .filter(|s| s.observation_count > 1)
            .map(|s| s.memory_var)
            .sum()
    }
}

impl Default for ResourcePredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_predictor() -> ResourcePredictor {
        let mut p = ResourcePredictor::new();
        p.observe(
            "cargo build",
            ResourceObservation {
                memory_bytes: 2_000_000_000,
                cpu_time_ms: 30000,
                duration_ms: 45000,
            },
        );
        p.observe(
            "cargo build",
            ResourceObservation {
                memory_bytes: 2_100_000_000,
                cpu_time_ms: 28000,
                duration_ms: 42000,
            },
        );
        p.observe(
            "cargo build",
            ResourceObservation {
                memory_bytes: 1_900_000_000,
                cpu_time_ms: 32000,
                duration_ms: 48000,
            },
        );
        p
    }

    // ------------------------------------------------------------------
    // Construction & basic properties
    // ------------------------------------------------------------------

    #[test]
    fn test_new_is_empty() {
        let p = ResourcePredictor::new();
        assert_eq!(p.num_commands(), 0);
    }

    #[test]
    fn test_custom_alpha() {
        let p = ResourcePredictor::with_alpha(0.1);
        assert_eq!(p.alpha(), 0.1);
    }

    #[test]
    fn test_default_alpha() {
        let p = ResourcePredictor::new();
        assert_eq!(p.alpha(), 0.3);
    }

    #[test]
    #[should_panic(expected = "alpha must be in [0, 1]")]
    fn test_invalid_alpha_gt_one() {
        ResourcePredictor::with_alpha(1.5);
    }

    #[test]
    #[should_panic(expected = "alpha must be in [0, 1]")]
    fn test_invalid_alpha_negative() {
        ResourcePredictor::with_alpha(-0.1);
    }

    // ------------------------------------------------------------------
    // Observations
    // ------------------------------------------------------------------

    #[test]
    fn test_observe_creates_entry() {
        let mut p = ResourcePredictor::new();
        p.observe(
            "ls",
            ResourceObservation {
                memory_bytes: 10_000_000,
                cpu_time_ms: 50,
                duration_ms: 100,
            },
        );
        assert_eq!(p.num_commands(), 1);
        assert_eq!(p.stats_for("ls").unwrap().observation_count, 1);
    }

    #[test]
    fn test_observe_multiple_commands() {
        let mut p = ResourcePredictor::new();
        p.observe("a", ResourceObservation { memory_bytes: 100, cpu_time_ms: 10, duration_ms: 20 });
        p.observe("b", ResourceObservation { memory_bytes: 200, cpu_time_ms: 20, duration_ms: 40 });
        assert_eq!(p.num_commands(), 2);
    }

    #[test]
    fn test_ema_updates_with_multiple_observations() {
        let p = make_predictor();
        let stats = p.stats_for("cargo build").unwrap();
        assert_eq!(stats.observation_count, 3);
        assert!(stats.memory_ema >= 1_900_000_000.0);
        assert!(stats.memory_ema <= 2_100_000_000.0);
    }

    #[test]
    fn test_min_max_tracking() {
        let p = make_predictor();
        let stats = p.stats_for("cargo build").unwrap();
        assert_eq!(stats.memory_min, 1_900_000_000);
        assert_eq!(stats.memory_max, 2_100_000_000);
    }

    #[test]
    fn test_observe_batch() {
        let mut p = ResourcePredictor::new();
        p.observe_batch(
            "npm test",
            &[
                ResourceObservation {
                    memory_bytes: 500_000_000,
                    cpu_time_ms: 10000,
                    duration_ms: 15000,
                },
                ResourceObservation {
                    memory_bytes: 600_000_000,
                    cpu_time_ms: 12000,
                    duration_ms: 18000,
                },
            ],
        );
        assert_eq!(p.stats_for("npm test").unwrap().observation_count, 2);
    }

    // ------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------

    #[test]
    fn test_predict_returns_ema_plus_std() {
        let p = make_predictor();
        let pred = p.predict("cargo build", None).unwrap();
        assert_eq!(pred.observation_count, 3);
        assert!(pred.predicted_memory_bytes >= 1_900_000_000);
    }

    #[test]
    fn test_predict_unknown_returns_none() {
        let p = make_predictor();
        assert!(p.predict("nonexistent", None).is_none());
    }

    #[test]
    fn test_predict_warning_when_exceeds_available() {
        let p = make_predictor();
        let avail = ResourceAvailability {
            free_memory_bytes: 500_000_000,
            total_memory_bytes: 4_000_000_000,
            cpu_cores: 4,
        };
        let pred = p.predict("cargo build", Some(&avail)).unwrap();
        assert!(pred.is_warning);
        assert!(pred.warning_message.is_some());
    }

    #[test]
    fn test_predict_no_warning_when_enough_memory() {
        let p = make_predictor();
        let avail = ResourceAvailability {
            free_memory_bytes: 10_000_000_000,
            total_memory_bytes: 16_000_000_000,
            cpu_cores: 8,
        };
        let pred = p.predict("cargo build", Some(&avail)).unwrap();
        assert!(!pred.is_warning);
    }

    #[test]
    fn test_predict_warning_message_contains_numbers() {
        let p = make_predictor();
        let avail = ResourceAvailability {
            free_memory_bytes: 100_000_000,
            total_memory_bytes: 4_000_000_000,
            cpu_cores: 4,
        };
        let pred = p.predict("cargo build", Some(&avail)).unwrap();
        let msg = pred.warning_message.as_ref().unwrap();
        assert!(msg.contains("GB"));
    }

    // ------------------------------------------------------------------
    // Display and description
    // ------------------------------------------------------------------

    #[test]
    fn test_prediction_display_format() {
        let p = make_predictor();
        let pred = p.predict("cargo build", None).unwrap();
        let formatted = format!("{}", pred);
        assert!(formatted.contains("cargo build"));
        assert!(formatted.contains("GB"));
    }

    #[test]
    fn test_prediction_display_with_warning() {
        let p = make_predictor();
        let avail = ResourceAvailability {
            free_memory_bytes: 500_000_000,
            total_memory_bytes: 4_000_000_000,
            cpu_cores: 4,
        };
        let pred = p.predict("cargo build", Some(&avail)).unwrap();
        let formatted = format!("{}", pred);
        assert!(formatted.contains("⚠️"));
    }

    #[test]
    fn test_memory_delta_description() {
        let p = make_predictor();
        let desc = p.memory_delta_description("cargo build").unwrap();
        assert!(desc.contains("cargo build"));
        assert!(desc.contains("MB"));
    }

    #[test]
    fn test_memory_delta_unknown_command() {
        let p = make_predictor();
        assert!(p.memory_delta_description("nonexistent").is_none());
    }

    // ------------------------------------------------------------------
    // Statistics & variance
    // ------------------------------------------------------------------

    #[test]
    fn test_variance_increases_for_variable_usage() {
        let mut p = ResourcePredictor::new();
        for _ in 0..10 {
            p.observe(
                "stable",
                ResourceObservation {
                    memory_bytes: 1_000_000_000,
                    cpu_time_ms: 1000,
                    duration_ms: 2000,
                },
            );
        }
        let stable_var = p.stats_for("stable").unwrap().memory_var;
        let mut vp = ResourcePredictor::new();
        for i in 0..10u64 {
            vp.observe(
                "variable",
                ResourceObservation {
                    memory_bytes: 1_000_000_000 + i * 500_000_000,
                    cpu_time_ms: 1000,
                    duration_ms: 2000,
                },
            );
        }
        let variable_var = vp.stats_for("variable").unwrap().memory_var;
        assert!(
            variable_var > stable_var,
            "variable usage should have higher variance"
        );
    }

    #[test]
    fn test_memory_cv_stable_is_small() {
        let mut p = ResourcePredictor::new();
        for _ in 0..5 {
            p.observe(
                "stable",
                ResourceObservation {
                    memory_bytes: 1_000_000_000,
                    cpu_time_ms: 1000,
                    duration_ms: 2000,
                },
            );
        }
        let cv = p.memory_cv("stable").unwrap();
        assert!(cv < 0.01, "stable command should have near-zero CV");
    }

    #[test]
    fn test_memory_cv_unknown_is_none() {
        let p = ResourcePredictor::new();
        assert!(p.memory_cv("nope").is_none());
    }

    #[test]
    fn test_total_variance_increases_with_commands() {
        let mut p = ResourcePredictor::new();
        assert_eq!(p.total_variance(), 0.0);

        p.observe("a", ResourceObservation { memory_bytes: 100, cpu_time_ms: 10, duration_ms: 20 });
        p.observe("a", ResourceObservation { memory_bytes: 200, cpu_time_ms: 10, duration_ms: 20 });
        let v1 = p.total_variance();
        assert!(v1 > 0.0);

        p.observe("b", ResourceObservation { memory_bytes: 300, cpu_time_ms: 10, duration_ms: 20 });
        p.observe("b", ResourceObservation { memory_bytes: 600, cpu_time_ms: 10, duration_ms: 20 });
        let v2 = p.total_variance();
        assert!(v2 > v1, "adding variable commands should increase total variance");
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    #[test]
    fn test_serialization_roundtrip() {
        let p = make_predictor();
        let json = p.to_json().unwrap();
        let restored = ResourcePredictor::from_json(&json).unwrap();
        assert_eq!(restored.num_commands(), p.num_commands());
        assert_eq!(
            restored.stats_for("cargo build").unwrap().observation_count,
            3
        );
        assert_eq!(restored.alpha(), 0.3);
    }

    #[test]
    fn test_serialization_empty() {
        let p = ResourcePredictor::new();
        let json = p.to_json().unwrap();
        let restored = ResourcePredictor::from_json(&json).unwrap();
        assert_eq!(restored.num_commands(), 0);
    }

    #[test]
    fn test_serialization_restores_predictions() {
        let p = make_predictor();
        let json = p.to_json().unwrap();
        let restored = ResourcePredictor::from_json(&json).unwrap();
        let pred = restored.predict("cargo build", None).unwrap();
        assert_eq!(pred.observation_count, 3);
    }

    // ------------------------------------------------------------------
    // Commands listing
    // ------------------------------------------------------------------

    #[test]
    fn test_commands_list() {
        let p = make_predictor();
        let cmds = p.commands();
        assert_eq!(cmds, vec!["cargo build"]);
    }

    #[test]
    fn test_commands_list_empty() {
        let p = ResourcePredictor::new();
        assert!(p.commands().is_empty());
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_stable_predictor_low_variance() {
        let mut p = ResourcePredictor::new();
        for _ in 0..20 {
            p.observe(
                "ping",
                ResourceObservation {
                    memory_bytes: 5_000_000,
                    cpu_time_ms: 100,
                    duration_ms: 500,
                },
            );
        }
        let pred = p.predict("ping", None).unwrap();
        // With very stable usage, prediction should be close to observed
        assert!(
            pred.predicted_memory_bytes >= 5_000_000
        );
    }

    #[test]
    fn test_stats_for_known_command() {
        let p = make_predictor();
        let stats = p.stats_for("cargo build").unwrap();
        assert_eq!(stats.command, "cargo build");
        assert!(stats.memory_var >= 0.0);
    }

    #[test]
    fn test_stats_for_unknown_command() {
        let p = make_predictor();
        assert!(p.stats_for("nope").is_none());
    }

    #[test]
    fn test_default_impl() {
        let p = ResourcePredictor::default();
        assert_eq!(p.num_commands(), 0);
        assert_eq!(p.alpha(), 0.3);
    }
}
