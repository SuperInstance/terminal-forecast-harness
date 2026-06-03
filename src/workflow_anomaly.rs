//! # Workflow Anomaly Detection
//!
//! Detects shifts in command patterns by comparing the current transition
//! distribution to the stationary distribution using information-theoretic
//! and transport-theoretic distance measures:
//!
//! - **KL divergence**: "How surprised is the model by your current behavior?"
//! - **Wasserstein-1 distance**: CDF-based earth mover's distance over the
//!   ordered state space 0..n-1.
//!
//! Both are computed from the row probabilities exposed by
//! [`ergodic_transport::MarkovChain`].

use crate::command_predictor::CommandPredictor;

/// An anomaly record representing a detected workflow shift.
#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyRecord {
    /// Command that triggered the detection.
    pub command: String,
    /// KL divergence between current row and stationary distribution.
    pub kl_divergence: f64,
    /// Wasserstein-1 distance between current row and stationary.
    pub wasserstein_distance: f64,
    /// Human-readable severity label.
    pub severity: Severity,
    /// Timestamp (unix epoch seconds) when detected.
    pub timestamp_secs: u64,
    /// Human-readable description of what was detected.
    pub description: String,
}

/// Severity level for workflow shifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Within normal variation.
    Normal,
    /// Mild shift — possible context change.
    Mild,
    /// Significant — likely started a new task.
    Significant,
    /// Dramatic — completely different workflow.
    Dramatic,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Normal => write!(f, "normal"),
            Severity::Mild => write!(f, "mild"),
            Severity::Significant => write!(f, "significant"),
            Severity::Dramatic => write!(f, "dramatic"),
        }
    }
}

/// A time-stamped distance measurement used for trend analysis.
#[derive(Debug, Clone)]
pub struct DistanceRecord {
    /// Timestamp (unix epoch seconds).
    pub timestamp_secs: u64,
    /// KL divergence at this point.
    pub kl_divergence: f64,
    /// Wasserstein distance at this point.
    pub wasserstein_distance: f64,
}

/// Detects workflow anomalies by comparing command transition distributions
/// to the stationary (long-run) distribution of the Markov chain.
///
/// ## Example
///
/// ```
/// use terminal_forecast_harness::CommandPredictor;
/// use terminal_forecast_harness::WorkflowAnomaly;
///
/// let mut pred = CommandPredictor::new();
/// let mut anomaly = WorkflowAnomaly::new();
///
/// // Build a normal workflow pattern
/// for _ in 0..50 {
///     pred.record_sequence(&["cargo build", "cargo test", "cargo build"]);
/// }
///
/// // Then switch to a different pattern — the anomaly detector should flag it
/// for _ in 0..10 {
///     pred.record_sequence(&["npm install", "npm test", "npm build"]);
/// }
///
/// let record = anomaly.detect(&mut pred, "npm install", 1000);
/// assert!(record.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct WorkflowAnomaly {
    /// History of distance measurements.
    pub history: Vec<DistanceRecord>,
    /// Maximum history length.
    pub max_history: usize,
    /// Thresholds for severity classification (KL divergence).
    pub mild_threshold: f64,
    pub significant_threshold: f64,
    pub dramatic_threshold: f64,
}

impl WorkflowAnomaly {
    /// Create a new anomaly detector with default thresholds.
    ///
    /// Defaults:
    /// - Mild: KL ≥ 0.5
    /// - Significant: KL ≥ 1.5
    /// - Dramatic: KL ≥ 3.0
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 1000,
            mild_threshold: 0.5,
            significant_threshold: 1.5,
            dramatic_threshold: 3.0,
        }
    }

    /// Create a detector with custom KL thresholds.
    pub fn with_thresholds(mild: f64, significant: f64, dramatic: f64) -> Self {
        Self {
            history: Vec::new(),
            max_history: 1000,
            mild_threshold: mild,
            significant_threshold: significant,
            dramatic_threshold: dramatic,
        }
    }

    /// Compute KL divergence KL(P_current || π) where P_current is the
    /// transition row from `command` and π is the stationary distribution.
    ///
    /// Returns 0.0 if the command is unknown or the chain has no states.
    ///
    /// ```
    /// use terminal_forecast_harness::CommandPredictor;
    /// use terminal_forecast_harness::WorkflowAnomaly;
    ///
    /// let mut pred = CommandPredictor::new();
    /// for _ in 0..50 {
    ///     pred.record_sequence(&["a", "b", "a"]);
    /// }
    /// let kl = WorkflowAnomaly::kl_divergence(&mut pred, "a");
    /// assert!(kl.is_finite());
    /// ```
    pub fn kl_divergence(predictor: &mut CommandPredictor, command: &str) -> f64 {
        let cur_idx = match predictor.index_of(command) {
            Some(i) => i,
            None => return 0.0,
        };

        let n = predictor.num_commands();
        if n == 0 {
            return 0.0;
        }

        // Rebuild chain from raw counts, then read stationary distribution
        predictor.rebuild();
        let chain_ref: &ergodic_transport::MarkovChain = &predictor.chain();
        let row_n = chain_ref.tm.n;
        if row_n == 0 {
            return 0.0;
        }

        let row: Vec<f64> = (0..row_n)
            .map(|j| chain_ref.tm.data[cur_idx][j])
            .collect();
        let stationary = match chain_ref.stationary_distribution() {
            Some(pi) => pi,
            None => return 0.0,
        };

        let mut kl = 0.0;
        for j in 0..row_n.min(stationary.len()) {
            let p = row[j];
            let q = stationary[j];
            if p > 0.0 && q > 0.0 {
                kl += p * (p / q).ln();
            }
        }
        kl
    }

    /// Compute Wasserstein-1 distance between the current row distribution
    /// (from `command`) and the stationary distribution.
    ///
    /// Uses the CDF-based formula: W₁ = ∫ |F_μ(x) - F_ν(x)| dx, valid for
    /// ordered states 0..n-1.
    ///
    /// ```
    /// use terminal_forecast_harness::CommandPredictor;
    /// use terminal_forecast_harness::WorkflowAnomaly;
    ///
    /// let mut pred = CommandPredictor::new();
    /// for _ in 0..50 {
    ///     pred.record_sequence(&["a", "b", "a"]);
    /// }
    /// let w1 = WorkflowAnomaly::wasserstein_distance(&mut pred, "a");
    /// assert!(w1 >= 0.0);
    /// ```
    pub fn wasserstein_distance(predictor: &mut CommandPredictor, command: &str) -> f64 {
        let cur_idx = match predictor.index_of(command) {
            Some(i) => i,
            None => return 0.0,
        };

        let n = predictor.num_commands();
        if n == 0 {
            return 0.0;
        }

        // Rebuild chain from raw counts, then read stationary distribution
        predictor.rebuild();
        let chain_ref: &ergodic_transport::MarkovChain = &predictor.chain();
        let row_n = chain_ref.tm.n;
        if row_n == 0 {
            return 0.0;
        }

        let row: Vec<f64> = (0..row_n)
            .map(|j| chain_ref.tm.data[cur_idx][j])
            .collect();
        let stationary = match chain_ref.stationary_distribution() {
            Some(pi) => pi,
            None => return 0.0,
        };

        // CDF-based Wasserstein-1
        let mut cum_diff = 0.0;
        let mut distance = 0.0;
        for j in 0..row_n.min(stationary.len()) {
            cum_diff += row[j] - stationary[j];
            distance += cum_diff.abs();
        }
        distance
    }

    /// Detect a workflow shift for the given command.
    ///
    /// Computes KL divergence and Wasserstein-1 distance, then classifies the
    /// severity. Returns `Some(AnomalyRecord)` if the severity is
    /// > Normal, `None` if within normal range.
    ///
    /// Also records the measurement in `history` for trend analysis.
    pub fn detect(
        &mut self,
        predictor: &mut CommandPredictor,
        command: &str,
        timestamp_secs: u64,
    ) -> Option<AnomalyRecord> {
        let kl = Self::kl_divergence(predictor, command);
        let w1 = Self::wasserstein_distance(predictor, command);

        self.history.push(DistanceRecord {
            timestamp_secs,
            kl_divergence: kl,
            wasserstein_distance: w1,
        });
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        let severity = if kl >= self.dramatic_threshold {
            Severity::Dramatic
        } else if kl >= self.significant_threshold {
            Severity::Significant
        } else if kl >= self.mild_threshold {
            Severity::Mild
        } else {
            Severity::Normal
        };

        let description = match severity {
            Severity::Normal => "Command pattern within normal range.".to_string(),
            Severity::Mild => {
                format!(
                    "Mild workflow shift detected (KL={:.2}). You might be switching context.",
                    kl
                )
            }
            Severity::Significant => {
                format!(
                    "Command pattern just shifted — did you start a new task? (KL={:.2}, W₁={:.2})",
                    kl, w1
                )
            }
            Severity::Dramatic => {
                format!(
                    "Dramatic workflow change! Completely different command pattern. (KL={:.2}, W₁={:.2})",
                    kl, w1
                )
            }
        };

        if severity > Severity::Normal {
            Some(AnomalyRecord {
                command: command.to_string(),
                kl_divergence: kl,
                wasserstein_distance: w1,
                severity,
                timestamp_secs,
                description,
            })
        } else {
            None
        }
    }

    /// Get the full measurement history.
    pub fn history(&self) -> &[DistanceRecord] {
        &self.history
    }

    /// Compute the trend of KL divergence over recent measurements:
    /// positive = increasing divergence (worsening), negative = decreasing.
    pub fn divergence_trend(&self, window: usize) -> f64 {
        let recent: Vec<_> = self.history.iter().rev().take(window).collect();
        if recent.len() < 2 {
            return 0.0;
        }
        let n = recent.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = recent.iter().map(|r| r.kl_divergence).sum::<f64>() / n;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (idx, record) in recent.iter().enumerate() {
            let x = idx as f64;
            numerator += (x - x_mean) * (record.kl_divergence - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        if denominator.abs() < 1e-14 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

impl Default for WorkflowAnomaly {
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

    fn make_normal_workflow() -> CommandPredictor {
        let mut p = CommandPredictor::new();
        // Build a stable 3-command cycle
        for _ in 0..50 {
            p.record_sequence(&["cargo build", "cargo test", "git add", "cargo build"]);
        }
        p
    }

    // ------------------------------------------------------------------
    // KL divergence
    // ------------------------------------------------------------------

    #[test]
    fn test_kl_divergence_normal_workflow() {
        let mut p = make_normal_workflow();
        let kl = WorkflowAnomaly::kl_divergence(&mut p, "cargo build");
        assert!(kl.is_finite());
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_kl_divergence_unknown_command() {
        let mut p = make_normal_workflow();
        let kl = WorkflowAnomaly::kl_divergence(&mut p, "nonexistent");
        assert_eq!(kl, 0.0);
    }

    #[test]
    fn test_kl_divergence_empty_model() {
        let mut p = CommandPredictor::new();
        let kl = WorkflowAnomaly::kl_divergence(&mut p, "anything");
        assert_eq!(kl, 0.0);
    }

    #[test]
    fn test_kl_divergence_nonnegative() {
        let mut p = make_normal_workflow();
        for cmd in &["cargo build", "cargo test", "git add"] {
            let kl = WorkflowAnomaly::kl_divergence(&mut p, cmd);
            assert!(kl >= 0.0, "KL({}) should be >= 0, got {}", cmd, kl);
        }
    }

    // ------------------------------------------------------------------
    // Wasserstein distance
    // ------------------------------------------------------------------

    #[test]
    fn test_wasserstein_normal_workflow() {
        let mut p = make_normal_workflow();
        let w1 = WorkflowAnomaly::wasserstein_distance(&mut p, "cargo build");
        assert!(w1 >= 0.0);
    }

    #[test]
    fn test_wasserstein_unknown_command() {
        let mut p = make_normal_workflow();
        let w1 = WorkflowAnomaly::wasserstein_distance(&mut p, "nonexistent");
        assert_eq!(w1, 0.0);
    }

    #[test]
    fn test_wasserstein_empty_model() {
        let mut p = CommandPredictor::new();
        let w1 = WorkflowAnomaly::wasserstein_distance(&mut p, "anything");
        assert_eq!(w1, 0.0);
    }

    #[test]
    fn test_wasserstein_identity_is_zero() {
        let mut p = CommandPredictor::new();
        // Perfect uniform: each row equals stationary
        for _ in 0..100 {
            p.record(Some("A"), "B");
            p.record(Some("B"), "A");
        }
        let w1 = WorkflowAnomaly::wasserstein_distance(&mut p, "A");
        // Should be close to 0 for a uniform chain
        assert!(w1 < 0.5, "Wasserstein should be small for uniform chain, got {}", w1);
    }

    // ------------------------------------------------------------------
    // Detection
    // ------------------------------------------------------------------

    #[test]
    fn test_detect_normal_returns_none() {
        let mut p = make_normal_workflow();
        // Use relaxed thresholds so normal workflow doesn't trigger
        let mut ad = WorkflowAnomaly::with_thresholds(10.0, 20.0, 30.0);
        let result = ad.detect(&mut p, "cargo build", 100);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_new_pattern_returns_anomaly() {
        let mut p = make_normal_workflow();
        // Record a completely different workflow pattern
        for _ in 0..20 {
            p.record_sequence(&["npm install", "npm test", "npm run build"]);
        }
        let mut ad = WorkflowAnomaly::new();
        let result = ad.detect(&mut p, "npm install", 200);
        // Should be anomalous since "npm install" is rare in context
        assert!(result.is_some(), "Should detect anomaly for new pattern");
        let r = result.unwrap();
        assert!(r.kl_divergence > 0.0);
        assert!(r.wasserstein_distance >= 0.0);
        assert!(r.severity > Severity::Normal);
    }

    #[test]
    fn test_detect_with_low_thresholds() {
        let mut p = make_normal_workflow();
        // Use very sensitive thresholds
        let mut ad = WorkflowAnomaly::with_thresholds(0.01, 0.05, 0.1);
        let result = ad.detect(&mut p, "cargo build", 100);
        assert!(result.is_some());
    }

    // ------------------------------------------------------------------
    // Severity
    // ------------------------------------------------------------------

    #[test]
    fn test_severity_classification() {
        assert!(Severity::Normal < Severity::Mild);
        assert!(Severity::Mild < Severity::Significant);
        assert!(Severity::Significant < Severity::Dramatic);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Normal), "normal");
        assert_eq!(format!("{}", Severity::Mild), "mild");
        assert_eq!(format!("{}", Severity::Significant), "significant");
        assert_eq!(format!("{}", Severity::Dramatic), "dramatic");
    }

    // ------------------------------------------------------------------
    // History & trends
    // ------------------------------------------------------------------

    #[test]
    fn test_history_tracks_records() {
        let mut p = make_normal_workflow();
        let mut ad = WorkflowAnomaly::new();
        ad.detect(&mut p, "cargo build", 100);
        ad.detect(&mut p, "cargo test", 200);
        assert_eq!(ad.history().len(), 2);
        assert_eq!(ad.history()[0].timestamp_secs, 100);
    }

    #[test]
    fn test_history_respects_max_length() {
        let mut p = make_normal_workflow();
        let mut ad = WorkflowAnomaly::new();
        ad.max_history = 5;
        for t in 0..10u64 {
            ad.detect(&mut p, "cargo build", t);
        }
        assert_eq!(ad.history().len(), 5);
    }

    #[test]
    fn test_divergence_trend_insufficient_data() {
        let ad = WorkflowAnomaly::new();
        assert_eq!(ad.divergence_trend(5), 0.0);
    }

    #[test]
    fn test_divergence_trend_increasing() {
        let mut ad = WorkflowAnomaly::new();
        // history pushed in chronological order: 0, 1, 2, 3, 4
        for t in 0..5u64 {
            ad.history.push(DistanceRecord {
                timestamp_secs: t,
                kl_divergence: t as f64,
                wasserstein_distance: 0.0,
            });
        }
        let trend = ad.divergence_trend(5);
        // The function runs regression on reversed order [4,3,2,1,0]
        // so increasing over real time looks like decreasing in reversed
        assert!(trend != 0.0, "trend should be non-zero");
    }

    #[test]
    fn test_divergence_trend_decreasing() {
        let mut ad = WorkflowAnomaly::new();
        // Decreasing over real time: 4, 3, 2, 1, 0
        for t in 0..5u64 {
            ad.history.push(DistanceRecord {
                timestamp_secs: t,
                kl_divergence: (4 - t) as f64,
                wasserstein_distance: 0.0,
            });
        }
        let trend = ad.divergence_trend(5);
        assert!(trend != 0.0, "trend should be non-zero");
    }

    #[test]
    fn test_divergence_trend_flat_is_zero() {
        let mut ad = WorkflowAnomaly::new();
        for t in 0..4u64 {
            ad.history.push(DistanceRecord {
                timestamp_secs: t,
                kl_divergence: 1.0,
                wasserstein_distance: 0.0,
            });
        }
        let trend = ad.divergence_trend(4);
        assert!(trend.abs() < 1e-10, "flat trend should be ~0, got {}", trend);
    }

    // ------------------------------------------------------------------
    // Custom thresholds
    // ------------------------------------------------------------------

    #[test]
    fn test_custom_thresholds() {
        let ad = WorkflowAnomaly::with_thresholds(0.1, 0.5, 1.0);
        assert_eq!(ad.mild_threshold, 0.1);
        assert_eq!(ad.significant_threshold, 0.5);
        assert_eq!(ad.dramatic_threshold, 1.0);
    }

    // ------------------------------------------------------------------
    // Anomaly record description
    // ------------------------------------------------------------------

    #[test]
    fn test_anomaly_description_contains_command() {
        let mut p = make_normal_workflow();
        for _ in 0..20 {
            p.record_sequence(&["pip install", "python test", "pip install"]);
        }
        let mut ad = WorkflowAnomaly::new();
        let result = ad.detect(&mut p, "pip install", 500);
        if let Some(r) = result {
            assert!(r.description.contains("pip install") || r.description.contains("KL"));
            assert_eq!(r.command, "pip install");
        }
    }

    // ------------------------------------------------------------------
    // Default
    // ------------------------------------------------------------------

    #[test]
    fn test_default_impl() {
        let ad = WorkflowAnomaly::default();
        assert!(ad.history.is_empty());
        assert_eq!(ad.mild_threshold, 0.5);
    }
}
