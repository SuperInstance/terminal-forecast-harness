//! # Command Predictor
//!
//! Wraps [`ergodic_transport::MarkovChain`] into a terminal-facing predictor
//! that maps command names to chain states and answers:
//!
//! > "Given you just ran `cargo build`, what's the most likely next command?"
//!
//! Uses the row-stochastic transition probabilities directly (not the
//! stationary distribution) for next-step prediction, because the transition
//! matrix tells us what happens *right now* given the current state.

use ergodic_transport::{MarkovChain, MAX_STATES};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single prediction item: a command name and its confidence score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    /// The predicted command name.
    pub command: String,
    /// P(next = command | current) — always positive due to Laplace smoothing.
    pub confidence: f64,
}

impl std::fmt::Display for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({:.1}%)", self.command, self.confidence * 100.0)
    }
}

/// A complete prediction result for a given current command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// The command the user just ran.
    pub current_command: String,
    /// Top-K predictions sorted by confidence descending.
    pub predictions: Vec<Prediction>,
    /// Number of distinct command states known to the model.
    pub num_states: usize,
    /// Total transitions recorded in the model.
    pub total_transitions: u64,
}

impl PredictionResult {
    /// Format as a compact one-line ghost text (tab-completion style).
    ///
    /// ```
    /// let mut cp = terminal_forecast_harness::CommandPredictor::new();
    /// cp.record_sequence(&["ls", "cd src", "cargo build", "cargo test"]);
    /// let r = cp.predict_top_k("cargo build", 2).unwrap();
    /// let ghost = r.ghost_text();
    /// assert!(ghost.starts_with("cargo test"));
    /// ```
    pub fn ghost_text(&self) -> String {
        self.predictions
            .iter()
            .map(|p| p.command.as_str())
            .collect::<Vec<_>>()
            .join("  ")
    }

    /// Format as a multi-line detailed explanation.
    ///
    /// ```
    /// let mut cp = terminal_forecast_harness::CommandPredictor::new();
    /// cp.record_sequence(&["ls", "cd src", "cargo build", "cargo test"]);
    /// let r = cp.predict_top_k("cargo build", 2).unwrap();
    /// println!("{}", r.detailed());
    /// ```
    pub fn detailed(&self) -> String {
        let mut lines = vec![format!("After `{}`:", self.current_command)];
        for p in &self.predictions {
            lines.push(format!("  → {} ({:.1}%)", p.command, p.confidence * 100.0));
        }
        lines.join("\n")
    }
}

/// A ghost-completion helper that returns only the single most likely command.
#[derive(Debug, Clone)]
pub struct GhostCompleter;

impl GhostCompleter {
    /// Return the single most likely next command, or `None` if unknown.
    pub fn complete(predictor: &mut CommandPredictor, current: &str) -> Option<String> {
        predictor
            .predict_top_k(current, 1)
            .and_then(|r| r.predictions.into_iter().next())
            .map(|p| p.command)
    }
}

// ---------------------------------------------------------------------------
// CommandPredictor
// ---------------------------------------------------------------------------

/// Terminal-facing predictor that maps command names onto a Markov chain.
///
/// Maintains a `command ↔ index` mapping and delegates to
/// [`ergodic_transport::MarkovChain`] for chain analysis.
///
/// Internally stores raw transition counts which are converted to a
/// row-stochastic (Laplace-smoothed) matrix on-demand in `build_chain()`.
///
/// ## Example
///
/// ```
/// use terminal_forecast_harness::CommandPredictor;
///
/// let mut pred = CommandPredictor::new();
/// pred.record_sequence(&["ls", "cd src", "cargo build", "cargo test"]);
///
/// let result = pred.predict_top_k("cargo build", 3).unwrap();
/// assert_eq!(result.current_command, "cargo build");
/// assert!(result.predictions[0].command == "cargo test");
/// ```
#[derive(Debug, Clone)]
pub struct CommandPredictor {
    /// Maps command name to dense state index.
    command_to_idx: HashMap<String, usize>,
    /// Reverse lookup: index → command name.
    idx_to_command: Vec<String>,
    /// Number of command states currently allocated.
    num_states: usize,
    /// The underlying Markov chain from ergodic-transport (lazily rebuilt).
    chain: MarkovChain,
    /// Total transitions recorded.
    total_transitions: u64,
    /// Raw transition counts, row-major, dimension MAX_STATES × MAX_STATES.
    raw_counts: Vec<u64>,
}

impl CommandPredictor {
    /// Create a new empty predictor with default max states (MAX_STATES = 64).
    pub fn new() -> Self {
        Self {
            command_to_idx: HashMap::new(),
            idx_to_command: Vec::new(),
            num_states: 0,
            chain: MarkovChain::new(),
            total_transitions: 0,
            raw_counts: vec![0u64; MAX_STATES * MAX_STATES],
        }
    }

    /// Number of distinct command states currently tracked.
    pub fn num_commands(&self) -> usize {
        self.num_states
    }

    /// Total number of transitions recorded.
    pub fn total_transitions(&self) -> u64 {
        self.total_transitions
    }

    /// Get or create an index for a command.  Returns `None` if MAX_STATES
    /// would be exceeded.
    fn ensure_index(&mut self, command: &str) -> Option<usize> {
        if let Some(&idx) = self.command_to_idx.get(command) {
            return Some(idx);
        }
        if self.num_states >= MAX_STATES {
            return None;
        }
        let idx = self.num_states;
        self.command_to_idx.insert(command.to_string(), idx);
        self.idx_to_command.push(command.to_string());
        self.num_states += 1;
        Some(idx)
    }

    /// Record a transition from `prev` command to `next` command.
    ///
    /// If `prev` is `None`, only registers `next` as a known state.
    /// If the state cap is exceeded, the transition is silently dropped.
    pub fn record(&mut self, prev: Option<&str>, next: &str) {
        let next_idx = match self.ensure_index(next) {
            Some(i) => i,
            None => return,
        };
        if let Some(p) = prev {
            let prev_idx = match self.ensure_index(p) {
                Some(i) => i,
                None => return,
            };
            self.raw_counts[prev_idx * MAX_STATES + next_idx] += 1;
            self.total_transitions += 1;
        }
    }

    /// Record a batch of transitions from an ordered command sequence.
    ///
    /// ```
    /// use terminal_forecast_harness::CommandPredictor;
    ///
    /// let mut pred = CommandPredictor::new();
    /// pred.record_sequence(&["git status", "git add", "git commit"]);
    /// assert_eq!(pred.num_commands(), 3);
    /// assert_eq!(pred.total_transitions(), 2);
    /// ```
    pub fn record_sequence(&mut self, commands: &[&str]) {
        if commands.is_empty() {
            return;
        }
        // Ensure all commands exist first
        for &cmd in commands {
            self.ensure_index(cmd);
        }
        // Record transitions
        for window in commands.windows(2) {
            let from_idx = self.command_to_idx[window[0]];
            let to_idx = self.command_to_idx[window[1]];
            self.raw_counts[from_idx * MAX_STATES + to_idx] += 1;
            self.total_transitions += 1;
        }
    }

    /// Predict the top `k` most likely next commands given `current`.
    ///
    /// Returns `None` if `current` is unknown or no transitions exist.
    ///
    /// ```
    /// use terminal_forecast_harness::CommandPredictor;
    ///
    /// let mut pred = CommandPredictor::new();
    /// pred.record_sequence(&["a", "b", "a", "c"]);
    /// let result = pred.predict_top_k("a", 2).unwrap();
    /// assert_eq!(result.predictions.len(), 2);
    /// ```
    pub fn predict_top_k(&mut self, current: &str, k: usize) -> Option<PredictionResult> {
        let cur_idx = *self.command_to_idx.get(current)?;
        self.build_chain();

        let n = self.chain.tm.n;
        if n == 0 {
            return None;
        }

        // Read the transition row for cur_idx
        let mut pairs: Vec<(String, f64)> = (0..n)
            .map(|j| {
                (
                    self.idx_to_command[j].clone(),
                    self.chain.tm.data[cur_idx][j],
                )
            })
            .collect();

        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let predictions: Vec<Prediction> = pairs
            .into_iter()
            .take(k)
            .map(|(cmd, prob)| Prediction {
                command: cmd,
                confidence: prob,
            })
            .collect();

        Some(PredictionResult {
            current_command: current.to_string(),
            predictions,
            num_states: self.num_states,
            total_transitions: self.total_transitions,
        })
    }

    /// Compute the stationary distribution — the long-run fraction of time
    /// spent in each command state.
    ///
    /// Returns `None` if the chain has fewer than 2 states or doesn't converge.
    ///
    /// ```
    /// use terminal_forecast_harness::CommandPredictor;
    ///
    /// let mut pred = CommandPredictor::new();
    /// for _ in 0..50 {
    ///     pred.record_sequence(&["A", "B", "A", "B"]);
    /// }
    /// let pi = pred.stationary_distribution();
    /// assert!(pi.is_some());
    /// let pi = pi.unwrap();
    /// assert_eq!(pi.len(), 2);
    /// // With many observations, stationary should be near-uniform
    /// assert!((pi[0] - pi[1]).abs() < 0.05);
    /// ```
    pub fn stationary_distribution(&mut self) -> Option<Vec<f64>> {
        self.build_chain();
        self.chain.stationary_distribution()
    }

    /// Get the stationary probability of a specific command.
    pub fn stationary_prob(&mut self, command: &str) -> f64 {
        let idx = match self.command_to_idx.get(command) {
            Some(&i) => i,
            None => return 0.0,
        };
        self.stationary_distribution()
            .and_then(|pi| pi.get(idx).copied())
            .unwrap_or(0.0)
    }

    /// Estimate the mixing time — steps to converge within ε of stationary.
    pub fn mixing_time(&mut self, epsilon: f64) -> Option<usize> {
        self.build_chain();
        self.chain.mixing_time(epsilon, 10_000)
    }

    /// Get the raw transition count from `from` to `to`.
    pub fn transition_count(&self, from: &str, to: &str) -> u64 {
        let i = match self.command_to_idx.get(from) {
            Some(&idx) => idx,
            None => return 0,
        };
        let j = match self.command_to_idx.get(to) {
            Some(&idx) => idx,
            None => return 0,
        };
        self.raw_counts[i * MAX_STATES + j]
    }

    /// Check whether the underlying Markov chain is ergodic.
    pub fn is_ergodic(&mut self) -> (bool, String) {
        self.build_chain();
        self.chain.is_ergodic()
    }

    /// Get a reference to the underlying ergodic-transport MarkovChain
    /// (useful for advanced use).
    pub fn chain(&self) -> &MarkovChain {
        &self.chain
    }

    /// Get a mutable reference to the underlying ergodic-transport MarkovChain.
    pub fn chain_mut(&mut self) -> &mut MarkovChain {
        &mut self.chain
    }

    /// Rebuild the internal MarkovChain from raw counts.
    /// This must be called before reading chain state after any recording.
    pub fn rebuild(&mut self) {
        self.build_chain();
    }

    /// Get all known command names in order of their state indices.
    pub fn commands(&self) -> &[String] {
        &self.idx_to_command
    }

    /// Look up the index for a command, if known.
    pub fn index_of(&self, command: &str) -> Option<usize> {
        self.command_to_idx.get(command).copied()
    }

    /// Rebuild the [MarkovChain] from raw counts with Laplace smoothing.
    fn build_chain(&mut self) {
        let n = self.num_states;
        if n == 0 {
            self.chain = MarkovChain::new();
            return;
        }

        let alpha = 1.0_f64;
        let mut flat = Vec::with_capacity(n * n);

        for i in 0..n {
            let row_sum: f64 = (0..n)
                .map(|j| self.raw_counts[i * MAX_STATES + j] as f64)
                .sum();
            let denom = row_sum + alpha * n as f64;
            for j in 0..n {
                let prob = (self.raw_counts[i * MAX_STATES + j] as f64 + alpha) / denom;
                flat.push(prob);
            }
        }

        self.chain = MarkovChain::from_flat(n, &flat);
    }

    /// Serialize counts and mapping to a JSON string for persistence.
    pub fn to_json(&self) -> serde_json::Result<String> {
        #[derive(Serialize)]
        struct Snapshot<'a> {
            command_to_idx: &'a HashMap<String, usize>,
            idx_to_command: &'a [String],
            raw_counts: &'a [u64],
            total_transitions: u64,
        }
        serde_json::to_string(&Snapshot {
            command_to_idx: &self.command_to_idx,
            idx_to_command: &self.idx_to_command,
            raw_counts: &self.raw_counts,
            total_transitions: self.total_transitions,
        })
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        #[derive(Deserialize)]
        struct Snapshot {
            command_to_idx: HashMap<String, usize>,
            idx_to_command: Vec<String>,
            raw_counts: Vec<u64>,
            total_transitions: u64,
        }
        let s: Snapshot = serde_json::from_str(json)?;
        let num_states = s.idx_to_command.len();
        let mut raw_counts = vec![0u64; MAX_STATES * MAX_STATES];
        for (idx, &val) in s.raw_counts.iter().enumerate() {
            if idx < raw_counts.len() {
                raw_counts[idx] = val;
            }
        }
        Ok(Self {
            command_to_idx: s.command_to_idx,
            idx_to_command: s.idx_to_command,
            num_states,
            chain: MarkovChain::new(),
            total_transitions: s.total_transitions,
            raw_counts,
        })
    }
}

impl Default for CommandPredictor {
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

    fn make_predictor() -> CommandPredictor {
        let mut p = CommandPredictor::new();
        p.record_sequence(&[
            "cargo build",
            "cargo test",
            "cargo build",
            "cargo run",
            "cargo build",
            "cargo test",
            "cargo build",
            "cargo test",
            "cargo build",
            "cargo clippy",
            "cargo build",
            "cargo test",
        ]);
        p
    }

    // ------------------------------------------------------------------
    // Construction & basic properties
    // ------------------------------------------------------------------

    #[test]
    fn test_new_is_empty() {
        let p = CommandPredictor::new();
        assert_eq!(p.num_commands(), 0);
        assert_eq!(p.total_transitions(), 0);
    }

    #[test]
    fn test_record_sequence_increases_state_count() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["a", "b", "c"]);
        assert_eq!(p.num_commands(), 3);
    }

    #[test]
    fn test_record_sequence_transition_count() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["a", "b", "c", "a"]);
        // transitions: a→b, b→c, c→a = 3
        assert_eq!(p.total_transitions(), 3);
    }

    #[test]
    fn test_empty_sequence_is_noop() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&[]);
        assert_eq!(p.num_commands(), 0);
        assert_eq!(p.total_transitions(), 0);
    }

    #[test]
    fn test_single_command_sequence_no_transitions() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["lonely"]);
        assert_eq!(p.num_commands(), 1);
        assert_eq!(p.total_transitions(), 0);
    }

    #[test]
    fn test_repeated_command_is_same_state() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["ls", "ls", "ls", "ls"]);
        assert_eq!(p.num_commands(), 1);
        assert_eq!(p.total_transitions(), 3);
    }

    #[test]
    fn test_record_single_transition() {
        let mut p = CommandPredictor::new();
        p.record(Some("git status"), "git add");
        assert_eq!(p.num_commands(), 2);
        assert_eq!(p.total_transitions(), 1);
    }

    #[test]
    fn test_record_without_prev_only_registers() {
        let mut p = CommandPredictor::new();
        p.record(None, "only");
        assert_eq!(p.num_commands(), 1);
        assert_eq!(p.total_transitions(), 0);
    }

    // ------------------------------------------------------------------
    // Transition counts
    // ------------------------------------------------------------------

    #[test]
    fn test_transition_count_known() {
        let p = make_predictor();
        assert!(p.transition_count("cargo build", "cargo test") >= 3);
    }

    #[test]
    fn test_transition_count_unknown_from() {
        let p = make_predictor();
        assert_eq!(p.transition_count("nonexistent", "cargo build"), 0);
    }

    #[test]
    fn test_transition_count_unknown_to() {
        let p = make_predictor();
        assert_eq!(p.transition_count("cargo build", "nonexistent"), 0);
    }

    #[test]
    fn test_transition_count_self_loop() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["x", "x", "x", "x"]);
        assert_eq!(p.transition_count("x", "x"), 3);
    }

    // ------------------------------------------------------------------
    // Commands index lookup
    // ------------------------------------------------------------------

    #[test]
    fn test_index_of_known() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["a", "b"]);
        assert_eq!(p.index_of("a"), Some(0));
        assert_eq!(p.index_of("b"), Some(1));
    }

    #[test]
    fn test_index_of_unknown() {
        let p = CommandPredictor::new();
        assert_eq!(p.index_of("anything"), None);
    }

    #[test]
    fn test_commands_list() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["z", "y", "x"]);
        let cmds = p.commands();
        assert_eq!(cmds.len(), 3);
        assert_eq!(cmds[0], "z");
        assert_eq!(cmds[2], "x");
    }

    // ------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------

    #[test]
    fn test_predict_top1() {
        let mut p = make_predictor();
        let r = p.predict_top_k("cargo build", 1).unwrap();
        assert_eq!(r.predictions.len(), 1);
        // "cargo test" is the most common follow-up
        assert_eq!(r.predictions[0].command, "cargo test");
    }

    #[test]
    fn test_predict_top3_respects_k() {
        let mut p = make_predictor();
        let r = p.predict_top_k("cargo build", 3).unwrap();
        assert_eq!(r.predictions.len(), 3);
    }

    #[test]
    fn test_predict_orders_by_confidence() {
        let mut p = make_predictor();
        let r = p.predict_top_k("cargo build", 5).unwrap();
        for w in r.predictions.windows(2) {
            assert!(
                w[0].confidence >= w[1].confidence,
                "predictions should be sorted descending"
            );
        }
    }

    #[test]
    fn test_predict_unknown_command_returns_none() {
        let mut p = make_predictor();
        assert!(p.predict_top_k("not_a_command", 3).is_none());
    }

    #[test]
    fn test_predict_confidence_in_range() {
        let mut p = CommandPredictor::new();
        for _ in 0..100 {
            p.record_sequence(&["x", "y"]);
        }
        let r = p.predict_top_k("x", 1).unwrap();
        assert!(r.predictions[0].confidence > 0.0);
        assert!(r.predictions[0].confidence <= 1.0);
    }

    #[test]
    fn test_predict_with_k_larger_than_states() {
        let mut p = make_predictor();
        // Request more predictions than states — should return all states
        let r = p.predict_top_k("cargo build", 100).unwrap();
        assert_eq!(r.predictions.len(), p.num_commands());
    }

    #[test]
    fn test_predict_deterministic_transition() {
        let mut p = CommandPredictor::new();
        // A always goes to B
        for _ in 0..1000 {
            p.record(Some("A"), "B");
        }
        let r = p.predict_top_k("A", 1).unwrap();
        assert_eq!(r.predictions[0].command, "B");
        assert!(r.predictions[0].confidence > 0.99);
    }

    // ------------------------------------------------------------------
    // PredictionResult formatting
    // ------------------------------------------------------------------

    #[test]
    fn test_ghost_text_format() {
        let mut p = make_predictor();
        let r = p.predict_top_k("cargo build", 2).unwrap();
        let ghost = r.ghost_text();
        assert!(ghost.starts_with("cargo test"));
        assert!(ghost.contains("  "));
    }

    #[test]
    fn test_detailed_format() {
        let mut p = make_predictor();
        let r = p.predict_top_k("cargo build", 3).unwrap();
        let detailed = r.detailed();
        assert!(detailed.starts_with("After `cargo build`:"));
        assert!(detailed.contains("cargo test"));
        assert!(detailed.contains("%"));
    }

    #[test]
    fn test_prediction_display_format() {
        let pred = Prediction {
            command: "cargo test".to_string(),
            confidence: 0.734,
        };
        let out = format!("{}", pred);
        assert!(out.contains("cargo test"));
        assert!(out.contains("73.4%"));
    }

    // ------------------------------------------------------------------
    // GhostCompleter
    // ------------------------------------------------------------------

    #[test]
    fn test_ghost_completer_returns_most_likely() {
        let mut p = make_predictor();
        let result = GhostCompleter::complete(&mut p, "cargo build");
        assert_eq!(result, Some("cargo test".to_string()));
    }

    #[test]
    fn test_ghost_completer_unknown_returns_none() {
        let mut p = make_predictor();
        let result = GhostCompleter::complete(&mut p, "nope");
        assert_eq!(result, None);
    }

    #[test]
    fn test_ghost_completer_empty_model_returns_none() {
        let mut p = CommandPredictor::new();
        let result = GhostCompleter::complete(&mut p, "anything");
        assert_eq!(result, None);
    }

    // ------------------------------------------------------------------
    // Stationary distribution
    // ------------------------------------------------------------------

    #[test]
    fn test_stationary_distribution_deterministic_cycle() {
        let mut p = CommandPredictor::new();
        for _ in 0..100 {
            p.record_sequence(&["A", "B", "C", "A"]);
        }
        let pi = p.stationary_distribution().unwrap();
        assert_eq!(pi.len(), 3);
        for prob in &pi {
            assert!((prob - 1.0 / 3.0).abs() < 0.02);
        }
    }

    #[test]
    fn test_stationary_distribution_absorbing() {
        let mut p = CommandPredictor::new();
        // A → A always (absorbing), B → A
        for _ in 0..50 {
            p.record(Some("A"), "A");
            p.record(Some("B"), "A");
        }
        let pi = p.stationary_distribution().unwrap();
        let a_idx = p.index_of("A").unwrap();
        assert!(pi[a_idx] > 0.9, "A should dominate: got {}", pi[a_idx]);
    }

    #[test]
    fn test_stationary_distribution_empty() {
        let mut p = CommandPredictor::new();
        assert!(p.stationary_distribution().is_none());
    }

    #[test]
    fn test_stationary_prob_known() {
        let mut p = make_predictor();
        let prob = p.stationary_prob("cargo build");
        assert!(prob > 0.0);
    }

    #[test]
    fn test_stationary_prob_unknown() {
        let mut p = make_predictor();
        assert_eq!(p.stationary_prob("nope"), 0.0);
    }

    // ------------------------------------------------------------------
    // Mixing time & ergodicity
    // ------------------------------------------------------------------

    #[test]
    fn test_mixing_time_returns_some() {
        let mut p = CommandPredictor::new();
        for _ in 0..100 {
            p.record_sequence(&["A", "B", "A"]);
        }
        let mt = p.mixing_time(0.01);
        assert!(mt.is_some());
    }

    #[test]
    fn test_mixing_time_empty_returns_none() {
        let mut p = CommandPredictor::new();
        assert!(p.mixing_time(0.01).is_none());
    }

    #[test]
    fn test_is_ergodic_with_cycle() {
        let mut p = CommandPredictor::new();
        for _ in 0..100 {
            p.record_sequence(&["A", "B", "A"]);
        }
        let (ok, _) = p.is_ergodic();
        assert!(ok);
    }

    #[test]
    fn test_is_ergodic_empty() {
        let mut p = CommandPredictor::new();
        let (ok, _) = p.is_ergodic();
        assert!(!ok);
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    #[test]
    fn test_serialization_roundtrip() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["cargo build", "cargo test", "cargo run"]);
        let json = p.to_json().unwrap();
        let restored = CommandPredictor::from_json(&json).unwrap();
        assert_eq!(restored.num_commands(), p.num_commands());
        assert_eq!(restored.total_transitions(), p.total_transitions());
        assert_eq!(
            restored.transition_count("cargo build", "cargo test"),
            1
        );
    }

    #[test]
    fn test_serialization_empty_model() {
        let p = CommandPredictor::new();
        let json = p.to_json().unwrap();
        let restored = CommandPredictor::from_json(&json).unwrap();
        assert_eq!(restored.num_commands(), 0);
        assert_eq!(restored.total_transitions(), 0);
    }

    #[test]
    fn test_serialization_restores_predictions() {
        let mut p = CommandPredictor::new();
        p.record_sequence(&["a", "b", "a", "c"]);
        let json = p.to_json().unwrap();
        let mut restored = CommandPredictor::from_json(&json).unwrap();
        let r = restored.predict_top_k("a", 2).unwrap();
        assert_eq!(r.predictions[0].command, "b");
    }

    // ------------------------------------------------------------------
    // Metadata in prediction result
    // ------------------------------------------------------------------

    #[test]
    fn test_predict_result_metadata() {
        let mut p = make_predictor();
        let r = p.predict_top_k("cargo build", 3).unwrap();
        assert!(r.num_states > 0);
        assert!(r.total_transitions > 0);
    }

    // ------------------------------------------------------------------
    // Default impl
    // ------------------------------------------------------------------

    #[test]
    fn test_command_predictor_default() {
        let p = CommandPredictor::default();
        assert_eq!(p.num_commands(), 0);
    }
}
