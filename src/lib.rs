//! # Terminal Forecast Harness
//!
//! A standalone crate wrapping [`ergodic_transport`] for terminal command
//! forecasting, anomaly detection, and resource prediction.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`command_predictor`] | Top-K next command prediction with confidence |
//! | [`workflow_anomaly`] | KL divergence + Wasserstein-1 shift detection |
//! | [`resource_predictor`] | EMA-based resource usage forecasting |
//!
//! ## Design
//!
//! The Markov chain lives in `ergodic_transport`. This crate adds:
//!
//! - **Terminal-facing nomenclature**: commands instead of abstract states
//! - **Command-state mapping**: dynamically growing, with an overflow cap
//! - **Top-K prediction**: sorted by transition probability from current state
//! - **Shift detection**: compares observed vs stationary distributions
//! - **Resource modelling**: EMA with variance estimates for memory/CPU/duration

pub mod command_predictor;
pub mod workflow_anomaly;
pub mod resource_predictor;

pub use command_predictor::{
    CommandPredictor, Prediction, PredictionResult, GhostCompleter,
};
pub use workflow_anomaly::{
    WorkflowAnomaly, AnomalyRecord, Severity,
};
pub use resource_predictor::{
    ResourcePredictor, ResourceObservation, ResourcePrediction, ResourceAvailability, ResourceStats,
};
