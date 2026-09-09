//! Conforming test evidence logging helper for bead f3d-01-upstream-pin-and-census-gl8.8.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Target Three.js r186 anchor commit for evidence summaries.
pub const UPSTREAM_COMMIT: &str = "148ef33ecb6d2502ff796d4554abd1549c95d519";

/// Structured JSON-lines test evidence event record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvidenceEvent {
    /// Wall clock timestamp in milliseconds.
    pub ts_wall: u64,
    /// Logical application clock timestamp when applicable.
    pub ts_app: Option<u64>,
    /// Test lane (unit, integration, fuzz, e2e-functional, e2e-forced-new-backend, e2e-exact, bench).
    pub lane: String,
    /// Bead identifier (e.g. "f3d-01-upstream-pin-and-census-gl8.8").
    pub bead: String,
    /// Test function name.
    pub test: String,
    /// Test step identifier.
    pub step: String,
    /// Severity level (info, warn, error, metric).
    pub level: String,
    /// Implementation route owner tag.
    pub owner: String,
    /// Submission route when rendering.
    pub route: Option<String>,
    /// Browser metadata for browser lanes.
    pub browser: Option<serde_json::Value>,
    /// Device generation when relevant.
    pub device_generation: Option<u32>,
    /// Scene generation when relevant.
    pub scene_generation: Option<u32>,
    /// Event message.
    pub msg: String,
    /// Additional bounded structured data.
    pub data: Option<serde_json::Value>,
}

/// Run-level evidence summary record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvidenceSummary {
    /// FrankenThreeD repository commit hash at HEAD.
    pub commit: String,
    /// Upstream Three.js pinned commit hash.
    pub upstream_commit: String,
    /// Bead identifier.
    pub bead: String,
    /// Execution run identifier.
    pub run_id: String,
    /// Count of passed test cases.
    pub pass: usize,
    /// Count of failed test cases.
    pub fail: usize,
    /// Diagnostic from first failure if any.
    pub first_failure: Option<String>,
}

/// Writer for conforming evidence archive layout: evidence/<bead-key>/<run-id>/
pub struct EvidenceWriter {
    /// Path to events.jsonl
    pub events_path: PathBuf,
    /// Path to summary.json
    pub summary_path: PathBuf,
}

impl EvidenceWriter {
    /// Initialize writer under base_dir/evidence/<bead-key>/<run-id>/
    pub fn init(base_dir: &Path, bead_key: &str, run_id: &str) -> io::Result<Self> {
        let dir = base_dir.join("evidence").join(bead_key).join(run_id);
        fs::create_dir_all(&dir)?;
        Ok(Self { events_path: dir.join("events.jsonl"), summary_path: dir.join("summary.json") })
    }

    /// Append one JSON-lines event to events.jsonl
    pub fn write_event(&self, event: &EvidenceEvent) -> io::Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(&self.events_path)?;
        let line = serde_json::to_string(event).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")
    }

    /// Write summary.json
    pub fn write_summary(&self, summary: &EvidenceSummary) -> io::Result<()> {
        let mut file = File::create(&self.summary_path)?;
        let json = serde_json::to_string_pretty(summary).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        file.write_all(json.as_bytes())?;
        file.write_all(b"\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_event_and_summary_roundtrip() {
        let base = if Path::new("/Volumes/USBNVME16TB/temp_agent_space").exists() {
            PathBuf::from("/Volumes/USBNVME16TB/temp_agent_space")
        } else {
            std::env::temp_dir()
        };
        let tmp = base.join(format!("f3d_ev_test_{}", std::process::id()));
        let writer = EvidenceWriter::init(&tmp, "01.8", "run-1").expect("init");

        let event = EvidenceEvent {
            ts_wall: 1725890000, ts_app: Some(16), lane: "unit".into(),
            bead: "f3d-01-upstream-pin-and-census-gl8.8".into(), test: "test_writer".into(),
            step: "start".into(), level: "info".into(), owner: "new-rust".into(),
            route: Some("specialized-webgpu".into()), browser: None,
            device_generation: Some(1), scene_generation: None,
            msg: "test event".into(), data: Some(serde_json::json!({"metric": 42})),
        };

        writer.write_event(&event).expect("write event");
        let content = fs::read_to_string(&writer.events_path).expect("read events");
        let parsed: EvidenceEvent = serde_json::from_str(content.trim()).expect("parse event");
        assert_eq!(event, parsed);

        let summary = EvidenceSummary {
            commit: "test_head".into(), upstream_commit: UPSTREAM_COMMIT.into(),
            bead: "01.8".into(), run_id: "run-1".into(), pass: 1, fail: 0, first_failure: None,
        };
        writer.write_summary(&summary).expect("write summary");
        let sum_content = fs::read_to_string(&writer.summary_path).expect("read summary");
        let parsed_sum: EvidenceSummary = serde_json::from_str(&sum_content).expect("parse summary");
        assert_eq!(summary, parsed_sum);
        let _ = fs::remove_dir_all(tmp);
    }
}
