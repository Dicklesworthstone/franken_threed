//! Conforming test evidence logging helper for bead f3d-01-upstream-pin-and-census-gl8.8.
use serde::{Deserialize, Serialize};
use std::{fs::{self, File, OpenOptions}, io::{self, Write}, path::{Path, PathBuf}};
/// Target Three.js r186 anchor commit for evidence summaries.
pub const UPSTREAM_COMMIT: &str = "148ef33ecb6d2502ff796d4554abd1549c95d519";
/// Structured JSON-lines test evidence event record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvidenceEvent {
    /// Wall-clock timestamp (ms).
    pub ts_wall: u64,
    /// Logical application clock timestamp (ms).
    pub ts_app: Option<u64>,
    /// Test lane identifier.
    pub lane: String,
    /// Bead identifier.
    pub bead: String,
    /// Test function name.
    pub test: String,
    /// Test step identifier.
    pub step: String,
    /// Severity level.
    pub level: String,
    /// Implementation route owner tag.
    pub owner: String,
    /// Submission route when rendering.
    pub route: Option<String>,
    /// Browser metadata.
    pub browser: Option<serde_json::Value>,
    /// Device generation counter.
    pub device_generation: Option<u32>,
    /// Scene generation counter.
    pub scene_generation: Option<u32>,
    /// Event message.
    pub msg: String,
    /// Additional structured payload.
    pub data: Option<serde_json::Value>,
}
/// Run-level evidence summary record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvidenceSummary {
    /// Commit hash at HEAD.
    pub commit: String,
    /// Upstream Three.js pinned commit hash.
    pub upstream_commit: String,
    /// Bead identifier.
    pub bead: String,
    /// Execution run identifier.
    pub run_id: String,
    /// Pass count.
    pub pass: usize,
    /// Fail count.
    pub fail: usize,
    /// First failure diagnostic.
    pub first_failure: Option<String>,
}
/// Validates that an evidence event has all required fields including a non-empty owner.
pub fn validate_event(ev: &EvidenceEvent) -> Result<(), &'static str> {
    if ev.owner.trim().is_empty() { Err("missing required field: owner") } else { Ok(()) }
}
/// Writer for conforming evidence archive layout.
pub struct EvidenceWriter {
    /// Path to events.jsonl
    pub events_path: PathBuf,
    /// Path to summary.json
    pub summary_path: PathBuf,
}
impl EvidenceWriter {
    /// Initialize writer under base_dir/evidence/<bead-key>/<run-id>/
    pub fn init(base: &Path, bead: &str, run: &str) -> io::Result<Self> {
        let d = base.join("evidence").join(bead).join(run); fs::create_dir_all(&d)?;
        Ok(Self { events_path: d.join("events.jsonl"), summary_path: d.join("summary.json") })
    }
    /// Append one JSON-lines event to events.jsonl after validation.
    pub fn write_event(&self, ev: &EvidenceEvent) -> io::Result<()> {
        validate_event(ev).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let s = serde_json::to_string(ev).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        writeln!(OpenOptions::new().create(true).append(true).open(&self.events_path)?, "{s}")
    }
    /// Write summary.json.
    pub fn write_summary(&self, sum: &EvidenceSummary) -> io::Result<()> {
        let s = serde_json::to_string_pretty(sum).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        writeln!(File::create(&self.summary_path)?, "{s}")
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn sample(owner: &str) -> EvidenceEvent {
        EvidenceEvent { ts_wall: 1, ts_app: None, lane: "unit".into(), bead: "01.8".into(), test: "t".into(), step: "s".into(), level: "info".into(), owner: owner.into(), route: None, browser: None, device_generation: None, scene_generation: None, msg: "m".into(), data: None }
    }
    #[test]
    fn test_evidence_event_and_summary_roundtrip() {
        let d = (if Path::new("/Volumes/USBNVME16TB/temp_agent_space").exists() { PathBuf::from("/Volumes/USBNVME16TB/temp_agent_space") } else { std::env::temp_dir() }).join(format!("f3d_{}", std::process::id()));
        let w = EvidenceWriter::init(&d, "01.8", "r1").unwrap();
        let ev = sample("new-rust");
        w.write_event(&ev).unwrap();
        let parsed_ev: EvidenceEvent = serde_json::from_str(fs::read_to_string(&w.events_path).unwrap().trim()).unwrap();
        assert_eq!(ev, parsed_ev);
        let sum = EvidenceSummary { commit: "c".into(), upstream_commit: UPSTREAM_COMMIT.into(), bead: "01.8".into(), run_id: "r1".into(), pass: 1, fail: 0, first_failure: None };
        w.write_summary(&sum).unwrap();
        let parsed_sum: EvidenceSummary = serde_json::from_str(&fs::read_to_string(&w.summary_path).unwrap()).unwrap();
        assert_eq!(sum, parsed_sum);
        let _ = fs::remove_dir_all(d);
    }
    #[test]
    fn test_validate_event_rejects_missing_owner() {
        assert!(validate_event(&sample("")).is_err());
        assert!(validate_event(&sample("   ")).is_err());
    }
}
