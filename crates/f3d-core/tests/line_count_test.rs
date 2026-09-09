use std::path::Path;
use std::process::Command;

#[test]
fn test_line_count_script_produces_valid_json() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root");

    let script = workspace_root.join("scripts").join("line-count.sh");
    assert!(script.exists(), "scripts/line-count.sh must exist");

    let output = Command::new("bash")
        .arg(&script)
        .arg("--json")
        .current_dir(workspace_root)
        .output()
        .expect("execute line-count.sh --json");

    assert!(
        output.status.success(),
        "line-count.sh failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout_str).expect("line-count output must be valid json");

    assert!(parsed.get("total_new_rust").is_some());
    assert!(parsed.get("budget").is_some());
    let within_ceiling = parsed["budget"]["within_ceiling"]
        .as_bool()
        .expect("within_ceiling is bool");
    assert!(within_ceiling, "Implementation must be within ceiling");
}
