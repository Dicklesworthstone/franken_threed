use std::fs;
use std::path::Path;

#[test]
fn test_crates_forbid_unsafe_code() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root");

    let crates_dir = workspace_root.join("crates");
    if !crates_dir.exists() {
        return;
    }

    for entry in fs::read_dir(&crates_dir).expect("read crates dir") {
        let entry = entry.expect("crate entry");
        let path = entry.path();
        if path.is_dir() {
            let lib_rs = path.join("src").join("lib.rs");
            if lib_rs.exists() {
                let content = fs::read_to_string(&lib_rs).expect("read lib.rs");
                assert!(
                    content.contains("#![forbid(unsafe_code)]"),
                    "Crate at {:?} must declare #![forbid(unsafe_code)]",
                    path
                );
            }
        }
    }
}

#[test]
fn test_cargo_toml_lint_configuration() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root");

    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml).expect("read root Cargo.toml");

    assert!(
        content.contains("unsafe_code = \"forbid\""),
        "Workspace root Cargo.toml must configure unsafe_code = forbid"
    );
    assert!(
        content.contains("crates/f3d-core"),
        "Workspace root Cargo.toml must include crates/f3d-core as member"
    );
}

#[test]
fn test_toolchain_file_pinned() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root");

    let toolchain_toml = workspace_root.join("rust-toolchain.toml");
    let content = fs::read_to_string(&toolchain_toml).expect("read rust-toolchain.toml");

    assert!(
        content.contains("nightly-2026-08-31"),
        "rust-toolchain.toml must pin dated nightly-2026-08-31"
    );
    assert!(
        content.contains("wasm32-unknown-unknown"),
        "rust-toolchain.toml must include wasm32-unknown-unknown target"
    );
}
