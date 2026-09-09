#!/usr/bin/env bash
# scripts/line-count.sh
# Deterministic budget accounting method for FrankenThreeD (Plan §17.2 / AGENTS.md).
# Tracks line counts across distinct categories:
# - New Rust (Execution core & compiler)
# - New Rust Tests
# - New JS/TS (Ingestion tooling & host adapters)
# - Authored WGSL
# - Retained Upstream Modules (Three.js r186)
# - Generated Artifacts
# - Documentation & Contracts
#
# Compares actual Rust implementation against:
# - Planned Subtotal: 205,000 lines
# - Asupersync Foundation: 10,000 lines
# - Planned Total: 215,000 lines
# - Hard Ceiling: 245,000 lines

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_MODE=false

for arg in "$@"; do
    case "$arg" in
        --json) JSON_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--json]"
            echo "Report line counts categorized by implementation ownership and language."
            exit 0
            ;;
    esac
done

count_lines() {
    local pattern="$1"
    local dir="$2"
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    find "$dir" -type f -name "$pattern" 2>/dev/null | xargs wc -l 2>/dev/null | tail -n 1 | awk '{print $1}' || echo 0
}

# 1. New Rust Core & Crate Sources (excluding standalone integration tests)
NEW_RUST_SRC=0
if [ -d "$REPO_ROOT/crates" ]; then
    while IFS= read -r file; do
        if [[ "$file" =~ /tests/ ]] || [[ "$file" =~ _test\.rs$ ]]; then
            continue
        fi
        lines=$(wc -l < "$file" 2>/dev/null || echo 0)
        NEW_RUST_SRC=$((NEW_RUST_SRC + lines))
    done < <(find "$REPO_ROOT/crates" -type f -name "*.rs" 2>/dev/null || true)
fi

# 2. Rust Tests (unit & integration)
RUST_TESTS=0
if [ -d "$REPO_ROOT/crates" ]; then
    while IFS= read -r file; do
        if [[ "$file" =~ /tests/ ]] || [[ "$file" =~ _test\.rs$ ]]; then
            lines=$(wc -l < "$file" 2>/dev/null || echo 0)
            RUST_TESTS=$((RUST_TESTS + lines))
        fi
    done < <(find "$REPO_ROOT/crates" -type f -name "*.rs" 2>/dev/null || true)
fi
if [ -d "$REPO_ROOT/tests" ]; then
    while IFS= read -r file; do
        lines=$(wc -l < "$file" 2>/dev/null || echo 0)
        RUST_TESTS=$((RUST_TESTS + lines))
    done < <(find "$REPO_ROOT/tests" -type f -name "*.rs" 2>/dev/null || true)
fi

TOTAL_NEW_RUST=$((NEW_RUST_SRC + RUST_TESTS))

# 3. New JS/TS (tools, host adapters, test harness)
NEW_JS_TS=0
if [ -d "$REPO_ROOT/tools" ]; then
    while IFS= read -r file; do
        if [[ "$file" =~ node_modules/ ]] || [[ "$file" =~ /upstream/ ]]; then
            continue
        fi
        lines=$(wc -l < "$file" 2>/dev/null || echo 0)
        NEW_JS_TS=$((NEW_JS_TS + lines))
    done < <(find "$REPO_ROOT/tools" -type f \( -name "*.js" -o -name "*.mjs" -o -name "*.ts" \) 2>/dev/null || true)
fi

# 4. Authored WGSL
AUTHORED_WGSL=0
while IFS= read -r file; do
    if [[ "$file" =~ /generated/ ]] || [[ "$file" =~ /target/ ]]; then
        continue
    fi
    lines=$(wc -l < "$file" 2>/dev/null || echo 0)
    AUTHORED_WGSL=$((AUTHORED_WGSL + lines))
done < <(find "$REPO_ROOT" -type f -name "*.wgsl" 2>/dev/null || true)

# 5. Retained Upstream Modules (Three.js checkout)
RETAINED_UPSTREAM=0
if [ -d "$REPO_ROOT/upstream" ]; then
    while IFS= read -r file; do
        lines=$(wc -l < "$file" 2>/dev/null || echo 0)
        RETAINED_UPSTREAM=$((RETAINED_UPSTREAM + lines))
    done < <(find "$REPO_ROOT/upstream" -type f \( -name "*.js" -o -name "*.ts" -o -name "*.glsl" \) 2>/dev/null || true)
fi

# 6. Documentation & Contracts
DOC_LINES=0
while IFS= read -r file; do
    if [[ "$file" =~ /node_modules/ ]] || [[ "$file" =~ /target/ ]]; then
        continue
    fi
    lines=$(wc -l < "$file" 2>/dev/null || echo 0)
    DOC_LINES=$((DOC_LINES + lines))
done < <(find "$REPO_ROOT" -type f -name "*.md" 2>/dev/null || true)

# Budget thresholds
PLANNED_SUBTOTAL=205000
PLANNED_CEILING=245000

if [ "$JSON_MODE" = true ]; then
    cat <<EOF
{
  "new_rust_src": $NEW_RUST_SRC,
  "rust_tests": $RUST_TESTS,
  "total_new_rust": $TOTAL_NEW_RUST,
  "new_js_ts": $NEW_JS_TS,
  "authored_wgsl": $AUTHORED_WGSL,
  "retained_upstream": $RETAINED_UPSTREAM,
  "documentation": $DOC_LINES,
  "budget": {
    "planned_subtotal": $PLANNED_SUBTOTAL,
    "ceiling": $PLANNED_CEILING,
    "remaining_budget": $((PLANNED_CEILING - TOTAL_NEW_RUST)),
    "within_ceiling": $( [ "$TOTAL_NEW_RUST" -le "$PLANNED_CEILING" ] && echo "true" || echo "false" )
  }
}
EOF
else
    echo "================================================================="
    echo " FrankenThreeD Code Budget & Line Count Report"
    echo "================================================================="
    printf "%-35s %12s lines\n" "New Rust Source (crates/*/src):" "$NEW_RUST_SRC"
    printf "%-35s %12s lines\n" "Rust Tests (unit & integration):" "$RUST_TESTS"
    printf "%-35s %12s lines\n" "Total New Rust:" "$TOTAL_NEW_RUST"
    printf "%-35s %12s lines\n" "New JS / TS Tooling & Adapters:" "$NEW_JS_TS"
    printf "%-35s %12s lines\n" "Authored WGSL Kernels:" "$AUTHORED_WGSL"
    printf "%-35s %12s lines\n" "Retained Upstream (Three.js):" "$RETAINED_UPSTREAM"
    printf "%-35s %12s lines\n" "Documentation & Contracts:" "$DOC_LINES"
    echo "-----------------------------------------------------------------"
    printf "%-35s %12s lines\n" "Planned Rust Ceiling:" "$PLANNED_CEILING"
    printf "%-35s %12s lines\n" "Remaining Rust Budget:" "$((PLANNED_CEILING - TOTAL_NEW_RUST))"
    echo "================================================================="
    if [ "$TOTAL_NEW_RUST" -gt "$PLANNED_CEILING" ]; then
        echo "WARNING: Rust line count exceeds 245,000 ceiling! Triggers review."
        exit 1
    else
        echo "Status: Within planned budget ceiling."
    fi
fi
