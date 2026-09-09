#!/usr/bin/env bash
# scripts/test-oracle-checkout.sh
#
# Focused unit and contract tests for scripts/oracle-checkout.sh and upstream pin files.
# Validates pin metadata, distinct hashes, CLI options, safety refusals, and verification behavior.
# Non-destructive: NEVER deletes test scratch directories (retains and reports them).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPECTED_COMMIT="148ef33ecb6d2502ff796d4554abd1549c95d519"
EXPECTED_TAG_HASH="819fadd6b663b74d828c6af72a543024f74d3877"
EXPECTED_VERSION="0.186.0"

PASS_COUNT=0
FAIL_COUNT=0

assert_eq() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    echo "  [PASS] ${label}"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "  [FAIL] ${label}: expected '${expected}', got '${actual}'" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

assert_true() {
  local label="$1"
  local condition="$2"
  if eval "${condition}"; then
    echo "  [PASS] ${label}"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "  [FAIL] ${label}" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

echo "=== Running Oracle Pin & Script Unit Tests ==="

# Test Suite 1: File Presence
echo "Test Suite 1: File Presence"
assert_true "tools/upstream/pin.json exists" "[[ -f '${REPO_ROOT}/tools/upstream/pin.json' ]]"
assert_true "tools/upstream/PIN.md exists" "[[ -f '${REPO_ROOT}/tools/upstream/PIN.md' ]]"
assert_true "upstream/pin.json exists" "[[ -f '${REPO_ROOT}/upstream/pin.json' ]]"
assert_true "upstream/PIN.md exists" "[[ -f '${REPO_ROOT}/upstream/PIN.md' ]]"
assert_true "scripts/oracle-checkout.sh is executable" "[[ -x '${REPO_ROOT}/scripts/oracle-checkout.sh' ]]"

# Test Suite 2: Metadata Field Values via Node JSON parsing
echo "Test Suite 2: Metadata Field Values"
PIN_JSON="${REPO_ROOT}/tools/upstream/pin.json"

NODE_CHECK="$(node -e '
  const fs = require("fs");
  const data = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  const o = data.oracle || data;
  console.log([o.source_commit, o.tag_object_hash, o.package_version].join("|"));
' "${PIN_JSON}")"

IFS="|" read -r ACTUAL_COMMIT ACTUAL_TAG ACTUAL_VER <<< "${NODE_CHECK}"

assert_eq "source_commit in pin.json" "${ACTUAL_COMMIT}" "${EXPECTED_COMMIT}"
assert_eq "tag_object_hash in pin.json" "${ACTUAL_TAG}" "${EXPECTED_TAG_HASH}"
assert_eq "package_version in pin.json" "${ACTUAL_VER}" "${EXPECTED_VERSION}"
assert_true "source_commit != tag_object_hash" "[[ '${ACTUAL_COMMIT}' != '${ACTUAL_TAG}' ]]"

# Test Suite 3: CLI Interface
echo "Test Suite 3: CLI Interface"
HELP_OUTPUT="$("${REPO_ROOT}/scripts/oracle-checkout.sh" --help)"
assert_true "CLI help mentions source commit" "echo '${HELP_OUTPUT}' | grep -q '${EXPECTED_COMMIT}'"
assert_true "CLI help mentions tag object hash" "echo '${HELP_OUTPUT}' | grep -q '${EXPECTED_TAG_HASH}'"

# Test Suite 4: Safety Refusal and Verification Behavior
echo "Test Suite 4: Safety Invariants & Verification Behavior"

# Prepare test scratch directory in configured agent space
TEMP_BASE="/Volumes/USBNVME16TB/temp_agent_space"
if [[ ! -d "${TEMP_BASE}" ]]; then
  TEMP_BASE="/tmp"
fi
TEST_SCRATCH="$(mktemp -d "${TEMP_BASE}/f3d_oracle_test.XXXXXX")"
echo "  Note: Test scratch directory retained at: ${TEST_SCRATCH} (no destructive cleanup)"

# Test 4a: Checkout refusal on non-git directory
NON_GIT_DIR="${TEST_SCRATCH}/non_git_target"
mkdir -p "${NON_GIT_DIR}"
touch "${NON_GIT_DIR}/some_file.txt"

set +e
"${REPO_ROOT}/scripts/oracle-checkout.sh" --checkout --target-dir "${NON_GIT_DIR}" >/dev/null 2>&1
EXIT_NON_GIT=$?
set -e
assert_eq "Checkout refuses non-git target directory without mutation" "${EXIT_NON_GIT}" "1"

# Test 4b: Checkout refusal on mismatched existing checkout
SYNTH_DIR="${TEST_SCRATCH}/synth_mismatched_repo"
mkdir -p "${SYNTH_DIR}"
git -C "${SYNTH_DIR}" init -q
git -C "${SYNTH_DIR}" config user.email "test@example.com"
git -C "${SYNTH_DIR}" config user.name "Test"
git -C "${SYNTH_DIR}" commit --allow-empty -m "Dummy commit" -q

set +e
"${REPO_ROOT}/scripts/oracle-checkout.sh" --checkout --target-dir "${SYNTH_DIR}" >/dev/null 2>&1
EXIT_MISMATCH=$?
set -e
assert_eq "Checkout refuses mismatched existing HEAD without mutation" "${EXIT_MISMATCH}" "1"

# Test 4c: Verify fails on mismatching commit
set +e
"${REPO_ROOT}/scripts/oracle-checkout.sh" --verify --target-dir "${SYNTH_DIR}" >/dev/null 2>&1
EXIT_VERIFY_FAIL=$?
set -e
assert_eq "Verify fails on commit mismatch" "${EXIT_VERIFY_FAIL}" "1"

# Test 4d: JSON report validity and null built_artifacts_exist when unrequested
JSON_REPORT="$("${REPO_ROOT}/scripts/oracle-checkout.sh" --verify --target-dir "${SYNTH_DIR}" --json || true)"
JSON_PARSE_CHECK="$(node -e '
  try {
    const report = JSON.parse(process.argv[1]);
    if (report.status === "FAIL" &&
        report.head_matches === false &&
        report.verify_built_requested === false &&
        report.built_artifacts_exist === null &&
        Array.isArray(report.missing_artifacts)) {
      console.log("VALID_REPORT");
    } else {
      console.log("INVALID_SCHEMA: " + JSON.stringify(report));
    }
  } catch (e) {
    console.log("PARSE_ERROR: " + e.message);
  }
' "${JSON_REPORT}")"

assert_eq "JSON output reports built_artifacts_exist: null when unrequested" "${JSON_PARSE_CHECK}" "VALID_REPORT"

# Test 4e: JSON report when --verify-built IS requested
JSON_REPORT_BUILT="$("${REPO_ROOT}/scripts/oracle-checkout.sh" --verify --verify-built --target-dir "${SYNTH_DIR}" --json || true)"
JSON_PARSE_CHECK_BUILT="$(node -e '
  try {
    const report = JSON.parse(process.argv[1]);
    if (report.verify_built_requested === true &&
        report.built_artifacts_exist === false &&
        Array.isArray(report.missing_artifacts) &&
        report.missing_artifacts.length > 0) {
      console.log("VALID_BUILT_REPORT");
    } else {
      console.log("INVALID_BUILT_SCHEMA: " + JSON.stringify(report));
    }
  } catch (e) {
    console.log("PARSE_ERROR: " + e.message);
  }
' "${JSON_REPORT_BUILT}")"

assert_eq "JSON output reports built_artifacts_exist: false when requested and missing" "${JSON_PARSE_CHECK_BUILT}" "VALID_BUILT_REPORT"

# Test 4f: Build refusal without ALLOW_LOCAL_BUILD=1
set +e
"${REPO_ROOT}/scripts/oracle-checkout.sh" --build --target-dir "${SYNTH_DIR}" >/dev/null 2>&1
EXIT_BUILD_REFUSE=$?
set -e
assert_eq "Build refuses unverified local execution (requires RCH)" "${EXIT_BUILD_REFUSE}" "1"

echo ""
echo "=== Test Summary ==="
echo "Passed: ${PASS_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Scratch directory preserved at: ${TEST_SCRATCH}"

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "Tests FAILED." >&2
  exit 1
else
  echo "All oracle tests PASSED."
  exit 0
fi
