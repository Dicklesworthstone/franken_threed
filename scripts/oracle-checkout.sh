#!/usr/bin/env bash
# scripts/oracle-checkout.sh
#
# FrankenThreeD Upstream Oracle Acquisition & Verification Script
# Pinned Oracle: Three.js r186 (commit 148ef33ecb6d2502ff796d4554abd1549c95d519)
#
# Requirements:
# - Plan §2.2, §5.11, §5.17
# - AGENTS.md "Upstream Pin Discipline"
# - Bead: f3d-01-upstream-pin-and-census-gl8.1
#
# Non-destructive & Immutable:
# - Strictly refuses to mutate an existing mismatched, corrupted, or dirty checkout.
# - Requires checked tag peel to source commit and explicit tag object hash separation.
# - Validates pin metadata JSON fields semantically via node.
# - Emits valid deterministic JSON report via node serialization.
# - Refuses unverified local builds (directs to RCH per policy).

set -euo pipefail

# Pinned immutable oracle constants
PINNED_RELEASE_NAME="Three.js r186"
PINNED_RELEASE_DATE="2026-09-08"
PINNED_SOURCE_COMMIT="148ef33ecb6d2502ff796d4554abd1549c95d519"
PINNED_TAG_OBJECT_HASH="819fadd6b663b74d828c6af72a543024f74d3877"
PINNED_TAG_NAME="r186"
PINNED_PACKAGE_VERSION="0.186.0"
PINNED_REPO_URL="https://github.com/mrdoob/three.js.git"
REQUIRED_USER_AGENT="OpenAI File Downloader, XaiImageApiFetch/1.0"

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_DIR="${REPO_ROOT}/upstream/three.js"
PIN_JSON="${REPO_ROOT}/tools/upstream/pin.json"
PIN_MD="${REPO_ROOT}/tools/upstream/PIN.md"

DO_CHECKOUT=0
DO_VERIFY=0
VERIFY_BUILT=0
JSON_OUTPUT=0
BUILD=0
RECONCILE=0
ALLOW_LOCAL_BUILD="${ALLOW_LOCAL_BUILD:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --checkout        Fetch/checkout pinned upstream commit into upstream/three.js
  --verify          Verify that checkout matches pinned source commit, tag hash, and pin metadata
  --verify-built    Additionally verify that required built artifacts exist
  --build           Prepare upstream build via RCH (or with ALLOW_LOCAL_BUILD=1)
  --reconcile       Run package reconciliation tool and require clean / explained report
  --all             Execute checkout, reconcile, and full verification
  --json            Output structured verification results in valid JSON
  --target-dir DIR  Override destination directory (default: upstream/three.js)
  -h, --help        Show this help message

Pinned Constants:
  Release:          ${PINNED_RELEASE_NAME} (${PINNED_RELEASE_DATE})
  Source Commit:    ${PINNED_SOURCE_COMMIT}
  Tag Object Hash:  ${PINNED_TAG_OBJECT_HASH} (refs/tags/${PINNED_TAG_NAME})
  Package Version:  ${PINNED_PACKAGE_VERSION}
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkout)
      DO_CHECKOUT=1
      shift
      ;;
    --verify)
      DO_VERIFY=1
      shift
      ;;
    --verify-built)
      VERIFY_BUILT=1
      shift
      ;;
    --build)
      BUILD=1
      shift
      ;;
    --reconcile)
      RECONCILE=1
      shift
      ;;
    --all)
      MODE="checkout"
      RECONCILE=1
      VERIFY_BUILT=1
      shift
      ;;
    --json)
      JSON_OUTPUT=1
      shift
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

do_checkout() {
  if [[ "${JSON_OUTPUT}" -eq 0 ]]; then
    echo "=== FrankenThreeD Oracle Acquisition ==="
    echo "Target directory: ${TARGET_DIR}"
    echo "Pinned Commit:    ${PINNED_SOURCE_COMMIT}"
    echo "Tag Object Hash:  ${PINNED_TAG_OBJECT_HASH}"
  fi

  # Check existing directory safety - NEVER mutate mismatched checkout in place
  if [[ -e "${TARGET_DIR}" ]]; then
    if [[ ! -d "${TARGET_DIR}/.git" ]]; then
      echo "ERROR: Target directory '${TARGET_DIR}' exists and is not a git repository." >&2
      echo "Refusing to overwrite or mutate non-repository destination." >&2
      exit 1
    fi

    # Check for uncommitted modifications
    local DIRTY_STATUS
    DIRTY_STATUS="$(git -C "${TARGET_DIR}" status --porcelain 2>/dev/null || echo "error")"
    if [[ -n "${DIRTY_STATUS}" ]]; then
      echo "ERROR: Existing checkout at '${TARGET_DIR}' has uncommitted changes or dirty status." >&2
      echo "Refusing to mutate dirty checkout in place." >&2
      exit 1
    fi

    local CURRENT_HEAD
    CURRENT_HEAD="$(git -C "${TARGET_DIR}" rev-parse HEAD 2>/dev/null || echo "")"
    if [[ "${CURRENT_HEAD}" != "${PINNED_SOURCE_COMMIT}" ]]; then
      echo "ERROR: Existing checkout at '${TARGET_DIR}' is at commit '${CURRENT_HEAD}'," >&2
      echo "which does not match pinned oracle commit '${PINNED_SOURCE_COMMIT}'." >&2
      echo "Refusing to mutate or fetch over an existing mismatched checkout in place." >&2
      exit 1
    fi

    if [[ "${JSON_OUTPUT}" -eq 0 ]]; then
      echo "Verified existing oracle checkout at ${TARGET_DIR} matches pinned commit ${PINNED_SOURCE_COMMIT}."
    fi
  else
    # Fresh checkout only when destination does not exist
    if [[ "${JSON_OUTPUT}" -eq 0 ]]; then
      echo "Performing clean acquisition into ${TARGET_DIR}..."
    fi
    mkdir -p "$(dirname "${TARGET_DIR}")"
    mkdir -p "${TARGET_DIR}"
    git -C "${TARGET_DIR}" init -q
    git -C "${TARGET_DIR}" remote add origin "${PINNED_REPO_URL}"
    git -C "${TARGET_DIR}" config http.userAgent "${REQUIRED_USER_AGENT}"
    git -C "${TARGET_DIR}" fetch --depth 1 origin "${PINNED_SOURCE_COMMIT}"
    git -C "${TARGET_DIR}" checkout -q --detach FETCH_HEAD
    git -C "${TARGET_DIR}" fetch --depth 1 origin "refs/tags/${PINNED_TAG_NAME}:refs/tags/${PINNED_TAG_NAME}"
  fi

  if [[ "${BUILD}" -eq 1 ]]; then
    do_build
  fi

  if [[ "${RECONCILE}" -eq 1 ]]; then
    do_reconcile
  fi

  do_verify
}

do_build() {
  if [[ "${JSON_OUTPUT}" -eq 0 ]]; then
    echo "=== Upstream Package Build Check ==="
  fi
  if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "ERROR: Target directory ${TARGET_DIR} does not exist. Run --checkout first." >&2
    exit 1
  fi

  if [[ "${ALLOW_LOCAL_BUILD}" -ne 1 ]]; then
    echo "NOTICE: Building upstream package must be offloaded to RCH per central build policy." >&2
    echo "Central verification command via RCH:" >&2
    echo "  rch exec -- env CARGO_TARGET_DIR=\"\${RCH_TARGET_BASE:-\${TMPDIR:-/tmp}}/rch_target_f3d_check\" bash -c 'cd ${TARGET_DIR} && npm ci && npm run build'" >&2
    echo "To allow local build explicitly in an isolated worker, set ALLOW_LOCAL_BUILD=1." >&2
    exit 1
  fi

  (
    cd "${TARGET_DIR}"
    npm ci
    npm run build
  )
}

do_reconcile() {
  local RECONCILE_SCRIPT="${REPO_ROOT}/tools/upstream/reconcile_package.mjs"
  if [[ ! -f "${RECONCILE_SCRIPT}" ]]; then
    echo "ERROR: Reconciliation script ${RECONCILE_SCRIPT} not found." >&2
    exit 1
  fi

  local REPORT_OUT="${REPO_ROOT}/evidence/01.1/reconciliation_report.json"
  mkdir -p "$(dirname "${REPORT_OUT}")"

  if [[ "${JSON_OUTPUT}" -eq 0 ]]; then
    echo "Running upstream package reconciliation (${RECONCILE_SCRIPT})..."
  fi

  # Run reconciliation tool; must succeed and fail on unexplained discrepancies
  node "${RECONCILE_SCRIPT}" \
    --package-dir "${TARGET_DIR}" \
    --out "${REPORT_OUT}" \
    --fail-on-unexplained
}

do_verify() {
  local CHECKOUT_EXISTS=false
  local HEAD_MATCHES=false
  local ACTUAL_HEAD=""
  local TAG_OBJECT_MATCHES=false
  local ACTUAL_TAG_OBJECT=""
  local TAG_PEEL_MATCHES=false
  local ACTUAL_TAG_PEEL=""
  local PIN_METADATA_EXISTS=false
  local PIN_JSON_VALID=false
  local BUILT_ARTIFACTS_EXIST="null"
  local MISSING_ARTIFACTS=()
  local STATUS="PASS"

  # 1. Check target git directory
  if [[ -d "${TARGET_DIR}/.git" ]]; then
    CHECKOUT_EXISTS=true
    ACTUAL_HEAD="$(git -C "${TARGET_DIR}" rev-parse HEAD 2>/dev/null || echo "")"
    if [[ "${ACTUAL_HEAD}" == "${PINNED_SOURCE_COMMIT}" ]]; then
      HEAD_MATCHES=true
    else
      STATUS="FAIL"
    fi

    # 2. Check annotated tag object hash
    ACTUAL_TAG_OBJECT="$(git -C "${TARGET_DIR}" rev-parse "refs/tags/${PINNED_TAG_NAME}" 2>/dev/null || echo "")"
    if [[ "${ACTUAL_TAG_OBJECT}" == "${PINNED_TAG_OBJECT_HASH}" ]]; then
      TAG_OBJECT_MATCHES=true
    else
      STATUS="FAIL"
    fi

    # 3. Check tag peeled commit (must resolve to the source commit)
    ACTUAL_TAG_PEEL="$(git -C "${TARGET_DIR}" rev-parse "refs/tags/${PINNED_TAG_NAME}^{commit}" 2>/dev/null || echo "")"
    if [[ "${ACTUAL_TAG_PEEL}" == "${PINNED_SOURCE_COMMIT}" ]]; then
      TAG_PEEL_MATCHES=true
    else
      STATUS="FAIL"
    fi
  else
    STATUS="FAIL"
  fi

  # 4. Check pin metadata files existence
  if [[ -f "${PIN_JSON}" && -f "${PIN_MD}" ]]; then
    PIN_METADATA_EXISTS=true

    # 5. Semantic field validation of pin.json via node (not regex or grep)
    if node -e '
      const fs = require("fs");
      const [pinPath, expectedCommit, expectedTag, expectedVer] = process.argv.slice(1);
      const data = JSON.parse(fs.readFileSync(pinPath, "utf8"));
      const o = data.oracle || data;
      if (o.source_commit !== expectedCommit) process.exit(1);
      if (o.tag_object_hash !== expectedTag) process.exit(2);
      if (o.source_commit === o.tag_object_hash) process.exit(3);
      if (o.package_version !== expectedVer) process.exit(4);
      process.exit(0);
    ' "${PIN_JSON}" "${PINNED_SOURCE_COMMIT}" "${PINNED_TAG_OBJECT_HASH}" "${PINNED_PACKAGE_VERSION}" 2>/dev/null; then
      PIN_JSON_VALID=true
    else
      PIN_JSON_VALID=false
      STATUS="FAIL"
    fi
  else
    STATUS="FAIL"
  fi

  # 6. Check required built artifacts if requested
  if [[ "${VERIFY_BUILT}" -eq 1 ]]; then
    BUILT_ARTIFACTS_EXIST="true"
    local REQUIRED_ARTIFACTS=(
      "build/three.module.js"
      "build/three.webgpu.js"
      "build/three.tsl.js"
      "build/three.cjs"
    )
    for art in "${REQUIRED_ARTIFACTS[@]}"; do
      if [[ ! -f "${TARGET_DIR}/${art}" ]]; then
        BUILT_ARTIFACTS_EXIST="false"
        MISSING_ARTIFACTS+=("${art}")
        STATUS="FAIL"
      fi
    done
  fi

  if [[ "${JSON_OUTPUT}" -eq 1 ]]; then
    # Deterministic JSON emission via node serialization using environment variables
    REPORT_STATUS="${STATUS}" \
    REPORT_RELEASE_NAME="${PINNED_RELEASE_NAME}" \
    REPORT_RELEASE_DATE="${PINNED_RELEASE_DATE}" \
    REPORT_EXP_COMMIT="${PINNED_SOURCE_COMMIT}" \
    REPORT_ACT_HEAD="${ACTUAL_HEAD}" \
    REPORT_HEAD_MATCHES="${HEAD_MATCHES}" \
    REPORT_EXP_TAG="${PINNED_TAG_OBJECT_HASH}" \
    REPORT_ACT_TAG="${ACTUAL_TAG_OBJECT}" \
    REPORT_TAG_OBJ_MATCHES="${TAG_OBJECT_MATCHES}" \
    REPORT_ACT_PEEL="${ACTUAL_TAG_PEEL}" \
    REPORT_TAG_PEEL_MATCHES="${TAG_PEEL_MATCHES}" \
    REPORT_TARGET_DIR="${TARGET_DIR}" \
    REPORT_CHECKOUT_EXISTS="${CHECKOUT_EXISTS}" \
    REPORT_PIN_META_EXISTS="${PIN_METADATA_EXISTS}" \
    REPORT_PIN_JSON_VALID="${PIN_JSON_VALID}" \
    REPORT_VERIFY_BUILT_REQ="${VERIFY_BUILT}" \
    REPORT_BUILT_EXIST="${BUILT_ARTIFACTS_EXIST}" \
    REPORT_MISSING_ARTS="${MISSING_ARTIFACTS[*]-}" \
    node -e '
      const e = process.env;
      const missing = e.REPORT_MISSING_ARTS ? e.REPORT_MISSING_ARTS.split(" ").filter(Boolean) : [];
      const builtExist = (e.REPORT_VERIFY_BUILT_REQ === "1")
        ? (e.REPORT_BUILT_EXIST === "true")
        : null;
      const report = {
        status: e.REPORT_STATUS,
        release_name: e.REPORT_RELEASE_NAME,
        release_date: e.REPORT_RELEASE_DATE,
        expected_source_commit: e.REPORT_EXP_COMMIT,
        actual_head: e.REPORT_ACT_HEAD,
        head_matches: e.REPORT_HEAD_MATCHES === "true",
        expected_tag_object_hash: e.REPORT_EXP_TAG,
        actual_tag_object_hash: e.REPORT_ACT_TAG,
        tag_object_matches: e.REPORT_TAG_OBJ_MATCHES === "true",
        actual_tag_peeled_commit: e.REPORT_ACT_PEEL,
        tag_peel_matches: e.REPORT_TAG_PEEL_MATCHES === "true",
        target_dir: e.REPORT_TARGET_DIR,
        checkout_exists: e.REPORT_CHECKOUT_EXISTS === "true",
        pin_metadata_exists: e.REPORT_PIN_META_EXISTS === "true",
        pin_json_valid: e.REPORT_PIN_JSON_VALID === "true",
        verify_built_requested: e.REPORT_VERIFY_BUILT_REQ === "1",
        built_artifacts_exist: builtExist,
        missing_artifacts: missing
      };
      console.log(JSON.stringify(report, null, 2));
    '
  else
    echo "=== FrankenThreeD Oracle Verification Report ==="
    echo "Status:                     ${STATUS}"
    echo "Target directory:           ${TARGET_DIR} (exists: ${CHECKOUT_EXISTS})"
    echo "Expected source commit:     ${PINNED_SOURCE_COMMIT}"
    echo "Actual HEAD commit:         ${ACTUAL_HEAD} (matches: ${HEAD_MATCHES})"
    echo "Expected tag object hash:   ${PINNED_TAG_OBJECT_HASH}"
    echo "Actual tag object hash:     ${ACTUAL_TAG_OBJECT} (matches: ${TAG_OBJECT_MATCHES})"
    echo "Actual tag peeled commit:   ${ACTUAL_TAG_PEEL} (matches: ${TAG_PEEL_MATCHES})"
    echo "Pin metadata files:         exists: ${PIN_METADATA_EXISTS}, valid: ${PIN_JSON_VALID}"
    if [[ "${VERIFY_BUILT}" -eq 1 ]]; then
      echo "Built artifacts check:      ${BUILT_ARTIFACTS_EXIST}"
      if [[ ${#MISSING_ARTIFACTS[@]} -gt 0 ]]; then
        echo "Missing built artifacts:    ${MISSING_ARTIFACTS[*]}"
      fi
    else
      echo "Built artifacts check:      null (not requested)"
    fi

    if [[ "${STATUS}" != "PASS" ]]; then
      echo "VERIFICATION FAILED." >&2
      return 1
    else
      echo "VERIFICATION PASSED."
      return 0
    fi
  fi

  if [[ "${STATUS}" != "PASS" ]]; then
    return 1
  fi
}

# Default behavior: if neither --checkout nor --verify is explicitly specified, do checkout (which verifies at end)
if [[ "${DO_CHECKOUT}" -eq 0 && "${DO_VERIFY}" -eq 0 ]]; then
  DO_CHECKOUT=1
fi

if [[ "${DO_CHECKOUT}" -eq 1 ]]; then
  do_checkout
elif [[ "${DO_VERIFY}" -eq 1 ]]; then
  do_verify
fi
