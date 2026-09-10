# Upstream Oracle Pin: Three.js r186

This document establishes the immutable upstream oracle definition for FrankenThreeD.

---

## 1. Upstream Pin Metadata

| Field | Value |
|---|---|
| **Upstream Release** | **Three.js r186** |
| **Release Date** | September 8, 2026 (`2026-09-08`) |
| **Source Commit Hash** | `148ef33ecb6d2502ff796d4554abd1549c95d519` |
| **Tag Object Hash** | `819fadd6b663b74d828c6af72a543024f74d3877` |
| **Tag Name** | `r186` |
| **Package Version** | `0.186.0` |
| **Upstream Repository** | `https://github.com/mrdoob/three.js.git` |
| **Inspection Date** | September 9, 2026 (`2026-09-09`) |
| **Local Oracle Target** | `upstream/three.js` |

---

## 2. Hash Distinction: Source Commit vs. Tag Object

> **CRITICAL HONESTY RULE (AGENTS.md & Plan §2.2, §5.17)**:
> The annotated tag object hash (`819fadd6b663b74d828c6af72a543024f74d3877`) is **not** the source commit.
> The source commit is `148ef33ecb6d2502ff796d4554abd1549c95d519`.
> Never conflate the two hashes.

In Git, an annotated tag is an independent tag object in the object database:
- `git rev-parse refs/tags/r186` produces the **annotated tag object** `819fadd6b663b74d828c6af72a543024f74d3877`.
- `git rev-parse refs/tags/r186^{commit}` (or inspecting the tag target) produces the **commit object** `148ef33ecb6d2502ff796d4554abd1549c95d519`.
- The checked-out `HEAD` in `upstream/three.js` must match the source commit `148ef33ecb6d2502ff796d4554abd1549c95d519` exactly.

---

## 3. Immutability and Non-Editing Rules

1. **Read-only Oracle**: The oracle checkout in `upstream/three.js` is an immutable reference oracle. It is **never edited** in place.
2. **No Vendoring in Main Tree**: The upstream checkout is cloned/fetched on demand via `scripts/oracle-checkout.sh`. The directory `upstream/three.js` is gitignored so vendor trees are not accidentally checked into FrankenThreeD.
3. **Reproducibility**: Any fresh clone or clean machine must be able to run `scripts/oracle-checkout.sh --verify` to reproduce the exact checkout and verify HEAD matches `148ef33ecb6d2502ff796d4554abd1549c95d519`.
4. **Isolated Retained Components**: When retained upstream components are needed in later phases, they are version-pinned and isolated with attribution, never ad hoc mutations of the oracle checkout.

---

## 4. Built Outputs and Reconciliation Contract

The upstream repository builds the following core bundles via `npm run build`:
- `build/three.module.js` — ESM core bundle
- `build/three.webgpu.js` — WebGPU renderer bundle
- `build/three.tsl.js` — Three.js Shading Language / Node system bundle
- `build/three.cjs` — CommonJS core bundle

Reconciliation against `package.json` (`exports`, `files`, subpath wildcards, and packaged assets) is performed by `tools/upstream/reconcile_package.mjs`. All discrepancies between metadata and actual filesystem entries are classified as explained (e.g. valid backward-compatible aliases) or unresolved.

---

## 5. E2E Runner Inputs (Bead 01.4)

Facts extracted from upstream Three.js r186 (`148ef33ecb6d2502ff796d4554abd1549c95d519`):

- **Puppeteer**: `^25.0.0` (declared in `package.json` `devDependencies`; resolves to `25.10.0` in upstream `package-lock.json`).
- **Chromium**: Managed directly by Puppeteer `25.10.0`; no separate `chromium` package is declared in `devDependencies`.
- **Playwright**: Not declared in upstream `devDependencies`.
- **Node Engine**:
  - Upstream `package.json` root: unpinned (`engines` field is omitted).
  - Upstream `package-lock.json` (`node_modules/puppeteer`): specifies `"node": ">=22.12.0"`.
  - FrankenThreeD toolchain environment (`tools/package.json`): `"node": ">=20.18.0"`.

