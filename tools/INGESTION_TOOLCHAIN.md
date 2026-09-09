# JavaScript Ingestion Toolchain Admission & Pinning Record

> **Bead:** `f3d-01-upstream-pin-and-census-gl8.2` (Toolchain/workspace bootstrap)  
> **Component:** Build-Time JavaScript Ingestion & AST Parser Toolchain  
> **Status:** Verified via locked install (`npm ci`: 6 packages added, 7 audited, 0 vulnerabilities on 2026-09-09)  
> **Authority:** Plan §4.1, §13.4; AGENTS.md "Dependency Policy"

---

## 1. Architectural Purpose & Scope Boundary

Per **Section 4.1** of `COMPREHENSIVE_PLAN_FOR_THE_DESIGN_OF_FRANKENTHREED.md`:
> *"The compiler understands Three.js-related semantics and a restricted, useful set of numeric operations. It does not attempt to reimplement all JavaScript language behavior in Rust. Use the existing project's build process to normalize TypeScript, JSX, and module syntax when present. A narrow build-time JavaScript parser/helper supplies a structured module representation to the Rust compiler. Initial preference: the existing Rollup ecosystem plus an Acorn-class parser, pinned and audited during Phase 0."*

### Strict Scope Boundaries
1. **Build-Time Only:** This package (`tools/package.json`) exists exclusively on the host build runner to ingest Three.js applications, parse ESM/CommonJS module graphs, and extract ASTs for the `f3d-compiler`.
2. **Never in Deployed Wasm:** Zero lines of JavaScript, Node dependencies, or npm modules from `tools/` enter the shipping Wasm runtime or client-side application bundle.
3. **No General Language Frontend:** This toolchain does not build a general JS-to-Rust compiler; it provides AST analysis and module graph closure for Three.js constructs.

---

## 2. Pinned Package Register & Admission Records

All packages are pinned to exact versions with lockfileVersion 3 integrity hashes in `tools/package-lock.json`.

| Package | Pinned Version | License | Category | Integrity (sha512) | Security Audit |
|---|---|---|---|---|---|
| `acorn` | `8.14.0` | MIT | Ingestion AST Parser | `cl669nCJTZBsL97OF4kUQm5g5hC2uihk0NxY3WENAC0TYdILVkAyHymAntgxGkl7K+t0cXIrH5siy5S4XkFycA==` | 0 vulnerabilities (`npm audit` observed 2026-09-09) |
| `acorn-walk` | `8.3.4` | MIT | AST Walker Companion | `ueEepnujpqee2o5aIYnvHU6C0A42MNdsIDeqy5BydrkuC5R1ZuUFnm27EeFJGoEHJQgn3uleRvmTXaJgfXbt4g==` | 0 vulnerabilities (`npm audit` observed 2026-09-09) |
| `rollup` | `4.63.1` | MIT | Module Bundler / Closure | `3Df9jsstwhccuEfmAMi9l8XUh/GOkVObmFTU7CCVBysEbcOZLl84jCtaAZMcPiMz2EGKsATzQcU+Xr3n/wU6cg==` | 0 vulnerabilities (`npm audit` observed 2026-09-09; GHSA-mw96-cpmx-2vgc patched) |
| `@types/estree` | `1.0.9` | MIT | ESTree AST Type Definitions | `GhdPgy1el4/ImP05X05Uw4cw2/M93BCUmnEvWZNStlCzEKME4Fkk+YpoA5OiHNQmoS7Cafb8Xa3Pya8m1Qrzeg==` | 0 vulnerabilities (`npm audit` observed 2026-09-09) |

---

## 3. Node.js & Package Manager Inputs and Reconciliation

- **Actual Host Runtime Inputs:**
  - Installed Node.js: `v25.9.0`
  - Installed npm: `11.12.1`
- **Supported Engine Range (declared in `package.json`):**
  - `node`: `>=20.18.0` (Active LTS baseline)
  - `npm`: `>=10.0.0`
- **Reconciliation with Workspace Documentation:**
  - While earlier drafts in `DEPENDENCY_ADMISSION.md` cited `Node.js v24 LTS`, the host runtime is currently `v25.9.0`.
  - The tool package engine constraint is set to `node >=20.18.0` in `tools/package.json`, which soundly admits Node 20, Node 22, Node 24 LTS, as well as the active host Node `v25.9.0`.

---

## 4. Verification Seams & Execution Evidence

1. **Locked Package Installation (`npm ci`):**
   ```bash
   cd tools && npm ci
   ```
   *Execution output (2026-09-09):*
   ```text
   added 6 packages, and audited 7 packages in 1s
   found 0 vulnerabilities
   ```

2. **Package Tree Verification:**
   ```bash
   npm ls --prefix tools
   ```
   *Execution output (2026-09-09):*
   ```text
   franken-threed-tools@0.1.0 /Users/jemanuel/projects/franken_threed/tools
   ├── acorn-walk@8.3.4
   ├── acorn@8.14.0
   └── rollup@4.63.1
   ```

3. **Toolchain Pin & AST Specification Test:**
   ```bash
   node tools/test_toolchain_pin.mjs
   ```
   *Execution output (2026-09-09):*
   ```text
   [PASS] tools/package.json configuration and dependency pins
   [PASS] Node runtime satisfies engine requirements
   [PASS] tools/package-lock.json lockfileVersion 3 and sha512 checksum integrity
   [PASS] AST grammar coverage contract specifies Three.js r186 AST requirements

   SUCCESS: All JS ingestion toolchain pin tests passed.
   ```
