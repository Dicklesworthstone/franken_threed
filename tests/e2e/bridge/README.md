# Independent WebGPU Bridge Counterexample Suite (`f3d-05.7`)

Executable adversarial browser tests against the real WebGPU bridge and host runtime (`f3d-05.6`), verified with independently issued direct-JavaScript WebGPU oracle reference operations.

## Purpose & Scope

Bead `f3d-05-ids-layouts-epochs-transport-vqa.7` implements the mandatory counterexamples required by plan §6.7, §8.2, §7.5, §6.5, §6.6, §6.9, §11.7, §14.8, §23 [S33], [S37], and root `AGENTS.md` ("Mandatory Counterexample Suite").

These tests guard against foundational WebGPU runtime hazards before any higher-level renderer or compiler abstraction is built:

## Single-Runner Suite Ownership Map (Agreed with NavyAspen in Mail #5694)

To ensure each test case is run by exactly one authoritative runner without duplicate execution or conflicting credit claims, ownership between `tests/fixtures/gpu_bridge/gpu_bridge_test.html` (ChartreuseFern, `f3d-05.6`) and `tests/e2e/bridge/index.html` (QuietSnow, `f3d-05.7`) is partitioned as follows:

| Runner / Page | Authoritative Cases Owned | Responsibility |
| --- | --- | --- |
| **`tests/fixtures/gpu_bridge/gpu_bridge_test.html`** (`f3d-05.6`) | • `negotiation_positive` & `negotiation_missing_feature`<br>• `error_scope_concurrency` & `error_scope_await_detection`<br>• `parser_truncated_fields_rejected` & `parser_invalid_format_rejected`<br>• `no_retained_renderer`<br>• `first_frame_pixel_identical` (smoke check) | **Bridge Host Unit & Boundary Verification**: Adapter negotiation limits, parser boundary decoding bounds, synchronous error scope serialization, and zero retained renderer assertions. |
| **`tests/e2e/bridge/index.html`** (`f3d-05.7`) | • `pixel_equivalence` (Dual Control: Rust AffineRows triangle vs Direct-JS Oracle, with real GPU broken transform negative rejection)<br>• `queue_snapshot_red_a_blue_b` (Dual Control: Rust PerUseByteBuffer dynamic offsets in single submit, with Rust unversioned & JS overwrite hazard rejections)<br>• `missing_wasm_rejection` (Runner fails if Wasm or required exports are missing)<br>• `stale_epoch_publication_gate` (Dual Control: matching epoch accepted vs advanced/stale epoch rejected via real Rust Wasm export)<br>• `bundle_then_direct_draw_state_reset` (Dual Control: Rust bundle packet vs Direct-JS Oracle, with real GPU omitted-rebind negative rejection)<br>• `generational_handle_aba_publication` (Dual Control: fresh generation check and advance in Rust slot table, with real GPU stale-generation publication rejection)<br>• `linear_memory_borrow_guards` (Dual Control: BorrowScope token validation, with real broken control rejecting growth inside active borrow)<br>• `affine_rows_gpu_layout_validation` (Dual Control: AffineRows wire validator codes 0, 1, 2, with GPU-side refusal before command submission)<br>• `pending_capabilities` (Zero Pending: 10/10 bridge capabilities backed by real Rust/Wasm exports) | **Integrated Adversarial Counterexamples**: Full dual-control tests (positive + isolated real GPU broken control rejected by the exact same assertion) against independent direct-JS WebGPU oracle reference operations. |

### Active Candidate Bridge Suites (Real GPU Execution)

| Suite | Ownership Status | Guards Against | Tested Invariant |
| --- | --- | --- | --- |
| **1. First-Frame Pixel Equivalence** | **Authoritative (05.7)** | Divergence from native WebGPU semantics | Candidate bridge executes Rust/Wasm binary packet (`gpu_bridge_build_triangle_packet()`) to render AffineRows triangle. Readback matches independent direct-JS WebGPU oracle (`renderDirectTriangleReference`) with 0 pixel diffs. Real GPU broken transform control (translation outside viewport) is demonstrably rejected by `assertExactPixelMatch`. |
| **2. Red-A / Blue-B Queue-Write Snapshot** | **Authoritative (05.7)** | Treating queue writes as per-draw snapshots | Two draws in a single command buffer submission sharing a uniform buffer use versioned slice offsets (0 and 256) via GrayFox's `PerUseByteBuffer` (`gpu_bridge_build_red_blue_packet(true)`). Target A is pure Red, Target B is pure Blue. Unversioned GPU queue hazard packet (`gpu_bridge_build_red_blue_packet(false)`) is demonstrably rejected by `assertRedABlueB`. |
| **3. Missing-Wasm Rejection Control** | **Authoritative (05.7)** | Silent fallback to JS packet generation | Test runner strictly asserts that positive acceptance suites reject execution if compiled application Wasm or required exports are missing. Silent JS fallback is forbidden. |
| **4. Malformed Packet Bounds Enforcement** | **Smoke Check (Authoritative in 05.6)** | Reading payload bytes as commands or defaulting invalid formats | Packet decoder validates every opcode field against command block boundary. Truncated commands, invalid texture format codes (99), and invalid render pass target types (99) are strictly rejected with structured decoder errors. Retained here for local harness integrity, authoritatively owned by `gpu_bridge_test.html`. |
| **5. Synchronous Error-Scope Discipline** | **Smoke Check (Authoritative in 05.6)** | Awaiting inside scopes or moving submit outside error scopes | Command encoding, canvas texture acquisition, `commandEncoder.finish()`, and `queue.submit()` execute synchronously inside error scopes before popping and awaiting errors. Interleaving and async yields inside scopes are strictly rejected. Retained here for local harness integrity, authoritatively owned by `gpu_bridge_test.html`. |
| **6. Stale-Epoch Readback Publication Gate** | **Authoritative (05.7)** | Accepting stale in-flight readbacks across epoch boundaries | Bridge returns 64-bit epoch stamp with readback buffer. Evaluated against active region epoch via compiled Rust/Wasm product export `gpu_bridge_try_publish_readback()`. Matching epoch (0, 1) is accepted (`true`); advanced region epoch (0, 2) and high-word mismatch (1, 1) are strictly rejected (`false`). |
| **7. Bundle-Then-Direct-Draw State Reset** | **Authoritative (05.7)** | Retaining render pass pipeline/bindgroup state across executeBundles | WebGPU spec mandates executeBundles clears pass state. Candidate bridge executes Rust/Wasm binary packet (`gpu_bridge_build_bundle_direct_draw_packet()`) rendering left green triangle via bundle and right blue triangle via direct draw with explicit rebind. Readback matches independent direct-JS WebGPU oracle (`renderDirectBundleDirectReference`) with 0 pixel diffs and verified sample points (16,32 Green, 48,32 Blue, 2,2 Black). Real GPU broken control (omitting post-bundle rebind) is demonstrably rejected by `assertBundleDirectDrawMatch`. |
| **8. Generational Handle ABA Publication Gate** | **Authoritative (05.7)** | Stale handle reuse (ABA) and zero generation across GPU resource lifecycles | Bridge slot table tracks typed NonZeroU32 generation counters (`gpu_bridge_check_resource_handle()`, `gpu_bridge_advance_resource_generation()`). Fresh generation (1) is accepted; advancing slot simulates release and reallocation to generation (2); older generation (1) is strictly rejected (`false`), generation (2) is accepted (`true`), and generation (0) is rejected (`false`). Real GPU broken publication attempt keyed by stale generation is demonstrably rejected by `assertGenerationalHandlePublication`. |
| **9. Linear Memory Borrow Guards** | **Authoritative (05.7)** | Linear memory growth or callback re-entry during active Rust borrows | Enforces that no WebAssembly memory growth or scheduler re-entry occurs while Rust linear memory borrows are held (`gpu_bridge_borrow_enter()`, `gpu_bridge_borrow_exit()`, `gpu_bridge_try_grow_memory()`). Idle memory growth succeeds; active borrow produces a non-zero `BorrowToken` during which growth attempts return `false`; invalid token exits return `false`; valid token exit succeeds and restores growth. Real broken control attempting growth inside an open borrow is demonstrably rejected by `assertMemoryGrowthAllowed`. |
| **10. AffineRows GPU Wire Layout Validation** | **Authoritative (05.7)** | Perspective components in affine layouts and corrupt wire payload lengths | Validates wire buffers against `AffineRows::from_column_major` and `AffineRows::from_bytes` (`gpu_bridge_validate_affine_rows()`). Real 64-byte column-major identity and translated matrices return code 0 (`LAYOUT_VALIDATION_OK`); 48-byte `AffineRows` wire record with nonzero translation returns code 0 (`LAYOUT_VALIDATION_OK`); 48-byte record with NaN translation returns code 2 (`NON_AFFINE_MATRIX`); non-affine 64-byte matrix with non-zero perspective term `e[11]` returns code 2 (`NON_AFFINE_MATRIX`); 47-byte and 56-byte buffers return code 1 (`BUFFER_TOO_SMALL`); 72-byte oversized buffer returns code 4 (`INCOMPATIBLE_TARGET`). GPU-side control uploading non-affine matrix through the validator gate refuses packet upload before any GPU command submission is encoded. |

### Pending Product Capabilities: ZERO PENDING (All Authoritatively Implemented)

Per OrangePelican root review and AGENTS credit rules, invented mock classes and simulated CPU byte edits are strictly forbidden.

**All 10 bridge capabilities are now fully backed by real Rust/Wasm product exports in `crates/f3d-runtime/src/gpu_host.rs`. The pending capability ledger is ZERO.** Zero mocks, zero synthetic JS fallbacks, zero unclassified items.

## Dual Verification: Real GPU Positive and Real GPU Broken Controls

Every active counterexample in this suite implements both:
1. **The Positive Implementation**: A valid path executing through the bridge that passes all assertions. Consumes the actual compiled application Wasm instance exports for binary packet generation.
2. **An Explicitly Isolated Real GPU Broken Control**: An actual faulty packet or operation submitted to the candidate bridge (e.g. out-of-bounds transform matrix on GPU, unversioned uniform queue hazard on GPU, truncated opcode buffer, or async yield inside error scope).

**Binding Honesty Rule**: The **EXACT SAME** readback assertion must reject the broken control. Mismatches are never hidden, softened, or relabeled as unsupported. Never regenerate oracle from candidate.

## File Map

- `README.md`: This architecture and invariant specification.
- `oracle_reference.js`: Independent direct-JS WebGPU reference implementations (completely decoupled from the candidate bridge).
- `counterexample_suite.js`: Standalone test suite executing positive paths and negative controls against the bridge.
- `index.html`: Self-contained browser harness page displaying rendering results, initializing Wasm, and reporting structured JSON to the test runner.
- `test_harness.mjs`: Node HTTP runner launching Chrome (`--headless=new`) or Safari, receiving test reports, and saving structured evidence to `evidence/05.7/`.

## Running the Suite

```bash
# Run on Chrome (headless with WebGPU enabled)
node tests/e2e/bridge/test_harness.mjs chrome

# Run on Safari (opens real Safari, receives report, closes tab)
node tests/e2e/bridge/test_harness.mjs safari

# Explicitly specifying compiled Wasm package directory:
node tests/e2e/bridge/test_harness.mjs out/browser-probe chrome
```

## No-Claim Boundary

Local Apple M4 GPU checks establish structural correctness, queue ordering semantics, and execution invariants. They do **not** establish M5, iPhone 17 Pro Max, NVIDIA, or AMD performance, and do not constitute full Three.js r186 compatibility.
