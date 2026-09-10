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
| **`tests/e2e/bridge/index.html`** (`f3d-05.7`) | • `pixel_equivalence` (Dual Control: Rust AffineRows triangle vs Direct-JS Oracle, with real GPU broken transform negative rejection)<br>• `queue_snapshot_red_a_blue_b` (Dual Control: Rust PerUseByteBuffer dynamic offsets in single submit, with Rust unversioned & JS overwrite hazard rejections)<br>• `missing_wasm_rejection` (Runner fails if Wasm or required exports are missing)<br>• `pending_capabilities` (4-item pending kill-gate audit ledger) | **Integrated Adversarial Counterexamples**: Full dual-control tests (positive + isolated real GPU broken control rejected by the exact same assertion) against independent direct-JS WebGPU oracle reference operations. |

### Active Candidate Bridge Suites (Real GPU Execution)

| Suite | Ownership Status | Guards Against | Tested Invariant |
| --- | --- | --- | --- |
| **1. First-Frame Pixel Equivalence** | **Authoritative (05.7)** | Divergence from native WebGPU semantics | Candidate bridge executes Rust/Wasm binary packet (`gpu_bridge_build_triangle_packet()`) to render AffineRows triangle. Readback matches independent direct-JS WebGPU oracle (`renderDirectTriangleReference`) with 0 pixel diffs. Real GPU broken transform control (translation outside viewport) is demonstrably rejected by `assertExactPixelMatch`. |
| **2. Red-A / Blue-B Queue-Write Snapshot** | **Authoritative (05.7)** | Treating queue writes as per-draw snapshots | Two draws in a single command buffer submission sharing a uniform buffer use versioned slice offsets (0 and 256) via GrayFox's `PerUseByteBuffer` (`gpu_bridge_build_red_blue_packet(true)`). Target A is pure Red, Target B is pure Blue. Unversioned GPU queue hazard packet (`gpu_bridge_build_red_blue_packet(false)`) is demonstrably rejected by `assertRedABlueB`. |
| **3. Missing-Wasm Rejection Control** | **Authoritative (05.7)** | Silent fallback to JS packet generation | Test runner strictly asserts that positive acceptance suites reject execution if compiled application Wasm or required exports are missing. Silent JS fallback is forbidden. |
| **4. Malformed Packet Bounds Enforcement** | **Smoke Check (Authoritative in 05.6)** | Reading payload bytes as commands or defaulting invalid formats | Packet decoder validates every opcode field against command block boundary. Truncated commands, invalid texture format codes (99), and invalid render pass target types (99) are strictly rejected with structured decoder errors. Retained here for local harness integrity, authoritatively owned by `gpu_bridge_test.html`. |
| **5. Synchronous Error-Scope Discipline** | **Smoke Check (Authoritative in 05.6)** | Awaiting inside scopes or moving submit outside error scopes | Command encoding, canvas texture acquisition, `commandEncoder.finish()`, and `queue.submit()` execute synchronously inside error scopes before popping and awaiting errors. Interleaving and async yields inside scopes are strictly rejected. Retained here for local harness integrity, authoritatively owned by `gpu_bridge_test.html`. |

### Pending Product Capabilities (Strictly Excluded from Acceptance Claims)

Per OrangePelican root review and AGENTS credit rules, invented mock classes and simulated CPU byte edits are forbidden. The following capabilities are pending product support in the candidate bridge:
- **Bundle-Then-Direct State Reset**: `bridge_runtime.js` does not yet implement `OPCODE_EXECUTE_BUNDLES`.
- **Generational Handle ABA Publication**: `gpu_host.rs` does not yet expose generational handle publication tables to JS.
- **Linear Memory Borrow Guards**: Browser Wasm linear memory borrow guard ABI is pending runtime integration.
- **Standalone Wire Validator**: Wire layout validator pending integration into `GpuSubmissionPacket` decoder.
- **Stale-Epoch Readback Publication Gate**: `gpu_host.rs` publication gate is currently native-only; browser Wasm export `gpu_bridge_try_publish_readback` is pending product support. No test-only JS mock permitted under root 5526.

These items are reported as `PENDING_PRODUCT_SUPPORT` and are never counted as green passes until real candidate product implementations exist.

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
