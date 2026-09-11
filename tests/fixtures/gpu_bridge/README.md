# Production Bulk WebGPU Bridge Fixture (`f3d-05.6`)

This fixture verifies the first real Rust/Wasm to WebGPU execution slice and rendered frame for FrankenThreeD.

## Verified Invariants & Capabilities

1. **Pre-Device Negotiation**:
   - Explicit required features and limits profile validated against `GPUAdapter` capabilities before `requestDevice`.
   - Missing required features (e.g. unknown feature names) fail immediately with structured errors without creating a device.
   - Capability record captures adapter vendor, architecture, fallback status, enabled features, and limits.

2. **Synchronous Error-Scope Stack Discipline**:
   - `withErrorScopes(kinds, syncAction)` pushes error scopes (`'validation'`, `'out-of-memory'`) synchronously.
   - Operations execute synchronously inside the error-scope stack without intervening `await` or microtask turn boundaries.
   - Asynchronous bodies returning a Promise inside the error-scope block are detected and rejected.
   - Concurrent tasks attempting to interleave error scopes on the same device are detected and rejected.
   - Error scope results are popped synchronously and their resolution promises awaited afterwards.

3. **First Correct Frame & Oracle Reference Equivalence**:
   - Procedural WGSL vertex shader transforms positions with `AffineRows` (48 bytes: 3 rows of `vec4<f32>` with translation in the 4th component).
   - Renders to an offscreen target texture and to the canvas.
   - Offscreen readback pixels match an independently issued direct JavaScript WebGPU reference pixel-for-pixel (0 differences).

4. **Queue Ordering & Snapshot Semantics (Red-A / Blue-B)**:
   - Two draws in the same queue submission using a shared uniform buffer with per-use versioned slices (offset 0 for Red, offset 256 for Blue).
   - Target A renders Red, Target B renders Blue.
   - Mutation check verifies that removing versioned offsets causes both draws to read Blue, proving the test detects the queue hazard.

5. **Fresh Canvas Texture Acquisition**:
   - Canvas swapchain texture (`context.getCurrentTexture().createView()`) is acquired fresh per rendering interval and never cached across frames.

6. **No Retained Renderer Submission**:
   - Confirms that zero Three.js / WebGL fallback renderer calls occurred; execution is driven exclusively through the new WebGPU bridge.

7. **Safe Copied Host Transport (`vqa.4`, `memory_transport.js`)**:
   - **Production Owned Copy-Out**: The shipping execution path relies on wasm-bindgen's native `Vec<u8>` return ABI (`getArrayU8FromWasm0(ptr, len).slice()`), which synchronously creates an owned `Uint8Array` copy decoupled from Wasm linear memory before freeing the Rust allocation.
   - **Packet Boundary Validation (`ensureSafePacketBytes`)**: Validates that incoming packet bytes are a valid, non-detached `Uint8Array` prior to asynchronous WebGPU command execution without making false claims of producing an owned copy in-place.
   - **Detachment Detection (`isDetached`)**: Immediately detects and rejects detached or zero-length `ArrayBuffer` instances resulting from Wasm linear memory growth (`memory.grow()`).
   - **Copy Accounting (`TransportCopyLedger`) & Bounded Helper (`copyFromWasmMemory`)**: Validates finite, non-negative integer ranges for Wasm memory slices and tracks actual transport copies without synthetic inference.

## No-Claim Boundary

- **Unbenchmarked Slice**: This is an unbenchmarked first bridge slice establishing structural correctness and execution invariants, not a production-selected fastest variant, full renderer, or passed M5/iPhone hardware gate.
- **Readback Publication Gate**: The readback publication gate (`ReadbackPublicationState` / `try_publish_readback` in `gpu_host.rs`) is exported to browser callers as `gpu_bridge_try_publish_readback(readback_epoch_hi, readback_epoch_lo, region_epoch_hi, region_epoch_lo) -> bool`. The browser bridge (`bridge_runtime.js`) decodes and stamps `epochHi` and `epochLo` onto resolved readback buffers in JavaScript, and passes them directly to the compiled Wasm product function (which reconstructs `Epoch` instances via `Epoch::from_words`), ensuring zero JavaScript reimplementation of epoch freshness validation.
