# CONTRACT.md: `f3d-math`

> Crate contract for `f3d-math` per AGENTS.md "Documentation and Contracts".

---

## 1. Purpose and Position in the Dependency Direction

`f3d-math` provides fixed-size mathematical primitives (`Vector3`, `Quaternion`, `Matrix4`) with `f64` public semantics matching Three.js r186 (source commit `148ef33ecb6d2502ff796d4554abd1549c95d519`).

Position in architecture:
```text
core <- math
core/math <- scene
```

- `f3d-math` depends only on `f3d-core` (for GPU wire layout types such as `AffineRows` and `ProjectiveMat4`).
- It does **not** depend on browser bindings (`wasm-bindgen`, `web-sys`), async executors, or rendering crates.
- It is strictly `#![no_std]` capable with `alloc`.

---

## 2. Public API Surface & Semantics

| Type | Semantics |
|---|---|
| `Vector3` | 3D vector `(x: f64, y: f64, z: f64)`. Implements `apply_matrix4` (with perspective divide matching Three.js r186), `apply_quaternion`, arithmetic, dot, length, and normalize (preserving signed zero and non-NaN components via `length || 1`). |
| `Quaternion` | 4D quaternion `(x: f64, y: f64, z: f64, w: f64)`. Implements Hamiltonian multiplication, conjugate, invert (via conjugate for unit quaternions), dot, length, and normalize (resets to identity on zero length). Non-unit authored quaternions are preserved and never normalized prematurely. |
| `Matrix4` | 4x4 matrix stored in column-major order `elements: [f64; 16]`. Implements `multiply_matrices`, `multiply`, `premultiply`, `compose(pos, quat, scale)`, `decompose`, `transpose`, `determinant`, `determinant_affine`, `invert`, `make_perspective`, and `make_orthographic`. |
| `CoordinateSystem` | Target clip space enum: `CoordinateSystem::WebGL` (depth range [-1, 1], value 2000) and `CoordinateSystem::WebGPU` (depth range [0, 1], value 2001). |
| `NarrowingTolerance` | Configurable tolerance parameters: `max_abs_err: f64` and `max_rel_err: f64`, with constants `EXACT`, `DEFAULT_FLOAT`, `PERMISSIVE`. |
| `NarrowingError` | Error enum with variants `NonFinite { index, value }`, `PrecisionLossExceeded { index, original, narrowed, abs_diff, rel_diff, max_abs_err, max_rel_err }`, and `NonAffineMatrix`. |
| Seam conversions | `Matrix4::to_affine_rows`, `to_affine_rows_checked`, `to_affine_rows_strict`, `to_projective_mat4`, `to_projective_mat4_checked`, and `to_projective_mat4_strict`, bridging CPU simulation math to the verified GPU layouts in `f3d-core` with explicit precision policy enforcement. |

---

## 3. Strict Invariants

1. **Operation Ordering and Formula Exactness**: Arithmetic follows the exact operations of Three.js r186 math modules:
   - `Matrix4.multiply_matrices` evaluates 16 dot products in exact column-major index order.
   - `Matrix4.compose(position, quaternion, scale)` evaluates Shoemake's quaternion matrix expansion without pre-normalizing the quaternion and with direct column scaling (`sx, sy, sz`).
   - `Vector3.apply_matrix4` evaluates the perspective denominator `w = 1.0 / (e[3]*x + e[7]*y + e[11]*z + e[15])` and multiplies by `w`.
   - `Vector3.normalize` implements `divideScalar(length || 1)` (Three.js `Vector3.js:794`), preserving signed zeros (`-0.0`) on zero vectors and preserving non-NaN components when one component is `NaN`.
   - `Matrix4.make_perspective` and `Matrix4.make_orthographic` preserve exact Three.js r186 formula ordering, supporting WebGL ([-1, 1]), WebGPU ([0, 1]), and reversedDepth ([1, 0]) depth intervals.
2. **Singular Matrix Behavior**: Calling `Matrix4::invert()` on a singular matrix (`determinant == 0.0`) produces an all-zero matrix (`[0.0; 16]`), matching the exact behavior of Three.js r186 `Matrix4.invert()`.
3. **Non-Unit Authored Quaternions**: `compose` does NOT normalize input quaternions; unnormalized authored rotations evaluate directly as written in upstream Three.js.
4. **Negative Scale & Shear**: Negative determinants, reflections, and non-orthogonal shear transformations are preserved accurately through multiplication, inversion, and point transformation.
5. **Exact Structural Affine Guard Before Narrowing**: `Matrix4::to_affine_rows` checks exact structural affine requirements (`e[3] == 0.0 && e[7] == 0.0 && e[11] == 0.0 && e[15] == 1.0`) at full `f64` precision before narrowing to `f32`. Any non-zero perspective term (including subnormals e.g. `1e-100`) is rejected with `LayoutError::NonAffineMatrix` to retain `ProjectiveMat4`.
6. **Signed Zero & IEEE 754**: Calculations use standard IEEE 754 `f64` double precision, preserving signed zero (`-0.0`) and exceptional values (`Infinity`, `NaN`) without artificial clamping.
7. **No Wire Layout Duplication**: GPU wire layouts (`AffineRows`, `ProjectiveMat4`) reside exclusively in `f3d-core::layout`; `f3d-math` implements conversion methods against those canonical types.
8. **Projection Matrices & Wire Layouts**: Perspective matrices (`e[11] = -1.0, e[15] = 0.0`) are strictly rejected by `to_affine_rows` and retained byte-for-byte by `to_projective_mat4`. Orthographic matrices (`e[11] = 0.0, e[15] = 1.0`) are affine and convert to `AffineRows`.
9. **Explicit f64 to f32 Wire Narrowing Policy**: Checked narrowing evaluates absolute error `|val - (val as f32 as f64)|` against `max_abs_err` and relative error against `max_rel_err`. A strict variant rejects non-finite values (`NaN`, `+Infinity`, `-Infinity`). Checked conversions `to_affine_rows_checked` and `to_projective_mat4_checked` reject precision loss exceeding tolerance, with affine shape guarded strictly at `f64` precision.

---

## 4. Implementation Ownership & Acceleration Status

- **Ownership route**: New Rust math core.
- **Acceleration**: Scalar-first implementation. SIMD (`simd128`) batch kernels will be introduced only under feature flags after measured crossover benchmarks on target browsers. No acceleration or speedup is claimed for this scalar slice.

---

## 5. Error Model

- Geometric operations that are total (e.g. `invert()` on singular returning all-zero matrix) match upstream Three.js without panicking.
- Safe fallible alternatives (`try_invert() -> Option<Matrix4>`) return `None` for singular matrices.
- Layout conversion `to_affine_rows()` returns `Result<AffineRows, LayoutError>` where `LayoutError::NonAffineMatrix` is returned if perspective elements are non-zero.
- Checked narrowing conversions `to_affine_rows_checked()` and `to_projective_mat4_checked()` return `Result<_, NarrowingError>`:
  - `NarrowingError::NonAffineMatrix`: Non-affine matrix supplied to affine layout.
  - `NarrowingError::PrecisionLossExceeded`: Element precision loss exceeds absolute and relative bounds.
  - `NarrowingError::NonFinite`: Non-finite value (`NaN`, `Infinity`) rejected by strict narrowing policy.


---

## 6. Determinism & Numeric-Fidelity Class

- Class: `Deterministic-Scalar-f64`.
- All operations are pure scalar arithmetic on fixed `f64` values.
- No hardware-dependent transcendental approximations are used in this transform slice.

---

## 7. Async / Cancellation Semantics

- All operations are purely synchronous, non-allocating, and bounded.
- No async runtime or cooperative yielding is required.

---

## 8. Unsafe Boundary Policy

- Declares **`#![forbid(unsafe_code)]`** at crate root.
- No unsafe blocks, pointer casts, or undefined memory layouts exist in this crate.

---

## 9. Feature Flags

- `default = ["std"]`
- `std`: Standard library support.
- `serde`: Enables serialization and deserialization for `Vector3`, `Quaternion`, and `Matrix4`.

---

## 10. Conformance & Verification Plan

- Independent analytical reference tests in `tests/transform_tests.rs` covering:
  - Positive: Identity, TRS composition, hierarchy parent/child multiplication, vector transformation, and invertible matrices.
  - Decompose reference cases: TRS roundtrip, negative determinant flipping `sx`, and det==0 singular fallback resetting to unit scale and identity quaternion matching Three.js r186 `Matrix4.js:1081-1088`.
  - Projection reference cases: Perspective and orthographic projections verifying NDC z near/far mapping across WebGL ([-1, 1]), WebGPU ([0, 1]), and reversed depth ([1, 0]).
  - Negative: Singular matrix (det == 0 produces all zeros), negative determinant reflection, shear, non-unit authored quaternion (catching premature normalization), perspective divide, and subnormal/non-affine matrix rejection.
  - Wire conversion seam roundtrip with `f3d-core::layout::AffineRows` and `ProjectiveMat4`.
  - Cross-crate projection agreement: `f3d_core::layout::ProjectiveMat4::transform_homogeneous` (divided by w) matches `Vector3::apply_matrix4` within 1e-6 (f32 narrowing).

---

## 11. Explicit No-Claims

- This first slice does **NOT** claim full Three.js math compatibility (Euler angles, Matrix3, Box3, Sphere, Ray, Plane, and transcendental functions are deferred to subsequent slices).
- It does **NOT** claim SIMD acceleration, batch processing crossover, packed scene execution, or GPU performance gains.
