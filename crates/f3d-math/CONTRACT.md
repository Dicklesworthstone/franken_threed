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
- It is strictly `#![no_std]` capable with zero heap allocations (no `alloc` crate requirement).

---

## 2. Public API Surface & Semantics

| Type | Semantics |
|---|---|
| `Vector2` | 2D vector `(x: f64, y: f64)`. Implements constructor/default `(0, 0)`, `set` / `copy`, `add` / `sub` / `multiply` / `divide` / scalars, `dot`, `cross` (2D scalar cross product), `length`, `length_sq`, `manhattan_length`, `normalize` (preserving signed zero and non-NaN components via `length || 1`), `distance_to`, `distance_to_squared`, `manhattan_distance_to`, `set_length`, `lerp`, `lerp_vectors`, `rotate_around`, `min` / `max` / `clamp` / `clamp_scalar` / `clamp_length`, `floor` / `ceil` / `round` / `round_to_zero`, `angle` / `angle_to` (with exact signed-zero atan2 semantics), `apply_matrix3`, equality, array conversions, and checked f32 narrowing matching Three.js r186 `Vector2.js`. |
| `Vector3` | 3D vector `(x: f64, y: f64, z: f64)`. Implements `apply_matrix4` (with perspective divide matching Three.js r186), `apply_quaternion`, `project` / `unproject`, `cross`, `lerp`, `angle_to`, `distance_to`, `distance_to_squared`, `manhattan_distance_to`, `set_from_matrix_column`, `set_from_matrix3_column`, `set_from_matrix_position`, `set_from_matrix_scale`, arithmetic, dot, length, and normalize (preserving signed zero and non-NaN components via `length || 1`). |
| `Quaternion` | 4D quaternion `(x: f64, y: f64, z: f64, w: f64)`. Implements Hamiltonian multiplication, conjugate, invert (via conjugate for unit quaternions), dot, length, and normalize (resets to identity on zero length). Non-unit authored quaternions are preserved and never normalized prematurely. |
| `Matrix4` | 4x4 matrix stored in column-major order `elements: [f64; 16]`. Implements `multiply_matrices`, `multiply`, `premultiply`, `compose(pos, quat, scale)`, `batch_compose(positions, quaternions, scales, outputs)`, `make_rotation_from_euler`, `make_rotation_from_quaternion`, `get_max_scale_on_axis` (with `js_max` NaN propagation), `decompose`, `transpose`, `determinant`, `determinant_affine`, `invert`, `make_perspective`, and `make_orthographic`. |
| `Affine3x4` | Packed 3x4 affine transform matrix in `f64` precision (`r0`, `r1`, `r2`: `[f64; 4]`), matching GPU `AffineRows` layout `[e0, e4, e8, e12]`, `[e1, e5, e9, e13]`, `[e2, e6, e10, e14]`. Implements checked/unchecked conversions from `Matrix4` and `[f64; 16]` (strictly rejecting non-affine matrices with `LayoutError::NonAffineMatrix`, including an intentional bitwise packing refusal of `-0.0` in the implicit row to avoid sign loss on round-trip expansion), checked composition API (`multiply`, `premultiply`, `multiply_affines`, `checked_multiply`) returning `Result<_, LayoutError>` by evaluating the actual full Matrix4 product bottom row and refusing non-affine products (e.g. when RHS has non-finite components making product row 3 `NaN`) before mutating the receiver, point and vector transformations (`transform_point` preserving homogeneous denominator `w = 1 / (0.0*x + 0.0*y + 0.0*z + 1.0)`, `transform_vector`), affine determinant, and explicit conversion to GPU wire layout `AffineRows`. |
| `BatchComposeError` | Error enum with variant `LengthMismatch { positions_len, quaternions_len, scales_len, outputs_len }` returned when batch compose input/output slices have mismatched lengths. |
| `CoordinateSystem` | Target clip space enum: `CoordinateSystem::WebGL` (depth range [-1, 1], value 2000) and `CoordinateSystem::WebGPU` (depth range [0, 1], value 2001). |
| `NarrowingTolerance` | Configurable tolerance parameters: `max_abs_err: f64` and `max_rel_err: f64`, with constants `EXACT`, `DEFAULT_FLOAT`, `PERMISSIVE`. |
| `NarrowingError` | Error enum with variants `NonFinite { index, value }`, `PrecisionLossExceeded { index, original, narrowed, abs_diff, rel_diff, max_abs_err, max_rel_err }`, and `NonAffineMatrix`. |
| `Euler` / `EulerOrder` | 3D Euler angles `(x: f64, y: f64, z: f64, order: EulerOrder)` across 6 rotation orders (`XYZ`, `YXZ`, `ZXY`, `ZYX`, `YZX`, `XZY`). Implements `set_from_rotation_matrix` with `0.9999999` gimbal threshold, `set_from_quaternion`, `to_quaternion`, and `reorder`. |
| `Matrix3` | 3x3 matrix in column-major order `elements: [f64; 9]`. Implements `set_from_matrix4`, `invert` (producing all-zero matrix on `det == 0.0`), `transpose`, and `get_normal_matrix(m4)` matching Three.js r186. |
| `Box3` | 3D axis-aligned bounding box (AABB) represented by `(min: Vector3, max: Vector3)`. Implements `set`, `set_from_points`, `set_from_buffer` (flat slice of f64 coordinates), `expand_by_point`, `expand_by_vector`, `expand_by_scalar`, `contains_point`, `contains_box`, `intersects_box`, `intersects_sphere`, `intersects_plane`, `intersects_triangle`, `clamp_point`, `distance_to_point`, `get_center`, `get_size`, `union`, `intersect`, `apply_matrix4` (8-corner method), `translate`, and `is_empty` with Three.js r186 empty conventions (`min = +inf, max = -inf`). |
| `Sphere` | 3D analytical bounding sphere `(center: Vector3, radius: f64)`. Implements `set`, `set_from_points` (with optional center fallback to AABB center), `contains_point`, `distance_to_point`, `intersects_sphere`, `intersects_box`, `intersects_plane`, `clamp_point`, `get_bounding_box`, `apply_matrix4` (`center.apply_matrix4(m)` and `radius *= m.get_max_scale_on_axis()`), `translate`, `expand_by_point`, `union`, and `is_empty` (`radius < 0.0`) matching Three.js r186. |
| `Line3` | 3D line segment `(start: Vector3, end: Vector3)`. Implements `set`, `get_center`, `delta`, `distance_sq`, `distance`, `at(t)`, `closest_point_to_point_parameter`, `closest_point_to_point`, `apply_matrix4`, and `equals`. |
| `Plane` | 2D infinite plane in Hessian normal form `(normal: Vector3, constant: f64)`. Implements `set_from_normal_and_coplanar_point`, `set_from_coplanar_points`, `distance_to_point`, `distance_to_sphere`, `project_point`, `coplanar_point`, `intersect_line`, `intersects_line`, `intersects_box`, `intersects_sphere`, `apply_matrix4` (with optional pre-computed normal matrix), `normalize`, `negate`, `translate`, and `equals` matching Three.js r186. |
| `Frustum` | 3D viewing frustum enclosed by six boundary planes `planes: [Plane; 6]` (Right, Left, Bottom, Top, Far, Near). Implements `set(p0..p5)`, `copy(&source)`, `set_from_projection_matrix(m, coordinate_system, reversed_depth)` matching Three.js r186 formulas across WebGL ([-1, 1]) and WebGPU ([0, 1]) clip intervals and reversed depth, `contains_point(point)`, `intersects_sphere(sphere)` (evaluating `distance < -radius` per plane), and `intersects_box(box3)` (corner test along plane normals) matching Three.js r186 `Frustum.js`. |
| `Ray` | 3D ray `(origin: Vector3, direction: Vector3)`. Implements `at(t)`, `look_at`, `recast`, `closest_point_to_point`, `distance_sq_to_point`, `distance_to_point`, `intersect_sphere`, `intersects_sphere`, `distance_to_plane`, `intersect_plane`, `intersects_plane`, `intersect_box` (slab method with NaN handling), `intersect_triangle` (watertight signed edge method with `backface_culling`), `apply_matrix4`, and `equals`. |
| `Triangle` | 3D geometric triangle `(a: Vector3, b: Vector3, c: Vector3)`. Implements `get_normal` (returning `(0, 0, 0)` for degenerate), `get_barycoord` (returning `None` for degenerate), `contains_point`, `closest_point_to_point` (Christer Ericson Voronoi region-walk method), `is_front_facing`, `get_area`, `get_midpoint`, `get_plane`, `intersects_box`, and `equals`. |
| `Color` | Three-component color `(r: f64, g: f64, b: f64)` stored in linear working color space ([`ColorSpace::LinearSRGB`]). Implements `set`, `set_scalar`, `set_rgb`, `set_hex`, `get_hex` (using `crate::jsnum::js_round`), `format_hex_string`, `write_hex_string`, `get_hex_string`, `set_hsl`, `get_hsl`, `offset_hsl`, `set_style`, `format_style`, `write_style`, `get_style`, `set_color_name`, `lerp`, `lerp_colors`, `lerp_hsl`, `convert_srgb_to_linear`, `convert_linear_to_srgb`, `add`, `sub` (clamped at zero), `multiply`, `apply_matrix3`, array/vector conversions matching Three.js r186 `Color.js`. |
| `StyleOutcome` | Outcome enum of `set_style` and `set_color_name`: `Applied` (success), `AlphaIgnored` (alpha < 1.0 ignored with upstream warning semantics), and `IgnoredInvalid` (malformed input leaves color unchanged). |
| `COLOR_NAMES` | Static table of 148 standard CSS color keyword tuples `[(&'static str, u32); 148]` matching Three.js r186 `_colorKeywords` (140 distinct colors plus 8 aliases). |
| `ColorSpace` | Supported color space enum matching Three.js r186: `LinearSRGB` (working color space default), `SRGB`, `DisplayP3`, `LinearDisplayP3`. |
| `Hsl` | HSL color structure `(h: f64, s: f64, l: f64)` normalized in `[0.0, 1.0]`. |
| Transfer & Matrices | Piecewise transfer functions `srgb_to_linear` and `linear_to_srgb` with exact `0.04045` and `0.0031308` thresholds; canonical upstream 3x3 matrices `LINEAR_REC709_TO_XYZ`, `XYZ_TO_LINEAR_REC709`, `LINEAR_DISPLAY_P3_TO_XYZ`, `XYZ_TO_LINEAR_DISPLAY_P3`, `LINEAR_SRGB_TO_LINEAR_DISPLAY_P3`, `LINEAR_DISPLAY_P3_TO_LINEAR_SRGB`. |
| `jsnum` lowering | Faithful ECMAScript numeric semantics: `to_int32`, `to_uint32`, `js_rem`, `js_shift_left`, `js_shift_right`, `js_shift_unsigned_right`, `js_min`, `js_max`, `js_round`, `js_trunc`, `js_sign`, exactly adhering to ECMA-262 and avoiding saturating Rust `as`-cast distortions. |
| Seam conversions | `Matrix4::to_affine_rows`, `to_affine_rows_checked`, `to_affine_rows_strict`, `to_projective_mat4`, `to_projective_mat4_checked`, and `to_projective_mat4_strict`, bridging CPU simulation math to the verified GPU layouts in `f3d-core` with explicit precision policy enforcement. |

---

## 3. Strict Invariants

1. **Operation Ordering and Formula Exactness**: Arithmetic follows the exact operations of Three.js r186 math modules:
   - `Matrix4.multiply_matrices` evaluates 16 dot products in exact column-major index order.
   - `Matrix4.compose(position, quaternion, scale)` evaluates Shoemake's quaternion matrix expansion without pre-normalizing the quaternion and with direct column scaling (`sx, sy, sz`).
   - `Vector3.apply_matrix4` evaluates the perspective denominator `w = 1.0 / (e[3]*x + e[7]*y + e[11]*z + e[15])` and multiplies by `w`.
   - `Vector3.normalize` implements `divideScalar(length || 1)` (Three.js `Vector3.js:794`), preserving signed zeros (`-0.0`) on zero vectors and preserving non-NaN components when one component is `NaN`.
   - `Matrix4.make_perspective` and `Matrix4.make_orthographic` preserve exact Three.js r186 formula ordering, supporting WebGL ([-1, 1]), WebGPU ([0, 1]), and reversedDepth ([1, 0]) depth intervals.
2. **Singular Matrix Behavior**: Calling `Matrix4::invert()` on a singular matrix (`determinant == 0.0`) produces an all-zero matrix (`[0.0; 16]`), matching the exact behavior of Three.js r186 `Matrix4.invert()`. Similarly, `Matrix3::invert()` produces an all-zero matrix (`[0.0; 9]`) on singular matrices.
3. **Non-Unit Authored Quaternions**: `compose` does NOT normalize input quaternions; unnormalized authored rotations evaluate directly as written in upstream Three.js.
4. **Negative Scale & Shear**: Negative determinants, reflections, and non-orthogonal shear transformations are preserved accurately through multiplication, inversion, and point transformation.
5. **Exact Structural Affine Guard Before Narrowing**: `Matrix4::to_affine_rows` checks exact structural affine requirements (`e[3] == 0.0 && e[7] == 0.0 && e[11] == 0.0 && e[15] == 1.0`) at full `f64` precision before narrowing to `f32`. Any non-zero perspective term (including subnormals e.g. `1e-100`) is rejected with `LayoutError::NonAffineMatrix` to retain `ProjectiveMat4`.
6. **Signed Zero & IEEE 754**: Calculations use standard IEEE 754 `f64` double precision, preserving signed zero (`-0.0`) and exceptional values (`Infinity`, `NaN`) without artificial clamping.
7. **No Wire Layout Duplication**: GPU wire layouts (`AffineRows`, `ProjectiveMat4`) reside exclusively in `f3d-core::layout`; `f3d-math` implements conversion methods against those canonical types.
8. **Projection Matrices & Wire Layouts**: Perspective matrices (`e[11] = -1.0, e[15] = 0.0`) are strictly rejected by `to_affine_rows` and retained byte-for-byte by `to_projective_mat4`. Orthographic matrices (`e[11] = 0.0, e[15] = 1.0`) are affine and convert to `AffineRows`.
9. **Explicit f64 to f32 Wire Narrowing Policy**: Checked narrowing evaluates absolute error `|val - (val as f32 as f64)|` against `max_abs_err` and relative error against `max_rel_err`. A strict variant rejects non-finite values (`NaN`, `+Infinity`, `-Infinity`). Checked conversions `to_affine_rows_checked` and `to_projective_mat4_checked` reject precision loss exceeding tolerance, with affine shape guarded strictly at `f64` precision.
10. **ECMAScript Numeric Lowering Invariants**: Functions in `jsnum` adhere strictly to ECMA-262: `to_int32` and `to_uint32` execute modulo $2^{32}$ wraparound without saturating; `js_rem` matches the sign of the dividend; `js_min` and `js_max` propagate `NaN` and order `-0.0 < +0.0`; `js_round` rounds halfway cases toward `+Infinity` (round-half-up); and bitwise shift operators mask shift counts to 5 bits (`& 0x1F`). Rust `as`-casts saturate on overflow and are forbidden as substitutes for JS numeric conversions.
11. **Euler Angles & Matrix3 Semantics**: `Euler::set_from_rotation_matrix` supports all 6 rotation orders with clamp on input within `[-1.0, 1.0]` and singular gimbal-lock threshold `0.9999999` matching Three.js r186 `Euler.js`. `Quaternion::set_from_euler` evaluates exact half-angle product formulas for all orders. `Matrix3::get_normal_matrix` computes `inverse().transpose()` of the upper 3x3 of a 4x4 matrix, and `Matrix3::invert` returns all zeros `[0.0; 9]` on singular matrices matching Three.js r186 `Matrix3.js`.
12. **Vector3 Operation Semantics**: `cross` computes vector cross products $\mathbf{a}\times\mathbf{b}$ preserving anti-commutativity; `lerp` implements `self += (v - self) * alpha`; `angle_to` evaluates $\arccos(\text{clamp}(\mathbf{a}\cdot\mathbf{b} / (\|\mathbf{a}\|\|\mathbf{b}\|), -1, 1))$ returning $\pi/2$ on zero length; `distance_to` and `distance_to_squared` evaluate Euclidean distance; `project` and `unproject` faithfully emulate camera transformation sequences (`matrixWorldInverse` $\to$ `projectionMatrix`, and `projectionMatrixInverse` $\to$ `matrixWorld`); and matrix column extractions pull column vectors directly from column-major element buffers without transposition.
13. **Box3 and Sphere Semantics and Conventions**:
    - **Box3 Empty State**: An empty box has `min = (+inf, +inf, +inf)` and `max = (-inf, -inf, -inf)`. `is_empty()` returns `true` when any component of `max` is less than `min`. Empty boxes return `Vector3::zero()` for `get_center()` and `get_size()`, remain unmodified under `apply_matrix4()`, and never intersect or contain any point.
    - **Box3 8-Corner Matrix Transform**: `apply_matrix4` generates all 8 corner vertices $(2^3)$ from `(min, max)`, transforms each corner point via `p.apply_matrix4(matrix)` with perspective divide, and resets the bounding box from the transformed points, matching Three.js r186 `Box3.applyMatrix4`.
    - **Sphere Empty State**: An empty sphere has `center = (0, 0, 0)` and `radius = -1.0`. `is_empty()` returns `radius < 0.0`. A sphere with `radius == 0.0` is non-empty (degenerate point sphere).
    - **Sphere Matrix Transform**: `apply_matrix4` transforms `center` via `center.apply_matrix4(matrix)` and scales `radius` by `matrix.get_max_scale_on_axis()`. `Matrix4::get_max_scale_on_axis()` computes the maximum Euclidean norm across the first three basis columns of `matrix`, matching Three.js r186 `Matrix4.getMaxScaleOnAxis()`.
    - **Reciprocal Intersections**: `Box3.intersects_sphere(&s)` and `Sphere.intersects_box(&b)` evaluate identically, finding the closest clamped point on the box to the sphere center and comparing the squared Euclidean distance against $r^2$. Empty bounding volumes never intersect.
14. **Plane, Ray, and Triangle Semantics**:
    - **Plane Hessian Normal Form**: Represents infinite planes via unit normal and constant (`dot(n, p) + c = 0`). `set_from_coplanar_points(a, b, c)` evaluates $(c - b) \times (a - b)$ normalized with constant $-a \cdot n$. `intersect_line` returns intersection point with optional segment clamping, returning line start if coplanar. `apply_matrix4` transforms coplanar point by matrix and normal by inverse transpose normal matrix, updating constant without distortion under non-uniform scale.
    - **Ray Intersections**: Parameterized as $p(t) = \text{origin} + \text{direction} \cdot t$. `intersect_sphere` returns front intersection $t_0$, or rear exit point $t_1$ if origin is inside. `intersect_box` implements slab intersection with strict IEEE 754 NaN handling. `intersect_triangle` implements the watertight ray/triangle intersection (Woop, Benthin, Wald 2013) evaluating signed edge functions $u, v, w$, supporting strict `backface_culling`.
    - **Triangle Geometry & Closest Point**: `get_normal` yields $(c - b) \times (a - b)$ normalized (or $(0,0,0)$ on degenerate zero area). `get_barycoord` computes exact barycentric coordinates summing to 1, returning `None` for degenerate/collinear triangles. `closest_point_to_point` implements Christer Ericson's Voronoi region-walk method across vertex, edge, and face regions with minimal redundant computation. `is_front_facing` evaluates $(c - b) \times (a - b) \cdot \text{direction} < 0.0$ strictly.
15. **Color Semantics and Color Space Management**:
    - **Linear Working Color Space**: Color components `(r, g, b)` are internally stored in `ColorSpace::LinearSRGB`. Hexadecimal inputs (`set_hex`), CSS strings (`set_style`), and explicit color spaces (`set_rgb`) automatically convert to the working color space via `color_space_to_working`.
    - **Exact Piecewise sRGB Transfer Functions**: Conversions strictly implement Three.js r186 `ColorManagement.js`:
      - `srgb_to_linear(c)`: if $c < 0.04045$, returns $c \times 0.0773993808$; else returns $(c \times 0.9478672986 + 0.0521327014)^{2.4}$.
      - `linear_to_srgb(c)`: if $c < 0.0031308$, returns $c \times 12.92$; else returns $1.055 \times c^{0.41666} - 0.055$.
      - Negative and non-positive inputs stay on the linear branch without crashing or returning `NaN`.
    - **Display P3 Conversion**: Color conversions across Rec. 709 (sRGB) and Display P3 evaluate through intermediate CIE 1931 XYZ space using canonical upstream matrices `LINEAR_REC709_TO_XYZ`, `XYZ_TO_LINEAR_REC709`, `LINEAR_DISPLAY_P3_TO_XYZ`, and `XYZ_TO_LINEAR_DISPLAY_P3`, matching Three.js r186 `ColorSpaces.js`.
    - **HSL Hue Modulo & hue2rgb**: `set_hsl` applies Euclidean modulo $h \bmod 1.0 = ((h \bmod 1) + 1) \bmod 1$ on hue and clamps saturation/lightness to `[0.0, 1.0]`. RGB conversion evaluates the 6-sector piecewise helper `hue2rgb` matching Three.js r186.
    - **CSS String Parsing & Upstream Exactness**: `set_style` strictly matches Three.js r186 `Color.js:286-410`:
      - Functional syntax `/^(\w+)\(([^\)]*)\)/` requires case-sensitive lowercase names (`'rgb'`, `'rgba'`, `'hsl'`, `'hsla'`), tolerates no leading whitespace (no outer trim), and tolerates trailing content after `)`.
      - Integer `rgb()` and `rgb(%)` strictly require `\d+` digits; floats and negative numbers are rejected.
      - Hex strings match `/^\#([A-Fa-f\d]+)$/` for 3-digit `#rgb` (divided by 15.0) and 6-digit `#rrggbb`.
      - Unmatched non-empty strings fall through to `set_color_name` across 148 standard CSS keywords (`COLOR_NAMES`).
      - Unrecognized or malformed styles leave color components completely unmodified and return `StyleOutcome::IgnoredInvalid`.
      - Discarded alpha < 1.0 returns `StyleOutcome::AlphaIgnored` matching Three.js warning semantics.
    - **Zero-Allocation Formatting**: Non-allocating `#![no_std]` methods `write_hex_string`, `format_hex_string`, `write_style`, and `format_style` write directly into caller byte buffers or `core::fmt::Write` destinations without heap allocation.
    - **Exact Rounding**: `get_hex` and `write_style` evaluate `crate::jsnum::js_round(clamp(c * 255, 0, 255))` adhering strictly to ECMAScript round-half-up semantics instead of saturating casts or standard floor offsets.
    - **Interpolation**: `lerp` evaluates linear interpolation per channel. `lerp_hsl` converts colors to HSL in working space, linearly interpolates hue, saturation, and lightness, and reapplies `set_hsl` with Euclidean modulo wrapping.
    - **Clamped Subtraction**: `Color.sub` clamps each component at `0.0` matching Three.js r186 `Math.max(0, this.r - color.r)`.

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
- Transcendental float operations (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `log`, `pow`, `sqrt`, `cbrt`, `hypot`) delegate to target floating-point methods:
  - `sqrt`: Correctly rounded (0 ULP difference, bit-for-bit exact with IEEE 754 and V8 reference).
  - Unary and binary transcendentals: Within 1 ULP of V8 reference vectors across 48,888 test cases (up to 2 ULP on compound reductions `tan` and `hypot` where host libm and V8 polynomial approximations round in opposite directions relative to the true real value).
  - `pow`: Conforms to IEEE 754-2008 (§9.2.1) / ISO C99 (`1.0.powf(any) == 1.0` and `(-1.0).powf(+-inf) == 1.0`), honestly documenting the 5 divergence cases where ECMAScript §21.2.2.26 requires `NaN`.

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
- `serde`: Enables serialization and deserialization for math primitives (`Vector3`, `Quaternion`, `Matrix4`, `Euler`, `Matrix3`, `Box3`, `Sphere`, `Line3`, `Plane`, `Ray`, `Triangle`).

---

## 10. Conformance & Verification Plan

- Independent analytical reference tests in `tests/transform_tests.rs` covering:
  - Positive: Identity, TRS composition, hierarchy parent/child multiplication, vector transformation, and invertible matrices.
  - Decompose reference cases: TRS roundtrip, negative determinant flipping `sx`, and det==0 singular fallback resetting to unit scale and identity quaternion matching Three.js r186 `Matrix4.js:1081-1088`.
  - Projection reference cases: Perspective and orthographic projections verifying NDC z near/far mapping across WebGL ([-1, 1]), WebGPU ([0, 1]), and reversed depth ([1, 0]).
  - Negative: Singular matrix (det == 0 produces all zeros), negative determinant reflection, shear, non-unit authored quaternion (catching premature normalization), perspective divide, and subnormal/non-affine matrix rejection.
  - Wire conversion seam roundtrip with `f3d-core::layout::AffineRows` and `ProjectiveMat4`.
  - Cross-crate projection agreement: `f3d_core::layout::ProjectiveMat4::transform_homogeneous` (divided by w) matches `Vector3::apply_matrix4` within 1e-6 (f32 narrowing).
- Bounding volume analytical reference tests in `tests/box3_sphere_tests.rs` covering:
  - `Box3`: empty convention (`min = +inf, max = -inf`), center/size of empty box, set from points, set from flat buffers, point/vector/scalar expansion, point containment, box containment, intersection tests, clamp, distance to point, union, intersection, and 8-corner `apply_matrix4` with non-uniform scale and rotation.
  - `Sphere`: empty convention (`radius = -1.0`), set from points with optional center fallback, point containment, signed distance, surface clamping, bounding box extraction, translation, point expansion, union, and `apply_matrix4` scaling radius by `Matrix4::get_max_scale_on_axis()`.
  - Reciprocal intersection tests between `Box3` and `Sphere` across concentric, face-touching, face-separated, corner-separated, and corner-overlapping geometric configurations.
- Plane, Ray, and Triangle analytical reference tests in `tests/plane_ray_triangle_tests.rs` covering:
  - `Plane`: Hessian normal form derivation from coplanar points and normal/point, signed point distance, sphere distance, point projection, line segment intersection (with clamping and coplanar start return), matrix transform via normal matrix, translation, and reciprocal box/sphere intersection.
  - `Ray`: point along ray (`at`), closest point to point (with origin clamping), squared distance to point, sphere intersection (entrance vs exit for inside rays), plane distance and intersection, box intersection (slab method with NaN handling, hits from outside/inside, misses in Y/away/parallel), watertight triangle intersection (hits, backface culling, collinear degenerate rejection), and matrix transform.
  - `Triangle`: unit normal derivation, exact barycentric coordinates, point containment, degenerate/collinear rejection (normal=(0,0,0), barycoord=None, contains=false), Christer Ericson Voronoi region-walk closest point across all 7 regions (vertices A/B/C, edges AB/AC/BC, and face), strictly front-facing test, and reciprocal box intersection.
- ECMAScript numeric differential reference tests in `tests/jsnum_differential_tests.rs`:
  - 11,844 V8-generated test vectors ingested via `include_str!` from `tests/fixtures/math/jsnum_expected.txt`.
  - Exact bit-for-bit comparison across all 11 numeric lowering operations (`to_int32`, `to_uint32`, `round`, `trunc`, `sign`, `shift_left`, `shift_right`, `shift_unsigned_right`, `rem`, `min`, `max`).
  - Verifies signed zero preservation (`-0.0` vs `+0.0`), IEEE 754 NaN propagation and sign bits, subnormals, halfway rounding toward $+Infinity$, modulo $2^{32}$ wraparound, and shift counts masked to 5 bits.
- Color and ColorManagement analytical reference tests in `tests/color_tests.rs`:
  - Piecewise sRGB transfer functions at exact threshold boundaries (`0.04045` and `0.0031308`), checking continuity, zero, unity, and negative handling.
  - Display P3 and Linear Display P3 conversions matching upstream Three.js r186 test fixtures (`ColorSpaces.tests.js`).
  - Hexadecimal round-trip tests (`0xRRGGBB`) across primary, secondary, and boundary colors with zero-allocation byte formatting.
  - HSL hue wraparound across multi-turn angles ($h = 0.75, 1.75, -0.25, 5.75$), pure saturated hues ($0^\circ, 60^\circ, 120^\circ, 180^\circ, 240^\circ, 300^\circ$), and achromatic lightness ($s = 0$).
  - CSS style string parsing across `#rgb`, `#rrggbb`, `rgb()`, `rgba()`, `hsl()`, `hsla()`, with whitespace tolerance, float rejection on integer rgb, case-sensitivity, no-outer-trim, trailing-content tolerance, and `StyleOutcome` validation.
  - Upstream divergence regressions asserting that invalid styles leave color components completely untouched.
  - 148 CSS named color keyword validation (`COLOR_NAMES`), case-insensitive lookup in `set_color_name`, and fallback from `set_style`.
  - Zero-allocation string formatting into fixed byte slices and custom `core::fmt::Write` destinations.
  - Exact ECMAScript §7.1.12.1 channel formatting in `get_style` / `format_style` matching Three.js r186 `Color.getStyle()`: nonfinites format as `"NaN"`, `"Infinity"`, `"-Infinity"`; signed zeros format as `"0"`; finite magnitudes $< 10^{21}$ use shortest decimal representation without decimal points or `i64` saturation (e.g. `9223372036854776000` at $2^{63}$, `1000000000000001000` at $1e18 + 1024$); magnitudes $\ge 10^{21}$ format in exponential notation with signed positive exponent (e.g. `"1e+21"`).
  - Rounding fidelity in `get_hex` verifying `crate::jsnum::js_round` round-half-up behavior.
  - Linear RGB interpolation (`lerp`) and HSL interpolation (`lerp_hsl`).
  - Upstream 3x3 matrix constants (`LINEAR_REC709_TO_XYZ`, `XYZ_TO_LINEAR_DISPLAY_P3`, `LINEAR_SRGB_TO_LINEAR_DISPLAY_P3`, `LINEAR_DISPLAY_P3_TO_LINEAR_SRGB`) and round-trip transformation accuracy.
  - Clamped subtraction and arithmetic operations.
- Transcendental differential reference tests in `tests/transcendental_differential_tests.rs`:
  - 48,888 V8-generated test vectors ingested via `include_str!` from `tests/fixtures/math/transcendental_expected.txt` (1,260 unary cases across 10 ops, 47,628 pairwise binary cases across `atan2`, `pow`, `hypot`).
  - Bit-for-bit evaluation of Rust core/std float methods against Node V8 reference vectors, collecting exact matches, 1-ULP differences, and worst ULP distances.
  - Verifies `sqrt` is bit-for-bit exact (0 ULP difference, 100% exact match across all 126 cases per IEEE 754 correctly rounded contract).
  - Verifies all transcendental operations match within 1 ULP on target (bounded by at most 2 ULP on compound reductions `tan` and `hypot`).
  - Captures and documents the 5 edge cases where ECMAScript §21.2.2.26 specifies `NaN` for `Math.pow(1, NaN)` and `Math.pow(+-1, +-inf)` while IEEE 754-2008 / ISO C99 / Rust `f64::powf` specify `1.0`.
  - Emits formatted empirical summary table documenting libm vs V8 characteristics.

---

## 11. Explicit No-Claims

- This math foundation does **NOT** claim full Three.js geometry/bounding compatibility (Frustum, Cylindrical, Spherical, and transcendental/spline curves are deferred to subsequent slices).
- It does **NOT** claim SIMD acceleration, batch processing crossover, packed scene execution, or GPU performance gains.
