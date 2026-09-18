# Quantized and compressed glTF inputs

## Quantized geometry

The existing `decodeGltfGeometry`, `decodeGltfAnimationModel`, CPU/GPU model
factories and owning asset-to-scene loader accept `KHR_mesh_quantization` for
positions, normals, tangents, UV sets and position/normal/tangent morph deltas.
The asset must declare the extension in `extensionsRequired`. Quantization does
not change the existing restrictions on material extensions or primitive types.

Positions and UVs accept signed/unsigned 8-bit and 16-bit components, normalized
or literal. Normal/tangent components must be signed and normalized. Position
morph deltas must be signed; normal/tangent deltas must also be normalized.
Core unsigned-normalized UVs, colors and skin weights remain valid without the
extension. Integer colors, joints and weights do not gain new formats implicitly.

The existing accessor reader supplies normalized Float64 values, including sparse
overlays and interleaved/strided storage. No new f32 rounding or source mutation
is introduced at decoding. Node transforms, inverse-bind matrices and
`KHR_texture_transform` already carry any required dequantization; the decoder
neither guesses scales from accessor bounds nor applies these transforms twice.
Quantized normals/tangents need not be exactly unit length; normalization remains
at the renderer's existing lighting boundary. Generated flat normals, independent
instances and all admitted skin influences keep their existing paths.

Encoded input size is not the expanded geometry budget. Existing accessor and
output component budgets continue to charge the decoded arrays and any triangle
expansion. Deformation and GPU allocation retain their separate limits. This is
JavaScript data decoding, not a retained renderer submission or a Rust rewrite.
UV/color morph animation is not added by this change; the existing unsupported
morph-field errors remain explicit. No additional quantization is applied.

Run the focused checks:

```sh
node --test tools/ingest/animation_quantization*.test.mjs
```

These tests exercise binary accessor decoding, geometry conversion, animation
metadata and material preflight. They are not native GPU pixel or performance
measurements. The format rules follow the Khronos KHR_mesh_quantization spec:
https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_mesh_quantization/README.md
