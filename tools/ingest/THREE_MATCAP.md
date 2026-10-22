# Native Matcap materials

The native WebGPU route admits pinned r186 `MeshMatcapMaterial` and
`MeshMatcapNodeMaterial` with their default node slots. It shades through the
existing F3D material renderer and texture owner; it does not construct or submit
through an upstream renderer.

## Shading and textures

The shader computes the pinned view-space normal and view direction, including
the orthographic camera's parallel direction, then constructs the Matcap basis
and generated UVs with the source's `0.495` scale. The texture's matrix transforms
those generated coordinates. A Matcap texture's channel selector does not choose
a geometry attribute, so it can render a geometry without UVs. Color and normal
maps retain their ordinary independent UV channels and matrices.

Matcap RGB multiplies the diffuse color, vertex/instance colors and color map.
The Matcap's alpha does not affect opacity. The absent-texture fallback is the
source's gray ramp from `0.2` to `0.8` along the generated vertical coordinate.
Normal maps, flat and double-sided normal handling, opacity/alpha maps, alpha
testing, blending, fog, clipping and the existing deformation/instancing paths
remain available. Native alpha maps use their red channel. Lights, environment
lighting and shadow receivers do not modify Matcap RGB.

Material color, normal scale and map matrices remain live per draw. Adding,
removing or replacing a texture uses the existing asynchronous preparation
boundary. Texture content updates use the shared texture owner. Custom material
nodes and render hooks retain the scene bridge's explicit admission limits.

The WebGL-authored route continues using its pinned ShaderLib Matcap program,
preserving that source renderer's own coordinate behavior.

## Verification

`node --test tools/ingest/three_matcap.test.mjs` checks source material admission,
generated coordinates without geometry UVs, independent transforms/bindings,
live values and preparation transitions. The existing browser parity runner's
`native_matcap_materials`, `native_matcap_updates` and
`native_matcap_instanced` scenes compare the same source scenes with the pinned
native WebGPU renderer. They cover the fallback, sRGB textures, ignored Matcap
alpha, normal maps, live changes and orthographic instancing.
