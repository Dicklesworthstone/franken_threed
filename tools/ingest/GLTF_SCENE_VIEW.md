# Authored glTF cameras and punctual lights

`gltf_scene_view.mjs` imports selected-scene camera and `KHR_lights_punctual`
instances, then evaluates them from an existing animation player's packed world
matrices. It does not load buffers, create another scene graph, advance animation,
allocate GPU resources, choose a frame loop, or modify the borrowed pose.

```js
import {
  decodeGltfSceneView, createGltfSceneView,
} from './gltf_scene_view.mjs';

const definition = decodeGltfSceneView(gltfJson, { scene: 0 });
const view = createGltfSceneView(pose, definition);
// After the application's existing animation update and GPU scene upload:
const frame = view.sample({ cameraNode: view.cameras[0].node, aspectRatio: width / height });
scene.render({ colorView, depthView,
  viewProjection: frame.viewProjection, lighting: frame.lighting });
```

`cameraNode` is the original **node index**, not an index into the glTF cameras
array. A camera definition instantiated on two nodes produces two separately
selectable views. With one selected-scene camera it can be omitted; with zero or
multiple cameras it cannot. `view.cameras` and `view.lights` retain frozen source
IDs, names, node names and projection/light properties for application selection.
Cameras and lights outside the selected scene, and uninstantiated definitions,
are not imported. The selected scene may have at most eight punctual lights,
matching the existing renderer; additional lights cause a preflight error, not
silent truncation. Unknown camera/light extensions require the source route.

`sample()` returns independent frozen `viewMatrix`, `projectionMatrix`,
`viewProjection`, `cameraPosition`, `viewDirection`, `lighting`, `poseVersion`,
`cameraNode`, `cameraIndex` and `type` fields. The matrices are column-major;
`viewProjection` maps world coordinates directly to **WebGPU clip depth 0..1**.
Finite and infinite perspective cameras and orthographic cameras are supported.
Orthographic lighting uses constant `viewDirection` toward the camera rather than
incorrectly varying the view vector with surface position.

An omitted perspective `aspectRatio` requires the viewport aspect on every sample,
so resize handling needs no camera reconstruction. An authored perspective aspect
or orthographic x/y magnification remains authoritative. The application must
match or letterbox its render viewport to that aspect; this helper does not infer
attachment dimensions, stretch/crop the authored image, or configure viewports.
The returned matrices can also be consumed directly by another compatible renderer.

Camera position includes all parent transforms and the pose's root transform.
The lens follows the transformed local -Z axis, while transformed +Y is projected
onto the perpendicular plane to produce an orthonormal camera basis. This removes
image scaling and reorthogonalizes shear inherited from rotated, nonuniformly
scaled ancestors. Degenerate forward/up axes are rejected instead of inventing a
view. This is an explicit forward/up convention for ambiguous sheared/reflected
world transforms, not a claim to match every source engine's decomposition.

Directional and spot directions follow transformed local -Z. Point/spot position
is the node's world location. Light colors are linear; intensity, range and spot
cone angles are copied without scaling. Directional intensity remains in lux;
point/spot intensity remains in candela. The renderer already implements their
attenuation and shading. `view.sampleLights()` returns just the world-space light
array for applications using an external camera, without requiring any imported
camera. No environment, area, shadow or ambient lighting is synthesized.

Only existing core node-transform animation is consumed. Animation-pointer
extensions that animate camera/light properties are not implemented here. Invalid
parameters, detached/nonfinite pose data and pose changes during a sample fail
before publishing a frame. A failed sample does not mutate previous snapshots or
the pose. A disposed pose makes subsequent sampling fail.

```sh
node --test tools/ingest/gltf_scene_view.test.mjs
```

These host tests verify source metadata selection, matrices and light descriptors
against independently known results. They do not execute a native GPU or assert
pixel parity. The pose boundary is supplied packed world matrices in these tests;
actual animation/model integration is a separate check.

## Integrated model APIs

Both `createCpuGltfAnimationModel` and the GPU model factories expose `view`,
`cameras` and `lights`. The owning `loadGpuGltfAnimationScene` URL/GLB entry exposes
them too. Model preparation snapshots `sceneView` metadata before reading binary
accessors or invoking texture resolvers; `KHR_lights_punctual` is accepted as a
required extension only through this validated model route. The existing pose
sampler still rejects animation-pointer extensions for camera/light properties.
`animation_model.mjs` also re-exports `createGltfSceneView` and its error type for
applications binding a decoded model to an existing pose.

GPU models provide the explicit convenience method:

```js
model.update(deltaSeconds);
model.renderCamera({ colorView, depthView }, {
  cameraNode: model.cameras[0].node,
  aspectRatio: width / height,
});
```

This fills the existing renderer's `viewProjection` and `lighting` fields from one
current pose sample. It does not replace the original `render(frame)` method or
implicitly choose an external camera. Supplying either generated field to
`renderCamera` is an error; custom lighting or mixed source/external views can use
`model.view.sample(...)` / `sampleLights()` and the original `render(frame)`.
Explicit camera rendering rejects a pose version that has not been uploaded to
the scene. Camera/frame getters cannot reenter, update or dispose the GPU model
during frame preparation. Recoverable view errors publish no draw and retain
resources; existing terminal GPU failures still dispose the model-owned pose.
A legacy geometry-only decoded payload without `sceneView` still constructs a
GPU model with empty camera/light lists and uses external rendering as before.

```sh
node --test tools/ingest/gltf_scene_view.test.mjs \
  tools/ingest/gltf_scene_view_model.test.mjs \
  tools/ingest/animation_model.test.mjs
```

The model integration suite exercises production accessor/material decoding,
animation sampling/blending, root/parent transforms and CPU deformation. Its GPU
scene is a test boundary; the two owning-loader seam tests also replace unchanged
asset transport and native texture preparation. Those tests verify forwarding and
preflight ordering, not HTTP/image decoding, shader execution or pixel equivalence.
The original model regression suite runs unchanged, including morph-plus-skin and
32-influence CPU skinning. Native browser/GPU and full-workspace gates are separate
from these focused host checks.
