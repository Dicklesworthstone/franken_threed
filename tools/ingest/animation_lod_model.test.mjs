/** Real model/scene/LOD wiring and package emission/relocation. Pose decoding,
 * camera projection and GPU resources are explicit doubles, not GPU evidence. */

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFile, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";

const lod = {
  groups: [
    {
      node: 0,
      levels: [
        { distance: 0, drawIndices: [0] },
        { distance: 10, hysteresis: 0.2, drawIndices: [1] },
      ],
    },
  ],
};
const definition = {
  nodeCount: 3,
  origins: [
    [0, 0, 0],
    [12, 0, 0],
    [9, 0, 0],
  ],
  ignoredChannels: [],
};
const identity = () => new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const drawables = () =>
  [0, 1, 2].map((id) => ({ geometry: { id, world: identity(), positions: new Float32Array(9) } }));
const ids = (frame) => frame.draws.map((draw) => draw.id);
async function environment(t) {
  const dir = await mkdtemp(join(tmpdir(), "f3d-lod-model-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  for (const name of [
    "animation_lod",
    "animation_scene",
    "animation_draw_order",
    "animation_scene_shadow",
    "animation_model_gpu",
    "build_animation",
  ])
    await copyFile(new URL(`./${name}.mjs`, import.meta.url), join(dir, `${name}.mjs`));
  const sources = {
    animation_runtime: `export class AnimationPoseError extends Error {constructor(code,message){super(message);this.code=code}}
      export function createAnimationPlayer(def){const pose={nodeCount:def.nodeCount,version:0,disposed:false,clips:[],instances:[],morphWeights:new Float64Array(),
        worldMatrices:new Float64Array(def.nodeCount*16),dispose(){this.disposed=true}};
        for(let i=0;i<def.nodeCount;i++){for(const j of [0,5,10,15])pose.worldMatrices[i*16+j]=1;
          pose.worldMatrices.set(def.origins[i],i*16+12)}return pose;}`,
    animation_controller: `export function createAnimationController(pose){return {update(){pose.version++},dispose(){}}}`,
    animation_webgpu: `export async function createGpuAnimationDeformer(device,pose,g){return {id:g.id,bufferBytes:64,poseVersion:pose.version,version:0,disposed:false,
      get worldMatrix(){return g.world},update(){this.poseVersion=pose.version;this.version++},whenIdle:async()=>{},dispose(){this.disposed=true}}}`,
    animation_render: `export class AnimationRenderError extends Error{constructor(code,message){super(message);this.code=code}}
      export async function createGpuAnimationRenderer(device){return {allocatedBytes:192,failed:false,
        async addMesh(gpu){return {id:gpu.id,dispose(){}}},render(frame){device.colors.push(frame)},whenIdle:async()=>{},dispose(){}}}`,
    animation_model: `export class GltfSceneViewError extends Error{constructor(code,message){super(message);this.code=code}}
      export class AnimationRaycastError extends GltfSceneViewError{};export class AnimationExportError extends GltfSceneViewError{};
      export const decodeGltfAnimationModel=prepared=>prepared;
      const helper=()=>({enabled:false,dispose(){}});
      export const createAnimationModelPicker=helper,createAnimationModelExporter=helper;
      export function createGltfSceneView(pose,definition){return {cameras:definition.cameras,lights:[],sample({cameraNode=1}={}){
        const c=definition.cameras.find(c=>c.node===cameraNode);if(!c)throw Error('unknown camera');
        const position=Array.from(pose.worldMatrices.slice(cameraNode*16+12,cameraNode*16+15));
        return {cameraNode,cameraIndex:c.camera,type:c.type,cameraPosition:position,poseVersion:pose.version,
          viewProjection:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
          lighting:c.type==='orthographic'?{lights:[],viewDirection:[0,0,1]}:{lights:[],cameraPosition:position}};
      }}}`,
    animation_bounds: `export function createAnimationBounds(){throw Error('unexpected culling')};export function animationBoundsVisible(){throw Error('unexpected culling')}`,
    animation_shadow_view: `export function fitAnimationShadowView(){throw Error('unexpected shadows')};export function animationShadowWorldBounds(){throw Error('unexpected shadows')}`,
    animation_shadow: `export function createGpuAnimationShadowMap(){throw Error('unexpected shadows')}`,
    animation_shadow_receiver: `export {};`,
    animation_deformer: `export function createAnimationDeformer(){throw Error('unused CPU deformer')}`,
    animation_gltf: `export const decodeGltfAnimation=json=>json.definition;`,
    gltf_instancing: `export const expandGltfInstances=json=>({json,instanceCount:0});`,
  };
  for (const [name, source] of Object.entries(sources))
    await writeFile(join(dir, `${name}.mjs`), source);
  const gpu = await import(pathToFileURL(join(dir, "animation_model_gpu.mjs")).href);
  const device = { colors: [] };
  const prepared = {
    definition,
    drawables: drawables(),
    source: [],
    diagnostics: [],
    sceneView: {
      cameras: [
        { node: 1, camera: 0, type: "perspective" },
        { node: 2, camera: 0, type: "orthographic" },
      ],
    },
  };
  async function model(options = { lod }) {
    const result = await gpu.createGpuDecodedAnimationScene(device, prepared, options);
    t.after(() => result.dispose());
    return result;
  }
  return { dir, gpu, device, prepared, model };
}

test("authored camera nodes sharing a camera asset have independent LOD history", async (t) => {
  const { model, device } = await environment(t),
    scene = await model();
  scene.renderCamera({}, { cameraNode: 1 });
  assert.deepEqual(ids(device.colors.at(-1)), [1, 2]);
  assert.equal(scene.lodStats.cameraKey, "gltf:1");
  // Orthographic lighting intentionally has no position, but its camera sample does.
  scene.renderCamera({}, { cameraNode: 2 });
  assert.deepEqual(ids(device.colors.at(-1)), [0, 2]);
  assert.equal(scene.lodStats.cameraKey, "gltf:2");
  assert.equal(scene.lodCameraCount, 2);
  assert.deepEqual(device.colors.at(-1).lighting, { lights: [], viewDirection: [0, 0, 1] });
  assert.equal(Object.hasOwn(device.colors.at(-1), "lodCamera"), false);
  scene.pose.worldMatrices[28] = 9;
  scene.pose.version++;
  scene.upload();
  scene.renderCamera({}, { cameraNode: 1 });
  assert.deepEqual(ids(device.colors.at(-1)), [1, 2]);
});

test("model reset forwards through the operation guard and resamples current camera positions", async (t) => {
  const { model, device } = await environment(t),
    scene = await model();
  scene.renderCamera({}, { cameraNode: 1 });
  assert.equal(scene.resetLodCamera("gltf:1"), scene);
  assert.equal(scene.lodStats, null);
  scene.pose.worldMatrices[28] = 9;
  scene.pose.version++;
  assert.throws(() => scene.renderCamera({}, { cameraNode: 1 }), { code: "GLTF_VIEW_STALE" });
  scene.upload();
  scene.renderCamera({}, { cameraNode: 1 });
  assert.deepEqual(ids(device.colors.at(-1)), [0, 2]);
  const settings = {
    get key() {
      scene.resetLodCamera("gltf:1");
      return "nested";
    },
    position: [0, 0, 0],
  };
  assert.throws(() => scene.renderCamera({ lodCamera: settings }, { cameraNode: 1 }), {
    code: "GLTF_VIEW_REENTRANT",
  });
});

test("explicit LOD camera overrides are preserved, including invalid null", async (t) => {
  const { model } = await environment(t),
    scene = await model();
  scene.renderCamera(
    { lodCamera: { position: [30, 0, 0], zoom: 10, key: "external" } },
    { cameraNode: 1 },
  );
  assert.equal(scene.lodStats.cameraKey, "external");
  assert.equal(scene.lodStats.groups[0].level, 0);
  const previous = scene.lodStats;
  assert.throws(() => scene.renderCamera({ lodCamera: null }, { cameraNode: 1 }), {
    code: "ANIMATION_LOD_OPTIONS",
  });
  assert.equal(scene.lodStats, previous);
});

test("explicit model draws bypass automatic camera LOD and preserve caller list", async (t) => {
  const { model, device } = await environment(t),
    scene = await model();
  scene.renderCamera({ draws: [scene.draws[2], scene.draws[0]] }, { cameraNode: 1 });
  assert.deepEqual(ids(device.colors.at(-1)), [2, 0]);
  assert.equal(scene.lodCameraCount, 0);
  assert.equal(scene.lodStats, null);
});

test("external model frames still require explicit LOD camera and disabled models are unchanged", async (t) => {
  const { model, device } = await environment(t),
    scene = await model();
  assert.throws(() => scene.render({ viewProjection: identity() }), {
    code: "ANIMATION_LOD_OPTIONS",
  });
  scene.render({
    viewProjection: identity(),
    lodCamera: { position: [12, 0, 0], key: "external" },
  });
  assert.equal(scene.lodStats.cameraKey, "external");
  const ordinary = await model({});
  ordinary.renderCamera({}, { cameraNode: 1 });
  assert.equal(ordinary.lodEnabled, false);
  assert.deepEqual(ids(device.colors.at(-1)), [0, 1, 2]);
});

test("generated GPU package includes hashed LOD runtime and executes after relocation", async (t) => {
  const { dir } = await environment(t);
  const { buildAnimation } = await import(pathToFileURL(join(dir, "build_animation.mjs")).href);
  const entry = join(dir, "asset.gltf"),
    out = join(dir, "gpu"),
    moved = join(dir, "relocated");
  await writeFile(entry, JSON.stringify({ definition }));
  const manifest = buildAnimation(entry, out, { webgpu: true });
  const artifact = manifest.artifacts.find((a) => a.file === "animation_lod.mjs");
  assert.ok(artifact);
  const bytes = await readFile(join(out, artifact.file));
  assert.equal(bytes.byteLength, artifact.bytes);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), artifact.sha256);
  assert.deepEqual(bytes, await readFile(new URL("./animation_lod.mjs", import.meta.url)));
  await rename(out, moved);
  const runtime = await import(pathToFileURL(join(moved, "gpu_playback.mjs")).href);
  const pose = runtime.createPlayer(),
    device = { colors: [] };
  const scene = await runtime.createGpuAnimationScene(device, pose, drawables(), { lod });
  try {
    scene.render({ viewProjection: identity(), lodCamera: { position: [12, 0, 0] } });
    assert.deepEqual(ids(device.colors.at(-1)), [1, 2]);
    assert.equal(scene.lodStats.suppressedDraws, 1);
  } finally {
    scene.dispose();
    pose.dispose();
  }
  const cpu = buildAnimation(entry, join(dir, "cpu"));
  assert.equal(cpu.emittedFiles.includes("animation_lod.mjs"), false);
  assert.equal(cpu.emittedFiles.includes("gpu_playback.mjs"), false);
});

test("top-level glTF scene option forwards LOD configuration without a second API", async (t) => {
  const { gpu, device, prepared } = await environment(t);
  const model = await gpu.createGpuGltfAnimationScene(device, prepared, [], { scene: { lod } });
  try {
    assert.equal(model.lodEnabled, true);
    model.renderCamera({}, { cameraNode: 1 });
    assert.deepEqual(ids(device.colors.at(-1)), [1, 2]);
  } finally {
    model.dispose();
  }
});
