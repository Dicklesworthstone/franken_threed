/** Real LOD, scene, ordering and shadow orchestration with deterministic GPU
 * resource doubles. These tests verify submission inputs/lifetimes, not pixels. */

import assert from "node:assert/strict";
import { copyFile, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";

const identity = () => new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const lod = (extra = {}) => ({
  groups: [
    {
      node: 0,
      levels: [
        { distance: 0, drawIndices: [0] },
        { distance: 10, hysteresis: 0.2, drawIndices: [1] },
      ],
    },
  ],
  ...extra,
});
const frame = (distance, key = "main", extra = {}) => ({
  viewProjection: identity(),
  lighting: { lights: [{ type: "directional", direction: [0, 0, -1] }] },
  lodCamera: { position: [distance, 0, 0], key },
  ...extra,
});
const ids = (draws) => draws.map((draw) => draw.id);

async function fixture(t, count = 3, options = {}, configure = () => {}) {
  const dir = await mkdtemp(join(tmpdir(), "f3d-lod-scene-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  for (const name of [
    "animation_scene",
    "animation_lod",
    "animation_draw_order",
    "animation_scene_shadow",
  ])
    await copyFile(new URL(`./${name}.mjs`, import.meta.url), join(dir, `${name}.mjs`));
  const sources = {
    state: `export const state={colors:[],shadows:[],fits:[],bounds:[],updates:[],resources:[],controllers:0};`,
    animation_controller: `import {state} from './state.mjs';
      export function createAnimationController(pose){return {update(){pose.version++},dispose(){state.controllers++}};}`,
    animation_webgpu: `import {state} from './state.mjs';
      export async function createGpuAnimationDeformer(device,pose,geometry){
        const gpu={id:geometry.id,bufferBytes:64,poseVersion:pose.version,version:0,disposed:false,failed:false,
          get worldMatrix(){return geometry.world},update(){this.poseVersion=pose.version;this.version++;state.updates.push(this.id)},
          whenIdle:async()=>{},dispose(){this.disposed=true}};state.resources.push(gpu);return gpu;}`,
    animation_render: `import {state} from './state.mjs';
      export class AnimationRenderError extends Error {constructor(code,message){super(message);this.code=code}}
      export async function createGpuAnimationRenderer(){
        state.allocations=(state.allocations??0)+1;
        const renderer={allocatedBytes:192,failed:false,disposed:false,
          async addMesh(gpu){const draw={id:gpu.id,dispose(){this.disposed=true}};state.resources.push(draw);return draw},
          render(input){if(Object.hasOwn(input,'lodCamera'))throw Error('LOD metadata leaked to renderer');
            state.onRender?.();if(state.colorFailure){this.failed=!!state.terminal;throw Error('color rejected')}
            state.colors.push(input);},whenIdle:async()=>{},dispose(){this.disposed=true}};
        state.resources.push(renderer);return renderer;}`,
    animation_bounds: `import {state} from './state.mjs';
      export function createAnimationBounds(pose,geometry){
        let snapshot={poseVersion:pose.version,id:geometry.id,visible:geometry.visible!==false};
        return {byteLength:48,sourceComponents:9,get snapshot(){return snapshot},
          update(){return snapshot={...snapshot,poseVersion:pose.version}},dispose(){}};}
      export function animationBoundsVisible(bounds){state.bounds.push(bounds.id);return bounds.visible;}`,
    animation_shadow_view: `import {state} from './state.mjs';
      export function animationShadowWorldBounds(entries){const ids=entries.map(e=>e.bounds.id);state.fits.push(ids);return {ids};}
      export function fitAnimationShadowView(light,world){return {viewProjection:new Float64Array(16),world};}`,
    animation_shadow: `import {state} from './state.mjs';
      export async function createGpuAnimationShadowMap(){const map={allocatedBytes:1024,version:0,failed:false,
        async addMesh(gpu){return {id:gpu.id}},render(input){if(state.shadowFailure)throw Error('shadow rejected');
          state.shadows.push(input);this.version++},whenIdle:async()=>{},dispose(){this.disposed=true}};
        state.resources.push(map);return map;}`,
  };
  for (const [name, source] of Object.entries(sources))
    await writeFile(join(dir, `${name}.mjs`), source);
  const { state } = await import(pathToFileURL(join(dir, "state.mjs")).href);
  const pose = { nodeCount: 1, version: 0, disposed: false, worldMatrices: identity() };
  const drawables = Array.from({ length: count }, (_, id) => ({
    geometry: { id, world: identity(), positions: new Float32Array(9) },
  }));
  configure({ state, pose, drawables });
  const { createGpuAnimationScene } = await import(
    pathToFileURL(join(dir, "animation_scene.mjs")).href
  );
  const scene = await createGpuAnimationScene({}, pose, drawables, options);
  t.after(() => {
    if (!scene.disposed) scene.dispose();
  });
  return { scene, state, pose, drawables, dir, createGpuAnimationScene };
}

test("implicit color draws select a level and retain ungrouped meshes", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod() });
  assert.equal(scene.lodEnabled, true);
  assert.equal(scene.lodStats, null);
  scene.render(frame(2));
  assert.deepEqual(ids(state.colors.at(-1).draws), [0, 2]);
  scene.render(frame(12));
  assert.deepEqual(ids(state.colors.at(-1).draws), [1, 2]);
  assert.equal(scene.lodStats.suppressedDraws, 1);
  assert.equal(scene.lodStats.groups[0].level, 1);
  assert.ok(Object.isFrozen(scene.lodStats.drawIndices));
  assert.equal(scene.lodCameraCount, 1);
});

test("hysteresis belongs to a camera and commits only after accepted color", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod() });
  scene.render(frame(12, "a"));
  scene.render(frame(9, "b"));
  assert.deepEqual(ids(state.colors.at(-1).draws), [0, 2]);
  scene.render(frame(9, "a"));
  assert.deepEqual(ids(state.colors.at(-1).draws), [1, 2]);
  const previous = scene.lodStats;
  state.colorFailure = true;
  assert.throws(() => scene.render(frame(1, "a")), /color rejected/);
  assert.equal(scene.lodStats, previous);
  state.colorFailure = false;
  scene.render(frame(9, "a"));
  assert.deepEqual(ids(state.colors.at(-1).draws), [1, 2]);
  scene.render(frame(7, "a"));
  assert.deepEqual(ids(state.colors.at(-1).draws), [0, 2]);
});

test("failed first frame consumes no camera slot and reset releases bounded history", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod({ maxCameras: 1 }) });
  state.colorFailure = true;
  assert.throws(() => scene.render(frame(12, "unused")), /color rejected/);
  assert.equal(scene.lodCameraCount, 0);
  state.colorFailure = false;
  scene.render(frame(12, "a"));
  const count = state.colors.length;
  assert.throws(() => scene.render(frame(12, "b")), { code: "ANIMATION_LOD_CAMERA_LIMIT" });
  assert.equal(state.colors.length, count);
  assert.equal(scene.resetLodCamera("a"), scene);
  assert.equal(scene.lodStats, null);
  assert.equal(scene.lodCameraCount, 0);
  scene.render(frame(9, "b"));
  assert.deepEqual(ids(state.colors.at(-1).draws), [0, 2]);
});

test("missing or invalid camera rejects before depth/color work and preserves history", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod(), shadow: {} });
  scene.render(frame(12));
  const previous = scene.lodStats,
    depth = state.shadows.length;
  for (const camera of [
    undefined,
    null,
    { position: [NaN, 0, 0] },
    { position: [1, 2, 3], zoom: 0 },
  ])
    assert.throws(() => scene.render(frame(0, "main", { lodCamera: camera })));
  assert.equal(state.shadows.length, depth);
  assert.equal(state.colors.length, 1);
  assert.equal(scene.lodStats, previous);
});

test("shadow selection uses LOD membership, not the color frustum result", async (t) => {
  const { scene, state } = await fixture(
    t,
    3,
    { lod: lod(), shadow: {}, frustumCulling: true },
    ({ drawables }) => {
      drawables[1].geometry.visible = false;
    },
  );
  scene.render(frame(12));
  assert.deepEqual(ids(state.colors.at(-1).draws), [2]);
  assert.deepEqual(state.bounds, [1, 2]);
  assert.deepEqual(state.fits.at(-1), [1, 2]);
  assert.deepEqual(ids(state.shadows.at(-1).draws), [1, 2]);
  assert.equal(scene.shadowStats.casterCount, 2);
  assert.deepEqual(scene.cullingStats, {
    poseVersion: 0,
    testedMeshes: 2,
    culledMeshes: 1,
    submittedDraws: 1,
  });
  assert.deepEqual(scene.lodStats.drawIndices, [1, 2]);
});

test("excluded casters do not shift LOD indices; selected receivers still bound the fit", async (t) => {
  const { scene, state } = await fixture(t, 3, {
    lod: lod(),
    shadow: { casters: [false, true, false] },
  });
  scene.render(frame(1));
  assert.deepEqual(ids(state.shadows.at(-1).draws), []);
  assert.deepEqual(state.fits.at(-1), [0, 2]);
  scene.render(frame(12));
  assert.deepEqual(ids(state.shadows.at(-1).draws), [1]);
  assert.deepEqual(state.fits.at(-1), [1, 2]);
  assert.equal(scene.shadowStats.casterCount, 1);
});

test("a submitted depth pass does not commit a rejected color selection", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod(), shadow: {} });
  scene.render(frame(12));
  const previous = scene.shadowStats;
  state.colorFailure = true;
  assert.throws(() => scene.render(frame(1)), /color rejected/);
  assert.deepEqual(ids(state.shadows.at(-1).draws), [0, 2]);
  assert.equal(scene.shadowStats, previous);
  state.colorFailure = false;
  scene.render(frame(9));
  assert.deepEqual(ids(state.shadows.at(-1).draws), [1, 2]);
});

test("shadow rejection cannot commit camera history or submit color", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod(), shadow: {} });
  state.shadowFailure = true;
  assert.throws(() => scene.render(frame(12)), /shadow rejected/);
  assert.equal(scene.lodCameraCount, 0);
  assert.equal(state.colors.length, 0);
  state.shadowFailure = false;
  scene.render(frame(9));
  assert.equal(scene.lodStats.groups[0].level, 0);
});

test("explicit lists preserve caller order, bypass LOD and require an explicit shadow choice", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod(), shadow: {} });
  scene.render(frame(12));
  const custom = [scene.draws[2], scene.draws[0]];
  assert.throws(() => scene.render({ draws: custom }), { code: "ANIMATION_SCENE_SHADOW" });
  scene.render({ draws: custom, shadow: null });
  assert.deepEqual(ids(state.colors.at(-1).draws), [2, 0]);
  assert.equal(scene.lodStats, null);
  assert.equal(scene.lodCameraCount, 1);
  scene.render(frame(9));
  assert.equal(scene.lodStats.groups[0].level, 1);
});

test("all meshes keep current deformation while moving group origins change selected levels", async (t) => {
  const { scene, state, pose } = await fixture(t, 3, { lod: lod() });
  scene.render(frame(12));
  pose.worldMatrices[12] = 12;
  pose.version++;
  assert.throws(() => scene.render(frame(12)), { code: "ANIMATION_SCENE_STALE" });
  scene.upload();
  assert.deepEqual(state.updates, [0, 1, 2]);
  scene.render(frame(12));
  assert.equal(scene.lodStats.groups[0].level, 0);
  assert.equal(scene.lodStats.poseVersion, 1);
  scene.render(frame(30, "zoom", { lodCamera: { position: [30, 0, 0], key: "zoom", zoom: 3 } }));
  assert.equal(scene.lodStats.groups[0].distance, 6);
  assert.equal(scene.lodStats.groups[0].level, 0);
});

test("LOD also selects without sorting or culling and supports multi-draw levels", async (t) => {
  const options = {
    groups: [
      {
        node: 0,
        levels: [
          { distance: 0, drawIndices: [2, 0] },
          { distance: 10, drawIndices: [1] },
        ],
      },
    ],
  };
  const { scene, state } = await fixture(t, 4, { lod: options, sortObjects: false });
  scene.render(frame(1));
  assert.deepEqual(ids(state.colors.at(-1).draws), [0, 2, 3]);
  scene.render(frame(12));
  assert.deepEqual(ids(state.colors.at(-1).draws), [1, 3]);
});

test("selection precedes alpha ordering and ignores invalid inactive sort transforms", async (t) => {
  const { scene, state, drawables } = await fixture(t, 4, { lod: lod() }, ({ drawables }) => {
    drawables[0].alphaMode = "BLEND";
    drawables[0].geometry.world[14] = NaN;
    drawables[1].alphaMode = "BLEND";
    drawables[1].geometry.world[14] = 1;
    drawables[2].alphaMode = "BLEND";
    drawables[2].geometry.world[14] = 5;
  });
  scene.render(frame(12));
  assert.deepEqual(ids(state.colors.at(-1).draws), [3, 2, 1]);
  assert.throws(() => scene.render(frame(1)), { code: "ANIMATION_SCENE_SORT" });
  drawables[0].geometry.world[14] = 0;
  scene.render(frame(9));
  assert.deepEqual(ids(state.colors.at(-1).draws), [3, 2, 1]);
});

test("nested LOD options are snapshotted before the first initialization await", async (t) => {
  const { createGpuAnimationScene, state, pose, drawables } = await fixture(t, 3);
  const settings = lod();
  const pending = createGpuAnimationScene({}, pose, drawables, { lod: settings });
  settings.groups[0].levels[1].distance = 100;
  settings.groups[0].levels[1].drawIndices[0] = 0;
  const scene = await pending;
  t.after(() => scene.dispose());
  scene.render(frame(12));
  assert.deepEqual(ids(state.colors.at(-1).draws), [1, 2]);
});

test("bad LOD admission allocates no GPU resources and disposes its controller", async (t) => {
  const { createGpuAnimationScene, state, pose, drawables } = await fixture(t, 3);
  const allocations = state.allocations,
    controllers = state.controllers;
  for (const options of [
    { groups: [] },
    { groups: new Array(4097) },
    lod({ unknown: true }),
    { groups: [{ node: 0, levels: [{ distance: 1, drawIndices: [0] }] }] },
    { groups: [{ node: 0, levels: [{ distance: 0, drawIndices: [0, 0] }] }] },
  ])
    await assert.rejects(createGpuAnimationScene({}, pose, drawables, { lod: options }));
  assert.equal(state.allocations, allocations);
  assert.equal(state.controllers, controllers + 5);
});

test("reentrant rendering cannot commit and terminal failure releases LOD and GPU ownership", async (t) => {
  const { scene, state } = await fixture(t, 3, { lod: lod() });
  scene.render(frame(12));
  state.onRender = () => scene.resetLodCamera("main");
  assert.throws(() => scene.render(frame(1)), { code: "ANIMATION_SCENE_REENTRANT" });
  state.onRender = null;
  scene.render(frame(9));
  assert.equal(scene.lodStats.groups[0].level, 1);
  state.colorFailure = true;
  state.terminal = true;
  assert.throws(() => scene.render(frame(1)), /color rejected/);
  assert.equal(scene.failed, true);
  assert.equal(scene.lodCameraCount, 0);
  assert.equal(scene.lodStats, null);
  assert.ok(state.resources.every((r) => r.disposed));
});

test("LOD omission preserves all source draws and dispose clears retained histories", async (t) => {
  const { scene, state } = await fixture(t, 3);
  scene.render({});
  assert.deepEqual(ids(state.colors.at(-1).draws), [0, 1, 2]);
  assert.equal(scene.lodEnabled, false);
  assert.equal(scene.lodStats, null);
  assert.throws(() => scene.resetLodCamera(), { code: "ANIMATION_SCENE_LOD" });
  const active = await fixture(t, 3, { lod: lod() });
  active.scene.render(frame(12));
  active.scene.dispose();
  assert.equal(active.scene.lodCameraCount, 0);
  assert.equal(active.scene.lodStats, null);
  assert.throws(() => active.scene.render(frame(1)), { code: "ANIMATION_SCENE_DISPOSED" });
});

test("ordering rejects malformed subsets before culling and retains source tie order", async (t) => {
  const { dir, state, pose } = await fixture(t, 3);
  const { createAnimationDrawOrder } = await import(
    pathToFileURL(join(dir, "animation_draw_order.mjs")).href
  );
  const entries = Array.from({ length: 3 }, (_, id) => ({
    mesh: { id },
    deformer: { worldMatrix: identity(), poseVersion: 0 },
    geometry: { id },
    alphaMode: "BLEND",
  }));
  const order = createAnimationDrawOrder(entries, { pose, frustumCulling: true });
  t.after(() => order.dispose());
  for (const indices of [[0, 0], [3], [-1], [0.5], new Uint32Array([0])])
    assert.throws(() => order.order(identity(), indices));
  assert.equal(state.bounds.length, 0);
  assert.deepEqual(ids(order.order(identity(), [2, 0])), [0, 2]);
  assert.deepEqual(state.bounds, [0, 2]);
  assert.deepEqual(order.order(identity(), []), []);
  assert.equal(order.lastCulling.testedMeshes, 0);
});
