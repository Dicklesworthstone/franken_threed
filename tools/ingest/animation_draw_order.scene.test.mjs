import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

// Execute the production scene, ordering and packaging code. These explicit
// dependency doubles record boundary calls; they do not execute deformation,
// material shaders, image decoding or a GPU. Existing animation_scene.test.mjs
// independently covers the real controller/deformer/renderer with a GPU spy.
const controllerSource = `export function createAnimationController(pose) {
  return {update(delta) { pose.blend(delta); }, dispose() { this.disposed = true; }};
}`;
const deformerSource = `export async function createGpuAnimationDeformer(device, pose, geometry, options) {
  const gpu = {bufferBytes:0, poseVersion:pose.version, failed:false, disposed:false,
    worldMatrix:pose.world[geometry.node].slice(),
    update() { this.worldMatrix = pose.world[geometry.node].slice(); this.poseVersion = pose.version; },
    whenIdle:async () => {}, dispose() { this.disposed = true; }};
  device.deformers.push(gpu); device.deformerOptions.push(options); return gpu;
}`;
const rendererSource = `export class AnimationRenderError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}
export async function createGpuAnimationRenderer(device) {
  device.creations++;
  return {allocatedBytes:0, failed:false,
    async addMesh(gpu, material) { const mesh = {gpu, material, dispose() { this.disposed = true; }};
      device.meshes.push(mesh); return mesh; },
    render(frame) { device.frames.push(frame); },
    whenIdle:async () => {}, dispose() { device.releases++; }};
}`;
const runtimeSource = `export class AnimationPoseError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}
export function createAnimationPlayer(definition) {
  return {version:0, nodeCount:definition.nodes.length, clips:[], instances:[], morphWeights:new Float64Array(0),
    world:definition.nodes.map(node => [1,0,0,0,0,1,0,0,0,0,1,0,...node.translation,1]),
    blend() { this.version++; }, dispose() { this.disposed = true; }};
}`;
const dataURL = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const sourceURL = new URL('./animation_scene.mjs', import.meta.url);
const imports = {
  './animation_draw_order.mjs':new URL('./animation_draw_order.mjs', import.meta.url).href,
  './animation_controller.mjs':dataURL(controllerSource),
  './animation_webgpu.mjs':dataURL(deformerSource),
  './animation_render.mjs':dataURL(rendererSource),
};
let source = fs.readFileSync(sourceURL, 'utf8');
for (const [name, url] of Object.entries(imports)) source = source.replaceAll(`'${name}'`, JSON.stringify(url));
const {createGpuAnimationScene} = await import(dataURL(source));
const {createAnimationPlayer} = await import(dataURL(runtimeSource));
const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const definition = () => ({nodes:[.2,.8,.1].map(z => ({translation:[0,0,z]}))});
const device = () => ({frames:[], meshes:[], deformers:[], deformerOptions:[], creations:0, releases:0});
const drawables = () => ['BLEND','BLEND','OPAQUE'].map((alphaMode,node) => ({alphaMode,
  geometry:{node,positions:[0,0,0,1,0,0,0,1,0]},baseColor:[1,0,0,.5]}));
const frame = () => ({viewProjection:identity(),colorView:{},depthView:{}});
const order = d => d.frames.at(-1).draws.map(mesh => d.meshes.indexOf(mesh));
const code = expected => error => error.code === expected;
async function setup(options) {
  const d = device(), p = createAnimationPlayer(definition());
  const scene = await createGpuAnimationScene(d,p,drawables(),options);
  return {d,p,scene};
}

test('implicit scene draws use opaque first and current uploaded depth', async () => {
  const {d,p,scene} = await setup();
  scene.render(frame()); assert.deepEqual(order(d),[2,1,0]);
  p.world[0][14] = .9; scene.update(.5); scene.render(frame());
  assert.deepEqual(order(d),[2,0,1]); assert.equal(scene.poseVersion,1);
  assert.deepEqual(scene.draws,d.meshes); scene.dispose();
  assert.ok(d.meshes.every(mesh => mesh.disposed)); assert.equal(p.disposed,undefined);
});

test('two views recompute depth without uploading or advancing animation', async () => {
  const {d,p,scene} = await setup(), reverse = frame();
  reverse.viewProjection[10] = -1; reverse.viewProjection[14] = 1;
  scene.render(frame()); assert.deepEqual(order(d),[2,1,0]);
  scene.render(reverse); assert.deepEqual(order(d),[2,0,1]);
  scene.render(frame()); assert.deepEqual(order(d),[2,1,0]);
  assert.equal(p.version,0); assert.equal(d.deformers.length,3); scene.dispose();
});

test('equal transparent depths retain original registration order after crossing', async () => {
  const {d,p,scene} = await setup(); scene.render(frame());
  p.world[0][14] = p.world[1][14]; scene.update(0); scene.render(frame());
  assert.deepEqual(order(d),[2,0,1]); scene.dispose();
});

test('explicit draws preserve caller identity, overrides, order and empty clears', async () => {
  const {d,scene} = await setup();
  const draws = [{mesh:scene.draws[1],worldMatrix:identity()},scene.draws[0]];
  scene.render({...frame(),draws}); assert.equal(d.frames.at(-1).draws,draws);
  const empty = []; scene.render({...frame(),draws:empty}); assert.equal(d.frames.at(-1).draws,empty);
  scene.dispose();
});

test('scene frame accessors are not read twice to choose explicit draws', async () => {
  const {d,scene} = await setup(); let reads = 0;
  const draws = [scene.draws[0]];
  scene.render({...frame(),get draws() { reads++; return draws; }});
  assert.equal(reads,1); assert.equal(d.frames.at(-1).draws,draws); scene.dispose();
});

test('sortObjects false preserves the original implicit list, true validates before GPU creation', async () => {
  const {d,scene} = await setup({sortObjects:false}); scene.render(frame());
  assert.equal(d.frames.at(-1).draws,scene.draws); assert.deepEqual(order(d),[0,1,2]); scene.dispose();
  const unused = device();
  await assert.rejects(createGpuAnimationScene(unused,createAnimationPlayer(definition()),drawables(),{sortObjects:'yes'}),code('ANIMATION_SCENE_SORT'));
  assert.equal(unused.creations,0);
});

test('bad sorting matrices fail before render and do not poison a usable scene', async () => {
  const {d,p,scene} = await setup();
  assert.throws(() => scene.render({...frame(),viewProjection:[]}),code('ANIMATION_SCENE_SORT'));
  assert.equal(d.frames.length,0); assert.equal(scene.failed,false); assert.equal(p.version,0);
  scene.render(frame()); assert.deepEqual(order(d),[2,1,0]); scene.dispose();
});

test('sorting retains stale-pose, reentry and disposal guards', async () => {
  const {d,p,scene} = await setup(); p.blend();
  assert.throws(() => scene.render(frame()),code('ANIMATION_SCENE_STALE')); assert.equal(d.frames.length,0);
  scene.upload();
  assert.throws(() => scene.render({...frame(),get draws() { scene.dispose(); }}),code('ANIMATION_SCENE_REENTRANT'));
  assert.equal(scene.disposed,false); scene.render(frame()); scene.dispose();
  assert.throws(() => scene.render(frame()),code('ANIMATION_SCENE_DISPOSED'));
});

test('alpha classification is frozen before awaits and new map coordinates survive sorting', async () => {
  const d = device(), p = createAnimationPlayer(definition()), items = drawables();
  const uv = [0,0,1,0,0,1]; items[0].baseColorTexture = {view:{},sampler:{}};
  items[0].mapCoordinates = {baseColorTexture:{texCoords:uv}};
  const pending = createGpuAnimationScene(d,p,items);
  items[0].alphaMode = 'OPAQUE'; uv.fill(99);
  const scene = await pending; scene.render(frame()); assert.deepEqual(order(d),[2,1,0]);
  assert.deepEqual(d.meshes[0].material.mapCoordinates.baseColorTexture.texCoords,[0,0,1,0,0,1]);
  scene.dispose();
});

test('relocated GPU packages include and hash the sorting dependency; CPU output stays cold', async () => {
  // Use the actual packer and scene/order source, with explicit fixture decoder
  // and runtime modules. This checks emission, manifest bytes and relocation,
  // not the unchanged glTF/accessor decoder or actual WebGPU execution.
  const root = fs.mkdtempSync(path.join(os.tmpdir(),'f3d-sort-package-'));
  try {
    const modules = path.join(root,'modules'); fs.mkdirSync(modules);
    const runtimeModules = {
      'animation_runtime.mjs':runtimeSource,'animation_controller.mjs':controllerSource,
      'animation_deformer.mjs':'export function createAnimationDeformer() {}\n',
      'animation_webgpu.mjs':deformerSource,'animation_render.mjs':rendererSource,
      'animation_gltf.mjs':'export function decodeGltfAnimation(model) { return model; }\n',
    };
    for (const name of ['animation_scene.mjs','animation_draw_order.mjs','animation_bounds.mjs','build_animation.mjs']) {
      fs.copyFileSync(new URL('./'+name,import.meta.url),path.join(modules,name));
    }
    for (const [name,content] of Object.entries(runtimeModules)) fs.writeFileSync(path.join(modules,name),content);
    const entry = path.join(root,'model.gltf'); fs.writeFileSync(entry,JSON.stringify(definition()));
    const {buildAnimation} = await import(pathToFileURL(path.join(modules,'build_animation.mjs')));
    const gpu = buildAnimation(entry,path.join(root,'gpu'),{webgpu:true});
    const cpu = buildAnimation(entry,path.join(root,'cpu'));
    assert.ok(gpu.emittedFiles.includes('animation_draw_order.mjs'));
    assert.equal(cpu.emittedFiles.includes('animation_draw_order.mjs'),false);
    const item = gpu.artifacts.find(item => item.file === 'animation_draw_order.mjs');
    const bytes = fs.readFileSync(path.join(root,'gpu',item.file));
    assert.equal(item.bytes,bytes.length); assert.equal(item.sha256,createHash('sha256').update(bytes).digest('hex'));
    fs.renameSync(path.join(root,'gpu'),path.join(root,'relocated'));
    fs.rmSync(modules,{recursive:true});
    const exported = await import(pathToFileURL(path.join(root,'relocated','gpu_playback.mjs')));
    const d = device(), p = exported.createPlayer();
    const scene = await exported.createGpuAnimationScene(d,p,drawables());
    scene.render(frame()); assert.deepEqual(order(d),[2,1,0]); scene.dispose();
  } finally { fs.rmSync(root,{recursive:true,force:true}); }
});