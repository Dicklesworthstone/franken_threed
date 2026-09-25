import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Production scene adapter AND deformation input capture run unchanged. Only
// Three constructors, native evaluation and renderer submission are doubled.
// These are host integration tests, not GPU pixels or native-math conformance.
const key = Symbol.for("f3d.deformed-scene-contract");
const dependency = `
export class BufferAttribute {
  constructor(array,itemSize){this.array=array;this.itemSize=itemSize;this.count=array.length/itemSize;}
  getX(i){return this.array[i*this.itemSize];} getY(i){return this.array[i*this.itemSize+1];}
  getZ(i){return this.array[i*this.itemSize+2];} getW(i){return this.array[i*this.itemSize+3];}
}
export class BufferGeometry {
  constructor(){this.isBufferGeometry=true;this.attributes={};this.morphAttributes={};this.boundingBox=null;this.boundingSphere=null;this.groups=[];this.drawRange={start:0,count:Infinity};}
  setAttribute(name,attribute){this.attributes[name]=attribute;return this;}
  computeBoundingBox(){const a=this.attributes.position;this.boundingBox={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};for(let i=0;i<a.count;i++)for(let j=0;j<3;j++){const v=a.array[i*3+j];this.boundingBox.min[j]=Math.min(this.boundingBox.min[j],v);this.boundingBox.max[j]=Math.max(this.boundingBox.max[j],v);}}
  computeBoundingSphere(){if(!this.boundingBox)this.computeBoundingBox();const center=this.boundingBox.min.map((v,i)=>(v+this.boundingBox.max[i])/2);let radius=0;for(let i=0;i<this.attributes.position.count;i++)radius=Math.max(radius,Math.hypot(...center.map((v,j)=>v-this.attributes.position.array[i*3+j])));this.boundingSphere={center,radius};}
}
export class Mesh { intersectsFrustum(){return true;} }
export class SkinnedMesh extends Mesh { intersectsFrustum(){return true;} }
const state=()=>globalThis[Symbol.for('f3d.deformed-scene-contract')];
export function canAdmitMesh(...args){return state().admit(...args);}
export function renderScene(...args){return state().render(...args);}
export function createAdmissionError(code,detail=''){const e=new Error(code+': '+detail);e.code=code;e.reason=code;return e;}
`;
const dataUrl = (text) => "data:text/javascript;base64," + Buffer.from(text).toString("base64");
const dependencyUrl = dataUrl(dependency);
let source = await readFile(new URL("./deformed_scene_adapter.mjs", import.meta.url), "utf8");
for (const specifier of ["../../upstream/three.js/build/three.module.js", "./mesh_adapter.mjs", "./deformation_inputs.mjs"]) {
  const target = specifier === "./deformation_inputs.mjs"
    ? new URL(specifier, import.meta.url).href : dependencyUrl;
  const pattern = new RegExp(`(["'])${specifier.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\1`);
  assert.match(source, pattern);
  source = source.replace(pattern, JSON.stringify(target));
}
const { prepareDeformedScene: prepare, renderDeformedScene: render } = await import(dataUrl(source));
const { BufferAttribute, BufferGeometry, Mesh, SkinnedMesh } = await import(dependencyUrl);
const identity = () => ({ elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
let nextId = 0;
function object() {
  return {
    id: ++nextId, uuid: `node-${nextId}`, children: [], parent: null,
    visible: true, matrixWorldAutoUpdate: true, matrixWorld: identity(), updates: 0,
    layers: { mask: 1, test(other) { return Boolean(this.mask & other.mask); } },
    updateMatrixWorld() { this.updates++; },
    traverse(callback) { callback(this); for (const child of this.children) child.traverse(callback); },
  };
}
function add(parent, ...children) {
  for (const child of children) { child.parent = parent; parent.children.push(child); }
  return parent;
}
function mesh(skinned = true) {
  const item = Object.assign(Object.create(skinned ? SkinnedMesh.prototype : Mesh.prototype), object());
  item.isMesh = true;
  item.isSkinnedMesh = skinned;
  item.frustumCulled = true;
  item.material = { visible: true, isMeshBasicMaterial: true, colorWrite: false };
  item.geometry = new BufferGeometry();
  item.geometry.setAttribute("position", new BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  item.geometry.setAttribute("color", new BufferAttribute(new Float32Array(9).fill(0.5), 3));
  item.geometry.index = new BufferAttribute(new Uint16Array([0, 1, 2]), 1);
  item.geometry.computeBoundingBox(); item.geometry.computeBoundingSphere();
  if (skinned) {
    item.geometry.setAttribute("skinIndex", new BufferAttribute(new Uint16Array(12), 4));
    item.geometry.setAttribute("skinWeight", new BufferAttribute(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 4));
    item.bindMatrix = identity(); item.bindMatrixInverse = identity();
    item.skeleton = { bones: [{}], boneMatrices: new Float32Array(identity().elements), updates: 0, update() { this.updates++; this.boneMatrices[12] = this.updates; } };
    item.boundingSphere = { radius: 999 }; // Deliberately stale bind-pose bounds.
  }
  return item;
}
function setup() {
  const scene = object(), camera = { ...object(), isCamera: true };
  const state = { nativeCalls: 0, renderCalls: 0, submissions: [], shift: 10 };
  state.admit = (item) => Object.hasOwn(item, "onBeforeRender") || !item.material.isMeshBasicMaterial
    ? { admitted: false, code: "UNSUPPORTED_CALLBACK", reason: "source hook or material" } : { admitted: true };
  state.render = async (host, staged, stagedCamera, context, wasm, options) => {
    state.renderCalls++; state.staged = staged; state.camera = stagedCamera; state.options = options;
    assert.equal(staged.matrixWorldAutoUpdate, false);
    assert.equal(stagedCamera.matrixWorldAutoUpdate, false);
    const nodes = []; staged.traverse((node) => nodes.push(node)); state.nodes = nodes;
    if (staged.fog) return { admitted: [], refused: [{ uuid: staged.uuid, reason: "UNSUPPORTED_SCENE_FEATURE" }] };
    await host.executePacket(new Uint8Array([70, 51, 68, 80]), context);
    return { admitted: nodes.filter((n) => n.isMesh).map((n) => n.uuid), refused: [], marker: 42 };
  };
  const wasm = { f3d_deform_position_batch(...args) { state.nativeCalls++; state.inputs = args; return Float32Array.from(args[1], (v) => v + state.shift); } };
  const host = { async executePacket(packet, context) { state.submissions.push({ packet, context }); } };
  globalThis[key] = state;
  return { scene, camera, state, wasm, host };
}
const refuses = (code) => (error) => error.code === code || error.reason === code;

test("mixed nested scene uses one native evaluation and fresh post-pose bounds", () => {
  const { scene, camera, wasm, state } = setup(), a = mesh(), b = mesh(false), group = object();
  b.geometry.morphAttributes.position = [new BufferAttribute(new Float32Array(9).fill(2), 3)];
  b.morphTargetInfluences = [0.5]; group.isGroup = true; group.renderOrder = 11;
  add(scene, add(group, a, b));
  const oldBounds = a.geometry.boundingSphere, oldPositions = a.geometry.attributes.position.array.slice();
  const result = prepare(scene, camera, wasm);
  const stagedGroup = result.scene.children[0], staged = stagedGroup.children[0];
  assert.equal(state.nativeCalls, 1);
  assert.equal(result.deformedMeshCount, 2); assert.equal(result.skinnedMeshCount, 1);
  assert.equal(result.deformationVertexCount, 6);
  assert.equal(staged.geometry.attributes.position.getX(0), 10);
  assert.deepEqual(staged.geometry.boundingBox.min, [10, 10, 10]);
  assert.deepEqual(staged.geometry.boundingSphere.center, [10.5, 10.5, 10]);
  assert.equal(staged.boundingSphere, undefined);
  assert.equal(Object.getPrototypeOf(staged), Mesh.prototype);
  assert.equal(staged.intersectsFrustum, Mesh.prototype.intersectsFrustum);
  assert.equal(staged.isSkinnedMesh, false);
  assert.equal(staged.parent, stagedGroup); assert.equal(stagedGroup.parent, result.scene);
  assert.equal(staged.uuid, a.uuid); assert.equal(stagedGroup.renderOrder, 11);
  for (const field of ["material", "matrixWorld", "layers"]) assert.equal(staged[field], a[field]);
  for (const field of ["index", "groups", "drawRange"]) assert.equal(staged.geometry[field], a.geometry[field]);
  assert.equal(staged.geometry.attributes.color, a.geometry.attributes.color);
  assert.deepEqual(a.geometry.attributes.position.array, oldPositions);
  assert.equal(a.geometry.boundingSphere, oldBounds); assert.equal(a.isSkinnedMesh, true);
  assert.equal(a.parent, group); assert.equal(b.morphTargetInfluences[0], 0.5);
});

test("matrix boundaries run once and shared skeleton palettes update once before capture", () => {
  const { scene, camera, wasm, state } = setup(), a = mesh(), b = mesh();
  b.skeleton = a.skeleton; add(scene, a, b);
  prepare(scene, camera, wasm);
  assert.equal(scene.updates, 1); assert.equal(camera.updates, 1); assert.equal(a.skeleton.updates, 1);
  assert.equal(state.inputs[6][12], 1); assert.equal(state.inputs[6][28], 1);
});

test("manual pose mode and parented cameras do not trigger automatic updates", () => {
  const { scene, camera, wasm } = setup(), a = mesh(); add(scene, a);
  prepare(scene, camera, wasm, { autoUpdate: false });
  assert.equal(scene.updates, 0); assert.equal(camera.updates, 0); assert.equal(a.skeleton.updates, 0);
  scene.matrixWorldAutoUpdate = false; camera.parent = object();
  prepare(scene, camera, wasm);
  assert.equal(scene.updates, 0); assert.equal(camera.updates, 0); assert.equal(a.skeleton.updates, 1);
});

test("hidden ancestors and materials are skipped, but a parent's layer does not mask children", () => {
  const { scene, camera, wasm, state } = setup();
  const hidden = object(), bad = mesh(), masked = mesh(), invisibleMaterial = mesh(), visible = mesh();
  hidden.visible = false; bad.geometry = null; masked.layers.mask = 2; invisibleMaterial.material.visible = false;
  const group = object(); group.layers.mask = 2;
  add(scene, add(hidden, bad), masked, invisibleMaterial, add(group, visible));
  const result = prepare(scene, camera, wasm);
  assert.equal(result.skinnedMeshCount, 1); assert.equal(state.inputs[0].length, 4);
  assert.equal(bad.skeleton.updates, 0); assert.equal(masked.skeleton.updates, 0);
  assert.equal(invisibleMaterial.skeleton.updates, 0); assert.equal(visible.skeleton.updates, 1);
  assert.equal(result.scene.children.length, 4);
});

test("external hidden ancestry and empty scenes need no deformation export", async () => {
  const { scene, camera, host, state } = setup(); add(object(), scene); scene.parent.visible = false;
  const a = mesh(); add(scene, a);
  await render(host, scene, camera, null, {});
  assert.equal(state.nativeCalls, 0); assert.equal(a.skeleton.updates, 0); assert.equal(state.renderCalls, 1);
  const empty = object(); const result = await render(host, empty, camera, null, {});
  assert.equal(result.deformationVertexCount, 0); assert.equal(state.submissions.length, 2);
});

test("own render hooks and custom culling are not hidden by staging", () => {
  const { scene, camera, wasm, state } = setup(), a = mesh(); add(scene, a);
  a.onBeforeRender = () => {};
  assert.throws(() => prepare(scene, camera, wasm), refuses("UNSUPPORTED_CALLBACK"));
  assert.equal(state.nativeCalls, 0);
  delete a.onBeforeRender;
  a.intersectsFrustum = () => true;
  assert.throws(() => prepare(scene, camera, wasm), refuses("UNSUPPORTED_CALLBACK"));
  assert.equal(state.nativeCalls, 0);
});

test("unsupported scene state and visible non-mesh content stay visible to renderer admission", async () => {
  const { scene, camera, wasm, host, state } = setup();
  const light = object(); light.isLight = true; add(scene, mesh(), light);
  scene.fog = { density: 2 }; scene.background = { isColor: true, r: 1, g: 0, b: 0 };
  const result = await render(host, scene, camera, null, wasm);
  assert.equal(result.refused.length, 1); assert.equal(state.submissions.length, 0);
  assert.equal(state.staged.background, scene.background); assert.equal(state.staged.fog, scene.fog);
  assert.equal(state.nodes.find((n) => n.uuid === light.uuid).isLight, true);
});

test("canvas/offscreen targets and renderer result fields are preserved", async () => {
  const { scene, camera, wasm, host, state } = setup(); add(scene, mesh());
  const context = { canvas: { width: 80, height: 60 } };
  const result = await render(host, scene, camera, context, wasm, { sourceBackend: "webgpu", sortObjects: false });
  assert.equal(result.marker, 42); assert.equal(result.skinnedMeshCount, 1);
  assert.equal(state.submissions[0].context, context);
  assert.equal(state.options.target, "canvas"); assert.equal(state.options.sourceBackend, "webgpu");
  assert.equal(state.options.sortObjects, false); assert.equal(state.options.autoUpdate, false);
  await render(host, scene, camera, undefined, wasm, { width: 20, height: 30 });
  assert.equal(state.submissions[1].context, null); assert.equal(state.options.target, "offscreen");
});

test("invalid hosts, dimensions, modes and target contradictions reject before updates", async () => {
  const { scene, camera, wasm, host, state } = setup(); add(scene, mesh());
  await assert.rejects(render({}, scene, camera, null, wasm), refuses("DEFORMATION_HOST"));
  await assert.rejects(render(host, scene, camera, null, wasm, { width: 0 }), refuses("INVALID_DIMENSIONS"));
  await assert.rejects(render(host, scene, camera, null, wasm, { target: "canvas" }), refuses("DEFORMATION_TARGET"));
  await assert.rejects(render(host, scene, camera, null, wasm, { autoUpdate: "false" }), refuses("DEFORMATION_UPDATE_BOUNDARY"));
  assert.equal(state.nativeCalls, 0); assert.equal(scene.updates, 0); assert.equal(state.renderCalls, 0);
});

test("native exceptions and malformed output cannot submit a partial scene", async () => {
  const { scene, camera, wasm, host, state } = setup(); add(scene, mesh());
  const error = new Error("native failure"); wasm.f3d_deform_position_batch = () => { throw error; };
  await assert.rejects(render(host, scene, camera, null, wasm), (e) => e === error);
  for (const output of [new Float32Array(2), new Float32Array(9).fill(NaN), new Float64Array(9)]) {
    wasm.f3d_deform_position_batch = () => output;
    await assert.rejects(render(host, scene, camera, null, wasm), refuses("DEFORMATION_WASM_OUTPUT"));
  }
  assert.equal(state.renderCalls, 0); assert.equal(state.submissions.length, 0);
});

test("each pose starts from source geometry and owns native scratch before reuse", () => {
  const { scene, camera, wasm, state } = setup(); add(scene, mesh()); const scratch = new Float32Array(9);
  wasm.f3d_deform_position_batch = (_, positions) => { scratch.set(Float32Array.from(positions, (v) => v + state.shift)); return scratch; };
  const first = prepare(scene, camera, wasm); state.shift = 20;
  const second = prepare(scene, camera, wasm);
  assert.equal(first.scene.children[0].geometry.attributes.position.getX(0), 10);
  assert.equal(second.scene.children[0].geometry.attributes.position.getX(0), 20);
  assert.equal(scene.children[0].geometry.attributes.position.getX(0), 0);
});

test("invalid topology and budgets reject before recursive source updates", () => {
  const { scene, camera, wasm, state } = setup(); const a = object(); add(scene, a);
  a.children.push(scene);
  assert.throws(() => prepare(scene, camera, wasm), refuses("DEFORMATION_SCENE_GRAPH"));
  a.children.length = 0; a.parent = null;
  assert.throws(() => prepare(scene, camera, wasm), refuses("DEFORMATION_SCENE_GRAPH"));
  a.parent = scene;
  assert.throws(() => prepare(scene, camera, wasm, { maxSceneNodes: 1 }), refuses("DEFORMATION_SCENE_BUDGET"));
  assert.throws(() => prepare(scene, camera, wasm, { maxSceneDepth: -1 }), refuses("DEFORMATION_SCENE_BUDGET"));
  add(a, object());
  assert.throws(() => prepare(scene, camera, wasm, { maxSceneDepth: 1 }), refuses("DEFORMATION_SCENE_BUDGET"));
  assert.equal(scene.updates, 0); assert.equal(state.nativeCalls, 0);
});

test("frozen source objects and structural descriptors stay unchanged in manual mode", () => {
  const { scene, camera, wasm } = setup(), a = mesh(); add(scene, a);
  Object.freeze(a.children); Object.freeze(a); Object.freeze(scene.children); Object.freeze(scene); Object.freeze(camera);
  const result = prepare(scene, camera, wasm, { autoUpdate: false });
  assert.equal(result.scene.children[0].parent, result.scene);
  assert.equal(a.parent, scene); assert.equal(scene.children[0], a);
});

test("deformation limits and missing palettes remain enforced by production capture", () => {
  const { scene, camera, wasm, state } = setup(), a = mesh(); add(scene, a);
  assert.throws(() => prepare(scene, camera, wasm, { deformationLimits: { maxVertices: 2 } }), refuses("DEFORMATION_BUDGET"));
  a.skeleton.boneMatrices = null;
  assert.throws(() => prepare(scene, camera, wasm, { autoUpdate: false }), refuses("DEFORMATION_SKELETON"));
  assert.equal(state.nativeCalls, 0);
});

test("renderer failure propagates with source topology and geometry intact", async () => {
  const { scene, camera, wasm, host, state } = setup(), a = mesh(); add(scene, a);
  const error = new Error("host failure"); host.executePacket = async () => { throw error; };
  await assert.rejects(render(host, scene, camera, null, wasm), (e) => e === error);
  assert.equal(state.renderCalls, 1); assert.equal(a.parent, scene);
  assert.equal(a.geometry.attributes.position.getX(0), 0); assert.equal(a.isSkinnedMesh, true);
});
