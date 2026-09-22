import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Exercise the production adapter with packet/Three constructor contract doubles.
// Only import specifiers are substituted; the adapter and input capture execute
// unchanged. This suite tests wiring/ownership, NOT native math or GPU pixels.
const key = Symbol.for("f3d.deformed-mesh-test-contract");
const dependency = `
export class BufferAttribute {
  constructor(array,itemSize){this.array=array;this.itemSize=itemSize;this.count=array.length/itemSize;}
  getX(i){return this.array[i*this.itemSize];} getY(i){return this.array[i*this.itemSize+1];}
  getZ(i){return this.array[i*this.itemSize+2];} getW(i){return this.array[i*this.itemSize+3];}
}
const state=()=>globalThis[Symbol.for('f3d.deformed-mesh-test-contract')];
export function canAdmitMesh(mesh,camera,options){return state().admit(mesh,camera,options);}
export function prepareMeshBatchPacket(...args){return state().prepare(...args);}
export function createAdmissionError(code,detail=''){const error=new Error(code+': '+detail);error.reason=code;return error;}
`;
const dependencyUrl = "data:text/javascript;base64," + Buffer.from(dependency).toString("base64");
let source = await readFile(new URL("./deformed_mesh_adapter.mjs", import.meta.url), "utf8");
for (const specifier of ["../../upstream/three.js/build/three.module.js", "./mesh_adapter.mjs"]) {
  assert.ok(source.includes(`'${specifier}'`));
  source = source.replace(`'${specifier}'`, `'${dependencyUrl}'`);
}
source = source.replace(
  "'./deformation_inputs.mjs'",
  JSON.stringify(new URL("./deformation_inputs.mjs", import.meta.url).href),
);
const { prepareDeformedMeshBatchPacket: prepare, renderDeformedMeshBatch: render } = await import(
  "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);
const { BufferAttribute } = await import(dependencyUrl);
const matrix = () => ({ elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
function mesh() {
  return {
    isMesh: true,
    isSkinnedMesh: true,
    visible: true,
    material: { isMeshBasicMaterial: true, side: 2, depthTest: true, colorWrite: false },
    matrixWorld: matrix(),
    renderOrder: 7,
    layers: { mask: 1 },
    parent: { visible: true },
    geometry: {
      isBufferGeometry: true,
      attributes: {
        position: new BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3),
        skinIndex: new BufferAttribute(new Uint32Array(12), 4),
        skinWeight: new BufferAttribute(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 4),
        color: new BufferAttribute(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 3),
      },
      index: new BufferAttribute(new Uint16Array([0, 1, 2]), 1),
      drawRange: { start: 0, count: 3 },
      groups: [],
      morphAttributes: {},
      boundingSphere: { radius: 1 },
      boundingBox: { min: 0 },
    },
    skeleton: { bones: [{}], boneMatrices: new Float32Array(matrix().elements) },
    bindMatrix: matrix(),
    bindMatrixInverse: matrix(),
  };
}
function setup() {
  const state = {
    nativeCalls: 0,
    packetCalls: 0,
    submissions: [],
    admitted: [],
    packet: new Uint8Array([70, 51, 68, 80, ...Array(12).fill(0)]),
  };
  state.admit = (source) => {
    state.admitted.push(source);
    return Object.hasOwn(source, "onBeforeRender")
      ? { admitted: false, code: "UNSUPPORTED_CALLBACK" }
      : { admitted: true };
  };
  state.prepare = (meshes, camera, width, height, wasm, options) => {
    state.packetCalls++;
    state.draws = meshes;
    state.options = options;
    state.width = width;
    state.height = height;
    return {
      packetBytes: state.packet,
      snapshots: meshes.map((m) => ({ positions: m.geometry.attributes.position.array })),
      meshCount: meshes.length,
      target: options.target === "canvas" ? "canvas" : "offscreen",
    };
  };
  const wasm = {
    f3d_deform_position_batch(layout, positions) {
      state.nativeCalls++;
      return Float32Array.from(positions, (x) => x + 2);
    },
  };
  const host = {
    async executePacket(bytes, context) {
      state.submissions.push({ bytes, context });
      return "submitted";
    },
  };
  globalThis[key] = state;
  return { state, wasm, host };
}

test("deformed positions reach one packet while original geometry stays immutable", () => {
  const { state, wasm } = setup(),
    a = mesh(),
    b = mesh();
  const original = a.geometry.attributes.position.array.slice();
  const result = prepare([a, b], {}, 32, 48, wasm, {
    sourceBackend: "webgl",
    clearColor: [0, 0, 0, 1],
  });
  assert.equal(state.nativeCalls, 1);
  assert.equal(state.packetCalls, 1);
  assert.equal(result.skinnedMeshCount, 2);
  assert.equal(result.deformationVertexCount, 6);
  assert.equal(state.draws[0].geometry.attributes.position.getX(0), 2);
  assert.deepEqual(a.geometry.attributes.position.array, original);
  assert.equal(a.isSkinnedMesh, true);
  assert.equal(state.draws[0].isSkinnedMesh, false);
  assert.equal(state.draws[0].geometry.boundingSphere, null);
});

test("draw ranges indices colors depth/cull material and transforms are preserved", () => {
  const { state, wasm } = setup(),
    a = mesh();
  prepare([a], {}, 32, 32, wasm);
  const view = state.draws[0];
  for (const field of ["material", "matrixWorld", "layers", "parent", "renderOrder"])
    assert.equal(view[field], a[field]);
  for (const field of ["index", "drawRange", "groups"])
    assert.equal(view.geometry[field], a.geometry[field]);
  assert.equal(view.geometry.attributes.color, a.geometry.attributes.color);
  assert.equal(view.material.colorWrite, false);
  assert.equal(view.material.side, 2);
  assert.equal(view.material.depthTest, true);
});

test("own callback overrides survive staging and reject before native evaluation", () => {
  const { state, wasm } = setup(),
    a = mesh();
  a.onBeforeRender = () => {};
  assert.throws(
    () => prepare([a], {}, 32, 32, wasm),
    (e) => e.reason === "UNSUPPORTED_CALLBACK",
  );
  assert.equal(state.nativeCalls, 0);
  assert.equal(state.packetCalls, 0);
});

test("material/camera admission remains delegated rather than silently weakened", () => {
  const { state, wasm } = setup();
  state.admit = () => ({ admitted: false, code: "UNSUPPORTED_MATERIAL" });
  assert.throws(
    () => prepare([mesh()], {}, 32, 32, wasm),
    (e) => e.reason === "UNSUPPORTED_MATERIAL",
  );
  assert.equal(state.nativeCalls, 0);
});

test("explicit source update boundaries and dimensions reject before native work", () => {
  const { state, wasm } = setup();
  assert.throws(
    () => prepare([mesh()], {}, 32, 32, wasm, { autoUpdate: true }),
    (e) => e.reason === "DEFORMATION_UPDATE_BOUNDARY",
  );
  assert.throws(
    () => prepare([mesh()], {}, 0, 32, wasm),
    (e) => e.reason === "INVALID_DIMENSIONS",
  );
  assert.equal(state.nativeCalls, 0);
});

test("visible canvas renders submit the prepared packet to the provided context", async () => {
  const { state, wasm, host } = setup(),
    context = { canvas: { width: 80, height: 60 } };
  const result = await render([mesh()], {}, host, wasm, context, { sourceBackend: "webgpu" });
  assert.equal(result.result, "submitted");
  assert.equal(result.target, "canvas");
  assert.equal(state.width, 80);
  assert.equal(state.height, 60);
  assert.equal(state.submissions[0].context, context);
  assert.equal(state.submissions[0].bytes, state.packet);
  assert.equal(state.options.autoUpdate, false);
});

test("offscreen submission remains offscreen and contradictory targets reject", async () => {
  const { state, wasm, host } = setup();
  await render([mesh()], {}, host, wasm, null, { width: 20, height: 30 });
  assert.equal(state.options.target, "offscreen");
  assert.equal(state.submissions[0].context, null);
  await assert.rejects(
    render([mesh()], {}, host, wasm, null, { target: "canvas" }),
    (e) => e.reason === "DEFORMATION_TARGET",
  );
  assert.equal(state.submissions.length, 1);
});

test("native failure cannot produce or submit a partial draw batch", async () => {
  const { state, wasm, host } = setup(),
    error = new Error("native");
  wasm.f3d_deform_position_batch = () => {
    throw error;
  };
  await assert.rejects(render([mesh()], {}, host, wasm), (e) => e === error);
  assert.equal(state.packetCalls, 0);
  assert.equal(state.submissions.length, 0);
});

test("packet and host failures propagate without mutating source state", async () => {
  const { state, wasm, host } = setup(),
    a = mesh(),
    error = new Error("packet");
  state.prepare = () => {
    throw error;
  };
  await assert.rejects(render([a], {}, host, wasm), (e) => e === error);
  assert.equal(state.submissions.length, 0);
  assert.equal(a.geometry.attributes.position.getX(0), 0);
  setup();
  host.executePacket = async () => {
    throw error;
  };
  await assert.rejects(render([a], {}, host, wasm), (e) => e === error);
});

test("each new pose is evaluated from original base arrays without accumulated deformation", () => {
  const { state, wasm } = setup(),
    a = mesh();
  const first = prepare([a], {}, 32, 32, wasm);
  const second = prepare([a], {}, 32, 32, wasm);
  assert.deepEqual(first.snapshots[0].positions, second.snapshots[0].positions);
  assert.equal(state.nativeCalls, 2);
});
