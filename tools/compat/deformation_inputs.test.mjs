import assert from "node:assert/strict";
import { test } from "node:test";
import { captureDeformationBatch, evaluateDeformationBatch } from "./deformation_inputs.mjs";

// Minimal attribute transport fixtures. Native arithmetic is never mocked with
// a competing implementation: numerical tests live in Rust and the real-Wasm
// conformance fixture. These tests check the complete snapshot/wire boundary.
class Attribute {
  constructor(values, size, normalized = false) {
    this.array = ArrayBuffer.isView(values) ? values : new Float32Array(values);
    this.itemSize = size;
    this.count = this.array.length / size;
    this.normalized = normalized;
  }
  component(i, c) {
    const value = this.array[i * this.itemSize + c];
    return this.normalized ? value / 255 : value;
  }
  getX(i) {
    return this.component(i, 0);
  }
  getY(i) {
    return this.component(i, 1);
  }
  getZ(i) {
    return this.component(i, 2);
  }
  getW(i) {
    return this.component(i, 3);
  }
  onUploadCallback() {}
}
function matrix(x = 0) {
  return { elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1] };
}
function mesh(positions = [1, 2, 3]) {
  return {
    isMesh: true,
    geometry: {
      isBufferGeometry: true,
      attributes: { position: new Attribute(positions, 3) },
      morphAttributes: {},
    },
  };
}
function morph(relative = false) {
  const result = mesh();
  result.geometry.morphTargetsRelative = relative;
  result.geometry.morphAttributes.position = [new Attribute([4, 5, 6], 3)];
  result.morphTargetInfluences = [0.25];
  return result;
}
function skin() {
  const result = mesh();
  result.isSkinnedMesh = true;
  result.geometry.attributes.skinIndex = new Attribute(new Uint32Array([0, 0xffffffff, 0, 0]), 4);
  result.geometry.attributes.skinWeight = new Attribute([1, 0, 0, 0], 4);
  result.skeleton = {
    bones: [{}],
    boneMatrices: new Float32Array([...matrix(7).elements, ...Array(48).fill(NaN)]),
  };
  result.bindMatrix = matrix(2);
  result.bindMatrixInverse = matrix(-2);
  return result;
}
const code = (expected) => (error) => error.code === expected;

test("mixed rigid absolute relative and skinned meshes pack one ordered bank", () => {
  const a = morph(true),
    b = skin();
  b.geometry.morphAttributes.position = [new Attribute([8, 9, 10], 3)];
  b.morphTargetInfluences = [-0.5];
  const input = captureDeformationBatch([mesh([0, 0, 0]), a, b, morph(false)]);
  assert.deepEqual([...input.layout], [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 0]);
  assert.deepEqual([...input.positions], [0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3]);
  assert.deepEqual([...input.morphPositions], [4, 5, 6, 8, 9, 10, 4, 5, 6]);
  assert.deepEqual([...input.morphWeights], [0.25, -0.5, 0.25]);
  assert.deepEqual(
    input.rows.map((row) => row.positionOffset),
    [0, 3, 6, 9],
  );
  assert.equal(input.deformedMeshCount, 3);
  assert.equal(input.skinnedCount, 1);
});

test("one compiled call receives all eight banks and result ownership is isolated", () => {
  const input = captureDeformationBatch([mesh(), skin()]),
    result = new Float32Array([9, 8, 7, 6, 5, 4]);
  let calls = 0;
  const output = evaluateDeformationBatch(input, {
    f3d_deform_position_batch(...args) {
      calls++;
      assert.equal(args.length, 8);
      assert.equal(args[0], input.layout);
      assert.equal(args[7], input.bindMatrices);
      return result;
    },
  });
  assert.equal(calls, 1);
  assert.deepEqual(output, result);
  result[0] = 123;
  assert.equal(output[0], 9);
});

test("texture padding is excluded and per-mesh bind matrices stay distinct", () => {
  const a = skin(),
    b = skin();
  b.skeleton = a.skeleton;
  b.bindMatrix = matrix(9);
  const input = captureDeformationBatch([a, b]);
  assert.equal(input.jointPalettes.length, 32);
  assert.equal(input.jointPalettes[12], 7);
  assert.deepEqual(
    [12, 28, 44, 60].map((i) => input.bindMatrices[i]),
    [2, -2, 9, -2],
  );
  assert.equal(input.jointIndices[1], 0xffffffff);
});

test("interleaved base attributes are decoded by component getters", () => {
  const a = mesh();
  const array = new Float32Array([99, 2, 3, 4, 98, 5, 6, 7]);
  a.geometry.attributes.position = {
    isInterleavedBufferAttribute: true,
    data: { array, stride: 4 },
    count: 2,
    itemSize: 3,
    getX: (i) => array[i * 4 + 1],
    getY: (i) => array[i * 4 + 2],
    getZ: (i) => array[i * 4 + 3],
  };
  assert.deepEqual([...captureDeformationBatch([a]).positions], [2, 3, 4, 5, 6, 7]);
});

test("normalized integer skin weights decode without implicit renormalization", () => {
  const a = skin();
  a.geometry.attributes.skinWeight = new Attribute(new Uint8Array([255, 128, 0, 0]), 4, true);
  assert.deepEqual([...captureDeformationBatch([a]).jointWeights], [1, 128 / 255, 0, 0]);
});

test("normalized joint indices are refused instead of truncated", () => {
  const a = skin();
  a.geometry.attributes.skinIndex.normalized = true;
  assert.throws(() => captureDeformationBatch([a]), code("DEFORMATION_ATTRIBUTE"));
});

test("source attributes palettes weights matrices and versions remain untouched", () => {
  const a = skin();
  a.geometry.attributes.position.version = 9;
  a.skeleton.update = () => assert.fail("implicit update");
  a.updateMatrixWorld = () => assert.fail("implicit traversal");
  const before = structuredClone(a.skeleton.boneMatrices);
  const input = captureDeformationBatch([a]);
  input.positions[0] = 500;
  input.jointPalettes[0] = 500;
  input.bindMatrices[0] = 500;
  assert.equal(a.geometry.attributes.position.getX(0), 1);
  assert.equal(a.bindMatrix.elements[0], 1);
  assert.deepEqual(a.skeleton.boneMatrices, before);
  assert.equal(a.geometry.attributes.position.version, 9);
});

test("later frames read changed influences membership and palette residency", () => {
  const a = skin();
  captureDeformationBatch([a]);
  a.skeleton.bones.push({});
  a.skeleton.boneMatrices = new Float32Array([...matrix(8).elements, ...matrix(9).elements]);
  assert.equal(captureDeformationBatch([a]).layout[2], 2);
  const b = morph();
  b.morphTargetInfluences[0] = 1.5;
  assert.equal(captureDeformationBatch([b]).morphWeights[0], 1.5);
});

test("normal-only morphs do not shift following position weight banks", () => {
  const a = mesh();
  a.geometry.morphAttributes.normal = [new Attribute([0, 1, 0], 3)];
  a.morphTargetInfluences = [0.75];
  const input = captureDeformationBatch([a, morph()]);
  assert.deepEqual([...input.morphWeights], [0.25]);
  assert.equal(input.layout[1], 0);
});

test("morph normals may accompany position targets on the unlit route", () => {
  const a = morph();
  a.geometry.morphAttributes.normal = [new Attribute([0, 1, 0], 3)];
  assert.equal(captureDeformationBatch([a]).morphPositions.length, 3);
});

test("color morphs unknown families and mismatched counts reject", () => {
  for (const family of ["color", "custom"]) {
    const a = morph();
    a.geometry.morphAttributes[family] = [new Attribute([1, 1, 1], 3)];
    assert.throws(() => captureDeformationBatch([a]), code("DEFORMATION_MORPH"));
  }
  const a = morph();
  a.morphTargetInfluences = [];
  assert.throws(() => captureDeformationBatch([a]), code("DEFORMATION_MORPH"));
  const b = morph();
  b.geometry.morphAttributes.position[0] = new Attribute([1, 2, 3, 4, 5, 6], 3);
  assert.throws(() => captureDeformationBatch([b]), code("DEFORMATION_ATTRIBUTE"));
});

test("empty vertices still consume authored morph weights and bind banks", () => {
  const a = mesh([]);
  a.geometry.morphAttributes.position = [new Attribute([], 3)];
  a.morphTargetInfluences = [1];
  const input = captureDeformationBatch([a, morph()]);
  assert.deepEqual([...input.morphWeights], [1, 0.25]);
  assert.deepEqual([...input.layout], [0, 1, 0, 0, 1, 1, 0, 0]);
});

test("size and cumulative normal-only morph budgets reject before allocation", () => {
  assert.throws(
    () => captureDeformationBatch([mesh()], { maxVertices: 0 }),
    code("DEFORMATION_BUDGET"),
  );
  assert.throws(
    () => captureDeformationBatch([morph()], { maxMorphComponents: 2 }),
    code("DEFORMATION_BUDGET"),
  );
  assert.throws(
    () => captureDeformationBatch([skin()], { maxJoints: 0 }),
    code("DEFORMATION_BUDGET"),
  );
  const a = mesh([]);
  a.geometry.morphAttributes.normal = [new Attribute([], 3), new Attribute([], 3)];
  a.morphTargetInfluences = [1, 1];
  assert.throws(
    () => captureDeformationBatch([a, a], { maxMorphComponents: 3 }),
    code("DEFORMATION_BUDGET"),
  );
});

test("nonfinite inputs and fractional joint indices are refused", () => {
  const a = morph();
  a.morphTargetInfluences[0] = NaN;
  assert.throws(() => captureDeformationBatch([a]), code("DEFORMATION_MORPH"));
  const b = mesh([Infinity, 0, 0]);
  assert.throws(() => captureDeformationBatch([b]), code("DEFORMATION_VALUE"));
  const c = skin();
  c.geometry.attributes.skinIndex = new Attribute([0.5, 0, 0, 0], 4);
  assert.throws(() => captureDeformationBatch([c]), code("DEFORMATION_VALUE"));
  const d = skin();
  d.bindMatrix.elements[0] = NaN;
  assert.throws(() => captureDeformationBatch([d]), code("DEFORMATION_BIND"));
});

test("skin weights stay authored including negatives and zero-weight sentinels", () => {
  const a = skin();
  a.geometry.attributes.skinWeight = new Attribute([-1, 0, 2, 0], 4);
  assert.deepEqual([...captureDeformationBatch([a]).jointWeights], [-1, 0, 2, 0]);
});

test("incomplete skeleton palettes are not replaced by identities", () => {
  const a = skin();
  a.skeleton.boneMatrices = new Float32Array(15);
  assert.throws(() => captureDeformationBatch([a]), code("DEFORMATION_SKELETON"));
});

test("shared and detached source storage reject", () => {
  const a = mesh();
  a.geometry.attributes.position.array = new Float32Array(new SharedArrayBuffer(12));
  assert.throws(() => captureDeformationBatch([a]), code("DEFORMATION_ATTRIBUTE"));
  const b = skin();
  structuredClone(b.skeleton.boneMatrices.buffer, { transfer: [b.skeleton.boneMatrices.buffer] });
  assert.throws(() => captureDeformationBatch([b]), code("DEFORMATION_SKELETON"));
});

test("custom upload callbacks on consumed morph and skin inputs reject", () => {
  for (const choose of [
    (a) => a.geometry.attributes.position,
    (a) => a.geometry.attributes.skinWeight,
  ]) {
    const a = skin();
    choose(a).onUploadCallback = () => assert.fail("not a callback boundary");
    assert.throws(() => captureDeformationBatch([a]), code("UNSUPPORTED_UPLOAD_CALLBACK"));
  }
  const a = morph();
  a.geometry.morphAttributes.position[0].onUploadCallback = () => {};
  assert.throws(() => captureDeformationBatch([a]), code("UNSUPPORTED_UPLOAD_CALLBACK"));
});

test("missing native export and invalid native outputs never select host math", () => {
  const input = captureDeformationBatch([mesh()]);
  assert.throws(() => evaluateDeformationBatch(input, {}), code("DEFORMATION_MISSING_WASM"));
  for (const output of [
    new Float32Array(2),
    new Float64Array(3),
    new Float32Array([NaN, 0, 0]),
    Promise.resolve(new Float32Array(3)),
  ]) {
    assert.throws(
      () => evaluateDeformationBatch(input, { f3d_deform_position_batch: () => output }),
      code("DEFORMATION_WASM_OUTPUT"),
    );
  }
});

test("native failure propagates unchanged without touching source data", () => {
  const a = mesh(),
    input = captureDeformationBatch([a]),
    error = new Error("native failure");
  assert.throws(
    () =>
      evaluateDeformationBatch(input, {
        f3d_deform_position_batch() {
          throw error;
        },
      }),
    (e) => e === error,
  );
  assert.equal(a.geometry.attributes.position.getX(0), 1);
});

test("4096 meshes retain exact segmentation and one coarse native invocation", () => {
  const input = captureDeformationBatch(Array.from({ length: 4096 }, (_, i) => mesh([i, 0, 0])));
  let calls = 0;
  const output = evaluateDeformationBatch(input, {
    f3d_deform_position_batch(layout, positions) {
      calls++;
      return new Float32Array(positions);
    },
  });
  assert.equal(calls, 1);
  for (let i = 0; i < 4096; i++) assert.equal(output[i * 3], i);
});
