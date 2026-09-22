import assert from "node:assert/strict";
import { test } from "node:test";
import { compileNumericKernel } from "./numeric_kernel.mjs";
import { instantiateNumericKernel } from "./numeric_kernel_runtime.mjs";
import { specializeNumericModule } from "./numeric_specialization.mjs";

function transform(position, matrix) {
  for (let i = 0; i < position.length; i += 3) {
    const x = position[i],
      y = position[i + 1],
      z = position[i + 2];
    position[i] = x * matrix[i] + y;
    position[i + 1] = y + z * matrix[i + 1];
    position[i + 2] = z - x * matrix[i + 2];
  }
}
function compiled(fn, parameterTypes) {
  const artifact = compileNumericKernel(fn.toString(), { parameterTypes });
  assert.equal(WebAssembly.validate(artifact.wasm), true);
  return instantiateNumericKernel(artifact.wasm, { fallback: fn });
}

test("vec3 in-place transforms retain source read/store ordering in mixed-width storage", () => {
  const run = compiled(transform, ["f32[]", "f64[]"]);
  assert.equal(run.manifest.version, 3);
  assert.equal(run.manifest.loopStride, 3);
  const position = Float32Array.from({ length: 3000 }, (_, i) => ((i % 23) - 11) / 7);
  const matrix = Float64Array.from({ length: 3000 }, (_, i) => ((i % 7) - 3) / 100);
  const expected = position.slice();
  for (let frame = 0; frame < 120; frame++) {
    run.run(position, matrix);
    transform(expected, matrix);
    assert.deepEqual(position, expected);
  }
  assert.equal(run.diagnostics.wasmCalls, 120);
  assert.equal(run.diagnostics.fallbackCalls, 0);
});

test("partial-channel writes preserve untouched RGBA channels across calls", () => {
  function update(color, alpha) {
    for (let i = 0; i < color.length; i += 4) {
      color[i + 3] = alpha;
    }
  }
  const run = compiled(update, ["f32[]", "f64"]);
  for (let frame = 0; frame < 20; frame++) {
    const a = new Float32Array([frame, 2, 3, 1, 4, 5, 6, 1]),
      expected = a.slice();
    run.run(a, 1 / 3);
    update(expected, 1 / 3);
    assert.deepEqual(a, expected);
  }
  assert.equal(run.manifest.parameters[0].read, true);
  assert.equal(run.diagnostics.wasmCalls, 20);
});

test("each store remains visible to later offset reads in the same record", () => {
  function update(a) {
    for (let i = 0; i < a.length; i += 3) {
      a[i] += 1;
      a[i + 1] = a[i] - 16777216;
      a[i + 2] = a[i + 1] + 2;
    }
  }
  const run = compiled(update, ["f32[]"]);
  const a = new Float32Array([16777216, 9, 9]);
  run.run(a);
  assert.deepEqual([...a], [16777216, 0, 2]);
  assert.equal(run.diagnostics.wasmCalls, 1);
});

test("incomplete final records fall back with original ignored-store/undefined-read behavior", () => {
  function update(a) {
    for (let i = 0; i < a.length; i += 3) {
      a[i] = a[i + 1] + a[i + 2];
      a[i + 2] = 100;
    }
  }
  const run = compiled(update, ["f32[]"]);
  for (const length of [1, 2, 4, 5, 7]) {
    const a = new Float32Array(length).fill(1),
      expected = a.slice();
    run.run(a);
    update(expected);
    assert.deepEqual(a, expected);
    assert.equal(run.diagnostics.lastGuardFailure, "KERNEL_LOOP_EXTENT");
  }
  assert.equal(run.diagnostics.wasmCalls, 0);
  assert.equal(run.diagnostics.fallbackCalls, 5);
});

test("constant record offsets outside the stride and cross-record dependencies remain refused", () => {
  for (const offset of ["3", "-1", "1.5", "99"]) {
    const source = `function update(a) { for(let i=0;i<a.length;i+=3) { a[i]=a[i+${offset}]; } }`;
    assert.throws(
      () => compileNumericKernel(source, { parameterTypes: ["f32[]"] }),
      /stride|offset/,
    );
  }
  for (const step of ["0", "-1", "17", "dt", "1.5"]) {
    const source = `function update(a,dt) { for(let i=0;i<a.length;i+=${step}) { a[i]=dt; } }`;
    assert.throws(
      () => compileNumericKernel(source, { parameterTypes: ["f32[]", "f64"] }),
      /stride/,
    );
  }
});

test("sixteen-component records encode high byte offsets without signed-immediate errors", () => {
  function update(a, factor) {
    for (let i = 0; i < a.length; i += 16) {
      a[i + 15] = a[i + 8] * factor;
    }
  }
  for (const [ArrayType, type] of [
    [Float64Array, "f64[]"],
    [Float32Array, "f32[]"],
  ]) {
    const run = compiled(update, [type, "f64"]);
    const a = ArrayType.from({ length: 32 }, (_, i) => i + 1),
      expected = a.slice();
    run.run(a, 2);
    update(expected, 2);
    assert.deepEqual(a, expected);
    assert.equal(run.diagnostics.wasmCalls, 1);
  }
});

test("conditional strided writes and empty arrays preserve source behavior", () => {
  function update(a) {
    for (let i = 0; i < a.length; i += 2) {
      if (a[i] > 0) a[i + 1] = -a[i];
    }
  }
  const run = compiled(update, ["f64[]"]);
  for (const values of [[], [1, 9, -1, 7], [0, 3, 2, 4]]) {
    const a = new Float64Array(values),
      expected = a.slice();
    run.run(a);
    update(expected);
    assert.deepEqual(a, expected);
  }
  assert.equal(run.diagnostics.wasmCalls, 3);
});

test("ordinary call-site specialization discovers vec3 kernels and both storage variants", () => {
  const source = `${transform.toString()}\nexport function tick(a, matrix) { return transform(a, matrix); }`;
  const result = specializeNumericModule(source);
  assert.equal(result.report.compiledKernels, 1);
  assert.equal(result.report.candidates[0].loopStride, 3);
  assert.equal(result.report.candidates[0].variants.length, 2);
});
