import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { skinPositions } from "./fixtures/numeric/linear_blend_skinning.mjs";
import { compileNumericCandidate } from "./numeric_candidate.mjs";
import { compileNumericKernel, NumericKernelCompileError } from "./numeric_kernel.mjs";
import { instantiateNumericKernel } from "./numeric_kernel_runtime.mjs";
import { specializeNumericModule } from "./numeric_specialization.mjs";

const type = (C) =>
  C === Float32Array
    ? "f32[]"
    : C === Float64Array
      ? "f64[]"
      : C === Uint16Array
        ? "u16[]"
        : "u32[]";
const same = (a, b) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    assert.ok(Object.is(a[i], b[i]), `element ${i}: ${a[i]} != ${b[i]}, including zero sign`);
};
function mesh(
  Storage = Float32Array,
  Index = Uint16Array,
  Matrix = Storage,
  count = 10000,
  Output = Storage,
) {
  const positions = Storage.from({ length: count * 3 }, (_, i) => ((i % 127) - 63) / 17);
  const joints = Index.from({ length: count * 4 }, (_, i) => (i * 7 + (i % 4)) % 64);
  const weights = Storage.from({ length: count * 4 }, (_, i) => [0.1, 0.2, 0.3, 0.4][i % 4]);
  const matrices = new Matrix(64 * 16),
    output = new Output(count * 3);
  pose(matrices, 0);
  return [positions, joints, weights, matrices, output];
}
function pose(matrices, frame) {
  for (let joint = 0; joint < 64; joint++) {
    const b = joint * 16,
      angle = (joint + frame) / 83,
      c = Math.cos(angle),
      s = Math.sin(angle);
    matrices[b] = c;
    matrices[b + 1] = s;
    matrices[b + 4] = -s;
    matrices[b + 5] = c;
    matrices[b + 10] = 1;
    matrices[b + 12] = ((joint % 7) - 3) / 11;
    matrices[b + 13] = frame / 90;
    matrices[b + 14] = ((joint % 3) - 1) / 13;
    matrices[b + 15] = 1;
  }
}
function build(args) {
  return compileNumericCandidate(skinPositions.toString(), {
    parameterTypes: args.map((value) => type(value.constructor)),
    sourceName: "skinning.mjs",
  });
}

for (const Storage of [Float32Array, Float64Array])
  for (const Index of [Uint16Array, Uint32Array])
    for (const Matrix of [Float32Array, Float64Array]) {
      test(`four-influence skinning, 10k vertices and 30 poses: ${Storage.name}/${Index.name}/${Matrix.name}`, () => {
        const args = mesh(Storage, Index, Matrix),
          artifact = build(args);
        assert.equal(artifact.manifest.version, 7);
        assert.equal(artifact.fixedLoops.expandedIterations, 4);
        assert.equal(artifact.fixedLoops.loops[0].iterations, 4);
        assert.ok(Object.isFrozen(artifact.fixedLoops.loops[0].sourceSpan));
        assert.match(artifact.manifest.sourceName, /#fixed-loop-expansion$/);
        const module = new WebAssembly.Module(artifact.wasm);
        assert.deepEqual(WebAssembly.Module.imports(module), []);
        const kernel = instantiateNumericKernel(artifact.wasm),
          expected = args[4].slice();
        for (let frame = 0; frame < 30; frame++) {
          pose(args[3], frame);
          skinPositions(...args.slice(0, 4), expected);
          assert.equal(kernel.run(...args), undefined);
          same(args[4], expected);
        }
        assert.equal(kernel.diagnostics.wasmCalls, 30);
        assert.equal(kernel.diagnostics.fallbackCalls, 0);
        assert.equal(
          kernel.diagnostics.copiedBytes,
          30 * (args.slice(0, 4).reduce((n, a) => n + a.byteLength, 0) + args[4].byteLength * 2),
        );
      });
    }

test("late invalid joints discard all scratch writes and preserve one original invocation", () => {
  const args = mesh(Float32Array, Uint16Array, Float32Array, 100),
    expected = args[4].slice();
  let calls = 0;
  const receiver = {};
  const kernel = instantiateNumericKernel(build(args).wasm, {
    fallback(...values) {
      calls++;
      assert.equal(this, receiver);
      return skinPositions(...values);
    },
  });
  args[1][399] = 65535;
  skinPositions(...args.slice(0, 4), expected);
  kernel.run.call(receiver, ...args);
  same(args[4], expected);
  assert.equal(calls, 1);
  assert.equal(kernel.diagnostics.wasmCalls, 0);
  assert.equal(kernel.diagnostics.copiedBytes, 0);
  args[1][399] = 1;
  skinPositions(...args.slice(0, 4), expected);
  kernel.run(...args);
  same(args[4], expected);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
  assert.equal(calls, 1);
});

test("unused influences, repeated joints, zero weights and IEEE values follow source behavior", () => {
  const args = mesh(Float64Array, Uint32Array, Float64Array, 4),
    expected = args[4].slice();
  args[0].set([
    0,
    -0,
    NaN,
    Infinity,
    -Infinity,
    1,
    Number.MIN_VALUE,
    -Number.MIN_VALUE,
    0,
    1,
    2,
    3,
  ]);
  args[1].fill(0);
  args[1][15] = 4294967295;
  args[2][15] = 0;
  const kernel = instantiateNumericKernel(build(args).wasm);
  skinPositions(...args.slice(0, 4), expected);
  kernel.run(...args);
  same(args[4], expected);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test("aliased input/output and short weights retain original semantics; empty meshes execute", () => {
  const args = mesh(Float64Array, Uint16Array, Float64Array, 3);
  const kernel = instantiateNumericKernel(build(args).wasm, { fallback: skinPositions });
  const expected = args[0].slice();
  skinPositions(expected, ...args.slice(1, 4), expected);
  kernel.run(...args.slice(0, 4), args[0]);
  same(args[0], expected);
  assert.equal(kernel.diagnostics.lastGuardFailure, "KERNEL_ARRAY_ALIAS");
  const short = args[2].subarray(0, 1),
    out = args[4].slice();
  skinPositions(args[0], args[1], short, args[3], out);
  kernel.run(args[0], args[1], short, args[3], args[4]);
  same(args[4], out);
  assert.equal(kernel.diagnostics.fallbackCalls, 2);
  const empty = mesh(Float64Array, Uint16Array, Float64Array, 0);
  kernel.run(...empty);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test("fixed neighborhood reduction and nested component loops use checked array addresses", () => {
  function neighborhood(values, out) {
    for (let i = 0; i < values.length; i++) {
      let sum = 0;
      for (let offset = -1; offset <= 1; offset++) {
        if (i + offset >= 0 && i + offset < values.length) sum += values[i + offset];
      }
      out[i] = sum;
    }
  }
  const input = new Float64Array([1, 2, 3, 4]),
    actual = new Float64Array(4),
    expected = actual.slice();
  const a = compileNumericCandidate(neighborhood.toString(), {
    parameterTypes: ["f64[]", "f64[]"],
  });
  instantiateNumericKernel(a.wasm).run(input, actual);
  neighborhood(input, expected);
  same(actual, expected);
  function components(values) {
    for (let i = 0; i < values.length; i += 16) {
      for (let row = 0; row < 4; row++)
        for (let col = 0; col < 4; col++) {
          values[i + row * 4 + col] += 1 / 3;
          values[i + row * 4 + col] *= 0.7;
        }
    }
  }
  const b = compileNumericCandidate(components.toString(), { parameterTypes: ["f32[]"] });
  const x = new Float32Array(32).fill(1 / 7),
    y = x.slice();
  instantiateNumericKernel(b.wasm).run(x);
  components(y);
  same(x, y);
  assert.equal(b.fixedLoops.expandedIterations, 20);
});

test("outer-loop state, scalar helper calls and individual Float32 stores remain ordered", () => {
  function twice(x) {
    return x + x;
  }
  function update(a) {
    let total = 0;
    for (let i = 0; i < a.length; i++) {
      for (let j = 3; j >= 0; j--) {
        a[i] = twice(a[i]) + j;
        total += a[i];
      }
    }
    return total;
  }
  const artifact = compileNumericCandidate(update.toString(), {
    parameterTypes: ["f32[]"],
    helperSources: new Map([["twice", twice.toString()]]),
  });
  const a = new Float32Array([0, -0, 1 / 3, 1e30]),
    expected = a.slice(),
    kernel = instantiateNumericKernel(artifact.wasm);
  assert.ok(Object.is(kernel.run(a), update(expected)));
  same(a, expected);
  assert.deepEqual(artifact.helpers, [{ name: "twice", arity: 1 }]);
});

test("unsupported operations and unbounded/mutable inner control retain the whole original function", () => {
  for (const body of [
    "break;",
    "continue;",
    "j+=1;",
    "a[i]=Math.sin(a[i]);",
    "a[i]=external();",
    "var state=1;",
  ]) {
    const source = `function f(a){for(let i=0;i<a.length;i++){for(let j=0;j<4;j++){${body}}a[i]+=1;}}`;
    assert.throws(
      () => compileNumericCandidate(source, { parameterTypes: ["f64[]"], allowMath: true }),
      NumericKernelCompileError,
    );
    const module = source + "\nexport function frame(a){return f(a);}";
    assert.equal(specializeNumericModule(module).code, module);
  }
  const dynamic = "function f(a,n){for(let i=0;i<a.length;i++){for(let j=0;j<n;j++)a[i]+=1;}}";
  assert.throws(
    () => compileNumericCandidate(dynamic, { parameterTypes: ["f64[]", "f64"] }),
    NumericKernelCompileError,
  );
  assert.throws(
    () =>
      compileNumericCandidate(skinPositions.toString(), {
        parameterTypes: ["f64[]", "u16[]", "f64[]", "f64[]", "f64[]"],
        checkedIndexing: false,
      }),
    { code: "INVALID_KERNEL_ABI" },
  );
});

async function generated(source) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-fixed-skinning-"));
  const runtimeModule = new URL("./numeric_dispatch.mjs", import.meta.url).href;
  const result = specializeNumericModule(source, { runtimeModule });
  const tokens = [...result.code.matchAll(/var (__f3d_numeric_token_\d+) =/g)].map((m) => m[1]);
  const observed =
    result.code +
    `\nimport {numericDispatchDiagnostics as __testStats} from ${JSON.stringify(runtimeModule)};\nexport const __stats=()=>[${tokens.map((name) => `__testStats(${name})`).join(",")}];`;
  const file = path.join(root, "app.mjs");
  fs.writeFileSync(file, observed);
  return { result, module: await import(pathToFileURL(file).href) };
}

test("ordinary application calls automatically select eight native input/output/joint layouts", async () => {
  const source =
    skinPositions.toString() +
    "\nexport {skinPositions};export const original=skinPositions;export function frame(...args){return skinPositions(...args);}";
  const { result, module } = await generated(source);
  assert.equal(result.report.compiledKernels, 1);
  assert.equal(result.report.rewrittenCalls, 1);
  assert.equal(module.original, module.skinPositions);
  assert.equal(module.skinPositions.toString(), skinPositions.toString());
  assert.equal(module.__stats()[0].initialized, false);
  for (const Input of [Float32Array, Float64Array])
    for (const Output of [Float32Array, Float64Array])
      for (const Index of [Uint16Array, Uint32Array]) {
        const args = mesh(Input, Index, Input, 1000, Output),
          expected = args[4].slice();
        for (let frame = 0; frame < 12; frame++) {
          pose(args[3], frame);
          module.skinPositions(...args.slice(0, 4), expected);
          module.frame(...args);
          same(args[4], expected);
        }
        assert.equal(module.__stats()[0].kernel.wasmCalls, 12);
        assert.equal(module.__stats()[0].kernel.fallbackCalls, 0);
      }
  assert.equal(module.__stats()[0].variants.filter((v) => v.initialized).length, 8);
});

test("original argument order, shadowed callees and no-Wasm behavior survive specialization", async () => {
  const source =
    skinPositions.toString() +
    `\nexport function run(log,...args){return skinPositions((log.push('first'),args[0]),...args.slice(1));}
    export function shadow(skinPositions,...args){return skinPositions(...args);}`;
  const { module } = await generated(source),
    args = mesh(Float32Array, Uint16Array, Float32Array, 2),
    log = [];
  const saved = globalThis.WebAssembly;
  try {
    globalThis.WebAssembly = undefined;
    module.run(log, ...args);
  } finally {
    globalThis.WebAssembly = saved;
  }
  assert.deepEqual(log, ["first"]);
  assert.equal(module.__stats()[0].retainedCalls, 1);
  assert.equal(
    module.shadow((...values) => values.length, ...args),
    5,
  );
  assert.equal(module.__stats()[0].identityMisses, 1);
});

test("legacy accepted kernels stay byte-identical, without new lowering metadata", () => {
  const cases = [
    ["function f(a){for(let i=0;i<a.length;i++)a[i]+=1;}", ["f64[]"]],
    ["function f(a){let sum=0;for(let i=0;i<a.length;i++)sum+=a[i];return sum;}", ["f32[]"]],
    ["function f(a,b){for(let i=0;i<a.length;i++)a[i]=b[i];}", ["f32[]", "f64[]"]],
    ["function f(a){for(let i=0;i<a.length;i+=3)a[i+2]*=2;}", ["f64[]"]],
    [
      "function f(a){for(let i=0;i<a.length;i++)a[i]+=1;for(let i=0;i<a.length;i++)a[i]*=2;}",
      ["f32[]"],
    ],
    ["function f(a,index){for(let i=0;i<index.length;i++)a[index[i]]+=1;}", ["f64[]", "u16[]"]],
  ];
  for (const [source, parameterTypes] of cases) {
    const checkedIndexing = parameterTypes.includes("u16[]");
    const direct = compileNumericKernel(source, { parameterTypes, checkedIndexing });
    const candidate = compileNumericCandidate(source, { parameterTypes });
    assert.deepEqual(candidate.wasm, direct.wasm);
    assert.deepEqual(candidate.manifest, direct.manifest);
    assert.equal(candidate.fixedLoops, undefined);
  }
});

test("standalone skinning package relocates, executes native code and keeps no-Wasm fallback", async () => {
  const { buildNumericKernel } = await import("./numeric_kernel_build.mjs");
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-skinning-package-"));
  const source = path.join(root, "skin.mjs"),
    destination = path.join(root, "package"),
    relocated = path.join(root, "moved");
  fs.writeFileSync(source, "export " + skinPositions.toString());
  const manifest = buildNumericKernel(source, destination, {
    parameterTypes: ["f32[]", "u16[]", "f32[]", "f32[]", "f32[]"],
  });
  fs.renameSync(destination, relocated);
  const module = await import(pathToFileURL(path.join(relocated, "kernel.mjs")).href);
  const args = mesh(Float32Array, Uint16Array, Float32Array, 1000),
    expected = args[4].slice();
  const kernel = module.createKernel();
  for (let frame = 0; frame < 12; frame++) {
    pose(args[3], frame);
    module.retained(...args.slice(0, 4), expected);
    kernel.run(...args);
    same(args[4], expected);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 12);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
  assert.equal(fs.readdirSync(relocated).length, 5);
  assert.match(manifest.kernel.sourceName, /#fixed-loop-expansion$/);
  const saved = globalThis.WebAssembly;
  let fallback;
  try {
    globalThis.WebAssembly = undefined;
    fallback = module.createKernel();
  } finally {
    globalThis.WebAssembly = saved;
  }
  fallback.run(...args);
  same(args[4], expected);
  assert.equal(fallback.diagnostics.fallbackCalls, 1);
  assert.throws(
    () =>
      buildNumericKernel(source, relocated, {
        parameterTypes: ["f32[]", "u16[]", "f32[]", "f32[]", "f32[]"],
      }),
    { code: "EEXIST" },
  );
});
