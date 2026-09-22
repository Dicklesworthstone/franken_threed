import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { recomputeNormals } from "./fixtures/numeric/indexed_normals.mjs";
import { numericDispatchDiagnostics } from "./numeric_dispatch.mjs";
import { numericKernelRollupPlugin } from "./numeric_rollup.mjs";
import { specializeNumericModule } from "./numeric_specialization.mjs";

const runtime = new URL("./numeric_dispatch.mjs", import.meta.url).href;
const same = (a, b) => assert.deepEqual([...a], [...b]);
async function application(source, options = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-indexed-app-"));
  // Observe the actual production dispatcher without modifying generated exports.
  fs.writeFileSync(
    path.join(root, "observe.mjs"),
    `import * as runtime from ${JSON.stringify(runtime)};
export const tokens=[];
export function createNumericDispatch(...args){const token=runtime.createNumericDispatch(...args);tokens.push(token);return token;}
export const dispatchNumericCall=runtime.dispatchNumericCall;`,
    { flag: "wx" },
  );
  const transformed = specializeNumericModule(source, {
    ...options,
    runtimeModule: "./observe.mjs",
  });
  fs.writeFileSync(path.join(root, "app.mjs"), transformed.code, { flag: "wx" });
  const app = await import(pathToFileURL(path.join(root, "app.mjs")).href);
  const { tokens } = await import(pathToFileURL(path.join(root, "observe.mjs")).href);
  return {
    app,
    transformed,
    diagnostics: (index = 0) => numericDispatchDiagnostics(tokens[index]),
  };
}
const normalsSource = `${recomputeNormals.toString()}
export { recomputeNormals };
export const reference = recomputeNormals;
export function frame(p,index,n){return recomputeNormals(p,index,n);}`;

for (const Input of [Float32Array, Float64Array])
  for (const Output of [Float32Array, Float64Array])
    for (const Index of [Uint16Array, Uint32Array]) {
      test(`ordinary frame call executes indexed normals in Wasm: ${Input.name}/${Output.name}/${Index.name}`, async () => {
        const { app, transformed, diagnostics } = await application(normalsSource);
        assert.equal(app.reference, app.recomputeNormals);
        assert.equal(diagnostics().initialized, false);
        assert.equal(transformed.report.compiledKernels, 1);
        assert.equal(transformed.report.rewrittenCalls, 1);
        const item = transformed.report.candidates[0];
        assert.equal(item.indexSemantics, "checked-integer-full-view-v1");
        assert.deepEqual(item.lengthParameters, [0, 1, 2]);
        assert.equal(item.loopCount, 3);
        assert.equal(item.variants.length, 12);
        assert.equal(transformed.report.accelerated, false);
        const p = new Input([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1]),
          n = new Output(12),
          expected = n.slice();
        const index = new Index([0, 1, 2, 0, 2, 3]);
        for (let frame = 0; frame < 12; frame++) {
          p[2] += 0.125;
          assert.equal(app.frame(p, index, n), recomputeNormals(p, index, expected));
          same(n, expected);
        }
        assert.equal(diagnostics().kernel.wasmCalls, 12);
        assert.equal(diagnostics().kernel.fallbackCalls, 0);
        assert.equal(diagnostics().variants.filter((variant) => variant.initialized).length, 1);
      });
    }

test("one call site switches topology and float storage without eagerly compiling other variants", async () => {
  const { app, diagnostics } = await application(`
function scatter(a,b,index){for(let i=0;i<index.length;i++)a[index[i]]+=b[i];}
export function run(a,b,index){return scatter(a,b,index);}`);
  for (const Float of [Float32Array, Float64Array])
    for (const Index of [Uint16Array, Uint32Array]) {
      const a = new Float([0, 0]),
        b = new Float([3, 4, 5]),
        index = new Index([0, 1, 0]);
      app.run(a, b, index);
      same(a, [8, 4]);
      assert.equal(diagnostics().kernel.wasmCalls, 1);
      assert.equal(diagnostics().kernel.fallbackCalls, 0);
    }
  assert.equal(diagnostics().variants.filter((variant) => variant.initialized).length, 4);
});

test("index failure discards scratch writes; later valid calls resume the same Wasm variant", async () => {
  const { app, diagnostics } = await application(`
export function scatter(a,index){for(let i=0;i<index.length;i++){a[0]+=1;a[index[i]]+=10;}return a[0];}
export function run(a,index){return scatter(a,index);}`);
  const a = new Float64Array([0, 1, 2]),
    expected = a.slice(),
    index = new Uint32Array([1, 99, 0]);
  assert.equal(app.run(a, index), app.scatter(expected, index));
  same(a, expected);
  assert.equal(diagnostics().kernel.wasmCalls, 0);
  assert.equal(diagnostics().kernel.fallbackCalls, 1);
  index[1] = 2;
  assert.equal(app.run(a, index), app.scatter(expected, index));
  same(a, expected);
  assert.equal(diagnostics().kernel.wasmCalls, 1);
});

test("storage hints do not replace actual callee identity or original argument evaluation", async () => {
  const { app, diagnostics } = await application(`
function scatter(a,index){for(let i=0;i<index.length;i++)a[index[i]]+=1;}
export function shadow(scatter,a,index){return scatter(a,index);}
export function run(a,index,trace){return scatter((trace.push('a'),a),(trace.push('index'),index));}`);
  const a = new Float32Array([0]),
    index = new Uint16Array([0]),
    trace = [];
  assert.equal(
    app.shadow(() => 42, a, index),
    42,
  );
  assert.equal(diagnostics().identityMisses, 1);
  assert.equal(diagnostics().initialized, false);
  app.run(a, index, trace);
  same(a, [1]);
  assert.deepEqual(trace, ["a", "index"]);
  assert.equal(diagnostics().kernel.wasmCalls, 1);
});

test("overlapping source/output and plain arrays retain original ordered behavior", async () => {
  const { app, diagnostics } = await application(`
export function scatter(a,b,index){for(let i=0;i<index.length;i++)a[index[i]]+=b[i];}
export function run(a,b,index){return scatter(a,b,index);}`);
  const a = new Float32Array([1, 2, 3]),
    expected = a.slice(),
    index = new Uint16Array([2, 0, 1]);
  app.scatter(expected, expected, index);
  app.run(a, a, index);
  same(a, expected);
  assert.equal(diagnostics().kernel.lastGuardFailure, "KERNEL_ARRAY_ALIAS");
  const plain = [1, 2, 3],
    reference = plain.slice();
  app.scatter(reference, reference, [2, 0, 1]);
  app.run(plain, plain, [2, 0, 1]);
  same(plain, reference);
  assert.equal(diagnostics().kernel.wasmCalls, 0);
});

test("live lexical Math replacement uses JavaScript rather than stale Wasm intrinsics", async () => {
  const { app, diagnostics } = await application(`let Math=globalThis.Math;
function update(a,index){for(let i=0;i<index.length;i++)a[Math.floor(index[i])]+=1;}
export function setMath(value){Math=value;}
export function run(a,index){return update(a,index);}`);
  const a = new Float32Array([0, 0]),
    index = new Float32Array([0, 1]);
  app.run(a, index);
  same(a, [1, 1]);
  app.setMath({
    floor() {
      return 0;
    },
  });
  app.run(a, index);
  same(a, [3, 1]);
  assert.equal(diagnostics().kernel.wasmCalls, 1);
  assert.equal(diagnostics().kernel.fallbackCalls, 1);
  assert.equal(diagnostics().kernel.lastGuardFailure, "KERNEL_MATH_BINDING");
});

test("host Wasm policy failure still executes the original indexed function exactly once", async () => {
  const { app, diagnostics } = await application(`
function scatter(a,index){for(let i=0;i<index.length;i++)a[index[i]]+=1;}
export function run(a,index){return scatter(a,index);}`);
  const saved = globalThis.WebAssembly,
    a = new Float32Array([0]),
    index = new Uint32Array([0, 0]);
  try {
    globalThis.WebAssembly = undefined;
    app.run(a, index);
  } finally {
    globalThis.WebAssembly = saved;
  }
  same(a, [2]);
  assert.equal(diagnostics().retainedCalls, 1);
  assert.equal(diagnostics().initializationFailure, "KERNEL_INITIALIZATION_FAILED");
});

test("closed helper indices compile, while mutable helpers and effectful passes remain JavaScript", async () => {
  const source = `function offset(i){return i*2;}
function update(a,index){for(let i=0;i<index.length;i++)a[offset(index[i])]+=1;}
export function run(a,index){return update(a,index);}`;
  const { app, diagnostics, transformed } = await application(source);
  const a = new Float32Array(5);
  app.run(a, new Uint32Array([2, 0, 2]));
  same(a, [1, 0, 0, 0, 2]);
  assert.equal(diagnostics().kernel.wasmCalls, 1);
  assert.deepEqual(
    transformed.report.candidates[0].scalarHelpers.map((helper) => helper.name),
    ["offset"],
  );
  for (const suffix of [
    "export function replace(){offset=x=>x;}",
    'export function inspect(){eval("offset");}',
  ]) {
    const refused = specializeNumericModule(source + suffix);
    assert.equal(refused.changed, false);
    assert.equal(refused.code, source + suffix);
  }
  const effectful =
    "function f(a,index){for(let i=0;i<index.length;i++)a[index[i]]=1;external();} f(a,index);";
  assert.equal(specializeNumericModule(effectful).changed, false);
});

test("variant enumeration is deterministic and bounded, even with many potential topology inputs", () => {
  const params = Array.from({ length: 20 }, (_, i) => `p${i}`);
  const body = params.map((name) => `a[${name}[i]]+=1;`).join("");
  const source = `function f(a,${params}){for(let i=0;i<a.length;i++){${body}}} f(a,${params});`;
  const first = specializeNumericModule(source),
    second = specializeNumericModule(source);
  assert.equal(first.report.compiledKernels, 1);
  assert.equal(first.report.candidates[0].variants.length, 17);
  assert.equal(first.code, second.code);
  assert.deepEqual(first.report, second.report);
});

test("post-link plugin emits executable indexed chunks and relocatable production runtime assets", async () => {
  const plugin = numericKernelRollupPlugin(),
    bundle = {};
  const context = {
    emitFile(asset) {
      bundle[asset.fileName] = asset;
      return asset.fileName;
    },
  };
  const chunk = {
    type: "chunk",
    name: "application",
    fileName: "chunks/application.mjs",
    moduleIds: ["normals.mjs", "application.mjs"],
    imports: [],
    importedBindings: {},
  };
  plugin.renderStart();
  const rendered = plugin.renderChunk.call(context, normalsSource, chunk, { format: "es" });
  assert.ok(rendered);
  chunk.code = rendered.code;
  bundle[chunk.fileName] = chunk;
  plugin.generateBundle.call(context, { format: "es" }, bundle);
  const report = plugin.api.getReport();
  assert.equal(report.compiledKernels, 1);
  assert.equal(report.runtimeAssets.length, 2);
  assert.equal(report.units[0].candidates[0].indexSemantics, "checked-integer-full-view-v1");
  assert.equal(chunk.imports.length, 1);
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-indexed-emission-"));
  for (const file of Object.values(bundle)) {
    const destination = path.join(root, file.fileName);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, file.type === "chunk" ? file.code : file.source, { flag: "wx" });
  }
  const app = await import(pathToFileURL(path.join(root, chunk.fileName)).href);
  // Count real Wasm executions, not just a build-time route label. The test
  // observes the host boundary; the emitted dispatcher/runtime are unmodified.
  const Instance = WebAssembly.Instance;
  let nativeCalls = 0;
  WebAssembly.Instance = new Proxy(Instance, {
    construct(target, args) {
      const instance = Reflect.construct(target, args);
      return {
        exports: {
          ...instance.exports,
          run(...values) {
            const result = instance.exports.run(...values);
            nativeCalls++;
            return result;
          },
        },
      };
    },
  });
  try {
    for (const Index of [Uint16Array, Uint32Array]) {
      const p = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
      const index = new Index([0, 1, 2]),
        n = new Float32Array(9),
        expected = n.slice();
      assert.equal(app.frame(p, index, n), recomputeNormals(p, index, expected));
      same(n, expected);
    }
  } finally {
    WebAssembly.Instance = Instance;
  }
  assert.equal(nativeCalls, 2);
});
