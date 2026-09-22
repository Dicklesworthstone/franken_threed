import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { buildNumericKernel } from "./numeric_kernel_build.mjs";
import { specializeNumericModule } from "./numeric_specialization.mjs";

const dispatchURL = new URL("./numeric_dispatch.mjs", import.meta.url).href;
const url = (root, name) => pathToFileURL(path.join(root, name)).href;
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-pipeline-integration-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}
function observer(root) {
  // Observe actual dispatch tokens without replacing the runtime, Wasm engine,
  // compiler, or retained implementation. Production exports remain untouched.
  fs.writeFileSync(
    path.join(root, "observer.mjs"),
    `
    import { createNumericDispatch as create, dispatchNumericCall, numericDispatchDiagnostics } from ${JSON.stringify(dispatchURL)};
    export { dispatchNumericCall };
    const tokens=[];
    export function createNumericDispatch(...args){const token=create(...args);tokens.push(token);return token;}
    export function diagnostics(){return tokens.map(numericDispatchDiagnostics);}
  `,
  );
  return url(root, "observer.mjs");
}
async function application(t, source, options = {}) {
  const root = fixture(t);
  const result = specializeNumericModule(source, {
    sourceName: "application.mjs",
    runtimeModule: observer(root),
    ...options,
  });
  fs.writeFileSync(path.join(root, "application.mjs"), result.code);
  fs.writeFileSync(path.join(root, "reference.mjs"), source);
  const module = await import(url(root, "application.mjs"));
  const reference = await import(url(root, "reference.mjs"));
  return {
    result,
    module,
    reference,
    diagnostics: (await import(url(root, "observer.mjs"))).diagnostics,
  };
}
const source = `
export function advance(position,velocity,dt){
  for(let i=0;i<position.length;i++)position[i]+=velocity[i]*dt;
  let sum=0;
  for(let i=0;i<position.length;i++)sum+=position[i];
  const mean=sum/position.length;
  for(let i=0;i<position.length;i++)position[i]-=mean;
  return mean;
}
export const original=advance;
export function frame(position,velocity,dt){return advance(position,velocity,dt);}
`;

test("ordinary three-pass application calls execute one native transaction per frame without changing exported identities", async (t) => {
  const app = await application(t, source);
  assert.equal(app.result.report.compiledKernels, 1);
  assert.equal(app.result.report.rewrittenCalls, 1);
  assert.equal(app.result.report.accelerated, false);
  const item = app.result.report.candidates[0];
  assert.equal(item.loopCount, 3);
  assert.equal(item.loopStride, undefined);
  assert.equal(item.resultType, "f64");
  assert.equal(item.iterationSemantics, "ordered");
  assert.deepEqual(item.boundParameters, [0]);
  for (const pass of item.loops) {
    assert.equal(pass.boundParameter, 0);
    assert.equal(pass.loopStride, 1);
    assert.ok(source.slice(pass.sourceSpan.start, pass.sourceSpan.end).startsWith("for(let i="));
  }
  assert.equal(app.diagnostics()[0].initialized, false);
  assert.deepEqual(Object.keys(app.module), Object.keys(app.reference));
  assert.equal(app.module.original, app.module.advance);
  assert.equal(app.module.advance.toString(), app.reference.advance.toString());
  for (const ArrayType of [Float32Array, Float64Array]) {
    const n = 10000;
    const x = ArrayType.from({ length: n }, (_, i) => ((i % 97) - 48) / 7);
    const velocity = ArrayType.from({ length: n }, (_, i) => ((i % 31) - 15) / 13);
    const expected = x.slice();
    for (let frame = 0; frame < 120; frame++) {
      const dt = (frame + 1) / 6000;
      assert.ok(
        Object.is(app.module.frame(x, velocity, dt), app.reference.frame(expected, velocity, dt)),
      );
      assert.deepEqual(x, expected);
    }
    assert.equal(app.diagnostics()[0].kernel.wasmCalls, 120);
    assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 0);
    assert.equal(
      app.diagnostics()[0].kernel.copiedBytes,
      3 * n * ArrayType.BYTES_PER_ELEMENT * 120,
    );
  }
});

test("native discovery handles independent bounds and strides in all four streamed/uniform layouts", async (t) => {
  const text = `function transform(x,offset){return x+offset;}
    function update(vertices,colors,matrix){
      let sum=0;
      for(let i=0;i<vertices.length;i+=3){
        vertices[i]=transform(vertices[i],matrix[0]);
        vertices[i+1]=transform(vertices[i+1],matrix[1]);
        vertices[i+2]=transform(vertices[i+2],matrix[2]);sum+=vertices[i];
      }
      const gain=sum/vertices.length;
      for(let i=0;i<colors.length;i+=4){colors[i]*=matrix[3];colors[i+3]=gain;}
      return gain;
    }
    export function frame(v,c,m){return update(v,c,m);}`;
  const app = await application(t, text),
    item = app.result.report.candidates[0];
  assert.equal(item.loopCount, 2);
  assert.deepEqual(item.boundParameters, [0, 1]);
  assert.deepEqual(
    item.loops.map((loop) => loop.loopStride),
    [3, 4],
  );
  assert.equal(item.variants.length, 4);
  assert.deepEqual(
    item.scalarHelpers.map((helper) => helper.name),
    ["transform"],
  );
  for (const StreamType of [Float32Array, Float64Array])
    for (const UniformType of [Float32Array, Float64Array]) {
      const v = StreamType.from({ length: 30000 }, (_, i) => (i % 37) / 11);
      const c = new StreamType(800).fill(0.5),
        m = new UniformType([0.01, -0.02, 0.03, 0.99]);
      const expectedV = v.slice(),
        expectedC = c.slice();
      for (let frame = 0; frame < 20; frame++) {
        assert.ok(
          Object.is(app.module.frame(v, c, m), app.reference.frame(expectedV, expectedC, m)),
        );
        assert.deepEqual(v, expectedV);
        assert.deepEqual(c, expectedC);
      }
      assert.equal(app.diagnostics()[0].kernel.wasmCalls, 20);
      assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 0);
    }
});

test("automatically specializes two-pass reductions with scalar results, empty inputs and read-only input aliases", async (t) => {
  const text = `function square(x){return x*x;}
    function variance(a,b){
      let total=0;for(let i=0;i<a.length;i++)total+=a[i];
      const mean=total/a.length;let sum=0;
      for(let i=0;i<b.length;i++)sum+=square(b[i]-mean);
      return sum/b.length;
    }
    export function result(a,b){return variance(a,b);}`;
  const app = await application(t, text);
  for (const count of [0, 1, 100000]) {
    const x = Float64Array.from({ length: count }, (_, i) => (i % 67) / 13);
    assert.ok(Object.is(app.module.result(x, x), app.reference.result(x, x)));
  }
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 3);
  assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 0);
});

test("parenthesized and spread call sites retain argument order and actual callee identity", async (t) => {
  const app = await application(
    t,
    source +
      `
    export function evaluated(a,b,c){return ((advance)) /* (comment) */ (a(),b(),c());}
    export function spread(args){return advance(...args);}
    export function shadow(advance,args){return advance(...args);}`,
  );
  const order = [],
    a = new Float64Array([1, 3]),
    b = new Float64Array([2, 4]);
  assert.equal(
    app.module.evaluated(
      () => {
        order.push("a");
        return a;
      },
      () => {
        order.push("b");
        return b;
      },
      () => {
        order.push("c");
        return 0.5;
      },
    ),
    3.5,
  );
  assert.deepEqual(order, ["a", "b", "c"]);
  assert.deepEqual([...a], [-1.5, 1.5]);
  assert.equal(app.module.spread([a, b, 0]), 0);
  const marker = {};
  assert.equal(
    app.module.shadow(
      (x, y) => {
        assert.equal(x, a);
        assert.equal(y, b);
        return marker;
      },
      [a, b],
    ),
    marker,
  );
  assert.equal(app.diagnostics()[0].identityMisses, 1);
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 2);
  const failure = new Error("argument failed");
  assert.throws(
    () =>
      app.module.evaluated(
        () => {
          throw failure;
        },
        () => b,
        () => 0,
      ),
    (error) => error === failure,
  );
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 2);
});

test("nested calls to different multi-pass kernels keep original evaluation order and use one token per function", async (t) => {
  const app = await application(
    t,
    `
    function total(a){let sum=0;for(let i=0;i<a.length;i++)sum+=a[i];for(let i=0;i<a.length;i++)sum+=a[i];return sum;}
    function scale(a,k){for(let i=0;i<a.length;i++)a[i]*=k;for(let i=0;i<a.length;i++)a[i]+=1;return k;}
    export function frame(a,b){return scale(a,total(b));}`,
  );
  const a = new Float64Array([1, 2]),
    b = new Float64Array([2, 3]);
  assert.equal(app.module.frame(a, b), 10);
  assert.deepEqual([...a], [11, 21]);
  assert.equal(app.result.report.compiledKernels, 2);
  assert.deepEqual(
    app.diagnostics().map((item) => item.kernel.wasmCalls),
    [1, 1],
  );
});

test("later-pass type and length guards fall back to the whole original sequence with original coercions and exceptions", async (t) => {
  const text = `function update(a,b,input,k){
    for(let i=0;i<a.length;i++)a[i]*=k;
    for(let i=0;i<b.length;i++)b[i]+=input[i];return a.length;
  } export function frame(a,b,input,k){return update(a,b,input,k);}`;
  const app = await application(t, text);
  const a = new Float64Array([2]),
    b = new Float64Array([3, 4]),
    input = new Float64Array([5]);
  assert.equal(app.module.frame(a, b, input, 2), 1);
  assert.deepEqual([...a], [4]);
  assert.deepEqual([...b], [8, NaN]);
  assert.equal(app.diagnostics()[0].kernel.lastGuardFailure, "KERNEL_ARRAY_LENGTH");
  let coercions = 0;
  app.module.frame(a, new Float64Array(), new Float64Array(), {
    valueOf() {
      coercions++;
      return 3;
    },
  });
  assert.equal(coercions, 1);
  assert.deepEqual([...a], [12]);
  const failure = new Error("later original length failed"),
    later = new Float64Array([1]);
  let reads = 0;
  Object.defineProperty(later, "length", {
    get() {
      reads++;
      throw failure;
    },
  });
  assert.throws(
    () => app.module.frame(a, later, input, 2),
    (error) => error === failure,
  );
  assert.deepEqual([...a], [24]);
  assert.equal(reads, 1);
  assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 3);
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 0);
});

test("unavailable Wasm preserves all passes and their numeric result", async (t) => {
  const app = await application(t, source);
  const prior = Object.getOwnPropertyDescriptor(globalThis, "WebAssembly");
  Object.defineProperty(globalThis, "WebAssembly", { configurable: true, value: undefined });
  try {
    const a = new Float64Array([1, 3]),
      b = new Float64Array([2, 4]);
    assert.equal(app.module.frame(a, b, 0.5), 3.5);
    assert.deepEqual([...a], [-1.5, 1.5]);
    assert.equal(app.diagnostics()[0].kernel, null);
    assert.equal(app.diagnostics()[0].retainedCalls, 1);
  } finally {
    Object.defineProperty(globalThis, "WebAssembly", prior);
  }
});

test("a later-pass unclosed effect retains the entire original function and creates no runtime dependency", async (t) => {
  const text = `let effects=0;function touch(){effects++;}
    function update(a){for(let i=0;i<a.length;i++)a[i]+=1;for(let i=0;i<a.length;i++){touch();a[i]*=2;}}
    export function frame(a){update(a);return effects;}`;
  let resolutions = 0;
  const result = specializeNumericModule(text, {
    runtimeModule() {
      resolutions++;
      return "./never.mjs";
    },
  });
  assert.equal(result.changed, false);
  assert.equal(result.code, text);
  assert.equal(result.report.compiledKernels, 0);
  assert.equal(resolutions, 0);
  assert.equal(result.report.candidates[0].reason, "KERNEL_NOT_CLOSED");
  const app = await application(t, text),
    a = new Float64Array([1, 2]);
  assert.equal(app.module.frame(a), 2);
  assert.deepEqual([...a], [4, 6]);
  assert.deepEqual(app.diagnostics(), []);
});

test("mutable helpers, direct eval, captured state and dynamic later loops cannot bypass whole-function closure", () => {
  const first = "for(let i=0;i<a.length;i++)a[i]+=1;";
  const inputs = [
    `function helper(x){return x*2;}function update(a){${first}for(let i=0;i<a.length;i++)a[i]=helper(a[i]);}
      export function frame(a){return update((helper=x=>x*3,a));}`,
    `const gain=3;function update(a){${first}for(let i=0;i<a.length;i++)a[i]*=gain;}export function frame(a){update(a);}`,
    `function update(a){${first}for(let i=0;i<a.length;i++)a[i]+=2;eval('');}export function frame(a){update(a);}`,
    `function update(a){${first}for(let i=0;i<a.length;i+=a[i])a[i]+=2;}export function frame(a){update(a);}`,
    `function update(a){${first}for(let i=0;i<a.length;i++)a[i]+=later;const later=2;}export function frame(a){update(a);}`,
  ];
  for (const text of inputs) {
    const result = specializeNumericModule(text);
    assert.equal(result.changed, false, text);
    assert.equal(result.code, text);
    assert.equal(result.report.compiledKernels, 0);
  }
});

test("ESM cycle calls before token initialization use original source, then later calls use native pipelines", async (t) => {
  const root = fixture(t),
    runtime = observer(root);
  const text = `import {early} from './cycle.mjs'; export {early};
    function advance(a){for(let i=0;i<a.length;i++)a[i]+=1;for(let i=0;i<a.length;i++)a[i]*=2;}
    export function frame(a){return advance(a);}`;
  const result = specializeNumericModule(text, { runtimeModule: runtime });
  fs.writeFileSync(path.join(root, "application.mjs"), result.code);
  fs.writeFileSync(
    path.join(root, "cycle.mjs"),
    `import {frame} from './application.mjs';
    export const early=new Float64Array([1,2]);frame(early);`,
  );
  const mod = await import(url(root, "application.mjs"));
  const { diagnostics } = await import(runtime);
  assert.deepEqual([...mod.early], [4, 6]);
  assert.equal(diagnostics()[0].initialized, false);
  const a = new Float64Array([2, 3]);
  mod.frame(a);
  assert.deepEqual([...a], [6, 8]);
  assert.equal(diagnostics()[0].kernel.wasmCalls, 1);
});

test("pipeline budgets count functions, not passes, and discovery output is deterministic", async (t) => {
  const text = `function first(a){for(let i=0;i<a.length;i++)a[i]+=1;for(let i=0;i<a.length;i++)a[i]+=2;}
    function second(a){for(let i=0;i<a.length;i++)a[i]*=2;for(let i=0;i<a.length;i++)a[i]*=3;}
    export function frame(a){first(a);second(a);}`;
  const options = { sourceName: "input.mjs", runtimeModule: "./runtime.mjs", maxKernels: 1 };
  const a = specializeNumericModule(text, options),
    b = specializeNumericModule(text, options);
  assert.deepEqual(a, b);
  assert.equal(a.report.compiledKernels, 1);
  assert.equal(a.report.candidates[0].loopCount, 2);
  assert.equal(a.report.candidates[1].reason, "KERNEL_BUDGET");
  const app = await application(t, text, { maxKernels: 1 }),
    x = new Float64Array([1, 2]);
  app.module.frame(x);
  assert.deepEqual([...x], [24, 30]);
  assert.equal(app.diagnostics().length, 1);
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 1);
});

const packageSource = `function square(x){return x*x;}
export function normalize(x){
  let sum=0;for(let i=0;i<x.length;i++)sum+=square(x[i]);
  const gain=sum===0?1:1/sum;
  for(let i=0;i<x.length;i++)x[i]*=gain;
  return sum;
}`;
function packageFixture(t, text = packageSource) {
  const root = fixture(t),
    entry = path.join(root, "input.mjs"),
    out = path.join(root, "package");
  fs.writeFileSync(entry, text);
  return { root, entry, out };
}
function frozenPipeline(manifest) {
  assert.ok(Object.isFrozen(manifest));
  assert.ok(Object.isFrozen(manifest.boundParameters));
  assert.ok(Object.isFrozen(manifest.loops));
  assert.ok(manifest.loops.every(Object.isFrozen));
  assert.ok(
    manifest.parameters.every((param) => !param.access || Object.isFrozen(param.access.loopBounds)),
  );
}

test("standalone packages preserve complete multi-pass source and helpers, relocate, and execute without runtime compiler dependencies", async (t) => {
  for (const [ArrayType, type] of [
    [Float32Array, "f32[]"],
    [Float64Array, "f64[]"],
  ]) {
    const { root, entry, out } = packageFixture(t);
    const result = buildNumericKernel(entry, out, { parameterTypes: [type] });
    assert.equal(result.kernel.version, 6);
    assert.equal(result.kernel.loops.length, 2);
    assert.equal(result.selectedFunction.name, "normalize");
    assert.equal(result.sourceSha256, digest(packageSource));
    assert.equal(result.wasmSha256, digest(fs.readFileSync(path.join(out, "kernel.wasm"))));
    assert.equal(result.accelerationClaim, false);
    assert.equal(
      fs.readFileSync(path.join(out, "retained.mjs"), "utf8"),
      `${packageSource}\nexport default normalize;\n`,
    );
    assert.deepEqual(fs.readdirSync(out).sort(), result.emittedFiles.slice().sort());
    assert.ok(!fs.readFileSync(path.join(out, "kernel.mjs"), "utf8").includes(root));
    const moved = path.join(root, "moved");
    fs.renameSync(out, moved);
    const mod = await import(url(moved, "kernel.mjs"));
    const retained = await import(url(moved, "retained.mjs"));
    assert.equal(mod.retained, retained.normalize);
    const engine = mod.createKernel();
    frozenPipeline(engine.manifest);
    const actual = new ArrayType([1, 2, 3]),
      expected = actual.slice();
    assert.ok(Object.is(engine.run(actual), mod.retained(expected)));
    assert.deepEqual(actual, expected);
    assert.equal(engine.diagnostics.wasmCalls, 1);
    assert.equal(engine.diagnostics.copiedBytes, actual.byteLength * 2);
    const plain = [1, 2, 3];
    assert.equal(engine.run(plain), 14);
    assert.deepEqual(plain, [1 / 14, 2 / 14, 3 / 14]);
    assert.equal(engine.diagnostics.fallbackCalls, 1);
  }
});

test("unavailable-Wasm package fallback retains all passes, results, disposal and deeply immutable pipeline metadata", async (t) => {
  const { entry, out } = packageFixture(t);
  buildNumericKernel(entry, out, { parameterTypes: ["f64[]"] });
  const { createKernel } = await import(url(out, "kernel.mjs"));
  const prior = Object.getOwnPropertyDescriptor(globalThis, "WebAssembly");
  Object.defineProperty(globalThis, "WebAssembly", { configurable: true, value: undefined });
  try {
    const engine = createKernel();
    frozenPipeline(engine.manifest);
    assert.throws(() => engine.manifest.boundParameters.push(1), TypeError);
    assert.throws(() => {
      engine.manifest.loops[0].loopStride = 2;
    }, TypeError);
    assert.throws(() => engine.manifest.parameters[0].access.loopBounds.push(1), TypeError);
    const x = new Float64Array([3, 4]);
    assert.equal(engine.run(x), 25);
    assert.deepEqual([...x], [3 * (1 / 25), 4 * (1 / 25)]);
    assert.equal(engine.diagnostics.wasmCalls, 0);
    assert.equal(engine.diagnostics.fallbackCalls, 1);
    engine.dispose();
    assert.throws(() => engine.run(x), /KERNEL_DISPOSED/);
  } finally {
    Object.defineProperty(globalThis, "WebAssembly", prior);
  }
});

test("pipeline packages select among multiple entries explicitly and retain fresh-output safety and deterministic artifacts", async (t) => {
  const text =
    packageSource +
    "\nexport function other(x){for(let i=0;i<x.length;i++)x[i]+=1;for(let i=0;i<x.length;i++)x[i]*=2;}";
  const { root, entry, out } = packageFixture(t, text);
  assert.throws(
    () => buildNumericKernel(entry, out, { parameterTypes: ["f64[]"] }),
    /functionName/,
  );
  assert.equal(fs.existsSync(out), false);
  const options = { parameterTypes: ["f64[]"], functionName: "normalize" };
  const result = buildNumericKernel(entry, out, options);
  const second = path.join(root, "second");
  buildNumericKernel(entry, second, options);
  for (const name of result.emittedFiles)
    assert.deepEqual(
      fs.readFileSync(path.join(out, name)),
      fs.readFileSync(path.join(second, name)),
    );
  assert.throws(() => buildNumericKernel(entry, out, options), { code: "EEXIST" });
  const { createKernel } = await import(url(out, "kernel.mjs"));
  assert.equal(createKernel().run(new Float64Array([3, 4])), 25);
});

test("unsafe late package work and helper rebinding refuse before any output or source evaluation", (t) => {
  for (const text of [
    packageSource.replace("x[i]*=gain;", "external();x[i]*=gain;"),
    packageSource + "\nexport function replace(){square=x=>x;}",
    packageSource.replace("const gain=sum===0?1:1/sum;", "const gain=captured;"),
    packageSource + '\nthrow new Error("BUILD_SOURCE_EXECUTED");',
  ]) {
    const { entry, out } = packageFixture(t, text);
    assert.throws(
      () => buildNumericKernel(entry, out, { parameterTypes: ["f64[]"] }),
      /KERNEL_NOT_CLOSED/,
    );
    assert.equal(fs.existsSync(out), false);
  }
});
