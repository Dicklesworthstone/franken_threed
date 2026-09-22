import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { specializeNumericModule } from "./numeric_specialization.mjs";

const STEP =
  "export function step(a, v, dt) { for (let i = 0; i < a.length; i++) a[i] += v[i] * dt; }";
const TICK = "export function tick(a, v, dt) { return step(a, v, dt); }";
const runtimeFiles = ["numeric_dispatch.mjs", "numeric_kernel_runtime.mjs"];

function directory() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-specialization-"));
  for (const name of runtimeFiles)
    fs.copyFileSync(new URL(name, import.meta.url), path.join(dir, name));
  return dir;
}
async function load(source, options = {}) {
  const result = specializeNumericModule(source, options);
  const dir = directory();
  fs.writeFileSync(path.join(dir, "entry.mjs"), result.code);
  const module = await import(pathToFileURL(path.join(dir, "entry.mjs")));
  return { ...result, module, dir };
}

// Only the counter is instrumented; every call executes a native Wasm instance.
function observeWasm(t) {
  const NativeInstance = WebAssembly.Instance;
  const counts = { instances: 0, calls: 0 };
  WebAssembly.Instance = (...args) => {
    const instance = Reflect.construct(NativeInstance, args);
    counts.instances++;
    return {
      exports: {
        memory: instance.exports.memory,
        run(...params) {
          counts.calls++;
          return instance.exports.run(...params);
        },
      },
    };
  };
  t.after(() => {
    WebAssembly.Instance = NativeInstance;
  });
  return counts;
}

test("ordinary frame calls execute one persistent Wasm instance, with unaltered exports and function identity", async (t) => {
  const counts = observeWasm(t);
  const { module, report, code } = await load(`${STEP}\n${TICK}\nexport const alias = step;`);
  assert.equal(report.compiledKernels, 1);
  assert.equal(report.rewrittenCalls, 1);
  assert.equal(report.accelerated, false);
  assert.deepEqual(Object.keys(module).sort(), ["alias", "step", "tick"]);
  assert.equal(module.alias, module.step);
  assert.equal(module.step.length, 3);
  assert.equal(module.step.name, "step");
  assert.equal(module.step.toString(), STEP.replace("export ", ""));
  assert.ok(code.includes(STEP));
  assert.equal(counts.instances, 0);
  const actual = new Float64Array([0, 1]),
    expected = actual.slice(),
    v = new Float64Array([1, -2]);
  for (let frame = 0; frame < 200; frame++) {
    module.tick(actual, v, 1 / 60);
    module.step(expected, v, 1 / 60);
    assert.deepEqual(actual, expected);
  }
  assert.deepEqual(counts, { instances: 1, calls: 200 });
});

test("shadowed same-name callee uses its own body, arguments, receiver and return value", async () => {
  const { module } = await load(
    `${STEP}\n${TICK}\nexport function shadow(step, ...args) { return step(...args); }`,
  );
  const actual = module.shadow(
    function (...args) {
      assert.equal(this, undefined);
      return args;
    },
    7,
    8,
  );
  assert.deepEqual(actual, [7, 8]);
  const sentinel = new Error("retained");
  assert.throws(
    () =>
      module.shadow(() => {
        throw sentinel;
      }),
    (e) => e === sentinel,
  );
});

test("callee lookup remains before argument evaluation and nested rewrites do not overwrite each other", async () => {
  const source = `${STEP}
export function other(a, dt) { for (let i = 0; i < a.length; i++) a[i] += dt; }
export function exercise() {
  let calls = [];
  let step = (...args) => { calls.push('original'); return args; };
  const out = step((step = () => { throw Error('wrong callee'); }, 7), other(new Float64Array([0]), 1));
  return { calls, out };
}`;
  const { module, report } = await load(source);
  // Conservative mutation refusal is independent of scope; the other kernel is still compiled.
  assert.equal(report.compiledKernels, 1);
  assert.equal(report.candidates[0].reason, "MUTABLE_FUNCTION_BINDING");
  assert.deepEqual(module.exercise(), { calls: ["original"], out: [7, undefined] });
});

test("parenthesized calls, misleading comments, zero arguments and spreads remain valid", async (t) => {
  const counts = observeWasm(t);
  const { module } = await load(`${STEP}
export function tick(...args) { return ((step)) /* ( fake */ ( /* ( */ ...args); }
export function missing() { return step(/* empty */); }
export function optional(a, v, dt) { return step?.(a, v, dt); }`);
  const a = new Float64Array([1]),
    v = new Float64Array([2]);
  module.tick(a, v, 3);
  assert.equal(a[0], 7);
  module.optional(a, v, 3);
  assert.equal(a[0], 13);
  assert.throws(() => module.missing(), TypeError);
  assert.equal(counts.calls, 1);
});

test("nested call sites and source argument order are retained", async (t) => {
  const counts = observeWasm(t);
  const { module, report } = await load(`${STEP}
export function inc(a, dt) { for (let i = 0; i < a.length; i++) a[i] += dt; }
export function tick(a, v, events) {
  return step((events.push('a'), a), (inc(v, 1), events.push('v'), v), (events.push('dt'), 2));
}`);
  const a = new Float64Array([0]),
    v = new Float64Array([1]),
    events = [];
  module.tick(a, v, events);
  assert.equal(report.compiledKernels, 2);
  assert.deepEqual([...a], [4]);
  assert.deepEqual([...v], [2]);
  assert.deepEqual(events, ["a", "v", "dt"]);
  assert.equal(counts.calls, 2);
});

test("source-level function hoisting and initialization side effects preserve order", async (t) => {
  const counts = observeWasm(t);
  const { module } = await load(`export const events = ['before'];
export const a = new Float64Array([0]); step(a, new Float64Array([2]), 3); events.push('after');
${STEP}`);
  assert.deepEqual([...module.a], [6]);
  assert.deepEqual(module.events, ["before", "after"]);
  assert.equal(counts.calls, 1);
});

test("early calls through cyclic imports retain original JS until registration", async () => {
  const dir = directory();
  const source = `import { early } from './b.mjs'; ${STEP} ${TICK} export const value = early;`;
  fs.writeFileSync(path.join(dir, "a.mjs"), specializeNumericModule(source).code);
  fs.writeFileSync(
    path.join(dir, "b.mjs"),
    `import { tick } from './a.mjs';
    const a = new Float64Array([1]); tick(a, new Float64Array([2]), 3); export const early = a[0];`,
  );
  const module = await import(pathToFileURL(path.join(dir, "a.mjs")));
  assert.equal(module.value, 7);
  const a = new Float64Array([1]);
  module.tick(a, new Float64Array([2]), 3);
  assert.equal(a[0], 7);
});

test("bindings that can be replaced before module evaluation are not associated with old Wasm", async () => {
  const dir = directory();
  const source = `import { early } from './b.mjs'; ${STEP} ${TICK}
export function replace(fn) { step = fn; } export const value = early;`;
  const result = specializeNumericModule(source);
  assert.equal(result.changed, false);
  assert.equal(result.report.candidates[0].reason, "MUTABLE_FUNCTION_BINDING");
  fs.writeFileSync(path.join(dir, "a.mjs"), result.code);
  fs.writeFileSync(
    path.join(dir, "b.mjs"),
    `import { replace, tick } from './a.mjs';
    replace(() => 123); export const early = tick();`,
  );
  const module = await import(pathToFileURL(path.join(dir, "a.mjs")));
  assert.equal(module.value, 123);
  assert.equal(module.tick(), 123);
});

test("runtime module is isolated from application names and generated names avoid nested bindings", async (t) => {
  const counts = observeWasm(t);
  const { module } = await load(`${STEP}
const Uint8Array = null, WeakMap = 1, Reflect = 2, globalThis = null;
export function tick(a, v, dt) { let __f3d_numeric_call_3 = 9; return step(a, v, dt); }`);
  const a = new Float64Array([0]);
  module.tick(a, new Float64Array([2]), 3);
  assert.equal(a[0], 6);
  assert.equal(counts.calls, 1);
});

test("unsupported argument families keep functioning instead of becoming feature refusals", async () => {
  const { module } = await load(`${STEP}\n${TICK}`);
  for (const make of [(xs) => xs, (xs) => new Float32Array(xs), (xs) => new Float64Array(xs)]) {
    const a = make([1, 2]),
      v = make([2, 3]);
    module.tick(a, v, 2);
    assert.deepEqual([...a], [5, 8]);
  }
});

test("custom length and scalar coercion effects run only in the original fallback", async () => {
  const { module } = await load(`${STEP}\n${TICK}`);
  let lengths = 0,
    coercions = 0;
  const a = new Float64Array([1, 2]),
    v = new Float64Array([2, 3]);
  Object.defineProperty(a, "length", {
    get() {
      lengths++;
      return 2;
    },
  });
  module.tick(a, v, {
    valueOf() {
      coercions++;
      return 2;
    },
  });
  assert.equal(lengths, 3);
  assert.equal(coercions, 2);
  assert.deepEqual([...a], [5, 8]);
});

test("Wasm-unavailable applications still import and perform their normal update calls", async (t) => {
  const native = globalThis.WebAssembly;
  t.after(() => {
    globalThis.WebAssembly = native;
  });
  globalThis.WebAssembly = undefined;
  const { module } = await load(`${STEP}\n${TICK}`);
  const a = new Float64Array([1]);
  module.tick(a, new Float64Array([2]), 3);
  assert.equal(a[0], 7);
});

test("conditional particle updates preserve exact arithmetic across hundreds of calls", async (t) => {
  const counts = observeWasm(t);
  const source = `export function bounce(a, v, dt, limit) { for (let i = 0; i < a.length; i++) {
    const next = a[i] + v[i] * dt;
    a[i] = next < -limit ? -limit : next > limit ? limit : next;
    if (next < -limit || next > limit) v[i] = -v[i];
  } }
  export function tick(a, v, dt, limit) { return bounce(a, v, dt, limit); }`;
  const { module } = await load(source);
  const actual = new Float64Array([0, -0, NaN, Infinity, -Infinity, 0.2]),
    expected = actual.slice();
  const v = new Float64Array([1, -1, 2, 1, 1, -2]),
    ev = v.slice();
  for (let i = 0; i < 300; i++) {
    module.tick(actual, v, 1 / 60, 1);
    module.bounce(expected, ev, 1 / 60, 1);
    assert.deepEqual(actual, expected);
    assert.deepEqual(v, ev);
  }
  assert.equal(counts.calls, 300);
});

test("refused loops remain byte-for-byte original and carry specific source evidence", async () => {
  const source = `export function step(a) { for (let i = 0; i < a.length; i++) a[i] = Math.sin(a[i]); }
export function tick(a) { step(a); }`;
  const { module, code, report } = await load(source, { sourceName: "updates.mjs" });
  assert.equal(code, source);
  assert.equal(report.compiledKernels, 0);
  assert.equal(report.candidates[0].reason, "KERNEL_NOT_CLOSED");
  assert.equal(report.candidates[0].sourceSpan.start, 7);
  const a = [1];
  module.tick(a);
  assert.equal(a[0], Math.sin(1));
});

test("direct eval and parse-unsupported source units are preserved without added bindings", () => {
  const source = `${STEP}\n${TICK}\nexport const inspect = text => eval(text);`;
  assert.equal(specializeNumericModule(source).code, source);
  assert.equal(specializeNumericModule(source).report.refusal.code, "DIRECT_EVAL");
  const ts = "const value: number = 1;";
  assert.equal(specializeNumericModule(ts).code, ts);
  assert.equal(specializeNumericModule(ts).report.refusal.code, "MODULE_PARSE_UNSUPPORTED");
});

test("uncalled exports remain JS; per-unit budget refuses additional kernels, not the application", () => {
  assert.equal(specializeNumericModule(STEP).report.candidates[0].reason, "NO_LOCAL_DIRECT_CALLS");
  const source = `${STEP}\n${TICK}\nfunction inc(a) { for (let i = 0; i < a.length; i++) a[i] += 1; } inc([]);`;
  const result = specializeNumericModule(source, { maxKernels: 1 });
  assert.equal(result.report.compiledKernels, 1);
  assert.equal(result.report.candidates[1].reason, "KERNEL_BUDGET");
});

test("hashbang, directive prologue and deterministic output survive insertion", () => {
  const source = `#!/usr/bin/env node\n'use strict';\n${STEP}\n${TICK}`;
  const a = specializeNumericModule(source),
    b = specializeNumericModule(source);
  assert.ok(a.code.startsWith("#!/usr/bin/env node\n'use strict';"));
  assert.deepEqual(a, b);
});
