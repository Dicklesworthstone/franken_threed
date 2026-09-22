import assert from "node:assert/strict";
import { test } from "node:test";
import {
  createNumericDispatch,
  dispatchNumericCall,
  numericDispatchDiagnostics,
} from "./numeric_dispatch.mjs";
import { compileNumericKernel } from "./numeric_kernel.mjs";

function integrate(a, v, dt) {
  for (let i = 0; i < a.length; i++) a[i] += v[i] * dt;
}
const artifact = compileNumericKernel(integrate.toString(), {
  parameterTypes: ["f64[]", "f64[]", "f64"],
});

test("lazy dispatcher executes real Wasm for repeated ordinary updates", () => {
  const token = createNumericDispatch(integrate, artifact.wasm);
  assert.equal(numericDispatchDiagnostics(token).initialized, false);
  const actual = new Float64Array([0, 1, -1]),
    expected = actual.slice(),
    velocity = new Float64Array([1, -3, 2]);
  for (let frame = 0; frame < 300; frame++) {
    assert.equal(dispatchNumericCall(token, integrate, [actual, velocity, 1 / 60]), undefined);
    integrate(expected, velocity, 1 / 60);
    assert.deepEqual(actual, expected);
  }
  const result = numericDispatchDiagnostics(token);
  assert.equal(result.kernel.wasmCalls, 300);
  assert.equal(result.kernel.fallbackCalls, 0);
  assert.equal(result.retainedCalls, 0);
});

test("callee identity misses do not initialize Wasm and retain receiver/result/exception semantics", () => {
  const token = createNumericDispatch(integrate, artifact.wasm);
  function shadow(...args) {
    assert.equal(this, undefined);
    return args;
  }
  assert.deepEqual(dispatchNumericCall(token, shadow, [1, 2]), [1, 2]);
  const sentinel = new Error("source exception");
  assert.throws(
    () =>
      dispatchNumericCall(token, () => {
        throw sentinel;
      }, []),
    (error) => error === sentinel,
  );
  assert.equal(numericDispatchDiagnostics(token).identityMisses, 2);
  assert.equal(numericDispatchDiagnostics(token).initialized, false);
});

test("undefined cycle token invokes the actual callee", () => {
  let calls = 0;
  assert.equal(
    dispatchNumericCall(
      undefined,
      function () {
        calls++;
        assert.equal(this, undefined);
        return 42;
      },
      [],
    ),
    42,
  );
  assert.equal(calls, 1);
  assert.throws(() => dispatchNumericCall(undefined, undefined, []), TypeError);
});

test("array alias guard retains source sequential dependencies", () => {
  const token = createNumericDispatch(integrate, artifact.wasm);
  const actual = new Float64Array([1, 2, 3, 4]),
    expected = actual.slice();
  dispatchNumericCall(token, integrate, [actual.subarray(1), actual.subarray(0, 3), 1]);
  integrate(expected.subarray(1), expected.subarray(0, 3), 1);
  assert.deepEqual(actual, expected);
  assert.equal(numericDispatchDiagnostics(token).kernel.lastGuardFailure, "KERNEL_ARRAY_ALIAS");
  assert.equal(numericDispatchDiagnostics(token).kernel.fallbackCalls, 1);
});

test("coercions occur only in fallback, in original per-element order", () => {
  const token = createNumericDispatch(integrate, artifact.wasm);
  let conversions = 0;
  const dt = {
    valueOf() {
      conversions++;
      return 2;
    },
  };
  const a = new Float64Array([0, 0]),
    v = new Float64Array([1, 2]);
  dispatchNumericCall(token, integrate, [a, v, dt]);
  assert.deepEqual([...a], [2, 4]);
  assert.equal(conversions, 2);
  assert.equal(numericDispatchDiagnostics(token).kernel.fallbackCalls, 1);
});

test("absence of Wasm leaves the application functional and initialization is tried once", (t) => {
  const token = createNumericDispatch(integrate, artifact.wasm);
  const native = globalThis.WebAssembly;
  t.after(() => {
    globalThis.WebAssembly = native;
  });
  globalThis.WebAssembly = undefined;
  const a = new Float64Array([1]),
    v = new Float64Array([2]);
  dispatchNumericCall(token, integrate, [a, v, 3]);
  globalThis.WebAssembly = native;
  dispatchNumericCall(token, integrate, [a, v, 3]);
  assert.equal(a[0], 13);
  assert.equal(numericDispatchDiagnostics(token).retainedCalls, 2);
  assert.equal(numericDispatchDiagnostics(token).kernel, null);
});

test("blocked Wasm with hostile thrown values does not escape fallback", (t) => {
  const native = WebAssembly.Module;
  let attempts = 0;
  t.after(() => {
    WebAssembly.Module = native;
  });
  WebAssembly.Module = () => {
    attempts++;
    throw {
      get code() {
        throw new Error("must not inspect policy exception");
      },
    };
  };
  const token = createNumericDispatch(integrate, artifact.wasm);
  const a = new Float64Array([0]),
    v = new Float64Array([1]);
  dispatchNumericCall(token, integrate, [a, v, 2]);
  dispatchNumericCall(token, integrate, [a, v, 2]);
  assert.equal(attempts, 1);
  assert.equal(a[0], 4);
});

test("invalid compiler artifact retains source without publishing scratch outputs", () => {
  const token = createNumericDispatch(integrate, [0, 1, 2]);
  const a = new Float64Array([1]),
    v = new Float64Array([2]);
  dispatchNumericCall(token, integrate, [a, v, 3]);
  assert.equal(a[0], 7);
  assert.equal(numericDispatchDiagnostics(token).retainedCalls, 1);
});
