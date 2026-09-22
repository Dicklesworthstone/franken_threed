import assert from "node:assert/strict";
import test from "node:test";
import { compileNumericKernel, NumericKernelCompileError } from "./numeric_kernel.mjs";
import { instantiateNumericKernel } from "./numeric_kernel_runtime.mjs";

const sources = (...helpers) => new Map(helpers.map((fn) => [fn.name, fn.toString()]));
const artifact = (fn, helpers, parameterTypes = ["f64[]"]) =>
  compileNumericKernel(fn.toString(), {
    parameterTypes,
    helperSources: sources(...helpers),
  });
const copy = (args) => args.map((value) => (ArrayBuffer.isView(value) ? value.slice() : value));
function compare(fn, helpers, parameterTypes, args, repeats = 1) {
  const built = artifact(fn, helpers, parameterTypes);
  assert.equal(WebAssembly.validate(built.wasm), true);
  const module = new WebAssembly.Module(built.wasm);
  assert.deepEqual(WebAssembly.Module.imports(module), []);
  assert.deepEqual(
    WebAssembly.Module.exports(module).map((item) => item.name),
    ["run", "memory"],
  );
  const engine = instantiateNumericKernel(built.wasm);
  const actual = copy(args),
    expected = copy(args);
  for (let frame = 0; frame < repeats; frame++) {
    assert.ok(Object.is(engine.run(...actual), fn(...expected)));
    // Compare Numbers, not engine-specific NaN payload bytes in typed storage.
    actual.forEach((value, p) => {
      if (ArrayBuffer.isView(value)) {
        value.forEach((number, i) =>
          assert.ok(Object.is(number, expected[p][i]), `parameter ${p}, element ${i}`),
        );
      } else assert.ok(Object.is(value, expected[p]));
    });
  }
  assert.equal(engine.diagnostics.wasmCalls, repeats);
  assert.equal(engine.diagnostics.fallbackCalls, 0);
  return { built, engine, actual };
}

function clamp(value, low, high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}
function smooth(value) {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function animate(x, targets, dt) {
  const blend = smooth(dt);
  for (let i = 0; i < x.length; i++) x[i] = lerp(x[i], targets[i], blend);
}

test("compiles a transitive helper graph once into private deterministic Wasm functions", () => {
  const helpers = [smooth, clamp, lerp];
  const a = artifact(animate, helpers, ["f64[]", "f64[]", "f64"]);
  const b = artifact(animate, helpers.slice().reverse(), ["f64[]", "f64[]", "f64"]);
  assert.deepEqual(a.wasm, b.wasm);
  assert.deepEqual(a.helpers, [
    { name: "smooth", arity: 1 },
    { name: "clamp", arity: 3 },
    { name: "lerp", arity: 3 },
  ]);
  assert.equal(Object.isFrozen(a.helpers), true);
  assert.ok(a.helpers.every(Object.isFrozen));
  assert.equal(a.manifest.automaticRouteAdmission, false);
});

test("runs 10,000-element animation updates for 120 frames with f32 and f64 storage", () => {
  for (const [ArrayType, type] of [
    [Float32Array, "f32[]"],
    [Float64Array, "f64[]"],
  ]) {
    compare(
      animate,
      [smooth, clamp, lerp],
      [type, type, "f64"],
      [
        ArrayType.from({ length: 10000 }, (_, i) => (i - 5000) / 7),
        ArrayType.from({ length: 10000 }, (_, i) => (5000 - i) / 11),
        1 / 60,
      ],
      120,
    );
  }
});

test("preserves nested lexical locals, parameter mutation, increments and early returns", () => {
  function scalar(x, y) {
    x += y;
    let result = x;
    if (x < 0) {
      let x = y;
      x++;
      result -= x;
      if (result < -10) return result / 2;
    } else {
      const y = x * 2;
      result += y;
      result--;
    }
    return result;
  }
  function update(x, delta) {
    for (let i = 0; i < x.length; i++) x[i] = scalar(x[i], delta);
  }
  compare(
    update,
    [scalar],
    ["f64[]", "f64"],
    [new Float64Array([-50, -3, -0, 0, 1, 7, NaN]), 0.5],
    7,
  );
});

test("accepts all-branch returns and numeric truthiness without boolean/number confusion", () => {
  function scalar(x) {
    if ((x !== 0 && x === x) || false) {
      return x < 0 ? -x : x;
    } else {
      if (!x) return -0;
      else return 7;
    }
  }
  function update(x) {
    for (let i = 0; i < x.length; i++) x[i] = scalar(x[i]);
  }
  compare(
    update,
    [scalar],
    ["f64[]"],
    [new Float64Array([0, -0, NaN, Infinity, -Infinity, 1e-300, -1])],
  );
});

test("helper calls preserve signed zeros, NaN, infinities and arithmetic grouping", () => {
  function ordered(x, y) {
    const first = x + y;
    return (first + -y) / y;
  }
  function update(x, divisor) {
    for (let i = 0; i < x.length; i++) x[i] = ordered(x[i], divisor);
  }
  for (const divisor of [0, -0, 1, -1, 1e20, Infinity, -Infinity, NaN]) {
    compare(
      update,
      [ordered],
      ["f64[]", "f64"],
      [new Float64Array([0, -0, 1, -1, 1e-300, Infinity, -Infinity, NaN]), divisor],
    );
  }
});

test("f32 stores still round individually when read back into a scalar helper", () => {
  function times(x, y) {
    return x * y;
  }
  function update(x, gain) {
    for (let i = 0; i < x.length; i++) {
      x[i] = times(x[i], gain);
      x[i] = times(x[i], 1 / gain);
    }
  }
  compare(
    update,
    [times],
    ["f32[]", "f64"],
    [new Float32Array([1 / 3, 1e-30, 1e30, -0]), 1.123456789],
    30,
  );
});

test("calls helpers from ordered reductions, indexed expressions and the final result", () => {
  function square(x) {
    return x * x;
  }
  function divide(x, count) {
    return x / count;
  }
  function energy(x) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += square(x[i]);
    return divide(sum, x.length);
  }
  for (const count of [0, 1, 100000]) {
    compare(
      energy,
      [square, divide],
      ["f64[]"],
      [Float64Array.from({ length: count }, (_, i) => (i % 71) / 7)],
    );
  }
});

test("works with streamed vec3 geometry and shared fixed-index matrix inputs", () => {
  function transform(x, y, z, a, b, c, d) {
    return a * x + b * y + c * z + d;
  }
  function update(vertices, matrix) {
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i],
        y = vertices[i + 1],
        z = vertices[i + 2];
      vertices[i] = transform(x, y, z, matrix[0], matrix[4], matrix[8], matrix[12]);
      vertices[i + 1] = transform(x, y, z, matrix[1], matrix[5], matrix[9], matrix[13]);
      vertices[i + 2] = transform(x, y, z, matrix[2], matrix[6], matrix[10], matrix[14]);
    }
  }
  compare(
    update,
    [transform],
    ["f32[]", "f64[]"],
    [
      Float32Array.from({ length: 30000 }, (_, i) => (i % 91) / 13),
      new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.125, -0.5, 0.75, 1]),
    ],
    120,
  );
});

test("supports zero-argument helpers and more than 127 scalar locals", () => {
  const helper = `function scalar(x) { ${Array.from({ length: 140 }, (_, i) => `const t${i}=x+${i};`).join("")} return t139; }`;
  const built = compileNumericKernel(
    "function f(x){for(let i=0;i<x.length;i++)x[i]=scalar(x[i])+constant();}",
    {
      parameterTypes: ["f64[]"],
      helperSources: new Map([
        ["scalar", helper],
        ["constant", "function constant(){return 42;}"],
      ]),
    },
  );
  const engine = instantiateNumericKernel(built.wasm);
  const x = new Float64Array([1, 2]);
  engine.run(x);
  assert.deepEqual([...x], [182, 183]);
});

test("ignores unreachable helper sources rather than evaluating or parsing application code", () => {
  const source = "function f(x){for(let i=0;i<x.length;i++)x[i]=x[i]+1;}";
  const base = compileNumericKernel(source, { parameterTypes: ["f64[]"] });
  const withHelpers = compileNumericKernel(source, {
    parameterTypes: ["f64[]"],
    helperSources: new Map([
      ["unused", "this is not valid JavaScript"],
      ["oversized", " ".repeat(300000)],
    ]),
  });
  assert.deepEqual(withHelpers.wasm, base.wasm);
  assert.deepEqual(withHelpers.helpers, []);
});

test("retains transactional alias/scalar/length fallback for helper-containing kernels", () => {
  function add(x, y) {
    return x + y;
  }
  function update(x, y, delta) {
    for (let i = 0; i < x.length; i++) x[i] = add(x[i], y[i] * delta);
  }
  const built = artifact(update, [add], ["f64[]", "f64[]", "f64"]);
  let calls = 0;
  const engine = instantiateNumericKernel(built.wasm, {
    fallback(...args) {
      calls++;
      return update(...args);
    },
  });
  const storage = new Float64Array([1, 2, 3, 4]);
  engine.run(storage.subarray(1), storage.subarray(0, 3), 1);
  assert.deepEqual([...storage], [1, 3, 6, 10]);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_ARRAY_ALIAS");
  let coercions = 0;
  engine.run(new Float64Array([1, 2]), new Float64Array([2, 3]), {
    valueOf() {
      coercions++;
      return 2;
    },
  });
  assert.equal(coercions, 2);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_SCALAR_TYPE");
  const x = new Float64Array([1, 2]);
  engine.run(x, new Float64Array([2]), 1);
  assert.deepEqual([...x], [3, NaN]);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_ARRAY_LENGTH");
  assert.equal(calls, 3);
  assert.equal(engine.diagnostics.wasmCalls, 0);
});

test("refuses effects, implicit returns, free variables, invalid scopes and non-scalar helpers", async (t) => {
  const refused = [
    "function scalar(x){return captured+x;}",
    "function scalar(x){return i+x;}",
    "function scalar(x){return x.length;}",
    "function scalar(x){return x[0];}",
    "function scalar(x){return this.value;}",
    "function scalar(x){return arguments[0];}",
    "function scalar(x){return Math.sin(x);}",
    "function scalar(x){if(false) external(); return x;}",
    "function scalar(x){return x; external();}",
    "function scalar(x){return x; function hidden(){return 1;}}",
    "function scalar(x){if(x) return x;}",
    "function scalar(x){return;}",
    "function scalar(x){return true;}",
    'function scalar(x){return "1"+x;}',
    "function scalar(x){return x||1;}",
    "function scalar(x){const y=y+1;return y;}",
    "function scalar(x){const y=1; {return y; const y=2;}}",
    "function scalar(x){const y=1;y++;return y;}",
    "function scalar(x){if(x){const y=1;}return y;}",
    "function scalar(x){var y=1;return y;}",
    "function scalar(x){while(x)x--;return x;}",
    "function scalar(x){throw x;}",
    "function scalar(x){x.value=1;return 1;}",
    "function scalar(x){x=1;return true;}",
    "function scalar(x=1){return x;}",
    "function scalar({x}){return x;}",
    "async function scalar(x){return x;}",
    "function* scalar(x){return x;}",
    "function wrong(x){return x;}",
    'function scalar(x){return x;} console.log("side effect");',
    "function scalar(x){return scalar(x);}",
    "function scalar(x){return other(x);}",
  ];
  for (const helper of refused)
    await t.test(helper, () => {
      assert.throws(
        () =>
          compileNumericKernel("function f(x){for(let i=0;i<x.length;i++)x[i]=scalar(x[i]);}", {
            parameterTypes: ["f64[]"],
            helperSources: new Map([
              ["scalar", helper],
              ["other", "function other(x){return scalar(x);}"],
            ]),
          }),
        NumericKernelCompileError,
      );
    });
});

test("refuses shadowed helpers, wrong arity, optional calls, spreads and array arguments", async (t) => {
  const snippets = [
    "for(let i=0;i<x.length;i++){const scalar=1;x[i]=scalar(x[i]);}",
    "for(let i=0;i<x.length;i++){x[i]=scalar(x[i]);const scalar=1;}",
    "for(let i=0;i<x.length;i++)x[i]=scalar();",
    "for(let i=0;i<x.length;i++)x[i]=scalar(x[i],1);",
    "for(let i=0;i<x.length;i++)x[i]=scalar?.(x[i]);",
    "for(let i=0;i<x.length;i++)x[i]=scalar(...x);",
    "for(let i=0;i<x.length;i++)x[i]=scalar(x);",
  ];
  for (const body of snippets)
    await t.test(body, () => {
      assert.throws(
        () =>
          compileNumericKernel(`function f(x){${body}}`, {
            parameterTypes: ["f64[]"],
            helperSources: new Map([["scalar", "function scalar(x){return x;}"]]),
          }),
        NumericKernelCompileError,
      );
    });
  const helperSources = new Map([
    ["scalar", "function scalar(x){return other(x);const other=1;}"],
    ["other", "function other(x){return x;}"],
  ]);
  assert.throws(
    () =>
      compileNumericKernel("function f(x){for(let i=0;i<x.length;i++)x[i]=scalar(x[i]);}", {
        parameterTypes: ["f64[]"],
        helperSources,
      }),
    /unshadowed/,
  );
});

test("refuses excessively deep or exponentially expanding acyclic helper graphs", () => {
  for (const [length, double] of [
    [34, false],
    [15, true],
  ]) {
    const helperSources = new Map(
      Array.from({ length }, (_, i) => [
        `h${i}`,
        `function h${i}(x){return ${i === length - 1 ? "x" : `h${i + 1}(x)${double ? `+h${i + 1}(x)` : ""}`};}`,
      ]),
    );
    assert.throws(
      () =>
        compileNumericKernel("function f(x){for(let i=0;i<x.length;i++)x[i]=h0(x[i]);}", {
          parameterTypes: ["f64[]"],
          helperSources,
        }),
      /limit/,
    );
  }
});

test("validates the explicit helper-source ABI instead of accepting dynamic providers", () => {
  assert.throws(
    () =>
      compileNumericKernel("function f(x){for(let i=0;i<x.length;i++)x[i]=1;}", {
        parameterTypes: ["f64[]"],
        helperSources: {},
      }),
    /INVALID_KERNEL_SOURCE/,
  );
});
