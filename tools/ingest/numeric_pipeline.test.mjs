import assert from "node:assert/strict";
import test from "node:test";
import { compileNumericKernel, NumericKernelCompileError } from "./numeric_kernel.mjs";
import { instantiateNumericKernel, NumericKernelGuardError } from "./numeric_kernel_runtime.mjs";

const typesFor = (args) =>
  args.map((value) =>
    value instanceof Float32Array ? "f32[]" : value instanceof Float64Array ? "f64[]" : "f64",
  );
function build(fn, args, helpers = []) {
  return compileNumericKernel(fn.toString(), {
    parameterTypes: typesFor(args),
    helperSources: new Map(helpers.map((helper) => [helper.name, helper.toString()])),
  });
}
const copy = (args) => args.map((value) => (ArrayBuffer.isView(value) ? value.slice() : value));
function equalArrays(actual, expected) {
  actual.forEach((value, index) => {
    if (!ArrayBuffer.isView(value)) return;
    assert.equal(value.length, expected[index].length);
    for (let i = 0; i < value.length; i++) {
      assert.ok(
        Object.is(value[i], expected[index][i]),
        `argument ${index}, element ${i}: ${value[i]} != ${expected[index][i]}`,
      );
    }
  });
}
function compare(fn, args, { frames = 1, helpers = [] } = {}) {
  const artifact = build(fn, args, helpers);
  assert.equal(artifact.manifest.version, 6);
  assert.equal(WebAssembly.validate(artifact.wasm), true);
  const module = new WebAssembly.Module(artifact.wasm);
  assert.deepEqual(WebAssembly.Module.imports(module), []);
  assert.deepEqual(
    WebAssembly.Module.exports(module).map((item) => item.name),
    ["run", "memory"],
  );
  const engine = instantiateNumericKernel(artifact.wasm);
  const actual = copy(args),
    expected = copy(args);
  for (let frame = 0; frame < frames; frame++) {
    assert.ok(Object.is(engine.run(...actual), fn(...expected)), `return at frame ${frame}`);
    equalArrays(actual, expected);
  }
  assert.equal(engine.diagnostics.wasmCalls, frames);
  assert.equal(engine.diagnostics.fallbackCalls, 0);
  return { artifact, engine, actual, expected };
}

function recenter(position, velocity, dt) {
  for (let i = 0; i < position.length; i++) position[i] += velocity[i] * dt;
  let sum = 0;
  for (let i = 0; i < position.length; i++) sum += position[i];
  const mean = sum / position.length;
  for (let i = 0; i < position.length; i++) position[i] -= mean;
  return mean;
}

test("executes 120 three-pass updates over 10,000 elements with one native call and one packing cycle per frame", () => {
  for (const ArrayType of [Float32Array, Float64Array]) {
    const n = 10000;
    const { engine } = compare(
      recenter,
      [
        ArrayType.from({ length: n }, (_, i) => ((i % 97) - 48) / 7),
        ArrayType.from({ length: n }, (_, i) => ((i % 31) - 15) / 13),
        1 / 60,
      ],
      { frames: 120 },
    );
    assert.equal(engine.diagnostics.copiedBytes, n * ArrayType.BYTES_PER_ELEMENT * 3 * 120);
  }
});

test("ABI records ordered passes, unique first-use count arguments, and accessed-prefix unions", () => {
  function update(a, b, output) {
    for (let i = 0; i < b.length; i++) output[i] = b[i];
    for (let j = 0; j < a.length; j++) output[j] += a[j];
    for (let k = 0; k < b.length; k++) output[k] *= 2;
  }
  const { artifact, engine, actual } = compare(update, [
    new Float64Array([5, 7]),
    new Float64Array([1, 2, 3, 4]),
    new Float64Array(6),
  ]);
  assert.deepEqual([...actual[2]], [12, 18, 6, 8, 0, 0]);
  assert.deepEqual(artifact.manifest.boundParameters, [1, 0]);
  assert.deepEqual(artifact.manifest.loops, [
    { boundParameter: 1, loopStride: 1 },
    { boundParameter: 0, loopStride: 1 },
    { boundParameter: 1, loopStride: 1 },
  ]);
  assert.deepEqual(artifact.manifest.parameters[2].access.loopBounds, [1, 0]);
  assert.equal(artifact.manifest.boundParameter, undefined);
  assert.equal(artifact.manifest.loopStride, undefined);
  for (const manifest of [artifact.manifest, engine.manifest]) {
    assert.ok(Object.isFrozen(manifest));
    assert.ok(Object.isFrozen(manifest.boundParameters));
    assert.ok(Object.isFrozen(manifest.loops));
    assert.ok(manifest.loops.every(Object.isFrozen));
    assert.ok(manifest.parameters.every((param) => Object.isFrozen(param.access.loopBounds)));
  }
});

test("mixes independently sized vec3 and vec4 passes with shared scalar state and f32/f64 arrays", () => {
  function update(vertices, colors, matrix, scale) {
    let energy = 0;
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i],
        y = vertices[i + 1],
        z = vertices[i + 2];
      vertices[i] = x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12];
      vertices[i + 1] = x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13];
      vertices[i + 2] = x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14];
      energy += vertices[i] * vertices[i];
    }
    const gain = energy / vertices.length;
    for (let i = 0; i < colors.length; i += 4) {
      colors[i] *= scale;
      if (gain > 0) colors[i + 3] = colors[i + 3] / (1 + gain);
    }
    return energy;
  }
  for (const ArrayType of [Float32Array, Float64Array]) {
    compare(
      update,
      [
        ArrayType.from({ length: 30000 }, (_, i) => (i % 23) / 11),
        ArrayType.from({ length: 800 }, (_, i) => (i % 17) / 19),
        new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.01, -0.02, 0.03, 1]),
        0.99,
      ],
      { frames: 120 },
    );
  }
});

test("f32 stores round before later passes and loop-carried reductions consume those rounded values", () => {
  function update(x, scale) {
    for (let i = 0; i < x.length; i++) x[i] *= scale;
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      sum += x[i];
      x[i] /= scale;
    }
    return sum;
  }
  compare(update, [new Float32Array([1 / 3, 1e-30, 1e30, -0]), 1.123456789], { frames: 30 });
});

test("scalar parameters, declarations, branches, and helper graphs carry state across passes without captures", () => {
  function square(x) {
    return x * x;
  }
  function divide(x, y) {
    if (y === 0) return x;
    return x / y;
  }
  function update(x, y, total) {
    for (let i = 0; i < x.length; i++) total += square(x[i]);
    const mean = divide(total, x.length);
    let gain = 1;
    if (mean > 0) {
      gain = divide(1, mean);
    } else {
      gain = 0;
    }
    for (let j = 0; j < y.length; j++) y[j] = y[j] * gain;
    total++;
    return total + mean;
  }
  const { artifact } = compare(
    update,
    [new Float64Array([1, 2, 3]), new Float64Array([3, 4, 5, 6]), 7],
    { helpers: [square, divide] },
  );
  assert.deepEqual(artifact.helpers, [
    { name: "square", arity: 1 },
    { name: "divide", arity: 2 },
  ]);
});

test("all-read reductions over the same array argument are native, including zero-sized and 100,000-element inputs", () => {
  function covariance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i];
    const mean = sum / a.length;
    let variance = 0;
    for (let i = 0; i < b.length; i++) variance += (a[i] - mean) * (b[i] - mean);
    return variance / b.length;
  }
  for (const count of [0, 1, 100000]) {
    const x = Float64Array.from({ length: count }, (_, i) => (i % 71) / 13);
    const engine = instantiateNumericKernel(build(covariance, [x, x]).wasm);
    assert.ok(Object.is(engine.run(x, x), covariance(x, x)));
    assert.equal(engine.diagnostics.wasmCalls, 1);
  }
});

test("empty early/middle passes do not skip later work and every pass index resets on each invocation", () => {
  function update(a, b, c) {
    let n = 0;
    for (let i = 0; i < a.length; i++) {
      a[i] += 1;
      n++;
    }
    for (let i = 0; i < b.length; i++) {
      b[i] += 2;
      n++;
    }
    for (let i = 0; i < c.length; i++) {
      c[i] += 3;
      n++;
    }
    return n;
  }
  for (const sizes of [
    [0, 0, 0],
    [0, 0, 3],
    [3, 0, 2],
    [0, 5, 0],
    [4, 7, 2],
  ]) {
    compare(
      update,
      sizes.map((n) => new Float64Array(n)),
      { frames: 3 },
    );
  }
});

test("union extents preserve untouched tails even when a later write covers only a shorter prefix", () => {
  function update(long, short, target) {
    let sum = 0;
    for (let i = 0; i < long.length; i++) sum += target[i];
    for (let i = 0; i < short.length; i++) if (short[i] > 0) target[i] = sum;
    return sum;
  }
  const args = [
    new Float64Array(7),
    new Float64Array([1, 0]),
    new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
  ];
  const { engine, actual } = compare(update, args, { frames: 4 });
  assert.deepEqual([...actual[2].slice(1)], [2, 3, 4, 5, 6, 7, 8]);
  // The long array supplies only a count; its elements are not packed or copied.
  assert.equal(engine.diagnostics.copiedBytes, (7 * 2 + 2) * 8 * 4);
});

test("reused scratch memory preserves skipped writes and handles increasing/decreasing independent lengths", () => {
  function update(a, b, enabled) {
    for (let i = 0; i < a.length; i++) if (enabled) a[i] = 2;
    for (let i = 0; i < b.length; i++) if (enabled) b[i] = 3;
  }
  const engine = instantiateNumericKernel(
    build(update, [new Float64Array(), new Float64Array(), 1]).wasm,
  );
  for (const [aLength, bLength] of [
    [2, 7],
    [100000, 60000],
    [1, 0],
    [70000, 2],
  ]) {
    for (const enabled of [1, 0]) {
      const a = new Float64Array(aLength).fill(7),
        b = new Float64Array(bLength).fill(11);
      const expectedA = a.slice(),
        expectedB = b.slice();
      engine.run(a, b, enabled);
      update(expectedA, expectedB, enabled);
      assert.deepEqual(a, expectedA);
      assert.deepEqual(b, expectedB);
    }
  }
  assert.equal(engine.diagnostics.wasmCalls, 8);
});

test("preserves signed zeros, NaNs, infinities and sequential reduction grouping across passes", () => {
  function update(x, y) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      x[i] = x[i] / y;
      sum += x[i];
    }
    for (let i = 0; i < x.length; i++) x[i] = x[i] * y;
    return sum;
  }
  for (const value of [0, -0, 1, -1, 1e20, Infinity, -Infinity, NaN]) {
    compare(update, [new Float64Array([0, -0, 1, -1, 1e-300, Infinity, -Infinity, NaN]), value]);
  }
  function ordered(x) {
    let total = 0;
    for (let i = 0; i < x.length; i++) total += x[i];
    for (let i = 0; i < x.length; i++) total -= x[i];
    return total;
  }
  compare(ordered, [new Float64Array([1e20, 1, -1e20, 3])]);
});

test("later-pass short inputs cause whole-function fallback before any original arrays are published", () => {
  function update(a, b, input) {
    for (let i = 0; i < a.length; i++) a[i] += 1;
    for (let i = 0; i < b.length; i++) b[i] += input[i];
    return a.length + b.length;
  }
  const a = new Float64Array([2]),
    b = new Float64Array([3, 4]),
    input = new Float64Array([1]);
  const artifact = build(update, [a, b, input]);
  let calls = 0;
  const receiver = {};
  const engine = instantiateNumericKernel(artifact.wasm, {
    fallback(...args) {
      assert.equal(this, receiver);
      assert.deepEqual([...a], [2]);
      assert.deepEqual([...b], [3, 4]);
      assert.equal(args[0], a);
      calls++;
      return update(...args);
    },
  });
  assert.equal(engine.run.call(receiver, a, b, input), 3);
  assert.deepEqual([...a], [3]);
  assert.deepEqual([...b], [4, NaN]);
  assert.equal(calls, 1);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_ARRAY_LENGTH");
  assert.equal(engine.diagnostics.wasmCalls, 0);
  assert.equal(engine.diagnostics.copiedBytes, 0);
});

test("every pass extent and intrinsic length is guarded, not just the first loop", () => {
  function update(a, b) {
    for (let i = 0; i < a.length; i++) a[i] += 1;
    for (let i = 0; i < b.length; i += 3) b[i + 2] *= 2;
  }
  const artifact = build(update, [new Float64Array(), new Float64Array()]);
  let calls = 0;
  const engine = instantiateNumericKernel(artifact.wasm, {
    fallback(...args) {
      calls++;
      return update(...args);
    },
  });
  const a = new Float64Array([2]),
    b = new Float64Array([3, 4, 5, 6]);
  engine.run(a, b);
  assert.deepEqual([...a], [3]);
  assert.deepEqual([...b], [3, 4, 10, 6]);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_LOOP_EXTENT");
  let reads = 0;
  const c = new Float64Array([7, 8, 9]);
  Object.defineProperty(c, "length", {
    get() {
      reads++;
      return 0;
    },
  });
  engine.run(a, c);
  assert.deepEqual([...a], [4]);
  assert.deepEqual([...c], [7, 8, 9]);
  assert.equal(reads, 1);
  assert.equal(calls, 2);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_MUTABLE_LENGTH");
});

test("cross-pass aliasing uses original JavaScript rather than independent scratch copies", () => {
  function update(a, b) {
    for (let i = 0; i < a.length; i++) a[i] += 1;
    for (let i = 0; i < b.length; i++) b[i] *= 2;
  }
  const storage = new Float64Array([1, 2, 3, 4]);
  const a = storage.subarray(0, 3),
    b = storage.subarray(1);
  let calls = 0;
  const engine = instantiateNumericKernel(build(update, [a, b]).wasm, {
    fallback(...args) {
      calls++;
      return update(...args);
    },
  });
  engine.run(a, b);
  assert.deepEqual([...storage], [2, 6, 8, 8]);
  assert.equal(calls, 1);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_ARRAY_ALIAS");
});

test("guards native types, fixed ownership, scalars, memory budgets and argument count before the first pass", () => {
  function update(a, b, scale) {
    for (let i = 0; i < a.length; i++) a[i] *= scale;
    for (let i = 0; i < b.length; i++) b[i] *= scale;
  }
  const artifact = build(update, [new Float64Array(), new Float64Array(), 2]);
  const engine = instantiateNumericKernel(artifact.wasm, { maxMemoryBytes: 65536 });
  const a = new Float64Array([7]);
  const expect = (code, ...args) => {
    assert.throws(
      () => engine.run(...args),
      (error) => error instanceof NumericKernelGuardError && error.code === code,
    );
    assert.deepEqual([...a], [7]);
  };
  expect("KERNEL_ARRAY_TYPE", a, [], 2);
  let traps = 0;
  expect(
    "KERNEL_ARRAY_TYPE",
    a,
    new Proxy(new Float64Array(1), {
      get() {
        traps++;
        throw new Error("trap");
      },
    }),
    2,
  );
  assert.equal(traps, 0);
  expect("KERNEL_ARRAY_OWNERSHIP", a, new Float64Array(new SharedArrayBuffer(8)), 2);
  expect(
    "KERNEL_ARRAY_OWNERSHIP",
    a,
    new Float64Array(new ArrayBuffer(8, { maxByteLength: 16 })),
    2,
  );
  const detached = new Float64Array(1);
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  expect("KERNEL_ARRAY_OWNERSHIP", a, detached, 2);
  expect("KERNEL_SCALAR_TYPE", a, new Float64Array(1), {
    valueOf() {
      throw new Error("coerced");
    },
  });
  expect("KERNEL_ARGUMENT_COUNT", a);
  expect("KERNEL_MEMORY_LIMIT", a, new Float64Array(10000), 2);
  assert.equal(engine.diagnostics.wasmCalls, 0);
  assert.equal(engine.diagnostics.copiedBytes, 0);
  engine.dispose();
  expect("KERNEL_DISPOSED", a, new Float64Array(), 2);
});

// Local binary utilities permit ABI fault injection without a mocked Wasm engine.
function u32(value) {
  const bytes = [];
  do {
    const b = value & 127;
    value >>>= 7;
    bytes.push(b | (value ? 128 : 0));
  } while (value);
  return bytes;
}
function readU32(bytes, offset) {
  let value = 0,
    shift = 0,
    b;
  do {
    b = bytes[offset++];
    value |= (b & 127) << shift;
    shift += 7;
  } while (b & 128);
  return [value, offset];
}
function sections(bytes) {
  const result = [];
  let offset = 8;
  while (offset < bytes.length) {
    const id = bytes[offset++];
    const [size, start] = readU32(bytes, offset);
    result.push({ id, payload: [...bytes.slice(start, start + size)] });
    offset = start + size;
  }
  return result;
}
function assemble(parts) {
  return new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    ...parts.flatMap(({ id, payload }) => [id, ...u32(payload.length), ...payload]),
  ]);
}
function changeManifest(artifact, mutate) {
  const parts = sections(artifact.wasm);
  const part = parts.find((section) => section.id === 0);
  const [size, start] = readU32(part.payload, 0);
  const manifest = JSON.parse(
    new TextDecoder().decode(new Uint8Array(part.payload.slice(start + size))),
  );
  mutate(manifest);
  part.payload = [
    ...part.payload.slice(0, start + size),
    ...new TextEncoder().encode(JSON.stringify(manifest)),
  ];
  return assemble(parts);
}

test("a real Wasm trap in a later pass never publishes the first pass and fallback runs the full original once", () => {
  function update(a, b) {
    for (let i = 0; i < a.length; i++) a[i] += 1;
    for (let i = 0; i < b.length; i++) b[i] += 2;
  }
  const a = new Float64Array([1]),
    b = new Float64Array([2]);
  const artifact = build(update, [a, b]);
  const parts = sections(artifact.wasm),
    code = parts.find((section) => section.id === 10);
  const [functions, first] = readU32(code.payload, 0);
  const [size, start] = readU32(code.payload, first);
  const body = code.payload.slice(start, start + size);
  // Each counted pass starts with block;loop. Insert an unreachable instruction
  // immediately before the second pass, after the first loop has really run.
  const starts = body.flatMap((_, i) =>
    body[i] === 2 && body[i + 1] === 64 && body[i + 2] === 3 && body[i + 3] === 64 ? [i] : [],
  );
  assert.equal(starts.length, 2);
  body.splice(starts[1], 0, 0x00);
  code.payload = [
    ...u32(functions),
    ...u32(body.length),
    ...body,
    ...code.payload.slice(start + size),
  ];
  const binary = assemble(parts);
  assert.equal(WebAssembly.validate(binary), true);
  let calls = 0;
  const engine = instantiateNumericKernel(binary, {
    fallback(...args) {
      assert.deepEqual([...a], [1]);
      assert.deepEqual([...b], [2]);
      calls++;
      return update(...args);
    },
  });
  engine.run(a, b);
  assert.deepEqual([...a], [2]);
  assert.deepEqual([...b], [4]);
  assert.equal(calls, 1);
  assert.equal(engine.diagnostics.lastGuardFailure, "KERNEL_EXECUTION_FAILED");
  assert.equal(engine.diagnostics.wasmCalls, 0);
  assert.equal(engine.diagnostics.copiedBytes, 0);
});

test("refuses corrupt pipeline metadata at instantiation", async (t) => {
  function update(a, b) {
    for (let i = 0; i < a.length; i++) a[i] += 1;
    for (let i = 0; i < b.length; i++) b[i] += 1;
  }
  const artifact = build(update, [new Float64Array(), new Float64Array()]);
  const cases = [
    [
      "missing bounds",
      (m) => {
        delete m.boundParameters;
      },
    ],
    [
      "empty bounds",
      (m) => {
        m.boundParameters = [];
      },
    ],
    [
      "duplicate bounds",
      (m) => {
        m.boundParameters = [0, 0];
      },
    ],
    [
      "reordered bounds",
      (m) => {
        m.boundParameters = [1, 0];
      },
    ],
    [
      "invalid bound",
      (m) => {
        m.boundParameters[1] = -1;
      },
    ],
    [
      "fractional bound",
      (m) => {
        m.boundParameters[1] = 0.5;
      },
    ],
    [
      "scalar bound",
      (m) => {
        m.parameters[1].type = "f64";
      },
    ],
    [
      "missing loops",
      (m) => {
        delete m.loops;
      },
    ],
    [
      "one pass",
      (m) => {
        m.loops.pop();
      },
    ],
    [
      "too many passes",
      (m) => {
        m.loops = Array(17).fill(m.loops[0]);
      },
    ],
    [
      "null pass",
      (m) => {
        m.loops[1] = null;
      },
    ],
    [
      "unknown count",
      (m) => {
        m.loops[1].boundParameter = 3;
      },
    ],
    [
      "zero stride",
      (m) => {
        m.loops[1].loopStride = 0;
      },
    ],
    [
      "large stride",
      (m) => {
        m.loops[1].loopStride = 17;
      },
    ],
    [
      "fractional stride",
      (m) => {
        m.loops[1].loopStride = 1.5;
      },
    ],
    [
      "stale singular count",
      (m) => {
        m.boundParameter = 0;
      },
    ],
    [
      "stale singular stride",
      (m) => {
        m.loopStride = 1;
      },
    ],
    [
      "unordered execution",
      (m) => {
        m.iterationSemantics = "parallel";
      },
    ],
    [
      "invalid result",
      (m) => {
        m.resultType = "i32";
      },
    ],
    [
      "missing access bounds",
      (m) => {
        delete m.parameters[0].access.loopBounds;
      },
    ],
    [
      "unknown access bound",
      (m) => {
        m.parameters[0].access.loopBounds = [5];
      },
    ],
    [
      "duplicate access bounds",
      (m) => {
        m.parameters[0].access.loopBounds = [0, 0];
      },
    ],
    [
      "inconsistent indexing",
      (m) => {
        m.parameters[0].access.indexed = false;
      },
    ],
    [
      "uninitialized write prefix",
      (m) => {
        m.parameters[0].read = false;
      },
    ],
    [
      "unsupported version",
      (m) => {
        m.version = 7;
      },
    ],
    [
      "wrong kind",
      (m) => {
        m.kind = "closed-numeric-loop";
      },
    ],
  ];
  for (const [name, mutate] of cases)
    await t.test(name, () => {
      assert.throws(
        () => instantiateNumericKernel(changeManifest(artifact, mutate)),
        (error) => error.code === "KERNEL_ABI_MISMATCH",
      );
    });
});

test("rejects unsafe later passes, nested loops, leaked lexical bindings and transitive helper shadowing", async (t) => {
  const first = "for(let i=0;i<a.length;i++)a[i]+=1;";
  const second = "for(let i=0;i<b.length;i++)b[i]+=1;";
  const refused = [
    `${first}for(let i=1;i<b.length;i++)b[i]+=1;`,
    `${first}for(let i=0;i<b.length;i--)b[i]+=1;`,
    `${first}for(let i=0;i<b.length;i+=17)b[i]+=1;`,
    `${first}for(let i=0;i<b.length;i++)b[i+1]+=1;`,
    `${first}for(let i=0;i<b.length;i++){for(let j=0;j<b.length;j++)b[j]+=1;}`,
    `${first}for(let i=0;i<b.length;i++){external();b[i]+=1;}`,
    `${first}for(let i=0;i<b.length;i++){return b[i];}`,
    `${first}for(let i=0;i<b.length;i++){break;}`,
    `${first}for(let i=0;i<b.length;i++){continue;}`,
    `${first}const mean=sum;for(let i=0;i<b.length;i++){const sum=2;b[i]=sum;}`,
    `for(let i=0;i<a.length;i++){const tmp=a[i];}${second}return tmp;`,
    `${first}${second}return i;`,
    `${first}let i=0;${second}`,
    `for(let i=0;i<a.length;i++)a[i]=later;const later=2;${second}`,
    `${first}for(let i=0;i<b.length;i++)b[i]=scalar(b[i]);const scalar=2;`,
    `${first}for(let i=0;i<b.length;i++)b[i]=a[0];`, // written arrays cannot be uniform inputs
    `${first}${second}return true;`,
    `${first}${second}while(true){}`,
    `${first}a[0]=2;${second}`,
    `${first}a.length=0;${second}`,
  ];
  for (const body of refused)
    await t.test(body, () => {
      assert.throws(
        () =>
          compileNumericKernel(`function update(a,b){${body}}`, {
            parameterTypes: ["f64[]", "f64[]"],
            helperSources: new Map([["scalar", "function scalar(x){return x*x;}"]]),
          }),
        NumericKernelCompileError,
      );
    });
});

test("supports the maximum 16 ordered passes deterministically and refuses a seventeenth", () => {
  const pass = "for(let i=0;i<a.length;i++)a[i]+=1;";
  const source = `function update(a){${pass.repeat(16)}}`;
  const options = { parameterTypes: ["f64[]"] };
  const artifact = compileNumericKernel(source, options);
  assert.deepEqual(artifact.wasm, compileNumericKernel(source, options).wasm);
  const engine = instantiateNumericKernel(artifact.wasm),
    a = new Float64Array([2, 3]);
  engine.run(a);
  assert.deepEqual([...a], [18, 19]);
  assert.equal(engine.diagnostics.wasmCalls, 1);
  assert.throws(
    () => compileNumericKernel(`function update(a){${pass.repeat(17)}}`, options),
    /16-pass limit/,
  );
});

test("sixteen independent count arguments and more than 127 locals retain their own indices", () => {
  const names = Array.from({ length: 16 }, (_, i) => `a${i}`);
  const source = `function update(${names.join(",")}){${Array.from({ length: 140 }, (_, i) => `const t${i}=${i};`).join("")}
    ${names.map((name, i) => `for(let j=0;j<${name}.length;j++)${name}[j]+=t139+${i};`).join("")}}`;
  const artifact = compileNumericKernel(source, { parameterTypes: names.map(() => "f64[]") });
  assert.deepEqual(
    artifact.manifest.boundParameters,
    names.map((_, i) => i),
  );
  const engine = instantiateNumericKernel(artifact.wasm);
  const args = names.map((_, i) => new Float64Array(i + 1).fill(i));
  engine.run(...args);
  args.forEach((array, i) => assert.deepEqual([...array], Array(i + 1).fill(139 + 2 * i)));
  assert.equal(engine.diagnostics.wasmCalls, 1);
});

test("fallback preserves coercion order through the complete sequence rather than restarting a later pass", () => {
  function update(a, b, scale) {
    for (let i = 0; i < a.length; i++) a[i] *= scale;
    for (let i = 0; i < b.length; i++) b[i] *= scale;
  }
  const a = new Float64Array([2, 3]),
    b = new Float64Array([5, 7]);
  const observations = [];
  const scalar = {
    valueOf() {
      observations.push([a[0], a[1], b[0], b[1]]);
      return 2;
    },
  };
  const engine = instantiateNumericKernel(build(update, [a, b, 2]).wasm, { fallback: update });
  engine.run(a, b, scalar);
  assert.deepEqual(observations, [
    [2, 3, 5, 7],
    [4, 3, 5, 7],
    [4, 6, 5, 7],
    [4, 6, 10, 7],
  ]);
  assert.deepEqual([...a, ...b], [4, 6, 10, 14]);
  assert.equal(engine.diagnostics.fallbackCalls, 1);
});

test("exceptions from retained later-pass length getters propagate unchanged after original earlier effects", () => {
  function update(a, b) {
    for (let i = 0; i < a.length; i++) a[i] += 1;
    for (let i = 0; i < b.length; i++) b[i] += 1;
  }
  const a = new Float64Array([1]),
    b = new Float64Array([2]);
  const sentinel = new Error("retained second-pass failure");
  let reads = 0;
  Object.defineProperty(b, "length", {
    get() {
      reads++;
      throw sentinel;
    },
  });
  const engine = instantiateNumericKernel(build(update, [a, b]).wasm, { fallback: update });
  assert.throws(
    () => engine.run(a, b),
    (error) => error === sentinel,
  );
  assert.deepEqual([...a], [2]);
  assert.deepEqual([...b], [2]);
  assert.equal(reads, 1);
  assert.equal(engine.diagnostics.fallbackCalls, 1);
});
