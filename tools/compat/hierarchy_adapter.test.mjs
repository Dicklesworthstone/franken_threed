import assert from "node:assert/strict";
import { test } from "node:test";
import { bindThreeTransformHierarchy } from "./hierarchy_adapter.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function matrix(x = 0) {
  const m = identity();
  m[12] = x;
  return m;
}
function node(x = 0, automatic = false) {
  return {
    isObject3D: true,
    parent: null,
    matrix: { elements: matrix(x) },
    matrixWorld: { elements: identity() },
    matrixAutoUpdate: automatic,
    matrixWorldAutoUpdate: true,
    matrixWorldNeedsUpdate: true,
    position: { x, y: 0, z: 0 },
    quaternion: { x: 0, y: 0, z: 0, w: 1 },
    scale: { x: 1, y: 1, z: 1 },
  };
}
function fixture(count = 2) {
  const calls = [],
    instances = [];
  const controls = {
    output: new Float64Array(Array.from({ length: count }, (_, i) => matrix(100 + i)).flat()),
    changed: Uint32Array.from({ length: count }, (_, i) => i),
    stats: new Uint32Array([count, Math.max(0, count - 1), count ? 1 : 0, 0, count]),
    hook: null,
    constructorHook: null,
    composeOutput: null,
  };
  // Deliberately no matrix multiplication or hierarchy algorithm in this fake.
  // These tests cover adapter transport/ownership. Numerical conformance uses
  // real compiled Wasm and pinned Three.js in hierarchy_conformance.mjs.
  const wasm = {
    F3dTransformHierarchy: class {
      constructor(parents, locals, worlds, flags) {
        this.epoch = 1n;
        this.frees = 0;
        calls.push(["construct", parents.slice(), locals.slice(), worlds.slice(), flags.slice()]);
        instances.push(this);
        controls.constructorHook?.();
      }
      reparent(value) {
        calls.push(["parents", value.slice()]);
      }
      setWorldAuto(ids, flags) {
        calls.push(["flags", ids.slice(), flags.slice()]);
      }
      setLocalMatrices(ids, values) {
        calls.push(["locals", ids.slice(), values.slice()]);
      }
      setWorldMatrices(ids, values) {
        calls.push(["worlds", ids.slice(), values.slice()]);
      }
      solve() {
        calls.push(["solve"]);
        controls.hook?.();
        return this.epoch;
      }
      worldMatrices(revision) {
        assert.equal(revision, this.epoch);
        return controls.output;
      }
      changedIndices(revision) {
        assert.equal(revision, this.epoch);
        return controls.changed;
      }
      solveStats() {
        return controls.stats;
      }
      get revision() {
        return this.epoch;
      }
      free() {
        this.frees++;
        calls.push(["free"]);
      }
    },
    f3d_batch_compose_scalar(p, q, s) {
      calls.push(["compose", p.slice(), q.slice(), s.slice()]);
      return (
        controls.composeOutput ??
        new Float64Array(Array.from({ length: p.length / 3 }, (_, i) => matrix(20 + i)).flat())
      );
    },
  };
  return { wasm, calls, controls, instances };
}
const code = (expected) => (error) => error.code === expected;

test("manual local matrices and arbitrary parent order are passed to Wasm unchanged", () => {
  const f = fixture(),
    parent = node(4),
    child = node(7);
  child.parent = parent;
  const localIdentity = child.matrix.elements,
    worldIdentity = child.matrixWorld.elements;
  const binding = bindThreeTransformHierarchy(f.wasm, [child, parent]);
  assert.deepEqual([...f.calls[0][1]], [1, -1]);
  const receipt = binding.update();
  assert.equal(receipt.revision, 1n);
  assert.equal(child.matrix.elements, localIdentity);
  assert.equal(child.matrixWorld.elements, worldIdentity);
  assert.equal(child.matrix.elements[12], 7);
  assert.equal(child.matrixWorld.elements[12], 100);
  assert.equal(parent.matrixWorld.elements[12], 101);
  assert.equal(
    child.matrixWorldNeedsUpdate,
    true,
    "snapshot binding does not masquerade as source updateMatrixWorld",
  );
  assert.equal(
    f.calls.some((call) => call[0] === "compose"),
    false,
  );
  binding.dispose();
});

test("automatic TRS composition is delegated to the existing f64 Wasm export", () => {
  const f = fixture(),
    a = node(1 + Number.EPSILON, true),
    b = node(9);
  a.quaternion.w = 2;
  a.scale.x = -3;
  const binding = bindThreeTransformHierarchy(f.wasm, [a, b]);
  binding.update();
  const compose = f.calls.filter((c) => c[0] === "compose").at(-1);
  assert.equal(compose[1][0], 1 + Number.EPSILON);
  assert.deepEqual([...compose[2]], [0, 0, 0, 2], "authored quaternion is not normalized in JS");
  assert.deepEqual([...compose[3]], [-3, 1, 1]);
  assert.equal(a.matrix.elements[12], 20);
  assert.equal(b.matrix.elements[12], 9);
  binding.dispose();
});

test("disabled world-auto matrices are sent as authoritative inputs and never overwritten", () => {
  const f = fixture(),
    a = node(),
    b = node();
  b.parent = a;
  a.matrixWorldAutoUpdate = false;
  a.matrixWorld.elements[12] = 55;
  const binding = bindThreeTransformHierarchy(f.wasm, [a, b]);
  binding.update();
  const worldCall = f.calls.find((call) => call[0] === "worlds");
  assert.deepEqual([...worldCall[1]], [0]);
  assert.equal(worldCall[2][12], 55);
  assert.equal(a.matrixWorld.elements[12], 55);
  assert.equal(b.matrixWorld.elements[12], 101);
  binding.dispose();
});

test("missing compiled exports fail instead of selecting a JavaScript math fallback", () => {
  assert.throws(() => bindThreeTransformHierarchy({}, []), code("HIERARCHY_MISSING_WASM"));
  const f = fixture(1),
    a = node(0, true);
  delete f.wasm.f3d_batch_compose_scalar;
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a]), code("HIERARCHY_MISSING_WASM"));
  assert.equal(f.instances.length, 0);
  a.matrixAutoUpdate = false;
  bindThreeTransformHierarchy(f.wasm, [a]).dispose();
});

test("parents must belong to the explicit fixed membership", () => {
  const f = fixture(1),
    a = node();
  a.parent = node();
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a]), code("HIERARCHY_PARENT"));
  assert.equal(f.instances.length, 0);
});

test("duplicate slots, cameras, skin hooks and own world update overrides reject", () => {
  const f = fixture(1),
    a = node();
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a, a]), code("HIERARCHY_NODES"));
  for (const change of [{ isCamera: true }, { isSkinnedMesh: true }, { updateMatrixWorld() {} }]) {
    assert.throws(
      () => bindThreeTransformHierarchy(f.wasm, [Object.assign(node(), change)]),
      code("HIERARCHY_NODES"),
    );
  }
});

test("a second writer cannot bind an already-owned object, but disposal releases it", () => {
  const f = fixture(1),
    a = node(),
    binding = bindThreeTransformHierarchy(f.wasm, [a]);
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a]), code("HIERARCHY_OWNER"));
  binding.dispose();
  binding.dispose();
  assert.equal(f.instances[0].frees, 1);
  assert.throws(() => binding.update(), code("HIERARCHY_DISPOSED"));
  bindThreeTransformHierarchy(f.wasm, [a]).dispose();
});

test("constructor failure frees a partially initialized native object and releases claims", () => {
  const f = fixture(1),
    a = node();
  f.controls.constructorHook = () => {
    f.instances[0].worldMatrices = null;
  };
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a]), code("HIERARCHY_MISSING_WASM"));
  assert.equal(f.instances[0].frees, 1);
  f.controls.constructorHook = null;
  bindThreeTransformHierarchy(f.wasm, [a]).dispose();
});

test("all destinations are preflighted before any retained output is modified", () => {
  const f = fixture(),
    a = node(),
    b = node();
  const binding = bindThreeTransformHierarchy(f.wasm, [a, b]);
  Object.freeze(b.matrixWorld.elements);
  assert.throws(() => binding.update(), code("HIERARCHY_MATRIX"));
  assert.equal(a.matrixWorld.elements[12], 0);
  assert.equal(
    f.calls.some((c) => c[0] === "solve"),
    false,
  );
  binding.dispose();
});

test("matrix aliases and overlapping typed views cannot corrupt another input bank", () => {
  const f = fixture(),
    a = node(),
    b = node();
  b.matrix.elements = a.matrixWorld.elements;
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a, b]), code("HIERARCHY_ALIAS"));
  const c = node(),
    buffer = new ArrayBuffer(24 * 8);
  c.matrix.elements = new Float64Array(buffer, 0, 16);
  c.matrixWorld.elements = new Float64Array(buffer, 8 * 8, 16);
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [c]), code("HIERARCHY_ALIAS"));
});

test("shared and accessor-backed input matrices reject without invoking element getters", () => {
  const f = fixture(1),
    a = node();
  a.matrix.elements = new Float64Array(new SharedArrayBuffer(128));
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a]), code("HIERARCHY_MATRIX"));
  a.matrix.elements = identity();
  let invoked = false;
  Object.defineProperty(a.matrix.elements, "0", {
    get() {
      invoked = true;
      return 1;
    },
  });
  assert.throws(() => bindThreeTransformHierarchy(f.wasm, [a]), code("HIERARCHY_MATRIX"));
  assert.equal(invoked, false);
});

for (const [label, mutate] of [
  [
    "local values",
    (a) => {
      a.matrix.elements[12] = 44;
    },
  ],
  [
    "world matrix identity",
    (a) => {
      a.matrixWorld = { elements: identity() };
    },
  ],
  [
    "parent identity",
    (a) => {
      a.parent = node();
    },
  ],
  [
    "ownership flags",
    (a) => {
      a.matrixWorldAutoUpdate = false;
    },
  ],
  [
    "TRS values",
    (a) => {
      a.position.x = 4;
    },
  ],
]) {
  test(`input changes during native execution reject stale publication: ${label}`, () => {
    const f = fixture(),
      a = node(0, true),
      b = node();
    const binding = bindThreeTransformHierarchy(f.wasm, [a, b]);
    f.controls.hook = () => mutate(a);
    assert.throws(() => binding.update(), code("HIERARCHY_STALE_INPUT"));
    assert.equal(b.matrixWorld.elements[12], 0);
    binding.dispose();
  });
}

test("freezing a later destination during the solve cannot cause partial publication", () => {
  const f = fixture(),
    a = node(),
    b = node();
  const binding = bindThreeTransformHierarchy(f.wasm, [a, b]);
  f.controls.hook = () => Object.freeze(b.matrixWorld.elements);
  assert.throws(() => binding.update(), code("HIERARCHY_MATRIX"));
  assert.equal(a.matrixWorld.elements[12], 0);
  binding.dispose();
});

test("reentrant updates reject; disposal during native work cancels publication and defers free", () => {
  const f = fixture(1),
    a = node(),
    binding = bindThreeTransformHierarchy(f.wasm, [a]);
  f.controls.hook = () => {
    assert.throws(() => binding.update(), code("HIERARCHY_REENTRANT"));
    binding.dispose();
    assert.equal(
      f.instances[0].frees,
      0,
      "native object remains alive until its call stack unwinds",
    );
  };
  assert.throws(() => binding.update(), code("HIERARCHY_DISPOSED"));
  assert.equal(f.instances[0].frees, 1);
  assert.equal(a.matrixWorld.elements[12], 0);
});

test("native revision changes after copy-out reject stale result receipts", () => {
  const f = fixture(1),
    a = node(),
    binding = bindThreeTransformHierarchy(f.wasm, [a]);
  f.instances[0].solveStats = () => {
    f.instances[0].epoch++;
    return f.controls.stats;
  };
  assert.throws(() => binding.update(), code("HIERARCHY_STALE_RESULT"));
  assert.equal(a.matrixWorld.elements[12], 0);
  binding.dispose();
});

test("the final native revision getter cannot resurrect a disposed binding", () => {
  const f = fixture(1),
    a = node(),
    binding = bindThreeTransformHierarchy(f.wasm, [a]);
  Object.defineProperty(f.instances[0], "revision", {
    get() {
      binding.dispose();
      return 1n;
    },
  });
  assert.throws(() => binding.update(), code("HIERARCHY_DISPOSED"));
  assert.equal(a.matrixWorld.elements[12], 0);
  assert.equal(f.instances[0].frees, 1);
});

for (const [label, mutate] of [
  [
    "short world bank",
    (c) => {
      c.output = new Float64Array(1);
    },
  ],
  [
    "wrong precision",
    (c) => {
      c.output = new Float32Array(32);
    },
  ],
  [
    "duplicate changed IDs",
    (c) => {
      c.changed = new Uint32Array([0, 0]);
    },
  ],
  [
    "out-of-range changed ID",
    (c) => {
      c.changed = new Uint32Array([0, 2]);
    },
  ],
  [
    "invalid work accounting",
    (c) => {
      c.stats = new Uint32Array([2, 2, 1, 0, 2]);
    },
  ],
]) {
  test(`malformed Wasm results fail closed: ${label}`, () => {
    const f = fixture(),
      a = node(),
      b = node(),
      binding = bindThreeTransformHierarchy(f.wasm, [a, b]);
    mutate(f.controls);
    assert.throws(() => binding.update(), code("HIERARCHY_WASM_OUTPUT"));
    assert.equal(a.matrixWorld.elements[12], 0);
    assert.equal(b.matrixWorld.elements[12], 0);
    binding.dispose();
  });
}

test("signed zeros and exceptional scalar values are not narrowed or sanitized by the shell", () => {
  const f = fixture(1),
    a = node();
  a.matrix.elements[1] = -0;
  a.matrix.elements[2] = Infinity;
  a.matrix.elements[3] = NaN;
  const binding = bindThreeTransformHierarchy(f.wasm, [a]);
  f.controls.output[1] = -0;
  f.controls.output[2] = Infinity;
  f.controls.output[3] = NaN;
  binding.update();
  assert.ok(Object.is(a.matrixWorld.elements[1], -0));
  assert.equal(a.matrixWorld.elements[2], Infinity);
  assert.ok(Number.isNaN(a.matrixWorld.elements[3]));
  binding.dispose();
});

test("empty membership stays a legal explicit zero-work snapshot", () => {
  const f = fixture(0),
    binding = bindThreeTransformHierarchy(f.wasm, []);
  assert.equal(binding.nodeCount, 0);
  assert.equal(binding.update().visited, 0);
  binding.dispose();
});
