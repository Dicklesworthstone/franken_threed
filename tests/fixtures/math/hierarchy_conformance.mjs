/** Numerical conformance using compiled Rust/Wasm and the pinned Three.js oracle.
 * Unlike hierarchy_adapter.test.mjs, this module contains no fake native class.
 * Run hierarchy.html after building /pkg/f3d_runtime.js and the pinned upstream.
 */
import { bindThreeTransformHierarchy } from "../../../tools/compat/hierarchy_adapter.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
function equal(actual, expected, label) {
  assert(actual.length === expected.length, `${label}: array length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    assert(
      Object.is(actual[i], expected[i]),
      `${label}[${i}]: expected ${expected[i]}, got ${actual[i]}`,
    );
  }
}
function rejects(action, pattern, label) {
  let failure;
  try {
    action();
  } catch (error) {
    failure = error;
  }
  assert(
    failure !== undefined && pattern.test(String(failure)),
    `${label}: expected rejection matching ${pattern}, got ${failure}`,
  );
}
const flatten = (matrices) => Float64Array.from(matrices.flatMap((matrix) => matrix.elements));

// Independent test-only traversal. Arithmetic is performed by pinned Three.js,
// not a JS translation of the Rust matrix or dirty-propagation implementation.
function oracleWorlds(THREE, parents, locals, initialWorlds, automatic) {
  const world = initialWorlds.map((matrix) => matrix.clone());
  const pending = new Set(parents.map((_, index) => index));
  while (pending.size) {
    let progress = false;
    for (const node of pending) {
      const parent = parents[node];
      if (parent !== -1 && pending.has(parent)) continue;
      if (automatic[node]) {
        if (parent === -1) world[node].copy(locals[node]);
        else world[node].multiplyMatrices(world[parent], locals[node]);
      }
      pending.delete(node);
      progress = true;
    }
    assert(progress, "Invalid cyclic test oracle input");
  }
  return flatten(world);
}

/** All groups throw on any mismatch; a missing Wasm export is a failure, not a skip. */
export function runHierarchyConformance(wasm, THREE) {
  assert(
    typeof wasm?.F3dTransformHierarchy === "function",
    "Missing compiled F3dTransformHierarchy",
  );
  assert(
    typeof wasm?.f3d_batch_compose_scalar === "function",
    "Missing compiled f64 TRS composition",
  );
  assert(
    String(THREE?.REVISION) === "186",
    "The oracle must be the repository-pinned Three.js r186",
  );
  const results = [];
  function group(name, action) {
    action();
    results.push({ name, passed: true });
  }
  const translation = (x) => new THREE.Matrix4().makeTranslation(x, 0, 0);

  group(
    "arbitrary-order forest, projective products, dirty ranges, revisions and atomic reparenting",
    () => {
      let parents = [3, -1, 0, 1, -1];
      const locals = [
        translation(2),
        translation(3),
        translation(4),
        translation(5),
        translation(6),
      ];
      locals[3].elements[0] = -2;
      locals[3].elements[3] = 0.25; // Full projective multiplication, not an affine-only fast path.
      const worlds = locals.map(() => new THREE.Matrix4());
      const flags = [1, 1, 1, 1, 1];
      const native = new wasm.F3dTransformHierarchy(
        new Int32Array(parents),
        flatten(locals),
        flatten(worlds),
        new Uint8Array(flags),
      );
      const compare = (label) =>
        equal(
          native.worldMatrices(native.revision),
          oracleWorlds(THREE, parents, locals, worlds, flags),
          label,
        );
      try {
        rejects(
          () => native.worldMatrices(native.revision),
          /not been solved/,
          "unsolved construction",
        );
        let revision = native.solve();
        assert(typeof revision === "bigint", "Revision transport must be BigInt");
        compare("initial forest");
        assert(native.solveStats()[0] === 5, "Initial solve must visit five nodes");
        native.solve();
        equal(native.solveStats(), new Uint32Array(5), "clean solve");

        locals[2].elements[12] = 9;
        native.setLocalMatrices(new Uint32Array([2]), flatten([locals[2]]));
        rejects(() => native.worldMatrices(revision), /stale/, "obsolete revision");
        rejects(() => native.worldMatrices(native.revision), /not been solved/, "unsolved edit");
        revision = native.solve();
        compare("leaf edit");
        assert(native.solveStats()[0] === 1, "An isolated leaf edit must visit exactly one node");
        equal(native.changedIndices(revision), new Uint32Array([2]), "leaf changed identities");

        locals[1].elements[12] = 10;
        native.setLocalMatrices(new Uint32Array([1, 2]), flatten([locals[1], locals[2]]));
        native.solve();
        compare("ancestor edit");
        assert(native.solveStats()[0] === 4, "Ancestor solve must skip the other root");

        const before = native.worldMatrices(native.revision);
        const beforeRevision = native.revision;
        rejects(
          () => native.reparent(new Int32Array([2, -1, 0, 1, -1])),
          /cycle/,
          "cyclic reparent",
        );
        rejects(
          () =>
            native.setLocalMatrices(
              new Uint32Array([0, 99]),
              flatten([translation(99), translation(99)]),
            ),
          /outside hierarchy/,
          "invalid batch tail",
        );
        assert(native.revision === beforeRevision, "Rejected mutations must not advance revision");
        equal(
          native.worldMatrices(beforeRevision),
          before,
          "rejected mutations retain the solved snapshot",
        );

        parents = [4, -1, 0, 1, -1];
        native.reparent(new Int32Array(parents));
        native.solve();
        compare("reparented stable IDs");
        // The exact same comparator must detect a missing parent contribution.
        const broken = native.worldMatrices(native.revision);
        broken.set(locals[0].elements, 0);
        rejects(
          () =>
            equal(
              broken,
              oracleWorlds(THREE, parents, locals, worlds, flags),
              "wrong parent control",
            ),
          /expected/,
          "wrong parent control",
        );
      } finally {
        native.free();
      }
    },
  );

  group("external world ownership, descendant propagation and re-enabling automatic worlds", () => {
    const parents = [-1, 0, 1];
    const locals = [translation(2), translation(3), translation(4)];
    const worlds = [translation(0), translation(20), translation(0)];
    const flags = [1, 0, 1];
    const native = new wasm.F3dTransformHierarchy(
      new Int32Array(parents),
      flatten(locals),
      flatten(worlds),
      new Uint8Array(flags),
    );
    const compare = (label) =>
      equal(
        native.worldMatrices(native.revision),
        oracleWorlds(THREE, parents, locals, worlds, flags),
        label,
      );
    try {
      native.solve();
      compare("external boundary");
      locals[0].elements[12] = 100;
      native.setLocalMatrices(new Uint32Array([0]), flatten([locals[0]]));
      native.solve();
      compare("ancestor cannot overwrite manual world");
      worlds[1].elements[12] = 40;
      native.setWorldMatrices(new Uint32Array([1]), flatten([worlds[1]]));
      native.solve();
      compare("manual world edit");
      equal(
        native.changedIndices(native.revision),
        new Uint32Array([1, 2]),
        "external edit identities",
      );
      flags[1] = 1;
      native.setWorldAuto(new Uint32Array([1]), new Uint8Array([1]));
      native.solve();
      compare("world-auto resumed");
      rejects(
        () => native.setWorldMatrices(new Uint32Array([1]), flatten([translation(99)])),
        /disable automatic/,
        "competing world writer",
      );
    } finally {
      native.free();
    }
  });

  group(
    "retained Object3D binding matches pinned source with mixed TRS and manual matrices",
    () => {
      function scene() {
        const nodes = Array.from({ length: 5 }, () => new THREE.Object3D());
        nodes[0].add(nodes[1], nodes[3]);
        nodes[1].add(nodes[2]);
        nodes.forEach((node, i) => node.position.set(i + 0.25, i * 0.5, -i));
        nodes[0].scale.set(-2, 3, 0.5);
        nodes[3].quaternion.set(0.25, -0.5, 0.125, 1.5); // Intentionally non-unit authored quaternion.
        nodes[2].matrixAutoUpdate = false;
        nodes[2].matrix.makeTranslation(4, 5, 6);
        nodes[2].matrix.elements[4] = 0.5;
        nodes[1].matrixWorldAutoUpdate = false;
        nodes[1].matrixWorld.makeTranslation(20, 30, 40);
        return nodes;
      }
      const candidate = scene(),
        reference = scene();
      const retained = candidate.map((node) => [node.matrix.elements, node.matrixWorld.elements]);
      const order = [2, 4, 1, 0, 3]; // Binding order is deliberately not topological.
      const binding = bindThreeTransformHierarchy(
        wasm,
        order.map((index) => candidate[index]),
      );
      function compare(label) {
        for (const root of reference.filter((node) => node.parent === null))
          root.updateMatrixWorld(true);
        candidate.forEach((node, i) => {
          equal(node.matrix.elements, reference[i].matrix.elements, `${label}: local ${i}`);
          equal(
            node.matrixWorld.elements,
            reference[i].matrixWorld.elements,
            `${label}: world ${i}`,
          );
          assert(
            node.matrix.elements === retained[i][0] && node.matrixWorld.elements === retained[i][1],
            "Retained matrix identities changed",
          );
        });
      }
      try {
        binding.update();
        compare("initial binding");
        assert(
          binding.update().visited === 0,
          "Clean binding must not recompute the native forest",
        );
        candidate[3].position.x += 1;
        reference[3].position.x += 1;
        assert(
          binding.update().visited === 1,
          "Binding leaf edit must not recompute unrelated nodes",
        );
        compare("bound leaf edit");
        candidate[1].matrixWorld.elements[12] = 70;
        reference[1].matrixWorld.elements[12] = 70;
        assert(
          binding.update().visited === 2,
          "Manual world edit must update that node and its child",
        );
        compare("bound manual world edit");
        candidate[1].matrixWorldAutoUpdate = true;
        reference[1].matrixWorldAutoUpdate = true;
        binding.update();
        compare("bound ownership transition");
      } finally {
        binding.dispose();
      }
    },
  );

  group("root signed zeros and exceptional values cross the f64 boundary unchanged", () => {
    const local = flatten([new THREE.Matrix4()]);
    local[1] = -0;
    local[3] = NaN;
    local[5] = Infinity;
    local[6] = -Infinity;
    const native = new wasm.F3dTransformHierarchy(
      new Int32Array([-1]),
      local,
      new Float64Array(),
      new Uint8Array(),
    );
    try {
      native.solve();
      equal(native.worldMatrices(native.revision), local, "exceptional root copy");
    } finally {
      native.free();
    }
  });

  group(
    "50,000-node chain solves without recursive stack growth; leaf edit visits one node",
    () => {
      const count = 50_000;
      const parents = Int32Array.from({ length: count }, (_, i) => i - 1);
      const locals = new Float64Array(count * 16);
      const step = translation(1).elements;
      for (let i = 0; i < count; i++) locals.set(step, i * 16);
      const native = new wasm.F3dTransformHierarchy(
        parents,
        locals,
        new Float64Array(),
        new Uint8Array(),
      );
      try {
        native.solve();
        assert(
          native.worldMatrices(native.revision)[(count - 1) * 16 + 12] === count,
          "Deep chain lost a parent contribution",
        );
        assert(native.solveStats()[0] === count, "Deep initial solve visit count");
        native.setLocalMatrices(new Uint32Array([count - 1]), flatten([translation(2)]));
        native.solve();
        assert(
          native.worldMatrices(native.revision)[(count - 1) * 16 + 12] === count + 1,
          "Deep leaf edit result",
        );
        assert(native.solveStats()[0] === 1, "Deep leaf update must not scan ancestors");
      } finally {
        native.free();
      }
    },
  );

  return {
    passed: true,
    oracleRevision: THREE.REVISION,
    groups: results,
    claim:
      "Exact f64 numerical and incremental-work conformance only; no rendering or performance claim.",
  };
}
