import { bindThreeTransformHierarchy } from "../../../tools/compat/hierarchy_adapter.mjs";
import { bindThreeSkeletonPalettes } from "../../../tools/compat/skeleton_adapter.mjs";

function requireThat(condition, message) {
  if (!condition) throw new Error(message);
}
function equal(actual, expected, label) {
  requireThat(actual.length === expected.length, `${label}: length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    requireThat(
      Object.is(actual[i], expected[i]),
      `${label}[${i}]: ${actual[i]} !== ${expected[i]}`,
    );
  }
}

/** Real compiled-Wasm comparisons, not a mock or a JavaScript kernel fallback. */
export function runSkeletonConformance(wasm, THREE) {
  requireThat(String(THREE.REVISION) === "186", "Pinned Three.js r186 is required");
  requireThat(
    typeof wasm.f3d_batch_skeleton_palette === "function",
    "Missing compiled skeleton palette export",
  );
  requireThat(
    typeof wasm.F3dTransformHierarchy === "function",
    "Missing compiled hierarchy export",
  );
  const passed = [];
  const identity = () => new THREE.Matrix4();
  const translated = (x) => new THREE.Matrix4().makeTranslation(x, 0, 0);
  const matrix = (values) => new THREE.Matrix4().fromArray(values);
  function skeleton(worlds, inverses) {
    const bones = worlds.map((world) => {
      if (world === null) return null;
      const bone = new THREE.Bone();
      bone.matrixWorld.copy(world);
      return bone;
    });
    return new THREE.Skeleton(
      bones,
      inverses.map((inverse) => inverse.clone()),
    );
  }
  function compareCase(name, worlds, inverses, textures = false) {
    const candidate = skeleton(worlds, inverses),
      reference = skeleton(worlds, inverses);
    const binding = bindThreeSkeletonPalettes(wasm, [candidate]);
    try {
      for (let step = 0; step < 3; step++) {
        if (textures && step === 1) {
          candidate.computeBoneTexture();
          reference.computeBoneTexture();
          candidate.boneMatrices.fill(0.75, worlds.length * 16);
          reference.boneMatrices.fill(0.75, worlds.length * 16);
        }
        const storage = candidate.boneMatrices;
        reference.update();
        binding.update();
        requireThat(candidate.boneMatrices === storage, `${name}: boneMatrices identity changed`);
        equal(candidate.boneMatrices, reference.boneMatrices, name);
        if (candidate.boneTexture) {
          requireThat(
            candidate.boneTexture.version === reference.boneTexture.version,
            `${name}: texture version mismatch`,
          );
          requireThat(
            candidate.boneTexture.image.data === storage,
            `${name}: texture no longer aliases its palette`,
          );
        }
      }
      passed.push(name);
    } finally {
      binding.dispose();
      candidate.dispose();
      reference.dispose();
    }
  }
  compareCase("empty skeleton with late texture residency", [], [], true);
  compareCase(
    "multiply in f64 before f32 rounding",
    [translated(16_777_217)],
    [translated(-16_777_216)],
    true,
  );
  compareCase(
    "missing bone keeps authored inverse bind",
    [null, translated(5)],
    [translated(-2), translated(-3)],
    true,
  );
  compareCase(
    "singular inverse bind and zero scale",
    [new THREE.Matrix4().makeScale(0, -2, 3)],
    [matrix(Array(16).fill(0))],
  );
  const signed = identity();
  signed.elements[0] = -0;
  signed.elements[12] = -0;
  compareCase("signed zero", [signed], [identity()]);
  const exceptional = identity();
  exceptional.elements[0] = NaN;
  exceptional.elements[12] = Infinity;
  compareCase("IEEE exceptional matrix values", [exceptional], [identity()]);
  compareCase("finite f64 overflow at the f32 public boundary", [translated(1e100)], [identity()]);

  // Deterministic arbitrary projective matrices, not only affine/translation cases.
  let seed = 0x4f3d186;
  function random() {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return ((seed >>> 0) / 0x100000000 - 0.5) * 32;
  }
  const worlds = Array.from({ length: 128 }, () => matrix(Array.from({ length: 16 }, random)));
  const inverses = Array.from({ length: 128 }, () => matrix(Array.from({ length: 16 }, random)));
  compareCase("128 full projective joint products", worlds, inverses, true);
  equal(
    wasm.f3d_batch_skeleton_palette(new Float64Array(), new Float64Array()),
    new Float32Array(),
    "empty export",
  );
  passed.push("empty compiled export");

  // Two rigs may share bones but keep distinct inverse binds and texture banks.
  const shared = new THREE.Bone();
  shared.matrixWorld.makeTranslation(7, 8, 9);
  const candidates = [
    new THREE.Skeleton([shared], [translated(-1)]),
    new THREE.Skeleton([shared, shared], [translated(-3), translated(2)]),
  ];
  const references = [
    skeleton([shared.matrixWorld], [translated(-1)]),
    skeleton([shared.matrixWorld, shared.matrixWorld], [translated(-3), translated(2)]),
  ];
  const batch = bindThreeSkeletonPalettes(wasm, candidates);
  try {
    const report = batch.update();
    references.forEach((rig) => rig.update());
    requireThat(report.jointCount === 3 && report.skeletonCount === 2, "multi-rig counts");
    candidates.forEach((rig, i) =>
      equal(rig.boneMatrices, references[i].boneMatrices, `shared rig ${i}`),
    );
    passed.push("shared bones across multiple rig palettes");
  } finally {
    batch.dispose();
    candidates.forEach((rig) => rig.dispose());
    references.forEach((rig) => rig.dispose());
  }

  // Actual hierarchy -> retained world matrices -> Rust palette composition.
  function articulated() {
    const root = new THREE.Object3D(),
      shoulder = new THREE.Bone(),
      hand = new THREE.Bone();
    root.add(shoulder);
    shoulder.add(hand);
    root.position.set(1, 2, 3);
    root.scale.set(-1, 2, 0.5);
    shoulder.position.set(2, 0, 0);
    hand.position.set(3, 1, 0);
    const rig = new THREE.Skeleton([hand, shoulder], [translated(-3), translated(-2)]);
    rig.computeBoneTexture();
    return { root, shoulder, hand, nodes: [root, shoulder, hand], rig };
  }
  const candidate = articulated(),
    reference = articulated();
  const hierarchy = bindThreeTransformHierarchy(wasm, candidate.nodes);
  const palettes = bindThreeSkeletonPalettes(wasm, [candidate.rig]);
  try {
    for (let step = 0; step < 5; step++) {
      for (const scene of [candidate, reference]) {
        if (step === 1) scene.hand.position.x += 0.125;
        if (step === 2) scene.rig.boneInverses[0].makeTranslation(-4, 1, 0);
        if (step === 3) scene.root.add(scene.hand);
        if (step === 4) {
          scene.shoulder.matrixWorldAutoUpdate = false;
          scene.shoulder.matrixWorld.makeTranslation(17, 18, 19);
        }
      }
      reference.root.updateMatrixWorld(true);
      reference.rig.update();
      hierarchy.update();
      palettes.update();
      equal(candidate.rig.boneMatrices, reference.rig.boneMatrices, `hierarchy step ${step}`);
      requireThat(
        candidate.rig.boneTexture.version === reference.rig.boneTexture.version,
        `hierarchy step ${step}: texture version`,
      );
      passed.push(`hierarchy-to-palette step ${step}`);
    }
  } finally {
    palettes.dispose();
    hierarchy.dispose();
    candidate.rig.dispose();
    reference.rig.dispose();
  }
  return Object.freeze({
    passed: true,
    caseCount: passed.length,
    cases: passed,
    oracle: "Three.js r186 / 148ef33ecb6d2502ff796d4554abd1549c95d519",
    scope:
      "Compiled Rust/Wasm CPU palette conformance; not GPU rendering or a performance measurement",
  });
}
