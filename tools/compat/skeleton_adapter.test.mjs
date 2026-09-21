import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bindThreeSkeletonPalettes } from './skeleton_adapter.mjs';

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function matrix(x = 0) { const elements = identity(); elements[12] = x; return { elements }; }
function rig(xs = [2, 3], padding = 0) {
  return { bones: xs.map(x => ({ matrixWorld: matrix(x) })),
    boneInverses: xs.map(x => matrix(-x)),
    boneMatrices: new Float32Array(xs.length * 16 + padding).fill(-10), boneTexture: null };
}
function texture() {
  return { version: 0, set needsUpdate(value) { if (value) this.version++; } };
}
function host() {
  const controls = { hook: null, result: undefined, calls: [] };
  const wasm = { f3d_batch_skeleton_palette(worlds, inverses) {
    controls.calls.push({ worlds: worlds.slice(), inverses: inverses.slice() });
    controls.hook?.(worlds, inverses);
    return controls.result ?? Float32Array.from({ length: worlds.length }, (_, i) => i + 0.5);
  } };
  // No matrix multiplication in this fake: these are transport/lifecycle tests.
  // The real numerical oracle lives in tests/fixtures/math/skeleton_conformance.mjs.
  return { wasm, controls };
}
const errorCode = code => error => error.code === code;
function unchanged(skeleton) { assert.ok(skeleton.boneMatrices.every(x => x === -10)); }

test('one bulk call preserves skeleton/joint order and f64 inputs', () => {
  const a = rig([16_777_217, 2]), b = rig([3]); const { wasm, controls } = host();
  const binding = bindThreeSkeletonPalettes(wasm, [a, b]);
  const report = binding.update();
  assert.equal(controls.calls.length, 1);
  assert.deepEqual([12, 28, 44].map(i => controls.calls[0].worlds[i]), [16_777_217, 2, 3]);
  assert.deepEqual([12, 28, 44].map(i => controls.calls[0].inverses[i]), [-16_777_217, -2, -3]);
  assert.deepEqual(report, { skeletonCount: 2, jointCount: 3, paletteBytes: 192 });
  assert.equal(b.boneMatrices[0], 32.5); binding.dispose();
});

test('publication preserves matrix storage identity and texture padding', () => {
  const a = rig([1], 48), original = a.boneMatrices; a.boneTexture = texture();
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
  binding.update(); binding.update();
  assert.equal(a.boneMatrices, original);
  assert.deepEqual(Array.from(original.slice(16)), Array(48).fill(-10));
  assert.equal(a.boneTexture.version, 2); binding.dispose();
});

test('missing bones pack identity worlds, while inverse binds remain authored', () => {
  const a = rig([2, 3]); a.bones = [null, undefined]; const { wasm, controls } = host();
  const binding = bindThreeSkeletonPalettes(wasm, [a]); binding.update();
  assert.deepEqual(Array.from(controls.calls[0].worlds), [...identity(), ...identity()]);
  assert.equal(controls.calls[0].inverses[12], -2); binding.dispose();
});

test('a present bone with missing or null matrixWorld is not a missing bone', () => {
  for (const value of [undefined, null]) {
    const a = rig([2]); a.bones[0].matrixWorld = value; const { wasm, controls } = host();
    const binding = bindThreeSkeletonPalettes(wasm, [a]);
    assert.throws(() => binding.update(), errorCode('SKELETON_MATRIX'));
    assert.equal(controls.calls.length, 0); unchanged(a); binding.dispose();
  }
});

test('shared/repeated bones across distinct rigs are valid', () => {
  const a = rig([2]), b = rig([3, 4]); b.bones = [a.bones[0], a.bones[0]];
  const { wasm, controls } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a, b]); binding.update();
  assert.deepEqual([12, 28, 44].map(i => controls.calls[0].worlds[i]), [2, 2, 2]); binding.dispose();
});

test('membership, inverse binds and GPU residency may change between updates', () => {
  const a = rig([2]); const { wasm, controls } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
  binding.update();
  a.bones.push({ matrixWorld: matrix(9) }); a.boneInverses.push(matrix(-8));
  a.boneMatrices = new Float32Array(64); a.boneTexture = texture();
  binding.update();
  assert.equal(controls.calls[1].worlds[28], 9); assert.equal(controls.calls[1].inverses[28], -8);
  assert.equal(a.boneTexture.version, 1); assert.equal(a.boneMatrices[32], 0); binding.dispose();
});

test('empty rigs still notify textures, without calling Wasm', () => {
  const a = rig([]); a.boneTexture = texture(); const { wasm, controls } = host();
  const binding = bindThreeSkeletonPalettes(wasm, [a]);
  assert.deepEqual(binding.update(), { skeletonCount: 1, jointCount: 0, paletteBytes: 0 });
  assert.equal(a.boneTexture.version, 1); assert.equal(controls.calls.length, 0); binding.dispose();
});

test('an empty binding is a legal zero-work update', () => {
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, []);
  assert.deepEqual(binding.update(), { skeletonCount: 0, jointCount: 0, paletteBytes: 0 }); binding.dispose();
});

test('IEEE exceptional components are transported without normalization', () => {
  const a = rig([1]); a.bones[0].matrixWorld.elements.splice(0, 3, NaN, -0, Infinity);
  const { wasm, controls } = host(); controls.result = new Float32Array(16);
  controls.result.set([NaN, -0, Infinity, -Infinity]);
  const binding = bindThreeSkeletonPalettes(wasm, [a]); binding.update();
  assert.ok(Number.isNaN(controls.calls[0].worlds[0])); assert.ok(Object.is(controls.calls[0].worlds[1], -0));
  assert.ok(Number.isNaN(a.boneMatrices[0])); assert.ok(Object.is(a.boneMatrices[1], -0));
  assert.equal(a.boneMatrices[2], Infinity); binding.dispose();
});

test('a malformed later skeleton prevents publication to earlier skeletons', () => {
  const a = rig([1]), b = rig([2]); b.boneInverses = [];
  const { wasm, controls } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a, b]);
  assert.throws(() => binding.update(), errorCode('SKELETON_SHAPE'));
  unchanged(a); unchanged(b); assert.equal(controls.calls.length, 0); binding.dispose();
});

test('output shape/type failures and accidental asynchronous exports publish nothing', () => {
  for (const result of [new Float32Array(15), new Float64Array(16), [], Promise.resolve(new Float32Array(16))]) {
    const a = rig([1]); const { wasm, controls } = host(); controls.result = result;
    const binding = bindThreeSkeletonPalettes(wasm, [a]);
    assert.throws(() => binding.update(), errorCode('SKELETON_WASM_OUTPUT')); unchanged(a); binding.dispose();
  }
});

test('native failure leaves every public palette untouched and permits retry', () => {
  const a = rig([1]); const { wasm, controls } = host(); const expected = new Error('native failure');
  controls.hook = () => { throw expected; }; const binding = bindThreeSkeletonPalettes(wasm, [a]);
  assert.throws(() => binding.update(), error => error === expected); unchanged(a);
  controls.hook = null; binding.update(); assert.equal(a.boneMatrices[0], 0.5); binding.dispose();
});

test('world and inverse-bind mutations during native evaluation reject stale publication', () => {
  for (const key of ['world', 'inverse']) {
    const a = rig([1]); const { wasm, controls } = host();
    controls.hook = () => { (key === 'world' ? a.bones[0].matrixWorld : a.boneInverses[0]).elements[12] += 1; };
    const binding = bindThreeSkeletonPalettes(wasm, [a]);
    assert.throws(() => binding.update(), errorCode('SKELETON_STALE_INPUT')); unchanged(a); binding.dispose();
  }
});

test('replacement of equal-valued matrix or joint identities still rejects stale publication', () => {
  const mutations = [a => { a.bones[0] = { matrixWorld: matrix(1) }; },
    a => { a.bones[0].matrixWorld = matrix(1); },
    a => { a.bones[0].matrixWorld.elements = a.bones[0].matrixWorld.elements.slice(); },
    a => { a.boneInverses[0] = matrix(-1); },
    a => { a.bones = a.bones.slice(); }];
  for (const mutate of mutations) {
    const a = rig([1]); const { wasm, controls } = host(); controls.hook = () => mutate(a);
    const binding = bindThreeSkeletonPalettes(wasm, [a]);
    assert.throws(() => binding.update(), errorCode('SKELETON_STALE_INPUT')); unchanged(a); binding.dispose();
  }
});

test('late destination or texture replacement is rejected before publication', () => {
  for (const mutate of [a => { a.boneMatrices = new Float32Array(16).fill(-10); }, a => { a.boneTexture = texture(); }]) {
    const a = rig([1]), original = a.boneMatrices; const { wasm, controls } = host(); controls.hook = () => mutate(a);
    const binding = bindThreeSkeletonPalettes(wasm, [a]);
    assert.throws(() => binding.update(), errorCode('SKELETON_STALE_INPUT')); unchanged(a);
    assert.ok(original.every(x => x === -10)); binding.dispose();
  }
});

test('external destination mutation is not overwritten by an older computed palette', () => {
  const a = rig([1]); const { wasm, controls } = host(); controls.hook = () => { a.boneMatrices[0] = 99; };
  const binding = bindThreeSkeletonPalettes(wasm, [a]);
  assert.throws(() => binding.update(), errorCode('SKELETON_STALE_INPUT'));
  assert.equal(a.boneMatrices[0], 99); assert.equal(a.boneMatrices[1], -10); binding.dispose();
});

test('overlapping destinations reject, including distinct views of one allocation', () => {
  const a = rig([1]), b = rig([2]), storage = new ArrayBuffer(96);
  a.boneMatrices = new Float32Array(storage, 0, 16); b.boneMatrices = new Float32Array(storage, 32, 16);
  const { wasm, controls } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a, b]);
  assert.throws(() => binding.update(), errorCode('SKELETON_ALIAS'));
  assert.equal(controls.calls.length, 0); binding.dispose();
});

test('disjoint destination slices of a shared ordinary allocation are valid', () => {
  const a = rig([1]), b = rig([2]), storage = new ArrayBuffer(128);
  a.boneMatrices = new Float32Array(storage, 0, 16); b.boneMatrices = new Float32Array(storage, 64, 16);
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a, b]); binding.update();
  assert.equal(a.boneMatrices[0], 0.5); assert.equal(b.boneMatrices[0], 16.5); binding.dispose();
});

test('writes overlapping another joint input are refused', () => {
  const a = rig([1]), storage = new ArrayBuffer(128);
  a.boneMatrices = new Float32Array(storage, 0, 16);
  a.boneInverses[0].elements = new Float64Array(storage); a.boneInverses[0].elements.set(identity());
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
  assert.throws(() => binding.update(), errorCode('SKELETON_ALIAS')); binding.dispose();
});

test('joint budgets fail before the compiled boundary', () => {
  const a = rig([1, 2]); const { wasm, controls } = host();
  const binding = bindThreeSkeletonPalettes(wasm, [a], { maxJoints: 1 });
  assert.throws(() => binding.update(), errorCode('SKELETON_BUDGET')); unchanged(a);
  assert.equal(controls.calls.length, 0); binding.dispose();
  for (const maxJoints of [-1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER]) {
    assert.throws(() => bindThreeSkeletonPalettes(wasm, [], { maxJoints }), errorCode('SKELETON_BUDGET'));
  }
});

test('duplicate identities and existing owners reject without partial ownership claims', () => {
  const a = rig([1]), b = rig([2]); const { wasm } = host();
  assert.throws(() => bindThreeSkeletonPalettes(wasm, [a, a]), errorCode('SKELETON_DUPLICATE'));
  const first = bindThreeSkeletonPalettes(wasm, [a]);
  assert.throws(() => bindThreeSkeletonPalettes(wasm, [b, a]), errorCode('SKELETON_OWNERSHIP'));
  const other = bindThreeSkeletonPalettes(wasm, [b]); other.dispose(); first.dispose();
  bindThreeSkeletonPalettes(wasm, [a, b]).dispose();
});

test('missing native export never selects a JavaScript math fallback', () => {
  assert.throws(() => bindThreeSkeletonPalettes({}, [rig()]), errorCode('SKELETON_MISSING_WASM'));
});

test('reentrant update rejects without corrupting the outer batch', () => {
  const a = rig([1]); const { wasm, controls } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
  controls.hook = () => assert.throws(() => binding.update(), errorCode('SKELETON_REENTRANT'));
  binding.update(); assert.equal(a.boneMatrices[0], 0.5); binding.dispose();
});

test('disposal during computation prevents late publication and releases ownership', () => {
  const a = rig([1]); const { wasm, controls } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
  controls.hook = () => binding.dispose();
  assert.throws(() => binding.update(), errorCode('SKELETON_DISPOSED')); unchanged(a);
  assert.equal(binding.disposed, true); binding.dispose();
  assert.throws(() => binding.update(), errorCode('SKELETON_DISPOSED'));
  bindThreeSkeletonPalettes(wasm, [a]).dispose();
});

test('texture notifications occur only after all palettes have committed, in source order', () => {
  const a = rig([1]), b = rig([2]), calls = [];
  a.boneTexture = { set needsUpdate(value) { assert.equal(b.boneMatrices[0], 16.5); calls.push(['a', value]); } };
  b.boneTexture = { set needsUpdate(value) { calls.push(['b', value]); } };
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a, b]); binding.update();
  assert.deepEqual(calls, [['a', true], ['b', true]]); binding.dispose();
});

test('throwing texture setter propagates after numeric publication without pretending rollback', () => {
  const a = rig([1]); const expected = new Error('texture callback');
  a.boneTexture = { set needsUpdate(_) { throw expected; } };
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
  assert.throws(() => binding.update(), error => error === expected);
  assert.equal(a.boneMatrices[0], 0.5); a.boneTexture = null; binding.update(); binding.dispose();
});

test('shared buffers are rejected for both source and destination storage', () => {
  for (const field of ['input', 'output']) {
    const a = rig([1]);
    if (field === 'input') a.bones[0].matrixWorld.elements = new Float64Array(new SharedArrayBuffer(128));
    else a.boneMatrices = new Float32Array(new SharedArrayBuffer(64));
    const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
    assert.throws(() => binding.update(), errorCode('SKELETON_SHARED_BUFFER')); binding.dispose();
  }
});

test('a detached zero-length destination is rejected before any other rig publishes', () => {
  const a = rig([1]), b = rig([]); structuredClone(b.boneMatrices.buffer, { transfer: [b.boneMatrices.buffer] });
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a, b]);
  assert.throws(() => binding.update(), errorCode('SKELETON_SHARED_BUFFER')); unchanged(a); binding.dispose();
});

test('detachment during compute does not permit partial publication to an earlier rig', () => {
  const a = rig([1]), b = rig([2]); const { wasm, controls } = host();
  controls.hook = () => structuredClone(b.boneMatrices.buffer, { transfer: [b.boneMatrices.buffer] });
  const binding = bindThreeSkeletonPalettes(wasm, [a, b]);
  assert.throws(() => binding.update(), errorCode('SKELETON_STALE_INPUT')); unchanged(a); binding.dispose();
});

test('accessor, sparse, and nonnumeric matrices are refused without running accessors', () => {
  for (const kind of ['accessor', 'hole', 'text']) {
    const a = rig([1]);
    if (kind === 'accessor') Object.defineProperty(a.bones[0].matrixWorld.elements, '0', { get() { throw new Error('must not run'); } });
    if (kind === 'hole') delete a.bones[0].matrixWorld.elements[0];
    if (kind === 'text') a.bones[0].matrixWorld.elements[0] = '1';
    const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
    assert.throws(() => binding.update(), errorCode('SKELETON_MATRIX')); unchanged(a); binding.dispose();
  }
});

test('numeric publication bypasses overridden destination set methods', () => {
  const a = rig([1]); a.boneMatrices.set = () => { throw new Error('do not call'); };
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]); binding.update();
  assert.equal(a.boneMatrices[0], 0.5); binding.dispose();
});


test('nested readonly aliases cannot hide a later overlapping output interval', () => {
  const a = rig([1]), storage = new ArrayBuffer(160);
  a.bones[0].matrixWorld.elements = new Float64Array(storage, 0, 16);
  a.boneInverses[0].elements = new Float32Array(storage, 0, 16);
  a.boneMatrices = new Float32Array(storage, 96, 16);
  const { wasm } = host(); const binding = bindThreeSkeletonPalettes(wasm, [a]);
  assert.throws(() => binding.update(), errorCode('SKELETON_ALIAS')); binding.dispose();
});

test('large batches retain one compiled call and exact destination segmentation', () => {
  const skeletons = Array.from({ length: 4096 }, (_, i) => rig([i]));
  const { wasm, controls } = host(); const binding = bindThreeSkeletonPalettes(wasm, skeletons);
  const report = binding.update();
  assert.equal(report.jointCount, 4096); assert.equal(controls.calls.length, 1);
  for (let i = 0; i < skeletons.length; i++) assert.equal(skeletons[i].boneMatrices[0], i * 16 + 0.5);
  binding.dispose();
});
