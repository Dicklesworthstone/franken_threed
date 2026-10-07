import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreePassSnapshots} from './three_pass_snapshot.mjs';
import {passThree as T, passFixture} from './fixtures/pass_snapshot_fixture.mjs';
const setup = options => ({...passFixture(), owner: createThreePassSnapshots(T, options)});

test('reused fullscreen uniforms retain every call without mutating live objects', () => {
  const {owner, root, camera, material} = setup();
  const first = owner.capture(root, camera);
  material.uniforms.factor.value = 2; material.uniforms.direction.value.x = 7;
  const second = owner.capture(root, camera);
  material.uniforms.factor.value = 99;
  first.install(); const installed = first.root.material;
  assert.equal(installed.uniforms.factor.value, 1); assert.equal(installed.uniforms.direction.value.x, 1);
  assert.equal(material.uniforms.factor.value, 99); assert.equal(root.material, material);
  second.install(); assert.equal(second.root, first.root); assert.equal(second.root.material, installed);
  assert.equal(installed.uniforms.factor.value, 2); assert.equal(installed.uniforms.direction.value.x, 7);
  assert.equal(material.uniforms.factor.value, 99);
  first.release(); second.release(); owner.dispose(); assert.equal(owner.allocatedBytes, 0);
});

test('material replacement and defines/shader/raster changes are captured separately', () => {
  const {owner, root, camera, material} = setup();
  material.defines.PASS = 1; material.opacity = 0.2;
  const a = owner.capture(root, camera);
  root.material = new T.RawShaderMaterial(); root.material.defines.PASS = 2; root.material.fragmentShader = 'other shader';
  const b = owner.capture(root, camera); material.defines.PASS = 99; material.opacity = 0.9;
  a.install(); assert.equal(a.root.material.defines.PASS, 1); assert.equal(a.root.material.opacity, 0.2);
  b.install(); assert.equal(b.root.material.isRawShaderMaterial, true); assert.equal(b.root.material.fragmentShader, 'other shader');
  assert.equal(b.root.material.defines.PASS, 2); assert.notEqual(b.root.material, root.material);
  a.release(); b.release(); owner.dispose();
});

test('typed-array slices, nested structs, aliases and math prototypes are copied by value', () => {
  const {owner, root, camera, material} = setup();
  const backing = new Float32Array([9, 1, 2, 9]), vector = new T.Vector3(1, 2, 3);
  material.uniforms.nested = {value: {array: backing.subarray(1, 3), aliases: [vector, vector], matrix: new T.Matrix4()}};
  const capture = owner.capture(root, camera); backing[1] = 8; vector.x = 9; capture.install();
  const value = capture.root.material.uniforms.nested.value;
  assert.deepEqual([...value.array], [1, 2]); assert.equal(value.aliases[0], value.aliases[1]);
  assert.ok(value.aliases[0] instanceof T.Vector3); assert.equal(value.aliases[0].x, 1);
  assert.ok(value.matrix instanceof T.Matrix4); assert.notEqual(value.matrix.elements, material.uniforms.nested.value.matrix.elements);
  capture.release(); owner.dispose();
});

test('textures keep identity; changing derived automatic matrices does not invalidate a pass', () => {
  const {owner, root, camera, material} = setup(), texture = new T.Texture();
  texture.clone = () => assert.fail('must never clone a texture'); material.uniforms.map = {value: texture};
  const c = owner.capture(root, camera); texture.matrix.elements[12] = 3;
  c.install(); assert.equal(c.root.material.uniforms.map.value, texture); c.check(); c.release(); owner.dispose();
});

for (const mutate of [t => t.version++, t => t.source.version++, t => t.minFilter++, t => t.offset.x++, t => t.dispose()])
  test(`source texture mutation/disposal fails explicitly: ${mutate}`, () => {
    const {owner, root, camera, material} = setup(), t = new T.Texture(); material.uniforms.map = {value: t};
    const c = owner.capture(root, camera); mutate(t);
    assert.throws(() => c.install(), {code: 'THREE_PASS_STALE'}); c.release(); owner.dispose(); assert.equal(owner.allocatedBytes, 0);
  });

test('manual texture matrices are guarded rather than silently sampled with later coordinates', () => {
  const {owner, root, camera, material} = setup(), t = new T.Texture(); t.matrixAutoUpdate = false;
  material.uniforms.map = {value: t}; const c = owner.capture(root, camera); t.matrix.elements[0] = 2;
  assert.throws(c.check, {code: 'THREE_PASS_STALE'}); c.release(); owner.dispose();
});

test('geometry upload data and draw ranges capture call-time values with stable input identity', () => {
  const {owner, root, camera, geometry} = setup();
  geometry.index = new T.BufferAttribute(new Uint16Array([0, 1, 2]), 1);
  const a = owner.capture(root, camera); const sourceIndex = geometry.index;
  geometry.attributes.position.array[0] = 20; geometry.attributes.position.version++;
  geometry.drawRange.count = 1; const b = owner.capture(root, camera);
  geometry.index = new T.BufferAttribute(new Uint16Array([2, 1, 0]), 1);
  a.install(); const stable = a.root.geometry.attributes.position, stableIndex = a.root.geometry.index;
  assert.equal(stable.array[0], -1); assert.equal(a.root.geometry.drawRange.count, Infinity);
  b.install(); assert.equal(b.root.geometry.attributes.position, stable); assert.equal(b.root.geometry.index, stableIndex);
  assert.equal(stable.array[0], 20); assert.equal(stable.version, 1); assert.equal(b.root.geometry.drawRange.count, 1);
  assert.notEqual(stableIndex, sourceIndex); assert.deepEqual([...stableIndex.array], [0, 1, 2]);
  a.release(); b.release(); owner.dispose();
});

test('interleaved attribute owner identity is captured before the application replaces it', () => {
  const {owner, root, camera, geometry} = setup();
  const source = new T.InterleavedBuffer(new Float32Array(15), 5);
  geometry.attributes.position = new T.InterleavedBufferAttribute(source, 3, 0);
  geometry.attributes.uv = new T.InterleavedBufferAttribute(source, 2, 3);
  const c = owner.capture(root, camera);
  geometry.attributes.position.data = new T.InterleavedBuffer(new Float32Array(15).fill(9), 5);
  c.install(); const attrs = c.root.geometry.attributes;
  assert.equal(attrs.position.data, attrs.uv.data); assert.equal(attrs.position.data.array[0], 0);
  c.release(); owner.dispose();
});

test('camera and mesh transforms are captured at the source boundary and do not update during install', () => {
  const {owner, root, camera} = setup(); root.position.x = 3; camera.position.x = 4;
  const c = owner.capture(root, camera); assert.equal(root.updates, 1); assert.equal(camera.updates, 1);
  root.position.x = camera.position.x = 99; c.install();
  assert.equal(c.root.matrixWorld.elements[12], 3); assert.equal(c.camera.matrixWorldInverse.elements[12], -4);
  assert.equal(c.root.matrixWorldAutoUpdate, false); assert.equal(c.camera.matrixWorldAutoUpdate, false);
  assert.equal(root.updates, 1); assert.equal(camera.updates, 1); c.release(); owner.dispose();
});

test('captures never retain or reparent the original graph or copy event listeners', () => {
  const {owner, root, camera, geometry} = setup();
  const parent = {children: [root]}; root.parent = parent; let sourceDisposed = 0;
  geometry.addEventListener('dispose', () => sourceDisposed++);
  const c = owner.capture(root, camera); c.install(); c.root.geometry.dispose();
  assert.equal(sourceDisposed, 0); assert.equal(root.parent, parent); assert.equal(parent.children[0], root);
  assert.equal(c.root.parent, null); c.release(); owner.dispose();
});

test('unadmitted roots and custom callbacks stay on the existing live-source route', () => {
  for (const edit of [f => f.root.children.push(new T.Object3D()), f => f.root.onBeforeRender = () => {},
    f => f.material.onBeforeCompile = () => {}, f => f.geometry.attributes.position.count = 100,
    f => f.geometry.attributes.uv.onUploadCallback = () => {}, f => f.camera.isArrayCamera = true]) {
    const f = setup(); edit(f); assert.equal(f.owner.capture(f.root, f.camera), null); assert.equal(f.owner.rootCount, 0); f.owner.dispose();
  }
});

test('accessor uniforms are rejected without executing getters or partially charging ownership', () => {
  const {owner, root, camera, material} = setup(); let calls = 0;
  material.uniforms.bad = {get value() { calls++; return 42; }};
  assert.throws(() => owner.capture(root, camera), {code: 'THREE_PASS_ACCESSOR'});
  assert.equal(calls, 0); assert.equal(owner.allocatedBytes, 0); assert.equal(owner.rootCount, 0); owner.dispose();
});

test('shared-memory input is rejected and cyclic plain uniform values remain bounded', () => {
  const f = setup(); f.material.uniforms.data = {value: new Float32Array(new SharedArrayBuffer(16))};
  assert.throws(() => f.owner.capture(f.root, f.camera), {code: 'THREE_PASS_VALUE'});
  const cyclic = {}; cyclic.self = cyclic; f.material.uniforms.data.value = cyclic;
  const c = f.owner.capture(f.root, f.camera); c.install();
  assert.equal(c.root.material.uniforms.data.value.self, c.root.material.uniforms.data.value); c.release(); f.owner.dispose();
});

test('aggregate budget covers pending plus installed state and recovers after release', () => {
  const {root, camera} = passFixture(), measured = createThreePassSnapshots(T), first = measured.capture(root, camera), bytes = first.bytes;
  first.release(); measured.dispose();
  const owner = createThreePassSnapshots(T, {maxBytes: bytes * 2, maxCaptureBytes: bytes + 100});
  const a = owner.capture(root, camera); a.install(); a.release(); assert.equal(owner.allocatedBytes, bytes);
  const b = owner.capture(root, camera); assert.throws(() => owner.capture(root, camera), {code: 'THREE_PASS_BUDGET'});
  b.install(); b.release(); assert.equal(owner.allocatedBytes, bytes); owner.dispose(); assert.equal(owner.allocatedBytes, 0);
});

test('release/disposal is idempotent and released captures cannot be replayed', () => {
  const {owner, root, camera} = setup(); const c = owner.capture(root, camera);
  c.release(); c.release(); assert.equal(owner.allocatedBytes, 0); assert.throws(c.install, {code: 'THREE_PASS_RELEASED'});
  owner.dispose(); owner.dispose(); assert.throws(() => owner.capture(root, camera), {code: 'THREE_PASS_DISPOSED'});
});


test('stable execution views keep renderer-owned disposal listeners across installs', () => {
  const {owner, root, camera} = setup(); const a = owner.capture(root, camera); a.install();
  let materialDisposed = 0, geometryDisposed = 0;
  a.root.material.addEventListener('dispose', () => materialDisposed++);
  a.root.geometry.addEventListener('dispose', () => geometryDisposed++);
  const b = owner.capture(root, camera); b.install();
  b.root.material.dispose(); b.root.geometry.dispose();
  assert.equal(materialDisposed, 1); assert.equal(geometryDisposed, 1); a.release(); b.release(); owner.dispose();
});

test('sparse uniform arrays keep length and are included in allocation bounds', () => {
  const {owner, root, camera, material} = setup(); const data = new Array(5); data[1] = 2;
  material.uniforms.data = {value: data}; const c = owner.capture(root, camera); c.install();
  const array = c.root.material.uniforms.data.value; assert.equal(array.length, 5); assert.equal(array[1], 2); assert.equal(0 in array, false);
  c.release(); material.uniforms.data.value = new Array(100000000);
  assert.throws(() => owner.capture(root, camera), {code: 'THREE_PASS_BUDGET'}); owner.dispose();
});

test('disposal clears installed payloads even when a renderer still holds the execution root', () => {
  const {owner, root, camera} = setup(), c = owner.capture(root, camera);
  c.install(); const executionRoot = c.root; c.release();
  assert.ok(executionRoot.geometry); owner.dispose();
  assert.equal(executionRoot.geometry, undefined); assert.equal(executionRoot.material, undefined);
  assert.equal(owner.allocatedBytes, 0); assert.equal(owner.rootCount, 0);
});
