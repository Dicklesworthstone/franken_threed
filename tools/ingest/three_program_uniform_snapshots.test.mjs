import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createThreeProgramUniformSnapshots} from './three_program_uniform_snapshots.mjs';
import {threeUniformGroupState} from './three_uniform_group_state.mjs';

const dictionary = value => ({gain: {value}});
const material = (value = .25, Type = T.ShaderMaterial) => new Type({uniforms: dictionary(value)});
const limit = {code: 'THREE_PROGRAM_UNIFORM_SNAPSHOT_LIMIT'};
function fixture(t, options) {
  const owner = createThreeProgramUniformSnapshots(T, options);
  t.after(() => owner.dispose()); return owner;
}

test('pending A/B/C dictionaries retain call-time values without publishing a uniform owner', t => {
  const owner = fixture(t), m = material(), a = m.uniforms;
  const first = owner.capture([m]);
  a.gain.value = .5; const b = m.uniforms = dictionary(2);
  const second = owner.capture([m]);
  a.gain.value = .75; b.gain.value = 3; const c = m.uniforms = dictionary(4);
  const third = owner.capture([m], () => a);
  a.gain.value = b.gain.value = c.gain.value = 99;
  assert.equal(first.sourceFor(m), a); assert.equal(second.sourceFor(m), b); assert.equal(third.sourceFor(m), c);
  assert.equal(first.uniformsFor(m, a).gain.value, .25);
  assert.equal(second.uniformsFor(m, a).gain.value, .5);
  assert.equal(second.uniformsFor(m, b).gain.value, 2);
  assert.deepEqual([a, b, c].map(source => third.uniformsFor(m, source).gain.value), [.75, 3, 4]);
  assert.equal(m.uniforms, c); assert.equal(m.uniforms.gain.value, 99);
  first.release(); second.release();
  assert.equal(third.uniformsFor(m, a).gain.value, .75, 'later packets independently retain candidate values');
  third.release(); assert.equal(owner.allocatedBytes, 0);
  const next = owner.capture([m], () => a);
  assert.equal(next.uniformsFor(m, a).gain.value, 99, 'the actual published owner is captured independently of material.uniforms');
  assert.equal(next.uniformsFor(m, b), b, 'released, unpublished candidates do not survive into later captures');
  next.release(); assert.equal(owner.allocatedBytes, 0);
});

test('failed earlier compilation can select the later accepted dictionary and does not retain old candidates', t => {
  const owner = fixture(t), m = material(), a = m.uniforms;
  const first = owner.capture([m]);
  const b = m.uniforms = dictionary(.75), second = owner.capture([m]);
  first.release(); // The compiler failed before publishing A.
  assert.equal(second.sourceFor(m), b);
  assert.equal(second.uniformsFor(m, second.sourceFor(m)).gain.value, .75);
  second.release();
  const third = owner.capture([m]);
  assert.equal(third.uniformsFor(m, a), a);
  third.release(); assert.equal(owner.allocatedBytes, 0);
});

test('plain structs, arrays, pinned math, Uniforms and typed views copy data while textures retain identity', t => {
  const owner = fixture(t), texture = new T.Texture(), vector = new T.Vector3(1, 2, 3);
  const matrix = new T.Matrix4().makeTranslation(4, 5, 6);
  const bits = new Uint32Array([0x7fc0002a, 0x80000000, 0x3f800000]);
  const floats = new Float32Array(bits.buffer, 4, 2), u = new T.Uniform(vector);
  Object.assign(u, {__data: new Float32Array(10000), __offset: 32, __cache: {derived: true}});
  const m = new T.RawShaderMaterial({uniforms: {controls: {value: {vector, matrix, nested: [true, null, {floats}]}},
    alias: u, sampler: {value: texture}, samplers: {value: [texture, null]}}});
  const accepted = owner.capture([m]), copied = accepted.uniformsFor(m, m.uniforms);
  assert.equal(copied.controls.value.vector, copied.alias.value, 'aliases within accepted data keep one copied value');
  assert.notEqual(copied.alias.value, vector);
  assert.equal(copied.sampler.value, texture); assert.equal(copied.samplers.value[0], texture);
  assert.deepEqual(new Uint8Array(copied.controls.value.nested[2].floats.buffer), new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength));
  assert.ok(copied.alias instanceof T.Uniform); assert.ok(copied.alias.value instanceof T.Vector3);
  for (const key of ['__data', '__offset', '__cache']) assert.equal(Object.hasOwn(copied.alias, key), false);
  vector.set(9, 9, 9); matrix.elements.fill(9); bits.fill(0);
  assert.deepEqual(copied.alias.value.toArray(), [1, 2, 3]);
  assert.deepEqual(copied.controls.value.matrix.elements.slice(12, 15), [4, 5, 6]);
  assert.ok(Object.is(copied.controls.value.nested[2].floats[0], -0));
  assert.equal(copied.controls.value.nested[2].floats[1], 1);
  assert.ok(accepted.bytes < 4000, 'derived upload storage is not duplicated');
  accepted.release(); assert.equal(owner.allocatedBytes, 0);
});

test('uniform blocks preserve source group and uniform identities across disposal generations', t => {
  const owner = fixture(t), m = material(), scalar = {value: 7}, vector = new T.Uniform(new T.Vector3(1, 2, 3));
  const group = new T.UniformsGroup().setName('Controls').add(scalar).add([vector]);
  Object.assign(group, {__size: 32, __cache: {data: new Float32Array(10000)}});
  Object.assign(scalar, {__data: new Float32Array([7]), __offset: 0});
  m.uniformsGroups = [group];
  const first = owner.capture([m]), a = first.groupsFor(m)[0], identity = threeUniformGroupState(a);
  assert.notEqual(a, group); assert.equal(identity.source, group);
  assert.equal(identity.generation, threeUniformGroupState(group).generation);
  assert.equal(identity.uniforms.get(a.uniforms[0]), scalar);
  assert.equal(identity.uniforms.get(a.uniforms[1][0]), vector);
  assert.equal(Object.hasOwn(a, '__cache'), false); assert.equal(Object.hasOwn(a, '__size'), false);
  assert.equal(Object.hasOwn(a.uniforms[0], '__data'), false); assert.equal(Object.hasOwn(a.uniforms[0], '__offset'), false);
  group.dispose(); scalar.value = 11; vector.value.set(4, 5, 6);
  const second = owner.capture([m]), b = second.groupsFor(m)[0];
  assert.notEqual(threeUniformGroupState(b).generation, identity.generation);
  assert.equal(a.uniforms[0].value, 7); assert.equal(b.uniforms[0].value, 11);
  assert.deepEqual(a.uniforms[1][0].value.toArray(), [1, 2, 3]);
  first.release(); second.release(); assert.equal(owner.allocatedBytes, 0);
});

test('accessors and custom value classes remain live without observing their values or callbacks', t => {
  const owner = fixture(t); let calls = 0;
  const accessor = material(); Object.defineProperty(accessor.uniforms.gain, 'value', {get() { calls++; return 7; }});
  const flag = material(); Object.defineProperty(flag.uniforms.gain, 'needsUpdate', {get() { calls++; return true; }});
  const nested = material(); nested.uniforms.gain.value = {get amount() { calls++; return 2; }};
  const callable = material(); callable.uniforms.gain.value = {clone() { calls++; return 2; }};
  class Custom { constructor() { this.amount = 2; } clone() { calls++; return this; } }
  const custom = material(); custom.uniforms.gain.value = new Custom();
  const guarded = material(); Object.defineProperty(guarded, 'uniforms', {get() { calls++; return dictionary(7); }});
  assert.equal(owner.capture([accessor, flag, nested, callable, custom, guarded]), null);
  assert.equal(calls, 0); assert.equal(owner.allocatedBytes, 0);
  const valid = material(), frame = owner.capture([nested, valid]);
  assert.equal(frame.has(nested), false); assert.equal(frame.has(valid), true);
  assert.equal(frame.uniformsFor(nested, nested.uniforms), nested.uniforms);
  frame.release();
});

test('skipped uploads opt out across current, active and pending dictionaries without reading value', t => {
  const owner = fixture(t), m = material(), original = m.uniforms, first = owner.capture([m]);
  let reads = 0;
  original.gain.needsUpdate = false;
  Object.defineProperty(original.gain, 'value', {configurable: true, get() { reads++; throw Error('skipped'); }});
  m.uniforms = dictionary(.75);
  assert.equal(owner.capture([m]), null, 'an earlier candidate can become the active dictionary');
  first.release();
  assert.equal(owner.capture([m], () => original), null, 'the active source dictionary is guarded too');
  m.uniforms = original; assert.equal(owner.capture([m]), null);
  assert.equal(reads, 0); assert.equal(owner.allocatedBytes, 0);
  const huge = material(); huge.uniforms.gain = {needsUpdate: false, value: new Float32Array(1000000)};
  assert.equal(owner.capture([huge]), null, 'a skipped value does not consume the data budget');
});

for (const [name, configure] of [
  ['onBeforeCompile', m => { m.onBeforeCompile = () => assert.fail('compile callback'); }],
  ['onBeforeRender', m => { m.onBeforeRender = () => assert.fail('render callback'); }],
  ['customProgramCacheKey', m => { m.customProgramCacheKey = () => assert.fail('program key callback'); }],
  ['lights', m => { m.lights = true; }], ['fog', m => { m.fog = true; }],
])
test(`shader hooks and derived owners stay live: ${name}`, t => {
  const owner = fixture(t), m = material(); configure(m);
  assert.equal(owner.capture([m]), null); assert.equal(owner.allocatedBytes, 0);
});

test('texture upload hooks stay live without running or inspecting callback accessors', t => {
  const owner = fixture(t), m = material(), texture = new T.Texture(); let calls = 0;
  m.uniforms.gain.value = texture;
  texture.onUpdate = () => { calls++; };
  assert.equal(owner.capture([m]), null);
  Object.defineProperty(texture, 'onUpdate', {get() { calls++; return null; }});
  assert.equal(owner.capture([m]), null); assert.equal(calls, 0);
});

test('shared, resizable, detached and customized typed storage remains on the live path', t => {
  const owner = fixture(t), m = material(), arrays = [new Float32Array(new SharedArrayBuffer(16))];
  const resizable = new ArrayBuffer(16, {maxByteLength: 32});
  if (resizable.resizable) arrays.push(new Float32Array(resizable));
  const detached = new Float32Array(4); structuredClone(detached.buffer, {transfer: [detached.buffer]}); arrays.push(detached);
  const customized = new Float32Array(4); customized.slice = () => assert.fail('slice callback'); arrays.push(customized);
  class CustomArray extends Float32Array {} arrays.push(new CustomArray(4));
  for (const value of arrays) { m.uniforms.gain.value = value; assert.equal(owner.capture([m]), null); }
  assert.equal(owner.allocatedBytes, 0);
});

test('byte and material budgets reject before publishing candidates, then recover after release', t => {
  const sample = fixture(t), m = material(), measured = sample.capture([m]), bytes = measured.bytes; measured.release();
  const owner = fixture(t, {maxBytes: bytes * 2, maxCaptureBytes: bytes, maxMaterials: 1});
  const first = owner.capture([m]), second = owner.capture([m]);
  assert.equal(owner.allocatedBytes, bytes * 2);
  assert.throws(() => owner.capture([m]), limit);
  first.release(); first.release(); const third = owner.capture([m]);
  assert.equal(owner.allocatedBytes, bytes * 2);
  second.release(); third.release(); assert.equal(owner.allocatedBytes, 0);
  const another = material(), retained = owner.capture([m]);
  assert.throws(() => owner.capture([another]), limit);
  retained.release(); const next = owner.capture([another]); next.release();
  assert.throws(() => owner.capture([m, another]), limit); assert.equal(owner.allocatedBytes, 0);
});

test('failed full capture publishes neither group generations nor candidates from earlier materials', t => {
  const owner = fixture(t), a = material(), b = material(), group = new T.UniformsGroup().add(new T.Uniform(1));
  a.uniformsGroups = [group]; const failure = Error('active owner failed');
  assert.throws(() => owner.capture([a, b], current => { if (current === b) throw failure; }), error => error === failure);
  assert.equal(owner.allocatedBytes, 0); assert.equal(group._listeners, undefined);
  const previous = a.uniforms; a.uniforms = dictionary(8);
  const next = owner.capture([a]);
  assert.equal(next.uniformsFor(a, previous), previous, 'an unpublished staged dictionary is not a later candidate');
  next.release();
  const limited = fixture(t, {maxValues: 8});
  assert.throws(() => limited.capture([a]), limit); assert.equal(limited.allocatedBytes, 0);
});

test('dispose and repeated release remove every pending value and make retained frames inert', t => {
  const owner = fixture(t), m = material(), first = owner.capture([m]);
  m.uniforms = dictionary(2); const second = owner.capture([m]);
  assert.ok(owner.allocatedBytes > 0);
  owner.dispose(); owner.dispose(); first.release(); second.release();
  assert.equal(owner.allocatedBytes, 0); assert.equal(first.has(m), false);
  assert.equal(first.sourceFor(m), undefined); assert.equal(first.groupsFor(m), undefined);
  assert.equal(first.uniformsFor(m, m.uniforms), m.uniforms);
  assert.throws(() => owner.capture([m]), {code: 'THREE_PROGRAM_UNIFORM_SNAPSHOT_DISPOSED'});
});
