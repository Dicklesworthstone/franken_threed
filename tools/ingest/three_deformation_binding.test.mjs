import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeDeformationBinding, inspectThreeDeformation, hasThreeDeformation} from './three_deformation_binding.mjs';
import {fixture, THREE, attr, Matrix4, Bone, BufferAttribute, InterleavedBuffer, InterleavedBufferAttribute} from './fixtures/animation/three_deformation_fixture.mjs';
const code = name => ({code: `THREE_DEFORMATION_${name}`});
const create = f => createThreeDeformationBinding(f.mesh, {three: THREE});
const transform = (m, v, w = 1) => [0,1,2].map(r => m[r]*v[0] + m[4+r]*v[1] + m[8+r]*v[2] + w*m[12+r]);

test('relative source morphs use the core pose/geometry contract without changing source arrays', () => {
  const f = fixture({morph: true}), original = f.geometry.attributes.position.array.slice(), b = create(f);
  assert.equal(hasThreeDeformation(f.mesh), true); assert.equal(b.pose.nodeCount, 1);
  assert.deepEqual([...b.pose.morphOffsets], [0,1]); assert.deepEqual([...b.pose.morphWeights], [0.5]);
  assert.deepEqual([...b.geometry.morphTargets[0].positions], [2,0,0,2,0,0,2,0,0]);
  assert.deepEqual(f.geometry.attributes.position.array, original);
  assert.throws(() => b.pose.sample(), code('CLOCK')); b.dispose();
});
test('absolute morph targets become deltas; negative and extrapolating weights are retained', () => {
  const f = fixture({morph: true}); f.geometry.morphTargetsRelative = false;
  f.geometry.morphAttributes.position[0] = attr([3,4,5,6,7,8,9,10,11]); f.mesh.morphTargetInfluences[0] = -2;
  const b = create(f); assert.ok(b.geometry.morphTargets[0].positions.every(v => v === 2));
  assert.equal(b.pose.morphWeights[0], -2); f.mesh.morphTargetInfluences[0] = 3.5; b.capture(); assert.equal(b.pose.morphWeights[0], 3.5); b.dispose();
});
test('normal-only morph channels retain their own deltas and base normals', () => {
  const f = fixture(); f.geometry.morphAttributes.normal = [attr([0,0,1,0,0,1,0,0,1])]; f.mesh.morphTargetInfluences = [0.25];
  const b = create(f); assert.deepEqual([...b.geometry.morphTargets[0].normals], [0,-1,1,0,-1,1,0,-1,1]);
  assert.equal(b.geometry.morphTargets[0].positions, undefined); b.dispose();
});
test('skin palettes apply inverse bind, bone world and mesh bind matrices in source order', () => {
  const f = fixture({skin: true}), m = f.mesh;
  m.bindMatrix.elements[12] = 2; m.bindMatrixInverse.elements[0] = 0.5; m.bindMatrixInverse.elements[12] = -5;
  m.skeleton.boneInverses[0].elements[12] = -2;
  m.skeleton.bones[0].matrixWorld = new Matrix4([0,1,0,0, -1,0,0,0, 0,0,1,0, 12,0,0,1]);
  m.matrixWorld.elements[0] = 2; m.matrixWorld.elements[12] = 10;
  const b = create(f), v = [1,2,3];
  const expected = transform(m.bindMatrixInverse.elements, transform(m.skeleton.bones[0].matrixWorld.elements,
    transform(m.skeleton.boneInverses[0].elements, transform(m.bindMatrix.elements, v))));
  assert.deepEqual(transform(b.pose.jointMatrices, v), expected);
  assert.deepEqual([...b.pose.worldMatrices], m.matrixWorld.elements);
  assert.deepEqual(b.pose.instances, [{node:0, offset:0, jointCount:1}]); b.dispose();
});
test('shared geometry does not share per-mesh morph state', () => {
  const f = fixture({morph:true}), other = new THREE.Mesh(f.geometry); other.morphTargetInfluences = [0.9];
  const a = create(f), b = createThreeDeformationBinding(other, {three: THREE});
  assert.notEqual(a.pose.morphWeights, b.pose.morphWeights); f.mesh.morphTargetInfluences[0] = 0.1; a.capture();
  assert.equal(b.pose.morphWeights[0], 0.9); a.dispose(); b.dispose();
});
test('capture reads fresh bones and inverse binds without invoking source update methods', () => {
  const f = fixture({skin:true}); f.mesh.skeleton.update = () => assert.fail('must not advance/update source');
  const b = create(f), version = b.pose.version;
  f.mesh.skeleton.bones[0].matrixWorld.elements[12] = 8; f.mesh.skeleton.boneInverses[0].elements[12] = -2;
  b.capture(); assert.equal(b.pose.jointMatrices[12], 6); assert.equal(b.pose.version, version+1); b.dispose();
});
test('invalid later joint preserves all previously published pose words and version', () => {
  const f = fixture({skin:true}); f.mesh.skeleton.bones.push(new Bone()); f.mesh.skeleton.boneInverses.push(new Matrix4());
  const b = create(f), old = [...b.pose.jointMatrices], version = b.pose.version;
  f.mesh.skeleton.bones[0].matrixWorld.elements[12] = 9; f.mesh.skeleton.bones[1].matrixWorld.elements[12] = NaN;
  assert.throws(() => b.capture(), code('VALUE')); assert.deepEqual([...b.pose.jointMatrices], old); assert.equal(b.pose.version, version); b.dispose();
});
test('normalized byte skin weights are decoded, not silently renormalized', () => {
  const f = fixture({skin:true}); f.geometry.attributes.skinWeight = new BufferAttribute(new Uint8Array([128,127,0,0,128,127,0,0,128,127,0,0]),4,true);
  const b = create(f); assert.ok(Math.abs(b.geometry.weights[0] - 128/255) < 1e-7); b.dispose();
  f.geometry.attributes.skinWeight.array[1] = 0; assert.throws(() => create(f), code('SKIN'));
});
test('interleaved native positions are decoded without rewriting storage', () => {
  const f = fixture({morph:true}), data = new InterleavedBuffer(new Float32Array([99,1,2,3,99,4,5,6,99,7,8,9]),4);
  f.geometry.attributes.position = new InterleavedBufferAttribute(data,3,1);
  const b = create(f); assert.deepEqual([...b.geometry.positions], [1,2,3,4,5,6,7,8,9]);
  data.version++; assert.equal(b.matches(), false); b.dispose();
});
test('index streams may have more entries than vertices; RGB surfaces expand to RGBA', () => {
  const f = fixture(); f.geometry.index = new BufferAttribute(new Uint16Array([0,1,2,0,1,2,0,1,2]),1);
  f.geometry.attributes.color = attr([1,0,0,0,1,0,0,0,1]); f.geometry.attributes.uv = attr([0,0,1,0,0,1],2);
  const b = createThreeDeformationBinding(f.mesh,{three:THREE,maxVertices:3});
  assert.equal(b.indexCount,9); assert.deepEqual([...b.surface.vertexColors],[1,0,0,1,0,1,0,1,0,0,1,1]); b.dispose();
});
test('static upload revisions and source geometry replacement require preparation', () => {
  const f = fixture({morph:true}), b = create(f); f.geometry.morphAttributes.position[0].needsUpdate = true;
  assert.equal(b.matches(),false); assert.throws(() => b.capture(),code('PREPARE')); b.dispose();
  const fresh = create(f); f.mesh.geometry = fixture().geometry; assert.equal(fresh.matches(),false); fresh.dispose();
});
test('source geometry disposal invalidates the binding without disposing the source twice', () => {
  const f = fixture({skin:true}), b = create(f); assert.equal(f.geometry.listeners.get('dispose').size,1);
  f.geometry.dispose(); assert.equal(b.matches(),false); assert.throws(() => b.capture(),code('PREPARE'));
  b.dispose(); b.dispose(); assert.equal(f.geometry.listeners.get('dispose').size,0); assert.equal(b.pose.disposed,true);
});
test('mismatched morph channels, color morphs and instanced meshes refuse explicitly', () => {
  for (const mutate of [f=>f.geometry.morphAttributes.normal=[attr(Array(9).fill(0)),attr(Array(9).fill(0))],
    f=>f.geometry.morphAttributes.color=[attr(Array(9).fill(0))],f=>f.mesh.isInstancedMesh=true]) {
    const f=fixture({morph:true}); mutate(f); assert.throws(()=>create(f),e=>e.code?.startsWith('THREE_DEFORMATION_'));
  }
});
test('component budgets reject before decoding oversized source data', () => {
  const f=fixture({skin:true,morph:true}); assert.throws(()=>createThreeDeformationBinding(f.mesh,{three:THREE,maxComponents:12}),code('LIMIT'));
  assert.throws(()=>inspectThreeDeformation(f.mesh,{three:THREE,maxVertices:2}),code('LIMIT'));
});
test('custom component readers and upload callbacks are not silently erased', () => {
  const f=fixture({morph:true}); f.geometry.attributes.position.getX=()=>1;
  assert.throws(()=>create(f),code('HOOK')); delete f.geometry.attributes.position.getX;
  f.geometry.attributes.position.onUploadCallback=()=>{}; assert.throws(()=>create(f),code('HOOK'));
});
test('invalid influence count and nonaffine matrices leave the captured pose unchanged', () => {
  const f=fixture({skin:true,morph:true}),b=create(f),v=b.pose.version;
  f.mesh.morphTargetInfluences=[]; assert.throws(()=>b.capture(),code('MORPH')); assert.equal(b.pose.version,v);
  f.mesh.morphTargetInfluences=[1]; f.mesh.bindMatrix.elements[3]=1;
  assert.throws(()=>b.capture(),code('SKIN')); assert.equal(b.pose.version,v); b.dispose();
});
test('foreign modules and unknown configuration do not broaden source admission', () => {
  const f=fixture({skin:true}); assert.throws(()=>createThreeDeformationBinding(f.mesh,{three:{...THREE,REVISION:'185'}}),code('SOURCE'));
  assert.throws(()=>createThreeDeformationBinding(f.mesh,{three:THREE,normalizeWeights:true}),code('OPTIONS'));
});
