import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms, hasThreeNodeProgram} from './three_node_program.mjs';
import {SunLight, SunLightNode} from './fixtures/three_sun_light.mjs';
const N = THREE.TSL;
function rig(t, options = {}) {
  const frame = new THREE.NodeFrame();
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: frame}, info: {calls: 1}};
  const owner = createThreeNodePrograms({three: THREE, renderer, ...options}); t.after(() => owner.dispose());
  const material = new THREE.MeshBasicNodeMaterial();
  const tint = N.uniform(new THREE.Color(.25, .5, .75)).setName('testTint');
  const shift = N.uniform(.125).setName('testShift');
  material.colorNode = tint; material.positionNode = N.positionLocal.add(N.vec3(shift, 0, 0));
  const object = new THREE.Mesh(new THREE.PlaneGeometry(), material), scene = new THREE.Scene(); scene.add(object);
  const camera = new THREE.PerspectiveCamera(60, 1, .1, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.position.z = 4; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {object, material, scene, camera, renderId: 1, frameId: 1, time: .2, deltaTime: .01};
  return {owner, renderer, frame, material, object, scene, camera, tint, shift, context,
    compile: () => owner.compile(material, object, context)};
}
const floats = packets => packets.flatMap(bytes => Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)));
function uniformMember(descriptor, name, stage = 'fragment') {
  const match = new RegExp(`@binding\\(\\s*(\\d+)\\s*\\)\\s*@group\\(\\s*(\\d+)\\s*\\)\\s*var<uniform>\\s+${name}\\s*:`).exec(descriptor[stage]);
  assert.ok(match, `${name} is bound in the ${stage} shader`);
  const group = Number(match[2]), member = descriptor.groups[group].uniforms.find(value => value.binding === Number(match[1]));
  assert.ok(member); return {group, ...member};
}
function uniformValues(packets, member, Type = Float32Array) {
  const bytes = packets[member.group];
  return Array.from(new Type(bytes.buffer, bytes.byteOffset + member.offset, member.byteLength / 4));
}
function dataTexture({linear = true, type = THREE.UnsignedByteType} = {}) {
  const pixels = type === THREE.FloatType ? new Float32Array(16) : type === THREE.HalfFloatType ? new Uint16Array(16) : new Uint8Array(16);
  const texture = new THREE.DataTexture(pixels, 2, 2, THREE.RGBAFormat, type);
  if (linear) texture.minFilter = texture.magFilter = THREE.LinearFilter;
  return texture;
}
function pbrRig(t, Material = THREE.MeshStandardNodeMaterial, parameters = {}) {
  const f = rig(t), material = new Material(parameters);
  material.colorNode = f.tint; f.object.material = material;
  const lights = [new THREE.AmbientLight(0xffffff, .125), new THREE.DirectionalLight(0xffffff, 2),
    new THREE.HemisphereLight(0x7799cc, 0x333333, .25), new THREE.PointLight(0xffddaa, 3),
    new THREE.SpotLight(0xaaddff, 4)];
  for (const light of lights) { light.position.set(1, 2, 3); f.scene.add(light); if (light.target) f.scene.add(light.target); }
  f.scene.updateMatrixWorld();
  const context = {...f.context, material, lights};
  return {...f, material, lights, context, compile: () => f.owner.compile(material, f.object, context)};
}
function instanceRig(t, Material = THREE.MeshBasicNodeMaterial, colorStride = 4) {
  const f = rig(t), material = new Material();
  material.colorNode = f.tint; material.positionNode = N.positionLocal.add(N.vec3(f.shift, 0, 0));
  const object = new THREE.InstancedMesh(f.object.geometry, material, 4);
  object.instanceMatrix = new THREE.StorageInstancedBufferAttribute(object.instanceMatrix.array, 16);
  object.instanceColor = new THREE.StorageInstancedBufferAttribute(3, colorStride); object.count = 3;
  f.scene.remove(f.object); f.scene.add(object); f.scene.updateMatrixWorld();
  const context = {...f.context, object, material, lights: []};
  return {...f, material, object, context, compile: () => f.owner.compile(material, object, context)};
}
test('real pinned node graphs preserve native WGSL groups, attributes and linear output', t => {
  const f = rig(t), d = f.compile();
  assert.equal(d.nodeProgram, true); assert.equal('reflection' in d, false);
  assert.match(d.vertex, /@vertex\s+fn main/); assert.match(d.fragment, /@fragment\s+fn main/);
  assert.match(d.fragment, /object\.testTint/); assert.doesNotMatch(d.fragment, /sRGBTransferOETF|LinearTosRGB/);
  assert.deepEqual(d.groups.map(g => g.name), ['render', 'object']);
  assert.deepEqual(d.attributes, [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}]);
  const captured = f.owner.capture(d, f.context);
  for (const v of [.25, .5, .75, .125]) assert.ok(floats(captured).includes(v));
  assert.equal(hasThreeNodeProgram(f.material), true);
  assert.equal(hasThreeNodeProgram(new THREE.MeshBasicNodeMaterial()), false);
});
test('uniform value edits reuse the program while old packets remain immutable', t => {
  const f = rig(t), d = f.compile(), first = f.owner.capture(d, f.context), saved = first.map(b => b.slice());
  f.tint.value.setRGB(.875, .625, .375); f.shift.value = .75;
  assert.equal(f.compile(), d);
  const second = f.owner.capture(d, {...f.context, renderId: 2});
  assert.deepEqual(first, saved); assert.ok(floats(second).includes(.875)); assert.notDeepEqual(second, first);
  f.material.colorNode = N.vec3(.3, .2, .1);
  assert.notEqual(f.compile(), d); assert.equal(f.owner.diagnostics.programs, 1);
});
test('authored BufferNodes retain all source bindings in aligned multi-buffer draw packets', t => {
  const f = rig(t);
  f.renderer._device = {limits: {minUniformBufferOffsetAlignment: 64}};
  const matrices = N.buffer(new Float32Array(32).fill(2), 'mat4', 2).setName('rawMatrices');
  const palette = N.buffer(new Float32Array(12).fill(3), 'vec4', 3).setName('rawPalette');
  const weights = N.buffer(new Float32Array(8).fill(5), 'vec4', 2).setName('rawWeights');
  f.material.positionNode = matrices.element(0).mul(N.vec4(N.positionLocal, 1)).xyz.add(palette.element(1).xyz);
  f.material.colorNode = palette.element(0).xyz.mul(weights.element(1).xyz).mul(f.tint);
  const d = f.compile(), packets = f.owner.capture(d, f.context), group = d.groups.find(value => value.name === 'object');
  assert.equal(group.uniforms.length, 5, 'three arrays retain four source stage bindings plus the ordinary uniform block');
  assert.deepEqual(group.uniforms.map(value => value.binding), group.entries.filter(value => value.buffer?.type === 'uniform').map(value => value.binding));
  let end = 0;
  for (const member of group.uniforms) {
    assert.equal(member.offset, Math.ceil(end / 64) * 64); assert.equal(member.offset % 64, 0);
    assert.equal(group.entries[member.binding].buffer.minBindingSize, member.byteLength);
    assert.ok(packets[group.index].subarray(end, member.offset).every(value => value === 0));
    end = member.offset + member.byteLength;
  }
  assert.equal(group.byteLength, end); assert.equal(packets[group.index].byteLength, end);
  const fragmentPalette = uniformMember(d, 'rawPalette'), vertexPalette = uniformMember(d, 'rawPalette', 'vertex');
  assert.notEqual(fragmentPalette.binding, vertexPalette.binding);
  for (const [name, stage, count, value] of [['rawMatrices', 'vertex', 32, 2], ['rawPalette', 'fragment', 12, 3],
    ['rawPalette', 'vertex', 12, 3], ['rawWeights', 'fragment', 8, 5]])
    assert.deepEqual(uniformValues(packets, uniformMember(d, name, stage)), Array(count).fill(value));
  assert.ok(d.groups.every(value => value.uniforms.length !== 1 || value.uniforms[0].offset === 0));
});
test('raw uniform arrays capture current data and compatible replacements without needsUpdate', t => {
  const f = rig(t), values = N.buffer(new Float32Array(8).fill(1), 'vec4', 2).setName('rawValues');
  f.material.fragmentNode = values.element(0);
  const d = f.compile(), member = uniformMember(d, 'rawValues'), first = f.owner.capture(d, f.context), saved = first.map(bytes => bytes.slice());
  values.value.fill(2); assert.equal(f.compile(), d);
  const second = f.owner.capture(d, {...f.context, renderId: 2});
  values.value = new Float32Array(8).fill(3); assert.equal(f.compile(), d);
  const third = f.owner.capture(d, {...f.context, renderId: 3}); values.value.fill(9);
  assert.deepEqual(uniformValues(first, member), Array(8).fill(1));
  assert.deepEqual(uniformValues(second, member), Array(8).fill(2));
  assert.deepEqual(uniformValues(third, member), Array(8).fill(3)); assert.deepEqual(first, saved);
});
test('raw uniform partial ranges preserve zero initialization and previous GPU-visible bytes', t => {
  const f = rig(t), values = N.buffer(Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), 'vec4', 2).setName('rawRanges');
  values.addUpdateRange(1, 2); f.material.fragmentNode = values.element(0);
  const d = f.compile(), member = uniformMember(d, 'rawRanges'), first = f.owner.capture(d, f.context);
  assert.deepEqual(uniformValues(first, member), [0, 2, 3, 0, 0, 0, 0, 0]); assert.deepEqual(values.updateRanges, []);
  values.value.fill(9); values.addUpdateRange(4, 2); values.addUpdateRange(6, 2);
  const second = f.owner.capture(d, {...f.context, renderId: 2});
  assert.deepEqual(uniformValues(second, member), [0, 2, 3, 0, 9, 9, 9, 9]);
  values.value = new Float32Array(8).fill(6); values.addUpdateRange(0, 1);
  assert.equal(f.compile(), d);
  const third = f.owner.capture(d, {...f.context, renderId: 3});
  assert.deepEqual(uniformValues(third, member), [6, 2, 3, 0, 9, 9, 9, 9]);
  assert.deepEqual(uniformValues(first, member), [0, 2, 3, 0, 0, 0, 0, 0]);
  assert.deepEqual(uniformValues(f.owner.capture(d, {...f.context, renderId: 4}), member), Array(8).fill(6));
});
test('cross-stage raw buffer bindings consume shared ranges in pinned source order', t => {
  const f = rig(t), values = N.buffer(Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), 'vec4', 2).setName('rawStages');
  f.material.positionNode = N.positionLocal.add(values.element(0).xyz); f.material.fragmentNode = values.element(1);
  values.addUpdateRange(0, 4);
  const d = f.compile(), fragment = uniformMember(d, 'rawStages'), vertex = uniformMember(d, 'rawStages', 'vertex');
  assert.equal(fragment.group, vertex.group); assert.ok(fragment.binding < vertex.binding);
  const first = f.owner.capture(d, f.context);
  assert.deepEqual(uniformValues(first, fragment), [1, 2, 3, 4, 0, 0, 0, 0]);
  assert.deepEqual(uniformValues(first, vertex), [1, 2, 3, 4, 5, 6, 7, 8]);
  values.value.fill(9); values.addUpdateRange(4, 4);
  const second = f.owner.capture(d, {...f.context, renderId: 2});
  assert.deepEqual(uniformValues(second, fragment), [1, 2, 3, 4, 9, 9, 9, 9]);
  assert.deepEqual(uniformValues(second, vertex), Array(8).fill(9));
});
test('raw custom uniform groups retain source version-gated uploads', t => {
  const f = rig(t), group = N.uniformGroup('manualPalette'), values = N.buffer(new Float32Array(8).fill(1), 'vec4', 2).setName('manualValues').setGroup(group);
  f.material.fragmentNode = values.element(0);
  const d = f.compile(), member = uniformMember(d, 'manualValues'); f.owner.capture(d, f.context);
  values.value.fill(2); values.addUpdateRange(0, 1);
  assert.deepEqual(uniformValues(f.owner.capture(d, {...f.context, renderId: 2}), member), Array(8).fill(1));
  assert.equal(values.updateRanges.length, 1, 'skipped source groups do not consume update ranges');
  group.needsUpdate = true;
  assert.deepEqual(uniformValues(f.owner.capture(d, {...f.context, renderId: 3}), member), [2, 1, 1, 1, 1, 1, 1, 1]);
  assert.deepEqual(values.updateRanges, []);
});
test('raw uniform layouts retain matrix padding, integer bytes and an omitted final vec3 padding word', t => {
  for (const [type, Type, stride] of [['vec3', Float32Array, 4], ['color', Float32Array, 4], ['vec4', Float32Array, 4],
    ['ivec3', Int32Array, 4], ['ivec4', Int32Array, 4], ['uvec3', Uint32Array, 4], ['uvec4', Uint32Array, 4],
    ['mat2', Float32Array, 4], ['mat3', Float32Array, 12], ['mat4', Float32Array, 16]]) {
    const f = rig(t), values = N.buffer(new Type(2 * stride).fill(7), type, 2).setName('rawLayout');
    if (type.startsWith('mat')) {
      const count = Number(type.at(-1)), vector = count === 2 ? N.positionLocal.xy : count === 3 ? N.positionLocal : N.vec4(N.positionLocal, 1);
      const transformed = values.element(1).mul(vector);
      f.material.positionNode = count === 2 ? N.vec3(transformed, 0) : transformed.xyz;
    } else f.material.fragmentNode = N.vec4(values.element(1));
    const d = f.compile(), member = uniformMember(d, 'rawLayout', type.startsWith('mat') ? 'vertex' : 'fragment');
    assert.deepEqual(uniformValues(f.owner.capture(d, f.context), member, Type), Array(2 * stride).fill(7), type);
  }
  const f = rig(t), backing = new Float32Array(12), view = new Float32Array(backing.buffer, 4, 7); view.fill(4);
  const values = N.buffer(view, 'vec3', 2).setName('rawTail'); f.material.fragmentNode = N.vec4(values.element(1), 1);
  const d = f.compile(), member = uniformMember(d, 'rawTail');
  assert.equal(member.byteLength, 32); assert.deepEqual(uniformValues(f.owner.capture(d, f.context), member), [4, 4, 4, 4, 4, 4, 4, 0]);
  values.value = new Float32Array(8).fill(6); assert.equal(f.compile(), d); f.owner.capture(d, {...f.context, renderId: 2});
  values.value = view; assert.equal(f.compile(), d);
  assert.deepEqual(uniformValues(f.owner.capture(d, {...f.context, renderId: 3}), member), [4, 4, 4, 4, 4, 4, 4, 6]);
});
test('raw buffer structural changes and Fn-created uniform aliases invalidate before callbacks', t => {
  const f = rig(t), nodes = {}; let shared = true, updates = 0;
  f.material.positionNode = null; f.material.colorNode = null;
  f.material.vertexNode = N.vec4(N.attribute('position', 'vec3'), 1);
  f.material.fragmentNode = N.Fn(() => {
    nodes.left ??= N.buffer(new Float32Array(8).fill(1), 'vec4', 2);
    nodes.right ??= N.buffer(nodes.left.value, 'vec4', 2);
    nodes.left.getUniformHash = () => 'left-uniform';
    nodes.right.getUniformHash = () => shared ? 'left-uniform' : 'right-uniform';
    return nodes.left.element(0).add(nodes.right.element(1));
  })();
  const aliased = f.compile(); assert.equal(aliased.groups.flatMap(group => group.uniforms).length, 1);
  shared = false;
  assert.throws(() => f.owner.capture(aliased, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const split = f.compile(); assert.notEqual(split, aliased); assert.equal(split.groups.flatMap(group => group.uniforms).length, 2);
  nodes.right.value = new Float32Array(8).fill(2); assert.equal(f.compile(), split, 'typed-array identity never aliases source BufferNodes');
  shared = true; assert.notEqual(f.compile(), split);
  f.material.fragmentNode = N.vec4(nodes.left.element(0).xyz, 1).mul(N.uniform(1).onObjectUpdate(() => { updates++; return 1; }));
  const initial = f.compile(); nodes.left.bufferCount = 1;
  assert.throws(() => f.owner.capture(initial, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, 0);
  const counted = f.compile(); nodes.left.nodeType = nodes.left.bufferType = 'vec3';
  assert.throws(() => f.owner.capture(counted, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const typed = f.compile(); nodes.left.setGroup(N.uniformGroup('differentGroup', 2, 'object'));
  assert.throws(() => f.owner.capture(typed, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.notEqual(f.compile(), typed);
});
test('raw buffer updates may replace compatible arrays but cannot change captured layouts', t => {
  const f = rig(t), values = N.buffer(new Float32Array(8), 'vec4', 2).setName('updatedRaw'); let next = new Float32Array(8).fill(3), calls = 0;
  const gain = N.uniform(1).onObjectUpdate(() => { calls++; values.value = next; return 1; });
  f.material.fragmentNode = values.element(0).mul(gain);
  const d = f.compile(), member = uniformMember(d, 'updatedRaw');
  assert.deepEqual(uniformValues(f.owner.capture(d, f.context), member), Array(8).fill(3));
  next = new Float32Array(12); assert.throws(() => f.owner.capture(d, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 2);
  assert.throws(() => f.owner.capture(d, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 2, 'already stale layouts refuse before observable updates');
});
test('invalid raw uniform arrays and read-only writes refuse before source updates', t => {
  const mutations = [node => { node.value = new Float64Array(8); }, node => { node.value = new Uint32Array(8); },
    node => { node.value = new Float32Array(3); }, node => { node.bufferCount = 0; }, node => { node.bufferCount = 2.5; },
    node => { node.nodeType = 'float'; }, node => { node.nodeType = 'vec2'; }, node => { node.nodeType = 'bvec4'; },
    node => { node.value = new Float32Array(new SharedArrayBuffer(32)); },
    node => { node.value = new Float32Array(new ArrayBuffer(32, {maxByteLength: 64})); },
    node => { node.addUpdateRange(7, 2); }, node => { node.addUpdateRange(-1, 1); }, node => { node.addUpdateRange(0, .5); }];
  for (const mutate of mutations) {
    const f = rig(t), node = N.buffer(new Float32Array(8), 'vec4', 2); let updates = 0;
    f.material.fragmentNode = node.element(0).mul(N.uniform(1).onObjectUpdate(() => { updates++; return 1; }));
    const d = f.compile(); mutate(node);
    assert.throws(() => f.owner.capture(d, f.context), {code: 'THREE_NODE_PROGRAM_UNIFORM'});
    assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_UNIFORM'}); assert.equal(updates, 0);
  }
  const f = rig(t), node = N.buffer(new Float32Array(8), 'vec4', 2);
  f.material.fragmentNode = N.Fn(() => { node.element(0).assign(N.vec4(1)); return node.element(0); })();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_UNIFORM'});
  f.material.fragmentNode = N.Fn(() => { const copy = node.element(0).toVar(); copy.assign(N.vec4(1)); return copy; })();
  assert.doesNotThrow(f.compile, 'a local copy remains writable');
});
test('uniform limits apply per binding and per pipeline while packed groups may exceed 64 KiB', t => {
  const f = rig(t), matrices = N.buffer(new Float32Array(16384), 'mat4', 1024).setName('largeMatrices');
  f.material.positionNode = matrices.element(0).mul(N.vec4(N.positionLocal, 1)).xyz;
  const d = f.compile(), group = d.groups.find(value => value.byteLength > 65536);
  assert.ok(group); assert.ok(group.uniforms.every(member => member.byteLength <= 65536));
  assert.equal(f.owner.capture(d, f.context)[group.index].byteLength, group.byteLength);
  f.renderer._device = {limits: {maxDynamicUniformBuffersPerPipelineLayout: 2}};
  f.material.needsUpdate = true; assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
  f.renderer._device = {limits: {maxUniformBuffersPerShaderStage: 1}};
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
  f.renderer._device = {limits: {maxUniformBufferBindingSize: 65520}};
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
  const bounded = rig(t, {maxUniformBytes: 65536}); bounded.material.positionNode = matrices.element(0).mul(N.vec4(N.positionLocal, 1)).xyz;
  assert.throws(bounded.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
});
test('Standard nodes retain source direct-light PBR, authored surface uniforms and the real RG16F DFG resource', t => {
  const f = pbrRig(t);
  f.material.metalnessNode = N.uniform(.375).setName('pbrMetalness');
  f.material.roughnessNode = N.uniform(.625).setName('pbrRoughness');
  f.material.emissiveNode = N.vec3(.015625, 0, 0);
  f.material.normalNode = N.normalView;
  const d = f.compile(), draw = f.owner.captureDraw(d, f.context);
  assert.match(d.fragment, /fn D_GGX/); assert.match(d.fragment, /fn V_GGX_SmithCorrelated/);
  assert.match(d.fragment, /object\.pbrMetalness/); assert.match(d.fragment, /object\.pbrRoughness/);
  assert.ok(floats(draw.uniforms).includes(.375)); assert.ok(floats(draw.uniforms).includes(.625));
  const sampled = draw.textures.filter(binding => binding.isSampledTexture);
  assert.equal(sampled.length, 1);
  const lut = sampled[0].texture;
  assert.ok(lut instanceof THREE.DataTexture); assert.equal(lut.name, 'DFG_LUT');
  assert.equal(lut.type, THREE.HalfFloatType); assert.equal(lut.format, THREE.RGFormat);
  assert.equal(lut.image.width, 16); assert.equal(lut.image.height, 16);
  assert.ok(lut.image.data instanceof Uint16Array); assert.equal(lut.image.data.length, 512);
  assert.equal(lut.minFilter, THREE.LinearFilter); assert.equal(lut.magFilter, THREE.LinearFilter);
  assert.equal(lut.generateMipmaps, false);
  assert.equal(draw.textures.filter(binding => binding.isSampler && !binding.isSampledTexture && binding.texture === lut).length, 1);
  const group = d.groups[sampled[0].group];
  assert.deepEqual(group.entries[sampled[0].binding].texture, {sampleType: 'float', viewDimension: '2d', multisampled: false});
  assert.equal(hasThreeNodeProgram(new THREE.MeshStandardNodeMaterial()), false);
  assert.equal(hasThreeNodeProgram(new THREE.MeshPhysicalNodeMaterial()), false);
  const saved = draw.uniforms.map(bytes => bytes.slice());
  f.material.metalnessNode.value = .875; f.material.roughnessNode.value = .25;
  assert.equal(f.compile(), d);
  const changed = f.owner.captureDraw(d, {...f.context, renderId: 2});
  assert.ok(floats(changed.uniforms).includes(.875)); assert.notDeepEqual(changed.uniforms, draw.uniforms);
  assert.deepEqual(draw.uniforms, saved); assert.equal(changed.textures[0].texture, lut);
});
test('Physical authored lobes keep the pinned shader functions and share DFG bindings with Standard', t => {
  const f = pbrRig(t, THREE.MeshPhysicalNodeMaterial), material = f.material;
  const values = {metalnessNode: .25, roughnessNode: .5, clearcoatNode: .75, clearcoatRoughnessNode: .125,
    sheenRoughnessNode: .625, iridescenceNode: .375, iridescenceIORNode: 1.375, iridescenceThicknessNode: 312,
    retroreflectivityNode: .5625, iorNode: 1.625, specularIntensityNode: .875};
  for (const [slot, value] of Object.entries(values)) material[slot] = N.uniform(value).setName(`physical_${slot}`);
  material.sheenNode = N.uniform(new THREE.Color(.125, .25, .375)).setName('physical_sheen');
  material.anisotropyNode = N.uniform(new THREE.Vector2(.5, .25)).setName('physical_anisotropy');
  material.specularColorNode = N.uniform(new THREE.Color(.75, .625, .5)).setName('physical_specular');
  material.clearcoatNormalNode = N.vec3(.125, 0, 1).normalize();
  const d = f.compile(), draw = f.owner.captureDraw(d, f.context);
  for (const pattern of [/fn D_Charlie/, /fn evalIridescence/, /fn D_GGX_Anisotropic/,
    /clearcoatSpecularDirect/, /Retroreflectivity/, /physical_anisotropy/, /physical_sheen/, /physical_specular/])
    assert.match(d.fragment, pattern);
  for (const [slot, value] of Object.entries(values)) {
    assert.match(d.fragment, new RegExp(`object\\.physical_${slot}`));
    assert.ok(floats(draw.uniforms).includes(value), slot);
  }
  assert.ok(d.attributes.some(attribute => attribute.name === 'uv'));
  const sampled = draw.textures.filter(binding => binding.isSampledTexture);
  assert.equal(sampled.length, 1, 'base and clearcoat DFG reads share the source texture binding');
  const standard = new THREE.MeshStandardNodeMaterial(); standard.colorNode = f.tint;
  const standardProgram = f.owner.compile(standard, f.object, {...f.context, material: standard});
  assert.equal(f.owner.textures(standardProgram).find(binding => binding.isSampledTexture).texture, sampled[0].texture);
});
for (const [feature, shader] of [['clearcoat', /clearcoatSpecularDirect/], ['sheen', /sheenSpecularDirect/],
  ['iridescence', /fn evalIridescence/], ['anisotropy', /fn D_GGX_Anisotropic/], ['retroreflectivity', /Retroreflectivity/]]) {
  test(`Physical ${feature} activation rebuilds the graph, while enabled value edits remain uniforms`, t => {
    const f = pbrRig(t, THREE.MeshPhysicalNodeMaterial), before = f.compile();
    assert.doesNotMatch(before.fragment, shader);
    f.material[feature] = .5;
    const enabled = f.compile(); assert.notEqual(enabled, before); assert.match(enabled.fragment, shader);
    const first = f.owner.capture(enabled, f.context), saved = first.map(bytes => bytes.slice());
    f.material[feature] = .75;
    assert.equal(f.compile(), enabled);
    const second = f.owner.capture(enabled, {...f.context, renderId: 2});
    assert.notDeepEqual(second, first); assert.deepEqual(first, saved);
    f.material[feature] = 0;
    const disabled = f.compile(); assert.notEqual(disabled, enabled); assert.doesNotMatch(disabled.fragment, shader);
    assert.equal(f.owner.diagnostics.programs, 1);
  });
}
test('PBR authored sampled values coexist with the source DFG and preserve mutable surface references', t => {
  const f = pbrRig(t, THREE.MeshPhysicalNodeMaterial), image = dataTexture(), sample = N.texture(image);
  f.material.colorNode = sample.rgb; f.material.roughnessNode = sample.a;
  f.material.clearcoat = .5; f.material.clearcoatNormalNode = N.normalMap(sample);
  const d = f.compile(), first = f.owner.captureDraw(d, f.context);
  assert.deepEqual(new Set(first.textures.map(binding => binding.texture.name)), new Set(['', 'DFG_LUT']));
  const saved = first.uniforms.map(bytes => bytes.slice());
  f.material.ior = 1.75; f.material.specularIntensity = .625;
  const replacement = dataTexture(); sample.value = replacement;
  assert.equal(f.compile(), d);
  const second = f.owner.captureDraw(d, {...f.context, renderId: 2});
  assert.ok(floats(second.uniforms).includes(1.75)); assert.ok(floats(second.uniforms).includes(.625));
  assert.ok(second.textures.some(binding => binding.texture === replacement));
  assert.ok(first.textures.some(binding => binding.texture === image)); assert.deepEqual(first.uniforms, saved);
});
test('PBR scene-color and environment dependencies refuse before source construction', t => {
  const f = pbrRig(t, THREE.MeshPhysicalNodeMaterial);
  for (const feature of ['transmission', 'dispersion']) {
    f.material[feature] = .25;
    assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_TRANSMISSION'});
    f.material[feature] = 0;
    f.material[`${feature}Node`] = N.float(0);
    assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_TRANSMISSION'});
    f.material[`${feature}Node`] = null;
  }
  for (const slot of ['thicknessNode', 'attenuationDistanceNode', 'attenuationColorNode']) {
    f.material[slot] = N.float(1);
    assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_GRAPH'}); f.material[slot] = null;
  }
  for (const slot of ['environment', 'environmentNode']) {
    f.scene[slot] = slot === 'environment' ? dataTexture() : N.vec3(1);
    assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT'}); f.scene[slot] = null;
  }
  f.material.lights = false; f.scene.environment = dataTexture();
  assert.doesNotThrow(() => f.compile(), 'the source ignores scene lighting when material.lights is false');
});
test('matrix, integer and boolean uniforms retain padded native values', t => {
  const f = rig(t), transform = N.uniform(new THREE.Matrix3().set(2, 0, 0, 0, 3, 0, 0, 0, 4)).setName('testMatrix');
  const choice = N.uniform(true, 'bool').setName('testChoice'), value = N.uniform(7, 'int').setName('testInteger');
  f.material.positionNode = transform.mul(N.positionLocal);
  f.material.fragmentNode = N.vec4(N.select(choice, value.toFloat(), N.float(2)), 0, 0, 1);
  const d = f.compile(), packets = f.owner.capture(d, f.context);
  assert.match(d.vertex, /mat3x3<f32>/); assert.match(d.fragment, /i32/);
  assert.ok(floats(packets).includes(3)); assert.ok(floats(packets).includes(4));
  assert.ok(packets.some(b => Array.from(new Int32Array(b.buffer)).includes(7)));
});
test('custom float attributes keep builder-assigned locations', t => {
  const f = rig(t);
  f.object.geometry.setAttribute('weight', new THREE.Float32BufferAttribute([.1, .2, .3, .4], 1));
  f.material.fragmentNode = N.vec4(N.attribute('weight', 'float'), N.uv(), 1);
  const d = f.compile();
  assert.ok(d.attributes.some(a => a.name === 'weight' && a.components === 1));
  assert.ok(d.attributes.some(a => a.name === 'uv' && a.components === 2));
});
test('base unlit NodeMaterial supports clip-space vertex and fragment graphs with no uniform groups', t => {
  const f = rig(t), material = new THREE.NodeMaterial();
  material.vertexNode = N.vec4(N.attribute('position', 'vec3'), 1);
  material.fragmentNode = N.vec4(.25, .5, .75, 1);
  const d = f.owner.compile(material, f.object, f.context);
  assert.equal(d.groups.length, 0); assert.equal(d.attributes[0].name, 'position');
  assert.deepEqual(f.owner.capture(d, {...f.context, material}), []);
});
test('source NodeFrame preserves FRAME, RENDER and OBJECT updates and all context identities', t => {
  const f = rig(t), calls = {frame: 0, render: 0, object: 0}, seen = [];
  const nodes = Object.keys(calls).map(kind => {
    const node = N.uniform(0); node.updateType = kind;
    node.update = current => { calls[kind]++; seen.push({renderer: current.renderer, scene: current.scene,
      camera: current.camera, object: current.object, material: current.material, compute: current.compute}); node.value = calls[kind]; };
    return node;
  });
  f.material.colorNode = N.vec3(...nodes); const d = f.compile();
  f.owner.capture(d, f.context); f.owner.capture(d, f.context);
  assert.deepEqual(calls, {frame: 1, render: 1, object: 2});
  f.owner.capture(d, {...f.context, renderId: 2});
  assert.deepEqual(calls, {frame: 1, render: 2, object: 3});
  f.owner.capture(d, {...f.context, renderId: 3, frameId: 2});
  assert.deepEqual(calls, {frame: 2, render: 3, object: 4});
  for (const context of seen) assert.deepEqual(context, {renderer: f.renderer, scene: f.scene,
    camera: f.camera, object: f.object, material: f.material, compute: null});
});
test('before/current/after false returns retain the pinned retry-marker behavior', t => {
  const f = rig(t), node = N.uniform(0), calls = {before: 0, current: 0, after: 0};
  node.updateBeforeType = 'render'; node.updateType = 'render'; node.updateAfterType = 'render';
  node.updateBefore = () => ++calls.before === 1 ? false : undefined;
  node.update = () => { node.value = ++calls.current; return calls.current === 1 ? false : undefined; };
  node.updateAfter = () => ++calls.after === 1 ? false : undefined;
  f.material.colorNode = N.vec3(node); const d = f.compile();
  for (let i = 0; i < 3; i++) { f.owner.capture(d, f.context); f.owner.after(d, f.context); }
  assert.deepEqual(calls, {before: 2, current: 2, after: 2});
});
test('shared update references are updated once per render across distinct nodes', t => {
  const f = rig(t), reference = {}, a = N.uniform(.1), b = N.uniform(.2); let calls = 0;
  for (const node of [a, b]) { node.updateType = 'render'; node.updateReference = () => reference; node.update = () => { calls++; }; }
  f.material.colorNode = N.vec3(a, b, 0); const d = f.compile(); f.owner.capture(d, f.context);
  assert.equal(calls, 1);
});
test('same material on two objects captures different object-update values', t => {
  const f = rig(t), other = new THREE.Mesh(f.object.geometry, f.material); other.position.x = 2; other.updateMatrixWorld();
  f.material.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(({object}, uniform) => uniform.value.setRGB(object === f.object ? .25 : .75, .5, 0));
  const d1 = f.compile(), d2 = f.owner.compile(f.material, other, {...f.context, object: other});
  const first = f.owner.capture(d1, f.context), second = f.owner.capture(d2, {...f.context, object: other});
  assert.ok(floats(first).includes(.25)); assert.ok(floats(second).includes(.75));
  assert.notDeepEqual(first, second);
});
test('pinned sampled graphs preserve mixed binding slots and separate vertex/fragment visibility', t => {
  const f = rig(t), texture = dataTexture();
  f.material.fragmentNode = N.texture(texture).mul(.75);
  f.material.positionNode = N.positionLocal.add(N.texture(texture).r.mul(.1));
  const d = f.compile(), group = d.groups.find(group => group.name === 'object');
  assert.deepEqual(group.entries.map(entry => [entry.binding, entry.buffer ? 'uniform' : entry.texture ? 'texture' : 'sampler', entry.visibility]),
    [[0, 'sampler', 2], [1, 'texture', 2], [2, 'uniform', 7], [3, 'sampler', 1], [4, 'texture', 1]]);
  assert.equal(group.entries[2].buffer.minBindingSize, group.byteLength);
  assert.match(d.vertex, /@binding\( 4 \).*@group\( 1 \)/); assert.match(d.vertex, /textureSampleLevel/);
  assert.match(d.fragment, /@binding\( 1 \).*@group\( 1 \)/); assert.match(d.fragment, /textureSample\(/);
  const captured = f.owner.captureDraw(d, f.context);
  assert.deepEqual(captured.uniforms.map(bytes => bytes.byteLength), d.groups.map(group => group.byteLength));
  assert.deepEqual(captured.textures.map(value => [value.group, value.binding, value.isSampledTexture]),
    [[1, 0, false], [1, 1, true], [1, 3, false], [1, 4, true]]);
  assert.ok(captured.textures.every(value => value.texture === texture && value.isSampler && value.store === false));
});
test('resource-only groups and nearest texture loads retain native binding cardinality', t => {
  for (const linear of [true, false]) {
    const f = rig(t), texture = dataTexture({linear}), material = new THREE.NodeMaterial();
    material.vertexNode = N.vec4(N.attribute('position', 'vec3'), 1);
    material.fragmentNode = N.texture(texture, N.vec2(.25, .75)).setUpdateMatrix(false);
    const d = f.owner.compile(material, f.object, f.context), captured = f.owner.captureDraw(d, {...f.context, material});
    assert.equal(d.groups.length, 1); assert.equal(d.groups[0].byteLength, 0);
    assert.equal(d.groups[0].entries.some(entry => entry.buffer), false);
    assert.equal(d.groups[0].entries.length, linear ? 2 : 1);
    assert.equal(captured.uniforms.length, 1); assert.equal(captured.uniforms[0].byteLength, 0);
    assert.equal(captured.textures.length, linear ? 2 : 1);
    assert.match(d.fragment, linear ? /textureSample\(/ : /textureLoad\(/);
  }
});
test('source texture channel, UV matrix and native color-space metadata survive capture', t => {
  const f = rig(t), texture = dataTexture(); texture.channel = 1; texture.colorSpace = THREE.SRGBColorSpace;
  texture.repeat.set(2, 3); texture.offset.set(.125, .375);
  f.object.geometry.setAttribute('uv1', f.object.geometry.attributes.uv);
  const inspected = [];
  f.renderer._getSampledTextureInputs = () => ({layoutOf(binding) {
    inspected.push(binding.texture); return {texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}};
  }});
  f.material.colorNode = N.texture(texture);
  const d = f.compile(), first = f.owner.captureDraw(d, f.context), saved = first.uniforms.map(bytes => bytes.slice());
  assert.ok(d.attributes.some(attribute => attribute.name === 'uv1'));
  assert.equal(d.attributes.some(attribute => attribute.name === 'uv'), false);
  for (const value of [2, 3, .125, .375]) assert.ok(floats(first.uniforms).includes(value));
  assert.ok(inspected.length > 0 && inspected.every(value => value === texture));
  assert.equal(first.textures[0].texture.colorSpace, THREE.SRGBColorSpace);
  // Source native shaders use the sampled texture view's color conversion.
  assert.doesNotMatch(d.fragment, /sRGBTransferEOTF/);
  texture.offset.x = .625;
  const second = f.owner.captureDraw(d, {...f.context, renderId: 2});
  assert.equal(f.compile(), d); assert.ok(floats(second.uniforms).includes(.625));
  assert.deepEqual(first.uniforms, saved);
});
test('compatible texture replacement reuses shaders and captures each logical update exactly once', t => {
  const f = rig(t), a = dataTexture(), b = dataTexture(), sampled = N.texture(a), contexts = [];
  b.offset.x = .75; let after = 0;
  const gain = N.uniform(.5).onObjectUpdate(frame => { contexts.push(frame.object); return .5; });
  gain.updateAfterType = 'object'; gain.updateAfter = () => { after++; };
  f.material.colorNode = sampled.mul(gain);
  const d = f.compile(); assert.equal(f.owner.textures(d)[0].texture, a); assert.equal(contexts.length, 0);
  const first = f.owner.captureDraw(d, f.context); f.owner.after(d, f.context);
  sampled.value = b; assert.equal(f.compile(), d);
  const second = f.owner.captureDraw(d, {...f.context, renderId: 2}); f.owner.after(d, {...f.context, renderId: 2});
  assert.ok(first.textures.every(binding => binding.texture === a));
  assert.ok(second.textures.every(binding => binding.texture === b));
  assert.ok(floats(second.uniforms).includes(.75)); assert.equal(contexts.length, 2); assert.equal(after, 2);
  assert.deepEqual(contexts, [f.object, f.object]);
  f.owner.textures(d); assert.equal(contexts.length, 2);
});
test('texture aliases, sampling mode and UV channels invalidate structural programs', t => {
  const f = rig(t), a = dataTexture(), b = dataTexture(), left = N.texture(a), right = N.texture(a);
  f.material.colorNode = left.add(right);
  const initial = f.compile(); assert.equal(f.owner.textures(initial).length, 2);
  right.value = b;
  assert.throws(() => f.owner.captureDraw(initial, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const split = f.compile(); assert.notEqual(split, initial); assert.equal(f.owner.textures(split).length, 4);
  b.minFilter = b.magFilter = THREE.NearestFilter;
  const nearest = f.compile(); assert.notEqual(nearest, split); assert.equal(f.owner.textures(nearest).length, 3);
  f.object.geometry.setAttribute('uv1', f.object.geometry.attributes.uv); b.channel = 1;
  const channel = f.compile(); assert.notEqual(channel, nearest); assert.ok(channel.attributes.some(attribute => attribute.name === 'uv1'));
});
test('render-target texture identity and explicit UVs use the native single-sampled layout', t => {
  const f = rig(t), target = new THREE.RenderTarget(4, 4); target.texture.channel = 3;
  f.material.fragmentNode = N.texture(target.texture, N.uv().yx).level(0);
  const d = f.compile(), captured = f.owner.captureDraw(d, f.context);
  assert.ok(captured.textures.every(binding => binding.texture === target.texture));
  assert.equal(d.attributes.some(attribute => attribute.name === 'uv3'), false);
  assert.match(d.fragment, /textureSampleLevel/); assert.doesNotMatch(d.fragment, /flipY/);
  for (const group of d.groups) for (const entry of group.entries) if (entry.texture)
    assert.deepEqual(entry.texture, {sampleType: 'float', viewDimension: '2d', multisampled: false});
});
test('callback texture edits cannot silently retain an incompatible compiled layout', t => {
  const f = rig(t), texture = dataTexture(), sampled = N.texture(texture); let updates = 0;
  f.object.geometry.setAttribute('uv1', f.object.geometry.attributes.uv);
  const gain = N.uniform(.5).onObjectUpdate(() => { updates++; texture.channel = 1; return .5; });
  f.material.colorNode = sampled.mul(gain); const d = f.compile();
  assert.throws(() => f.owner.captureDraw(d, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, 1);
  const next = f.compile(); assert.notEqual(next, d); f.owner.captureDraw(next, f.context); assert.equal(updates, 2);
});
test('unsupported texture families, actual borrowed layouts and device binding limits refuse explicitly', t => {
  const f = rig(t);
  const values = [new THREE.CubeTexture(), new THREE.Data3DTexture(new Uint8Array(4), 1, 1, 1),
    new THREE.DataArrayTexture(new Uint8Array(4), 1, 1, 1), new THREE.DepthTexture(1, 1),
    new THREE.StorageTexture(1, 1), dataTexture({type: THREE.FloatType})];
  const integral = dataTexture(); integral.type = THREE.UnsignedIntType; integral.format = THREE.RGBAIntegerFormat; values.push(integral);
  const video = dataTexture(); video.isVideoTexture = true; values.push(video);
  for (const texture of values) { f.material.colorNode = N.texture(texture); assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'}); }
  const a = dataTexture(); f.material.colorNode = N.texture(a);
  f.renderer._getSampledTextureInputs = () => ({layoutOf: () => ({texture: {sampleType: 'unfilterable-float', viewDimension: '2d', multisampled: false}})});
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  delete f.renderer._getSampledTextureInputs;
  f.renderer._device = {limits: {maxSampledTexturesPerShaderStage: 1}};
  f.material.colorNode = N.texture(a).add(N.texture(dataTexture()));
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
});
test('native storage preserves source stage aliases and forces read-only graphics bindings', t => {
  const f = rig(t), attribute = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(attribute, 'vec3', 4);
  f.material.positionNode = N.positionLocal.add(storage.element(N.vertexIndex));
  f.material.colorNode = storage.element(0);
  const d = f.compile(), bindings = f.owner.storage(d), captured = f.owner.captureDraw(d, f.context);
  assert.equal(storage.access, 'readWrite', 'graphics admission does not mutate the shared compute node');
  assert.match(d.vertex, /var<storage,\s*read>/); assert.match(d.fragment, /var<storage,\s*read>/);
  assert.doesNotMatch(d.vertex + d.fragment, /var<storage,\s*read_write>/);
  assert.equal(bindings.length, 2, 'the source emits separate vertex and fragment binding slots');
  assert.equal(new Set(bindings.map(binding => `${binding.group}/${binding.binding}`)).size, 2);
  assert.ok(bindings.every(binding => binding.attribute === attribute));
  assert.deepEqual(captured.storage, bindings); assert.equal(captured.textures.length, 0);
  assert.ok(Object.isFrozen(bindings) && bindings.every(Object.isFrozen));
  const layouts = bindings.map(({group, binding}) => d.groups[group].entries[binding]);
  assert.deepEqual(new Set(layouts.map(layout => layout.visibility)), new Set([1, 2]));
  for (const layout of layouts) {
    assert.equal(layout.buffer.type, 'read-only-storage'); assert.equal(layout.buffer.minBindingSize, 64);
    assert.notEqual(layout.buffer.hasDynamicOffset, true);
  }
  const old = bindings;
  storage.value = new THREE.StorageBufferAttribute(4, 4);
  assert.equal(f.compile(), d); assert.equal(f.owner.storage(d)[0].attribute, storage.value);
  assert.equal(old[0].attribute, attribute, 'captured attribute identities do not follow a mutable node');
});
test('native storage admits source f32, i32 and u32 vectors with explicit vec3 padding', t => {
  const f = rig(t);
  for (const [ArrayType, prefix, scalar] of [[Float32Array, '', 'float'], [Int32Array, 'i', 'int'], [Uint32Array, 'u', 'uint']]) {
    for (const width of [1, 2, 3, 4]) {
      const stride = width === 3 ? 4 : width, attribute = new THREE.StorageBufferAttribute(2, stride, ArrayType);
      attribute.array.fill(ArrayType === Int32Array ? -19 : 29);
      const storage = N.storage(attribute, width === 1 ? scalar : `${prefix}vec${width}`, 2);
      f.material.fragmentNode = N.vec4(storage.element(0));
      const d = f.compile(), captured = f.owner.captureDraw(d, f.context).storage;
      assert.equal(captured.length, 1); assert.equal(captured[0].attribute, attribute);
      assert.equal(captured[0].attribute.array.constructor, ArrayType);
      const binding = d.groups[captured[0].group].entries[captured[0].binding];
      assert.equal(binding.buffer.minBindingSize, 2 * stride * 4);
      assert.match(d.fragment, new RegExp(`(?:array<\\s*|vec${width}<\\s*)${ArrayType === Float32Array ? 'f32' : ArrayType === Int32Array ? 'i32' : 'u32'}`));
    }
  }
  const packed = new THREE.StorageBufferAttribute(2, 3);
  f.material.fragmentNode = N.vec4(N.storage(packed, 'vec3', 2).element(0), 1);
  const packedProgram = f.compile();
  assert.equal(f.owner.storage(packedProgram)[0].attribute, packed);
  assert.equal(packed.itemSize, 3, 'graphics does not normalize or upload the compute attribute');
});
test('storage alias, count and layout changes invalidate programs before captured use', t => {
  const f = rig(t), a = new THREE.StorageBufferAttribute(4, 4), b = new THREE.StorageBufferAttribute(4, 4);
  const left = N.storage(a, 'vec4'), right = N.storage(a, 'vec4');
  f.material.fragmentNode = left.element(0).add(right.element(1));
  const aliased = f.compile(); assert.equal(f.owner.storage(aliased).length, 1);
  right.value = b;
  assert.throws(() => f.owner.captureDraw(aliased, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const split = f.compile(); assert.notEqual(split, aliased); assert.equal(f.owner.storage(split).length, 2);
  right.bufferCount = 2;
  const bounded = f.compile(); assert.notEqual(bounded, split);
  const bigger = new THREE.StorageBufferAttribute(8, 4); right.value = bigger;
  assert.throws(() => f.owner.captureDraw(bounded, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.notEqual(f.compile(), bounded);
  right.value = new THREE.StorageBufferAttribute(8, 4, Uint32Array);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_STORAGE'});
});
test('storage changed by an object update is captured only when its compiled shape still matches', t => {
  const f = rig(t), initial = new THREE.StorageBufferAttribute(4, 4), next = new THREE.StorageBufferAttribute(4, 4),
    storage = N.storage(initial, 'vec4', 4); let replacement = next, updates = 0;
  const gain = N.uniform(1).onObjectUpdate(() => { updates++; storage.value = replacement; return 1; });
  f.material.fragmentNode = storage.element(0).mul(gain);
  const d = f.compile(); assert.equal(updates, 0);
  assert.equal(f.owner.captureDraw(d, f.context).storage[0].attribute, next); assert.equal(updates, 1);
  replacement = new THREE.StorageBufferAttribute(8, 4);
  assert.throws(() => f.owner.captureDraw(d, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, 2);
});
test('compatible vector and struct replacements between updates retain the program and each captured identity', t => {
  for (const type of ['vec4', N.struct({color: 'vec4'}, 'ReplacementRecord')]) {
    const f = rig(t), a = new THREE.StorageBufferAttribute(4, 4), b = new THREE.StorageBufferAttribute(4, 4),
      after = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(a, type, 4);
    let updates = 0;
    const gain = N.uniform(1).onObjectUpdate(() => { storage.value = ++updates === 1 ? a : b; return 1; });
    gain.updateAfterType = 'object'; gain.updateAfter = () => { storage.value = after; };
    f.material.fragmentNode = (type === 'vec4' ? storage.element(0) : storage.element(0).get('color')).mul(gain);
    const d = f.compile(), first = f.owner.captureDraw(d, f.context);
    f.owner.after(d, f.context); assert.equal(storage.value, after);
    assert.equal(f.compile(), d, 'after-update replacement preserves the source binding layout');
    const second = f.owner.captureDraw(d, {...f.context, renderId: 2});
    assert.equal(updates, 2); assert.equal(first.storage[0].attribute, a); assert.equal(second.storage[0].attribute, b);
  }
});
test('fixed storage bindings stay independent when replacements change their aliases with unsized nodes', t => {
  const f = rig(t), a = new THREE.StorageBufferAttribute(4, 4), b = new THREE.StorageBufferAttribute(4, 4);
  const left = N.storage(a, 'vec4', 4), right = N.storage(a, 'vec4', 4), unsized = N.storage(a, 'vec4');
  f.material.fragmentNode = left.element(0).add(right.element(1)).add(unsized.element(2));
  const d = f.compile(); assert.equal(f.owner.storage(d).length, 3);
  left.value = b;
  assert.equal(f.compile(), d); assert.deepEqual(f.owner.storage(d).map(binding => binding.attribute), [b, a, a]);
  right.value = unsized.value = b;
  assert.equal(f.compile(), d); assert.ok(f.owner.storage(d).every(binding => binding.attribute === b));
});
test('storage aliases created inside Fn invalidate on splitting and merging', t => {
  const f = rig(t), a = new THREE.StorageBufferAttribute(4, 4), b = new THREE.StorageBufferAttribute(4, 4), nodes = {};
  f.material.fragmentNode = N.Fn(() => {
    nodes.left ??= N.storage(a, 'vec4'); nodes.right ??= N.storage(a, 'vec4');
    return nodes.left.element(0).add(nodes.right.element(1));
  })();
  const aliased = f.compile(); assert.equal(f.owner.storage(aliased).length, 1);
  nodes.right.value = b;
  assert.throws(() => f.owner.captureDraw(aliased, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const split = f.compile(); assert.equal(f.owner.storage(split).length, 2);
  assert.deepEqual(new Set(f.owner.captureDraw(split, f.context).storage.map(value => value.attribute)), new Set([a, b]));
  nodes.right.value = a;
  assert.throws(() => f.owner.captureDraw(split, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const merged = f.compile(); assert.equal(f.owner.storage(merged).length, 1);
  nodes.left.value = nodes.right.value = b;
  assert.equal(f.compile(), merged); assert.equal(f.owner.captureDraw(merged, f.context).storage[0].attribute, b);
});
test('graphics storage assignments fail before a descriptor is published, including Fn and component targets', t => {
  const f = rig(t), storage = N.storage(new THREE.StorageBufferAttribute(4, 4), 'vec4', 4);
  for (const [target, value] of [[storage.element(0), N.vec4(1)], [storage.element(0).xy, N.vec2(1)],
    [storage.element(0).element(1), N.float(1)], [storage.element(0).context({}), N.vec4(1)]]) {
    f.material.fragmentNode = N.Fn(() => { target.assign(value); return N.vec4(1); })();
    assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_STORAGE'});
    assert.equal(f.owner.diagnostics.programs, 0);
  }
  f.material.fragmentNode = N.Fn(() => {
    const local = storage.element(0).toVar(); local.x.assign(.5); return local;
  })();
  assert.equal(f.owner.storage(f.compile())[0].attribute, storage.value, 'writes to a local copy remain valid');
});
for (const Material of [THREE.MeshBasicNodeMaterial, THREE.MeshStandardNodeMaterial]) {
  test(`${Material.name} storage instancing retains source matrix, normal, color and authored instance-index reads`, t => {
    const f = instanceRig(t, Material, 3), extra = new THREE.StorageBufferAttribute(4, 4);
    f.material.colorNode = f.tint.mul(N.storage(extra, 'vec4', 4).element(N.instanceIndex).xyz);
    const d = f.compile(), draw = f.owner.captureDraw(d, f.context);
    assert.equal(d.instanceCapacity, 3);
    assert.match(d.vertex, /@builtin\( instance_index \)/);
    assert.match(d.vertex, /array< mat4x4<f32> >/);
    assert.match(d.vertex, /normalLocal = normalize\(.*transpose\( tsl_inverse_mat3/);
    assert.match(d.vertex, /varyings\.vInstanceColor = .*\[ instanceIndex \]/);
    assert.match(d.fragment, /vInstanceColor/); assert.match(d.fragment, /var<storage, read>/);
    assert.ok(d.vertex.indexOf('positionLocal = ( NodeBuffer_') < d.vertex.indexOf('positionLocal = ( positionLocal +'),
      'the source applies instance transforms before authored positionNode');
    assert.ok(d.attributes.every(attribute => f.object.geometry.attributes[attribute.name]), 'instance data adds no vertex streams');
    for (const [attribute, bytes] of [[f.object.instanceMatrix, 256], [f.object.instanceColor, 48], [extra, 64]]) {
      const binding = draw.storage.find(value => value.attribute === attribute); assert.ok(binding);
      assert.equal(d.groups[binding.group].entries[binding.binding].buffer.minBindingSize, bytes);
    }
    assert.equal(f.object.instanceColor.itemSize, 3, 'graphics does not normalize or upload source storage');
  });
}
test('instance graph keys retain data/count updates but rebuild captured matrix/color identities and capacities', t => {
  const f = instanceRig(t, THREE.MeshBasicNodeMaterial, 3), d = f.compile();
  f.object.instanceMatrix.array[12] = .25; f.object.instanceMatrix.needsUpdate = true;
  f.object.instanceColor.array = new Float32Array(12); f.object.instanceColor.itemSize = 4;
  f.object.instanceColor.needsUpdate = true;
  for (const count of [3, 1, 0, 99, -2]) {
    f.object.count = count; assert.equal(f.compile(), d, 'only post-update scene capture bounds draw counts');
  }
  f.object.instanceMatrix = new THREE.StorageInstancedBufferAttribute(4, 16);
  assert.throws(() => f.owner.captureDraw(d, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const matrix = f.compile(); assert.notEqual(matrix, d); assert.equal(matrix.instanceCapacity, 3);
  f.object.instanceColor = new THREE.StorageInstancedBufferAttribute(2, 4);
  assert.throws(() => f.owner.captureDraw(matrix, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const color = f.compile(); assert.notEqual(color, matrix); assert.equal(color.instanceCapacity, 2);
  f.object.instanceColor = null;
  const plain = f.compile(); assert.equal(plain.instanceCapacity, 4); assert.doesNotMatch(plain.fragment, /vInstanceColor/);
  assert.equal(f.owner.captureDraw(plain, f.context).storage.length, 1);
  f.object.instanceMatrix.array = new Float32Array(32); f.object.instanceMatrix.count = 2;
  const smaller = f.compile(); assert.notEqual(smaller, plain); assert.equal(smaller.instanceCapacity, 2);
  assert.equal(rig(t).compile().instanceCapacity, undefined);
});
test('instance replacements during source updates fail before stale generated bindings can be captured', t => {
  const f = instanceRig(t); let updates = 0, replace = false;
  f.material.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(() => {
    updates++;
    if (replace) f.object.instanceMatrix = new THREE.StorageInstancedBufferAttribute(4, 16);
  });
  const d = f.compile(); f.object.instanceColor = new THREE.StorageInstancedBufferAttribute(3, 4);
  assert.throws(() => f.owner.captureDraw(d, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, 0);
  const changed = f.compile(); replace = true;
  assert.throws(() => f.owner.captureDraw(changed, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, 1);
  replace = false;
  const current = f.compile(); assert.ok(f.owner.captureDraw(current, {...f.context, renderId: 2})); assert.equal(updates, 2);
  f.object.morphTexture = new THREE.DataTexture();
  assert.throws(() => f.owner.captureDraw(current, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_OBJECT'});
  assert.equal(updates, 2);
});
test('instance admission refuses CPU streams, malformed storage and deformation before node updates', t => {
  for (const [change, code] of [
    [object => { object.instanceMatrix = new THREE.InstancedBufferAttribute(new Float32Array(64), 16); }, 'OBJECT'],
    [object => { object.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(9), 3); }, 'OBJECT'],
    [object => { object.instanceMatrix = new THREE.StorageInstancedBufferAttribute(4, 4); }, 'STORAGE'],
    [object => { object.instanceMatrix = new THREE.StorageInstancedBufferAttribute(4, 16, Uint32Array); }, 'STORAGE'],
    [object => { object.instanceColor = new THREE.StorageInstancedBufferAttribute(3, 2); }, 'STORAGE'],
    [object => { object.instanceMatrix = new THREE.StorageInstancedBufferAttribute(0, 16); }, 'STORAGE'],
    [object => { object.morphTexture = new THREE.DataTexture(); }, 'OBJECT'],
    [object => { object.isSkinnedMesh = true; }, 'OBJECT'],
    [object => { object.geometry = new THREE.InstancedBufferGeometry(); }, 'OBJECT'],
  ]) {
    const f = instanceRig(t); let updates = 0;
    f.material.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(() => { updates++; }); change(f.object);
    assert.throws(f.compile, {code: `THREE_NODE_PROGRAM_${code}`}); assert.equal(updates, 0);
  }
});
test('nested captures and thrown callbacks restore the exact outer NodeFrame context', t => {
  const f = rig(t), innerMaterial = new THREE.MeshBasicNodeMaterial(), innerObject = new THREE.Mesh(f.object.geometry, innerMaterial);
  innerMaterial.colorNode = N.vec3(.2); const inner = f.owner.compile(innerMaterial, innerObject, f.context);
  const initial = Object.fromEntries(['renderer', 'scene', 'camera', 'object', 'material', 'compute', 'frameId', 'renderId', 'time', 'deltaTime'].map(k => [k, f.frame[k]]));
  const failure = Error('callback failure'); let nested = false;
  f.material.colorNode = N.uniform(0).onObjectUpdate(current => {
    f.owner.capture(inner, {...f.context, object: innerObject, material: innerMaterial, renderId: 2});
    assert.equal(current.object, f.object); assert.equal(current.material, f.material); assert.equal(current.renderId, 1);
    nested = true; throw failure;
  });
  assert.throws(() => f.owner.capture(f.compile(), f.context), e => e === failure); assert.equal(nested, true);
  assert.deepEqual(Object.fromEntries(Object.keys(initial).map(k => [k, f.frame[k]])), initial);
});
for (const [name, change, code] of [
  ['depth texture graph', f => { f.material.colorNode = N.texture(new THREE.DepthTexture(1, 1)); }, 'TEXTURE'],
  ['atomic storage graph', f => { f.material.colorNode = N.storage(new THREE.StorageBufferAttribute(4, 1, Uint32Array), 'uint', 4).toAtomic().element(0).toFloat(); }, 'STORAGE'],
  ['managed uniform array graph', f => { f.material.colorNode = N.uniformArray([new THREE.Vector4(1, 0, 0, 1)]).element(0).xyz; }, 'STORAGE'],
  ['buffer attribute graph', f => { f.material.colorNode = N.bufferAttribute(new THREE.Float32BufferAttribute(new Float32Array(12), 3)); }, 'STORAGE'],
  ['material environment texture', f => { f.material.envMap = new THREE.Texture(); }, 'TEXTURE'],
  ['MRT graph', f => { f.material.fragmentNode = N.mrt({output: N.vec4(1)}); }, 'MRT'],
  ['lighting slot', f => { f.material.lightsNode = N.vec3(1); }, 'GRAPH'],
  ['transparent double side', f => { f.material.side = THREE.DoubleSide; f.material.transparent = true; }, 'RASTER'],
  ['custom renderer context', f => { f.renderer.contextNode = N.context({getOutput: output => output}); }, 'GRAPH'],
  ['spoofed instancing', f => { f.object.isInstancedMesh = true; }, 'OBJECT'],
  ['fog', f => { f.scene.fog = new THREE.Fog(0, 1, 10); }, 'EFFECT'],
]) test(`${name} is explicitly refused`, t => {
  const f = rig(t); change(f); assert.throws(f.compile, {code: `THREE_NODE_PROGRAM_${code}`});
});
test('native storage reads retain source binding order, read-only stage access and live attribute identity', t => {
  const f = rig(t), attribute = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(attribute, 'vec4', 4);
  f.material.positionNode = storage.element(N.vertexIndex).xyz;
  f.material.colorNode = storage.element(0).xyz.mul(f.tint);
  const d = f.compile(), bindings = d.groups.flatMap(group => group.entries.filter(entry => entry.buffer?.type === 'read-only-storage')
    .map(entry => ({group: group.index, ...entry})));
  assert.ok(bindings.length > 0);
  for (const binding of bindings) {
    assert.equal(binding.buffer.hasDynamicOffset, false); assert.equal(binding.buffer.minBindingSize, 64);
    assert.ok((binding.visibility & 3) !== 0); assert.equal(binding.visibility & 4, 0);
  }
  assert.match(d.vertex, /var<storage, read>/); assert.match(d.fragment, /var<storage, read>/);
  assert.doesNotMatch(d.vertex + d.fragment, /var<storage, read_write>/);
  const captured = f.owner.captureDraw(d, f.context);
  assert.equal(captured.storage.length, bindings.length);
  for (const binding of captured.storage) {
    assert.equal(binding.attribute, attribute);
    assert.ok(bindings.some(entry => entry.group === binding.group && entry.binding === binding.binding));
  }
  attribute.array.fill(.25); attribute.needsUpdate = true;
  assert.equal(f.compile(), d, 'upload versions do not recompile a shader');
  storage.value = new THREE.StorageBufferAttribute(4, 4);
  assert.equal(f.compile(), d, 'source NodeStorageBuffer.attribute follows node.value without changing its binding layout');
  const replaced = f.owner.captureDraw(d, {...f.context, renderId: 2});
  assert.ok(replaced.storage.every(binding => binding.attribute === storage.value));
  assert.ok(captured.storage.every(binding => binding.attribute === attribute), 'earlier draw metadata keeps its captured identity');
});
test('native storage supports typed scalar/vector inputs and vec3 padded residency', t => {
  for (const [type, components, array] of [['float', 1, Float32Array], ['ivec2', 2, Int32Array], ['uvec4', 4, Uint32Array], ['vec3', 3, Float32Array]]) {
    const f = rig(t), attribute = new THREE.StorageBufferAttribute(4, components, array);
    f.material.colorNode = N.vec3(N.storage(attribute, type, 4).element(0));
    const d = f.compile(), entry = d.groups.flatMap(group => group.entries).find(value => value.buffer?.type === 'read-only-storage');
    assert.equal(entry.buffer.minBindingSize, 4 * (components === 3 ? 4 : components) * 4);
    if (components === 3) {
      attribute.array = new Float32Array(16); attribute.itemSize = 4;
      assert.equal(f.compile(), d, 'compute normalization keeps the shader shape');
    }
  }
});
test('native storage rejects writes, unsupported shapes and stage limits before node updates', t => {
  const f = rig(t), attribute = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(attribute, 'vec4', 4);
  f.material.colorNode = N.Fn(() => { storage.element(0).assign(N.vec4(1)); return N.vec3(1); })();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_STORAGE'});
  f.material.colorNode = storage.element(0).xyz;
  attribute.normalized = true; assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_STORAGE'}); attribute.normalized = false;
  attribute.onUploadCallback = () => {}; assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_HOOK'});
  attribute.onUploadCallback = THREE.BufferAttribute.prototype.onUploadCallback;
  f.renderer._device = {limits: {maxStorageBufferBindingSize: 32}};
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
  f.renderer._device.limits = {maxStorageBuffersPerShaderStage: 1};
  f.material.colorNode = storage.element(0).xyz.add(N.storage(new THREE.StorageBufferAttribute(4, 4), 'vec4', 4).element(0).xyz);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
});
test('native H8 particle structs preserve padded source layout and shared vertex/fragment storage', t => {
  const f = rig(t), schema = N.struct({position: 'vec3', velocity: 'vec3', C: 'mat3'}, 'FluidParticle');
  const array = new Float32Array(5 * 20), particles = N.instancedArray(array, schema);
  const particle = particles.element(N.instanceIndex);
  f.material.positionNode = particle.get('C').mul(N.positionLocal).add(particle.get('position'));
  f.material.colorNode = particle.get('velocity'); f.object.count = 5;
  const d = f.compile(), draw = f.owner.captureDraw(d, f.context);
  assert.equal(schema.getLength(), 20); assert.equal(particles.value.itemSize, 20);
  assert.equal(particles.value.array, array, 'graphics retains the padded source bytes');
  for (const shader of [d.vertex, d.fragment]) {
    assert.match(shader, /struct FluidParticle/);
    assert.match(shader, /position : vec3<f32>/); assert.match(shader, /velocity : vec3<f32>/);
    assert.match(shader, /C : mat3x3<f32>/); assert.match(shader, /var<storage, read>/);
    assert.doesNotMatch(shader, /var<storage, read_write>/);
  }
  for (const entry of d.groups.flatMap(group => group.entries).filter(entry => entry.buffer?.type === 'read-only-storage'))
    assert.equal(entry.buffer.minBindingSize, 5 * 80);
  assert.ok(draw.storage.length > 0);
  assert.ok(draw.storage.every(binding => binding.attribute === particles.value));
  array.fill(.25); particles.value.needsUpdate = true;
  assert.equal(f.compile(), d, 'host upload versions do not change the struct shader');
});
test('native storage structs retain mixed scalar/vector and matrix fields as 32-bit backing bytes', t => {
  for (const Type of [Float32Array, Int32Array, Uint32Array]) {
    const f = rig(t), schema = N.struct({weight: 'float', code: 'uint', bias: 'ivec2', direction: 'vec3',
      index: 'int', basis: 'mat2', transform: 'mat4'}, 'MixedParticle');
    assert.equal(schema.getLength(), 28);
    const array = new Type(3 * 28), source = N.storage(new THREE.StorageBufferAttribute(array, 28), schema);
    const p = source.element(0);
    f.material.positionNode = p.get('transform').mul(N.vec4(p.get('direction'), 1)).xyz
      .add(N.vec3(p.get('basis').mul(N.vec2(1)), 0));
    f.material.colorNode = N.vec3(p.get('weight'), p.get('code').toFloat(), p.get('index').toFloat())
      .add(N.vec3(p.get('bias'), 0));
    const d = f.compile(), draw = f.owner.captureDraw(d, f.context);
    assert.equal(draw.storage[0].attribute.array, array);
    assert.match(d.fragment, /code : u32/); assert.match(d.fragment, /bias : vec2<i32>/);
    assert.match(d.vertex, /basis : mat2x2<f32>/); assert.match(d.vertex, /transform : mat4x4<f32>/);
    for (const entry of d.groups.flatMap(group => group.entries).filter(entry => entry.buffer?.type === 'read-only-storage'))
      assert.equal(entry.buffer.minBindingSize, 3 * 28 * 4);
  }
});
test('same-size struct schema changes invalidate descriptors before source update callbacks', t => {
  const f = rig(t), schema = N.struct({position: 'vec3', velocity: 'vec3', C: 'mat3'}, 'MutableParticle');
  const particles = N.instancedArray(4, schema); let updates = 0;
  const value = N.uniform(1).onRenderUpdate(() => { updates++; return 1; });
  f.material.positionNode = particles.element(0).get('position');
  f.material.colorNode = particles.element(0).get('velocity').mul(value);
  let d = f.compile();
  for (const mutate of [
    () => { schema.membersLayout[1].type = 'ivec3'; },
    () => { schema.membersLayout[2].name = 'deformation'; },
    () => { [schema.membersLayout[0], schema.membersLayout[1]] = [schema.membersLayout[1], schema.membersLayout[0]]; },
    () => { schema.name = 'RenamedParticle'; },
  ]) {
    mutate(); assert.equal(schema.getLength(), 20);
    assert.throws(() => f.owner.captureDraw(d, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    assert.equal(updates, 0);
    const next = f.compile(); assert.notEqual(next, d); d = next;
  }
  assert.match(d.fragment, /velocity : vec3<i32>/); assert.match(d.vertex, /struct RenamedParticle/);
  schema.membersLayout[0].atomic = true;
  assert.throws(() => f.owner.captureDraw(d, f.context), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(updates, 0);
});
test('structured storage rejects unsupported layouts, short strides and atomic members before updates', t => {
  const cases = [
    [{position: {type: 'int', atomic: true}}, 1, Float32Array],
    [{position: 'bool'}, 1, Float32Array],
    [{position: 'bvec4'}, 4, Float32Array],
    [{position: 'NestedParticle'}, 4, Float32Array],
    [{position: 'float', velocity: 'int', id: 'uint'}, 3, Float32Array],
    [{position: 'vec3', velocity: 'vec3', C: 'mat3'}, 19, Float32Array],
    [{position: 'vec4'}, 4, Uint16Array],
  ];
  for (const [members, stride, Type] of cases) {
    const f = rig(t), source = N.storage(new THREE.StorageBufferAttribute(4, stride, Type), N.struct(members));
    let updates = 0;
    f.material.positionNode = N.vec3(source.element(0).get('position'));
    f.material.colorNode = N.uniform(new THREE.Color()).onRenderUpdate(() => { updates++; });
    assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_STORAGE'});
    assert.equal(updates, 0);
  }
  const f = rig(t), source = N.instancedArray(4, N.struct({position: 'vec3', velocity: 'vec3', C: 'mat3'}));
  f.material.positionNode = source.element(0).get('position');
  f.renderer._device = {limits: {maxStorageBufferBindingSize: 319}};
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'});
});
test('struct member, matrix-column and whole-element writes are rejected while local copies remain writable', t => {
  for (const write of [
    p => p.element(0).get('position').assign(N.vec3(1)),
    p => p.element(0).get('C').element(0).x.assign(1),
    p => p.element(0).assign(p.element(1)),
  ]) {
    const f = rig(t), particles = N.instancedArray(4, N.struct({position: 'vec3', velocity: 'vec3', C: 'mat3'}));
    f.material.positionNode = N.Fn(() => { write(particles); return N.positionLocal; })();
    assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_STORAGE'});
  }
  const f = rig(t), particles = N.instancedArray(4, N.struct({position: 'vec3', velocity: 'vec3', C: 'mat3'}));
  f.material.positionNode = N.Fn(() => {
    const local = particles.element(0).toVar('localParticle');
    local.get('position').addAssign(N.vec3(.125)); return local.get('position');
  })();
  const d = f.compile(); assert.match(d.vertex, /localParticle\.position =/);
  assert.match(d.vertex, /var<storage, read>/); assert.equal(f.owner.captureDraw(d, f.context).storage.length, 1);
});
test('source missing attributes retain the builder constant fallback without a fake vertex stream', t => {
  const f = rig(t); f.material.positionNode = N.attribute('absent', 'vec3');
  const d = f.compile(); assert.equal(d.attributes.some(a => a.name === 'absent'), false);
  assert.match(d.vertex, /positionLocal = vec3<f32>\( 0\.0, 0\.0, 0\.0 \)/);
});
test('program budget, material disposal and owner identity are enforced', t => {
  const f = rig(t, {maxPrograms: 1}), d = f.compile(), other = new THREE.Mesh(f.object.geometry, f.material);
  assert.throws(() => f.owner.compile(f.material, other, f.context), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  f.material.dispose(); assert.equal(f.owner.diagnostics.programs, 0);
  assert.ok(f.owner.compile(f.material, other, f.context));
  const foreign = rig(t); assert.throws(() => foreign.owner.capture(d, f.context), {code: 'THREE_NODE_PROGRAM_SOURCE'});
  f.owner.dispose(); assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_DISPOSED'});
});


for (const Material of [THREE.MeshBasicMaterial, THREE.MeshToonMaterial, THREE.MeshLambertMaterial,
  THREE.MeshPhongMaterial, THREE.MeshStandardMaterial, THREE.MeshPhysicalMaterial]) {
  test(`source material ${Material.name} retains authored geometry and the pinned surface graph`, t => {
    const f = pbrRig(t, Material), material = f.material;
    delete material.colorNode;
    material.positionNode = N.positionLocal.add(N.vec3(f.shift, 0, 0));
    material.map = dataTexture(); material.map.colorSpace = THREE.SRGBColorSpace;
    assert.equal(hasThreeNodeProgram(material), true);
    assert.equal(hasThreeNodeProgram(new Material()), false, 'ordinary built-ins retain the fixed route');
    const d = f.compile(), draw = f.owner.captureDraw(d, f.context);
    assert.match(d.vertex, /object\.testShift/);
    assert.match(d.fragment, /textureSample/);
    assert.equal(draw.textures.find(binding => binding.texture === material.map).texture.colorSpace, THREE.SRGBColorSpace);
    assert.ok(draw.textures.some(binding => binding.texture === material.map));
    assert.ok(floats(draw.uniforms).includes(.125));
    if (material.isMeshStandardMaterial) assert.match(d.fragment, /fn D_GGX/);
    assert.equal(f.compile(), d);
    material.dispose(); assert.equal(f.owner.diagnostics.programs, 0, 'the source material owns cache lifetime');
  });
}
test('source material H7 Physical maps coexist with compute-authored instanced geometry', t => {
  const f = pbrRig(t, THREE.MeshPhysicalMaterial), material = f.material, vertexCount = f.object.geometry.attributes.position.count;
  delete material.colorNode;
  material.side = THREE.DoubleSide; material.ior = 1.4500000476837158; material.metalness = .5;
  material.map = dataTexture(); material.map.colorSpace = THREE.SRGBColorSpace;
  material.normalMap = dataTexture(); material.specularColorMap = dataTexture();
  material.specularColorMap.colorSpace = THREE.SRGBColorSpace;
  material.metalnessMap = material.roughnessMap = dataTexture();
  const vertices = new THREE.StorageBufferAttribute(new Float32Array(2 * vertexCount * 2 * 4), 4);
  const storage = N.storage(vertices, 'vec4', vertices.count), index = N.instanceIndex.mul(N.uint(vertexCount)).add(N.vertexIndex).mul(N.uint(2));
  material.positionNode = storage.element(index).xyz;
  material.normalNode = N.transformNormalToView(storage.element(index.add(N.uint(1))).xyz).toVarying();
  f.object.count = 2;
  const d = f.compile(), draw = f.owner.captureDraw(d, f.context);
  assert.match(d.vertex, /instance_index/); assert.match(d.vertex, /vertex_index/);
  assert.match(d.vertex, /var<storage, read>/); assert.match(d.fragment, /fn D_GGX/);
  assert.ok(draw.storage.some(binding => binding.attribute === vertices));
  const sampled = draw.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
  assert.equal(sampled.length, 4, 'map, specular color, shared metallic/roughness map and source DFG');
  for (const texture of [material.map, material.specularColorMap, material.metalnessMap]) assert.ok(sampled.includes(texture));
  assert.equal(sampled.includes(material.normalMap), false, 'authored H7 normalNode overrides the normal map');
  assert.ok(sampled.some(texture => texture.name === 'DFG_LUT'));
  assert.ok(floats(draw.uniforms).includes(Math.fround(material.ior)));
});
test('source material preparation preserves builder and NodeFrame source identities', t => {
  const f = pbrRig(t, THREE.MeshStandardMaterial), material = f.material, seen = [];
  delete material.colorNode;
  material.map = dataTexture(); material.color.setRGB(.125, .375, .625);
  const offset = N.uniform(0).onObjectUpdate(frame => {
    seen.push([frame.material, frame.object, frame.scene, frame.camera]); return frame.material.roughness;
  });
  material.positionNode = N.Fn(builder => {
    seen.push([builder.material, builder.object, builder.scene, builder.camera]);
    assert.equal(builder.context.material, material);
    return N.positionLocal.add(N.vec3(offset, 0, 0));
  })();
  const d = f.compile(), first = f.owner.captureDraw(d, f.context);
  assert.ok(seen.length >= 2);
  for (const context of seen) assert.deepEqual(context, [material, f.object, f.scene, f.camera]);
  for (const value of [.125, .375, .625]) assert.ok(floats(first.uniforms).includes(value));
  material.color.setRGB(.75, .5, .25); material.roughness = .875;
  assert.equal(f.compile(), d);
  const second = f.owner.captureDraw(d, {...f.context, renderId: 2});
  for (const value of [.75, .5, .25, .875]) assert.ok(floats(second.uniforms).includes(value));
  assert.notDeepEqual(second.uniforms, first.uniforms);
});
test('source material graph edits, removed authored slots and Physical feature switches rebuild correctly', t => {
  const f = pbrRig(t, THREE.MeshPhysicalMaterial), material = f.material;
  delete material.colorNode;
  material.positionNode = N.positionLocal.add(N.vec3(f.shift, 0, 0));
  const before = f.compile(); assert.match(before.vertex, /testShift/);
  material.clearcoat = .5;
  const enabled = f.compile(); assert.notEqual(enabled, before); assert.match(enabled.fragment, /clearcoatSpecularDirect/);
  material.clearcoat = .75; assert.equal(f.compile(), enabled);
  material.positionNode = N.positionLocal;
  const changed = f.compile(); assert.notEqual(changed, enabled); assert.doesNotMatch(changed.vertex, /testShift/);
  delete material.positionNode;
  const removed = f.compile(); assert.notEqual(removed, changed); assert.equal(hasThreeNodeProgram(material), false);
  material.clearcoat = 0;
  const disabled = f.compile(); assert.notEqual(disabled, removed); assert.doesNotMatch(disabled.fragment, /clearcoatSpecularDirect/);
  assert.equal(f.owner.diagnostics.programs, 1);
});
test('source material compatible map replacements retain programs and immutable draw captures', t => {
  const f = pbrRig(t, THREE.MeshStandardMaterial), material = f.material;
  delete material.colorNode; material.positionNode = N.positionLocal;
  material.map = dataTexture(); material.map.offset.set(.125, .25);
  const original = material.map, d = f.compile(), first = f.owner.captureDraw(d, f.context), saved = first.uniforms.map(bytes => bytes.slice());
  const replacement = dataTexture(); replacement.offset.set(.375, .625); material.map = replacement;
  assert.equal(f.compile(), d);
  const second = f.owner.captureDraw(d, {...f.context, renderId: 2});
  assert.ok(first.textures.some(binding => binding.texture === original));
  assert.ok(second.textures.some(binding => binding.texture === replacement));
  assert.equal(second.textures.some(binding => binding.texture === original), false);
  assert.deepEqual(first.uniforms, saved); assert.notDeepEqual(second.uniforms, first.uniforms);
  assert.ok(floats(second.uniforms).includes(.375)); assert.ok(floats(second.uniforms).includes(.625));
});
test('source material map topology, UV channels and normal mode invalidate the graph before callbacks', t => {
  const f = pbrRig(t, THREE.MeshStandardMaterial), material = f.material;
  delete material.colorNode; let calls = 0;
  material.positionNode = N.positionLocal.add(N.vec3(N.uniform(0).onObjectUpdate(() => ++calls), 0, 0));
  const noMap = f.compile();
  material.map = dataTexture();
  assert.throws(() => f.owner.captureDraw(noMap, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(calls, 0);
  const mapped = f.compile(); assert.notEqual(mapped, noMap);
  material.metalnessMap = material.roughnessMap = dataTexture();
  const shared = f.compile(), first = f.owner.captureDraw(shared, f.context);
  material.metalnessMap = dataTexture();
  assert.throws(() => f.owner.captureDraw(shared, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(calls, 1);
  const split = f.compile(); assert.notEqual(split, shared);
  const second = f.owner.captureDraw(split, {...f.context, renderId: 2});
  assert.equal(second.textures.filter(binding => binding.isSampledTexture).length,
    first.textures.filter(binding => binding.isSampledTexture).length + 1);
  f.object.geometry.setAttribute('uv1', f.object.geometry.attributes.uv.clone());
  material.map.channel = 1;
  const uv1 = f.compile(); assert.notEqual(uv1, split); assert.ok(uv1.attributes.some(attribute => attribute.name === 'uv1'));
  material.normalMap = dataTexture();
  const tangent = f.compile(); assert.match(tangent.fragment, /dFdx|dpdx/);
  material.normalMapType = THREE.ObjectSpaceNormalMap;
  assert.throws(() => f.owner.captureDraw(tangent, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const objectNormal = f.compile(); assert.notEqual(objectNormal, tangent);
});
test('source material references switch correctly between alternating objects and maps', t => {
  const f = pbrRig(t, THREE.MeshStandardMaterial), a = f.material;
  delete a.colorNode; a.positionNode = N.positionLocal; a.map = dataTexture(); a.map.offset.x = .125;
  a.color.setRGB(.125, .25, .375);
  const b = new THREE.MeshStandardMaterial({map: dataTexture()}); b.positionNode = N.positionLocal; b.map.offset.x = .625;
  b.color.setRGB(.625, .75, .875);
  const objectB = new THREE.Mesh(f.object.geometry, b); f.scene.add(objectB); f.scene.updateMatrixWorld();
  const contextB = {...f.context, object: objectB, material: b};
  const da = f.compile(), db = f.owner.compile(b, objectB, contextB);
  for (const [descriptor, context, material, value] of [[da, f.context, a, .125], [db, contextB, b, .625], [da, f.context, a, .125]]) {
    const draw = f.owner.captureDraw(descriptor, context);
    assert.ok(draw.textures.some(binding => binding.texture === material.map));
    assert.equal(draw.textures.some(binding => binding.texture === (material === a ? b.map : a.map)), false);
    assert.ok(floats(draw.uniforms).includes(value));
  }
});
test('source material conversion refuses unregistered classes, shader hooks and unsupported map dependencies', t => {
  const f = pbrRig(t, THREE.MeshStandardMaterial), material = f.material;
  const Custom = class extends THREE.MeshStandardMaterial {};
  const custom = new Custom(); custom.positionNode = N.positionLocal;
  assert.throws(() => f.owner.compile(custom, f.object, {...f.context, material: custom}), {code: 'THREE_NODE_PROGRAM_MATERIAL'});
  const originalConversion = f.renderer.library.fromMaterial;
  f.renderer.library.fromMaterial = () => new THREE.MeshStandardNodeMaterial();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_MATERIAL'}); f.renderer.library.fromMaterial = originalConversion;
  material.onBeforeCompile = () => {};
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_HOOK'}); delete material.onBeforeCompile;
  material.customProgramCacheKey = () => 'custom';
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_HOOK'}); delete material.customProgramCacheKey;
  for (const slot of ['envMap', 'lightMap', 'aoMap']) {
    material[slot] = dataTexture(); assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'}); material[slot] = null;
  }
  material.map = new THREE.CubeTexture(); assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});

test('source material maps and authored samples rebuild when a shared binding separates', t => {
  const f = pbrRig(t, THREE.MeshBasicMaterial), material = f.material;
  material.map = dataTexture(); const sampled = N.texture(material.map);
  material.colorNode = N.materialColor.mul(sampled);
  const shared = f.compile();
  assert.equal(f.owner.captureDraw(shared, f.context).textures.filter(binding => binding.isSampledTexture).length, 1);
  material.map = dataTexture();
  const separate = f.compile(); assert.notEqual(separate, shared);
  const draw = f.owner.captureDraw(separate, {...f.context, renderId: 2});
  const textures = draw.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
  assert.equal(textures.length, 2); assert.ok(textures.includes(material.map)); assert.ok(textures.includes(sampled.value));
});

test('source material map capture respects source reference-update order', t => {
  const f = pbrRig(t, THREE.MeshBasicMaterial), material = f.material, original = dataTexture(), replacement = dataTexture();
  material.map = original;
  const lateChange = N.uniform(1).onObjectUpdate(() => { material.map = replacement; return 1; });
  material.colorNode = N.materialColor.mul(lateChange);
  const d = f.compile(), first = f.owner.captureDraw(d, f.context);
  assert.equal(material.map, replacement);
  assert.ok(first.textures.some(binding => binding.texture === original), 'the source map reference updates before this fragment callback');
  assert.equal(first.textures.some(binding => binding.texture === replacement), false);
  assert.equal(f.compile(), d);
  const second = f.owner.captureDraw(d, {...f.context, renderId: 2});
  assert.ok(second.textures.some(binding => binding.texture === replacement));
});


function sunRig(t) {
  const f = pbrRig(t, THREE.MeshPhysicalMaterial), sun = new SunLight(new THREE.Color(.125, .25, .5), 2), group = new THREE.Group();
  for (const light of f.lights) f.scene.remove(light);
  group.position.set(3, .25, -2); sun.position.set(-4, 10, 8); group.add(sun); f.scene.add(group); f.scene.updateMatrixWorld();
  f.renderer.shadowMap = {enabled: false}; f.renderer.library.addLight(SunLightNode, SunLight);
  const context = {...f.context, lights: [sun]};
  return {...f, sun, group, context, compile: () => f.owner.compile(f.material, f.object, context)};
}
function hasVector(packets, values) {
  return packets.some(bytes => {
    const words = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    return words.some((_, index) => values.every((value, offset) => words[index + offset] === value));
  });
}
test('registered pinned SunLight uses source position-to-origin lighting with no shadow bindings', t => {
  const f = sunRig(t), d = f.compile(), draw = f.owner.captureDraw(d, f.context);
  assert.equal(f.sun.target, undefined);
  assert.ok(hasVector(draw.uniforms, [-1, 10.25, 6]), 'the source world-space light position reaches the render group');
  assert.ok(hasVector(draw.uniforms, [.25, .5, 1]), 'the analytic source node applies color times intensity');
  assert.match(d.fragment, /fn D_GGX/);
  const direction = /(\w+) = \( render\.cameraViewMatrix \* (\w+) \);/.exec(d.fragment);
  assert.ok(direction, 'the source sun direction uses the view matrix');
  assert.ok(new RegExp(`${direction[2]} = vec4<f32>\\( render\\.\\w+, 0\\.0 \\);`).test(d.fragment), 'a direction has zero homogeneous translation');
  assert.ok(new RegExp(`normalize\\( ${direction[1]}\\.xyz \\)`).test(d.fragment), 'the transformed source direction is normalized');
  assert.doesNotMatch(d.fragment, /texture_depth|textureSampleCompare/);
  assert.ok(draw.textures.every(binding => binding.texture.name === 'DFG_LUT'));
  assert.equal(f.sun.shadow.map, null); assert.equal(f.sun._listeners.dispose.length, 1);
});
test('registered SunLight retains one source FRAME node across objects and graph replacements', t => {
  const f = sunRig(t), first = f.compile();
  assert.ok(hasVector(f.owner.capture(first, f.context), [.25, .5, 1]));
  const material = new THREE.MeshStandardMaterial(); material.colorNode = N.vec3(.5);
  const object = new THREE.Mesh(f.object.geometry, material); f.scene.add(object);
  f.sun.intensity = 3; f.sun.position.set(-2, 4, 6); f.scene.updateMatrixWorld();
  const context = {...f.context, material, object, renderId: 2};
  const second = f.owner.compile(material, object, context), captured = f.owner.capture(second, context);
  assert.ok(hasVector(captured, [.25, .5, 1]), 'same-frame compilation does not repeat the shared analytic update');
  assert.ok(hasVector(captured, [1, 4.25, 4]), 'source light position still updates for a new render');
  material.roughnessNode = N.float(.375);
  const replacement = f.owner.compile(material, object, context); assert.notEqual(replacement, second);
  assert.ok(hasVector(f.owner.capture(replacement, {...context, renderId: 3}), [.25, .5, 1]));
  assert.ok(hasVector(f.owner.capture(replacement, {...context, renderId: 4, frameId: 2}), [.375, .75, 1.5]));
  assert.equal(f.sun._listeners.dispose.length, 1, 'all builders retain the original cached analytic node');
});
test('registered SunLight uses the source shared identity across renderer owners', t => {
  const a = sunRig(t), first = a.compile(); a.owner.capture(first, a.context);
  const b = pbrRig(t, THREE.MeshStandardMaterial); b.renderer.library.addLight(SunLightNode, SunLight);
  a.sun.intensity = 4;
  const context = {...b.context, lights: [a.sun], renderId: 7, frameId: 3};
  const second = b.owner.compile(b.material, b.object, context);
  assert.ok(hasVector(b.owner.capture(second, context), [.5, 1, 2]));
  assert.equal(a.sun._listeners.dispose.length, 1, 'a second renderer must not instantiate a duplicate source light node');
});
test('registered direct-light graph and constructor edits invalidate cached programs', t => {
  const f = pbrRig(t, THREE.MeshLambertMaterial);
  class CustomLight extends THREE.Light {}
  class OtherLight extends THREE.Light {}
  class CustomNode extends THREE.AnalyticLightNode {
    setupDirect() { return {lightDirection: N.vec3(1, 0, 0), lightColor: this.colorNode}; }
  }
  const light = new CustomLight(); f.scene.add(light);
  f.renderer.library.addLight(CustomNode, CustomLight); f.renderer.library.addLight(CustomNode, OtherLight);
  const context = {...f.context, lights: [light]}, first = f.owner.compile(f.material, f.object, context);
  CustomNode.prototype.setupDirect = function() { return {lightDirection: N.vec3(0, 1, 0), lightColor: this.colorNode}; };
  assert.throws(() => f.owner.capture(first, context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const changed = f.owner.compile(f.material, f.object, context); assert.notEqual(changed, first);
  assert.notEqual(changed.fragment, first.fragment);
  Object.setPrototypeOf(light, OtherLight.prototype);
  assert.throws(() => f.owner.capture(changed, context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const other = f.owner.compile(f.material, f.object, context); assert.notEqual(other, changed);
  assert.doesNotThrow(() => f.owner.capture(other, context));
});
test('registered light remapping refuses the source cached node instead of silently reusing its old class', t => {
  const f = sunRig(t), first = f.compile(); let updates = 0;
  f.material.colorNode = N.uniform(new THREE.Color(.5, .5, .5)).onObjectUpdate(() => { updates++; });
  class ReplacementNode extends THREE.AnalyticLightNode {
    setupDirect() { return {lightDirection: N.vec3(0, 0, 1), lightColor: this.colorNode}; }
  }
  f.renderer.library.lightNodes.set(SunLight, ReplacementNode);
  assert.throws(() => f.owner.capture(first, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'}); assert.equal(updates, 0);
  const fresh = new SunLight(), context = {...f.context, lights: [fresh]};
  assert.doesNotThrow(() => f.owner.compile(f.material, f.object, context), 'a fresh light identity uses the newly registered class');
});
test('registered SunLight admission never turns requested shadows or projected maps into unshadowed lighting', t => {
  const f = sunRig(t); let updates = 0;
  f.material.colorNode = N.uniform(new THREE.Color(.5, .5, .5)).onObjectUpdate(() => { updates++; });
  f.renderer.library.lightNodes.delete(SunLight);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'});
  f.renderer.library.addLight(SunLightNode, SunLight);
  for (const enabled of [false, true]) {
    f.renderer.shadowMap.enabled = enabled; f.sun.castShadow = true;
    assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'}); f.sun.castShadow = false;
    for (const key of ['castShadow', 'receiveShadow']) {
      f.object[key] = true; assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_SHADOW'}); f.object[key] = false;
    }
  }
  f.sun.map = dataTexture(); assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'}); f.sun.map = null;
  f.sun.colorNode = N.vec3(1); assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'});
  assert.equal(updates, 0); assert.equal(f.owner.diagnostics.programs, 0);
});
test('registered direct analytic nodes retain source lifecycle and native graph resource admission', t => {
  for (const kind of ['prototype lifecycle', 'instance lifecycle', 'storage write']) {
    const f = pbrRig(t, THREE.MeshLambertMaterial); let updates = 0;
    f.material.colorNode = N.uniform(new THREE.Color(.5, .5, .5)).onObjectUpdate(() => { updates++; });
    class CustomLight extends THREE.Light {}
    class CustomNode extends THREE.AnalyticLightNode {
      constructor(light) { super(light); if (kind === 'instance lifecycle') this.updateType = 'object'; }
      setupDirect() {
        const lightColor = kind === 'storage write' ? N.Fn(() => {
          N.storage(new THREE.StorageBufferAttribute(1, 4), 'vec4', 1).element(0).assign(N.vec4(1)); return N.vec3(1);
        })() : this.colorNode;
        return {lightDirection: N.vec3(0, 1, 0), lightColor};
      }
    }
    if (kind === 'prototype lifecycle') CustomNode.prototype.update = () => {};
    const light = new CustomLight(); f.scene.add(light); f.renderer.library.addLight(CustomNode, CustomLight);
    assert.throws(() => f.owner.compile(f.material, f.object, {...f.context, lights: [light]}),
      {code: kind === 'storage write' ? 'THREE_NODE_PROGRAM_STORAGE' : 'THREE_NODE_PROGRAM_LIGHTS'});
    assert.equal(updates, 0); assert.equal(f.owner.diagnostics.programs, 0);
  }
});
