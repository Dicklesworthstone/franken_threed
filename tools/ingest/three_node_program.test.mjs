import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms, hasThreeNodeProgram} from './three_node_program.mjs';
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
  ['material texture', f => { f.material.map = new THREE.Texture(); }, 'TEXTURE'],
  ['MRT graph', f => { f.material.fragmentNode = N.mrt({output: N.vec4(1)}); }, 'MRT'],
  ['lighting slot', f => { f.material.lightsNode = N.vec3(1); }, 'GRAPH'],
  ['transparent double side', f => { f.material.side = THREE.DoubleSide; f.material.transparent = true; }, 'RASTER'],
  ['custom renderer context', f => { f.renderer.contextNode = N.context({getOutput: output => output}); }, 'GRAPH'],
  ['instancing', f => { f.object.isInstancedMesh = true; }, 'OBJECT'],
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
  assert.throws(() => f.owner.captureDraw(d, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.notEqual(f.compile(), d);
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
