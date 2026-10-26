import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';

const N = THREE.TSL;
function rig(t) {
  const frame = new THREE.NodeFrame(); Object.assign(frame, {frameId: 7, time: 1.25, deltaTime: .125});
  let recorder = null, copyId = 0;
  const released = [], created = [];
  const textures = {captureDispatch: run => run(), release: lease => released.push(lease),
    captureBinding: binding => Object.freeze({binding, texture: binding.texture})};
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true}, sortObjects: true,
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null,
    getCanvasTarget: () => null, getDrawingBufferSize: size => size.set(32, 16),
    getViewport: value => value.set(0, 0, 32, 16), getPixelRatio: () => 1,
    _getSampledTextureInputs: () => ({layoutOf: ({texture}) => texture.isFramebufferTexture
      ? {texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}} : null}),
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: frame}, info: {calls: 3},
    copyFramebufferToTexture(texture) {
      assert.ok(recorder, 'copies belong to a physical draw capture');
      const marker = Object.freeze({lease: Object.freeze({id: ++copyId, texture}), width: texture.image.width,
        height: texture.image.height, mipmaps: texture.generateMipmaps});
      created.push(marker); recorder(marker);
    }};
  const programs = createThreeNodePrograms({three: THREE, renderer}); t.after(() => programs.dispose());
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 1, .1, 100); camera.position.z = 5;
  const mesh = (name, options = {}) => {
    const material = new THREE.MeshBasicNodeMaterial(); material.colorNode = N.vec3(.25, .5, .75);
    // RenderList classifies by these public properties for every material type.
    // Basic isolates ordering from the separate physical-lighting shader tests.
    Object.assign(material, options);
    const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); object.name = name; scene.add(object); return object;
  };
  const capture = (options = {}) => captureThreeNodeScene(THREE, scene, camera, {
    getPrograms: () => programs, renderer, getTextures: () => textures,
    raster: material => Object.freeze({side: material.side}),
    withFramebufferCopies(record, run) {
      const previous = recorder; recorder = record;
      try { return run(); } finally { recorder = previous; }
    }, ...options});
  return {renderer, programs, scene, camera, frame, mesh, capture, textures, released, created};
}
function sourceTransparents(objects, scene, camera, observe) {
  const renderList = objects.map(object => ({object, geometry: object.geometry, material: object.material, group: null}));
  const doublePass = renderList.filter(({material}) =>
    (material.transmission > 0 || material.transmissionNode?.isNode) && material.side === THREE.DoubleSide && material.forceSinglePass === false);
  const source = {_currentSourceMaterial: null,
    _handleObjectFunction(object, material, _scene, _camera, _lights, _group, _clipping, passId) { observe(object, material.side, passId); },
    _renderObjects(items, selectedCamera, selectedScene, lights, passId = null) {
      for (const {object, geometry, material, group} of items)
        THREE.WebGPURenderer.prototype.renderObject.call(this, object, selectedScene, selectedCamera,
          geometry, material, group, lights, null, passId);
    }};
  THREE.WebGPURenderer.prototype._renderTransparents.call(source, renderList, doublePass, camera, scene, null);
}
test('native transmission captures the pinned all-backfaces prefix before mixed transparent objects', t => {
  for (const sortObjects of [false, true]) {
    const f = rig(t), first = f.mesh('first glass', {transmission: .8, side: THREE.DoubleSide}),
      alpha = f.mesh('alpha', {transparent: true, side: THREE.DoubleSide}),
      second = f.mesh('second glass', {transmission: .4, side: THREE.DoubleSide}), opaque = f.mesh('opaque');
    first.renderOrder = 2; alpha.renderOrder = 1; second.renderOrder = 0; f.renderer.sortObjects = sortObjects;
    const observed = [], sides = () => [first.material.side, second.material.side];
    for (const object of [first, alpha, second, opaque]) object.material.colorNode = N.vec3(
      N.uniform(.5).onObjectUpdate(() => { observed.push([object.name, object.material.side, sides()]); return .5; }));
    const ordered = sortObjects ? [second, alpha, first] : [first, alpha, second], expected = [];
    sourceTransparents(ordered, f.scene, f.camera, (object, side) => expected.push([object.name, side, sides()]));
    const result = f.capture();
    assert.deepEqual(observed, [[opaque.name, THREE.FrontSide, [THREE.DoubleSide, THREE.DoubleSide]], ...expected]);
    assert.deepEqual(result.draws.filter(draw => draw.transmissionBack).map(draw => draw.object),
      ordered.filter(object => object !== alpha));
    assert.deepEqual(result.draws.map(draw => [draw.object.name, draw.binding.raster.side]),
      observed.map(([name, side]) => [name, side]));
    assert.deepEqual(sides(), [THREE.DoubleSide, THREE.DoubleSide]);
    assert.equal(result.bytes, result.draws.reduce((sum, draw) =>
      sum + draw.nodeUniforms.reduce((n, bytes) => n + bytes.byteLength, 0), 0));
  }
});
test('shared transmissive materials retain one back and front draw per source group', t => {
  const f = rig(t), first = f.mesh('first', {transmission: 1, side: THREE.DoubleSide}), second = f.mesh('second');
  second.material = first.material;
  first.geometry.clearGroups(); first.geometry.addGroup(0, 3, 0); first.geometry.addGroup(3, 3, 0);
  first.material = [first.material]; f.renderer.sortObjects = false;
  const result = f.capture();
  assert.deepEqual(result.draws.map(draw => [draw.object.name, draw.first, draw.transmissionBack, draw.binding.raster.side]), [
    ['first', 0, true, THREE.BackSide], ['first', 3, true, THREE.BackSide], ['second', 0, true, THREE.BackSide],
    ['first', 0, false, THREE.FrontSide], ['first', 3, false, THREE.FrontSide], ['second', 0, false, THREE.FrontSide],
  ]);
  assert.equal(second.material.side, THREE.DoubleSide);
  assert.equal(new Set(result.bindings.map(binding => binding.nodeProgram)).size, 4, 'each object retains both source pass identities');
});
test('transmission respects forceSinglePass and explicit sides without transparent=true', t => {
  for (const [side, forceSinglePass, expected] of [
    [THREE.DoubleSide, false, [THREE.BackSide, THREE.FrontSide]],
    [THREE.DoubleSide, true, [THREE.DoubleSide]],
    [THREE.FrontSide, false, [THREE.FrontSide]],
    [THREE.BackSide, false, [THREE.BackSide]],
  ]) {
    const f = rig(t), object = f.mesh('glass', {transmission: .5, side, forceSinglePass});
    const result = f.capture();
    assert.deepEqual(result.draws.map(draw => draw.binding.raster.side), expected);
    assert.equal(object.material.side, side); assert.equal(object.material.transparent, false);
  }
});
test('transmissive variants preflight draw budgets and compile without source updates', t => {
  const f = rig(t), object = f.mesh('glass', {transmission: 1, side: THREE.DoubleSide}); let updates = 0;
  object.material.colorNode = N.vec3(N.uniform(.25).onObjectUpdate(() => { updates++; return .25; }));
  const prepared = f.capture({uniforms: false});
  assert.equal(prepared.bindings.length, 2); assert.equal(prepared.draws.length, 0); assert.equal(updates, 0);
  assert.throws(() => f.capture({maxDraws: 1}), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  assert.equal(updates, 0); assert.equal(object.material.side, THREE.DoubleSide);
});
test('copies from before/current/after updates stay attached to their exact physical draw', t => {
  const f = rig(t), texture = new THREE.FramebufferTexture(8, 4), object = f.mesh('glass', {transmission: 1, side: THREE.DoubleSide});
  const value = N.uniform(.25).onObjectUpdate(frame => {
    texture.image.width = frame.material.side === THREE.BackSide ? 12 : 24;
    frame.renderer.copyFramebufferToTexture(texture); return .25;
  });
  value.updateBeforeType = 'object'; value.updateBefore = frame => frame.renderer.copyFramebufferToTexture(texture);
  value.updateAfterType = 'object'; value.updateAfter = frame => {
    texture.generateMipmaps = true; frame.renderer.copyFramebufferToTexture(texture); texture.generateMipmaps = false;
  };
  object.material.colorNode = N.vec3(value);
  const result = f.capture();
  assert.deepEqual(result.draws.map(draw => draw.nodeFramebufferCopies.map(copy => copy.width)), [[8, 12], [12, 24]]);
  assert.deepEqual(result.draws.map(draw => draw.nodeFramebufferAfterCopies.map(copy => [copy.width, copy.mipmaps])),
    [[[12, true]], [[24, true]]]);
  assert.deepEqual(result.textureUses, f.created.map(copy => copy.lease));
  assert.ok(result.draws.every(draw => Object.isFrozen(draw.nodeFramebufferCopies) && Object.isFrozen(draw.nodeFramebufferAfterCopies)));
  f.renderer.info.calls++; texture.image.width = 64;
  const next = f.capture();
  assert.equal(result.draws[0].nodeFramebufferCopies[0].width, 8);
  assert.equal(next.draws[0].nodeFramebufferCopies[0].width, 64);
  assert.equal(new Set([...result.textureUses, ...next.textureUses]).size, 12);
});
test('copy leases and every transmissive side unwind when a later draw update fails', t => {
  const f = rig(t), first = f.mesh('first', {transmission: 1, side: THREE.DoubleSide}),
    second = f.mesh('second', {transmission: 1, side: THREE.DoubleSide}), texture = new THREE.FramebufferTexture(8, 4);
  f.renderer.sortObjects = false;
  for (const object of [first, second]) object.material.colorNode = N.vec3(N.uniform(.25).onObjectUpdate(frame => {
    frame.renderer.copyFramebufferToTexture(texture);
    if (object === second && frame.material.side === THREE.BackSide) throw new Error('backface update failed');
    return .25;
  }));
  assert.throws(f.capture, /backface update failed/);
  assert.deepEqual(f.released, f.created.map(copy => copy.lease)); assert.equal(f.released.length, 2);
  assert.equal(first.material.side, THREE.DoubleSide); assert.equal(second.material.side, THREE.DoubleSide);
  assert.throws(() => f.renderer.copyFramebufferToTexture(texture), /copies belong to a physical draw capture/);
});
test('empty native draws keep their before and after copy boundaries', t => {
  const f = rig(t), object = f.mesh('empty'), texture = new THREE.FramebufferTexture(8, 4);
  object.geometry.setDrawRange(0, 0);
  const value = N.uniform(.25).onObjectUpdate(frame => { frame.renderer.copyFramebufferToTexture(texture); return .25; });
  value.updateAfterType = 'object'; value.updateAfter = frame => frame.renderer.copyFramebufferToTexture(texture);
  object.material.colorNode = N.vec3(value);
  const result = f.capture();
  assert.equal(result.draws[0].count, 0); assert.equal(result.draws[0].nodeFramebufferCopies.length, 1);
  assert.equal(result.draws[0].nodeFramebufferAfterCopies.length, 1); assert.equal(result.textureUses.length, 2);
});
test('physical and authored transmission share source front/back copies with a viewport-sampling alpha draw', t => {
  const f = rig(t), first = f.mesh('ordinary glass'), second = f.mesh('node glass'), alpha = f.mesh('alpha', {transparent: true});
  first.material = new THREE.MeshPhysicalMaterial({transmission: .75, thickness: .25, dispersion: .5, side: THREE.DoubleSide});
  second.material = new THREE.MeshPhysicalNodeMaterial({thickness: .5, side: THREE.DoubleSide});
  second.material.transmissionNode = N.uniform(.5);
  alpha.material.colorNode = N.viewportOpaqueMipTexture().rgb;
  f.scene.add(new THREE.AmbientLight(0xffffff, .5));
  alpha.renderOrder = 0; first.renderOrder = 1; second.renderOrder = 2;
  const prepared = f.capture({uniforms: false});
  assert.equal(prepared.bindings.length, 5); assert.equal(f.created.length, 0, 'precompile does not snapshot the framebuffer');
  const result = f.capture();
  assert.deepEqual(result.draws.map(draw => [draw.object.name, draw.binding.raster.side]), [
    ['ordinary glass', THREE.BackSide], ['node glass', THREE.BackSide], ['alpha', THREE.FrontSide],
    ['ordinary glass', THREE.FrontSide], ['node glass', THREE.FrontSide],
  ]);
  assert.equal(f.created.length, 2, 'each source RENDER-frequency front/back texture is copied exactly once');
  assert.deepEqual(result.draws.map(draw => draw.nodeFramebufferCopies.length), [1, 0, 1, 0, 0]);
  const [back, front] = f.created;
  assert.notEqual(back.lease.texture, front.lease.texture);
  assert.ok(result.draws.slice(3).every(draw => draw.nodeTextures.some(binding => binding.lease.texture === front.lease.texture)));
  assert.ok(result.draws.slice(0, 2).every(draw => draw.nodeTextures.some(binding => binding.lease.texture === back.lease.texture)));
  assert.ok(f.created.every(copy => copy.mipmaps && copy.width === 32 && copy.height === 16));
  assert.equal(first.material.side, THREE.DoubleSide); assert.equal(second.material.side, THREE.DoubleSide);
});
test('distinct transmissive DoubleSide overrides reject before replacing the source four-pass expansion', t => {
  const f = rig(t), object = f.mesh('glass');
  object.material = new THREE.MeshPhysicalMaterial({transmission: .75, side: THREE.DoubleSide});
  const override = new THREE.MeshPhysicalNodeMaterial({transparent: true, side: THREE.DoubleSide});
  let updates = 0;
  override.colorNode = N.uniform(new THREE.Color(.25, .5, .75)).onObjectUpdate(() => { updates++; });
  f.scene.overrideMaterial = override;
  const expected = [];
  sourceTransparents([object], f.scene, f.camera, (_object, side) => expected.push(side));
  assert.deepEqual(expected, [THREE.BackSide, THREE.FrontSide, THREE.BackSide, THREE.FrontSide]);
  assert.throws(() => f.capture({overrideMaterial: override}), error =>
    error.code === 'THREE_NODE_PROGRAM_EFFECT' && /original and effective pass expansion/.test(error.message));
  assert.equal(updates, 0); assert.equal(f.created.length, 0);
  assert.equal(object.material.side, THREE.DoubleSide); assert.equal(override.side, THREE.DoubleSide);
});
