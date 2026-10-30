/** Public source sort callbacks through native and WebGL-authored renderers.
 * Pinned RenderList implementations supply the ordering/callback oracle; the
 * recording GPU establishes the actual submitted draw order, not pixel parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createWebGPURendererClass, createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const source = name => import(pathToFileURL(path.join(root, name)));
const T = await source('build/three.webgpu.js'), GL = await source('build/three.module.js');
const {default: RenderList} = await source('src/renderers/common/RenderList.js');
const {WebGLRenderList} = await source('src/renderers/webgl/WebGLRenderLists.js');
const key = item => `${item.object.name}:${item.group?.materialIndex ?? '-'}`;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };

async function fixture(t, native = true) {
  const three = native ? T : GL, device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); },
  })};
  const Renderer = native ? createWebGPURendererClass(three) : createWebGLRendererClass(three);
  const renderer = new Renderer({canvas, device});
  if (native) renderer.outputColorSpace = three.LinearSRGBColorSpace;
  t.after(() => renderer.dispose()); await renderer.init();
  const scene = new three.Scene(), camera = new three.PerspectiveCamera(50, 1, .1, 20);
  camera.position.z = 4;
  camera.coordinateSystem = native ? three.WebGPUCoordinateSystem : three.WebGLCoordinateSystem;
  camera.updateProjectionMatrix();
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const result = capture.call(this, request); if (result) captures.push(result); return result;
  };
  let marker = 0;
  const material = (options = {}) => {
    const result = native ? new three.MeshBasicNodeMaterial(options) : new three.MeshBasicMaterial(options);
    if (native) result.colorNode = three.TSL.vec3(.25, .5, .75);
    t.after(() => result.dispose()); return result;
  };
  const mesh = (name, rank, selected = material(), parent = scene) => {
    const geometry = new three.PlaneGeometry(); geometry.attributes.position.setX(0, ++marker / 16);
    const object = new three.Mesh(geometry, selected); object.name = name; object.userData.rank = rank;
    object.frustumCulled = false; parent.add(object); t.after(() => geometry.dispose()); return object;
  };
  return {three, native, device, renderer, scene, camera, captures, material, mesh};
}

function referenceList(f) {
  const {three, scene, camera, native} = f;
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const list = native ? new RenderList({enabled: true, getNode: () => ({setLights() {}})}, scene, camera) : WebGLRenderList();
  native ? list.begin() : list.init();
  const projection = new three.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const center = native ? new three.Vector4() : new three.Vector3();
  const visit = (object, groupOrder) => {
    if (!object.visible) return;
    if (object.isGroup) groupOrder = object.renderOrder;
    if (object.isMesh && object.layers.test(camera.layers)) {
      const geometry = object.geometry;
      if (geometry.boundingSphere === null) geometry.computeBoundingSphere();
      const z = center.copy(geometry.boundingSphere.center).applyMatrix4(object.matrixWorld).applyMatrix4(projection).z;
      if (Array.isArray(object.material)) {
        for (const group of geometry.groups) if (object.material[group.materialIndex]?.visible)
          list.push(object, geometry, object.material[group.materialIndex], groupOrder, z, group, native ? null : camera);
      } else if (object.material.visible) list.push(object, geometry, object.material, groupOrder, z, null, native ? null : camera);
    }
    for (const child of object.children) visit(child, groupOrder);
  };
  visit(scene, 0); list.finish(); return list;
}

function comparator(trace, direction = 1, label = 'sort', native = true) {
  const seen = new Map();
  return function(a, b) {
    assert.equal(this, undefined, 'Array.sort calls the comparator without a renderer receiver');
    for (const item of [a, b]) {
      assert.deepEqual(Object.keys(item).sort(), ['id', 'object', 'geometry', 'material', 'groupOrder', 'renderOrder', 'z', 'group',
        native ? 'clippingContext' : 'materialVariant'].sort());
      assert.equal(item.id, item.object.id); assert.equal(item.geometry, item.object.geometry);
      assert.equal(item.material, Array.isArray(item.object.material) ? item.object.material[item.group.materialIndex] : item.object.material);
      if (item.group) assert.ok(item.geometry.groups.includes(item.group), 'the original source group reaches the callback');
      if (!native) assert.equal(item.materialVariant, (item.object.isInstancedMesh ? 2 : 0) + (item.object.isSkinnedMesh ? 1 : 0));
      const identity = key(item);
      if (seen.has(identity)) assert.equal(item, seen.get(identity), 'one item identity is reused within its list');
      else seen.set(identity, item);
    }
    trace.push([label, key(a), key(b)]);
    return direction * (a.object.userData.rank - b.object.userData.rank || (a.group?.materialIndex ?? 0) - (b.group?.materialIndex ?? 0));
  };
}

function submittedMarkers(device) {
  return device.snapshots.flat().filter(draw => draw.streams.size).map(draw => {
    const bytes = draw.contents.get(draw.streams.get(0));
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)[0];
  });
}
const markers = items => items.map(item => item.object.geometry.attributes.position.getX(0));

test('native custom sorting shares original items across mixed materials, groups and source callbacks', async t => {
  const f = await fixture(t), group = new T.Group(); group.renderOrder = 9; f.scene.add(group);
  const ordinary = new T.MeshBasicMaterial(); t.after(() => ordinary.dispose());
  const a = f.mesh('ordinary', 4, ordinary), b = f.mesh('node', 1), c = f.mesh('groups', 2, [f.material(), f.material()], group);
  c.geometry.clearGroups(); c.geometry.addGroup(0, 3, 0); c.geometry.addGroup(3, 3, 1);
  f.mesh('alpha-a', 3, f.material({transparent: true})); f.mesh('alpha-b', 0, f.material({transparent: true}));
  a.renderOrder = -5; b.renderOrder = 12;
  const expectedTrace = [], reference = referenceList(f);
  reference.sort(comparator(expectedTrace, 1, 'opaque'), comparator(expectedTrace, -1, 'transparent'));
  const expected = [...reference.opaque, ...reference.transparent], events = [];
  f.scene.traverse(object => {
    if (!object.isMesh) return;
    object.onBeforeRender = (_renderer, _scene, _camera, _geometry, material, group) => {
      events.push(['before', key({object, group})]);
      assert.equal(material, Array.isArray(object.material) ? object.material[group.materialIndex] : object.material);
    };
    object.onAfterRender = (_renderer, _scene, _camera, _geometry, _material, group) => events.push(['after', key({object, group})]);
  });
  assert.equal(f.renderer.setOpaqueSort(comparator(events, 1, 'opaque')), undefined);
  assert.equal(f.renderer.setTransparentSort(comparator(events, -1, 'transparent')), undefined);
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, [...expectedTrace, ...expected.flatMap(item => [['before', key(item)], ['after', key(item)]])]);
  assert.deepEqual(f.captures[0].draws.map(draw => key({object: draw.object, group: draw.sourceGroup})), expected.map(key));
  const synchronous = events.slice(); await drain(f.renderer);
  assert.deepEqual(events, synchronous, 'preparation and submission never replay the sorter or object hooks');
  assert.deepEqual(submittedMarkers(f.device), markers(expected));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const native of [true, false]) test(`${native ? 'native' : 'WebGL'} queued canvas/target draws retain comparator changes, reset and disabled sorting`, async t => {
  const f = await fixture(t, native), events = [];
  for (const [name, rank] of [['a', 1], ['b', 3], ['c', 2]]) {
    const object = f.mesh(name, rank);
    object.onBeforeRender = () => events.push(name);
  }
  const target = native ? new T.RenderTarget(32, 32) : new GL.WebGLRenderTarget(32, 32);
  t.after(() => target.dispose());
  const expected = [], comparisons = [];
  for (const [index, direction] of [-1, 1, null, false].entries()) {
    const list = referenceList(f), sorter = direction === null || direction === false ? null : comparator([], direction, 'sort', native);
    if (direction !== false) list.sort(sorter, null);
    expected.push(...list.opaque);
    f.renderer.sortObjects = direction !== false;
    f.renderer.setOpaqueSort(direction === false ? () => assert.fail('disabled sorter executed') :
      direction === null ? null : comparator(comparisons, direction, `call-${index}`, native));
    f.renderer.setRenderTarget(index % 2 === 0 ? target : null);
    f.renderer.render(f.scene, f.camera);
  }
  f.renderer.setOpaqueSort(() => assert.fail('a later sorter replaced a queued request'));
  const beforeDrain = comparisons.length;
  await drain(f.renderer);
  assert.deepEqual(events, expected.map(item => item.object.name));
  assert.deepEqual(submittedMarkers(f.device), markers(expected));
  if (native) assert.equal(comparisons.length, beforeDrain, 'native comparators completed at render-call capture');
  else assert.deepEqual([...new Set(comparisons.map(event => event[0]))], ['call-0', 'call-1']);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._pending.length, 0);
});

test('WebGL comparator inputs and opaque/transparent callback order match the pinned render list', async t => {
  const f = await fixture(t, false), group = new GL.Group(); group.renderOrder = 7; f.scene.add(group);
  f.mesh('opaque-a', 2); f.mesh('opaque-b', 1);
  const grouped = f.mesh('grouped', 3, [f.material(), f.material()], group);
  grouped.geometry.clearGroups(); grouped.geometry.addGroup(0, 3, 0); grouped.geometry.addGroup(3, 3, 1);
  f.mesh('alpha-a', 1, f.material({transparent: true})); f.mesh('alpha-b', 2, f.material({transparent: true}));
  const reference = referenceList(f), expectedTrace = [], actualTrace = [];
  reference.sort(comparator(expectedTrace, -1, 'opaque', false), comparator(expectedTrace, 1, 'transparent', false));
  f.renderer.setOpaqueSort(comparator(actualTrace, -1, 'opaque', false));
  f.renderer.setTransparentSort(comparator(actualTrace, 1, 'transparent', false));
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.deepEqual(actualTrace, expectedTrace);
  assert.deepEqual(submittedMarkers(f.device), markers([...reference.opaque, ...reference.transparent]));
});

test('native transmission sorts the separate back-face list before transparent objects without callback replay', async t => {
  const f = await fixture(t);
  for (const [name, rank, glass] of [['glass-a', 1, true], ['alpha', 2, false], ['glass-b', 3, true]]) {
    const material = f.material({side: T.DoubleSide});
    // The source list uses these properties on every material class; Basic
    // isolates ordering from Physical's separate scene-color copy machinery.
    if (glass) material.transmission = 1; else material.transparent = true;
    f.mesh(name, rank, material);
  }
  const reference = referenceList(f), expectedTrace = [], trace = [];
  const stateful = events => {
    const inspect = comparator(events);
    return function(a, b) { const result = inspect(a, b); return events.length === 1 ? -result : result; };
  };
  reference.sort(null, stateful(expectedTrace));
  f.renderer.setTransparentSort(stateful(trace)); f.renderer.render(f.scene, f.camera);
  assert.deepEqual(trace, expectedTrace, 'both lists use the source comparison sequence and shared item identities');
  const expected = [...reference.transparentDoublePass, ...reference.transparent.flatMap(item =>
    item.material.transparent ? [item, item] : [item])];
  assert.deepEqual(f.captures[0].draws.map(draw => draw.object.name), expected.map(item => item.object.name));
  await drain(f.renderer);
  assert.deepEqual(trace, expectedTrace); assert.deepEqual(submittedMarkers(f.device), markers(expected));
});

test('native compilation leaves comparators inert and comparator errors occur before any GPU effects', async t => {
  const f = await fixture(t); f.mesh('a', 2); f.mesh('b', 1);
  const failure = new Error('sort callback failed'); let calls = 0;
  f.renderer.setOpaqueSort(() => { calls++; throw failure; });
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(calls, 0, 'pinned compileAsync projects without sorting');
  assert.throws(() => f.renderer.render(f.scene, f.camera), error => error === failure);
  assert.equal(calls, 1); assert.equal(f.device.submissions.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._pending.length, 0);
  f.renderer.setOpaqueSort(null); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(submittedMarkers(f.device).length, 2);
});
