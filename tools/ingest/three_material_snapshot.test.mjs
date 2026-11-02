import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.core.js';
import {captureThreeMaterialState} from './three_material_snapshot.mjs';

const sceneWith = (...materials) => {
  const scene = new THREE.Scene();
  for (const material of materials) scene.add(new THREE.Mesh(new THREE.BufferGeometry(), material));
  return scene;
};

test('material snapshots retain source identities and independent ordinary values', () => {
  const material = new THREE.MeshPhysicalMaterial({color: new THREE.Color(.125, .375, .625)});
  const map = new THREE.DataTexture(new Uint8Array(4), 1, 1);
  material.map = map;
  const scene = sceneWith(material, material), mesh = scene.children[0];
  const first = captureThreeMaterialState(THREE, scene);
  material.sheen = .5; material.color.setRGB(.875, .625, .375);
  const second = captureThreeMaterialState(THREE, scene);
  material.sheen = 0; material.color.setRGB(0, 0, 0); material.map = null;
  assert.equal(first.materials.size, 1);
  assert.equal(first.objects.get(mesh), material);
  assert.equal(first.materials.get(material).id, material.id);
  assert.equal(Object.getPrototypeOf(first.materials.get(material)), Object.getPrototypeOf(material));
  assert.deepEqual(first.materials.get(material).color.toArray(), [.125, .375, .625]);
  assert.deepEqual(second.materials.get(material).color.toArray(), [.875, .625, .375]);
  assert.equal(first.materials.get(material).sheen, 0);
  assert.equal(second.materials.get(material).sheen, .5);
  assert.equal(first.materials.get(material).map, map);
  assert.equal(second.materials.get(material).map, map);
  assert.equal(mesh.material, material);
  assert.ok(first.values > 0);
});

test('material arrays, override values and clipping planes retain their accepted selection', () => {
  const a = new THREE.MeshBasicMaterial(), b = new THREE.MeshPhongMaterial();
  const original = [a, b], scene = sceneWith(original);
  const override = new THREE.MeshBasicMaterial({color: 0x804020});
  const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 2);
  a.clippingPlanes = [plane]; a.allowOverride = false;
  const captured = captureThreeMaterialState(THREE, scene, {overrideMaterial: override});
  original[0] = b; scene.children[0].material = override;
  a.allowOverride = true; plane.normal.set(0, 1, 0); plane.constant = 8; a.clippingPlanes.push(plane);
  override.color.setRGB(0, 0, 0);
  assert.deepEqual(captured.objects.get(scene.children[0]), [a, b]);
  const state = captured.materials.get(a);
  assert.equal(state.allowOverride, false);
  assert.equal(state.clippingPlanes.length, 1);
  assert.notEqual(state.clippingPlanes[0], plane);
  assert.deepEqual(state.clippingPlanes[0].normal.toArray(), [1, 0, 0]);
  assert.equal(state.clippingPlanes[0].constant, 2);
  assert.notDeepEqual(captured.materials.get(override).color.toArray(), [0, 0, 0]);
});

test('native packets and excluded objects are not recaptured or updated', () => {
  const a = new THREE.MeshBasicMaterial(), b = new THREE.MeshPhongMaterial();
  const scene = sceneWith(a, b);
  let gets = 0;
  Object.defineProperty(a, 'color', {get() { gets++; throw new Error('native material reread'); }, configurable: true});
  const frame = {objects: new Set([scene.children[0]]), excludedObjects: new Set([scene.children[1]])};
  assert.equal(captureThreeMaterialState(THREE, scene, {nodeFrame: frame, maxValues: 0}), null);
  assert.equal(gets, 0);
});

test('capture preserves rejected material accessors without invoking them', () => {
  const material = new THREE.MeshBasicMaterial(), scene = sceneWith(material);
  let reads = 0;
  const getter = () => { reads++; return .5; };
  Object.defineProperty(material, 'opacity', {get: getter, configurable: true});
  const captured = captureThreeMaterialState(THREE, scene);
  assert.equal(reads, 0);
  assert.equal(Object.getOwnPropertyDescriptor(captured.materials.get(material), 'opacity').get, getter);
  const color = new THREE.Color();
  Object.defineProperty(color, 'r', {get: getter, configurable: true});
  material.color = color;
  const next = captureThreeMaterialState(THREE, scene);
  assert.equal(reads, 0);
  assert.equal(next.materials.get(material).color, color, 'custom accessor-backed values keep their existing live receiver');
});

test('material value, identity and graph limits reject before returning a partial packet', () => {
  const scene = sceneWith(new THREE.MeshBasicMaterial(), new THREE.MeshBasicMaterial());
  assert.throws(() => captureThreeMaterialState(THREE, scene, {maxValues: 1}), {code: 'THREE_MATERIAL_SNAPSHOT_LIMIT'});
  assert.throws(() => captureThreeMaterialState(THREE, scene, {maxMaterials: 1}), {code: 'THREE_MATERIAL_SNAPSHOT_LIMIT'});
  assert.throws(() => captureThreeMaterialState(THREE, scene, {maxNodes: 1}), {code: 'THREE_MATERIAL_SNAPSHOT_LIMIT'});
  scene.children.push(scene);
  assert.throws(() => captureThreeMaterialState(THREE, scene), {code: 'THREE_MATERIAL_SNAPSHOT_GRAPH'});
});

test('array-valued userData remains borrowed metadata outside the capture budget', () => {
  const material = new THREE.MeshBasicMaterial(), scene = sceneWith(material);
  const metadata = new Array(100000).fill(0); let nested = metadata;
  for (let i = 0; i < 6000; i++) { const next = []; nested[0] = next; nested = next; }
  material.userData = metadata;
  const captured = captureThreeMaterialState(THREE, scene, {maxValues: 256});
  assert.equal(captured.materials.get(material).userData, metadata);
  assert.ok(captured.values < 256);
});

test('exotic material selectors preserve their live receiver without invoking custom methods', () => {
  const material = new THREE.MeshBasicMaterial(); let calls = 0;
  class Selectors extends Array {
    [Symbol.iterator]() { calls++; return super[Symbol.iterator](); }
  }
  const subclass = new Selectors(material), iterable = [material], mapped = [material];
  iterable[Symbol.iterator] = function () { calls++; return Array.prototype[Symbol.iterator].call(this); };
  mapped.map = function (...args) { calls++; return Array.prototype.map.apply(this, args); };
  for (const selector of [subclass, iterable, mapped]) {
    const scene = sceneWith(selector);
    assert.equal(captureThreeMaterialState(THREE, scene), null);
    assert.equal(scene.children[0].material, selector);
  }
  assert.equal(calls, 0);
});

test('WebGL selections retain material identities and live shader uniform aliases', () => {
  const light = new THREE.DirectionalLight(), camera = new THREE.PerspectiveCamera();
  const a = new THREE.ShaderMaterial({uniforms: {lightPosition: {value: light.position}}});
  const b = new THREE.MeshPhongMaterial(), original = [a, b], scene = sceneWith(original);
  const mesh = scene.children[0], geometry = mesh.geometry;
  const first = captureThreeMaterialState(THREE, scene, {selectionOnly: true, camera});
  original[0] = b; mesh.material = b;
  light.position.set(2, 3, 4); a.opacity = .25;
  assert.deepEqual(first.objects.get(mesh), [a, b]);
  assert.equal(first.materials.get(a), a);
  assert.equal(first.materials.get(b), b);
  assert.equal(first.materials.get(a).uniforms.lightPosition.value, light.position);
  assert.deepEqual(first.materials.get(a).uniforms.lightPosition.value.toArray(), [2, 3, 4]);
  assert.equal(first.materials.get(a).opacity, .25);
  assert.equal(mesh.material, b); assert.equal(mesh.geometry, geometry);
  const next = captureThreeMaterialState(THREE, scene, {selectionOnly: true, camera});
  assert.equal(next.objects.get(mesh), b);
  assert.equal(next.materials.has(a), false);
});

test('WebGL material selection leaves projection and upload hooks on their source path', () => {
  let calls = 0;
  const hook = () => { calls++; };
  const mutations = [
    ({scene}) => { scene.onBeforeRender = hook; },
    ({mesh}) => { mesh.onAfterRender = hook; },
    ({scene}) => { scene.updateMatrixWorld = hook; },
    ({mesh}) => { mesh.updateMatrix = hook; },
    ({camera}) => { camera.updateMatrixWorld = hook; },
    ({camera}) => { camera.updateMatrix = hook; },
    ({material}) => { material.onBeforeRender = hook; },
    ({material}) => { material.onBeforeCompile = hook; },
    ({material}) => { material.customProgramCacheKey = hook; },
    ({mesh}) => { mesh.intersectsFrustum = hook; },
    ({mesh}) => { mesh.layers.test = hook; },
    ({mesh}) => { mesh.boundingSphere = null; mesh.computeBoundingSphere = hook; },
    ({geometry}) => { geometry.computeBoundingSphere = hook; },
    ({geometry}) => { geometry.attributes.position.onUpload(hook); },
    ({geometry}) => { geometry.setIndex([0, 1, 2]); geometry.index.onUpload(hook); },
    ({geometry}) => {
      const data = new THREE.InterleavedBuffer(new Float32Array(9), 3);
      data.onUpload(hook); geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(data, 3, 0));
    },
    ({geometry}) => { geometry.morphAttributes.position = [new THREE.Float32BufferAttribute(9, 3).onUpload(hook)]; },
    ({scene}) => { const lod = new THREE.LOD(); lod.update = hook; scene.add(lod); },
  ];
  for (const mutate of mutations) {
    const material = new THREE.MeshBasicMaterial(), scene = sceneWith(material), mesh = scene.children[0];
    const camera = new THREE.PerspectiveCamera(), geometry = mesh.geometry;
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(9, 3));
    mutate({scene, mesh, material, camera, geometry});
    assert.equal(captureThreeMaterialState(THREE, scene, {selectionOnly: true, camera}), null);
  }
  assert.equal(calls, 0, 'capture never invokes the hooks it detects');
});

test('WebGL selector eligibility never invokes accessor-backed or exotic source state', () => {
  let calls = 0;
  const getter = () => { calls++; throw new Error('capture must not invoke source accessor'); };
  const mutations = [
    ({mesh}) => { Object.defineProperty(mesh, 'material', {get: getter}); },
    ({mesh}) => { Object.defineProperty(mesh, 'onBeforeRender', {get: getter}); },
    ({scene}) => { Object.defineProperty(scene, 'children', {get: getter}); },
    ({camera}) => { Object.defineProperty(camera, 'updateMatrixWorld', {get: getter}); },
    ({material}) => { Object.defineProperty(material, 'onBeforeCompile', {get: getter}); },
    ({mesh, material}) => { const array = [material]; Object.defineProperty(array, '0', {get: getter}); mesh.material = array; },
    ({mesh, material}) => { const array = [material]; array[Symbol.iterator] = getter; mesh.material = array; },
    ({geometry}) => { Object.defineProperty(geometry.attributes.position, 'onUploadCallback', {get: getter}); },
    ({mesh}) => { Object.defineProperty(mesh.layers, 'mask', {get: getter}); },
    ({geometry}) => { geometry.addGroup(0, 3, 0); Object.defineProperty(geometry.groups[0], 'materialIndex', {get: getter}); },
    ({geometry}) => { geometry.groups.push(Object.assign(Object.create({get materialIndex() { return getter(); }}), {start: 0, count: 3})); },
  ];
  for (const mutate of mutations) {
    const material = new THREE.MeshBasicMaterial(), scene = sceneWith(material), mesh = scene.children[0];
    const camera = new THREE.PerspectiveCamera(), geometry = mesh.geometry;
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(9, 3));
    mutate({scene, mesh, material, camera, geometry});
    assert.equal(captureThreeMaterialState(THREE, scene, {selectionOnly: true, camera}), null);
  }
  assert.equal(calls, 0);
});

test('WebGL selector capture enforces pending-value, identity and graph budgets', () => {
  const scene = sceneWith(new THREE.MeshBasicMaterial(), new THREE.MeshPhongMaterial());
  for (const limits of [{maxValues: 2}, {maxMaterials: 1}, {maxNodes: 1}])
    assert.throws(() => captureThreeMaterialState(THREE, scene, {selectionOnly: true, ...limits}),
      {code: 'THREE_MATERIAL_SNAPSHOT_LIMIT'});
  scene.children.push(scene);
  assert.throws(() => captureThreeMaterialState(THREE, scene, {selectionOnly: true}),
    {code: 'THREE_MATERIAL_SNAPSHOT_GRAPH'});
});
