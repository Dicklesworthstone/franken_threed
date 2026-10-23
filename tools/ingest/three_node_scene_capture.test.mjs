import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';
import {withThreeNativeCapture, assertThreeNativeTargetWrite, assertThreeNativeComputeOutsideCapture} from './three_node_shadows.mjs';
import {createGpuCompute} from './gpu_compute.mjs';
import {captureComputeStorage} from './gpu_compute_attributes.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const N = THREE.TSL;
function rig(t) {
  const frame = new THREE.NodeFrame(); Object.assign(frame, {frameId: 7, time: 1.25, deltaTime: .125});
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true}, sortObjects: true,
    backend: {hasFeature: () => false}, hasFeature: () => false, getRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: frame}, info: {calls: 3}};
  const owner = createThreeNodePrograms({three: THREE, renderer}); t.after(() => owner.dispose());
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 1, .1, 100); camera.position.z = 5;
  const material = () => { const m = new THREE.MeshBasicNodeMaterial(); m.colorNode = N.vec3(.25, .5, .75); return m; };
  const mesh = (m = material(), parent = scene) => { const o = new THREE.Mesh(new THREE.PlaneGeometry(), m); parent.add(o); return o; };
  const capture = (options = {}) => captureThreeNodeScene(THREE, scene, camera, {
    getPrograms: () => owner, renderer, raster: m => Object.freeze({side: m.side, depthWrite: m.depthWrite}), ...options});
  return {owner, frame, renderer, scene, camera, material, mesh, capture};
}
const floats = packets => packets.flatMap(bytes => Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)));
test('ordinary scenes do not create a node compiler or change the source camera', t => {
  const f = rig(t); f.mesh(new THREE.MeshBasicMaterial());
  assert.equal(f.capture({getPrograms() { assert.fail('unexpected native compiler'); }}), null);
  assert.equal(f.camera.coordinateSystem, THREE.WebGLCoordinateSystem); assert.equal(f.frame.frameId, 7);
});
test('native shadow intent promotes default mesh materials without changing fixed-only scenes', t => {
  const fixed = rig(t); fixed.renderer.shadowMap = {enabled: true}; fixed.mesh(new THREE.MeshBasicMaterial());
  assert.equal(fixed.capture({getPrograms() { assert.fail('fixed-only shadows keep their existing owner'); }}), null);
  for (const authored of [false, true]) {
    const f = rig(t); f.renderer.shadowMap = {enabled: true};
    const ordinary = f.mesh(new THREE.MeshBasicMaterial()), native = f.mesh(authored ? f.material() : new THREE.MeshBasicNodeMaterial());
    const result = f.capture();
    assert.deepEqual([...result.objects], [ordinary, native]);
    assert.equal(result.bindings[0].material, ordinary.material); assert.equal(result.bindings.length, 2);
    assert.deepEqual(result.draws.map(draw => draw.object), [ordinary, native]);
  }
});
test('unsupported mixed shadow materials fail before native draw updates', t => {
  const f = rig(t); f.renderer.shadowMap = {enabled: true};
  const material = f.material(); let updates = 0;
  material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; }); f.mesh(material);
  f.mesh(new THREE.ShaderMaterial()).visible = false;
  assert.throws(() => f.capture(), {code: 'THREE_NODE_PROGRAM_MATERIAL'}); assert.equal(updates, 0);
});
test('private native backgrounds precede scene draws independently of sorting, layers and overrides', t => {
  for (const sortObjects of [false, true]) {
    const f = rig(t), seen = [], foreground = f.mesh(), fixed = f.mesh(new THREE.MeshBasicMaterial());
    foreground.renderOrder = -100; f.renderer.sortObjects = sortObjects;
    const override = f.material();
    override.colorNode = N.uniform(.75).onObjectUpdate(frame => { seen.push(frame.object); return .75; });
    const material = f.material();
    const background = new THREE.Mesh(new THREE.PlaneGeometry(), material);
    background.visible = false; background.layers.set(4); background.renderOrder = 100;
    background.position.x = 100; // projection traversal would also cull it
    material.colorNode = N.uniform(.25).onObjectUpdate(frame => {
      assert.equal(frame.scene, f.scene); assert.equal(frame.camera, f.camera);
      assert.equal(frame.object, background); assert.equal(frame.material, material);
      assert.equal(frame.object.modelViewMatrix.elements[12], 100);
      assert.equal(frame.object.modelViewMatrix.elements[14], -5);
      seen.push(frame.object); return .25;
    });
    const children = f.scene.children.slice();
    const captured = f.capture({backgroundMesh: background, overrideMaterial: override});
    assert.equal(captured.background, true);
    assert.equal(captured.bindings[0].object, background);
    assert.equal(captured.bindings[0].material, material, 'scene override cannot replace the private background material');
    assert.deepEqual(captured.draws.map(draw => draw.object), [background, foreground, fixed]);
    assert.deepEqual(captured.draws.map(draw => draw.background), [true, false, false]);
    assert.deepEqual(seen, [background, foreground, fixed]);
    assert.deepEqual(f.scene.children, children); assert.equal(background.parent, null);
    assert.ok(children.every(child => child.parent === f.scene));
  }
});
test('a private background prepares and captures when all user scene materials use the fixed route', t => {
  const f = rig(t), fixed = f.mesh(new THREE.MeshBasicMaterial()), material = f.material(); let updates = 0;
  const background = new THREE.Mesh(new THREE.PlaneGeometry(), material);
  material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  f.scene.visible = false; background.layers.set(5);
  const children = f.scene.children.slice();
  const prepared = f.capture({backgroundMesh: background, uniforms: false});
  assert.equal(prepared.background, true); assert.equal(prepared.bindings.length, 1);
  assert.equal(prepared.bindings[0].object, background); assert.equal(prepared.draws.length, 0); assert.equal(updates, 0);
  const captured = f.capture({backgroundMesh: background});
  assert.equal(captured.background, true); assert.deepEqual([...captured.objects], [background]);
  assert.equal(captured.draws.length, 1); assert.equal(captured.draws[0].object, background);
  assert.equal(captured.draws[0].background, true); assert.equal(updates, 1);
  assert.deepEqual(f.scene.children, children); assert.equal(fixed.parent, f.scene); assert.equal(background.parent, null);
  const attached = new THREE.Group(); attached.add(background);
  assert.throws(() => f.capture({backgroundMesh: background}), {code: 'THREE_NODE_PROGRAM_BACKGROUND'});
});
test('native draws capture source identities, matrices, frame context and independent uniform histories', t => {
  const f = rig(t), seen = [], m = f.material(), tint = N.uniform(new THREE.Color());
  tint.onObjectUpdate(frame => { seen.push({object: frame.object, material: frame.material, camera: frame.camera,
    scene: frame.scene, renderer: frame.renderer, frameId: frame.frameId, renderId: frame.renderId,
    time: frame.time, deltaTime: frame.deltaTime}); return frame.object.userData.tint; });
  let after = 0; tint.updateAfterType = 'object'; tint.updateAfter = () => { after++; };
  m.colorNode = tint;
  const a = f.mesh(m), b = f.mesh(m); a.userData.tint = new THREE.Color(.125, .25, .375);
  b.userData.tint = new THREE.Color(.625, .75, .875); a.position.x = -.5; b.position.x = .5; b.scale.x = -1;
  const captured = f.capture();
  assert.equal(captured.bindings.length, 2); assert.equal(captured.draws.length, 2); assert.equal(after, 2);
  assert.equal(f.camera.coordinateSystem, THREE.WebGPUCoordinateSystem);
  assert.deepEqual([captured.draws[0].object, captured.draws[1].object], [a, b]);
  assert.ok(floats(captured.draws[0].nodeUniforms).includes(.125));
  assert.ok(floats(captured.draws[1].nodeUniforms).includes(.875));
  assert.equal(captured.draws[1].frontFaceCW, true); assert.equal(captured.draws[0].nodeRangeCaptured, true);
  assert.equal(a.modelViewMatrix.elements[14], -5); assert.equal(a.modelViewMatrix.elements[12], -.5);
  assert.equal(captured.bytes, captured.draws.reduce((sum, draw) => sum + draw.nodeUniforms.reduce((n, bytes) => n + bytes.byteLength, 0), 0));
  for (const [index, context] of seen.entries()) assert.deepEqual(context, {object: [a, b][index], material: m,
    camera: f.camera, scene: f.scene, renderer: f.renderer, frameId: 7, renderId: 3, time: 1.25, deltaTime: .125});
  const old = captured.draws.map(draw => draw.nodeUniforms.map(bytes => bytes.slice())), key = captured.bindings[0].nodeProgram;
  m.colorNode = N.vec3(1, 0, 0); m.transparent = true; a.renderOrder = 9; a.visible = false;
  const later = f.capture();
  assert.notEqual(later.bindings[0].nodeProgram, key); assert.equal(later.draws.length, 1);
  assert.equal(captured.draws[0].listMaterial.transparent, false); assert.equal(captured.draws[0].renderOrder, 0);
  assert.deepEqual(captured.draws.map(draw => draw.nodeUniforms), old); assert.equal(captured.draws.length, 2);
  assert.equal(f.frame.frameId, 7); assert.equal(f.renderer.info.calls, 3);
});
test('nested preparation preserves the outer render clock and source update frequency', t => {
  const f = rig(t), m = f.material(), seen = [];
  const observe = (phase, frame) => seen.push([phase, frame.frameId, frame.renderId, frame.time, frame.deltaTime]);
  const tint = N.uniform(.5).onRenderUpdate(frame => { observe('current', frame); return .5; });
  tint.updateBeforeType = 'render'; tint.updateBefore = frame => { observe('before', frame); };
  tint.updateAfterType = 'render'; tint.updateAfter = frame => { observe('after', frame); };
  m.colorNode = N.vec3(tint); f.mesh(m); f.mesh(m);
  let prepared = false;
  const programs = {...f.owner, compile(...args) {
    if (!prepared) {
      prepared = true;
      // A retained preparation helper may render before returning its program.
      // Its global call counters belong to those nested passes, not this draw.
      f.renderer.info.calls += 13;
      Object.assign(f.frame, {frameId: 8, time: 2, deltaTime: .25});
    }
    return f.owner.compile(...args);
  }};
  const captured = f.capture({getPrograms: () => programs});
  assert.equal(captured.draws.length, 2);
  assert.deepEqual(seen, ['before', 'current', 'after'].map(phase => [phase, 7, 3, 1.25, .125]));
  assert.equal(f.renderer.info.calls, 16, 'capturing the outer clock does not roll back nested counters');
  assert.deepEqual([f.frame.frameId, f.frame.time, f.frame.deltaTime], [8, 2, .25]);
});
test('compile-only capture prepares hidden bindings without node updates or frame advancement', t => {
  const f = rig(t), m = f.material(); let calls = 0;
  m.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(() => { calls++; return new THREE.Color(1, 0, 0); });
  const a = f.mesh(m), b = f.mesh(m); a.visible = false; b.visible = false;
  const result = f.capture({uniforms: false, maxUniformBytes: 0});
  assert.equal(result.objects.size, 2); assert.equal(result.bindings.length, 2);
  assert.equal(result.draws.length, 0); assert.equal(result.bytes, 0); assert.equal(calls, 0);
  assert.deepEqual([f.frame.frameId, f.frame.time, f.frame.deltaTime, f.renderer.info.calls], [7, 1.25, .125, 3]);
});
test('native Mesh.count is captured after updates, including empty draws, before after-updates', t => {
  const f = rig(t), m = f.material(), requested = [3, 0, -2, undefined], seen = [];
  const tint = N.uniform(.5).onObjectUpdate(frame => {
    frame.object.count = requested.shift(); seen.push('update'); return .5;
  });
  tint.updateAfterType = 'object'; tint.updateAfter = frame => { seen.push('after'); frame.object.count = 99; };
  m.colorNode = N.vec3(tint); m.positionNode = N.positionLocal.add(N.vec3(N.instanceIndex.toFloat().mul(.1), 0, 0));
  const object = f.mesh(m);
  const prepared = f.capture({uniforms: false}); assert.equal(seen.length, 0);
  const frames = Array.from({length: 4}, () => f.capture());
  assert.deepEqual(frames.map(frame => frame.draws[0].instanceCount), [3, 0, 0, 1]);
  assert.deepEqual(seen, ['update', 'after', 'update', 'after', 'update', 'after', 'update', 'after']);
  assert.equal(object.count, 99);
  assert.ok(frames.every(frame => frame.bindings[0].nodeProgram === prepared.bindings[0].nodeProgram));
  assert.match(prepared.bindings[0].nodeProgram.vertex, /@builtin\(\s*instance_index\s*\)/);
});

test('source selection keeps hidden bindings and follows visibility, layers, frustum and LOD', t => {
  const f = rig(t), hidden = new THREE.Group(); hidden.visible = false; f.scene.add(hidden);
  const hiddenChild = f.mesh(undefined, hidden), offscreen = f.mesh(); offscreen.position.x = 100;
  const wrongLayer = f.mesh(); wrongLayer.layers.set(2);
  const differentLayerParent = new THREE.Group(); differentLayerParent.layers.set(2); f.scene.add(differentLayerParent);
  const visibleChild = f.mesh(undefined, differentLayerParent);
  const lod = new THREE.LOD(); f.scene.add(lod);
  const near = f.mesh(undefined, lod), far = f.mesh(undefined, lod); lod.addLevel(near, 0); lod.addLevel(far, 4);
  const result = f.capture();
  assert.equal(result.bindings.length, 6); assert.ok(result.objects.has(hiddenChild));
  assert.deepEqual(new Set(result.draws.map(draw => draw.object)), new Set([visibleChild, far]));
  assert.equal(near.visible, false); assert.equal(far.visible, true);
});
test('native opaque sorting uses depth before object id, with source group and transparency order', t => {
  const f = rig(t), earlyMaterial = f.material(), lateMaterial = f.material();
  const farOpaque = f.mesh(earlyMaterial), nearOpaque = f.mesh(lateMaterial); nearOpaque.position.z = 1;
  const tm = f.material(); tm.transparent = true;
  const nearTransparent = f.mesh(tm), farTransparent = f.mesh(tm); nearTransparent.position.z = 1; farTransparent.position.z = -1;
  const group = new THREE.Group(); group.renderOrder = -2; f.scene.add(group); const ordered = f.mesh(undefined, group);
  assert.deepEqual(f.capture().draws.map(draw => draw.object), [ordered, nearOpaque, farOpaque, farTransparent, nearTransparent]);
  f.renderer.sortObjects = false;
  assert.deepEqual(f.capture().draws.map(draw => draw.object), [farOpaque, nearOpaque, ordered, nearTransparent, farTransparent]);
});
test('material groups capture their intersections and update once per logical draw', t => {
  const f = rig(t), a = f.material(), b = f.material(); let updates = 0;
  const value = N.uniform(.25).onObjectUpdate(() => { updates++; return .25; }); a.colorNode = N.vec3(value); b.colorNode = N.vec3(value);
  const object = f.mesh([a, b]), geometry = object.geometry; geometry.clearGroups();
  geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 1); geometry.addGroup(1, 3, 0); geometry.setDrawRange(1, 4);
  const result = f.capture();
  assert.equal(result.bindings.length, 2); assert.equal(updates, 3);
  assert.deepEqual(result.draws.map(draw => [draw.first, draw.count]), [[1, 2], [3, 2], [1, 3]]);
  geometry.groups[0].start = 5; geometry.setDrawRange(4, 1);
  assert.deepEqual(result.draws.map(draw => [draw.first, draw.count]), [[1, 2], [3, 2], [1, 3]]);
  assert.equal(result.draws[0].group.start, 0); assert.equal(result.draws[0].binding, result.draws[2].binding);
});
test('hidden invalid graphs, mixed arrays and effectful upload/object hooks refuse before updates', t => {
  const cases = [
    f => { const m = f.material(); m.colorNode = N.texture(new THREE.Texture()); f.mesh(m).visible = false; },
    f => { f.mesh([f.material(), new THREE.MeshBasicMaterial()]); },
    f => { f.mesh().geometry.attributes.position.onUpload(() => {}); },
    f => { f.mesh().geometry.index.onUpload(() => {}); },
    f => { const o = f.mesh(); const data = new THREE.InterleavedBuffer(new Float32Array(12), 3); data.onUpload(() => {});
      o.geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(data, 3, 0)); },
    f => { f.mesh().onBeforeRender = () => {}; },
    f => { f.mesh().intersectsFrustum = () => true; },
  ];
  for (const bad of cases) {
    const f = rig(t), m = f.material(); let updates = 0;
    m.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; }); f.mesh(m); bad(f);
    assert.throws(() => f.capture(), error => error.code?.startsWith('THREE_NODE_PROGRAM_'));
    assert.equal(updates, 0);
  }
});
test('all compile and raster admission completes before callbacks, and budgets preflight without partial updates', t => {
  const f = rig(t), m = f.material(); let updates = 0;
  m.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; }); f.mesh(m); f.mesh(m);
  const compiled = f.capture({uniforms: false, maxUniformBytes: 0});
  const bytes = compiled.bindings.reduce((sum, binding) => sum + binding.nodeProgram.groups.reduce((n, group) => n + group.byteLength, 0), 0);
  assert.throws(() => f.capture({maxUniformBytes: bytes - 1}), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  assert.equal(updates, 0);
  let rasterCalls = 0;
  assert.throws(() => f.capture({raster() { if (++rasterCalls === 2) throw Error('second raster refuses'); return {}; }}), /second raster refuses/);
  assert.equal(updates, 0);
  assert.throws(() => f.capture({maxDraws: 1}), {code: 'THREE_NODE_PROGRAM_LIMIT'}); assert.equal(updates, 0);
  assert.equal(f.capture({maxUniformBytes: bytes}).bytes, bytes); assert.equal(updates, 2);
});
test('equivalent overrides preserve source allowOverride, original visibility and callback material identities', t => {
  const f = rig(t), original = f.material(), override = f.material(), excluded = f.material(); excluded.allowOverride = false;
  const a = f.mesh(original), b = f.mesh(excluded); override.visible = false;
  let selected = null; override.colorNode = N.uniform(.25).onObjectUpdate(frame => { selected = frame.material; return .25; });
  const captured = f.capture({overrideMaterial: override});
  assert.deepEqual(captured.draws.map(draw => draw.material), [override, excluded]); assert.equal(selected, override);
  original.visible = false;
  assert.deepEqual(f.capture({overrideMaterial: override}).draws.map(draw => draw.object), [b]);
  original.visible = true; original.positionNode = N.positionLocal.add(N.vec3(1, 0, 0));
  assert.throws(() => f.capture({overrideMaterial: override}), {code: 'THREE_NODE_PROGRAM_EFFECT'});
  assert.equal(override.positionNode, null); assert.equal(a.material, original);
});
test('capture freezes the render context across callbacks that mutate renderer counters', t => {
  const f = rig(t), seen = [], m = f.material();
  m.colorNode = N.uniform(.25).onObjectUpdate(frame => { seen.push([frame.frameId, frame.renderId, frame.time]);
    f.renderer.info.calls = 91; return .25; }); f.mesh(m); f.mesh(m);
  f.capture(); assert.deepEqual(seen, [[7, 3, 1.25], [7, 3, 1.25]]);
});
test('capture bounds cyclic hierarchies and hidden node growth before matrix recursion', t => {
  const f = rig(t), object = f.mesh();
  assert.throws(() => f.capture({maxNodes: 1}), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  object.children.push(f.scene);
  assert.throws(() => f.capture(), {code: 'THREE_NODE_PROGRAM_GRAPH'});
});
test('storage-only compile capture needs no GPU publication and runs no node callbacks', t => {
  const f = rig(t), device = geometryDevice(); f.renderer._device = device;
  const attribute = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(attribute, 'vec4', 4);
  let updates = 0, transactions = 0;
  const gain = N.uniform(1).onObjectUpdate(() => { updates++; return 1; });
  const material = f.material(); material.fragmentNode = storage.element(0).mul(gain); f.mesh(material);
  const guard = {captureDispatch(run) { transactions++; return run(); }, validate() {}};
  const result = f.capture({uniforms: false, getStorage: () => guard});
  assert.equal(result.bindings.length, 1); assert.equal(result.draws.length, 0);
  assert.equal(f.owner.storage(result.bindings[0].nodeProgram)[0].attribute, attribute);
  assert.equal(updates, 0); assert.equal(device.buffers.length, 0); assert.equal(device.writes.length, 0);
  assert.equal(result.storageOwner, guard, 'storage-only graphs join the shared capture transaction');
  assert.equal(transactions, 1);
});
test('storage captures lease post-update identities before later draws and after-updates can replace them', t => {
  const f = rig(t), device = geometryDevice(), gpu = createGpuCompute(device); f.renderer._device = device;
  t.after(() => gpu.dispose());
  const a = new THREE.StorageBufferAttribute(4, 4), b = new THREE.StorageBufferAttribute(4, 4),
    after = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(a, 'vec4', 4);
  a.array.fill(.125); b.array.fill(.875);
  const aBuffer = gpu.upload(a, a.array, a.version), bBuffer = gpu.upload(b, b.array, b.version);
  const material = f.material(), first = f.mesh(material), second = f.mesh(material);
  let inside = false, transactions = 0, updates = 0;
  const guard = {captureDispatch(run) {
    assert.equal(inside, false); transactions++; inside = true;
    try { return run(); } finally { inside = false; }
  }, capture: attribute => captureComputeStorage(device, attribute), validate() {}, release() {}};
  const gain = N.uniform(1).onObjectUpdate(frame => {
    assert.equal(inside, true, 'storage callbacks run inside the shared capture transaction');
    updates++; storage.value = frame.object === first ? a : b; return 1;
  });
  gain.updateAfterType = 'object'; gain.updateAfter = () => { storage.value = after; };
  material.fragmentNode = storage.element(0).mul(gain);
  const writes = device.writes.length, result = f.capture({getStorage: () => guard});
  assert.equal(updates, 2); assert.equal(transactions, 2, 'each draw captures after its source before-updates'); assert.equal(storage.value, after);
  assert.deepEqual(result.draws.map(draw => draw.object), [first, second]);
  assert.deepEqual(result.draws.map(draw => draw.nodeStorage[0].lease.resolve().buffer), [aBuffer, bBuffer]);
  assert.ok(result.draws.every(draw => Object.isFrozen(draw.nodeStorage) && draw.nodeStorage.every(Object.isFrozen)));
  assert.equal(result.textureUses.length, 0); assert.equal(device.writes.length, writes, 'draw capture never uploads storage');
  a.array.fill(99); b.array.fill(99);
  assert.deepEqual(result.draws.map(draw => Array.from(new Float32Array(draw.nodeStorage[0].lease.resolve().buffer.data)).slice(0, 4)),
    [[.125, .125, .125, .125], [.875, .875, .875, .875]]);
});

test('per-draw capture releases earlier leases when a later source update fails', t => {
  const f = rig(t), material = f.material(), first = f.mesh(material), second = f.mesh(material);
  const storage = N.storage(new THREE.StorageBufferAttribute(4, 4), 'vec4', 4);
  const leases = [], released = [];
  let inside = false;
  const guard = {captureDispatch(run) {
    assert.equal(inside, false); inside = true;
    try { return run(); } finally { inside = false; }
  }, capture(attribute) { const lease = {attribute}; leases.push(lease); return lease; },
  release(lease) { released.push(lease); }, validate() { assert.fail('failed capture cannot validate'); }};
  const gain = N.uniform(1).onObjectUpdate(() => 1);
  gain.updateAfterType = 'object'; gain.updateAfter = frame => {
    assert.equal(inside, true);
    if (frame.object === second) throw new Error('late shadow draw failure');
  };
  material.fragmentNode = storage.element(0).mul(gain);
  assert.throws(() => f.capture({getStorage: () => guard}), /late shadow draw failure/);
  assert.equal(leases.length, 2); assert.deepEqual(released, leases); assert.equal(inside, false);
  assert.equal(f.frame.object, null); assert.equal(first.material, material);
});

test('source before-updates can capture child draws outside the parent resource transaction', t => {
  const f = rig(t), material = f.material(), storage = N.storage(new THREE.StorageBufferAttribute(4, 4), 'vec4', 4);
  material.fragmentNode = storage.element(0); f.mesh(material);
  let inside = false, nested = false, transactions = 0;
  const guard = {captureDispatch(run) {
    assert.equal(inside, false); inside = true; transactions++;
    try { return run(); } finally { inside = false; }
  }, capture(attribute) { return {attribute}; }, release() {}, validate() {}};
  const programs = {...f.owner, before(descriptor, context) {
    assert.equal(inside, false);
    if (!nested) { nested = true; f.capture({getStorage: () => guard}); }
    f.owner.before?.(descriptor, context);
  }};
  const result = f.capture({getPrograms: () => programs, getStorage: () => guard});
  assert.equal(result.draws.length, 1); assert.equal(nested, true); assert.equal(transactions, 2);
  assert.equal(inside, false);
});

test('nested writes reject an earlier parent depth read, and release the guard after failure', () => {
  const renderer = {}, target = new THREE.RenderTarget(8, 8);
  target.depthTexture = new THREE.DepthTexture(8, 8);
  const unrelated = new THREE.RenderTarget(8, 8);
  assert.throws(() => withThreeNativeCapture(renderer, record => {
    assert.doesNotThrow(() => assertThreeNativeTargetWrite(renderer, target));
    record(target.depthTexture);
    withThreeNativeCapture(renderer, recordChild => {
      recordChild(unrelated.texture);
      assert.throws(() => assertThreeNativeTargetWrite(renderer, target), {code: 'THREE_NODE_PROGRAM_SHADOW'});
      assert.throws(() => assertThreeNativeTargetWrite(renderer, unrelated), {code: 'THREE_NODE_PROGRAM_SHADOW'});
      throw new Error('nested capture failed');
    });
  }), /nested capture failed/);
  assert.doesNotThrow(() => assertThreeNativeTargetWrite(renderer, target));
  assert.doesNotThrow(() => assertThreeNativeTargetWrite(renderer, unrelated));
});

test('compute reentry stays refused until the complete native capture unwinds', () => {
  const renderer = {};
  assert.doesNotThrow(() => assertThreeNativeComputeOutsideCapture(renderer));
  withThreeNativeCapture(renderer, () => {
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer), {code: 'THREE_COMPUTE_REENTRY'});
    withThreeNativeCapture(renderer, () =>
      assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer), {code: 'THREE_COMPUTE_REENTRY'}));
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer), {code: 'THREE_COMPUTE_REENTRY'});
  });
  assert.doesNotThrow(() => assertThreeNativeComputeOutsideCapture(renderer));
});
