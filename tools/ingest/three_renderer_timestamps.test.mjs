/** Real query descriptors and mapped nanoseconds, never CPU clock estimates. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t, {supported = true, enabled = true} = {}) {
  const device = geometryDevice(), querySets = [];
  device.features = new Set(supported ? ['timestamp-query'] : []);
  device.limits.maxTextureDimension2D = 8192;
  device.createTexture = descriptor => ({...descriptor, width: descriptor.size[0], height: descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() {this.destroyed = true;}});
  device.createSampler = value => value;
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout() {return {};}});
  device.createComputePipeline = device.createRenderPipeline;
  device.createComputePipelineAsync = async descriptor => device.createComputePipeline(descriptor);
  device.computePasses = [];
  device.createQuerySet = descriptor => {
    const query = {...descriptor, values: [0n, 0n], destroy() {this.destroyed = true;}};
    querySets.push(query); return query;
  };
  const allocate = device.createBuffer;
  device.createBuffer = descriptor => {
    const buffer = allocate(descriptor);
    if (descriptor.usage & 1) buffer.mapAsync = async () => {};
    return buffer;
  };
  let clock = 1000000n;
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = encode();
    return {...encoder, beginRenderPass(descriptor) {
      const pass = encoder.beginRenderPass(descriptor), timestamps = descriptor.timestampWrites;
      if (timestamps?.beginningOfPassWriteIndex !== undefined)
        timestamps.querySet.values[timestamps.beginningOfPassWriteIndex] = clock;
      clock += 250000n;
      return {...pass, end() {
        pass.end();
        if (timestamps?.endOfPassWriteIndex !== undefined) timestamps.querySet.values[timestamps.endOfPassWriteIndex] = clock;
      }};
    }, beginComputePass(descriptor) {
      device.computePasses.push(descriptor);
      const timestamps = descriptor.timestampWrites;
      if (timestamps?.beginningOfPassWriteIndex !== undefined)
        timestamps.querySet.values[timestamps.beginningOfPassWriteIndex] = clock;
      clock += 250000n;
      return {setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {},
        end() { if (timestamps?.endOfPassWriteIndex !== undefined) timestamps.querySet.values[timestamps.endOfPassWriteIndex] = clock; }};
    }, resolveQuerySet(query, first, count, buffer, offset) {
      new BigUint64Array(buffer.data, offset, count).set(query.values.slice(first, first + count));
    }, copyBufferToBuffer(source, from, destination, to, size) {
      new Uint8Array(destination.data, to, size).set(new Uint8Array(source.data, from, size));
    }};
  };
  const makeCanvas = () => {
    let configured;
    return {width: 32, height: 32, style: {}, getContext: () => ({configure(value) { configured = value; },
      unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [32, 32], format: configured.format, usage: 16})})};
  };
  const canvas = makeCanvas();
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device, trackTimestamp: enabled});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicNodeMaterial();
  scene.add(new T.Mesh(geometry, material));
  t.after(() => {renderer.dispose(); geometry.dispose(); material.dispose();});
  return {renderer, scene, camera, device, querySets, makeCanvas};
}

test('native renderer exposes asynchronous GPU timestamp resolution and honors the source constructor option', t => {
  const f = fixture(t);
  assert.equal(typeof f.renderer.resolveTimestampsAsync, 'function');
  assert.equal(f.renderer.backend.trackTimestamp, true);
  assert.equal(f.querySets.length, 0);
});

test('cold queued renders retain their captured frame and publish only mapped GPU duration', async t => {
  const f = fixture(t);
  const uids = [];
  f.renderer.inspector.beginRender = uid => uids.push(uid);
  f.renderer.info.frame = 3;
  const first = f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.info.frame = 4;
  const second = f.renderer.renderAsync(f.scene, f.camera);
  const resolving = f.renderer.resolveTimestampsAsync();
  assert.ok(resolving instanceof Promise);
  assert.equal(f.renderer.info.render.timestamp, 0);
  assert.equal(await resolving, .25);
  await Promise.all([first, second]);
  assert.deepEqual(f.renderer.backend.getTimestampFrames('render'), [3, 4]);
  for (const uid of uids) {
    assert.equal(f.renderer.backend.hasTimestampQuery(uid), true);
    assert.equal(f.renderer.backend.getTimestamp(uid), .25);
  }
  assert.equal(f.renderer.info.render.timestamp, .25);
  assert.equal(f.querySets.length, 2);
  assert.ok(f.querySets.every(query => query.destroyed));
  assert.equal(await f.renderer.resolveTimestampsAsync('compute'), undefined);
});

test('warm render timing covers scissor-clear and geometry passes with one query pair', async t => {
  const f = fixture(t);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.querySets.length, 0, 'compilation does not create a timing sample');
  f.renderer.setScissor(1, 2, 4, 5); f.renderer.setScissorTest(true);
  f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const writes = f.device.passes.filter(pass => pass.desc.timestampWrites).map(pass => pass.desc.timestampWrites);
  assert.equal(writes.length, 2);
  assert.equal(writes[0].querySet, writes[1].querySet);
  assert.equal(writes[0].beginningOfPassWriteIndex, 0);
  assert.equal(writes[1].beginningOfPassWriteIndex, undefined);
  assert.equal(await f.renderer.backend.resolveTimestampsAsync('render'), .5);
});

test('unsupported devices and disabled tracking submit the same draws without query allocation', async t => {
  for (const options of [{supported: false}, {enabled: false}]) {
    const f = fixture(t, options);
    await f.renderer.init();
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    assert.equal(await f.renderer.resolveTimestampsAsync(), undefined);
    assert.equal(f.renderer.info.render.timestamp, 0);
    assert.equal(f.querySets.length, 0);
    assert.ok(f.device.passes.some(pass => pass.draws.length));
    assert.equal(f.renderer.backend.hasTimestamp, options.supported !== false);
    if (options.enabled === false) {
      f.renderer.backend.trackTimestamp = true;
      f.renderer.render(f.scene, f.camera);
      assert.equal(await f.renderer.resolveTimestampsAsync(), .25);
    }
  }
});

test('a queued compute group measures its actual kernels under the captured Inspector UID', async t => {
  const f = fixture(t), uids = [], kernel = T.TSL.Fn(() => {})().compute(1);
  await f.renderer.init();
  f.renderer.inspector.beginCompute = uid => uids.push(uid);
  let release;
  f.renderer._preparing = new Promise(resolve => { release = resolve; });
  f.renderer.info.frame = 7;
  const issued = f.renderer.compute([kernel, kernel]);
  f.renderer.info.frame = 8;
  const resolving = f.renderer.resolveTimestampsAsync('compute');
  release(); await issued;
  assert.equal(await resolving, .5);
  assert.deepEqual(f.renderer.backend.getTimestampFrames('compute'), [7]);
  assert.equal(f.renderer.backend.getTimestamp(uids[0]), .5);
  assert.equal(f.renderer.info.compute.timestamp, .5);
  assert.equal(f.device.computePasses.length, 2);
  assert.equal(f.querySets.length, 1);
  assert.equal(f.device.computePasses[0].timestampWrites.querySet, f.device.computePasses[1].timestampWrites.querySet);
});

test('queued CanvasTarget switches preserve separate call timestamps in one logical frame', async t => {
  const f = fixture(t), uids = [];
  await f.renderer.init();
  f.renderer.inspector.beginRender = uid => uids.push(uid);
  const main = f.renderer.getCanvasTarget(), preview = new T.CanvasTarget(f.makeCanvas());
  f.renderer.info.frame = 9;
  f.renderer.render(f.scene, f.camera);
  f.renderer.setCanvasTarget(preview); f.renderer.render(f.scene, f.camera);
  f.renderer.setCanvasTarget(main); f.renderer.render(f.scene, f.camera);
  assert.equal(await f.renderer.resolveTimestampsAsync(), .75);
  assert.deepEqual(f.renderer.backend.getTimestampFrames('render'), [9]);
  assert.equal(new Set(uids).size, 3);
  assert.ok(uids.every(uid => f.renderer.backend.getTimestamp(uid) === .25));
  assert.equal(f.renderer.getCanvasTarget(), main);
});

test('disposal interrupts an outstanding timestamp map and destroys owned queries and buffers', async t => {
  const f = fixture(t);
  await f.renderer.compileAsync(f.scene, f.camera);
  const allocate = f.device.createBuffer;
  let issued;
  const mapping = new Promise(resolve => { issued = resolve; });
  f.device.createBuffer = descriptor => {
    const buffer = allocate(descriptor);
    if (descriptor.usage & 1) buffer.mapAsync = () => { issued(); return new Promise(() => {}); };
    return buffer;
  };
  f.renderer.render(f.scene, f.camera);
  const pending = f.renderer.resolveTimestampsAsync(); pending.catch(() => {});
  await mapping; f.renderer.dispose();
  await assert.rejects(pending, {code: 'THREE_TIMESTAMP_DISPOSED'});
  assert.ok(f.querySets.every(query => query.destroyed));
  assert.ok(f.device.buffers.filter(buffer => buffer.label?.startsWith('f3d-timestamp')).every(buffer => buffer.destroyed));
});
