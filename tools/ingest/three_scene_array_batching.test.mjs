// Actual r186 BatchedMesh callbacks and program shaders with byte-accurate
// submitted texture histories. Rasterization is checked by the browser case.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function recordingDevice() {
  const device = geometryDevice(); device.textures = []; device.events = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.features = new Set(['float32-filterable']);
  const texelBytes = format => format === 'rgba32float' ? 16 : 4;
  device.createTexture = descriptor => {
    const width = descriptor.size.width ?? descriptor.size[0], height = descriptor.size.height ?? descriptor.size[1];
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor, width, height,
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      pixels: new Uint8Array(width * height * texelBytes(descriptor.format)),
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.queue.writeTexture = ({texture}, data) => {
    texture.pixels.set(new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength));
  };
  const create = device.createCommandEncoder;
  device.createCommandEncoder = descriptor => {
    const base = create(), commands = [];
    return {...base, beginRenderPass(descriptor) {
      const pass = base.beginRenderPass(descriptor), record = device.passes.at(-1); commands.push(record);
      return {...pass, setViewport(...value) { record.viewport = value; pass.setViewport(...value); }};
    }, copyTextureToBuffer(source, destination, size) {
      commands.push({kind: 'capture', source, destination, size, draws: []});
    }, copyBufferToTexture(source, destination, size) {
      commands.push({kind: 'restore', source, destination, size, draws: []});
    }, finish() { return commands; }};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const list of commands) for (const command of list) {
      if (command.kind) {
        const capture = command.kind === 'capture', image = capture ? command.source : command.destination;
        const memory = capture ? command.destination : command.source;
        assert.ok(image.texture.usage & (capture ? 1 : 2));
        assert.ok(memory.buffer.usage & (capture ? 8 : 4));
        const [width, height] = command.size, row = width * texelBytes(image.texture.format);
        const bytes = new Uint8Array(memory.buffer.data);
        for (let y = 0; y < height; y++) {
          if (capture) bytes.set(image.texture.pixels.subarray(y * row, (y + 1) * row), y * memory.bytesPerRow);
          else image.texture.pixels.set(bytes.subarray(y * memory.bytesPerRow, y * memory.bytesPerRow + row), y * row);
        }
      }
      device.events.push(command);
    }
    submit(commands);
    for (const draw of device.snapshots.at(-1)) {
      draw.textureContents = new Map();
      for (const {group} of draw.groups.values()) for (const entry of group.entries ?? []) {
        const texture = entry.resource?.texture;
        if (texture?.pixels) draw.textureContents.set(texture, texture.pixels.slice());
      }
    }
  };
  return device;
}

async function fixture(t, views = 2, classOptions = {}) {
  const device = recordingDevice(); let configuration;
  const canvas = {width: 64, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 32], format: configuration.format, usage: configuration.usage}); }
  })};
  const Renderer = createWebGLRendererClass(T, classOptions), renderer = new Renderer({canvas, device});
  const material = new T.MeshBasicMaterial({side: T.DoubleSide}), geometry = new T.PlaneGeometry(.5, .5);
  const batch = new T.BatchedMesh(8, 32, 48, material), scene = new T.Scene(); scene.add(batch);
  batch.frustumCulled = batch.perObjectFrustumCulled = false;
  const geometryId = batch.addGeometry(geometry), ids = [batch.addInstance(geometryId), batch.addInstance(geometryId)];
  ids.forEach((id, i) => {
    batch.setMatrixAt(id, new T.Matrix4().makeTranslation(i ? .6 : -.6, 0, 0));
    batch.setColorAt(id, new T.Color(i ? 0x00ff00 : 0xff0000));
  });
  batch.setCustomSort((list, camera) => list.sort((a, b) => Number(camera.name) % 2 ? b.index - a.index : a.index - b.index));
  const children = Array.from({length: views}, (_, i) => {
    const camera = new T.OrthographicCamera(-2, 2, 1, -1, .1, 10);
    camera.name = String(i); camera.position.z = 4; camera.updateMatrixWorld();
    camera.viewport = new T.Vector4(i * 16, 0, 32, 32); return camera;
  });
  const camera = new T.ArrayCamera(children); camera.position.z = 4;
  await renderer.init(); await renderer.compileAsync(scene, camera);
  t.after(() => { renderer.dispose(); batch.dispose(); material.dispose(); geometry.dispose(); });
  return {device, renderer, scene, camera, batch, ids};
}
const submitted = device => device.snapshots.flat().filter(draw => [...draw.textureContents.keys()].some(texture => texture.format === 'r32uint'));
function words(draw, source, format, Type) {
  const [, bytes] = [...draw.textureContents].find(([texture]) => texture.format === format && texture.width === source.image.width);
  return Array.from(new Type(bytes.buffer, bytes.byteOffset, bytes.byteLength / Type.BYTES_PER_ELEMENT));
}
const historyBuffers = device => device.buffers.filter(buffer => buffer.label === 'f3d-program-batch-history');

test('each camera consumes its requested batching IDs, matrices and colors before later versions overwrite them', async t => {
  const f = await fixture(t), expected = [];
  f.batch.onBeforeRender = function(renderer, scene, camera, geometry, material) {
    const view = Number(camera.name);
    this.setMatrixAt(f.ids[0], new T.Matrix4().makeTranslation(view ? .25 : -.75, 0, 0));
    this.setColorAt(f.ids[0], new T.Color(view ? 0x0000ff : 0xff0000));
    T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, camera, geometry, material);
    expected.push({ids: Array.from(this._indirectTexture.image.data),
      matrices: Array.from(this._matricesTexture.image.data), colors: Array.from(this._colorsTexture.image.data)});
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = submitted(f.device); assert.equal(draws.length, 4); assert.equal(expected.length, 2);
  assert.notDeepEqual(expected[0].ids, expected[1].ids);
  for (let i = 0; i < draws.length; i++) {
    const frame = expected[Math.floor(i / 2)];
    assert.deepEqual(words(draws[i], f.batch._indirectTexture, 'r32uint', Uint32Array), frame.ids);
    assert.deepEqual(words(draws[i], f.batch._matricesTexture, 'rgba32float', Float32Array), frame.matrices);
    assert.deepEqual(words(draws[i], f.batch._colorsTexture, 'rgba32float', Float32Array), frame.colors);
  }
  const buffers = historyBuffers(f.device).length;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(historyBuffers(f.device).length, buffers, 'steady frames reuse their GPU history storage');
  f.renderer.dispose(); assert.ok(historyBuffers(f.device).every(buffer => buffer.destroyed));
});

test('GPU history capture does not upload versionless CPU matrix or color edits', async t => {
  const f = await fixture(t); let expectedMatrix, expectedColor;
  f.batch.onBeforeRender = function(renderer, scene, camera, geometry, material) {
    if (camera.name === '0') {
      this.setMatrixAt(f.ids[0], new T.Matrix4().makeTranslation(-.25, 0, 0));
      this.setColorAt(f.ids[0], new T.Color(0xff0000));
      expectedMatrix = Array.from(this._matricesTexture.image.data); expectedColor = Array.from(this._colorsTexture.image.data);
    } else {
      this._matricesTexture.image.data[12] = .875;
      this._colorsTexture.image.data.set([0, 0, 1, 1], 0);
    }
    T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, camera, geometry, material);
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = submitted(f.device); assert.equal(draws.length, 4);
  for (const draw of draws) {
    assert.deepEqual(words(draw, f.batch._matricesTexture, 'rgba32float', Float32Array), expectedMatrix);
    assert.deepEqual(words(draw, f.batch._colorsTexture, 'rgba32float', Float32Array), expectedColor);
  }
  assert.equal(f.batch._matricesTexture.image.data[12], .875, 'public CPU edits remain observable');
  assert.equal(f.batch._colorsTexture.image.data[2], 1);
});

test('a final view with no visible instances retains its last uploaded shared texture contents', async t => {
  const f = await fixture(t);
  f.batch.onBeforeRender = function(renderer, scene, camera, geometry, material) {
    if (camera.name === '1') {
      this.setColorAt(f.ids[0], new T.Color(0x0000ff));
      for (const id of f.ids) this.setVisibleAt(id, false);
    }
    T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, camera, geometry, material);
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = submitted(f.device); assert.equal(draws.length, 2);
  assert.deepEqual(words(draws[0], f.batch._colorsTexture, 'rgba32float', Float32Array).slice(0, 4), [1, 0, 0, 1]);
  const texture = [...draws[0].textureContents.keys()].find(texture => texture.format === 'rgba32float' && texture.width === f.batch._colorsTexture.image.width);
  assert.deepEqual(Array.from(new Float32Array(texture.pixels.buffer)).slice(0, 4), [0, 0, 1, 1]);
});

test('growing source textures cannot exceed the shared budget through cached batch histories', async t => {
  const budget = 8192, f = await fixture(t, 2, {scene: {texture: {maxTextureBytes: budget}}});
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const old = historyBuffers(f.device); assert.ok(old.length > 0);
  const residentBytes = () => f.device.textures.filter(texture => !texture.destroyed && texture.label === 'f3d-three-textures')
    .reduce((sum, texture) => sum + texture.pixels.byteLength, 0) + historyBuffers(f.device)
    .filter(buffer => !buffer.destroyed).reduce((sum, buffer) => sum + buffer.size, 0);
  const allocationBytes = [], create = f.device.createTexture;
  f.device.createTexture = descriptor => {
    const result = create(descriptor); allocationBytes.push(residentBytes()); return result;
  };
  const texture = new T.DataTexture(new Uint8Array(40 * 40 * 4), 40, 40); texture.needsUpdate = true;
  const material = new T.MeshBasicMaterial({map: texture}), geometry = new T.PlaneGeometry(.5, .5);
  t.after(() => { material.dispose(); geometry.dispose(); texture.dispose(); });
  f.scene.remove(f.batch); f.scene.add(new T.Mesh(geometry, material));
  await f.renderer.compileAsync(f.scene, f.camera);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.ok(allocationBytes.length > 0);
  assert.ok(allocationBytes.every(bytes => bytes <= budget), `combined residency: ${allocationBytes.join(', ')}`);
  assert.ok(residentBytes() <= budget);
  assert.ok(old.every(buffer => buffer.destroyed), 'old history storage retires before the larger texture allocation');
});
