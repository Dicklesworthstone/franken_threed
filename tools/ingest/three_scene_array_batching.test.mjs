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
      for (const draw of command.draws) {
        draw.pass = command.desc;
        draw.textureContents = new Map();
        for (const {group} of draw.groups.values()) for (const entry of group.entries ?? []) {
          const texture = entry.resource?.texture;
          if (texture?.pixels) draw.textureContents.set(texture, texture.pixels.slice());
        }
      }
      device.events.push(command);
    }
    submit(commands);
  };
  return device;
}

async function fixture(t, views = 2, classOptions = {}, {configure = null, indexed = true, glassBatch = false} = {}) {
  const device = recordingDevice(); let configuration;
  const canvas = {width: 64, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 32], format: configuration.format, usage: configuration.usage}); }
  })};
  const Renderer = createWebGLRendererClass(T, classOptions), renderer = new Renderer({canvas, device});
  const material = glassBatch ? new T.MeshPhysicalMaterial({transmission: .75, thickness: .25, side: T.DoubleSide})
    : new T.MeshBasicMaterial({side: T.DoubleSide});
  let geometry = new T.PlaneGeometry(.5, .5);
  if (!indexed) { const original = geometry; geometry = original.toNonIndexed(); original.dispose(); }
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
  configure?.({device, renderer, scene, camera, batch, ids});
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

for (const indexed of [true, false]) test(`transmission consumes ${indexed ? 'indexed' : 'nonindexed'} opaque batches with per-pass camera histories`, async t => {
  const f = await fixture(t, 2, {}, {indexed, configure({scene, renderer}) {
    renderer.transmissionResolutionScale = .5;
    const geometry = new T.PlaneGeometry(1, 1), material = new T.MeshPhysicalMaterial({transmission: .75, thickness: .25});
    const glass = new T.Mesh(geometry, material); glass.position.z = .25; scene.add(glass);
    t.after(() => { geometry.dispose(); material.dispose(); });
  }}), expected = [];
  f.batch.geometry.setDrawRange(0, 0);
  f.batch.onBeforeRender = function(renderer, scene, camera, geometry, material) {
    const transmission = renderer.getRenderTarget() !== null, view = Number(camera.name);
    this.setMatrixAt(f.ids[0], new T.Matrix4().makeTranslation(transmission ? -.5 + view * .2 : .25 + view * .1, 0, 0));
    this.setColorAt(f.ids[0], new T.Color(transmission ? view ? 0x00ff00 : 0xff0000 : view ? 0xffff00 : 0x0000ff));
    T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, camera, geometry, material);
    expected.push({transmission, ids: Array.from(this._indirectTexture.image.data),
      matrices: Array.from(this._matricesTexture.image.data), colors: Array.from(this._colorsTexture.image.data)});
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = submitted(f.device);
  assert.equal(expected.length, 4, 'one callback per logical object in each camera/pass');
  assert.equal(draws.length, 8, 'two physical subdraws for each accepted batch');
  assert.deepEqual(expected.map(value => value.transmission), [true, true, false, false]);
  for (let i = 0; i < draws.length; i++) {
    const draw = draws[i], frame = expected[Math.floor(i / 2)];
    assert.equal(draw.indexed, indexed);
    assert.equal(draw.args[0], 6, 'batch ranges are not clipped by ordinary geometry.drawRange');
    assert.deepEqual(words(draw, f.batch._indirectTexture, 'r32uint', Uint32Array), frame.ids);
    assert.deepEqual(words(draw, f.batch._matricesTexture, 'rgba32float', Float32Array), frame.matrices);
    assert.deepEqual(words(draw, f.batch._colorsTexture, 'rgba32float', Float32Array), frame.colors);
    const compiled = f.renderer._programSupport.compile(f.batch.material, f.batch, {renderTarget: frame.transmission});
    const offset = compiled.program.reflection.uniforms.find(value => value.name === '_gl_DrawID').offset;
    const group = draw.groups.get(0), buffer = group.group.entries[0].resource.buffer, bytes = draw.contents.get(buffer);
    assert.equal(new DataView(bytes.buffer, bytes.byteOffset).getInt32((group.offsets[0] ?? 0) + offset, true), i % 2);
  }
});

test('transmissive batches retain back-face selections and leave unrequested matrix edits GPU-stale', async t => {
  const f = await fixture(t, 1, {}, {glassBatch: true, configure({scene}) {
    const geometry = new T.PlaneGeometry(2, 1), material = new T.MeshBasicMaterial({color: 0x2468ac});
    const solid = new T.Mesh(geometry, material); solid.position.z = -.5; scene.add(solid);
    t.after(() => { geometry.dispose(); material.dispose(); });
  }}), events = [];
  let uploaded;
  f.batch.onBeforeRender = function(renderer, scene, camera, geometry, material) {
    const transmission = renderer.getRenderTarget() !== null;
    if (transmission) {
      this.setMatrixAt(f.ids[0], new T.Matrix4().makeTranslation(-.2, 0, 0));
      uploaded = Array.from(this._matricesTexture.image.data);
    } else this._matricesTexture.image.data[12] = .875;
    T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, camera, geometry, material);
    events.push([transmission, material.side]);
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(events, [[true, T.BackSide], [false, T.DoubleSide]]);
  const draws = submitted(f.device); assert.equal(draws.length, 4);
  for (const draw of draws)
    assert.deepEqual(words(draw, f.batch._matricesTexture, 'rgba32float', Float32Array), uploaded);
  assert.equal(f.batch._matricesTexture.image.data[12], .875);
  assert.equal(f.batch.material.side, T.DoubleSide);
});

test('point-shadow faces and ArrayCamera views consume their own captured batch selections', async t => {
  const f = await fixture(t, 2, {}, {configure({scene, renderer, batch}) {
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
    batch.castShadow = true;
    const light = new T.PointLight(); light.position.z = 3; light.castShadow = true;
    light.shadow.mapSize.set(8, 8); scene.add(light);
  }}), expected = [];
  f.batch.geometry.setDrawRange(0, 0);
  let phase = 0, beforeShadow = 0, afterShadow = 0, afterRender = 0;
  f.batch.setCustomSort(list => list.sort((a, b) => phase % 2 ? b.index - a.index : a.index - b.index));
  f.batch.onBeforeShadow = function(...args) {
    beforeShadow++; T.BatchedMesh.prototype.onBeforeShadow.apply(this, args);
  };
  f.batch.onAfterShadow = () => afterShadow++;
  f.batch.onAfterRender = () => afterRender++;
  f.batch.onBeforeRender = function(renderer, scene, camera, geometry, material) {
    phase = expected.length;
    this.setMatrixAt(f.ids[0], new T.Matrix4().makeTranslation(-.75 + phase / 8, 0, 0));
    this.setColorAt(f.ids[0], new T.Color().setRGB(phase / 8, 1 - phase / 8, .25));
    T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, camera, geometry, material);
    expected.push({shadow: scene === null, camera: camera.name, ids: Array.from(this._indirectTexture.image.data),
      matrices: Array.from(this._matricesTexture.image.data), colors: Array.from(this._colorsTexture.image.data)});
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(beforeShadow, 6); assert.equal(afterShadow, 6); assert.equal(afterRender, 2);
  assert.deepEqual(expected.map(frame => frame.shadow), [true, true, true, true, true, true, false, false]);
  assert.deepEqual(expected.slice(6).map(frame => frame.camera), ['0', '1']);
  const draws = submitted(f.device); assert.equal(draws.length, 16);
  for (let i = 0; i < draws.length; i++) {
    const draw = draws[i], frame = expected[Math.floor(i / 2)], depth = draw.pass.depthStencilAttachment.view;
    assert.equal(depth.texture.format === 'depth32float', frame.shadow);
    if (frame.shadow) assert.equal(depth.baseArrayLayer ?? 0, Math.floor(i / 2));
    assert.equal(draw.args[0], 6, 'ordinary geometry.drawRange does not clip source batch ranges');
    assert.deepEqual(words(draw, f.batch._indirectTexture, 'r32uint', Uint32Array), frame.ids);
    assert.deepEqual(words(draw, f.batch._matricesTexture, 'rgba32float', Float32Array), frame.matrices);
    if (!frame.shadow) assert.deepEqual(words(draw, f.batch._colorsTexture, 'rgba32float', Float32Array), frame.colors);
  }
  assert.notDeepEqual(expected[0].ids, expected[1].ids);
  assert.notDeepEqual(expected[5].matrices, expected[6].matrices);
});

for (const requested of [true, false])
test(`afterShadow ${requested ? 'requested uploads survive' : 'versionless matrix edits stay stale across'} batch replay before color draws`, async t => {
  const f = await fixture(t, 1, {}, {configure({scene, renderer, batch}) {
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.BasicShadowMap; batch.castShadow = true;
    const light = new T.DirectionalLight(); light.position.z = 3; light.castShadow = true;
    light.shadow.mapSize.set(8, 8); scene.add(light);
  }});
  let shadowMatrix, colorMatrix, colorValues;
  f.batch.onBeforeShadow = function(...args) {
    this.setMatrixAt(f.ids[0], new T.Matrix4().makeTranslation(-.25, 0, 0));
    this.setColorAt(f.ids[0], new T.Color(0xff0000));
    T.BatchedMesh.prototype.onBeforeShadow.apply(this, args);
    shadowMatrix = [...this._matricesTexture.image.data];
    colorValues = [...this._colorsTexture.image.data];
  };
  f.batch.onAfterShadow = function() {
    if (requested) {
      this.setMatrixAt(f.ids[0], new T.Matrix4().makeTranslation(.875, 0, 0));
      this.setColorAt(f.ids[0], new T.Color(0x00ff00));
      colorMatrix = [...this._matricesTexture.image.data]; colorValues = [...this._colorsTexture.image.data];
    } else {
      this._matricesTexture.image.data[12] = .875;
      colorMatrix = shadowMatrix;
    }
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = submitted(f.device); assert.equal(draws.length, 4);
  for (const draw of draws.slice(0, 2))
    assert.deepEqual(words(draw, f.batch._matricesTexture, 'rgba32float', Float32Array), shadowMatrix);
  for (const draw of draws.slice(2)) {
    assert.deepEqual(words(draw, f.batch._matricesTexture, 'rgba32float', Float32Array), colorMatrix);
    assert.deepEqual(words(draw, f.batch._colorsTexture, 'rgba32float', Float32Array), colorValues);
  }
});

test('nonindexed batch wireframe transmission uses the selected source edge-index stride', async t => {
  const f = await fixture(t, 1, {}, {indexed: false, configure({scene, batch, ids}) {
    const plane = new T.PlaneGeometry(.5, .5), extra = plane.toNonIndexed();
    batch.setGeometryIdAt(ids[1], batch.addGeometry(extra)); batch.material.wireframe = true;
    const material = new T.MeshPhysicalMaterial({transmission: .75, thickness: .25}), glass = new T.Mesh(plane, material);
    glass.position.z = .25; scene.add(glass);
    t.after(() => { plane.dispose(); extra.dispose(); material.dispose(); });
  }});
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.batch.geometry.index, null);
  assert.deepEqual(Array.from(f.batch._multiDrawStarts).slice(0, 2), [0, 24]);
  const draws = submitted(f.device); assert.equal(draws.length, 4);
  assert.ok(draws.every(draw => draw.indexed && draw.pipeline.primitive.topology === 'line-list'));
  assert.deepEqual(draws.map(draw => draw.args[0]), [12, 12, 12, 12]);
  assert.deepEqual(draws.map(draw => draw.args[2]), [0, 12, 0, 12],
    'both transmission and main divide source starts by the selected Uint16 wireframe index width');
});
