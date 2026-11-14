/** Public TSL BufferAttributeNode inputs through the owned renderer. The
 * recording device checks submitted bytes and bindings, not rasterized pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const Renderer = createWebGPURendererClass(T), N = T.TSL;

function fixture(t) {
  const device = geometryDevice();
  device.limits.maxTextureDimension2D = 8192; device.limits.maxBindingsPerBindGroup = 1000;
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc});
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({configure(value) { configured = value; },
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [64, 64], format: configured.format, usage: 16})})};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  mesh.frustumCulled = false; scene.add(mesh);
  return {renderer, device, scene, camera, material, mesh};
}
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
// Output conversion also uses main; select the public inputs exercised here.
const draws = device => device.snapshots.flat().filter(draw =>
  draw.pipeline.vertex.entryPoint === 'main' &&
  /@location\(\s*\d+\s*\)\s*(?:nodeAttribute\d+|color)\s*:/.test(draw.pipeline.vertex.module.code));
function attributeWords(draw, input, Type = Float32Array) {
  const shader = draw.pipeline.vertex.module.code;
  const location = Number(new RegExp(`@location\\(\\s*(\\d+)\\s*\\)\\s*${input.name}\\s*:`).exec(shader)?.[1]);
  assert.ok(Number.isSafeInteger(location), `source input ${input.name} has a vertex location`);
  const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.attributes.some(a => a.shaderLocation === location));
  const buffer = draw.streams.get(slot), layout = draw.pipeline.vertex.buffers[slot];
  assert.ok(buffer && layout);
  return {words: [...new Type(draw.contents.get(buffer).buffer)], buffer, layout};
}

for (const interleaved of [false, true]) test(`public ${interleaved ? 'interleaved' : 'direct'} node inputs retain queued partial uploads and disposal`, async t => {
  const f = fixture(t), geometry = f.mesh.geometry;
  const source = interleaved ? new T.InterleavedBuffer(new Float32Array(24), 6)
    : new T.BufferAttribute(new Float32Array(12).fill(.25), 3);
  if (interleaved) {
    for (let i = 0; i < 4; i++) {
      source.array.set(geometry.attributes.position.array.subarray(i * 3, i * 3 + 3), i * 6);
      source.array.fill(.25, i * 6 + 3, i * 6 + 6);
    }
    geometry.setAttribute('position', new T.InterleavedBufferAttribute(source, 3, 0));
    geometry.setAttribute('color', new T.InterleavedBufferAttribute(source, 3, 3));
  } else geometry.setAttribute('color', source);
  const input = interleaved ? N.bufferAttribute(source, 'vec3', 6, 3) : N.bufferAttribute(source);
  f.material.colorNode = input; f.material.vertexColors = true;
  await f.renderer.init();
  const first = [...source.array], offset = interleaved ? 3 : 0;
  f.renderer.render(f.scene, f.camera);
  source.array.fill(.875); source.array.fill(.5, offset, offset + 3);
  source.addUpdateRange(offset, 3); source.needsUpdate = true;
  f.renderer.render(f.scene, f.camera);
  const expected = first.slice(); expected.fill(.5, offset, offset + 3);
  source.array = new Float32Array(source.array.length).fill(.125);
  f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.equal(draws(f.device).length, 3);
  assert.deepEqual(draws(f.device).map(draw => attributeWords(draw, input).words), [first, expected, expected]);
  assert.deepEqual(source.updateRanges, []);
  assert.equal(input.value, source); assert.equal(interleaved ? input.attribute.data : input.attribute, source);
  geometry.dispose();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(attributeWords(draws(f.device).at(-1), input).words, [...source.array], 'disposal/reuse allocates current source bytes');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const Type of [Int32Array, Uint32Array]) test(`public ${Type.name} node inputs preserve integer vertex formats and values`, async t => {
  const f = fixture(t), value = Type === Int32Array ? -16777219 : 16777219;
  const source = new T.BufferAttribute(new Type(12).fill(value), 3); source.gpuType = T.IntType;
  const input = N.bufferAttribute(source);
  f.material.colorNode = N.vec3(input).abs().div(20000000);
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(draws(f.device).length, 0, 'compilation does not upload or draw a source attribute');
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const actual = attributeWords(draws(f.device).at(-1), input, Type);
  assert.deepEqual(actual.words, Array(12).fill(value));
  assert.ok(actual.layout.attributes.some(a => a.format === (Type === Int32Array ? 'sint32x3' : 'uint32x3')));
  assert.equal(input.value, source); assert.equal(source.version, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('public dynamic node inputs share the first upload across ordinary companion draws', async t => {
  const f = fixture(t), source = new T.BufferAttribute(new Float32Array(12).fill(.25), 3).setUsage(T.DynamicDrawUsage);
  const input = N.dynamicBufferAttribute(source), gain = N.uniform(1).onObjectUpdate(() => { source.array.fill(.75); return 1; });
  f.material.colorNode = input.mul(gain);
  const other = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial({vertexColors: true}));
  other.geometry.setAttribute('color', source); other.frustumCulled = false; other.renderOrder = 1; f.scene.add(other);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(draws(f.device).length, 2);
  assert.deepEqual(attributeWords(draws(f.device)[0], input).words, Array(12).fill(.25));
  const ordinary = draws(f.device)[1], color = {name: 'color'};
  assert.deepEqual(attributeWords(ordinary, color).words, Array(12).fill(.25), 'the companion observes the first source upload, not later CPU edits');
  assert.deepEqual([...source.array], Array(12).fill(.75), 'the current-node callback runs after geometry capture');
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(draws(f.device).length, 4);
  assert.deepEqual(attributeWords(draws(f.device)[2], input).words, Array(12).fill(.75));
  assert.equal(source.version, 0, 'dynamic source uploads do not fabricate needsUpdate');
});

test('external public node inputs cannot replace an earlier uncaptured fixed upload history', async t => {
  const f = fixture(t), source = new T.BufferAttribute(new Float32Array(12).fill(.25), 3);
  f.mesh.geometry.setAttribute('color', source); f.mesh.material = new T.MeshBasicMaterial({vertexColors: true});
  f.renderer.shadowMap.enabled = true;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.ok(f.device.snapshots.flat().some(draw => draw.pipeline.vertex.entryPoint === 'vertex_main'));
  source.array.fill(.75); f.renderer.shadowMap.enabled = false;
  let updates = 0;
  f.material.colorNode = N.bufferAttribute(source).mul(N.uniform(1).onObjectUpdate(() => { updates++; return 1; }));
  f.scene.remove(f.mesh);
  const consumer = new T.Mesh(new T.PlaneGeometry(), f.material); consumer.frustumCulled = false; f.scene.add(consumer);
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, 0, 'history validation precedes user node callbacks');
  assert.equal(draws(f.device).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const registered of [false, true]) for (const Type of [Float32Array, Int8Array, Uint8Array, Int16Array, Uint16Array])
for (const interleaved of Type === Float32Array ? [true] : [false, true])
test(`public material extensions retain queued ${Type.name} ${interleaved ? 'interleaved' : 'normalized'} bytes (registered=${registered})`, async t => {
  const f = fixture(t), normalized = Type !== Float32Array, stride = interleaved ? 8 : 3, offset = interleaved ? 4 : 0;
  const max = new Map([[Float32Array, 1], [Int8Array, 127], [Uint8Array, 255], [Int16Array, 32767], [Uint16Array, 65535]]).get(Type);
  const min = Type === Int8Array ? -128 : Type === Int16Array ? -32768 : 0;
  const normalize = value => normalized ? Math.fround(Math.max(value / max, -1)) : value;
  let uploads = 0;
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder, output) { return super.setupOutput(builder, output.mul(.75)); }
  }
  class Application extends T.MeshBasicMaterial { constructor() { super(); this.type = 'QueuedGeometryExtension'; } }
  if (registered) f.renderer.library.addMaterial(Extension, 'QueuedGeometryExtension');
  f.mesh.material = new (registered ? Application : Extension)(); f.mesh.material.vertexColors = true;
  const make = array => {
    const owner = interleaved ? new T.InterleavedBuffer(array, stride) : new T.BufferAttribute(array, 3, normalized);
    owner.onUpload(() => { uploads++; });
    return {owner, attribute: interleaved ? new T.InterleavedBufferAttribute(owner, 3, offset, normalized) : owner};
  };
  const array = new Type(4 * stride).fill(7);
  for (let i = 0; i < 4; i++) array.set([min, max, max / 2], i * stride + offset);
  const source = make(array); f.mesh.geometry.setAttribute('color', source.attribute);
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(draws(f.device).length, 0);
  const geometryDraws = () => draws(f.device).filter(draw => /@location\(\s*\d+\s*\)\s*color\s*:/.test(draw.pipeline.vertex.module.code));
  const first = [...array], expected = first.slice();
  f.renderer.render(f.scene, f.camera);
  array.fill(min); array.set([max, 0, max / 2], offset);
  const start = interleaved ? offset : 1, count = interleaved ? 4 : 0;
  source.owner.addUpdateRange(start, count); source.owner.needsUpdate = true;
  for (let i = interleaved ? start : 0; i < (interleaved ? start + count : 3); i++) expected[i] = array[i];
  f.renderer.render(f.scene, f.camera);
  array.fill(max); f.renderer.render(f.scene, f.camera);
  const replacement = make(new Type(array.length).fill(max / 2));
  f.mesh.geometry.setAttribute('color', replacement.attribute); f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.equal(geometryDraws().length, 4);
  assert.deepEqual(geometryDraws().map(draw => attributeWords(draw, {name: 'color'}).words),
    [first, expected, expected, [...replacement.owner.array]].map(values => values.map(normalize)));
  assert.deepEqual(source.owner.updateRanges, []); assert.equal(uploads, 0, 'native uploads do not invoke WebGL onUpload callbacks');
  assert.equal(source.attribute.normalized, normalized); assert.equal(source.attribute.itemSize, 3);
  f.mesh.geometry.dispose(); replacement.owner.array.fill(max);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(attributeWords(geometryDraws().at(-1), {name: 'color'}).words, [...replacement.owner.array].map(normalize));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
