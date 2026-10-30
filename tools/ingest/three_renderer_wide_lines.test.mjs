/** Pinned wide-line addons through native preparation, snapshots and commands.
 * The recording device checks resource/order contracts, not GPU pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {Line2} from '../../upstream/three.js/examples/jsm/lines/webgpu/Line2.js';
import {LineSegments2} from '../../upstream/three.js/examples/jsm/lines/webgpu/LineSegments2.js';
import {LineGeometry} from '../../upstream/three.js/examples/jsm/lines/LineGeometry.js';
import {LineSegmentsGeometry} from '../../upstream/three.js/examples/jsm/lines/LineSegmentsGeometry.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {nativePrimitiveScenarios} from '../../tests/e2e/webgpu_renderer/scenarios_native_primitives.js';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const isLine = draw => /instanceStart/.test(draw.pipeline.vertex.module.code);
const lineDraws = device => device.snapshots.flat().filter(isLine);
// Shared source accessor nodes can number uniforms or emit independent groups
// in a different order on a second builder. Retain group/binding numbers,
// member order/types and every use; normalize only those spelling differences.
function sourceShader(shader) {
  const blocks = [];
  shader = shader.replace(/struct \w+Struct \{[^}]+\};\n@binding\( (\d+) \) @group\( (\d+) \)\nvar<uniform> \w+ : \w+Struct;\n/g,
    (block, binding, group) => { blocks.push({block, binding: +binding, group: +group}); return ''; });
  blocks.sort((a, b) => a.group - b.group || a.binding - b.binding);
  const names = new Map();
  return (blocks.map(value => value.block).join('') + shader).replace(/\bnodeUniform\d+\b/g, name => {
    if (!names.has(name)) names.set(name, `nodeUniform${names.size}`); return names.get(name);
  });
}
const uniformValues = draw => [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
  .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
  .flatMap((entry, index) => {
    const resource = group.entries.find(value => value.binding === entry.binding).resource;
    const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
    return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
  }));
const streams = draw => draw.pipeline.vertex.buffers.flatMap((layout, slot) => {
  if (layout.stepMode !== 'instance') return [];
  const buffer = draw.streams.get(slot), bytes = draw.contents.get(buffer);
  return [{layout, buffer, values: [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]}];
});

async function fixture(t, {polyline = false, worldUnits = false, dashed = false, samples = 0,
  vertexColors = false, transparent = false} = {}) {
  const device = geometryDevice(), events = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = createEncoder(), commands = [];
    return {...encoder, beginRenderPass(descriptor) {
      const pass = encoder.beginRenderPass(descriptor), record = device.passes.at(-1); commands.push(record);
      return {...pass, setViewport(...value) { record.viewport = value; }};
    }, copyTextureToTexture(source, destination, size) {
      commands.push({copy: true, source, destination, size, draws: []});
    }, finish() { return commands; }};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => { for (const list of commands) events.push(...list); submit(commands); };
  let configuration;
  const canvas = {width: 64, height: 48, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configuration.format,
      usage: configuration.usage}); }
  })};
  const renderer = new Renderer({canvas, device, samples});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20); camera.position.z = 4;
  const geometry = polyline ? new LineGeometry() : new LineSegmentsGeometry();
  const positions = polyline ? [-1, -.5, 0, -.25, .5, 0, .5, -.5, 0, 1, .5, 0]
    : [-1, -.5, 0, -.25, .5, 0, .25, -.5, 0, 1, .5, 0];
  geometry.setPositions(positions);
  geometry.setColors([.125, .25, .5, .25, .5, .75, .5, .75, 1, .75, .5, .25]);
  const material = new T.Line2NodeMaterial({linewidth: worldUnits ? .125 : 5.5, worldUnits, dashed,
    dashSize: .375, gapSize: .1875, vertexColors, transparent, opacity: transparent ? .625 : 1});
  const object = polyline ? new Line2(geometry, material) : new LineSegments2(geometry, material);
  object.computeLineDistances(); object.frustumCulled = false; scene.add(object);
  const frames = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const result = capture.call(this, request); if (result) frames.push(result); return result;
  };
  t.after(() => { renderer.dispose(); geometry.dispose(); material.dispose(); });
  await renderer.init();
  return {renderer, scene, camera, object, geometry, material, device, events, frames, canvas};
}

for (const worldUnits of [false, true]) for (const dashed of [false, true])
test(`${worldUnits ? 'world' : 'screen'}-width ${dashed ? 'dashed' : 'solid'} addons preserve source WGSL and instanced streams`, async t => {
  const f = await fixture(t, {worldUnits, dashed, polyline: dashed, vertexColors: true});
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const [draw] = lineDraws(f.device), descriptor = f.frames[0].draws[0].binding.nodeProgram;
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.material = f.material; builder.context.material = f.material;
  builder.scene = f.scene; builder.camera = f.camera; builder.build();
  assert.equal(sourceShader(descriptor.vertex), sourceShader(builder.vertexShader));
  assert.equal(sourceShader(descriptor.fragment), sourceShader(builder.fragmentShader));
  assert.equal(draw.pipeline.primitive.topology, 'triangle-list');
  assert.deepEqual(draw.args, [18, dashed ? 3 : 2, 0, 0, 0]);
  assert.equal(streams(draw).length, dashed ? 3 : 2);
  assert.ok(streams(draw).some(stream => stream.layout.arrayStride === 24 &&
    stream.layout.attributes.map(attribute => attribute.offset).sort((a, b) => a - b).join() === '0,12'));
  assert.equal(f.material.colorNode, null, 'source wide lines require no authored placeholder slot');
  assert.deepEqual(f.object._resolution.toArray(), [64, 48], 'the actual addon render hook still updates public raycast inputs');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued width, dash controls and colors retain each draw before later mutations', async t => {
  const f = await fixture(t, {dashed: true});
  for (const width of [3.25, 7.75]) {
    f.material.linewidth = width; f.material.dashOffset = width / 8;
    f.material.color.setRGB(width / 16, .5, .75);
    f.renderer.render(f.scene, f.camera);
  }
  f.material.linewidth = 99; f.material.dashOffset = 99; f.material.color.setRGB(99, 99, 99);
  await f.renderer.waitForGPU();
  const draws = lineDraws(f.device); assert.equal(draws.length, 2);
  assert.equal(draws[0].pipeline, draws[1].pipeline, 'uniform controls reuse the source program');
  for (const [index, width] of [3.25, 7.75].entries()) {
    const values = uniformValues(draws[index]);
    assert.ok(values.includes(width)); assert.ok(values.includes(width / 8)); assert.ok(values.includes(width / 16));
    assert.ok(!values.includes(99));
  }
});

test('source world-unit, dash and coverage setters rebuild programs while retaining queued variants', async t => {
  const f = await fixture(t, {samples: 4});
  f.renderer.render(f.scene, f.camera);
  f.material.dashed = true; f.renderer.render(f.scene, f.camera);
  f.material.worldUnits = true; f.material.dashed = false; f.renderer.render(f.scene, f.camera);
  f.material.alphaToCoverage = false; f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const draws = lineDraws(f.device);
  assert.equal(draws.length, 4); assert.equal(new Set(draws.map(draw => draw.pipeline)).size, 4);
  assert.doesNotMatch(draws[0].pipeline.vertex.module.code, /instanceDistanceStart/);
  assert.match(draws[1].pipeline.vertex.module.code, /instanceDistanceStart/);
  assert.notEqual(draws[1].pipeline.vertex.module.code, draws[2].pipeline.vertex.module.code);
  assert.notEqual(draws[2].pipeline.fragment.module.code, draws[3].pipeline.fragment.module.code);
});

test('authored dash nodes retain source updates and complete line-distance attributes', async t => {
  const f = await fixture(t, {dashed: true}); let updates = 0;
  f.material.offsetNode = N.uniform(.3125).onObjectUpdate(() => { updates++; return updates / 8; });
  f.material.dashScaleNode = N.uniform(.75); f.material.dashSizeNode = N.uniform(.625); f.material.gapSizeNode = N.uniform(.4375);
  f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  assert.equal(updates, 2);
  for (const [index, draw] of lineDraws(f.device).entries()) {
    const values = uniformValues(draw);
    assert.ok(values.includes((index + 1) / 8)); assert.ok(values.includes(.625)); assert.ok(values.includes(.4375));
    assert.ok(streams(draw).some(stream => stream.layout.arrayStride === 8 && stream.layout.attributes.length === 2));
  }
});

test('interleaved endpoint partial uploads preserve unrequested GPU words', async t => {
  const f = await fixture(t), owner = f.geometry.attributes.instanceStart.data;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const original = streams(lineDraws(f.device)[0])[0].values;
  owner.array[0] = -.875; owner.array[1] = 99; owner.addUpdateRange(0, 1); owner.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(streams(lineDraws(f.device)[1])[0].values, original.map((value, index) => index === 0 ? -.875 : value));
  assert.equal(owner.array[1], 99); assert.deepEqual(owner.updateRanges, []);
});

test('queued screen widths preserve viewport, pixel ratio and addon raycast resolution after resize', async t => {
  const f = await fixture(t), observed = [];
  for (const [ratio, width, height] of [[1.5, 80, 60], [2, 96, 72]]) {
    f.renderer.setPixelRatio(ratio); f.renderer.setSize(width, height, false);
    f.renderer.setViewport(1, 2, width - 4, height - 6);
    f.renderer.render(f.scene, f.camera); observed.push(f.object._resolution.toArray());
  }
  await f.renderer.waitForGPU();
  assert.deepEqual(observed, [[76, 54], [92, 66]]);
  const draws = lineDraws(f.device);
  assert.ok(uniformValues(draws[0]).includes(114)); assert.ok(uniformValues(draws[0]).includes(81));
  assert.ok(uniformValues(draws[1]).includes(184)); assert.ok(uniformValues(draws[1]).includes(132));
  assert.ok(uniformValues(draws[0]).includes(1.5)); assert.ok(uniformValues(draws[1]).includes(2));
});

test('multisampled wide lines preserve source alpha-to-coverage pipeline state', async t => {
  const f = await fixture(t, {samples: 4});
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const [draw] = lineDraws(f.device);
  assert.equal(draw.pipeline.multisample.count, 4); assert.equal(draw.pipeline.multisample.alphaToCoverageEnabled, true);
});

test('source sample counts zero and one select different wide-line coverage shaders', async t => {
  const f = await fixture(t), target = new T.RenderTarget(64, 48, {samples: 0});
  f.scene.add(new T.Mesh(new T.PlaneGeometry(3, 3), new T.MeshBasicMaterial()));
  t.after(() => target.dispose()); f.renderer.setRenderTarget(target);
  await f.renderer.compileAsync(f.scene, f.camera);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  target.samples = 1; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const descriptors = f.frames.flatMap(frame => frame.draws.filter(draw => draw.material === f.material).map(draw => draw.binding.nodeProgram));
  assert.notEqual(descriptors[0], descriptors[1]);
  assert.doesNotMatch(descriptors[0].fragment, /fwidth/); assert.match(descriptors[1].fragment, /fwidth/);
});

test('transparent wide lines sample the captured opaque framebuffer and its mip chain', async t => {
  const f = await fixture(t, {transparent: true});
  f.scene.add(new T.Mesh(new T.PlaneGeometry(3, 3), new T.MeshBasicMaterial({color: 0x804020})));
  f.object.position.z = .5;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const copies = f.events.filter(event => event.copy); assert.equal(copies.length, 1);
  const [draw] = lineDraws(f.device);
  const sampled = [...draw.groups.values()].flatMap(({group}) => group.entries)
    .map(entry => entry.resource?.texture).filter(Boolean);
  assert.ok(sampled.includes(copies[0].destination.texture)); assert.ok(copies[0].destination.texture.mipLevelCount > 1);
  const at = f.events.indexOf(copies[0]);
  assert.ok(f.events.slice(0, at).some(event => event.draws.some(value => !isLine(value))));
  assert.ok(f.events.slice(at + 1).some(event => event.draws.some(isLine)));
  assert.equal(draw.pipeline.fragment.targets[0].blend, undefined, 'source Line2 composes opacity inside its shader');
});

test('precompiled mixed wide-line controls preserve every native binding across source transitions', async t => {
  const f = await fixture(t), target = new T.RenderTarget(128, 96, {type: T.UnsignedByteType});
  t.after(() => target.dispose()); f.renderer.setRenderTarget(target);
  const {scene, camera} = nativePrimitiveScenarios.native_wide_lines(T, f.renderer);
  await f.renderer.compileAsync(scene, camera);
  for (let frame = 1; frame <= 13; frame++) { f.renderer.render(scene, camera); await f.renderer.waitForGPU(); }
  assert.equal(lineDraws(f.device).length, 13 * 4);
});
