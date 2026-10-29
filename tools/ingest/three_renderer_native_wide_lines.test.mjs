/** Official wide-line addons through the owned WebGPU renderer. The recording
 * device proves commands and captured data; browser lanes prove raster parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {Line2} from '../../upstream/three.js/examples/jsm/lines/webgpu/Line2.js';
import {LineSegments2} from '../../upstream/three.js/examples/jsm/lines/webgpu/LineSegments2.js';
import {LineGeometry} from '../../upstream/three.js/examples/jsm/lines/LineGeometry.js';
import {LineSegmentsGeometry} from '../../upstream/three.js/examples/jsm/lines/LineSegmentsGeometry.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const native = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
// Shared source matrix accessors get generated uniform numbers in encounter
// order. Preserve every member's identity/type/offset and binding while
// normalizing only that naming and independent block declaration order.
function canonical(shader) {
  const names = new Map(), blocks = [];
  const body = shader.replace(/struct (\w+)Struct \{\n([\s\S]*?)\n\};\n@binding\( (\d+) \) @group\( (\d+) \)\nvar<uniform> \1 : \1Struct;\n/g,
    (block, group, fields, binding, index) => {
      let member = 0;
      fields.replace(/\bnodeUniform\d+\b/g, name => { names.set(name, `${group}Member${member++}`); return name; });
      blocks.push({block, key: Number(index) * 1000 + Number(binding)}); return '';
    });
  const replace = value => value.replace(/\bnodeUniform\d+\b/g, name => names.get(name) ?? name).replace(/\n{3,}/g, '\n\n');
  return blocks.sort((a, b) => a.key - b.key).map(({block}) => replace(block)).join('\n') + replace(body);
}
const floats = packets => packets.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
function fixture(t, parameters = {}) {
  const device = geometryDevice(), copies = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = desc => ({...desc, getBindGroupLayout: () => ({})});
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => ({...encode(), copyTextureToTexture(...args) { copies.push(args); }});
  let configured;
  const canvas = {width: 80, height: 60, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device, ...parameters}); renderer.setSize(80, 60, false);
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20); camera.position.z = 4;
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) { const value = capture.call(this, request); if (value) captures.push(value); return value; };
  const add = object => { scene.add(object); t.after(() => { object.geometry.dispose(); object.material.dispose(); }); return object; };
  return {device, renderer, scene, camera, captures, copies, add};
}
function line(material, joined = false) {
  const geometry = joined ? new LineGeometry() : new LineSegmentsGeometry();
  geometry.setPositions(joined ? [-.8, -.3, 0, 0, .4, 0, .8, -.2, 0]
    : [-.8, -.3, 0, -.1, .4, 0, .1, -.4, 0, .8, .2, 0]);
  geometry.setColors(joined ? [1, .25, .5, .25, 1, .5, .25, .5, 1]
    : [1, .25, .5, .25, 1, .5, .5, .25, 1, .25, .5, 1]);
  return (joined ? new Line2(geometry, material) : new LineSegments2(geometry, material)).computeLineDistances();
}
function captured(f, object) {
  return f.captures.flatMap(value => value.draws ?? []).filter(draw => draw.object === object);
}

for (const worldUnits of [false, true]) for (const dashed of [false, true])
test(`official ${dashed ? 'Line2' : 'LineSegments2'} retains ${worldUnits ? 'world' : 'screen'} widths, colors and source WGSL`, async t => {
  const f = fixture(t), material = new T.Line2NodeMaterial({worldUnits, dashed, vertexColors: true,
    linewidth: worldUnits ? .125 : 7, color: 0xffffff, dashSize: .25, gapSize: .125});
  const object = f.add(line(material, dashed));
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draw = native(f.device)[0], descriptor = captured(f, object)[0].binding.nodeProgram;
  const source = new T.WGSLNodeBuilder(object, f.renderer); source.scene = f.scene; source.camera = f.camera; source.build();
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
  assert.equal(draw.pipeline.primitive.topology, 'triangle-list'); assert.equal(draw.indexed, true);
  assert.deepEqual(draw.args.slice(0, 2), [18, 2]);
  assert.equal(draw.pipeline.multisample.alphaToCoverageEnabled, false);
  assert.ok(draw.pipeline.vertex.buffers.some(layout => layout.stepMode === 'instance'));
  assert.match(descriptor.vertex, /instanceStart/); assert.match(descriptor.vertex, /instanceEnd/);
  assert.match(descriptor.vertex, /instanceColorStart/); assert.match(descriptor.vertex, /instanceColorEnd/);
  if (dashed) assert.match(descriptor.vertex, /instanceDistanceStart/);
  assert.deepEqual(object._resolution.toArray(), [80, 60], 'source onBeforeRender remains observable by raycasting');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('wide-line updates capture width, DPR, viewport, dash nodes and source callbacks before queued mutations', async t => {
  const f = fixture(t), counts = {frame: 0, render: 0, object: 0, before: 0, after: 0};
  const material = new T.Line2NodeMaterial({dashed: true, linewidth: 7});
  material.colorNode = N.uniform(new T.Color(.25, .5, .75)).onFrameUpdate(() => { counts.frame++; });
  material.offsetNode = N.uniform(.125).onRenderUpdate(() => { counts.render++; });
  material.dashScaleNode = N.uniform(1.5).onObjectUpdate(() => { counts.object++; });
  material.dashSizeNode = N.float(.375); material.gapSizeNode = N.float(.25);
  const object = f.add(line(material)), sourceBefore = object.onBeforeRender;
  object.onBeforeRender = function (...args) { counts.before++; sourceBefore.apply(this, args); };
  object.onAfterRender = () => { counts.after++; };
  await f.renderer.init(); f.renderer.setPixelRatio(1.5); f.renderer.setViewport(3, 5, 42, 26);
  f.renderer.render(f.scene, f.camera);
  const first = captured(f, object)[0], saved = first.nodeUniforms.map(bytes => bytes.slice());
  assert.deepEqual(object._resolution.toArray(), [42, 26]);
  f.renderer.setPixelRatio(2); f.renderer.setViewport(1, 2, 37, 23); material.linewidth = 19;
  material.offsetNode.value = .625; object.geometry.instanceCount = 1;
  f.renderer.render(f.scene, f.camera);
  material.linewidth = 99; material.offsetNode.value = 99; object.geometry.instanceCount = 0;
  const second = captured(f, object)[1];
  assert.deepEqual(counts, {frame: 1, render: 2, object: 2, before: 2, after: 2});
  assert.equal(first.binding.nodeProgram, second.binding.nodeProgram, 'uniform changes keep the same source program');
  for (const value of [7, 1.5, 63, 39, .125]) assert.ok(floats(saved).includes(value));
  for (const value of [19, 2, 74, 46, .625]) assert.ok(floats(second.nodeUniforms).includes(value));
  assert.deepEqual(first.nodeUniforms, saved);
  await drain(f.renderer);
  assert.deepEqual(native(f.device).map(draw => draw.args[1]), [2, 1]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('wide-line structural controls recompile while queued draws retain their original variants', async t => {
  const f = fixture(t, {antialias: true}), material = new T.Line2NodeMaterial({linewidth: 9});
  const object = f.add(line(material)); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  material.dashed = true; f.renderer.render(f.scene, f.camera);
  material.worldUnits = true; material.linewidth = .125; f.renderer.render(f.scene, f.camera);
  material.alphaToCoverage = false; f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  const draws = native(f.device); assert.equal(draws.length, 4);
  assert.equal(new Set(draws.map(draw => draw.pipeline)).size, 4);
  assert.deepEqual(draws.map(draw => draw.pipeline.multisample.alphaToCoverageEnabled), [true, true, true, false]);
  assert.doesNotMatch(draws[0].pipeline.vertex.module.code, /instanceDistanceStart/);
  assert.match(draws[1].pipeline.vertex.module.code, /instanceDistanceStart/);
  assert.notEqual(draws[1].pipeline.vertex.module.code, draws[2].pipeline.vertex.module.code);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('stale wide-line programs reject structural changes before running authored updates', async t => {
  const f = fixture(t), material = new T.Line2NodeMaterial({dashed: true}); let updates = 0;
  material.offsetNode = N.uniform(.25).onObjectUpdate(() => { updates++; });
  const object = f.add(line(material)); await f.renderer.init(); f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const owner = createThreeNodePrograms({three: T, renderer: f.renderer}); t.after(() => owner.dispose());
  const context = {object, material, scene: f.scene, camera: f.camera, renderId: 1, frameId: 1};
  const descriptor = owner.compile(material, object, context); owner.capture(descriptor, context);
  material.worldUnits = true;
  assert.throws(() => owner.capture(descriptor, context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, 1); assert.notEqual(owner.compile(material, object, context), descriptor);
});

test('transparent wide lines retain source framebuffer sampling before their native draw', async t => {
  const f = fixture(t), material = new T.Line2NodeMaterial({transparent: true, opacity: .5, linewidth: 9});
  const object = f.add(line(material)); await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draw = native(f.device).find(draw => draw.streams.size);
  assert.ok(f.copies.length > 0); assert.equal(draw.args[1], 2);
  assert.ok(captured(f, object)[0].binding.nodeProgram.groups.some(group => group.entries.some(entry => entry.texture)));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
