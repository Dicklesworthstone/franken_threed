/** Actual WebGL facade: rendered depth -> custom comparison shader consumers.
 * The recording device proves bindings, orientation commands and queued inputs;
 * it does not execute shaders or establish pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const COPY = 'Source target comparison depth UV orientation';
async function fixture(t, {comparison = true, depthType = T.FloatType} = {}) {
  const device = geometryDevice(), textures = [], events = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = descriptor => {
    const encoder = createEncoder();
    return {...encoder, beginRenderPass(options) {
      const pass = encoder.beginRenderPass(options); device.passes.at(-1).label = descriptor?.label; return pass;
    }};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => { events.push(...commands.flat()); submit(commands); };
  let configuration;
  const canvas = {width: 32, height: 24, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 24], format: configuration.format, usage: configuration.usage}); },
  })};
  const renderer = new (createWebGLRendererClass(T))({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const depth = new T.DepthTexture(8, 6, depthType);
  if (comparison) { depth.compareFunction = T.LessEqualCompare; depth.minFilter = depth.magFilter = T.LinearFilter; }
  const target = new T.WebGLRenderTarget(8, 6, {depthTexture: depth});
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10); camera.position.z = 4;
  const mesh = material => {
    const object = new T.Mesh(new T.PlaneGeometry(2, 2), material); object.frustumCulled = false;
    t.after(() => { object.geometry.dispose(); material.dispose(); }); return object;
  };
  const producer = new T.Scene(); producer.add(mesh(new T.MeshBasicMaterial({color: 0x804020})));
  await renderer.init(); t.after(() => { renderer.dispose(); target.dispose(); depth.dispose(); });
  const produce = async () => {
    renderer.setRenderTarget(target); await renderer.renderAsync(producer, camera); renderer.setRenderTarget(null);
  };
  return {renderer, device, textures, events, depth, target, camera, mesh, producer, produce};
}
function material(texture, {raw = false, stage = 'fragment', offset = false, shadow = true} = {}) {
  const sampler = `uniform highp ${shadow ? 'sampler2DShadow' : 'sampler2D'} shadowImage;uniform vec4 shadowCoord;`;
  const tap = `textureProj${offset ? 'Offset' : ''}(shadowImage,shadowCoord${offset ? ',ivec2(1,-1)' : ''})${shadow ? '' : '.r'}`;
  const output = raw ? 'out vec4 color;' : '', assign = raw ? 'color' : 'gl_FragColor';
  return new (raw ? T.RawShaderMaterial : T.ShaderMaterial)({
    ...(raw ? {glslVersion: T.GLSL3} : {}), depthTest: false, depthWrite: false, blending: T.NoBlending,
    uniforms: {shadowImage: {value: texture}, shadowCoord: {value: new T.Vector4(.25, .75, .625, 2)}},
    vertexShader: (raw ? 'precision highp float;in vec3 position;' : '') +
      (stage === 'vertex' ? sampler + `${raw ? 'out' : 'varying'} float visibility;` : '') +
      `void main(){${stage === 'vertex' ? `visibility=${tap};` : ''}gl_Position=vec4(position,1.0);}`,
    fragmentShader: 'precision highp float;' + output +
      (stage === 'vertex' ? `${raw ? 'in' : 'varying'} float visibility;` : sampler) +
      `void main(){${assign}=vec4(${stage === 'vertex' ? 'visibility' : tap});}`,
  });
}
const copies = f => f.events.filter(event => event.label === COPY);
const consumers = f => f.device.snapshots.flat().filter(draw =>
  [draw.pipeline.vertex.module.code, draw.pipeline.fragment.module.code].some(code => code.includes('shadowImage')));
const input = draw => {
  const group = draw.groups.get(1).group;
  return {view: group.entries.find(entry => entry.binding === 0).resource,
    sampler: group.entries.find(entry => entry.binding === 1).resource};
};
function uniformBytes(draw) {
  const {group, offsets} = draw.groups.get(0), resource = group.entries.find(entry => entry.resource?.buffer).resource;
  const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[0] ?? 0);
  return new Uint8Array(bytes.buffer, bytes.byteOffset + offset, resource.size);
}
const uniformValues = draw => {
  const bytes = uniformBytes(draw); return [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)];
};
const contains = (values, expected) => values.some((_, index) => expected.every((value, offset) => values[index + offset] === value));

for (const depthType of [T.FloatType, T.UnsignedIntType]) for (const raw of [false, true]) for (const stage of ['fragment', 'vertex'])
test(`${raw ? 'raw GLSL3' : 'converted ShaderMaterial'} projected ${stage} ${depthType === T.FloatType ? 'float' : 'fixed'} depth comparisons bind the rendered depth owner`, async t => {
  const f = await fixture(t, {depthType}), shader = material(f.depth, {raw, stage, offset: raw || stage === 'vertex'});
  const object = f.mesh(shader); await f.produce(); await f.renderer.renderAsync(object, f.camera);
  const rendered = consumers(f); assert.equal(rendered.length, 1);
  const draw = rendered[0], {view, sampler} = input(draw), copy = copies(f)[0];
  assert.equal(copies(f).length, 1); assert.equal(view.texture.format, 'depth32float'); assert.equal(view.aspect, 'depth-only');
  assert.equal(sampler.compare, 'less-equal'); assert.equal(sampler.minFilter, 'linear'); assert.equal(sampler.magFilter, 'linear');
  assert.equal(f.device.features.has('float32-filterable'), false);
  assert.ok(!f.textures.some(texture => texture.format === 'r32float'), 'comparison filtering uses depth storage, not a float color substitute');
  assert.deepEqual(draw.pipeline.layout.bindGroupLayouts[1].entries, [
    {binding: 0, visibility: 3, texture: {sampleType: 'depth', viewDimension: '2d'}},
    {binding: 1, visibility: 3, sampler: {type: 'comparison'}},
  ]);
  const wgsl = draw.pipeline[stage].module.code;
  assert.match(wgsl, /texture_depth_2d/); assert.match(wgsl, /sampler_comparison/);
  assert.match(wgsl, stage === 'vertex' ? /textureSampleCompareLevel\(/ : /textureSampleCompare\(/);
  assert.match(wgsl, /\.xyz \/ .*\.w/);
  // Inspect the submitted bytes using this actual facade program's reflection:
  // the depth32float sampling replica must retain the source image's domain.
  const {program} = f.renderer._programSupport.compile(shader, object);
  assert.equal(wgsl, program[stage]);
  const reflected = program.reflection.textures.find(texture => texture.name === 'shadowImage');
  assert.ok(Number.isInteger(reflected.depthReferenceClampOffset));
  assert.match(wgsl, /f3d_depth_reference_clamp_0\s*:\s*vec4<u32>/);
  assert.match(wgsl, /select\([^\n]*clamp\([^\n]*f3d_u\.f3d_depth_reference_clamp_0\.x != 0u/);
  const bytes = uniformBytes(draw);
  assert.deepEqual([...new Uint32Array(bytes.buffer, bytes.byteOffset + reflected.depthReferenceClampOffset, 4)],
    [depthType === T.FloatType ? 0 : 1, 0, 0, 0]);
  assert.equal(copy.desc.colorAttachments.length, 0);
  assert.equal(copy.desc.depthStencilAttachment.view, view);
  assert.deepEqual(copy.draws[0].pipeline.depthStencil, {format: 'depth32float', depthCompare: 'always', depthWriteEnabled: true});
  assert.match(copy.draws[0].pipeline.fragment.module.code, /@builtin\(frag_depth\)/);
  assert.match(copy.draws[0].pipeline.fragment.module.code, /i32\(size.y\) - 1 - i32\(p.y\)/);
  const sourceDepth = copy.draws[0].groups.get(0).group.entries[0].resource.texture;
  assert.equal(sourceDepth.format, depthType === T.FloatType ? 'depth32float' : 'depth24plus'); assert.notEqual(sourceDepth, view.texture);
  const producer = f.events.find(event => event.desc.depthStencilAttachment?.view.texture === sourceDepth && event.draws.some(value => value.indexed));
  const consumer = f.events.find(event => event.draws.some(value => value.pipeline === draw.pipeline));
  assert.ok(producer); assert.ok(f.events.indexOf(producer) < f.events.indexOf(copy));
  assert.ok(f.events.indexOf(copy) < f.events.indexOf(consumer));
  assert.equal(f.renderer.info.f3d.route, 'general-webgpu');
});

test('queued fullscreen comparisons retain each coordinate vector and reuse the orientation image until a producer writes', async t => {
  const f = await fixture(t), shader = material(f.depth, {offset: true}), consumer = f.mesh(shader);
  await f.produce();
  f.renderer.render(consumer, f.camera);
  shader.uniforms.shadowCoord.value.set(.75, .25, .875, 4); f.renderer.render(consumer, f.camera);
  shader.uniforms.shadowCoord.value.set(9, 8, 7, 6); await f.renderer.waitForGPU();
  const rendered = consumers(f); assert.equal(rendered.length, 2); assert.equal(copies(f).length, 1);
  assert.ok(contains(uniformValues(rendered[0]), [.25, .75, .625, 2]));
  assert.ok(contains(uniformValues(rendered[1]), [.75, .25, .875, 4]));
  assert.equal(input(rendered[0]).view, input(rendered[1]).view);
  assert.equal(input(rendered[0]).sampler, input(rendered[1]).sampler);
  await f.renderer.renderAsync(consumer, f.camera); assert.equal(copies(f).length, 1);
  await f.produce(); await f.renderer.renderAsync(consumer, f.camera);
  assert.equal(copies(f).length, 2); assert.equal(input(consumers(f).at(-1)).view, input(rendered[0]).view);
  assert.equal(f.renderer._pending.length, 0);
});

test('facade comparison mode keeps allocation state until needsUpdate and retains its last non-null mode', async t => {
  const f = await fixture(t), shader = material(f.depth), object = f.mesh(shader);
  await f.produce(); await f.renderer.renderAsync(object, f.camera);
  const first = input(consumers(f).at(-1)); assert.equal(first.sampler.compare, 'less-equal');
  f.depth.compareFunction = T.GreaterCompare;
  await f.renderer.renderAsync(object, f.camera);
  const unversioned = input(consumers(f).at(-1));
  assert.equal(unversioned.sampler, first.sampler); assert.equal(unversioned.sampler.compare, 'less-equal');
  f.depth.needsUpdate = true; await f.renderer.renderAsync(object, f.camera);
  const acknowledged = input(consumers(f).at(-1));
  assert.notEqual(acknowledged.sampler, first.sampler); assert.equal(acknowledged.sampler.compare, 'greater');
  assert.equal(acknowledged.view, first.view); assert.equal(first.sampler.compare, 'less-equal');
  f.depth.compareFunction = null; f.depth.needsUpdate = true;
  await f.renderer.renderAsync(object, f.camera);
  const cleared = input(consumers(f).at(-1));
  assert.equal(cleared.sampler, acknowledged.sampler); assert.equal(cleared.sampler.compare, 'greater');
  assert.equal(cleared.view, first.view); assert.equal(copies(f).length, 1);
  assert.equal(consumers(f).length, 4); assert.equal(f.renderer._pending.length, 0);
});

for (const kind of ['color image', 'missing comparison', 'ordinary sampler'])
test(`custom comparison binding refuses ${kind} sampler/image mismatch`, async t => {
  const f = await fixture(t, {comparison: kind !== 'missing comparison'}); await f.produce();
  const shader = material(kind === 'color image' ? f.target.texture : f.depth, {shadow: kind !== 'ordinary sampler'});
  await assert.rejects(f.renderer.renderAsync(f.mesh(shader), f.camera), error => {
    assert.match(error.code ?? '', /TEXTURE|SAMPLER|PROGRAM/); return true;
  });
  assert.equal(consumers(f).length, 0);
});

test('custom comparison feedback is refused before copying from the active attachment', async t => {
  const f = await fixture(t); await f.produce();
  f.renderer.setRenderTarget(f.target);
  await assert.rejects(f.renderer.renderAsync(f.mesh(material(f.depth)), f.camera), error => {
    assert.match(error.code ?? '', /FEEDBACK/); return true;
  });
  assert.equal(copies(f).length, 0); assert.equal(consumers(f).length, 0);
});
