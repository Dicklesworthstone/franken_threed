/** Actual WebGL facade, source scene callbacks and ESSL programs on a recording
 * GPU boundary. These assertions inspect command order, not rasterized pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const Renderer = createWebGLRendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
function fixture(t, samples = 0) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.textures = [];
  device.createTexture = desc => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = desc => ({...desc});
  device.createRenderPipeline = desc => ({...desc, getBindGroupLayout: () => ({})});
  device.textureCopies = [];
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = (...args) => {
    const encoder = encode(...args), copies = [];
    return {...encoder, copyTextureToTexture(source, destination, size) {
      assert.equal(source.texture.sampleCount, 1); assert.equal(destination.texture.sampleCount, 1);
      assert.equal(source.texture.format, destination.texture.format);
      assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
      assert.ok((source.mipLevel ?? 0) < source.texture.mipLevelCount);
      assert.ok((destination.mipLevel ?? 0) < destination.texture.mipLevelCount);
      const copy = {source, destination, size, submission: device.submissions.length};
      copies.push({draws: [], copy}); device.textureCopies.push(copy);
    }, finish() { return [...encoder.finish(), ...copies]; }};
  };
  let configured;
  const canvas = {width: 8, height: 4, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [8, 4], format: configured.format, usage: 16}),
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const target = new T.WebGLRenderTarget(8, 4, {count: 2, samples, depthBuffer: false});
  target.textures[1].type = T.HalfFloatType;
  for (const texture of target.textures) texture.generateMipmaps = true;
  t.after(() => target.dispose());
  const camera = new T.PerspectiveCamera(50, 2, .1, 30); camera.position.z = 4;
  const material = new T.ShaderMaterial({glslVersion: T.GLSL3,
    vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'layout(location=0) out vec4 first; layout(location=1) out vec4 second; void main() { first=vec4(1.0); second=vec4(0.5); }'});
  const geometry = new T.PlaneGeometry(); t.after(() => { geometry.dispose(); material.dispose(); });
  const scene = new T.Scene(), mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; scene.add(mesh);
  renderer.setRenderTarget(target);
  const mipPasses = start => device.passes.slice(start).filter(pass => (pass.desc.colorAttachments[0].view.baseMipLevel ?? 0) > 0);
  return {renderer, device, target, camera, material, geometry, scene, mesh, mipPasses};
}
async function initialize(f) {
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.mipPasses(0).length, 6, 'both initialized attachments retain a complete chain');
}

for (const deferred of [false, true]) for (const samples of [0, 4])
test(`WebGL MRT draw callbacks select final mip flags (${deferred ? 'deferred' : 'warm'}, ${samples} samples)`, async t => {
  const f = fixture(t, samples); await initialize(f);
  for (const phase of [0, 1]) {
    // A fresh source root forces the facade's asynchronous scene preparation.
    // The warm case retains the original prepared objects and callback identity.
    const scene = deferred ? new T.Scene() : f.scene;
    const mesh = deferred ? new T.Mesh(f.geometry, f.material) : f.mesh;
    if (deferred) { mesh.frustumCulled = false; scene.add(mesh); }
    const events = [], expected = phase === 0 ? 'rgba8unorm' : 'rgba16float';
    f.target.texture.generateMipmaps = false; f.target.textures[1].generateMipmaps = true;
    mesh.onBeforeRender = function(renderer, sourceScene, camera, geometry, material) {
      assert.equal(this, mesh); assert.equal(renderer, f.renderer); assert.equal(sourceScene, scene);
      assert.equal(camera, f.camera); assert.equal(geometry, f.geometry); assert.equal(material, f.material);
      events.push('object before'); f.target.texture.generateMipmaps = true;
    };
    f.material.onBeforeRender = function(renderer, sourceScene, camera, geometry, object) {
      assert.equal(this, f.material); assert.equal(renderer, f.renderer); assert.equal(sourceScene, scene);
      assert.equal(camera, f.camera); assert.equal(geometry, f.geometry); assert.equal(object, mesh);
      events.push('material before'); f.target.textures[1].generateMipmaps = false;
    };
    mesh.onAfterRender = function() {
      assert.equal(this, mesh); events.push('object after');
      if (phase === 1) { f.target.texture.generateMipmaps = false; f.target.textures[1].generateMipmaps = true; }
    };
    const start = f.device.passes.length;
    f.renderer.render(scene, f.camera);
    assert.equal(events.length, deferred ? 0 : 3, 'hooks retain the existing immediate/deferred boundary');
    if (deferred) {
      f.target.texture.generateMipmaps = false; f.target.textures[1].generateMipmaps = false;
      await drain(f.renderer);
    }
    assert.deepEqual(events, ['object before', 'material before', 'object after']);
    const generated = f.mipPasses(start);
    assert.equal(generated.length, 3); assert.ok(generated.every(pass => pass.desc.colorAttachments[0].view.texture.format === expected));
    assert.deepEqual(generated.map(pass => pass.desc.colorAttachments[0].view.baseMipLevel), [1, 2, 3]);
    const draw = f.device.passes.slice(start).find(pass => pass.desc.colorAttachments.length === 2);
    assert.ok(draw, 'the source MRT program reaches the command owner');
    assert.ok(f.device.passes.indexOf(draw) < f.device.passes.indexOf(generated[0]), 'generation follows the completed source draw');
    if (samples === 4) assert.ok(draw.desc.colorAttachments.every(color => color.resolveTarget));
  }
});

test('callback-free WebGL MRT draws retain separate queued generation masks through the real scene', async t => {
  const f = fixture(t); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  f.target.texture.generateMipmaps = false; f.renderer.render(f.scene, f.camera);
  f.target.textures[1].generateMipmaps = false;
  await drain(f.renderer);
  const generated = f.mipPasses(0);
  assert.equal(generated.length, 9);
  assert.deepEqual(generated.map(pass => pass.desc.colorAttachments[0].view.texture.format),
    [...Array(3).fill('rgba8unorm'), ...Array(6).fill('rgba16float')]);
});

test('a material-only WebGL MRT hook can enable a previously disabled allocated chain', async t => {
  const f = fixture(t); await initialize(f);
  for (const texture of f.target.textures) texture.generateMipmaps = false;
  let calls = 0;
  f.material.onBeforeRender = () => { calls++; f.target.textures[1].generateMipmaps = true; };
  const start = f.device.passes.length; f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 1);
  assert.deepEqual(f.mipPasses(start).map(pass => pass.desc.colorAttachments[0].view.texture.format), Array(3).fill('rgba16float'));
});

for (const deferred of [false, true]) for (const samples of [0, 4])
test(`WebGL MRT callbacks grow an initialized base-only color after its source draw (${deferred ? 'deferred' : 'warm'}, ${samples} samples)`, async t => {
  const f = fixture(t, samples); f.target.textures[1].generateMipmaps = false;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const entry = f.renderer._targets.capture(f.target);
  assert.deepEqual(entry.shape.colorMipLevelCounts, [4, 1]);
  const before = f.device.passes.find(pass => pass.desc.colorAttachments.length === 2);
  const base = before.desc.colorAttachments[1].resolveTarget?.texture ?? before.desc.colorAttachments[1].view.texture;
  const scene = deferred ? new T.Scene() : f.scene, mesh = deferred ? new T.Mesh(f.geometry, f.material) : f.mesh;
  if (deferred) { mesh.frustumCulled = false; scene.add(mesh); }
  f.target.texture.generateMipmaps = false;
  let calls = 0;
  f.material.onBeforeRender = () => { calls++; f.target.textures[1].generateMipmaps = true; };
  const start = f.device.passes.length;
  f.renderer.render(scene, f.camera);
  assert.equal(calls, deferred ? 0 : 1, 'the existing source callback boundary is preserved');
  if (deferred) { f.target.textures[1].generateMipmaps = false; await drain(f.renderer); }
  assert.equal(calls, 1); assert.equal(f.renderer._targets.capture(f.target), entry);
  assert.deepEqual(entry.shape.colorMipLevelCounts, [4, 4]);
  const generated = f.mipPasses(start), draw = f.device.passes.slice(start).find(pass => pass.desc.colorAttachments.length === 2);
  assert.equal(generated.length, 3);
  assert.ok(generated.every(pass => pass.desc.colorAttachments[0].view.texture.format === 'rgba16float'));
  assert.equal(f.device.textureCopies.length, 1);
  const copy = f.device.textureCopies[0];
  assert.equal(copy.source.texture, base);
  assert.equal(generated[0].desc.colorAttachments[0].view.texture, copy.destination.texture);
  const drawSubmission = f.device.submissions.findIndex(commands => commands.some(passes => passes.includes(draw)));
  assert.ok(drawSubmission < copy.submission, 'copying the completed base follows the source draw and resolve');
  assert.ok(f.device.passes.indexOf(draw) < f.device.passes.indexOf(generated[0]));
  assert.equal(draw.desc.colorAttachments[0].view.texture, before.desc.colorAttachments[0].view.texture);
  if (samples === 4) assert.equal(draw.desc.colorAttachments[1].view.texture, before.desc.colorAttachments[1].view.texture);
});

test('callback-free queued WebGL MRT growth occurs only at the accepted generating draw', async t => {
  const f = fixture(t);
  for (const texture of f.target.textures) texture.generateMipmaps = false;
  await f.renderer.init();
  const entry = f.renderer._targets.capture(f.target);
  f.renderer.render(f.scene, f.camera);
  f.target.textures[1].generateMipmaps = true; f.renderer.render(f.scene, f.camera);
  f.target.textures[1].generateMipmaps = false; f.renderer.render(f.scene, f.camera);
  assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 1], 'queued capture has not issued native growth');
  await drain(f.renderer);
  assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 4]);
  const draws = f.device.passes.filter(pass => pass.desc.colorAttachments.length === 2);
  assert.equal(draws.length, 3); assert.equal(f.device.textureCopies.length, 1);
  const copy = f.device.textureCopies[0];
  assert.equal(draws[0].desc.colorAttachments[1].view.texture, copy.source.texture);
  assert.equal(draws[1].desc.colorAttachments[1].view.texture, copy.source.texture);
  assert.equal(draws[2].desc.colorAttachments[1].view.texture, copy.destination.texture);
  assert.equal(f.mipPasses(0).length, 3, 'the later public flag does not erase or duplicate accepted generation');
  assert.equal(f.renderer._targets.capture(f.target), entry);
});

test('nested WebGL MRT callbacks update only their own execution masks', async t => {
  const f = fixture(t); await initialize(f);
  const inner = new T.WebGLRenderTarget(8, 4, {count: 2, depthBuffer: false});
  inner.textures[1].type = T.HalfFloatType;
  for (const texture of inner.textures) texture.generateMipmaps = true;
  t.after(() => inner.dispose());
  const scene = new T.Scene(), mesh = new T.Mesh(f.geometry, f.material.clone()); mesh.frustumCulled = false; scene.add(mesh);
  t.after(() => mesh.material.dispose());
  f.renderer.setRenderTarget(inner); f.renderer.render(scene, f.camera); await drain(f.renderer);
  const events = [];
  f.mesh.onBeforeRender = () => {
    events.push('outer before'); f.target.texture.generateMipmaps = true; f.target.textures[1].generateMipmaps = false;
    f.renderer.setRenderTarget(inner); f.renderer.render(scene, f.camera); f.renderer.setRenderTarget(f.target);
    events.push('outer resumed');
  };
  mesh.onBeforeRender = () => {
    events.push('inner before'); inner.texture.generateMipmaps = false; inner.textures[1].generateMipmaps = true;
    f.target.textures[1].generateMipmaps = true;
  };
  f.mesh.onAfterRender = () => { events.push('outer after'); f.target.texture.generateMipmaps = false; };
  f.target.texture.generateMipmaps = true; f.target.textures[1].generateMipmaps = false;
  inner.texture.generateMipmaps = true; inner.textures[1].generateMipmaps = false;
  f.renderer.setRenderTarget(f.target); const start = f.device.passes.length;
  f.renderer.render(f.scene, f.camera);
  assert.equal(f.renderer._drain, null, 'both nested scenes are already prepared');
  assert.deepEqual(events, ['outer before', 'inner before', 'outer resumed', 'outer after']);
  const generated = f.mipPasses(start);
  assert.equal(generated.length, 6); assert.ok(generated.every(pass => pass.desc.colorAttachments[0].view.texture.format === 'rgba16float'));
  assert.notEqual(generated[0].desc.colorAttachments[0].view.texture, generated[3].desc.colorAttachments[0].view.texture);
});
