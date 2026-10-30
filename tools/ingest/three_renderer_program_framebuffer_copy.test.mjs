// Actual public WebGL facade, framebuffer owner and source-material binding.
// The command recorder checks ordering/lifetime; browser comparison checks pixels.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

async function fixture(t) {
  const device = geometryDevice(), textures = [], events = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.features = new Set(['float32-filterable']);
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(descriptor = {}) {
        if (descriptor.format && descriptor.format !== this.format) assert.ok(this.viewFormats?.includes(descriptor.format));
        return {texture: this, descriptor};
      }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const create = device.createCommandEncoder;
  device.createCommandEncoder = descriptor => {
    const encoder = create();
    return {...encoder, beginRenderPass(options) {
      const pass = encoder.beginRenderPass(options), record = device.passes.at(-1); record.label = descriptor?.label;
      return {...pass, setBindGroup(index, group, ...rest) {
        for (const entry of group.entries ?? []) if (entry.resource?.texture) assert.ok(entry.resource.texture.usage & 4);
        pass.setBindGroup(index, group, ...rest);
      }, setViewport(...value) { record.viewport = value; pass.setViewport(...value); }};
    }, copyTextureToTexture() {}};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => { commands.flat().forEach(command => events.push(command)); submit(commands); };
  let configuration;
  const canvas = {width: 32, height: 24, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configuration.format, usage: configuration.usage}); },
  })};
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device});
  await renderer.init(); t.after(() => renderer.dispose());
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10); camera.position.z = 4;
  const scene = material => {
    const result = new T.Scene(), geometry = new T.PlaneGeometry(2, 2), mesh = new T.Mesh(geometry, material);
    result.add(mesh); t.after(() => { material.dispose(); geometry.dispose(); }); return result;
  };
  return {renderer, device, events, textures, camera, scene};
}
const blits = f => f.events.filter(event => event.label === 'f3d-public-framebuffer-copy');
const draw = event => event.draws[0];
const binding = (draw, index) => draw.groups.get(0).group.entries.find(entry => entry.binding === index).resource;
const region = event => Array.from(new Uint32Array(binding(draw(event), 1).buffer.data));

test('post-render WebGL framebuffer copies share native residency with ordinary material maps', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(8, 6), version = image.version, sourceVersion = image.source.version;
  const producer = f.scene(new T.MeshBasicMaterial({color: 0xff0000})), consumer = f.scene(new T.MeshBasicMaterial({map: image}));
  f.renderer.render(producer, f.camera);
  f.renderer.copyFramebufferToTexture(image, {x: 3.9, y: 2.9, z: 1, w: 1});
  f.renderer.render(consumer, f.camera); await f.renderer.waitForGPU();
  assert.equal(blits(f).length, 1); const copy = blits(f)[0], native = copy.desc.colorAttachments[0].view.texture;
  assert.deepEqual(region(copy), [3, 2, 24, 0]); assert.deepEqual(copy.viewport, [0, 0, 8, 6, 0, 1]);
  assert.equal(image.type, T.UnsignedByteType); assert.equal(image.version, version); assert.equal(image.source.version, sourceVersion);
  const at = f.events.indexOf(copy);
  assert.ok(f.events.slice(0, at).some(event => event.draws.some(value => value.indexed)));
  assert.ok(f.events.slice(at + 1).some(event => event.draws.some(value => [...value.groups.values()]
    .some(({group}) => group.entries.some(entry => entry.resource?.texture === native)))));
  assert.ok(f.device.buffers.filter(buffer => buffer.label === 'WebGL framebuffer copy region').every(buffer => buffer.destroyed));
  assert.equal(f.renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('destination initialization precedes current-target capture and preserves its own type and allocated extent', async t => {
  const f = await fixture(t), first = new T.WebGLRenderTarget(16, 16), second = new T.WebGLRenderTarget(32, 24);
  const image = new T.FramebufferTexture(4, 3); let callbacks = 0;
  t.after(() => { first.dispose(); second.dispose(); image.dispose(); });
  image.onUpdate = value => {
    assert.equal(value, image); callbacks++; f.renderer.setRenderTarget(second); image.image.width = 99;
    image.needsUpdate = true; // Source acknowledges this callback's version before returning.
  };
  f.renderer.setRenderTarget(first); f.renderer.copyFramebufferToTexture(image, new T.Vector4(2, 1, 99, 99));
  await f.renderer.waitForGPU();
  const copy = blits(f)[0]; assert.equal(callbacks, 1); assert.deepEqual(region(copy), [2, 1, 24, 0]);
  assert.equal(binding(draw(copy), 0).texture.format, 'rgba8unorm');
  assert.equal(copy.desc.colorAttachments[0].view.texture.format, 'rgba8unorm'); assert.equal(image.type, T.UnsignedByteType);
  assert.deepEqual(copy.viewport, [0, 0, 4, 3, 0, 1]);
  image.image.width = 4; f.renderer.copyFramebufferToTexture(image); await f.renderer.waitForGPU(); assert.equal(callbacks, 1);
  image.onUpdate = () => { callbacks++; }; image.needsUpdate = true;
  f.renderer.copyFramebufferToTexture(image); await f.renderer.waitForGPU(); assert.equal(callbacks, 2);
  assert.equal(blits(f)[2].desc.colorAttachments[0].view.texture, copy.desc.colorAttachments[0].view.texture);
});

test('an initialization callback exception prevents copying and is acknowledged before retry', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(4, 4), error = new Error('initialization failed'); let callbacks = 0;
  image.onUpdate = () => { callbacks++; throw error; };
  assert.throws(() => f.renderer.copyFramebufferToTexture(image), value => value === error);
  assert.equal(blits(f).length, 0); assert.equal(f.renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
  f.renderer.copyFramebufferToTexture(image); await f.renderer.waitForGPU();
  assert.equal(callbacks, 1); assert.equal(blits(f).length, 1);
});

test('selected source face and destination mip stay captured without post-copy mip generation', async t => {
  const f = await fixture(t), target = new T.WebGLCubeRenderTarget(16), image = new T.FramebufferTexture(8, 8);
  image.minFilter = T.NearestMipmapNearestFilter; image.needsUpdate = true;
  t.after(() => { target.dispose(); image.dispose(); });
  f.renderer.setRenderTarget(target, 3); f.renderer.copyFramebufferToTexture(image, new T.Vector2(2, 1), 1);
  f.renderer.setRenderTarget(target, 5); await f.renderer.waitForGPU();
  const copy = blits(f)[0]; assert.equal(binding(draw(copy), 0).descriptor.baseArrayLayer, 3);
  assert.equal(copy.desc.colorAttachments[0].view.descriptor.baseMipLevel, 1);
  assert.equal(copy.desc.colorAttachments[0].view.texture.mipLevelCount, 4);
  assert.deepEqual(region(copy), [2, 1, 16, 3]); assert.deepEqual(copy.viewport, [0, 0, 4, 4, 0, 1]);
  assert.equal(f.events.filter(event => event.label?.includes('framebuffer')).length, 1);
});

test('requested mip generation precedes copying and sRGB storage uses raw views', async t => {
  const f = await fixture(t), target = new T.WebGLRenderTarget(8, 8), image = new T.FramebufferTexture(8, 8);
  target.texture.colorSpace = T.SRGBColorSpace; image.colorSpace = T.SRGBColorSpace;
  image.generateMipmaps = true; image.minFilter = T.LinearMipmapLinearFilter; image.needsUpdate = true;
  t.after(() => { target.dispose(); image.dispose(); });
  f.renderer.setRenderTarget(target); f.renderer.copyFramebufferToTexture(image); await f.renderer.waitForGPU();
  const copy = blits(f)[0], at = f.events.indexOf(copy);
  assert.equal(f.events.slice(0, at).filter(event => event.label === 'f3d-webgl-framebuffer-initialize').length, 3);
  assert.equal(f.events.slice(at + 1).length, 0);
  assert.equal(binding(draw(copy), 0).descriptor.format, 'rgba8unorm');
  assert.equal(copy.desc.colorAttachments[0].view.descriptor.format, 'rgba8unorm');
  assert.equal(copy.desc.colorAttachments[0].view.texture.format, 'rgba8unorm-srgb');
});

test('draw callbacks copy after the preceding draw prefix and resume with loaded attachments', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(4, 4), scene = f.scene(new T.MeshBasicMaterial());
  const second = new T.Mesh(scene.children[0].geometry, new T.MeshBasicMaterial());
  scene.add(second); second.renderOrder = 1; t.after(() => second.material.dispose());
  let initialized = 0; const callbacks = [];
  image.onUpdate = () => { initialized++; callbacks.push('upload'); };
  second.onBeforeRender = () => { callbacks.push('before'); f.renderer.copyFramebufferToTexture(image); };
  second.onAfterRender = () => { callbacks.push('after'); f.renderer.copyFramebufferToTexture(image); };
  await f.renderer.compileAsync(scene, f.camera);
  f.renderer.render(scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(initialized, 1); assert.deepEqual(callbacks, ['before', 'upload', 'after']);
  const copies = blits(f); assert.equal(copies.length, 2);
  const first = f.events.indexOf(copies[0]), last = f.events.indexOf(copies[1]);
  const earlier = f.events.slice(0, first).filter(event => event.draws.some(value => value.indexed));
  const between = f.events.slice(first + 1, last).filter(event => event.draws.some(value => value.indexed));
  assert.equal(earlier.length, 1); assert.equal(earlier[0].draws.length, 1);
  assert.equal(between.length, 1); assert.equal(between[0].draws.length, 1);
  assert.equal(between[0].desc.colorAttachments[0].loadOp, 'load');
  assert.equal(between[0].desc.depthStencilAttachment.depthLoadOp, 'load');
  assert.equal(f.events.slice(last + 1).some(event => event.draws.some(value => value.indexed)), false);
  assert.equal(f.renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
  assert.equal(f.renderer._programDrawContext, undefined);
});

test('draw-time initialization failures preserve the already submitted color prefix', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(4, 4), scene = f.scene(new T.MeshBasicMaterial());
  const failure = new Error('source initialization failed'), second = new T.Mesh(scene.children[0].geometry, new T.MeshBasicMaterial());
  scene.add(second); second.renderOrder = 1; t.after(() => second.material.dispose());
  image.onUpdate = () => { throw failure; };
  second.onBeforeRender = () => f.renderer.copyFramebufferToTexture(image);
  await f.renderer.compileAsync(scene, f.camera);
  assert.throws(() => f.renderer.render(scene, f.camera), error => error === failure);
  assert.equal(blits(f).length, 0);
  assert.equal(f.events.filter(event => event.draws.some(value => value.indexed)).length, 1);
  assert.equal(f.renderer._programDrawContext, undefined);
  assert.equal(f.renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('draw-time requested mip initialization follows the preceding draw and precedes its copy', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(8, 8), scene = f.scene(new T.MeshBasicMaterial());
  image.generateMipmaps = true; image.minFilter = T.LinearMipmapLinearFilter; image.needsUpdate = true;
  scene.children[0].onAfterRender = () => f.renderer.copyFramebufferToTexture(image);
  await f.renderer.compileAsync(scene, f.camera);
  f.renderer.render(scene, f.camera); await f.renderer.waitForGPU();
  const copy = blits(f)[0], copyIndex = f.events.indexOf(copy);
  const mipIndices = f.events.map((event, index) => event.label === 'f3d-webgl-framebuffer-initialize' ? index : -1).filter(index => index !== -1);
  assert.equal(mipIndices.length, 3);
  assert.ok(f.events.slice(0, mipIndices[0]).some(event => event.draws.some(value => value.indexed)));
  assert.ok(mipIndices.every(index => index < copyIndex));
  assert.equal(f.renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('queued draw callbacks observe their captured target while later public selections remain intact', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(4, 4), scene = f.scene(new T.MeshBasicMaterial());
  const target = new T.WebGLRenderTarget(16, 12), later = new T.WebGLRenderTarget(32, 24), seen = [];
  t.after(() => { target.dispose(); later.dispose(); image.dispose(); });
  scene.children[0].onAfterRender = () => { seen.push(f.renderer.getRenderTarget()); f.renderer.copyFramebufferToTexture(image); };
  f.renderer.setRenderTarget(target); f.renderer.render(scene, f.camera);
  f.renderer.setRenderTarget(later); await f.renderer.waitForGPU();
  assert.deepEqual(seen, [target]); assert.equal(f.renderer.getRenderTarget(), later);
  assert.deepEqual(region(blits(f)[0]), [0, 0, 12, 0]);
});

function directMaterial(t) {
  const material = new T.RawShaderMaterial({uniforms: {tint: {value: new T.Vector3(1, 0, 0)}},
    vertexShader: 'precision highp float; attribute vec3 position; void main() { gl_Position = vec4(position, 1.0); }',
    fragmentShader: 'precision highp float; uniform vec3 tint; void main() { gl_FragColor = vec4(tint, 1.0); }',
    transparent: true, side: T.DoubleSide, depthTest: false, depthWrite: false});
  t.after(() => material.dispose()); return material;
}

test('cold callback-discovered direct shaders submit once per call without source render hooks or matrix updates', async t => {
  const f = await fixture(t), scene = f.scene(new T.MeshBasicMaterial()), material = directMaterial(t);
  const direct = new T.Mesh(scene.children[0].geometry, material), image = new T.FramebufferTexture(4, 4);
  const expected = direct.modelViewMatrix.makeTranslation(19, 23, 29).toArray();
  direct.onBeforeRender = direct.onAfterRender = material.onBeforeRender = () => assert.fail('direct calls do not run renderObject hooks');
  let callbacks = 0;
  scene.children[0].onBeforeRender = () => {
    callbacks++;
    material.uniforms.tint.value.set(1, 0, 0); material.uniformsNeedUpdate = true;
    f.renderer.renderBufferDirect(f.camera, null, direct.geometry, material, direct, null);
    assert.equal(material.uniformsNeedUpdate, false);
    material.uniforms.tint.value.set(0, 1, 0); material.uniformsNeedUpdate = true;
    f.renderer.renderBufferDirect(f.camera, null, direct.geometry, material, direct, null);
    Object.defineProperty(material.uniforms.tint, 'value', {configurable: true,
      get() { assert.fail('unchanged direct material does not read an unrequested custom uniform'); }});
    f.renderer.renderBufferDirect(f.camera, null, direct.geometry, material, direct, null);
    f.renderer.copyFramebufferToTexture(image);
  };
  f.renderer.render(scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(callbacks, 1); assert.deepEqual(direct.modelViewMatrix.toArray(), expected);
  const copyAt = f.events.indexOf(blits(f)[0]), before = f.events.slice(0, copyAt).filter(event => event.draws.some(value => value.indexed));
  assert.equal(before.length, 3); assert.ok(before.every(event => event.draws.length === 1));
  assert.equal(f.events.slice(copyAt + 1).filter(event => event.draws.some(value => value.indexed)).length, 1);
  const snapshots = f.device.snapshots.flat().filter(value => value.pipeline.fragment.module.code.includes('tint'));
  const bytes = snapshots.map(value => {
    const {group, offsets} = value.groups.get(0), {buffer, size} = group.entries[0].resource;
    return value.contents.get(buffer).slice(offsets[0], offsets[0] + size);
  });
  assert.equal(bytes.length, 3); assert.notDeepEqual(bytes[0], bytes[1], 'requested direct uniforms publish their new values');
  assert.deepEqual(bytes[1], bytes[2], 'unrequested custom edits retain the preceding uploaded values');
});

test('a later callback exception preserves every direct draw already submitted', async t => {
  const f = await fixture(t), scene = f.scene(new T.MeshBasicMaterial()), material = directMaterial(t), failure = Error('after direct draws');
  const direct = new T.Mesh(scene.children[0].geometry, material);
  scene.children[0].onBeforeRender = () => {
    f.renderer.renderBufferDirect(f.camera, null, direct.geometry, material, direct, null);
    throw failure;
  };
  await f.renderer.compileAsync(scene, f.camera);
  assert.throws(() => f.renderer.render(scene, f.camera), error => error === failure);
  assert.equal(f.events.filter(event => event.draws.some(value => value.indexed)).length, 1);
  assert.equal(f.renderer._programDrawContext, undefined);
});

test('queued consumers retain the framebuffer generation preceding a later sampler and version change', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(8, 6);
  const producer = f.scene(new T.MeshBasicMaterial()), consumer = f.scene(new T.MeshBasicMaterial({map: image}));
  // The first cold scene preparation holds this ordinary render/copy FIFO.
  f.renderer.render(producer, f.camera);
  f.renderer.copyFramebufferToTexture(image);
  f.renderer.render(consumer, f.camera);
  image.magFilter = T.LinearFilter; image.needsUpdate = true;
  f.renderer.copyFramebufferToTexture(image);
  f.renderer.render(consumer, f.camera);
  await f.renderer.waitForGPU();
  const [first, second] = blits(f); assert.equal(blits(f).length, 2);
  const old = first.desc.colorAttachments[0].view.texture, next = second.desc.colorAttachments[0].view.texture;
  assert.notEqual(old, next);
  const consumes = (events, texture) => events.some(event => event.draws.some(value => value.indexed &&
    [...value.groups.values()].some(({group}) => group.entries.some(entry => entry.resource?.texture === texture))));
  const a = f.events.indexOf(first), b = f.events.indexOf(second);
  assert.ok(consumes(f.events.slice(a + 1, b), old), 'earlier accepted draw reads its retained allocation');
  assert.ok(!consumes(f.events.slice(a + 1, b), next));
  assert.ok(consumes(f.events.slice(b + 1), next), 'later draw reads the requested replacement');
  assert.equal(f.renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('ordinary rendering does not inspect unrelated material custom data', async t => {
  const f = await fixture(t), material = new T.MeshBasicMaterial();
  Object.defineProperty(material.userData, 'unrelated', {enumerable: true, get() { assert.fail('source rendering never reads this property'); }});
  const scene = f.scene(material); f.renderer.render(scene, f.camera); await f.renderer.waitForGPU();
  assert.ok(f.events.some(event => event.draws.some(value => value.indexed)));
  assert.equal(f.renderer._sampledTextureInputs, undefined);
});

test('requested mip initialization stays after prior queued copies and before its own copy', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(8, 8);
  image.generateMipmaps = true; image.minFilter = T.LinearMipmapLinearFilter; image.needsUpdate = true;
  const producer = f.scene(new T.MeshBasicMaterial());
  f.renderer.render(producer, f.camera);
  f.renderer.copyFramebufferToTexture(image);
  image.needsUpdate = true;
  f.renderer.copyFramebufferToTexture(image);
  await f.renderer.waitForGPU();
  const labels = f.events.map(event => event.label).filter(label => label?.includes('framebuffer'));
  assert.deepEqual(labels, [...Array(3).fill('f3d-webgl-framebuffer-initialize'), 'f3d-public-framebuffer-copy',
    ...Array(3).fill('f3d-webgl-framebuffer-initialize'), 'f3d-public-framebuffer-copy']);
  assert.equal(f.renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('source-invalid format and storage encoding copies fail instead of introducing a conversion', async t => {
  const f = await fixture(t), image = new T.FramebufferTexture(4, 4); image.type = T.HalfFloatType;
  assert.throws(() => f.renderer.copyFramebufferToTexture(image), /matching normalized or floating-point format classes/);
  for (const sourceColorSpace of [T.NoColorSpace, T.SRGBColorSpace]) {
    const target = new T.WebGLRenderTarget(8, 8), destination = new T.FramebufferTexture(4, 4);
    target.texture.colorSpace = sourceColorSpace;
    destination.colorSpace = sourceColorSpace === T.SRGBColorSpace ? T.NoColorSpace : T.SRGBColorSpace;
    t.after(() => { target.dispose(); destination.dispose(); });
    f.renderer.setRenderTarget(target);
    assert.throws(() => f.renderer.copyFramebufferToTexture(destination), /matching storage color encoding/);
  }
  assert.equal(blits(f).length, 0);
});
