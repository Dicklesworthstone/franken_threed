/** Source 3D targets through the public facade and native command owner.
 * Record attachments, source WGSL, bindings and queued values; these tests do
 * not execute shaders or claim rasterized pixel equivalence.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import * as GL from '../../upstream/three.js/build/three.module.js';
import {createWebGPURendererClass, createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t, webgl = false) {
  const device = geometryDevice(), textures = [], captures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.createTexture = desc => {
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
      createView(options = {}) {
        if (texture.dimension === '3d') {
          assert.equal(options.dimension ?? '3d', '3d', 'a volume cannot expose a 2D slice view');
          assert.equal(options.baseArrayLayer ?? 0, 0);
        }
        return {texture, dimension: texture.dimension, ...options};
      }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createRenderPipeline = desc => {
    const pipeline = {...desc, getBindGroupLayout: () => ({})}; device.pipelines.push(pipeline); return pipeline;
  };
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const Type = webgl ? createWebGLRendererClass(GL) : Renderer;
  const renderer = new Type({canvas, device}); renderer.outputColorSpace = webgl ? GL.SRGBColorSpace : T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const material = webgl ? new GL.MeshBasicMaterial({color: 0x20609f}) : new T.MeshBasicNodeMaterial();
  if (!webgl) material.colorNode = N.vec3(.125, .375, .625);
  const mesh = new T.Mesh(new T.PlaneGeometry(), material); mesh.frustumCulled = false; scene.add(mesh);
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  return {renderer, device, scene, camera, material, mesh, textures, captures};
}
function target(t, options = {}, Type = T.RenderTarget3D) {
  const value = new Type(8, 6, 5, options); t.after(() => value.dispose()); return value;
}
const drawPasses = device => device.passes.filter(pass => pass.draws.length);
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const textureEntries = draw => entries(draw).filter(entry => entry.resource?.texture);
const layouts = draw => draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries);
const volumeDraws = device => nativeDraws(device).filter(draw => /texture_3d/.test(draw.pipeline.fragment.module.code));
function assertSource(f, descriptor) {
  const source = new T.WGSLNodeBuilder(f.mesh, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
}

// An ordinary PassNode sizes a 2D output. This source subclass owns the volume
// extent and slice loop; its original PassNode texture methods still rotate the
// complete current/previous images once per NodeFrame update.
class VolumeHistoryPass extends T.PassNode {
  constructor(scene, camera, depth = 5) {
    super(T.PassNode.COLOR, scene, camera, {samples: 0, depthBuffer: false});
    this.slices = depth; this.updates = 0; this.levels = 3;
    this.renderTarget = new T.RenderTarget3D(8, 4, depth, {type: T.HalfFloatType, depthBuffer: false});
    this.renderTarget.texture.name = 'output';
    this.renderTarget.texture.mipmaps = [{}, {}, {}];
    this._textures = {output: this.renderTarget.texture};
    this.getPreviousTextureNode();
    this.currentSample = N.texture3D(this.getTexture('output'), N.vec3(.5)).level(1);
    this.previousSample = N.texture3D(this.getPreviousTexture('output'), N.vec3(.5)).level(1);
  }
  setSize(width, height) { this.renderTarget.setSize(width, height, this.slices); }
  setup(builder) { super.setup(builder); return this.currentSample; }
  updateBefore(frame) {
    this.previousSample.value = this.getTexture('output');
    super.updateBefore(frame);
    this.currentSample.value = this.getTexture('output');
    const r = frame.renderer, target = r.getRenderTarget(), face = r.getActiveCubeFace(), mip = r.getActiveMipmapLevel();
    try {
      for (let level = 0; level < this.levels; level++)
        for (let slice = level === 0 ? 1 : 0; slice < Math.max(1, this.slices >> level); slice++) {
          r.setRenderTarget(this.renderTarget, slice, level); r.render(this.scene, this.camera);
        }
    } finally { r.setRenderTarget(target, face, mip); }
    this.updates++;
  }
}

test('RenderPipeline executes whole-volume history feedback through original PassNode rotation and 3D sampling', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = new VolumeHistoryPass(f.scene, f.camera);
  const weight = N.uniform(.5).setName('volumeHistoryFeedback');
  f.material.colorNode = pass.previousSample.rgb.mul(weight).add(N.vec3(.125, .25, .375));
  const pipeline = new T.RenderPipeline(f.renderer, pass);
  t.after(() => { pipeline.dispose(); pass.dispose(); });
  const generations = [];
  for (let frame = 0; frame < 3; frame++) {
    const before = f.device.passes.length;
    f.renderer._nodes.nodeFrame.update(); pipeline.render(); await f.renderer.waitForGPU();
    const passes = f.device.passes.slice(before).filter(p => p.draws.length), producer = passes.filter(p =>
      p.desc.colorAttachments[0].view.texture.dimension === '3d');
    assert.deepEqual(producer.map(p => [p.desc.colorAttachments[0].view.baseMipLevel,
      p.desc.colorAttachments[0].depthSlice]), [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [2, 0]]);
    const destination = producer[0].desc.colorAttachments[0].view.texture;
    assert.ok(producer.every(p => p.desc.colorAttachments[0].view.texture === destination));
    for (const p of producer) {
      const draw = p.draws[0], input = textureEntries(draw).find(entry => entry.resource.dimension === '3d').resource;
      assert.notEqual(input.texture, destination, 'every slice samples the other complete temporal volume');
      if (frame) assert.equal(input.texture, generations[frame - 1]);
      assert.equal(input.mipLevelCount, 3); assert.match(draw.pipeline.fragment.module.code, /textureSampleLevel/);
    }
    const output = passes.at(-1).draws[0];
    assert.equal(textureEntries(output)[0].resource.texture, destination);
    assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer._nodeCaptureBytes, 0);
    generations.push(destination);
  }
  assert.equal(pass.updates, 3); assert.equal(generations[0], generations[2]);
  assert.notEqual(generations[0], generations[1]);
  assert.equal(f.textures.filter(texture => texture.dimension === '3d').length, 2);
  assert.equal(pass.getPreviousTexture('output'), pass.previousSample.value);
  assert.equal(pass.getTexture('output'), pass.currentSample.value);
  assert.equal(pass.getTextureNode().value, pass.getTexture('output'));
  assert.equal(pass.getPreviousTextureNode().value, pass.getPreviousTexture('output'));
});

test('queued RenderPipeline volume histories retain XYZ generations across source effect resizes', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = new VolumeHistoryPass(f.scene, f.camera), pipeline = new T.RenderPipeline(f.renderer, pass);
  f.material.colorNode = pass.previousSample.rgb.mul(.5).add(N.vec3(.125));
  t.after(() => { pipeline.dispose(); pass.dispose(); });
  pipeline.render();
  f.renderer.setSize(20, 12, false); pass.slices = 2;
  f.renderer._nodes.nodeFrame.update(); pipeline.render();
  // A Z-only resize has its own dispose event even when the canvas is unchanged.
  pass.slices = 7; f.renderer._nodes.nodeFrame.update(); pipeline.render();
  await f.renderer.waitForGPU();
  const consumers = drawPasses(f.device).filter(p => p.desc.colorAttachments[0].view.texture.dimension !== '3d');
  assert.deepEqual(consumers.map(p => {
    const volume = textureEntries(p.draws[0])[0].resource.texture;
    return [volume.width, volume.height, volume.depthOrArrayLayers];
  }), [[32, 32, 5], [20, 12, 2], [20, 12, 7]]);
  const producers = drawPasses(f.device).filter(p => p.desc.colorAttachments[0].view.texture.dimension === '3d');
  assert.deepEqual([5, 2, 7].map(depth => producers.filter(p =>
    p.desc.colorAttachments[0].view.texture.depthOrArrayLayers === depth).length), [8, 4, 11]);
  for (const p of producers) {
    const output = p.desc.colorAttachments[0].view.texture;
    const input = textureEntries(p.draws[0])[0].resource.texture;
    assert.notEqual(input, output);
    assert.deepEqual([input.width, input.height, input.depthOrArrayLayers],
      [output.width, output.height, output.depthOrArrayLayers]);
  }
  assert.equal(pass.updates, 3); assert.equal(pass.renderTarget.depth, 7);
  assert.equal(pass.getPreviousTexture('output').image.depth, 7);
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('borrowed volume history pairs preserve old XYZ mip layouts until both leases and the GPU fence finish', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = new VolumeHistoryPass(f.scene, f.camera, 9), target = pass.renderTarget, owner = f.renderer._targets;
  t.after(() => pass.dispose()); target.setSize(1, 1, 9);
  const sources = [pass.getTexture('output'), pass.getPreviousTexture('output')];
  for (const texture of sources) texture.mipmaps = [{}, {}, {}, {}];
  owner.registerNativePass(pass);
  const oldBindings = sources.map(texture => owner.captureNativeTexture(texture));
  const releases = oldBindings.map(binding => owner.retainGeneration(binding.entry));
  let oldImages;
  owner.withNativeTextures(oldBindings, values => { oldImages = sources.map(texture => values.get(texture)); });
  assert.equal(owner.allocatedBytes, 16 * 8 * 2);
  let finish;
  const completed = f.device.queue.onSubmittedWorkDone, pending = new Promise(resolve => { finish = resolve; });
  f.device.queue.onSubmittedWorkDone = () => pending;
  let next;
  try {
    for (const texture of sources) texture.mipmaps = [{}, {}];
    owner.withNativeEffectTargets([target], () => target.setSize(1, 1, 2));
    owner.registerNativePass(pass);
    next = sources.map(texture => owner.captureNativeTexture(texture));
    owner.withNativeTextures(next, values => {
      for (const texture of sources) {
        const image = values.get(texture);
        assert.equal(image.dimension, '3d'); assert.equal(image.depthOrArrayLayers, 2); assert.equal(image.mipLevelCount, 2);
      }
    });
    owner.withNativeTextures(oldBindings, values => {
      assert.deepEqual(sources.map(texture => values.get(texture)), oldImages);
      assert.ok(oldImages.every(image => image.depthOrArrayLayers === 9 && image.mipLevelCount === 4));
    });
    assert.equal(owner.allocatedBytes, (16 + 3) * 8 * 2);
    releases[0](); releases[1]();
    assert.ok(oldImages.every(image => !image.destroyed));
  } finally {
    releases[0](); releases[1](); f.device.queue.onSubmittedWorkDone = completed; finish();
  }
  await owner.whenIdle(); await owner.whenIdle();
  assert.ok(oldImages.every(image => image.destroyed)); assert.equal(owner.allocatedBytes, 3 * 8 * 2);
  target.dispose();
  assert.throws(() => owner.withNativeTexture(next[0], () => {}), {code: 'THREE_TARGET_STALE'});
});

test('queued volume renders retain slices, including zero, with one shared 2D depth attachment', async t => {
  const f = fixture(t), volume = target(t); await f.renderer.init();
  for (const slice of [4, 0, 2]) { f.renderer.setRenderTarget(volume, slice); f.renderer.render(f.scene, f.camera); }
  f.renderer.setRenderTarget(volume, 1); await f.renderer.waitForGPU();
  const passes = drawPasses(f.device), colors = passes.map(pass => pass.desc.colorAttachments[0]);
  assert.deepEqual(colors.map(color => color.depthSlice), [4, 0, 2]);
  assert.ok(colors.every(color => color.view.dimension === '3d' && color.view.mipLevelCount === 1));
  assert.equal(new Set(colors.map(color => color.view.texture)).size, 1);
  const native = colors[0].view.texture;
  assert.deepEqual([native.dimension, native.width, native.height, native.depthOrArrayLayers], ['3d', 8, 6, 5]);
  const depths = passes.map(pass => pass.desc.depthStencilAttachment);
  assert.equal(new Set(depths.map(depth => depth.view.texture)).size, 1);
  assert.ok(depths.every(depth => depth.depthSlice === undefined && depth.view.dimension === '2d' && depth.view.texture.depthOrArrayLayers === 1));
  assert.equal(f.renderer.getActiveCubeFace(), 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('authored volume mips select shrinking Z slices and independent private depth extents', async t => {
  const f = fixture(t), volume = target(t); volume.texture.mipmaps = [{}, {}, {}, {}];
  const viewport = volume.viewport.clone(); await f.renderer.init();
  for (const [slice, mip] of [[1, 1], [0, 2], [0, 3]]) {
    f.renderer.setRenderTarget(volume, slice, mip); f.renderer.render(f.scene, f.camera);
  }
  await f.renderer.waitForGPU();
  const passes = drawPasses(f.device), colors = passes.map(pass => pass.desc.colorAttachments[0]);
  assert.deepEqual(colors.map(color => [color.depthSlice, color.view.baseMipLevel, color.view.mipLevelCount]), [[1, 1, 1], [0, 2, 1], [0, 3, 1]]);
  assert.ok(colors.every(color => color.view.dimension === '3d' && color.view.texture.mipLevelCount === 4));
  assert.deepEqual(passes.map(pass => {
    const texture = pass.desc.depthStencilAttachment.view.texture;
    return [texture.dimension, texture.width, texture.height, texture.depthOrArrayLayers];
  }), [['2d', 4, 3, 1], ['2d', 2, 1, 1], ['2d', 1, 1, 1]]);
  assert.equal(f.renderer.getActiveMipmapLevel(), 3); assert.deepEqual(volume.viewport, viewport);
});

test('WebGL shaders consume rendered volumes with one persistent 3D binding and queued Z coordinates', async t => {
  const f = fixture(t, true), volume = target(t, {depthBuffer: false}, GL.WebGL3DRenderTarget);
  volume.texture.wrapR = GL.RepeatWrapping; await f.renderer.init();
  f.renderer.setRenderTarget(volume, 4); await f.renderer.renderAsync(f.scene, f.camera);
  const native = drawPasses(f.device)[0].desc.colorAttachments[0].view.texture;
  const material = new GL.ShaderMaterial({glslVersion: GL.GLSL3,
    uniforms: {volumeData: {value: volume.texture}, sampleDepth: {value: .9}},
    vertexShader: 'out vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
    fragmentShader: 'precision highp sampler3D;uniform sampler3D volumeData;uniform float sampleDepth;in vec2 vUv;out vec4 color;void main(){color=texture(volumeData,vec3(vUv,sampleDepth));}'});
  const consumer = new GL.Mesh(new GL.PlaneGeometry(2, 2), material);
  t.after(() => { material.dispose(); consumer.geometry.dispose(); });
  f.renderer.setRenderTarget(null); await f.renderer.compileAsync(consumer, f.camera);
  f.renderer.render(consumer, f.camera); material.uniforms.sampleDepth.value = .1;
  f.renderer.render(consumer, f.camera); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => /volumeData/.test(draw.pipeline.fragment.module.code));
  assert.equal(draws.length, 2);
  const views = draws.map(draw => textureEntries(draw)[0].resource);
  assert.equal(views[0], views[1]); assert.equal(views[0].dimension, '3d'); assert.notEqual(views[0].texture, native);
  assert.equal(views[0].texture.dimension, '3d'); assert.equal(views[0].texture.depthOrArrayLayers, 5);
  const oriented = f.device.passes.filter(pass => pass.desc.colorAttachments[0].view.texture === views[0].texture);
  assert.deepEqual(oriented.map(pass => pass.desc.colorAttachments[0].depthSlice), [0, 1, 2, 3, 4]);
  assert.deepEqual(oriented.map(pass => pass.draws[0].args), [0, 1, 2, 3, 4].map(slice => [3, 1, 0, slice]));
  const uniforms = draws.map(draw => [...draw.contents.values()].map(bytes => Array.from(bytes)));
  assert.notDeepEqual(uniforms[0], uniforms[1], 'queued normalized Z coordinates retain call-time values');
  assert.equal(f.renderer._targets.capture(volume).copies, 1);
  f.renderer.setRenderTarget(volume, 1); await f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.setRenderTarget(null); await f.renderer.renderAsync(consumer, f.camera);
  assert.equal(f.renderer._targets.capture(volume).copies, 2); assert.equal(volume.texture.renderTarget, null);
});

test('WebGL volume mipmaps retain each queued generation flag and never regenerate on clear', async t => {
  const f = fixture(t, true), volume = target(t, {depthBuffer: false, generateMipmaps: true}, GL.WebGL3DRenderTarget);
  await f.renderer.init();
  for (const [slice, enabled] of [[4, true], [0, false], [2, true]]) {
    volume.texture.generateMipmaps = enabled;
    f.renderer.setRenderTarget(volume, slice); f.renderer.render(f.scene, f.camera);
  }
  volume.texture.generateMipmaps = false;
  await f.renderer.waitForGPU();
  const entry = f.renderer._targets.capture(volume);
  assert.equal(entry.storage.diagnostics.mipGenerations, 2);
  assert.equal(entry.storage.mipLevelCount, 4);
  const generated = f.device.passes.filter(pass => pass.draws.some(draw => /input\.slice/.test(draw.pipeline.fragment.module.code)));
  assert.deepEqual(generated.map(pass => [pass.desc.colorAttachments[0].view.baseMipLevel,
    pass.desc.colorAttachments[0].depthSlice]), [[1, 0], [1, 1], [2, 0], [3, 0], [1, 0], [1, 1], [2, 0], [3, 0]]);
  const image = generated[0].desc.colorAttachments[0].view.texture;
  assert.ok(generated.every(pass => pass.desc.colorAttachments[0].view.texture === image));
  volume.texture.generateMipmaps = true; f.renderer.clear(); await f.renderer.waitForGPU();
  assert.equal(f.renderer._targets.capture(volume), entry);
  assert.equal(entry.storage.diagnostics.mipGenerations, 2, 'source clear leaves higher levels stale');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('MRT volume output retains the selected slice on every color attachment after renderer state restoration', async t => {
  const f = fixture(t), volume = target(t, {count: 2, type: T.HalfFloatType});
  volume.textures[0].name = 'output'; volume.textures[1].name = 'auxiliary';
  await f.renderer.init(); f.renderer.setRenderTarget(volume, 3);
  f.renderer.setMRT(N.mrt({output: N.output, auxiliary: N.vec4(.2, .4, .6, 1)}));
  const rendering = f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.setRenderTarget(null); f.renderer.setMRT(null); await rendering; await f.renderer.waitForGPU();
  const [pass] = drawPasses(f.device), colors = pass.desc.colorAttachments;
  assert.equal(colors.length, 2); assert.deepEqual(colors.map(color => color.depthSlice), [3, 3]);
  assert.ok(colors.every(color => color.view.dimension === '3d' && color.view.texture.format === 'rgba16float'));
  assert.notEqual(colors[0].view.texture, colors[1].view.texture);
  const [draw] = nativeDraws(f.device);
  assert.equal(draw.pipeline.fragment.targets.length, 2); assert.match(draw.pipeline.fragment.module.code, /output\.m1/);
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer.getMRT(), null);
});

for (const requestedBy of ['option', 'depth image'])
test(`volume depth selected by ${requestedBy} uses an independent 2D array and source comparison sampling`, async t => {
  const f = fixture(t), volume = target(t, {useArrayDepthTexture: requestedBy === 'option'});
  const depth = volume.depthTexture = new T.DepthTexture(8, 6);
  if (requestedBy === 'depth image') depth.image.depth = 5;
  depth.compareFunction = T.LessEqualCompare;
  await f.renderer.init(); f.renderer.setRenderTarget(volume, 4); await f.renderer.renderAsync(f.scene, f.camera);
  const pass = drawPasses(f.device)[0], attachment = pass.desc.depthStencilAttachment;
  assert.equal(pass.desc.colorAttachments[0].depthSlice, 4); assert.equal(attachment.depthSlice, undefined);
  assert.deepEqual([attachment.view.dimension, attachment.view.baseArrayLayer, attachment.view.arrayLayerCount], ['2d', 4, 1]);
  assert.deepEqual([attachment.view.texture.dimension, attachment.view.texture.depthOrArrayLayers], ['2d', 5]);
  f.material.colorNode = N.vec3(N.texture(depth, N.vec2(.25, .75)).depth(4).compare(.5));
  f.renderer.setRenderTarget(null); await f.renderer.renderAsync(f.scene, f.camera);
  const draw = nativeDraws(f.device).at(-1), view = textureEntries(draw)[0].resource;
  assert.match(draw.pipeline.fragment.module.code, /texture_depth_2d_array/);
  assert.match(draw.pipeline.fragment.module.code, /textureSampleCompare/);
  assert.deepEqual(layouts(draw).filter(entry => entry.texture).map(entry => entry.texture),
    [{sampleType: 'depth', viewDimension: '2d-array', multisampled: false}]);
  assert.deepEqual([view.dimension, view.baseArrayLayer, view.arrayLayerCount, view.aspect], ['2d-array', 0, 5, 'depth-only']);
  assert.equal(view.texture, attachment.view.texture); assert.equal(depth.isArrayTexture, true);
});

for (const Type of [T.RenderTarget3D, T.WebGL3DRenderTarget])
test(`${Type.name} sampling borrows the rendered volume and preserves queued source coordinates`, async t => {
  const f = fixture(t), volume = target(t, {minFilter: T.LinearFilter, magFilter: T.LinearFilter}, Type);
  volume.texture.wrapR = T.RepeatWrapping;
  const sourceOwner = volume.texture.renderTarget;
  await f.renderer.init(); f.renderer.setRenderTarget(volume, 3); await f.renderer.renderAsync(f.scene, f.camera);
  const rendered = drawPasses(f.device)[0].desc.colorAttachments[0].view.texture;
  const uvw = N.uniform(new T.Vector3(.25, .5, .75)).setName('volumeCoordinates');
  f.material.colorNode = N.texture3D(volume.texture, uvw).rgb; f.renderer.setRenderTarget(null);
  f.renderer.render(f.scene, f.camera);
  const first = f.captures.at(-1).draws[0].nodeUniforms.map(bytes => bytes.slice());
  uvw.value.set(.75, .25, .5); f.renderer.render(f.scene, f.camera);
  const second = f.captures.at(-1).draws[0].nodeUniforms.map(bytes => bytes.slice());
  uvw.value.set(99, 99, 99); await f.renderer.waitForGPU();
  assert.notDeepEqual(first, second); assert.deepEqual(f.captures.at(-2).draws[0].nodeUniforms, first);
  assertSource(f, f.captures.at(-1).draws[0].binding.nodeProgram);
  const draws = volumeDraws(f.device); assert.equal(draws.length, 2);
  for (const draw of draws) {
    assert.match(draw.pipeline.fragment.module.code, /texture_3d<f32>/);
    assert.match(draw.pipeline.fragment.module.code, /textureSample\(/);
    const view = textureEntries(draw)[0].resource;
    assert.equal(view.texture, rendered); assert.equal(view.dimension, '3d');
    assert.deepEqual(layouts(draw).filter(entry => entry.texture).map(entry => entry.texture),
      [{sampleType: 'float', viewDimension: '3d', multisampled: false}]);
    assert.equal(entries(draw).find(entry => entry.resource?.addressModeW).resource.addressModeW, 'repeat');
  }
  assert.equal(f.textures.filter(texture => texture.dimension === '3d').length, 1);
  assert.equal(volume.texture.image.data, null); assert.equal(volume.texture.renderTarget, sourceOwner);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('explicit volume mip sampling binds all authored levels from the existing native image', async t => {
  const f = fixture(t), volume = target(t, {minFilter: T.LinearMipmapLinearFilter, magFilter: T.LinearFilter});
  volume.texture.mipmaps = [{}, {}, {}, {}];
  await f.renderer.init(); f.renderer.setRenderTarget(volume, 1, 1); await f.renderer.renderAsync(f.scene, f.camera);
  const rendered = drawPasses(f.device)[0].desc.colorAttachments[0].view.texture;
  f.material.colorNode = N.texture3D(volume.texture, N.vec3(.25, .5, .75)).level(1).rgb;
  f.renderer.setRenderTarget(null); await f.renderer.renderAsync(f.scene, f.camera);
  const [draw] = volumeDraws(f.device), view = textureEntries(draw)[0].resource;
  assert.match(draw.pipeline.fragment.module.code, /textureSampleLevel/);
  assert.equal(view.texture, rendered); assert.deepEqual([view.dimension, view.baseMipLevel, view.mipLevelCount], ['3d', 0, 4]);
  assert.equal(entries(draw).find(entry => entry.resource?.addressModeW).resource.lodMaxClamp, 3);
});

test('first-use ordinary volume depth sampling stays 2D while its color storage is 3D', async t => {
  const f = fixture(t), volume = target(t); volume.depthTexture = new T.DepthTexture(8, 6);
  f.material.colorNode = N.vec3(N.texture(volume.depthTexture, N.vec2(.25, .75)).level(0));
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const [draw] = nativeDraws(f.device), view = textureEntries(draw)[0].resource;
  assert.match(draw.pipeline.fragment.module.code, /texture_depth_2d\s*;/);
  assert.match(draw.pipeline.fragment.module.code, /textureLoad/);
  assert.deepEqual(layouts(draw).filter(entry => entry.texture).map(entry => entry.texture),
    [{sampleType: 'depth', viewDimension: '2d', multisampled: false}]);
  assert.deepEqual([view.dimension, view.texture.depthOrArrayLayers, view.aspect], ['2d', 1, 'depth-only']);
  assert.equal(f.textures.filter(texture => texture.dimension === '3d').length, 1);
  assert.equal(volume.depthTexture.isArrayTexture, false);
});

test('native compute loads the rendered volume through the same 3D texture ownership', async t => {
  const f = fixture(t), volume = target(t), passes = [];
  f.device.createComputePipeline = descriptor => ({...descriptor,
    getBindGroupLayout: index => descriptor.layout.bindGroupLayouts[index]});
  f.device.createComputePipelineAsync = descriptor => Promise.resolve(f.device.createComputePipeline(descriptor));
  const createEncoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const encoder = createEncoder(), commands = [];
    return {...encoder, beginComputePass() {
      const pass = {draws: [], groups: new Map(), dispatches: []}; passes.push(pass); commands.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; },
        setBindGroup(index, group) { pass.groups.set(index, group); },
        dispatchWorkgroups(...size) { pass.dispatches.push(size); }, end() {}};
    }, finish() { return [...encoder.finish(), ...commands]; }};
  };
  await f.renderer.init(); f.renderer.setRenderTarget(volume, 4); await f.renderer.renderAsync(f.scene, f.camera);
  const rendered = drawPasses(f.device)[0].desc.colorAttachments[0].view.texture;
  const output = N.storage(new T.StorageBufferAttribute(1, 1), 'float', 1);
  const value = N.texture3DLoad(volume.texture, N.ivec3(0, 0, 4));
  const kernel = N.Fn(() => { output.element(0).assign(value.r); })().compute(1);
  f.renderer.setRenderTarget(null); await f.renderer.computeAsync(kernel); await f.renderer.waitForGPU();
  assert.equal(passes.length, 1); assert.deepEqual(passes[0].dispatches, [[1, 1, 1]]);
  const {pipeline, groups} = passes[0];
  assert.match(pipeline.compute.module.code, /texture_3d<f32>/); assert.match(pipeline.compute.module.code, /textureLoad\(/);
  const layout = pipeline.layout.bindGroupLayouts.flatMap(group => group.entries).filter(entry => entry.texture);
  assert.deepEqual(layout.map(entry => entry.texture), [{sampleType: 'float', viewDimension: '3d', multisampled: false}]);
  const view = [...groups.values()].flatMap(group => group.entries).find(entry => entry.resource?.texture).resource;
  assert.equal(view.dimension, '3d'); assert.equal(view.texture, rendered);
  assert.equal(f.textures.filter(texture => texture.dimension === '3d').length, 1);
});

test('source volume resize preserves public textures while replacing color and per-slice depth storage', async t => {
  const f = fixture(t), volume = target(t, {useArrayDepthTexture: true}), sourceColor = volume.texture;
  volume.depthTexture = new T.DepthTexture(8, 6); const sourceDepth = volume.depthTexture;
  await f.renderer.init(); f.renderer.setRenderTarget(volume, 4); await f.renderer.renderAsync(f.scene, f.camera);
  const first = drawPasses(f.device)[0].desc, oldColor = first.colorAttachments[0].view.texture, oldDepth = first.depthStencilAttachment.view.texture;
  volume.setSize(4, 3, 2); f.renderer.setRenderTarget(volume, 1); await f.renderer.renderAsync(f.scene, f.camera); await f.renderer.waitForGPU();
  const next = drawPasses(f.device).at(-1).desc, color = next.colorAttachments[0].view.texture, depth = next.depthStencilAttachment.view.texture;
  assert.deepEqual([color.dimension, color.width, color.height, color.depthOrArrayLayers], ['3d', 4, 3, 2]);
  assert.deepEqual([depth.dimension, depth.width, depth.height, depth.depthOrArrayLayers], ['2d', 4, 3, 2]);
  assert.equal(next.colorAttachments[0].depthSlice, 1); assert.equal(next.depthStencilAttachment.view.baseArrayLayer, 1);
  assert.equal(volume.texture, sourceColor); assert.equal(volume.depthTexture, sourceDepth);
  assert.equal(volume.texture.image.depth, 2); assert.equal(volume.depthTexture.image.depth, 2);
  assert.notEqual(color, oldColor); assert.notEqual(depth, oldDepth);
  assert.equal(oldColor.destroyed, true); assert.equal(oldDepth.destroyed, true); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('invalid volume slice and mip selections fail before native allocation or queued work', async t => {
  const f = fixture(t), volume = target(t); volume.texture.mipmaps = [{}, {}, {}, {}];
  for (const [slice, mip] of [[-1, 0], [5, 0], [.5, 0], [NaN, 0], [Infinity, 0], [2, 1], [1, 2], [1, 3]])
    assert.throws(() => f.renderer.setRenderTarget(volume, slice, mip), error => error.code?.endsWith('_FACE'));
  for (const mip of [-1, 4, .5, NaN])
    assert.throws(() => f.renderer.setRenderTarget(volume, 0, mip), error => error.code?.endsWith('_MIP'));
  assert.equal(f.renderer.setOutputRenderTarget(volume), undefined);
  assert.equal(f.renderer.getOutputRenderTarget(), volume);
  assert.equal(f.textures.length, 0); assert.equal(f.device.buffers.length, 0); assert.equal(f.renderer._pending.length, 0);
});
