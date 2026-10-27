/** The pinned VXGI addon owns voxelization, lighting and cone tracing. This
 * recording device proves native command/resource ordering; it does not execute
 * the shaders or make a pixel-equivalence claim. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {vxgi} from '../../upstream/three.js/examples/jsm/lighting/vxgi/VXGINode.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL;
const resources = groups => [...groups.values()].flatMap(value => (value.group ?? value).entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
const attachments = pass => pass.desc.colorAttachments.map(value => (value.resolveTarget ?? value.view).texture);

function fixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000, maxColorAttachments: 8,
    maxColorAttachmentBytesPerSample: 32, maxStorageTexturesPerShaderStage: 8});
  device.features = new Set(); device.textures = []; device.computePipelines = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = () => {};
  device.createComputePipeline = descriptor => {
    const pipeline = {...descriptor, getBindGroupLayout: index => ({index})};
    device.computePipelines.push(pipeline); return pipeline;
  };
  device.createComputePipelineAsync = descriptor => Promise.resolve(device.createComputePipeline(descriptor));
  const createEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = descriptor => {
    const encoder = createEncoder(descriptor), passes = [];
    return {...encoder, beginComputePass(desc) {
      const pass = {desc, draws: [], groups: new Map(), dispatches: []}; passes.push(pass);
      return {setPipeline(pipeline) { pass.pipeline = pipeline; },
        setBindGroup(index, group) { pass.groups.set(index, group); },
        dispatchWorkgroups(...size) { pass.dispatches.push(size); }, end() {}};
    }, finish() { return [...encoder.finish(), ...passes]; }};
  };
  const submit = device.queue.submit.bind(device.queue);
  device.queue.submit = commandBuffers => {
    for (const pass of commandBuffers.flat()) if (pass.dispatches) {
      pass.uniforms = [...pass.groups.values()].flatMap(group => group.entries)
        .map(entry => entry.resource).filter(resource => resource?.buffer?.usage & 64)
        .map(({buffer, offset = 0, size = buffer.size - offset}) =>
          Array.from(new Float32Array(buffer.data, offset, size / 4)));
    }
    submit(commandBuffers);
  };
  let configured;
  const canvas = {width: 32, height: 24, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configured.format, usage: 16}); },
  })};
  const renderer = new (createWebGPURendererClass(T))({canvas, device});
  t.after(() => renderer.dispose());
  return {device, renderer};
}

test('retained VXGI computes volume lighting before its GI-context beauty across updates and resize', async t => {
  const {device, renderer} = fixture(t); await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20);
  camera.position.set(2, 1.5, 4); camera.lookAt(0, 0, 0);
  const geometry = new T.BoxGeometry(), material = new T.MeshStandardNodeMaterial({color: 0x3366aa, roughness: .7});
  const mesh = new T.Mesh(geometry, material); scene.add(mesh);
  const light = new T.PointLight(0xffddaa, 15, 12, 2); light.position.set(-1, 2, 2); scene.add(light);
  scene.add(new T.AmbientLight(0xffffff, .2));

  // Use the pinned example's packed-normal/depth prepass and material context.
  const prepass = N.pass(scene, camera); prepass.transparent = false;
  prepass.setMRT(N.mrt({output: N.packNormalToRGB(N.normalView), velocity: N.velocity}));
  prepass.getTexture('output').type = T.UnsignedByteType;
  const normal = N.sample(uv => N.unpackRGBToNormal(prepass.getTextureNode().sample(uv)));
  const effect = vxgi(prepass.getTextureNode('depth'), normal, scene, camera, 16);
  effect.useTemporalFiltering = false;
  // Stable source bounds keep geometry invalidation independent of allocation
  // size changes; the fourth frame separately exercises render-target resizing.
  effect.volume.bounds.set(new T.Vector3(-2, -2, -2), new T.Vector3(2, 2, 2));
  const beauty = N.pass(scene, camera);
  beauty.contextNode = N.builtinGIContext(effect.getAONode().sample(N.screenUV).r,
    effect.getGINode().sample(N.screenUV).rgb);
  const pipeline = new T.RenderPipeline(renderer, beauty);
  t.after(() => { pipeline.dispose(); beauty.dispose(); effect.dispose(); prepass.dispose(); material.dispose(); geometry.dispose(); });
  const originalContext = renderer.contextNode, frames = [];
  for (let frame = 0; frame < 4; frame++) {
    if (frame === 1) { light.intensity = 25; light.position.x = .75; effect.volume.lightingNeedsUpdate = true; }
    if (frame === 2) { geometry.scale(1.25, .75, 1); effect.volume.needsUpdate = true; }
    if (frame === 3) { renderer.setSize(24, 16, false); camera.aspect = 3 / 2; camera.updateProjectionMatrix(); }
    const start = device.submissions.length;
    renderer._nodes.nodeFrame.update(); pipeline.render();
    await renderer.waitForGPU();
    const commands = device.submissions.slice(start).flat(2), compute = commands.filter(pass => pass.dispatches);
    const renders = commands.filter(pass => pass.draws.length);
    const cones = renders.find(pass => attachments(pass)[0].format === 'r8unorm');
    assert.ok(cones);
    const coneInputs = resources(cones.draws[0].groups).map(view => view.texture);
    const pre = renders.find(pass => coneInputs.includes(attachments(pass)[0]));
    const lit = renders.find(pass => resources(pass.draws[0].groups)
      .some(view => view.texture === attachments(cones)[0]));
    const present = renders.at(-1);
    assert.ok(pre && cones && lit && present !== lit,
      JSON.stringify(renders.map(pass => attachments(pass).map(texture => texture.format))));
    assert.ok(commands.indexOf(pre) < commands.indexOf(cones));
    assert.ok(commands.indexOf(cones) < commands.indexOf(lit));
    assert.ok(commands.indexOf(lit) < commands.indexOf(present));
    assert.ok(compute.every(pass => commands.indexOf(pass) < commands.indexOf(cones)));
    assert.deepEqual(attachments(cones).map(texture => texture.format), ['r8unorm', 'rgba16float']);
    const beautyInputs = resources(lit.draws[0].groups).map(view => view.texture);
    assert.ok(coneInputs.includes(attachments(pre)[0]));
    assert.ok(coneInputs.includes(pre.desc.depthStencilAttachment.view.texture));
    for (const texture of attachments(cones)) assert.ok(beautyInputs.includes(texture));
    assert.ok(resources(present.draws[0].groups).some(view => view.texture === attachments(lit)[0]));
    const volumes = coneInputs.filter(texture => texture.dimension === '3d');
    assert.ok(volumes.length >= 2);
    if (compute.length) {
      const computedVolumes = new Set(compute.flatMap(pass => resources(pass.groups).map(view => view.texture)));
      assert.ok(volumes.some(texture => computedVolumes.has(texture)), 'cone tracing samples the actual compute allocations');
    }
    assert.equal(effect.volume.needsUpdate, false); assert.equal(effect.volume.lightingNeedsUpdate, false);
    assert.ok(effect.volume._triangleCount > 0);
    assert.equal(renderer.contextNode, originalContext); assert.equal(renderer.getRenderTarget(), null);
    assert.equal(renderer.getMRT(), null); assert.equal(renderer._deferredError, null);
    assert.equal(renderer._pending.length, 0); assert.equal(renderer._nodeCaptureBytes, 0);
    for (const target of [prepass.renderTarget, effect._renderTarget, beauty.renderTarget]) {
      const owner = renderer._targets.capture(target);
      assert.equal(owner.references, 0); assert.equal(owner.busy, false);
    }
    frames.push({compute: compute.length, volume: volumes[0], color: attachments(cones)[0],
      triangles: effect.volume._triangleAttribute});
  }
  assert.ok(frames[0].compute > frames[1].compute && frames[1].compute > 0);
  assert.equal(frames[2].compute, frames[0].compute); assert.equal(frames[3].compute, 0);
  assert.equal(frames[1].triangles, frames[0].triangles);
  assert.notEqual(frames[2].triangles, frames[1].triangles);
  assert.ok(frames.every(frame => frame.volume === frames[0].volume));
  assert.deepEqual(frames.map(frame => [frame.color.width, frame.color.height]), [[32, 24], [32, 24], [32, 24], [24, 16]]);
  assert.notEqual(frames[3].color, frames[2].color);

  // A public compile request reserves the renderer FIFO before its first
  // asynchronous boundary. Both already-warm source pipelines capture while
  // that request is pending, before either lighting version reaches the GPU.
  const queuedStart = device.submissions.length;
  const compilation = renderer.compileComputeAsync(effect.volume._kernels.clear);
  for (const intensity of [7.5, 19.25]) {
    light.intensity = intensity; effect.volume.lightingNeedsUpdate = true;
    renderer._nodes.nodeFrame.update(); pipeline.render();
  }
  assert.equal(device.submissions.length, queuedStart); assert.ok(renderer._pending.length > 0);
  light.intensity = 99;
  await Promise.all([compilation, renderer.waitForGPU()]);
  const queued = device.submissions.slice(queuedStart).flat(2);
  const inject = queued.flatMap((pass, index) => {
    // The source light array stores position/type, direction/range, then
    // irradiance/decay. Its red component is exactly this light's intensity.
    const lights = pass.uniforms?.find(words => words.length === effect.volume._lightsArray.length * 4 &&
      words[0] === .75 && words[1] === 2 && words[2] === 2 && words[3] === 1 && words[7] === 12 && words[11] === 2);
    return lights ? [{index, intensity: lights[8]}] : [];
  });
  const beautyIndices = queued.flatMap((pass, index) => pass.draws.some(draw =>
    resources(draw.groups).some(view => view.texture.format === 'r8unorm')) ? [index] : []);
  assert.deepEqual(inject.map(pass => pass.intensity), [7.5, 19.25]);
  assert.equal(beautyIndices.length, 2);
  assert.ok(inject[0].index < beautyIndices[0] && beautyIndices[0] < inject[1].index && inject[1].index < beautyIndices[1]);
  assert.equal(renderer._pending.length, 0); assert.equal(renderer._nodeCaptureBytes, 0);
  for (const target of [prepass.renderTarget, effect._renderTarget, beauty.renderTarget])
    assert.equal(renderer._targets.capture(target).references, 0);

  const computeSource = device.computePipelines.map(pipeline => pipeline.compute.module.code).join('\n');
  assert.match(computeSource, /atomicOr\(/); assert.match(computeSource, /texture_storage_3d/);
  assert.match(computeSource, /textureStore\(/);
  assert.ok(device.pipelines.some(pipeline => /texture_3d<f32>/.test(pipeline.fragment.module.code)));
});
