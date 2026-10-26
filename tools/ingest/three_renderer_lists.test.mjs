/** Real facade/capture/resource/scene integration with recorded WebGPU commands.
 * Render-list selection, callback timing and copies are observed; WGSL is not rasterized here. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const N = THREE.TSL;

async function fixture(t, options = {}) {
  const device = geometryDevice(), events = [], textures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.features = new Set(['float32-filterable', 'float32-blendable']);
  device.createTexture = descriptor => {
    const width = descriptor.size.width ?? descriptor.size[0], height = descriptor.size.height ?? descriptor.size[1];
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor, width, height,
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(descriptor = {}) { return {texture: this, descriptor}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = descriptor => {
    const encoder = createEncoder(), commands = [];
    return {...encoder,
      beginRenderPass(descriptor) {
        const pass = encoder.beginRenderPass(descriptor), record = device.passes.at(-1);
        commands.push(record);
        return {...pass, setViewport(...value) { record.viewport = value; pass.setViewport(...value); },
          setScissorRect(...value) { record.scissor = value; pass.setScissorRect(...value); }};
      },
      copyTextureToTexture(source, destination, size) {
        assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
        assert.equal(source.texture.sampleCount, 1); assert.equal(source.texture.format, destination.texture.format);
        commands.push({copy: true, source, destination, size, draws: [], label: descriptor?.label});
      }, finish() { return commands; }};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const list of commands) for (const command of list) events.push(command);
    submit(commands);
  };
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configuration.format, usage: configuration.usage}); }
  })};
  const Renderer = createWebGPURendererClass(THREE), renderer = new Renderer({canvas, device, ...options});
  t.after(() => renderer.dispose()); await renderer.init();
  const scene = new THREE.Scene(); scene.add(new THREE.AmbientLight(0xffffff, .5));
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const mesh = material => {
    const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); object.frustumCulled = false; scene.add(object); return object;
  };
  return {device, renderer, scene, camera, mesh, events, textures, canvas};
}

const code = draw => draw.pipeline.fragment?.module?.code ?? '';
const meshDraws = events => events.flatMap(event => event.draws).filter(draw => draw.indexed);
const copies = events => events.filter(event => event.copy);
function native(name, transparent = false, update = null) {
  const material = new THREE.MeshBasicNodeMaterial({transparent, opacity: transparent ? .5 : 1});
  material.colorNode = N.uniform(new THREE.Color(.25, .5, .75)).setName(name);
  if (update) material.colorNode.onObjectUpdate(frame => { update(frame); return material.colorNode.value; });
  return material;
}
const names = f => meshDraws(f.events).map(draw =>
  ['opaqueNode', 'alphaNode', 'alphaSecond', 'innerOpaque', 'innerAlpha', 'outerSample'].find(name => code(draw).includes(name)) ?? 'fixed');

test('opaque and transparent controls select mixed native and fixed source lists', async t => {
  const f = await fixture(t), updates = [], {renderer, scene, camera} = f; renderer.sortObjects = false;
  f.mesh(native('opaqueNode', false, () => updates.push('opaque')));
  f.mesh(native('alphaNode', true, () => updates.push('alpha')));
  f.mesh(new THREE.MeshBasicMaterial());
  f.mesh(new THREE.MeshBasicMaterial({transparent: true, opacity: .5}));
  renderer.opaque = false; renderer.transparent = true;
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.deepEqual(updates, ['alpha']);
  assert.deepEqual(names(f), ['alphaNode', 'fixed']);
  assert.equal(renderer.info.render.drawCalls, 2);
  const start = f.events.length; updates.length = 0;
  renderer.opaque = true; renderer.transparent = false;
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.deepEqual(updates, ['opaque']);
  assert.deepEqual(names({...f, events: f.events.slice(start)}), ['opaqueNode', 'fixed']);
});

test('both lists disabled still clear the source color background without building node programs', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  scene.background = new THREE.Color(.25, .5, .75);
  const bad = new THREE.MeshBasicNodeMaterial(); bad.fragmentNode = N.texture(new THREE.Texture());
  f.mesh(bad); f.mesh(native('alphaNode', true, () => assert.fail('disabled callback')));
  renderer.opaque = false; renderer.transparent = false;
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(meshDraws(f.events).length, 0);
  assert.equal(renderer._nodeProgramSupport, null);
  assert.ok(f.events.some(event => event.desc?.colorAttachments?.[0]?.loadOp === 'clear'));
  assert.equal(renderer.info.render.drawCalls, 0);
});

test('source list controls require true rather than truthy values', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(native('opaqueNode')); f.mesh(native('alphaNode', true));
  for (const [opaque, transparent] of [[1, 'true'], [null, undefined], [0, {}]]) {
    renderer.opaque = opaque; renderer.transparent = transparent;
    renderer.render(scene, camera);
  }
  await renderer.waitForGPU();
  assert.equal(meshDraws(f.events).length, 0);
});

test('queued source list selections survive later renderer changes', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(native('opaqueNode')); f.mesh(native('alphaNode', true));
  renderer.opaque = true; renderer.transparent = false; renderer.render(scene, camera);
  renderer.opaque = false; renderer.transparent = true; renderer.render(scene, camera);
  renderer.opaque = true; renderer.transparent = true; renderer.render(scene, camera);
  renderer.opaque = false; renderer.transparent = false;
  await renderer.waitForGPU();
  assert.deepEqual(names(f), ['opaqueNode', 'alphaNode', 'opaqueNode', 'alphaNode']);
});

test('opaque node updates may disable the later transparent phase and its framebuffer copies', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(native('opaqueNode', false, ({renderer}) => { renderer.transparent = false; }));
  const alpha = native('alphaNode', true, () => assert.fail('disabled transparent callback'));
  alpha.colorNode = N.viewportOpaqueMipTexture().rgb.add(alpha.colorNode);
  f.mesh(alpha); f.mesh(new THREE.MeshBasicMaterial({transparent: true, opacity: .5}));
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.deepEqual(names(f), ['opaqueNode']); assert.equal(copies(f.events).length, 0);
});

test('opaque node updates may enable the later transparent phase and its framebuffer copies', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(native('opaqueNode', false, ({renderer}) => { renderer.transparent = true; }));
  const alpha = native('alphaNode', true);
  alpha.colorNode = N.viewportOpaqueMipTexture().rgb.add(alpha.colorNode);
  f.mesh(alpha); renderer.transparent = false;
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.deepEqual(names(f), ['opaqueNode', 'alphaNode']); assert.equal(copies(f.events).length, 1);
  const index = f.events.indexOf(copies(f.events)[0]);
  assert.deepEqual(names({...f, events: f.events.slice(0, index)}), ['opaqueNode']);
  assert.deepEqual(names({...f, events: f.events.slice(index + 1)}), ['alphaNode']);
});

test('a transparent callback cannot retroactively disable either accepted source list', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; renderer.sortObjects = false;
  f.mesh(native('opaqueNode'));
  f.mesh(native('alphaNode', true, ({renderer}) => { renderer.opaque = false; renderer.transparent = false; }));
  f.mesh(native('alphaSecond', true));
  f.mesh(new THREE.MeshBasicMaterial()); f.mesh(new THREE.MeshBasicMaterial({transparent: true, opacity: .5}));
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.deepEqual(names(f), ['opaqueNode', 'fixed', 'alphaNode', 'alphaSecond', 'fixed']);
});

test('compileAsync constructs only enabled native phases and never executes their updates', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f, built = [], updates = [];
  for (const [name, transparent] of [['opaqueNode', false], ['alphaNode', true]]) {
    const material = native(name, transparent, () => updates.push(name)), color = material.colorNode;
    material.colorNode = N.Fn(() => { built.push(name); return color; })();
    f.mesh(material);
  }
  renderer.opaque = false; await renderer.compileAsync(scene, camera);
  assert.deepEqual([...new Set(built)], ['alphaNode']); assert.deepEqual(updates, []);
  const count = built.length;
  renderer.opaque = true; renderer.transparent = false; await renderer.compileAsync(scene, camera);
  assert.deepEqual([...new Set(built.slice(count))], ['opaqueNode']); assert.deepEqual(updates, []);
  assert.equal(meshDraws(f.events).length, 0);
});

test('native shader backgrounds follow opaque selection while transparent geometry remains visible', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; let backgroundUpdates = 0;
  scene.backgroundNode = N.uniform(new THREE.Color(.3, .4, .5)).onObjectUpdate(() => {
    backgroundUpdates++; return new THREE.Color(.3, .4, .5);
  });
  f.mesh(native('alphaNode', true));
  renderer.opaque = false; renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(backgroundUpdates, 0); assert.deepEqual(names(f), ['alphaNode']);
  renderer.opaque = true; renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(backgroundUpdates, 1);
});

test('ordinary fixed opaque meshes retain transparent child traversal when the opaque list is disabled', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const parent = f.mesh(new THREE.MeshBasicMaterial());
  const child = new THREE.Mesh(new THREE.PlaneGeometry(), new THREE.MeshBasicMaterial({transparent: true, opacity: .5}));
  child.frustumCulled = false; parent.add(child);
  renderer.opaque = false; renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(meshDraws(f.events).length, 1);
});

test('pinned PassNode filters its nested scene and restores caller list controls', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const inner = new THREE.Scene();
  for (const material of [native('innerOpaque'), native('innerAlpha', true)]) {
    const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); object.frustumCulled = false; inner.add(object);
  }
  const pass = N.pass(inner, camera); pass.opaque = false; pass.transparent = true; t.after(() => pass.dispose());
  const material = new THREE.MeshBasicNodeMaterial(); material.fragmentNode = pass.getTextureNode().mul(N.uniform(1).setName('outerSample'));
  f.mesh(material);
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.deepEqual(names(f), ['innerAlpha', 'outerSample']);
  assert.equal(renderer.opaque, true); assert.equal(renderer.transparent, true);
  assert.equal(renderer.getRenderTarget(), null);
  assert.equal(renderer._nodeCaptureBytes, 0);
});

test('fixed texture backgrounds allocate and draw only when the opaque list is enabled', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const texture = new THREE.DataTexture(new Uint8Array([64, 128, 192, 255]), 1, 1);
  texture.needsUpdate = true; t.after(() => texture.dispose()); scene.background = texture;
  let uploads = 0; f.device.queue.writeTexture = () => { uploads++; };
  f.mesh(new THREE.MeshBasicMaterial({transparent: true, opacity: .5}));
  renderer.opaque = false; renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(uploads, 0); assert.equal(meshDraws(f.events).length, 1);
  renderer.opaque = true; renderer.render(scene, camera); await renderer.waitForGPU();
  assert.ok(uploads > 0);
});

test('fixed ArrayCamera views retain source list filtering in each viewport', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const children = [camera.clone(), camera.clone()];
  for (const [index, child] of children.entries()) {
    child.viewport = new THREE.Vector4(index * 8, 0, 8, 16); child.updateMatrixWorld();
  }
  const array = new THREE.ArrayCamera(children);
  f.mesh(new THREE.MeshBasicMaterial()); f.mesh(new THREE.MeshBasicMaterial({transparent: true, opacity: .5}));
  renderer.opaque = false; renderer.render(scene, array); await renderer.waitForGPU();
  assert.equal(meshDraws(f.events).length, 2);
  const drawnPasses = f.events.filter(event => event.draws.some(draw => draw.indexed));
  assert.deepEqual(drawnPasses.map(pass => pass.viewport.slice(0, 4)), [[0, 0, 8, 16], [8, 0, 8, 16]]);
});

for (const toneMapping of [THREE.NoToneMapping, THREE.ReinhardToneMapping])
test(`opaque=false suppresses the source output quad while retaining the linear scene (${toneMapping})`, async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; renderer.toneMapping = toneMapping;
  f.mesh(native('opaqueNode')); f.mesh(native('alphaNode', true));
  renderer.opaque = false; renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(renderer._hdr, true);
  assert.deepEqual(names(f), ['alphaNode']);
  assert.equal(f.events.flatMap(event => event.draws).filter(draw => !draw.indexed).length, 0);
  assert.ok(meshDraws(f.events)[0].pipeline.fragment.targets.some(target => target.format === 'rgba16float'));
});

test('transparent updates gate queued output quads independently of accepted opaque scene draws', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; renderer.toneMapping = THREE.ReinhardToneMapping;
  let disableOutput = true;
  f.mesh(native('opaqueNode'));
  f.mesh(native('alphaNode', true, ({renderer}) => { if (disableOutput) renderer.opaque = false; }));
  renderer.render(scene, camera);
  disableOutput = false; renderer.opaque = true; renderer.render(scene, camera);
  renderer.opaque = false;
  await renderer.waitForGPU();
  assert.deepEqual(names(f), ['opaqueNode', 'alphaNode', 'opaqueNode', 'alphaNode']);
  assert.equal(f.events.flatMap(event => event.draws).filter(draw => !draw.indexed).length, 1);
});

test('transparent updates can enable the output quad without admitting the earlier opaque list', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; renderer.toneMapping = THREE.ReinhardToneMapping;
  f.mesh(native('opaqueNode'));
  f.mesh(native('alphaNode', true, ({renderer}) => { renderer.opaque = true; }));
  renderer.opaque = false; renderer.render(scene, camera); await renderer.waitForGPU();
  assert.deepEqual(names(f), ['alphaNode']);
  assert.equal(f.events.flatMap(event => event.draws).filter(draw => !draw.indexed).length, 1);
});

test('ordinary canvas rendering retains clear RGB above one in its linear output buffer', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  renderer.setClearColor(new THREE.Color().setRGB(4, 2, .5));
  f.mesh(new THREE.MeshBasicMaterial()); renderer.render(scene, camera); await renderer.waitForGPU();
  const pass = f.events.find(event => event.draws.some(draw => draw.indexed));
  assert.equal(pass.desc.colorAttachments[0].view.texture.format, 'rgba16float');
  assert.deepEqual(pass.desc.colorAttachments[0].clearValue, {r: 4, g: 2, b: .5, a: 1});
});
