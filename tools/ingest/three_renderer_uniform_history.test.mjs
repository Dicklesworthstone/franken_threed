/** Accepted shader data through the real facade/compiler/command owners.
 * Recording-device bytes prove FIFO inputs; no GPU pixel claim is made. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

async function fixture(t, {raw = false, uniforms, fragmentShader, uniformsGroups = []} = {}) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createRenderPipeline = descriptor => {
    const pipeline = {...descriptor, getBindGroupLayout: () => ({})}; device.pipelines.push(pipeline); return pipeline;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 32, height: 24, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [32, 24], format: configuration.format, usage: 16}),
  })};
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device}); await renderer.init();
  const material = new (raw ? T.RawShaderMaterial : T.ShaderMaterial)({
    glslVersion: T.GLSL3, uniforms: uniforms ?? {tint: {value: new T.Color(.125, .375, .625)}, gain: {value: .25}}, uniformsGroups,
    vertexShader: `${raw ? 'precision highp float; in vec3 position;' : ''} void main() { gl_Position = vec4(position, 1.); }`,
    fragmentShader: fragmentShader ?? 'precision highp float; uniform vec3 tint; uniform float gain; out vec4 color; void main() { color = vec4(tint * gain, 1.); }',
  });
  const scene = new T.Scene(), mesh = new T.Mesh(new T.PlaneGeometry(2, 2), material), camera = new T.PerspectiveCamera();
  mesh.frustumCulled = false; scene.add(mesh);
  t.after(() => { renderer.dispose(); material.dispose(); mesh.geometry.dispose(); });
  return {renderer, device, scene, mesh, material, camera};
}
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
function packets(f, support = f.renderer._programSupport) {
  const {reflection} = support.compile(f.material, f.mesh).program;
  return f.device.snapshots.flat().filter(draw => draw.streams.size).map(draw => {
    const {group, offsets} = draw.groups.get(0), resource = group.entries.find(entry => entry.binding === 0).resource;
    const bytes = draw.contents.get(resource.buffer), base = (resource.offset ?? 0) + (offsets[0] ?? 0);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return {draw, read(name, count = 1) {
      const uniform = reflection.uniforms.find(value => value.name === name); assert.ok(uniform, name);
      return Array.from({length: count}, (_, i) => view.getFloat32(base + uniform.offset + 4 * i, true));
    }};
  });
}

for (const raw of [false, true]) for (const warm of [false, true])
test(`queued shader values retain each accepted color and scalar (raw=${raw}, warm=${warm})`, async t => {
  const f = await fixture(t, {raw});
  if (warm) {
    f.renderer.render(f.scene, f.camera); await drain(f.renderer); f.device.snapshots.length = 0;
    f.material.side = T.BackSide;
  }
  f.renderer.render(f.scene, f.camera);
  f.material.uniforms.tint.value.setRGB(.875, .625, .375); f.material.uniforms.gain.value = .75;
  f.renderer.render(f.scene, f.camera);
  f.material.uniforms.tint.value.setRGB(9, 9, 9); f.material.uniforms.gain.value = 9;
  await drain(f.renderer);
  assert.deepEqual(packets(f).map(packet => [packet.read('tint', 3), packet.read('gain')[0]]),
    [[[.125, .375, .625], .25], [[.875, .625, .375], .75]]);
  assert.deepEqual(f.material.uniforms.tint.value.toArray(), [9, 9, 9]);
  assert.equal(f.renderer._programUniformSnapshots.allocatedBytes, 0);
});

test('cold queued uniform dictionary replacements retain the source linked dictionary', async t => {
  const f = await fixture(t), original = f.material.uniforms;
  f.renderer.render(f.scene, f.camera);
  original.gain.value = .5;
  f.material.uniforms = {tint: {value: new T.Color(.75, .5, .25)}, gain: {value: .875}};
  f.renderer.render(f.scene, f.camera);
  original.gain.value = 9; f.material.uniforms.gain.value = 10;
  await drain(f.renderer);
  assert.deepEqual(packets(f).map(packet => packet.read('gain')[0]), [.25, .5]);
  assert.equal(f.renderer._programSupport.uniformsSourceFor(f.material), original);
  f.material.defines = {NEXT_VARIANT: 1}; f.material.needsUpdate = true;
  f.material.uniforms.gain.value = .875; f.renderer.render(f.scene, f.camera);
  f.material.uniforms.gain.value = 11; await drain(f.renderer);
  assert.equal(packets(f).at(-1).read('gain')[0], .875);
  assert.equal(f.renderer._programSupport.uniformsSourceFor(f.material), f.material.uniforms);
  assert.equal(f.renderer._programUniformSnapshots.allocatedBytes, 0);
});

test('warm dictionary replacement snapshots edits to the already linked original owner', async t => {
  const f = await fixture(t), original = f.material.uniforms;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer); f.device.snapshots.length = 0;
  f.material.uniforms = {tint: {value: new T.Color()}, gain: {value: 20}};
  f.renderer.render(new T.Scene(), f.camera);
  assert.ok(f.renderer._drain, 'a newly prepared empty scene defers following draws without changing their program');
  original.gain.value = .375; f.renderer.render(f.scene, f.camera);
  original.gain.value = .625; f.renderer.render(f.scene, f.camera);
  original.gain.value = 9; await drain(f.renderer);
  assert.deepEqual(packets(f).map(packet => packet.read('gain')[0]), [.375, .625]);
});

test('one shader renders red to target A and blue to target B before preparation completes', async t => {
  const f = await fixture(t), a = new T.WebGLRenderTarget(8, 8), b = new T.WebGLRenderTarget(12, 10);
  t.after(() => { a.dispose(); b.dispose(); });
  f.material.uniforms.tint.value.setRGB(1, 0, 0); f.renderer.setRenderTarget(a); f.renderer.render(f.scene, f.camera);
  f.material.uniforms.tint.value.setRGB(0, 0, 1); f.renderer.setRenderTarget(b); f.renderer.render(f.scene, f.camera);
  f.material.uniforms.tint.value.setRGB(0, 1, 0); await drain(f.renderer);
  assert.deepEqual(packets(f, f.renderer._targets.capture(a)._programSupport).map(packet => packet.read('tint', 3)), [[1, 0, 0], [0, 0, 1]]);
  assert.deepEqual(f.device.passes.filter(pass => pass.draws.length).map(pass => pass.desc.colorAttachments[0].view.texture.width), [8, 12]);
});

test('nested structs, matrix values and detached typed arrays keep their accepted bytes', async t => {
  const weights = new Float32Array([.25, .5, .75]);
  const controls = {tint: new T.Vector3(.125, .375, .625), gain: .875};
  const basis = new T.Matrix3().set(1, 2, 3, 4, 5, 6, 7, 8, 9);
  const f = await fixture(t, {raw: true, uniforms: {controls: {value: controls}, basis: {value: basis}, weights: {value: weights}},
    fragmentShader: 'precision highp float; struct Controls { vec3 tint; float gain; }; uniform Controls controls; uniform mat3 basis; uniform float weights[3]; out vec4 color;' +
      'void main() { color = vec4(basis * controls.tint * controls.gain + vec3(weights[0], weights[1], weights[2]), 1.); }'});
  f.renderer.render(f.scene, f.camera);
  controls.tint.set(9, 9, 9); controls.gain = 9; basis.elements.fill(9);
  structuredClone(weights.buffer, {transfer: [weights.buffer]}); await drain(f.renderer);
  const packet = packets(f)[0];
  assert.deepEqual(packet.read('controls', 4), [.125, .375, .625, .875]);
  assert.deepEqual(packet.read('basis', 12), [1, 4, 7, 0, 2, 5, 8, 0, 3, 6, 9, 0]);
  assert.deepEqual(packet.read('weights', 12), [.25, 0, 0, 0, .5, 0, 0, 0, .75, 0, 0, 0]);
});

test('queued sampler selection is consistent during preparation and uniform packing', async t => {
  const a = new T.DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1);
  const b = new T.DataTexture(new Uint8Array(8).fill(255), 2, 1); a.needsUpdate = b.needsUpdate = true;
  t.after(() => { a.dispose(); b.dispose(); });
  const f = await fixture(t, {raw: true, uniforms: {image: {value: a}},
    fragmentShader: 'precision highp float; uniform sampler2D image; out vec4 color; void main() { color = texture(image, vec2(.5)); }'});
  f.renderer.render(f.scene, f.camera); f.material.uniforms.image.value = b; f.renderer.render(f.scene, f.camera);
  f.material.uniforms.image.value = null; await drain(f.renderer);
  const widths = packets(f).map(({draw}) => [...draw.groups.values()].flatMap(({group}) => group.entries)
    .filter(entry => entry.resource?.texture).map(entry => entry.resource.texture.width));
  assert.deepEqual(widths, [[1], [2]]); assert.equal(f.material.uniforms.image.value, null);
});

test('ordinary scenes retain uniform block values without replacing their source layout owner', async t => {
  const scalar = new T.Uniform(.25), vector = new T.Uniform(new T.Vector3(.125, .375, .625));
  const group = new T.UniformsGroup().setName('Controls').add(scalar).add(vector); t.after(() => group.dispose());
  const f = await fixture(t, {raw: true, uniforms: {}, uniformsGroups: [group],
    fragmentShader: 'precision highp float; layout(std140) uniform Controls { float gain; vec3 tint; } controls; out vec4 color; void main() { color = vec4(controls.tint * controls.gain, 1.); }'});
  f.renderer.render(f.scene, f.camera); scalar.value = .75; vector.value.set(.875, .625, .375);
  f.renderer.render(f.scene, f.camera); scalar.value = 9; vector.value.set(9, 9, 9); await drain(f.renderer);
  assert.deepEqual(packets(f).map(packet => [packet.read('controls_gain')[0], packet.read('controls_tint', 3)]),
    [[.25, [.125, .375, .625]], [.75, [.875, .625, .375]]]);
  assert.equal(group.__size, 32); assert.equal(scalar.value, 9); assert.equal(f.material.uniformsGroups[0], group);
});

test('a hook added after acceptance keeps the original live material and callback boundary', async t => {
  const f = await fixture(t); let calls = 0;
  f.renderer.render(f.scene, f.camera);
  f.mesh.onBeforeRender = (_renderer, _scene, camera, _geometry, material) => {
    calls++; assert.equal(camera, f.camera); assert.equal(material, f.material); material.uniforms.gain.value = .875;
  };
  await drain(f.renderer);
  assert.equal(calls, 1); assert.equal(packets(f)[0].read('gain')[0], .875);
  assert.equal(f.renderer._programUniformSnapshots.allocatedBytes, 0);
});

test('warm uniform-only edits stay synchronous and keep the same compiled pipeline', async t => {
  const f = await fixture(t); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const pipeline = packets(f)[0].draw.pipeline, count = f.device.pipelines.length;
  f.material.uniforms.gain.value = .875; f.renderer.render(f.scene, f.camera);
  assert.equal(f.renderer._drain, null); assert.equal(f.device.pipelines.length, count);
  assert.equal(packets(f).at(-1).draw.pipeline, pipeline); assert.equal(packets(f).at(-1).read('gain')[0], .875);
  assert.equal(f.renderer._programUniformSnapshots.allocatedBytes, 0);
});

test('failed preparation releases pending shader captures and dictionary candidates', async t => {
  const f = await fixture(t);
  f.device.createRenderPipelineAsync = async () => { throw new Error('shader pipeline failed'); };
  f.renderer.render(f.scene, f.camera); f.material.uniforms.gain.value = .75; f.renderer.render(f.scene, f.camera);
  assert.ok(f.renderer._programUniformSnapshots.allocatedBytes > 0);
  await assert.rejects(drain(f.renderer), /shader pipeline failed/);
  assert.equal(f.renderer._programUniformSnapshots.allocatedBytes, 0); assert.equal(f.renderer._pending.length, 0);
});
