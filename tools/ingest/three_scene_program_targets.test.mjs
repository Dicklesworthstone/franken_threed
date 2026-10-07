import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import {T, fixture} from './fixtures/scene_program_fixture.mjs';
register('./fixtures/scene_program_loader.mjs', import.meta.url);
const {createGpuThreeScene} = await import('./three_scene.mjs');

for (const format of ['rgba8unorm', 'rgba8unorm-srgb', 'rgba16float'])
  for (const sampleCount of [1, 4]) test(`source programs draw into ${format}, ${sampleCount} samples without shader sRGB transfer`, async () => {
    const f = fixture(); f.mesh.material.uniforms.map = {value: f.texture};
    const bridge = await createGpuThreeScene(f.device, f.scene, {three: T, textures: f.textures, program: f.program, renderer: {format, sampleCount}});
    const frame = {colorView: {}, targetSize: [64, 32, 48, 24]};
    f.mesh.material.uniforms.value.value = 3;
    bridge.render(f.camera, frame);
    const draw = f.device.frames[0];
    assert.equal(draw.options.format, format); assert.equal(draw.options.sampleCount, sampleCount);
    assert.equal(draw.options.outputTransfer, undefined); assert.equal(draw.frame.draws.length, 1);
    assert.equal(new DataView(draw.frame.draws[0].programUniforms.buffer).getFloat32(0, true), 3);
    assert.deepEqual(f.sizes.at(-1), [64, 32, 48, 24]);
    assert.equal(draw.frame.targetSize, undefined, 'program-only dimensions never leak to core frame validation');
    assert.equal(f.device.registrations[0].description.textures[0].view, f.binding.view);
    bridge.dispose(); assert.ok(f.device.geometries.every(g => g.disposed), 'program geometry retires with the target dispatcher');
  });

test('linear source programs retain live numeric uniforms and stable borrowed texture bindings', async () => {
  const f = fixture(); f.mesh.material.uniforms.map = {value: f.texture};
  const bridge = await createGpuThreeScene(f.device, f.scene, {three: T, textures: f.textures, program: f.program, renderer: {format: 'rgba16float'}});
  for (const value of [1, 7, 13]) { f.mesh.material.uniforms.value.value = value; bridge.render(f.camera, {colorView: {}}); }
  assert.deepEqual(f.device.frames.map(d => new DataView(d.frame.draws[0].programUniforms.buffer).getFloat32(0, true)), [1, 7, 13]);
  assert.equal(f.device.registrations.length, 1); bridge.dispose();
});

test('linear ShaderLib fallback is available independently of canvas encoding', async () => {
  const f = fixture(); f.mesh.material.isShaderMaterial = false; f.mesh.material.type = 'MeshMatcapMaterial';
  const bridge = await createGpuThreeScene(f.device, f.scene, {three: T, program: f.program, renderer: {format: 'rgba16float'}});
  bridge.render(f.camera, {colorView: {}});
  assert.equal(f.device.frames[0].frame.draws.length, 1); bridge.dispose();
});

test('changing a borrowed target view requires preparation, not silent stale sampling', async () => {
  const f = fixture(); f.mesh.material.uniforms.map = {value: f.texture};
  const bridge = await createGpuThreeScene(f.device, f.scene, {three: T, textures: f.textures, program: f.program});
  f.textures.set(f.texture, {...f.binding, view: {replacement: true}});
  assert.throws(() => bridge.render(f.camera, {colorView: {}}), {code: 'THREE_SCENE_PREPARE'});
  assert.equal(f.device.frames.length, 0);
  await bridge.prepare(); bridge.render(f.camera, {colorView: {}});
  assert.equal(f.device.frames.length, 1); bridge.dispose();
});

test('absence of a program implementation remains an explicit failure', async () => {
  const f = fixture();
  await assert.rejects(createGpuThreeScene(f.device, f.scene, {three: T}), {code: 'THREE_SCENE_MATERIAL'});
  assert.equal(f.device.registrations.length, 0);
});
