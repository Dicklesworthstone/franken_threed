import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createWebGPURendererClass} from '../../tools/ingest/three_renderer.mjs';
import {gpuDevice} from './compute_geometry_fixture.mjs';

// The pinned builder, TSL graphs and F3D renderer are real. GPU commands are
// recorded by the existing test device, so this proves preparation/submission
// ownership and byte layout, not shader execution or driver validation.
test('pinned H8 particle and atomic structs build through the production compute renderer', async t => {
  const source = process.env.F3D_THREE_ROOT
    ? pathToFileURL(resolve(process.env.F3D_THREE_ROOT, 'build/three.webgpu.js'))
    : new URL('../../upstream/three.js/build/three.webgpu.js', import.meta.url);
  if (!existsSync(source)) {
    assert.equal(process.env.F3D_THREE_ROOT, undefined, 'F3D_THREE_ROOT must contain the pinned Three build');
    t.skip('Requires the pinned upstream checkout from scripts/oracle-checkout.sh'); return;
  }
  const THREE = await import(source.href);
  assert.equal(THREE.REVISION, '186');
  const device = gpuDevice(), shaders = [], makeModule = device.createShaderModule;
  device.createShaderModule = descriptor => { shaders.push(descriptor.code); return makeModule(descriptor); };
  const Renderer = createWebGPURendererClass(THREE);
  const renderer = new Renderer({canvas: {width: 16, height: 16, style: {}}});
  // Supply the initialized device at the existing renderer seam. No retained
  // Three renderer/backend is created and no browser context is impersonated.
  renderer._device = device;
  renderer._session = {device, dispose() {}, async whenIdle() {}};
  renderer._initialized = true;
  const {Fn, struct, instancedArray, instanceIndex, atomicAdd} = THREE.TSL;
  const particleType = struct({position: {type: 'vec3'}, velocity: {type: 'vec3'}, C: {type: 'mat3'}});
  const sourceWords = new Uint32Array(40).fill(0x3f800000);
  sourceWords[3] = 0x7fc01234;
  const particles = instancedArray(new Float32Array(sourceWords.buffer), particleType);
  const cells = instancedArray(4, struct({
    x: {type: 'int', atomic: true}, y: {type: 'int', atomic: true},
    z: {type: 'int', atomic: true}, mass: {type: 'int', atomic: true},
  }));
  const advance = Fn(() => {
    const particle = particles.element(instanceIndex);
    particle.get('position').addAssign(particle.get('C').mul(particle.get('velocity')));
  })().compute(2, [2]);
  const transfer = Fn(() => { atomicAdd(cells.element(instanceIndex).get('mass'), 1); })().compute(4, [2]);
  try {
    assert.equal(renderer.debug.checkShaderErrors, true);
    assert.equal(renderer.debug.diagnostics.keywords, false);
    assert.equal(particleType.getLength(), 20);
    assert.equal(renderer.compute([advance, transfer]), undefined);
    await renderer.waitForGPU();
    assert.equal(device.events.filter(e => e[0] === 'compute').length, 2);
    assert.deepEqual(device.buffers.filter(buffer => buffer.usage & 128).map(buffer => buffer.size), [160, 64]);
    assert.match(shaders[0], /position\s*:\s*vec3<f32>/);
    assert.match(shaders[0], /velocity\s*:\s*vec3<f32>/);
    assert.match(shaders[0], /C\s*:\s*mat3x3<f32>/);
    assert.match(shaders[0], /if \( instanceIndex >= .+ \) \{ return; \}/);
    assert.match(shaders[1], /mass\s*:\s*atomic<\s*i32\s*>/);
    assert.match(shaders[1], /atomicAdd\(/);
    assert.deepEqual([...new Uint32Array(await renderer.getArrayBufferAsync(particles.value))], [...sourceWords]);
  } finally { advance.dispose(); transfer.dispose(); renderer.dispose(); }
});
