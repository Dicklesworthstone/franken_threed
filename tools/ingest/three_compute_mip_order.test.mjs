/** Source binding-order comparison for compute storage mip generation.
 * Executes pinned Bindings._update and the production texture owners against
 * an initialized recording image. Uniform-valued levels expose queue ordering;
 * this does not execute WGSL or establish browser pixel parity.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import Bindings from '../../upstream/three.js/src/renderers/common/Bindings.js';
import {createThreeStorageTextures} from './three_storage_textures.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';

class Texture {
  constructor() {
    Object.assign(this, {isTexture: true, isStorageTexture: true, image: {width: 4, height: 4},
      source: {}, version: 0, type: 'u8', format: 'rgba', colorSpace: '',
      generateMipmaps: true, mipmapsAutoUpdate: true, mipmaps: [],
      minFilter: 'nearest', magFilter: 'nearest', wrapS: 'clamp', wrapT: 'clamp',
      anisotropy: 1, compareFunction: null});
    this.listeners = new Map();
  }
  addEventListener(type, callback) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(callback); this.listeners.set(type, listeners);
  }
  removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
  dispose() { for (const callback of [...this.listeners.get('dispose') ?? []]) callback(); }
}
const THREE = {Texture, UnsignedByteType: 'u8', HalfFloatType: 'f16', FloatType: 'f32',
  RedFormat: 'r', RGFormat: 'rg', RGBAFormat: 'rgba', NoColorSpace: '', LinearSRGBColorSpace: 'linear',
  NearestFilter: 'nearest', LinearFilter: 'linear', NearestMipmapNearestFilter: 'nn', NearestMipmapLinearFilter: 'nl',
  LinearMipmapNearestFilter: 'ln', LinearMipmapLinearFilter: 'll',
  ClampToEdgeWrapping: 'clamp', RepeatWrapping: 'repeat', MirroredRepeatWrapping: 'mirror'};

function device() {
  const events = [], scopes = [];
  return {events, limits: {maxTextureDimension2D: 64}, features: new Set(),
    pushErrorScope(type) { scopes.push(type); },
    popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(null); },
    createTexture(descriptor) {
      return {descriptor, levels: Array(descriptor.mipLevelCount).fill(0),
        createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    },
    createSampler: descriptor => descriptor,
    createShaderModule: descriptor => descriptor,
    createBindGroupLayout: descriptor => descriptor,
    createPipelineLayout: descriptor => descriptor,
    createRenderPipeline: descriptor => ({descriptor, getBindGroupLayout: () => ({})}),
    createBindGroup: descriptor => descriptor,
    createCommandEncoder({label}) {
      const commands = [];
      return {beginRenderPass(descriptor) {
        let group;
        return {setPipeline() {}, setBindGroup(_index, selected) { group = selected; },
          draw() {
            const source = group.entries[0].resource, target = descriptor.colorAttachments[0].view;
            commands.push(() => {
              assert.equal(source.texture.destroyed, undefined); assert.equal(target.texture.destroyed, undefined);
              target.texture.levels[target.baseMipLevel] = source.texture.levels[source.baseMipLevel];
            });
          }, end() {}};
      }, finish: () => ({label, commands})};
    },
    queue: {submit(commandBuffers) {
      for (const commandBuffer of commandBuffers) {
        events.push(commandBuffer.label);
        for (const command of commandBuffer.commands) command();
      }
    }, onSubmittedWorkDone: () => Promise.resolve()},
  };
}

function rig(t, source = false) {
  const gpu = device(), texture = new Texture();
  const storage = createThreeStorageTextures(THREE, {getDevice: () => gpu});
  const owner = createThreeComputeTextures({storage, poolFactory() { throw new Error('No CPU image in storage fixture'); }});
  // Start with an existing image and distinct base/mip content. Initial upload
  // and initial mip allocation have their own established texture-owner tests.
  const entry = storage.ensure(storage.capture(texture)), native = entry.native;
  native.levels.splice(0, 3, 11, 3, 2);
  const backendData = new WeakMap(), backend = {
    get(value) { let data = backendData.get(value); if (!data) backendData.set(value, data = {}); return data; },
    generateMipmaps(value) {
      assert.equal(value, texture);
      gpu.queue.submit([{label: 'f3d-storage-mips', commands: [() => {
        for (let level = 1; level < native.levels.length; level++) native.levels[level] = native.levels[level - 1];
      }]}]);
    },
    updateBindings() {},
  };
  const reference = new Bindings(backend, {updateGroup: () => true}, {
    get: () => ({}), needsMipmaps: value => value.generateMipmaps || value.mipmaps.length > 0,
  }, {}, {}, {});
  const binding = (kind, options = {}) => ({texture, isSampler: true,
    isSampledTexture: kind !== 'sampler', store: kind === 'storage',
    access: 'writeOnly', mipLevel: 0, update: () => false, ...options});
  const update = (kind, options = {}) => {
    const selected = binding(kind, options);
    if (source) reference._update({bindings: [selected]}, []);
    else return owner.resolveForCompute(owner.captureBinding(selected));
  };
  const sample = () => {
    const selected = binding('sampled');
    if (source) reference._update({bindings: [selected]}, []);
    else owner.resolve(owner.captureBinding(selected)); // ordinary graphics consumption
    return native.levels.slice();
  };
  const submit = commands => gpu.queue.submit([{label: 'parent-compute', commands}]);
  t.after(() => { owner.dispose(); storage.dispose(); });
  return {gpu, texture, native, storage, owner, binding, reference, update, sample, submit};
}
const paired = (t, run) => {
  const expected = run(rig(t, true)), actual = run(rig(t));
  assert.deepEqual(actual, expected); return actual;
};

test('a grouped writer/sample generates mips before the still-unsubmitted parent dispatches', t => {
  const actual = paired(t, f => {
    f.update('storage');
    const commands = [() => { f.native.levels[0] = 47; }];
    assert.deepEqual(f.gpu.events, [], 'binding a writer records intent without submitting mips');
    f.update('sampled');
    const seen = [];
    commands.push(() => seen.push(f.native.levels[1]));
    f.submit(commands);
    // The sampled binding already consumed the source dirty flag. Completion
    // of the parent store does not invent a second automatic generation.
    return {events: f.gpu.events.slice(), seen, levels: f.sample()};
  });
  assert.deepEqual(actual, {events: ['f3d-storage-mips', 'parent-compute'], seen: [11], levels: [47, 11, 11]});
});

test('a later graphics sample regenerates a submitted writer-only group once', t => {
  const actual = paired(t, f => {
    f.update('storage'); f.submit([() => { f.native.levels[0] = 47; }]);
    const before = f.native.levels.slice(), first = f.sample(), second = f.sample();
    return {events: f.gpu.events.slice(), before, first, second};
  });
  assert.deepEqual(actual, {events: ['parent-compute', 'f3d-storage-mips'],
    before: [47, 3, 2], first: [47, 47, 47], second: [47, 47, 47]});
});

test('source binding order preserves a later store intent after an earlier sample', t => {
  const actual = paired(t, f => {
    f.update('storage'); f.submit([() => { f.native.levels[0] = 17; }]);
    f.update('sampled'); f.update('storage');
    f.submit([() => { f.native.levels[0] = 29; }]);
    return {levels: f.sample(), events: f.gpu.events.slice()};
  });
  assert.deepEqual(actual, {levels: [29, 29, 29],
    events: ['parent-compute', 'f3d-storage-mips', 'parent-compute', 'f3d-storage-mips']});
});

for (const access of ['readOnly', 'writeOnly', 'readWrite']) for (const mipLevel of [0, 1])
  test('source store flag marks pending mips with ' + access + ' at mip ' + mipLevel, t => {
    const actual = paired(t, f => {
      f.update('storage', {access, mipLevel}); f.update('sampler');
      assert.deepEqual(f.gpu.events, [], 'a sampler alone does not consume the source pending flag');
      return {levels: f.sample(), events: f.gpu.events.slice()};
    });
    assert.deepEqual(actual, {levels: [11, 11, 11], events: ['f3d-storage-mips']});
  });

test('disabled auto updates preserve earlier intent and do not mark new storage bindings', t => {
  const actual = paired(t, f => {
    f.update('storage'); f.texture.mipmapsAutoUpdate = false;
    f.update('sampled'); f.submit([() => { f.native.levels[0] = 17; }]);
    assert.deepEqual(f.gpu.events, ['parent-compute']);
    f.texture.mipmapsAutoUpdate = true; const first = f.sample();
    f.texture.mipmapsAutoUpdate = false; f.update('storage');
    f.submit([() => { f.native.levels[0] = 29; }]);
    f.texture.mipmapsAutoUpdate = true;
    return {first, last: f.sample(), events: f.gpu.events.slice()};
  });
  assert.deepEqual(actual, {first: [17, 17, 17], last: [29, 17, 17],
    events: ['parent-compute', 'f3d-storage-mips', 'parent-compute']});
});

test('captured compute and graphics uses retain their accepted auto-update controls', t => {
  const f = rig(t);
  const write = f.owner.captureBinding(f.binding('storage'));
  f.texture.mipmapsAutoUpdate = false;
  f.owner.resolveForCompute(write);
  const disabled = f.owner.captureBinding(f.binding('sampled'));
  f.texture.mipmapsAutoUpdate = true;
  f.owner.resolveForCompute(disabled); f.owner.resolve(disabled);
  assert.deepEqual(f.gpu.events, []);
  const enabled = f.owner.captureBinding(f.binding('sampled'));
  f.texture.mipmapsAutoUpdate = false;
  f.owner.resolveForCompute(enabled);
  assert.deepEqual(f.gpu.events, ['f3d-storage-mips']);
});

test('an abandoned parent keeps issued binding intent without publishing its recorded store', t => {
  const actual = paired(t, f => {
    f.update('storage');
    const error = new Error('after update');
    assert.throws(() => { throw error; }, error); // parent encoder is abandoned
    assert.deepEqual(f.gpu.events, []);
    return {levels: f.sample(), events: f.gpu.events.slice()};
  });
  assert.deepEqual(actual, {levels: [11, 11, 11], events: ['f3d-storage-mips']});
});

test('explicit native writes retain their eager policy independently of source compute intent', t => {
  const f = rig(t), write = f.owner.captureBinding(f.binding('storage'));
  f.owner.resolveForCompute(write);
  f.native.levels[0] = 47; f.owner.written([write]);
  assert.deepEqual(f.native.levels, [47, 47, 47]);
  assert.deepEqual(f.gpu.events, ['f3d-storage-mips']);
  f.sample();
  assert.deepEqual(f.gpu.events, ['f3d-storage-mips', 'f3d-storage-mips'],
    'the separate source binding flag is still consumed by a later sample');
});

test('disposing pending storage rejects consumption before another mip submission', t => {
  const f = rig(t);
  f.update('storage'); const selected = f.owner.captureBinding(f.binding('sampled'));
  f.texture.dispose();
  assert.throws(() => f.owner.resolveForCompute(selected), {code: 'THREE_STORAGE_TEXTURE_STALE'});
  assert.deepEqual(f.gpu.events, []);
});
