import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuThreeEnvironment, inspectThreeEnvironment, captureThreeEnvironmentPixels,
  threeEnvironmentDescriptor} from './three_environment.mjs';
import {planAnimationEnvironment} from './animation_environment.mjs';

// Run both production owners and the actual filter command generator. Only the
// source object identities and GPU boundary are fixtures; no shader pixels or
// pinned-source/browser equivalence are inferred from this recording device.
class Texture {
  constructor(image, type = 1) {
    Object.assign(this, {isTexture: true, image, type, format: 3, internalFormat: null,
      colorSpace: 'linear', flipY: false, premultiplyAlpha: false, unpackAlignment: 1,
      onUpdate: null, updateRanges: [], mipmaps: [], version: 1});
    this.source = {data: image, dataReady: true, version: 1};
    this.listeners = new Map();
  }
  addEventListener(key, fn) { const set = this.listeners.get(key) ?? new Set(); set.add(fn); this.listeners.set(key, set); }
  removeEventListener(key, fn) { const set = this.listeners.get(key); set?.delete(fn); if (!set?.size) this.listeners.delete(key); }
  dispose() { for (const fn of [...(this.listeners.get('dispose') ?? [])]) fn(); }
}
class DataTexture extends Texture {
  constructor(data, width, height, type = 1) { super({data, width, height}, type); this.isDataTexture = true; this.mapping = 4; }
  get data() { return this.image.data; } set data(v) { this.image.data = v; }
  get width() { return this.image.width; } set width(v) { this.image.width = v; }
  get height() { return this.image.height; } set height(v) { this.image.height = v; }
}
class CubeTexture extends Texture {
  constructor(images, type = 1) { super(images, type); this.isCubeTexture = true; this.mapping = 5; }
}
class Matrix4 {
  makeRotationFromEuler(e) { const c = Math.cos(e.y), s = Math.sin(e.y);
    this.elements = [c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1]; return this; }
}
const T = {REVISION: '186', DataTexture, CubeTexture, Matrix4, HalfFloatType: 1, FloatType: 2,
  RGBAFormat: 3, EquirectangularReflectionMapping: 4, CubeReflectionMapping: 5, UnsignedByteType: 6,
  LinearSRGBColorSpace: 'linear', NoColorSpace: '', SRGBColorSpace: 'srgb'};
const quality = {size: 2, diffuseSize: 1, lutSize: 2, samples: 2, maxSampleWork: 1024};
const code = c => e => e.code === `THREE_ENVIRONMENT_${c}`;
const tick = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve, reject; const promise = new Promise((a,b) => {resolve = a; reject = b;}); return {promise, resolve, reject}; }
function cube(type = T.HalfFloatType, size = 2, wrapped = true) {
  const C = type === T.HalfFloatType ? Uint16Array : type === T.FloatType ? Float32Array : Uint8Array;
  const images = Array.from({length: 6}, (_, i) => {
    const data = new C(size * size * 4).fill(type === T.HalfFloatType ? 0x3c00 + i : i + 1);
    return wrapped ? new DataTexture(data, size, size, type) : {data, width: size, height: size};
  });
  return new CubeTexture(images, type);
}
function setup() {
  const loss = deferred(), state = {textures: [], buffers: [], uploads: [], shaders: [], passes: [], layouts: [],
    scopes: [], submits: 0, pipelineCalls: 0, pops: 0};
  const device = {
    limits: {maxTextureDimension2D: 4096, maxTextureArrayLayers: 256, minUniformBufferOffsetAlignment: 256, maxBufferSize: 1 << 24},
    lost: loss.promise,
    pushErrorScope(kind) { state.scopes.push(kind); },
    popErrorScope() { assert.ok(state.scopes.pop()); return Promise.resolve(++state.pops === state.invalidPop ? {message: 'validation failure'} : null); },
    createTexture(options) {
      if (state.throwTextureAt === state.textures.length) throw new Error('allocation failure');
      const t = {options, width: options.size[0], height: options.size[1], depthOrArrayLayers: options.size[2],
        dimension: options.dimension, format: options.format, sampleCount: options.sampleCount ?? 1,
        usage: options.usage, destroyed: 0, views: [], destroy() { this.destroyed++; },
        createView(options = {}) { const view = {texture: this, options}; this.views.push(view); return view; }};
      state.textures.push(t); return t;
    },
    createBuffer(options) { const b = {options, destroyed: 0, destroy() { this.destroyed++; }}; state.buffers.push(b); return b; },
    createSampler(options) { return {options}; },
    createBindGroupLayout(options) { state.layouts.push(options); return options; },
    createPipelineLayout(options) { return options; },
    createShaderModule(options) { state.shaders.push(options.code); return options; },
    async createRenderPipelineAsync(options) { state.pipelineCalls++; if (state.pipelineGate) await state.pipelineGate; return options; },
    createBindGroup(options) { return options; },
    createCommandEncoder() { return {
      beginRenderPass(options) {
        const p = {options, setPipeline(v) {this.pipeline = v;}, setBindGroup(...v) {this.binding = v;},
          draw(n) {this.vertices = n;}, end() {this.ended = true;}}; state.passes.push(p); return p;
      }, finish() { return {}; },
    }; },
    queue: {
      writeTexture(target, data, layout, extent) {
        if (state.uploadError) throw new Error('upload failure');
        state.uploads.push({target, data: data.slice(), layout, extent});
      },
      writeBuffer(buffer, offset, data) { buffer.data = data.slice(0); },
      submit() { state.submits++; },
      onSubmittedWorkDone() { return state.workGate ?? Promise.resolve(); },
    },
  };
  const make = (texture, options = {}) => createGpuThreeEnvironment(device, texture, {three: T, ...quality, ...options});
  return {state, device, loss, make};
}
function released(state) {
  assert.ok(state.textures.every(t => t.destroyed === 1), 'all owned textures destroyed exactly once');
  assert.ok(state.buffers.every(b => b.destroyed === 1), 'all owned buffers destroyed exactly once');
  assert.equal(state.scopes.length, 0);
}
function halfFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1, e = (bits >>> 10) & 31, m = bits & 1023;
  return sign * (e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15));
}

test('six source faces reach the real cube filter, its full mip chain and same-device receiver', async () => {
  const {state, device, make} = setup(), texture = cube(), originals = texture.image.map(f => f.data.slice());
  const owner = await make(texture), upload = state.uploads[0], source = state.textures[0];
  assert.deepEqual(source.options.size, [2,2,6]); assert.equal(source.options.format, 'rgba16float');
  assert.deepEqual(upload.layout, {bytesPerRow: 16, rowsPerImage: 2}); assert.deepEqual(upload.extent, [2,2,6]);
  assert.equal(upload.data.byteLength, 192); assert.equal(source.destroyed, 1);
  assert.equal(state.layouts[0].entries[2].texture.viewDimension, 'cube');
  assert.equal(source.views[0].options.arrayLayerCount, 6); assert.match(state.shaders[0], /texture_cube<f32>/);
  assert.equal(state.passes.length, 19); assert.equal(state.submits, 1);
  assert.ok(state.passes.every(p => p.ended && p.vertices === 3));
  assert.equal(owner.sample(device).specularView.options.dimension, 'cube');
  assert.equal(owner.sample(device).mipLevelCount, 2); assert.equal(owner.allocatedBytes, owner.plan.textureBytes);
  assert.throws(() => owner.sample({}), /same GPU device/);
  assert.deepEqual(texture.image.map(f => f.data), originals); assert.equal(owner.matches(), true);
  await owner.whenIdle(); owner.dispose(); owner.dispose(); released(state); assert.equal(texture.listeners.size, 0);
});

// Independent major-axis lookup, rather than repeating the implementation's
// face permutation. Off-axis rays expose orientation errors on all six faces.
function address([x,y,z], size) {
  const a = [Math.abs(x), Math.abs(y), Math.abs(z)];
  let face, u, v, m;
  if (a[0] > a[1] && a[0] > a[2]) { m = a[0]; face = x > 0 ? 0 : 1; u = x > 0 ? -z : z; v = -y; }
  else if (a[1] > a[2]) { m = a[1]; face = y > 0 ? 2 : 3; u = x; v = y > 0 ? z : -z; }
  else { m = a[2]; face = z > 0 ? 4 : 5; u = z > 0 ? x : -x; v = -y; }
  return [face, Math.min(size-1, Math.floor((u / m + 1) * size / 2)), Math.min(size-1, Math.floor((v / m + 1) * size / 2))];
}
for (const flipY of [false, true]) test(`source X reflection and flipY=${flipY} preserve all face interiors`, async () => {
  const {state, make} = setup(), texture = cube(T.FloatType, 4); texture.flipY = flipY;
  for (let f = 0; f < 6; f++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++)
    texture.image[f].data.set([f * 100 + y * 10 + x, -1, 2, 1], (y * 4 + x) * 4);
  const owner = await make(texture), output = state.uploads[0].data;
  for (let axis = 0; axis < 3; axis++) for (const sign of [-1,1])
    for (const a of [-.75,-.25,.25,.75]) for (const b of [-.75,-.25,.25,.75]) {
      const d = []; d[axis] = sign; d[(axis+1)%3] = a; d[(axis+2)%3] = b;
      const [outFace, x, y] = address(d,4), [srcFace, sx, sy] = address([-d[0],d[1],d[2]],4);
      const sourceFace = flipY && (srcFace === 2 || srcFace === 3) ? 5-srcFace : srcFace;
      const expected = sourceFace * 100 + (flipY ? 3-sy : sy) * 10 + sx;
      assert.equal(halfFloat(output[((outFace * 4 + y) * 4 + x) * 4]), expected, `ray ${d}`);
    }
  owner.dispose(); released(state);
});

for (const wrapped of [true]) test(`half-float ${wrapped ? 'DataTexture' : 'data'} faces preserve finite bits, signs and subnormals`, async () => {
  const {state, make} = setup(), texture = cube(T.HalfFloatType, 1, wrapped), bits = [0x8000, 0x7bff, 1, 0x3c00];
  for (const f of texture.image) (wrapped ? f.image : f).data.set(bits);
  const owner = await make(texture);
  for (let i = 0; i < 6; i++) assert.deepEqual([...state.uploads[0].data.slice(i*4,i*4+4)], bits);
  owner.dispose(); released(state);
});

test('float HDR rounds ties to even without clipping bright or negative energy', async () => {
  const {state, make} = setup(), texture = cube(T.FloatType, 1);
  for (const f of texture.image) f.data.set([1+2**-11, 1+3*2**-11, -4, 65504]);
  const owner = await make(texture);
  assert.deepEqual([...state.uploads[0].data.slice(0,4)], [0x3c00, 0x3c02, 0xc400, 0x7bff]); owner.dispose();
});

for (const clamped of [false, true]) test(`sRGB ${clamped ? 'clamped ' : ''}bytes decode RGB, never alpha`, async () => {
  const {state, make} = setup(), texture = cube(T.UnsignedByteType, 1); texture.colorSpace = T.SRGBColorSpace;
  for (const f of texture.image) f.data = new (clamped ? Uint8ClampedArray : Uint8Array)([10,128,255,128]);
  const owner = await make(texture);
  assert.deepEqual([...state.uploads[0].data.slice(0,4)], [0x1a37,0x32e8,0x3c00,0x3804]); owner.dispose();
});
for (const space of [T.NoColorSpace,T.LinearSRGBColorSpace]) test(`byte colorSpace=${JSON.stringify(space)} stays linear`, async () => {
  const {state, make} = setup(), texture = cube(T.UnsignedByteType, 1); texture.colorSpace = space;
  for (const f of texture.image) f.data.set([0,128,255,128]);
  const owner = await make(texture);
  assert.deepEqual([...state.uploads[0].data.slice(0,4)], [0,0x3804,0x3c00,0x3804]); owner.dispose();
});

for (const [label, mutate, expected] of [
  ['raw descriptor face', t => t.image[5] = t.image[5].image, 'SOURCE'],
  ['missing face', t => t.image.pop(), 'PROFILE'], ['extra face', t => t.image.push(t.image[0]), 'PROFILE'],
  ['nonsquare face', t => t.image[5].height = 1, 'LIMIT'], ['unequal faces', t => t.image[5].width = t.image[5].height = 1, 'LIMIT'],
  ['wrong storage type', t => t.image[5].data = new Float32Array(16), 'STORAGE'],
  ['short final face', t => t.image[5].data = new Uint16Array(15), 'STORAGE'],
  ['shared final face', t => t.image[5].data = new Uint16Array(new SharedArrayBuffer(32)), 'STORAGE'],
  ['resizable storage', t => t.image[5].data = new Uint16Array(new ArrayBuffer(32, {maxByteLength: 64})), 'STORAGE'],
  ['detached storage', t => structuredClone(t.image[5].data, {transfer: [t.image[5].data.buffer]}), 'STORAGE'],
  ['not ready', t => t.source.dataReady = false, 'NOT_READY'], ['wrong source identity', t => t.source.data = [], 'NOT_READY'],
  ['unuploaded parent', t => t.version = 0, 'LIMIT'], ['refraction', t => t.mapping = 99, 'PROFILE'],
  ['render target', t => t.isRenderTargetTexture = true, 'SOURCE'], ['compressed cube', t => t.isCompressedTexture = true, 'SOURCE'],
  ['sRGB HDR', t => t.colorSpace = T.SRGBColorSpace, 'PROFILE'], ['unknown color space', t => t.colorSpace = 'display-p3', 'PROFILE'],
  ['upload hook', t => t.onUpdate = () => {}, 'PROFILE'], ['authored mips', t => t.mipmaps.push({}), 'PROFILE'],
  ['partial updates', t => t.updateRanges.push({}), 'PROFILE'], ['premultiplied alpha', t => t.premultiplyAlpha = true, 'FORMAT'],
  ['nonfinite final pixel', t => t.image[5].data[15] = 0x7c00, 'VALUE'],
]) test(`preflight rejects ${label} before any native allocation`, async () => {
  const {state, make} = setup(), texture = cube(); mutate(texture);
  await assert.rejects(make(texture), code(expected)); assert.equal(state.textures.length,0); assert.equal(state.pipelineCalls,0);
});

test('float nonfinite and overflow in the sixth face reject before upload', async () => {
  for (const value of [NaN,Infinity,-Infinity,65520,-65520]) {
    const {state, make} = setup(), texture = cube(T.FloatType); texture.image[5].data[15] = value;
    await assert.rejects(make(texture),code('VALUE')); assert.equal(state.textures.length,0);
  }
});

test('wrapped face readiness, identity and layout are validated, never silently ignored', async () => {
  for (const mutate of [f => f.type = T.FloatType, f => f.mipmaps.push({}), f => f.onUpdate = () => {},
    f => f.source.dataReady = false, f => f.source.data = {}, f => f.source.version = -1]) {
    const {state,make} = setup(), texture = cube(T.HalfFloatType,2,true); mutate(texture.image[5]);
    await assert.rejects(make(texture)); assert.equal(state.textures.length,0);
  }
});

test('aggregate six-face pixel, device dimension/layer and transient byte budgets preflight', async () => {
  const {state,device,make} = setup(), texture = cube(), plan = planAnimationEnvironment({...quality});
  await assert.rejects(make(texture,{maxPixels:23}),code('LIMIT'));
  const budget = 192 + plan.textureBytes + plan.uniformBytes;
  await assert.rejects(make(texture,{maxBytes:budget-1}),code('LIMIT'));
  device.limits.maxTextureArrayLayers = 5; await assert.rejects(make(texture),code('LIMIT'));
  device.limits.maxTextureArrayLayers = 256; device.limits.maxTextureDimension2D = 1;
  await assert.rejects(make(texture),code('LIMIT')); assert.equal(state.textures.length,0);
  device.limits.maxTextureDimension2D = 4096;
  const owner = await make(texture,{maxBytes:budget,maxPixels:24}); owner.dispose(); released(state);
});

test('matching checks never rescan pixels; explicit source and face edits require prepare', async () => {
  for (const mutate of [t => t.version++, t => t.source.version++, t => t.image[5] = new DataTexture(t.image[5].data,2,2),
    t => t.image[5].data = t.image[5].data.slice(), t => t.flipY = true]) {
    const {make} = setup(), texture = cube(), owner = await make(texture);
    texture.image[0].data[0] = 0x7c00; assert.equal(owner.matches(),true);
    mutate(texture); assert.equal(owner.matches(),false); assert.throws(() => owner.check(),code('PREPARE')); owner.dispose();
  }
  const {make} = setup(), texture = cube(T.FloatType,1,true), owner = await make(texture);
  texture.image[5].version++; assert.equal(owner.matches(),false); owner.dispose();
});

test('face replacement during real filtering cannot publish a stale owner', async () => {
  const {state,make} = setup(), gate = deferred(), texture = cube(); state.workGate = gate.promise;
  const work = make(texture); await tick(); assert.equal(state.submits,1);
  texture.image[5].data = texture.image[5].data.slice(); gate.resolve();
  await assert.rejects(work,code('PREPARE')); released(state); assert.equal(texture.listeners.size,0);
});

test('abort rejects promptly while native pipeline construction is blocked', async () => {
  const {state,make} = setup(), gate = deferred(), controller = new AbortController(), texture = cube();
  state.pipelineGate = gate.promise; const work = make(texture,{signal:controller.signal}); await tick();
  assert.equal(state.pipelineCalls,1); controller.abort(); await assert.rejects(work,code('ABORTED'));
  gate.resolve(); await tick(); released(state); assert.equal(state.submits,0); assert.equal(texture.listeners.size,0);
});

test('already-aborted preparation allocates nothing', async () => {
  const {state,make} = setup(), controller = new AbortController(); controller.abort();
  await assert.rejects(make(cube(),{signal:controller.signal}),code('ABORTED')); assert.equal(state.textures.length,0);
});

for (const kind of ['source','device','abort']) test(`${kind} revokes a completed owner without destroying borrowed source faces`, async () => {
  const {state,make,loss} = setup(), texture = cube(), controller = new AbortController();
  const owner = await make(texture,{signal:controller.signal});
  if (kind === 'source') texture.dispose(); else if (kind === 'device') loss.resolve({message:'device lost'}); else controller.abort();
  await tick(); assert.equal(owner.failed,true); assert.throws(() => owner.check()); owner.dispose(); released(state);
  assert.equal(texture.image[5].data.length,16); assert.equal(texture.listeners.size,0);
});

for (const kind of ['upload','allocation','scope','filter']) test(`${kind} failure releases partial native resources`, async () => {
  const {state,make} = setup(), texture = cube();
  if (kind === 'upload') state.uploadError = true;
  if (kind === 'allocation') state.throwTextureAt = 2;
  if (kind === 'scope') state.invalidPop = 1;
  if (kind === 'filter') state.invalidPop = 3;
  await assert.rejects(make(texture)); released(state); assert.equal(texture.listeners.size,0);
});

test('the existing panorama route stays 2D and does not require cube layer limits', async () => {
  const {state,device,make} = setup(), texture = new DataTexture(new Uint16Array(32).fill(0x3c00),4,2);
  delete device.limits.maxTextureArrayLayers; const owner = await make(texture);
  assert.deepEqual(state.uploads[0].extent,[4,2,1]); assert.deepEqual(state.uploads[0].layout,{bytesPerRow:32});
  assert.equal(state.layouts[0].entries[2].texture.viewDimension,'2d'); owner.dispose(); released(state);
});

test('a cube cannot accidentally enter the panorama-only background copy API', () => {
  assert.throws(() => captureThreeEnvironmentPixels(cube(),T),code('PROFILE'));
});

test('scene intensity and Euler rotation stay live without refiltering a cube', async () => {
  const {state,make} = setup(), owner = await make(cube()), scene = {environmentIntensity:2,
    environmentRotation:{isEuler:true,x:0,y:Math.PI/2,z:0,order:'XYZ'}};
  const packet = threeEnvironmentDescriptor(owner,scene,T);
  assert.equal(packet.map,owner); assert.equal(packet.intensity,2); assert.ok(Math.abs(packet.rotation[2]-1)<1e-12);
  scene.environmentIntensity = .25; assert.equal(threeEnvironmentDescriptor(owner,scene,T).intensity,.25);
  assert.equal(state.pipelineCalls,1); owner.dispose();
});
