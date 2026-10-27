/** Production graph preparation and storage residency with a recording device.
 * Source objects and the builder are structural fixtures, not a WGSL compiler.
 * Unrelated graph contexts, copy planning and float-only mip generation fail
 * loudly if entered. No fixture replaces the two production modules under test.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {register} from 'node:module';

const unexpected = `const unexpected = () => { throw new Error('Unexpected non-integer graphics dependency'); };`;
const contexts = {
  'three_node_program.mjs': {
    './three_node_attributes.mjs': `${unexpected}
      export const createThreeNodeAttributes = () => ({dispose() {}, register: unexpected, capture: unexpected, validate: unexpected});`,
    './three_node_shadows.mjs': `${unexpected}
      export const registerThreeNativeShadow = unexpected, updateThreeNativeShadow = unexpected;`,
    './three_node_passes.mjs': `${unexpected}
      export const inspectThreePipelineContext = () => null, inspectThreeMRT = () => null;
      export const inspectThreeNativePass = unexpected, updateThreeNativePass = unexpected;`,
    './three_node_clipping.mjs': `
      export const inspectThreeNodeClipping = () => '0/0';
      export const updateThreeNodeClipping = () => {};`,
  },
  'three_storage_textures.mjs': {
    './gpu_storage_array_mips.mjs': `${unexpected} export const submitStorageArrayMips = unexpected;`,
    './gpu_storage_volume_mips.mjs': `${unexpected} export const submitStorageVolumeMips = unexpected;`,
    './three_texture_copy_plan.mjs': `${unexpected} export const inspectStorageTextureCopy = unexpected;`,
  },
};
register('data:text/javascript,' + encodeURIComponent(`
  const contexts = ${JSON.stringify(contexts)};
  export async function resolve(specifier, context, next) {
    const parent = context.parentURL && new URL(context.parentURL).pathname.split('/').pop();
    const source = contexts[parent]?.[specifier];
    if (source) return {url: 'data:text/javascript,' + encodeURIComponent(source), shortCircuit: true};
    return next(specifier, context);
  }
`), import.meta.url);
const {createThreeNodePrograms} = await import('./three_node_program.mjs');
const {createThreeStorageTextures} = await import('./three_storage_textures.mjs');

let serial = 0;
class Events {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, fn) { const set = this.listeners.get(type) ?? new Set(); set.add(fn); this.listeners.set(type, set); }
  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
  dispose() { for (const fn of [...this.listeners.get('dispose') ?? []]) fn({type: 'dispose', target: this}); }
}
class Texture extends Events {
  constructor() {
    super(); this.id = ++serial; this.uuid = `texture-${serial}`; this.isTexture = true;
    Object.assign(this, {image: {width: 4, height: 4}, source: {version: 0}, version: 0,
      type: 'u32', format: 'rgba', colorSpace: '', channel: 0, minFilter: 'linear', magFilter: 'linear',
      wrapS: 'clamp', wrapT: 'clamp', wrapR: 'clamp', anisotropy: 1, compareFunction: null,
      generateMipmaps: false, mipmapsAutoUpdate: true, mipmaps: [], internalFormat: null});
  }
}
class StorageTexture extends Texture { constructor() { super(); this.isStorageTexture = true; } }
class StorageArrayTexture extends StorageTexture {
  constructor() { super(); this.isArrayTexture = true; this.image.depth = 3; this.mipmapsAutoUpdate = undefined; }
}
class Storage3DTexture extends StorageTexture {
  constructor() { super(); this.is3DTexture = true; this.image.depth = 3; this.mipmapsAutoUpdate = undefined; }
}
class Node {
  constructor(children = []) { this.isNode = true; this.children = children; }
  getChildren() { return this.children; }
}
class TextureNode extends Node {
  constructor(value, {stage = 2, sampler = false, layer = null} = {}) {
    super(); Object.assign(this, {value, isTextureNode: true, sampler, stage, depthNode: layer,
      levelNode: null, biasNode: null, gradNode: null, gatherNode: null, updateMatrix: false,
      groupNode: {name: 'object'}});
  }
  getBase() { return this; }
  isSampleCompare() { return false; }
}
class Texture3DNode extends TextureNode {
  constructor(value, options) { super(value, options); this.isTexture3DNode = true; }
}
class Material extends Events {
  constructor() { super(); Object.assign(this, {isMaterial: true, version: 0, lights: false, fog: false, alphaToCoverage: false}); }
  onBeforeCompile() {}
  onBeforeRender() {}
  customProgramCacheKey() { return 'fixture'; }
}
class NodeMaterial extends Material {
  constructor() { super(); this.isNodeMaterial = true; }
  customProgramCacheKey() { return 'fixture-node'; }
}
class MeshBasicNodeMaterial extends NodeMaterial { constructor() { super(); this.isMeshBasicNodeMaterial = true; } }
class BufferGeometry extends Events { constructor() { super(); this.id = ++serial; this.attributes = {}; this.morphAttributes = {}; } }
class Mesh { constructor(geometry, material) { this.geometry = geometry; this.material = material; } }
class WGSLNodeBuilder {
  constructor(object, renderer) { Object.assign(this, {object, renderer, context: {}}); }
  build() {
    this.renderer.builds++;
    this.nodes = new Set([this.material.colorNode, this.material.positionNode].filter(Boolean));
    for (const node of this.nodes) for (const child of node.getChildren()) this.nodes.add(child);
    this.renderer.buildHook?.(this);
    this.updateBeforeNodes = [...this.nodes].filter(node => node.before);
    this.updateNodes = [...this.nodes].filter(node => node.update);
    this.updateAfterNodes = [...this.nodes].filter(node => node.after);
    const bindings = [], seen = new Set();
    for (const node of this.nodes) if (node.isTextureNode && !seen.has(node.value.uuid)) {
      seen.add(node.value.uuid);
      if (node.sampler || node.forceSampler) bindings.push({textureNode: node, isSampler: true, visibility: node.stage});
      bindings.push({textureNode: node, isSampler: true, isSampledTexture: true, visibility: node.stage, store: node.forceStore === true});
    }
    this.bindings = [{name: 'object', bindings}]; this.structs = {fragment: []};
    this.vertexShader = '// recording vertex builder'; this.fragmentShader = '// recording fragment builder';
  }
  getBindings() { return this.bindings; }
  getAttributesArray() { return []; }
}
const THREE = {REVISION: '186', Texture, StorageTexture, StorageArrayTexture, Storage3DTexture, Texture3DNode,
  Node, TextureNode, Material, NodeMaterial, MeshBasicNodeMaterial, BufferGeometry, Mesh, WGSLNodeBuilder,
  TSL: {lights: () => new Node()},
  UnsignedByteType: 'u8', HalfFloatType: 'f16', FloatType: 'f32', IntType: 'i32', UnsignedIntType: 'u32',
  RGBAFormat: 'rgba', RedFormat: 'r', RGFormat: 'rg', RedIntegerFormat: 'ri', RGIntegerFormat: 'rgi', RGBAIntegerFormat: 'rgbai', RGBIntegerFormat: 'rgbi',
  NoColorSpace: '', LinearSRGBColorSpace: 'linear', SRGBColorSpace: 'srgb',
  NearestFilter: 'nearest', LinearFilter: 'linear', NearestMipmapNearestFilter: 'nn', NearestMipmapLinearFilter: 'nl',
  LinearMipmapNearestFilter: 'ln', LinearMipmapLinearFilter: 'll', ClampToEdgeWrapping: 'clamp', RepeatWrapping: 'repeat', MirroredRepeatWrapping: 'mirror',
  CubeReflectionMapping: 'cube', CubeRefractionMapping: 'cube-refraction', CubeUVReflectionMapping: 'cubeuv',
  EquirectangularReflectionMapping: 'equirect', EquirectangularRefractionMapping: 'equirect-refraction', DoubleSide: 2,
};
for (const name of ['MeshToonNodeMaterial', 'MeshLambertNodeMaterial', 'MeshPhongNodeMaterial', 'MeshStandardNodeMaterial', 'MeshPhysicalNodeMaterial', 'ShadowNodeMaterial'])
  THREE[name] = class extends NodeMaterial {};
for (const name of ['MeshBasicMaterial', 'MeshToonMaterial', 'MeshLambertMaterial', 'MeshPhongMaterial', 'MeshStandardMaterial', 'MeshPhysicalMaterial', 'ShadowMaterial'])
  THREE[name] = class extends Material {};
for (const name of ['AmbientLight', 'AmbientLightNode', 'DirectionalLight', 'DirectionalLightNode', 'HemisphereLight', 'HemisphereLightNode',
  'PointLight', 'PointLightNode', 'SpotLight', 'SpotLightNode', 'LightProbe', 'LightProbeNode', 'RectAreaLight', 'RectAreaLightNode',
  'StandardNodeLibrary', 'BufferNode', 'UniformArrayNode', 'BufferAttributeNode', 'PMREMNode', 'MaterialReferenceNode', 'ShadowNode',
  'OutputStructNode', 'MRTNode', 'Fog', 'FogExp2', 'Color', 'Camera']) THREE[name] = class {};
THREE.DepthTexture = class extends Texture {}; THREE.CubeTexture = class extends Texture {};
THREE.CubeDepthTexture = class extends THREE.DepthTexture {}; THREE.RenderTarget = class {};

function rig(t, {dimension = '2d', type = 'u32', format = 'rgba', stage = 2, features = [], maxBytes} = {}) {
  const textures = [], events = [], samplers = [], scopes = [];
  let lose;
  const device = {features: new Set(features), textures, events, samplers, lost: new Promise(resolve => { lose = resolve; }),
    limits: {maxTextureDimension3D: 32, maxTextureDimension2D: 64, maxTextureArrayLayers: 8},
    pushErrorScope: type => scopes.push(type), popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(null); },
    createTexture(descriptor) {
      const channels = /^(rgba|rg|r)32(?:uint|sint|float)$/.exec(descriptor.format)?.[1];
      const bytes = channels ? ({rgba: 16, rg: 8, r: 4})[channels] : descriptor.format === 'rgba16float' ? 8 : 4;
      const native = {descriptor, pixels: new Uint8Array(descriptor.size.reduce((a, b) => a * b, 1) * bytes), destroyed: false,
        createView(options = {}) { return {texture: native, options}; }, destroy() { this.destroyed = true; }};
      textures.push(native); return native;
    },
    createSampler(descriptor) { samplers.push(descriptor); return descriptor; },
    queue: {submit: commands => events.push(commands), onSubmittedWorkDone() { events.push('fence'); return Promise.resolve(); }},
  };
  const lifetime = new AbortController();
  const storage = createThreeStorageTextures(THREE, {getDevice: () => device, signal: lifetime.signal, maxBytes});
  const shared = {layoutOf: binding => storage.layout(storage.captureBinding(binding))};
  const renderer = {_device: device, _nodes: {nodeFrame: {
    updateNode: node => node.update?.(), updateBeforeNode: node => node.before?.(), updateAfterNode: node => node.after?.(),
  }}, info: {calls: 1}, builds: 0, hasFeature: name => device.features.has(name), getRenderTarget: () => null,
    _getSampledTextureInputs: () => shared,
    backend: {utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}}};
  const owner = createThreeNodePrograms({three: THREE, renderer});
  const material = new MeshBasicNodeMaterial(), object = new Mesh(new BufferGeometry(), material);
  const TextureClass = dimension === '3d' ? Storage3DTexture : dimension === '2d-array' ? StorageArrayTexture : StorageTexture;
  const texture = new TextureClass(); Object.assign(texture, {type, format});
  const node = dimension === '3d' ? new Texture3DNode(texture, {stage}) : new TextureNode(texture, {stage, layer: dimension === '2d-array' ? new Node() : null});
  material.colorNode = node;
  t.after(() => { owner.dispose(); storage.dispose(); });
  return {device, storage, shared, renderer, owner, material, object, texture, node, lifetime, lose,
    compile: () => owner.compile(material, object),
    capture: descriptor => owner.captureDraw(descriptor).textures.map(binding => storage.captureBinding(binding)),
    write: (value = texture, mipLevel = 0) => storage.captureBinding({texture: value, isSampledTexture: true, store: true, access: 'writeOnly', mipLevel})};
}

const dimensions = ['2d', '2d-array', '3d'];
const formats = [['r', 'r'], ['ri', 'r'], ['rg', 'rg'], ['rgi', 'rg'], ['rgba', 'rgba'], ['rgbai', 'rgba']];
for (const dimension of dimensions) for (const type of ['i32', 'u32']) for (const [format, channels] of formats)
  for (const stage of [1, 2, 3]) test(`${dimension} ${type}/${format} stage ${stage}: lossless shared integer residency`, t => {
    const f = rig(t, {dimension, type, format, stage}), descriptor = f.compile(), snapshots = f.capture(descriptor);
    assert.equal(snapshots.length, 1); assert.equal(snapshots[0].kind, 'sampled');
    assert.equal(f.device.textures.length, 0, 'preparation is allocation-free');
    const sampleType = type === 'i32' ? 'sint' : 'uint';
    assert.deepEqual(descriptor.groups[0].entries, [{binding: 0, visibility: stage,
      texture: {sampleType, viewDimension: dimension, multisampled: false}}]);
    assert.deepEqual(f.storage.layout(snapshots[0]), {texture: descriptor.groups[0].entries[0].texture});
    const write = f.write(), producer = f.storage.resolve(write).texture;
    const Words = type === 'i32' ? Int32Array : Uint32Array;
    const words = new Words(producer.pixels.buffer), expected = type === 'i32'
      ? [-2147483648, -16777217, -1, 2147483647] : [0, 16777217, 2147483648, 4294967295];
    words.set(expected); f.storage.written([write]);
    f.storage.validateDispatch(snapshots);
    const consumer = f.storage.resolve(snapshots[0]);
    assert.equal(consumer.texture, producer);
    assert.deepEqual([...new Words(consumer.texture.pixels.buffer).slice(0, 4)], expected);
    assert.equal(producer.descriptor.format, `${channels}32${sampleType}`);
    assert.equal(producer.descriptor.dimension, dimension === '3d' ? '3d' : '2d');
    assert.equal(f.compile(), descriptor, 'content writes do not rebuild the graph');
    assert.equal(f.device.textures.length, 1); assert.equal(f.device.samplers.length, 0);
    assert.equal(f.device.events.length, 0, 'reads add no copy, submission or fence');
    assert.equal(f.texture.image.data, undefined);
  });

for (const dimension of dimensions) test(`${dimension}: replacements preserve queued image identity`, t => {
  const f = rig(t, {dimension}), d = f.compile(), first = f.capture(d), replacement = new f.texture.constructor();
  f.node.value = replacement; assert.equal(f.compile(), d); const second = f.capture(d);
  const a = f.storage.resolve(first[0]).texture, b = f.storage.resolve(second[0]).texture;
  assert.notEqual(a, b); new Uint32Array(a.pixels.buffer)[0] = 4294967295; new Uint32Array(b.pixels.buffer)[0] = 16777217;
  assert.equal(new Uint32Array(f.storage.resolve(first[0]).texture.pixels.buffer)[0], 4294967295);
  assert.equal(new Uint32Array(f.storage.resolve(second[0]).texture.pixels.buffer)[0], 16777217);
});
for (const [from, to] of [['i32', 'u32'], ['u32', 'i32'], ['u32', 'f32'], ['f32', 'u32']]) test(`${from} -> ${to}: reject stale descriptors before callbacks`, t => {
  const f = rig(t, {type: from}); f.texture.minFilter = f.texture.magFilter = 'nearest';
  let calls = 0; f.node.before = () => calls++;
  const d = f.compile(); f.texture.type = to;
  assert.throws(() => f.owner.captureDraw(d), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(calls, 0);
  assert.notEqual(f.compile(), d);
});
for (const [name, mutate] of [
  ['sampler', f => { f.node.sampler = true; }], ['bias', f => { f.node.biasNode = new Node(); }],
  ['gradients', f => { f.node.gradNode = new Node(); }], ['gather', f => { f.node.gatherNode = new Node(); }],
  ['comparison coordinate', f => { f.node.compareNode = new Node(); }],
  ['comparison', f => { f.node.isSampleCompare = () => true; }], ['material map', f => { f.material.map = f.texture; }],
  ['CPU ownership', f => { f.texture.isStorageTexture = false; }], ['normalized', f => { f.texture.normalized = true; }],
  ['color space', f => { f.texture.colorSpace = 'linear'; }], ['missing owner', f => { delete f.renderer._getSampledTextureInputs; }],
  ['wrong layout', f => { f.shared.layoutOf = () => ({texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}}); }],
]) for (const dimension of dimensions) test(`${dimension}: reject ${name} before construction and allocation`, t => {
  const f = rig(t, {dimension}); mutate(f);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(f.renderer.builds, 0); assert.equal(f.device.textures.length, 0);
});
for (const key of ['sampler', 'compareNode', 'biasNode', 'gradNode', 'gatherNode']) test(`cached integer nodes revalidate ${key} before callbacks`, t => {
  const f = rig(t); let calls = 0; f.node.before = () => calls++;
  const d = f.compile(); f.node[key] = key === 'sampler' ? true : new Node();
  assert.throws(() => f.owner.textures(d), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.throws(() => f.owner.captureDraw(d), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'}); assert.equal(calls, 0);
});
test('integer changes made inside source updates are revalidated after the update', t => {
  const f = rig(t); f.node.update = () => { f.node.sampler = true; };
  const d = f.compile(); assert.throws(() => f.owner.captureDraw(d), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(f.device.textures.length, 0);
});
for (const key of ['forceSampler', 'forceStore']) test(`builder-introduced ${key} cannot bypass admission`, t => {
  const f = rig(t); f.node[key] = true; assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_BINDING'});
});
test('an integer read introduced during source construction is admitted', t => {
  const f = rig(t); f.material.colorNode = new Node(); f.renderer.buildHook = builder => builder.nodes.add(f.node);
  const d = f.compile(); assert.equal(d.groups[0].entries[0].texture.sampleType, 'uint');
});
test('a sampled integer read introduced during source construction is rejected', t => {
  const f = rig(t); f.material.colorNode = new Node(); f.node.sampler = true;
  f.renderer.buildHook = builder => builder.nodes.add(f.node);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});
test('alias partition changes rebuild instead of silently sharing stale integer bindings', t => {
  const f = rig(t), sibling = new TextureNode(f.texture); f.material.colorNode = new Node([f.node, sibling]);
  const d = f.compile(); assert.equal(d.groups[0].entries.length, 1);
  sibling.value = new StorageTexture(); assert.throws(() => f.owner.captureDraw(d), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const next = f.compile(); assert.notEqual(next, d); assert.equal(next.groups[0].entries.length, 2);
});
for (const dimension of dimensions) test(`${dimension}: explicit integer mips remain producer-owned`, t => {
  const f = rig(t, {dimension}); f.texture.generateMipmaps = true; f.texture.mipmapsAutoUpdate = false;
  f.node.levelNode = new Node(); const d = f.compile(), snapshots = f.capture(d), write = f.write(f.texture, 1);
  const native = f.storage.resolve(write).texture; f.storage.written([write]);
  assert.equal(f.storage.resolve(snapshots[0]).texture, native); assert.equal(f.device.events.length, 0);
  assert.equal(native.descriptor.mipLevelCount, 3);
});
for (const dimension of dimensions) test(`${dimension}: integer auto-reduction remains rejected at producer admission`, t => {
  const f = rig(t, {dimension}); f.texture.generateMipmaps = true; f.texture.mipmapsAutoUpdate = true;
  f.capture(f.compile()); assert.throws(() => f.write(), {code: 'THREE_STORAGE_TEXTURE_MIPS'});
  assert.equal(f.device.textures.length, 0);
});
for (const change of [texture => texture.dispose(), texture => { texture.image.width = 8; },
  texture => { texture.type = 'i32'; }, texture => { texture.generateMipmaps = true; }]) test(`integer generation invalidation: ${change}`, t => {
  const f = rig(t), snapshots = f.capture(f.compile()); change(f.texture);
  assert.throws(() => f.storage.validateDispatch(snapshots), {code: 'THREE_STORAGE_TEXTURE_STALE'});
});
for (const dimension of dimensions) test(`${dimension}: allocation budget includes 32-bit channels and depth`, t => {
  const f = rig(t, {dimension, maxBytes: dimension === '2d' ? 255 : 767}), snapshots = f.capture(f.compile());
  assert.throws(() => f.storage.resolve(snapshots[0]), {code: 'THREE_STORAGE_TEXTURE_BUDGET'});
  assert.equal(f.device.textures.length, 0);
});
test('integer sampled reads retain storage feedback validation', t => {
  const f = rig(t), snapshots = f.capture(f.compile());
  assert.throws(() => f.storage.validateDispatch([...snapshots, f.write()]), {code: 'THREE_STORAGE_TEXTURE_FEEDBACK'});
});
test('integer reads observe each preceding producer without copying capture-time bytes', t => {
  const f = rig(t), snapshots = f.capture(f.compile()), write = f.write(), observed = [];
  for (const value of [4294967295, 16777217, 2147483648]) {
    new Uint32Array(f.storage.resolve(write).texture.pixels.buffer)[0] = value;
    f.storage.written([write]); observed.push(new Uint32Array(f.storage.resolve(snapshots[0]).texture.pixels.buffer)[0]);
  }
  assert.deepEqual(observed, [4294967295, 16777217, 2147483648]); assert.equal(f.device.events.length, 0);
});
test('integer graphics consumers preserve producer validation failures', async t => {
  const f = rig(t), snapshots = f.capture(f.compile()), write = f.write(), error = new Error('integer producer failed');
  f.storage.withTexture(write.entry, () => {}, write, {validation: Promise.reject(error)});
  await assert.rejects(f.storage.whenIdle(), error);
  assert.throws(() => f.storage.resolve(snapshots[0]), error);
});
test('device loss invalidates integer graphics inputs', async t => {
  const f = rig(t), snapshots = f.capture(f.compile()); f.lose({message: 'lost'}); await Promise.resolve();
  assert.throws(() => f.storage.resolve(snapshots[0]), {code: 'THREE_STORAGE_TEXTURE_DEVICE_LOST'});
});
test('renderer cancellation invalidates integer graphics inputs', t => {
  const f = rig(t), snapshots = f.capture(f.compile()), error = new Error('cancelled'); f.lifetime.abort(error);
  assert.throws(() => f.storage.resolve(snapshots[0]), error);
});
for (const dimension of dimensions) for (const type of ['u8', 'f16', 'f32']) for (const sampler of [false, true])
  test(`${dimension} ${type} sampler=${sampler}: floating-point bindings retain their layouts`, t => {
    const f = rig(t, {dimension, type}); f.texture.minFilter = f.texture.magFilter = 'nearest'; f.node.sampler = sampler;
    const d = f.compile(), snapshots = f.capture(d);
    const sample = d.groups[0].entries.find(entry => entry.texture);
    assert.equal(sample.texture.sampleType, type === 'f32' ? 'unfilterable-float' : 'float');
    assert.equal(d.groups[0].entries.length, sampler ? 2 : 1);
    snapshots.forEach((snapshot, i) => assert.deepEqual(f.storage.layout(snapshot), d.groups[0].entries[i].texture
      ? {texture: d.groups[0].entries[i].texture} : {sampler: d.groups[0].entries[i].sampler}));
  });
for (const type of ['u32', 'f32']) test(`depth ${type} is not misclassified as an integer color texture`, t => {
  const f = rig(t), depth = new THREE.DepthTexture(), target = new THREE.RenderTarget();
  Object.assign(depth, {type, isDepthTexture: true, renderTarget: target, minFilter: 'nearest', magFilter: 'nearest'});
  target.depthTexture = depth; f.node.value = depth;
  f.shared.layoutOf = () => ({texture: {sampleType: 'depth', viewDimension: '2d', multisampled: false}});
  assert.equal(f.compile().groups[0].entries[0].texture.sampleType, 'depth');
});

for (const type of ['i32', 'u32']) test(`${type}: float filtering features never change integer sample types`, t => {
  const f = rig(t, {type, features: ['float32-filterable']});
  assert.equal(f.compile().groups[0].entries[0].texture.sampleType, type === 'i32' ? 'sint' : 'uint');
});
for (const format of ['rgbi', 'unknown']) test(`unsupported integer ${format} retains owner format admission`, t => {
  const f = rig(t, {format});
  assert.throws(f.compile, {code: 'THREE_STORAGE_TEXTURE_FORMAT'});
  assert.equal(f.renderer.builds, 0); assert.equal(f.device.textures.length, 0);
});
test('integer bindings retain per-stage sample limits', t => {
  const f = rig(t); f.device.limits.maxSampledTexturesPerShaderStage = 1;
  f.material.colorNode = new Node([f.node, new TextureNode(new StorageTexture())]);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIMIT'}); assert.equal(f.device.textures.length, 0);
});
test('compatible integer reads preserve source object properties', t => {
  const f = rig(t), before = {...f.texture}, image = {...f.texture.image};
  const snapshots = f.capture(f.compile()); f.storage.resolve(snapshots[0]);
  assert.deepEqual({...f.texture}, before); assert.deepEqual(f.texture.image, image);
  assert.equal(Object.hasOwn(f.node, 'generateNodeType'), false, 'no application node type patches');
});
