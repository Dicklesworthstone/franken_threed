/** Production graph preparation and storage residency, with structural r186
 * descriptors, a recording builder/device and a controlled submission driver.
 * The three unrelated graph-context imports are isolated below. This suite
 * does not compile WGSL, execute a compute shader or instantiate the renderer.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {register} from 'node:module';
import {createThreeStorageTextures} from './three_storage_textures.mjs';

// Keep the actual program module intact. Fail if this texture-only fixture
// unexpectedly enters generated-attribute, shadow or render-pipeline behavior.
const unsupported = `const unexpected = () => { throw new Error('Unexpected graph context in storage-input fixture'); };`;
const contexts = {
  './three_node_attributes.mjs': `${unsupported}
    export const createThreeNodeAttributes = () => ({dispose() {}, register: unexpected, capture: unexpected, validate: unexpected});`,
  './three_node_shadows.mjs': `${unsupported}
    export const registerThreeNativeShadow = unexpected, updateThreeNativeShadow = unexpected;`,
  './three_node_passes.mjs': `${unsupported}
    export const inspectThreePipelineContext = () => null, inspectThreeMRT = () => null;
    export const inspectThreeNativePass = unexpected, updateThreeNativePass = unexpected;`,
};
register('data:text/javascript,' + encodeURIComponent(`
  const contexts = ${JSON.stringify(contexts)};
  export async function resolve(specifier, context, next) {
    if (context.parentURL && new URL(context.parentURL).pathname.endsWith('/three_node_program.mjs') &&
        Object.hasOwn(contexts, specifier))
      return {url: 'data:text/javascript,' + encodeURIComponent(contexts[specifier]), shortCircuit: true};
    return next(specifier, context);
  }
`), import.meta.url);
const {createThreeNodePrograms} = await import('./three_node_program.mjs');

let id = 0;
class Events {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, fn) { const set = this.listeners.get(type) ?? new Set(); set.add(fn); this.listeners.set(type, set); }
  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
  dispose() { for (const fn of [...this.listeners.get('dispose') ?? []]) fn({type: 'dispose', target: this}); }
}
class Texture extends Events {
  constructor() {
    super(); this.id = ++id; this.uuid = `texture-${id}`; this.isTexture = true;
    Object.assign(this, {image: {width: 4, height: 4}, source: {version: 0}, version: 0,
      type: 'u8', format: 'rgba', colorSpace: '', channel: 0, minFilter: 'nearest', magFilter: 'nearest',
      wrapS: 'clamp', wrapT: 'clamp', anisotropy: 1, compareFunction: null,
      generateMipmaps: false, mipmapsAutoUpdate: true, mipmaps: [], internalFormat: null});
  }
}
class StorageTexture extends Texture { constructor() { super(); this.isStorageTexture = true; } }
class StorageArrayTexture extends StorageTexture {
  constructor() { super(); this.isArrayTexture = true; this.image.depth = 3; this.mipmapsAutoUpdate = undefined; }
}
class DataArrayTexture extends Texture { constructor() { super(); this.isDataArrayTexture = this.isArrayTexture = true; this.image.depth = 3; } }
class Node {
  constructor(children = []) { this.isNode = true; this.children = children; }
  getChildren() { return this.children; }
}
class TextureNode extends Node {
  constructor(value, {stage = 2, sampler = true, layer = null} = {}) {
    super(); Object.assign(this, {value, isTextureNode: true, sampler, stage, depthNode: layer,
      levelNode: null, biasNode: null, gradNode: null, gatherNode: null, updateMatrix: true,
      groupNode: {name: 'object'}});
  }
  getBase() { return this; }
  isSampleCompare() { return false; }
}
class Material extends Events {
  constructor() { super(); this.isMaterial = true; this.version = 0; this.lights = false; this.fog = false; }
  onBeforeCompile() {}
  onBeforeRender() {}
  customProgramCacheKey() { return 'fixture'; }
}
class NodeMaterial extends Material {
  constructor() { super(); this.isNodeMaterial = true; }
  customProgramCacheKey() { return 'fixture-node'; }
}
class MeshBasicNodeMaterial extends NodeMaterial { constructor() { super(); this.isMeshBasicNodeMaterial = true; } }
class BufferGeometry extends Events { constructor() { super(); this.id = ++id; this.attributes = {}; this.morphAttributes = {}; } }
class Mesh { constructor(geometry, material) { this.geometry = geometry; this.material = material; } }
class WGSLNodeBuilder {
  constructor(object, renderer) { Object.assign(this, {object, renderer, context: {}}); }
  build() {
    this.renderer.builds++;
    this.nodes = new Set([this.material.colorNode, this.material.positionNode].filter(Boolean));
    for (const node of this.nodes) for (const child of node.getChildren()) this.nodes.add(child);
    if (this.material.map) this.nodes.add(new TextureNode(this.material.map));
    this.updateBeforeNodes = [...this.nodes].filter(node => node.before);
    this.updateNodes = [...this.nodes].filter(node => node.update);
    this.updateAfterNodes = [...this.nodes].filter(node => node.after);
    const bindings = [], seen = new Set();
    for (const node of this.nodes) if (node.isTextureNode && !seen.has(node.value.uuid)) {
      seen.add(node.value.uuid);
      if (node.sampler) bindings.push({textureNode: node, isSampler: true, visibility: node.stage});
      bindings.push({textureNode: node, isSampler: true, isSampledTexture: true, visibility: node.stage,
        store: node.forceStore === true});
    }
    this.bindings = [{name: 'object', bindings}]; this.structs = {fragment: []};
    this.vertexShader = '// recording vertex builder'; this.fragmentShader = '// recording fragment builder';
  }
  getBindings() { return this.bindings; }
  getAttributesArray() { return []; }
}
const THREE = {REVISION: '186', Texture, StorageTexture, StorageArrayTexture, DataArrayTexture,
  Node, TextureNode, Material, NodeMaterial, MeshBasicNodeMaterial, BufferGeometry, Mesh, WGSLNodeBuilder,
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
  'PointLight', 'PointLightNode', 'SpotLight', 'SpotLightNode', 'StandardNodeLibrary', 'BufferNode', 'UniformArrayNode', 'BufferAttributeNode',
  'PMREMNode', 'MaterialReferenceNode', 'ShadowNode', 'OutputStructNode', 'MRTNode', 'Fog', 'FogExp2', 'Color', 'Camera'])
  THREE[name] = class {};
THREE.DepthTexture = class extends Texture {}; THREE.CubeTexture = class extends Texture {};
THREE.CubeDepthTexture = class extends THREE.DepthTexture {}; THREE.RenderTarget = class {};

function gpu(features = []) {
  const textures = [], events = [], samplers = [], scopes = [];
  let lose; const lost = new Promise(resolve => { lose = resolve; });
  const device = {features: new Set(features), textures, events, samplers, lost, lose,
    limits: {maxTextureDimension2D: 64, maxTextureArrayLayers: 8, maxComputeWorkgroupsPerDimension: 65535},
    pushErrorScope(type) { scopes.push(type); }, popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(null); },
    createTexture(descriptor) {
      const bytes = descriptor.format === 'rgba32float' ? 16 : ['rgba16float', 'rg32float'].includes(descriptor.format) ? 8 : 4;
      const native = {descriptor, pixels: new Uint8Array(descriptor.size.reduce((a, b) => a * b, 1) * bytes), destroyed: false,
        createView(options = {}) { return {texture: native, options}; }, destroy() { this.destroyed = true; }};
      textures.push(native); return native;
    },
    createSampler(descriptor) { const value = {...descriptor}; samplers.push(value); return value; },
    createShaderModule(descriptor) { return descriptor; }, createBindGroupLayout(descriptor) { return descriptor; },
    createPipelineLayout(descriptor) { return descriptor; }, createBindGroup(descriptor) { return descriptor; },
    createRenderPipeline(descriptor) { return {descriptor, getBindGroupLayout: () => ({})}; },
    createComputePipeline(descriptor) { return {descriptor, getBindGroupLayout: () => ({})}; },
    createCommandEncoder(descriptor) {
      const commands = [];
      const pass = kind => ({setPipeline() {}, setBindGroup() {}, draw() {}, dispatchWorkgroups() {}, end() { commands.push(kind); }});
      return {beginRenderPass: () => pass('render-mip'), beginComputePass: () => pass('array-mip'),
        finish: () => ({descriptor, commands})};
    },
  };
  device.queue = {submit(commands) { events.push(...commands); }, onSubmittedWorkDone() { events.push('fence'); return Promise.resolve(); }};
  return device;
}
function rig(t, {features = [], maxBytes, maxTextures} = {}) {
  const device = gpu(features), lifetime = new AbortController();
  const storage = createThreeStorageTextures(THREE, {getDevice: () => device, signal: lifetime.signal, maxBytes, maxTextures});
  // This fixture drives the same storage-owner methods as the renderer's
  // unified sampled-input adapter. No CPU upload pool exists in this fixture.
  const shared = {layoutOf: binding => storage.layout(storage.captureBinding(binding))};
  const frame = {updateNode: node => node.update?.(), updateBeforeNode: node => node.before?.(), updateAfterNode: node => node.after?.()};
  const renderer = {_device: device, _nodes: {nodeFrame: frame}, info: {calls: 1}, builds: 0,
    hasFeature: name => device.features.has(name), getRenderTarget: () => null,
    _getSampledTextureInputs: () => shared,
    backend: {utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}}};
  const owner = createThreeNodePrograms({three: THREE, renderer});
  const material = new MeshBasicNodeMaterial(), object = new Mesh(new BufferGeometry(), material);
  const texture = new StorageTexture(), node = new TextureNode(texture); material.colorNode = node;
  t.after(() => { owner.dispose(); storage.dispose(); });
  const compile = () => owner.compile(material, object);
  const capture = descriptor => owner.captureDraw(descriptor).textures.map(binding => storage.captureBinding(binding));
  return {device, storage, shared, renderer, owner, material, object, texture, node, compile, capture, lifetime};
}
const formats = [
  ['rgba8unorm', 'u8', 'rgba'], ['rgba16float', 'f16', 'rgba'], ['r32float', 'f32', 'r'],
  ['rg32float', 'f32', 'rg'], ['rgba32float', 'f32', 'rgba'],
];
for (const array of [false, true]) for (const [format, type, channels] of formats) {
  test(`${array ? 'array' : '2D'} ${format}: graph layouts agree with the authoritative storage owner`, t => {
    const f = rig(t), texture = array ? new StorageArrayTexture() : f.texture;
    texture.type = type; texture.format = channels; f.node.value = texture;
    if (array) f.node.depthNode = new Node();
    const d = f.compile(), snapshots = f.capture(d), entries = d.groups[0].entries;
    assert.equal(f.device.textures.length, 0, 'compilation and capture do not allocate a GPU image');
    snapshots.forEach((snapshot, i) => assert.deepEqual(f.storage.layout(snapshot),
      entries[i].texture ? {texture: entries[i].texture} : {sampler: entries[i].sampler}));
    assert.equal(entries[1].texture.viewDimension, array ? '2d-array' : '2d');
    assert.equal(entries[1].texture.sampleType, type === 'f32' ? 'unfilterable-float' : 'float');
    const write = f.storage.captureBinding({texture, isSampledTexture: true, store: true, access: 'writeOnly'});
    const producer = f.storage.resolve(write).texture; producer.pixels.fill(73); f.storage.written([write]);
    f.storage.validateDispatch(snapshots);
    const consumer = f.storage.resolve(snapshots[1]);
    assert.equal(consumer.texture, producer); assert.ok(consumer.texture.pixels.every(value => value === 73));
    assert.equal(f.device.textures.length, 1); assert.equal(f.device.events.length, 0, 'reading adds no transfer or fence');
    assert.equal(f.compile(), d, 'GPU content updates do not recompile the graph');
  });
}
for (const stage of [1, 2, 3]) test(`storage sampling retains stage visibility ${stage}`, t => {
  const f = rig(t); f.node.stage = stage; const d = f.compile();
  for (const binding of d.groups[0].entries) assert.equal(binding.visibility, stage);
});
test('unfiltered texture loads need no sampler binding', t => {
  const f = rig(t); f.node.sampler = false; f.texture.type = 'f32'; f.texture.format = 'r';
  const d = f.compile(), snapshots = f.capture(d);
  assert.equal(snapshots.length, 1); assert.equal(snapshots[0].kind, 'sampled');
  f.storage.resolve(snapshots[0]); assert.equal(f.device.samplers.length, 0);
});
test('linear float32 sampling is admitted only with the real device feature', t => {
  const f = rig(t, {features: ['float32-filterable']});
  f.texture.type = 'f32'; f.texture.minFilter = f.texture.magFilter = 'linear';
  const d = f.compile(), snapshots = f.capture(d);
  assert.equal(d.groups[0].entries[0].sampler.type, 'filtering');
  assert.equal(f.storage.layout(snapshots[0]).sampler.type, 'filtering');
  assert.equal(d.groups[0].entries[1].texture.sampleType, 'float');
  assert.equal(f.storage.resolve(snapshots[0]).minFilter, 'linear');
});
test('linear float32 sampling without its feature fails before builder or GPU work', t => {
  const f = rig(t); f.texture.type = 'f32'; f.texture.minFilter = f.texture.magFilter = 'linear';
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(f.renderer.builds, 0); assert.equal(f.device.textures.length, 0);
});
test('compatible texture replacement reuses the graph but queued bindings retain the earlier image', t => {
  const f = rig(t), d = f.compile(), first = f.capture(d), secondTexture = new StorageTexture();
  f.node.value = secondTexture; assert.equal(f.compile(), d); const second = f.capture(d);
  const a = f.storage.resolve(first[1]), b = f.storage.resolve(second[1]);
  assert.notEqual(a.texture, b.texture); assert.equal(first[1].entry.texture, f.texture);
  assert.equal(second[1].entry.texture, secondTexture);
  a.texture.pixels.fill(11); b.texture.pixels.fill(29);
  assert.equal(f.storage.resolve(first[1]).texture.pixels[0], 11);
  assert.equal(f.storage.resolve(second[1]).texture.pixels[0], 29);
});
test('the ordered consumer sees preceding GPU writes, not capture-time pixels', t => {
  const f = rig(t), d = f.compile(), snapshots = f.capture(d), samples = [];
  const write = f.storage.captureBinding({texture: f.texture, isSampledTexture: true, store: true, access: 'writeOnly'});
  const queue = [
    () => { f.storage.resolve(write).texture.pixels.fill(7); f.storage.written([write]); },
    () => { f.storage.validateDispatch(snapshots); samples.push(f.storage.resolve(snapshots[1]).texture.pixels.slice()); },
    () => { f.storage.resolve(write).texture.pixels.fill(83); f.storage.written([write]); },
    () => { f.storage.validateDispatch(snapshots); samples.push(f.storage.resolve(snapshots[1]).texture.pixels.slice()); },
  ];
  queue.forEach(run => run());
  assert.ok(samples[0].every(value => value === 7)); assert.ok(samples[1].every(value => value === 83));
  assert.equal(f.device.textures.length, 1); assert.equal(f.device.events.length, 0);
});
test('source sampler state is captured independently for queued draws', t => {
  const f = rig(t), d = f.compile(), first = f.capture(d);
  f.texture.wrapS = 'repeat'; f.texture.wrapT = 'mirror';
  assert.equal(f.compile(), d); const second = f.capture(d);
  assert.equal(f.storage.resolve(first[0]).addressModeU, 'clamp-to-edge');
  assert.equal(f.storage.resolve(second[0]).addressModeU, 'repeat');
  assert.equal(f.storage.resolve(second[0]).addressModeV, 'mirror-repeat');
});
test('2D/array replacement rebuilds the graph and rejects the stale descriptor', t => {
  const f = rig(t), d = f.compile(); f.node.value = new StorageArrayTexture();
  assert.throws(() => f.owner.captureDraw(d), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const next = f.compile(); assert.notEqual(next, d); assert.equal(next.groups[0].entries[1].texture.viewDimension, '2d-array');
});
test('texture alias partition changes rebuild bindings, while live content does not', t => {
  const f = rig(t), sibling = new TextureNode(f.texture);
  f.material.colorNode = new Node([f.node, sibling]); const d = f.compile();
  assert.equal(d.groups[0].entries.length, 2);
  sibling.value = new StorageTexture();
  assert.throws(() => f.owner.captureDraw(d), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const next = f.compile(); assert.notEqual(next, d); assert.equal(next.groups[0].entries.length, 4);
});
test('2D storage images work in an admitted material map slot', t => {
  const f = rig(t); f.material.map = f.texture; const d = f.compile();
  assert.equal(d.groups[0].entries.length, 2, 'source identity aliases the authored sample and map');
  assert.ok(f.owner.captureDraw(d).textures.every(binding => binding.texture === f.texture));
});
for (const change of [texture => { texture.image.width = 8; }, texture => texture.dispose(),
  texture => { texture.type = 'f16'; }, texture => { texture.generateMipmaps = true; }]) {
  test(`accepted storage bindings reject generation changes: ${change.toString()}`, t => {
    const f = rig(t), snapshots = f.capture(f.compile()); change(f.texture);
    assert.throws(() => f.storage.validateDispatch(snapshots), {code: 'THREE_STORAGE_TEXTURE_STALE'});
    assert.equal(f.device.textures.length, 0);
  });
}
for (const [name, change, code] of [
  ['integer formats', t => { t.type = 'u32'; }, 'THREE_NODE_PROGRAM_TEXTURE'],
  ['volume', t => { t.is3DTexture = true; t.image.depth = 4; }, 'THREE_NODE_PROGRAM_TEXTURE'],
  ['CPU pixel shadow', t => { t.image.data = new Uint8Array(64); }, 'THREE_STORAGE_TEXTURE_FORMAT'],
  ['sRGB storage', t => { t.colorSpace = 'srgb'; }, 'THREE_STORAGE_TEXTURE_FORMAT'],
  ['invalid extent', t => { t.image.width = 0; }, 'THREE_STORAGE_TEXTURE_SHAPE'],
  ['depth storage', t => { t.isDepthTexture = true; }, 'THREE_NODE_PROGRAM_TEXTURE'],
]) test(`${name} remains rejected before callbacks or GPU allocation`, t => {
  const f = rig(t); let calls = 0; f.node.before = () => calls++; change(f.texture);
  assert.throws(f.compile, {code}); assert.equal(calls, 0); assert.equal(f.device.textures.length, 0);
});
test('graphics storage-store nodes stay rejected', t => {
  const f = rig(t); f.node.isStorageTextureNode = true;
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});
test('writable bindings introduced by a builder stay rejected', t => {
  const f = rig(t); f.node.forceStore = true;
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_BINDING'});
});
test('storage arrays do not implicitly become 2D material maps', t => {
  const f = rig(t); f.material.map = new StorageArrayTexture();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});
test('a 2D storage sample cannot silently gain an array-layer coordinate', t => {
  const f = rig(t); f.node.depthNode = new Node();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});
for (const layout of [null, {}, {texture: {sampleType: 'float', viewDimension: 'cube', multisampled: false}},
  {texture: {sampleType: 'float', viewDimension: '2d', multisampled: true}}]) {
  test(`storage admission refuses missing or contradictory owner layout ${JSON.stringify(layout)}`, t => {
    const f = rig(t); f.shared.layoutOf = () => layout;
    assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'}); assert.equal(f.renderer.builds, 0);
  });
}
test('storage admission cannot fall back to an inferred CPU texture path', t => {
  const f = rig(t); delete f.renderer._getSampledTextureInputs;
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});
test('shared allocation and device limits still apply to graphics consumers', t => {
  const f = rig(t, {maxBytes: 63}), snapshots = f.capture(f.compile());
  assert.throws(() => f.storage.resolve(snapshots[1]), {code: 'THREE_STORAGE_TEXTURE_BUDGET'});
  assert.equal(f.device.textures.length, 0);
});
test('graphics consumers cannot bypass device array layer limits', t => {
  const f = rig(t); f.node.value = new StorageArrayTexture(); f.node.value.image.depth = 9;
  const snapshots = f.capture(f.compile());
  assert.throws(() => f.storage.resolve(snapshots[1]), {code: 'THREE_STORAGE_TEXTURE_LIMIT'});
  assert.equal(f.device.textures.length, 0);
});
for (const array of [false, true]) test(`${array ? 'array' : '2D'} automatic mips remain shared with compute`, t => {
  const f = rig(t), texture = array ? new StorageArrayTexture() : f.texture;
  f.node.value = texture; texture.generateMipmaps = true; texture.mipmapsAutoUpdate = true;
  const d = f.compile(), snapshots = f.capture(d);
  const write = f.storage.captureBinding({texture, isSampledTexture: true, store: true, access: 'writeOnly'});
  const producer = f.storage.resolve(write).texture; f.storage.written([write]);
  const before = f.device.events.length; assert.equal(before, 1);
  const view = f.storage.resolve(snapshots[1]); assert.equal(view.texture, producer);
  assert.equal(f.device.events.length, before, 'the draw does not regenerate an already-current mip chain');
  assert.equal(f.device.textures.length, 1);
});
test('a dispatch cannot combine the graphics sampled view with a writer to that same image', t => {
  const f = rig(t), snapshots = f.capture(f.compile());
  const write = f.storage.captureBinding({texture: f.texture, isSampledTexture: true, store: true, access: 'writeOnly'});
  assert.throws(() => f.storage.validateDispatch([...snapshots, write]), {code: 'THREE_STORAGE_TEXTURE_FEEDBACK'});
});
test('device loss invalidates accepted graphics storage inputs', async t => {
  const f = rig(t), snapshots = f.capture(f.compile()); f.device.lose({message: 'lost-test'}); await Promise.resolve();
  assert.throws(() => f.storage.resolve(snapshots[1]), {code: 'THREE_STORAGE_TEXTURE_DEVICE_LOST'});
});
test('renderer lifetime cancellation invalidates accepted graphics storage inputs', t => {
  const f = rig(t), snapshots = f.capture(f.compile()), error = new Error('renderer stopped'); f.lifetime.abort(error);
  assert.throws(() => f.storage.resolve(snapshots[1]), error);
});
test('source update callbacks run once per capture, not during admission or cached compilation', t => {
  const f = rig(t); let before = 0, current = 0, after = 0;
  f.node.before = () => before++; f.node.update = () => current++; f.node.after = () => after++;
  const d = f.compile(); assert.equal(f.compile(), d); assert.deepEqual([before, current, after], [0, 0, 0]);
  f.capture(d); f.owner.after(d); assert.deepEqual([before, current, after], [1, 1, 1]);
  f.capture(d); f.owner.after(d); assert.deepEqual([before, current, after], [2, 2, 2]);
});

test('ordinary 2D/data-array inputs retain their previous graph layout', t => {
  const f = rig(t);
  f.shared.layoutOf = ({texture}) => ({texture: {sampleType: 'float',
    viewDimension: texture.isDataArrayTexture ? '2d-array' : '2d', multisampled: false}});
  for (const texture of [new Texture(), new DataArrayTexture()]) {
    f.node.value = texture; f.node.depthNode = texture.isDataArrayTexture ? new Node() : null;
    const d = f.compile();
    assert.equal(d.groups[0].entries[1].texture.viewDimension, texture.isDataArrayTexture ? '2d-array' : '2d');
    assert.ok(f.owner.captureDraw(d).textures.every(binding => binding.texture === texture));
  }
  assert.equal(f.storage.size, 0, 'ordinary textures never enter storage residency');
});
test('switching between CPU and storage ownership rebuilds an otherwise identical graph', t => {
  const f = rig(t), storageLayout = f.shared.layoutOf;
  f.shared.layoutOf = binding => binding.texture.isStorageTexture ? storageLayout(binding)
    : {texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}};
  const storage = f.compile(); f.node.value = new Texture();
  assert.throws(() => f.owner.captureDraw(storage), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const cpu = f.compile(); assert.notEqual(cpu, storage);
  f.node.value = f.texture;
  assert.throws(() => f.owner.captureDraw(cpu), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.notEqual(f.compile(), cpu);
});
test('manual storage mip writes do not trigger an automatic graphics-side regeneration', t => {
  const f = rig(t); f.texture.generateMipmaps = true; f.texture.mipmapsAutoUpdate = false;
  const snapshots = f.capture(f.compile());
  const write = f.storage.captureBinding({texture: f.texture, isSampledTexture: true,
    store: true, access: 'writeOnly', mipLevel: 1});
  const native = f.storage.resolve(write).texture; f.storage.written([write]);
  assert.equal(f.storage.resolve(snapshots[1]).texture, native);
  assert.equal(f.device.events.length, 0, 'the owner preserves manually produced levels');
});
for (const features of [[], ['float32-filterable']]) test(`integer compute samplers stay non-filtering with features ${features}`, t => {
  const f = rig(t, {features}); f.texture.type = 'u32';
  const sampler = f.storage.captureBinding({texture: f.texture, isSampler: true});
  assert.equal(f.storage.layout(sampler).sampler.type, 'non-filtering');
  assert.equal(f.storage.resolve(sampler).minFilter, 'nearest');
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'}, 'graphics still excludes integer samples');
});
test('filterable RGBA8 stays compatible with a filtering layout even for nearest filters', t => {
  const f = rig(t), d = f.compile(), snapshots = f.capture(d);
  assert.equal(d.groups[0].entries[0].sampler.type, 'filtering');
  assert.equal(f.storage.layout(snapshots[0]).sampler.type, 'filtering');
  assert.equal(f.storage.resolve(snapshots[0]).minFilter, 'nearest');
});
test('producer validation failures are not bypassed by a later graphics consumer', async t => {
  const f = rig(t), snapshots = f.capture(f.compile());
  const entry = f.storage.capture(f.texture), error = new Error('producer validation failed');
  f.storage.withTexture(entry, () => {}, f.storage.captureBinding({texture: f.texture,
    isSampledTexture: true, store: true, access: 'writeOnly'}), {validation: Promise.reject(error)});
  await assert.rejects(f.storage.whenIdle(), error);
  assert.throws(() => f.storage.resolve(snapshots[1]), error);
  assert.equal(f.storage.allocatedBytes, 0);
});
test('graphics preparation never modifies source texture pixels, identity, versions or ownership flags', t => {
  const f = rig(t), texture = f.texture;
  const before = {id: texture.id, uuid: texture.uuid, source: texture.source, image: texture.image,
    version: texture.version, sourceVersion: texture.source.version, storage: texture.isStorageTexture,
    renderTarget: texture.isRenderTargetTexture, keys: Object.keys(texture)};
  const snapshots = f.capture(f.compile()); f.storage.resolve(snapshots[1]);
  assert.deepEqual({id: texture.id, uuid: texture.uuid, source: texture.source, image: texture.image,
    version: texture.version, sourceVersion: texture.source.version, storage: texture.isStorageTexture,
    renderTarget: texture.isRenderTargetTexture, keys: Object.keys(texture)}, before);
  assert.equal(texture.image.data, undefined);
});
