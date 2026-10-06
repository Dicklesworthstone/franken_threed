/** Source-owned HDR panoramas for the native GGX environment renderer.
 * The source Texture and pixels are borrowed. Preparation copies current ready
 * pixels and filters once; needsUpdate/identity changes require a new owner.
 * This is the core's explicit GGX/DFG profile, not Three PMREM pixel parity.
 */
import {createGpuAnimationEnvironment, planAnimationEnvironment} from './animation_environment.mjs';

export class ThreeEnvironmentError extends Error {
  constructor(code, message) {
    super(`THREE_ENVIRONMENT_${code}: ${message}`);
    this.name = 'ThreeEnvironmentError'; this.code = `THREE_ENVIRONMENT_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeEnvironmentError(code, message); };
const integer = (n, min, max, label) => {
  if (!Number.isSafeInteger(n) || n < min || n > max) fail('LIMIT', `Invalid ${label}`);
  return n;
};
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/** Metadata-only check: never scan or copy panorama pixels in a render loop. */
export function inspectThreeEnvironment(texture, three, {maxPixels = 16 * 1024 * 1024} = {}) {
  if (texture?.isF3DSceneEnvironment === true) return inspectSceneCapture(texture, three, maxPixels);
  if (three?.REVISION !== '186' || typeof three.DataTexture !== 'function' ||
      !(texture instanceof three.DataTexture) || texture.isCubeTexture || texture.isRenderTargetTexture)
    fail('SOURCE', 'Supply a ready r186 DataTexture panorama');
  integer(maxPixels, 1, Number.MAX_SAFE_INTEGER, 'pixel budget');
  const half = texture.type === three.HalfFloatType;
  if ((!half && texture.type !== three.FloatType) || texture.format !== three.RGBAFormat ||
      texture.internalFormat !== null || texture.compareFunction != null)
    fail('FORMAT', 'Expected RGBA half-float or float storage without internal overrides');
  if (texture.mapping !== three.EquirectangularReflectionMapping ||
      ![three.LinearSRGBColorSpace, three.NoColorSpace].includes(texture.colorSpace))
    fail('PROFILE', 'Expected a linear-sRGB equirectangular reflection panorama');
  if (typeof texture.flipY !== 'boolean' || texture.premultiplyAlpha !== false ||
      ![1, 2, 4, 8].includes(texture.unpackAlignment))
    fail('FORMAT', 'Expected explicit row orientation and unpremultiplied HDR pixels');
  if (texture.onUpdate != null || !Array.isArray(texture.updateRanges) || texture.updateRanges.length ||
      !Array.isArray(texture.mipmaps) || texture.mipmaps.length)
    fail('PROFILE', 'Upload hooks, partial updates and authored source mipmaps need another profile');
  const image = texture.image, data = image?.data, source = texture.source;
  if (!source || source.dataReady !== true || source.data !== image)
    fail('NOT_READY', 'Provide ready source pixels before preparation');
  integer(texture.version, 1, Number.MAX_SAFE_INTEGER, 'texture version');
  integer(source.version, 0, Number.MAX_SAFE_INTEGER, 'source version');
  const width = integer(image?.width, 2, 32768, 'panorama width');
  const height = integer(image?.height, 1, 16384, 'panorama height');
  if (width !== 2 * height || width * height > maxPixels)
    fail('LIMIT', 'Expected a 2:1 panorama within the pixel budget');
  if (!(data instanceof (half ? Uint16Array : Float32Array)) || !(data.buffer instanceof ArrayBuffer) ||
      data.buffer.resizable || data.length !== width * height * 4)
    fail('STORAGE', 'Expected complete, fixed, unshared RGBA source storage');
  try { new Uint8Array(data.buffer, 0, 0); } catch { fail('STORAGE', 'Source pixels are detached'); }
  return Object.freeze({texture, source, image, data, half, width, height, sourceBytes: width * height * 8,
    signature: Object.freeze([texture, source, image, data, data.buffer, texture.version, source.version,
      texture.type, texture.mapping, texture.colorSpace, texture.flipY, width, height])});
}

/** A PMREMGenerator.fromScene() request from three_renderer.mjs: the capture
 * runs at this owner's preparation boundary (not at the source call). */
function inspectSceneCapture(texture, three, maxPixels) {
  const c = texture.f3dCapture;
  if (three?.REVISION !== '186' || !c || !(c.scene instanceof three.Scene) || !(c.position instanceof three.Vector3))
    fail('SOURCE', 'Expected a captured r186 source scene environment');
  integer(c.size, 1, 4096, 'capture size');
  if (![c.near, c.far, c.sigma].every(v => typeof v === 'number' && Number.isFinite(v)) || c.near <= 0 || c.far <= c.near || c.sigma < 0)
    fail('VALUE', 'Invalid environment capture near/far/sigma');
  if (c.size * c.size * 6 > maxPixels) fail('LIMIT', 'Environment capture exceeds the pixel budget');
  return Object.freeze({texture, capture: c, width: c.size, height: c.size, sourceBytes: c.size * c.size * 6 * 8 + c.size * c.size * 4,
    signature: Object.freeze([texture, c, texture.version])});
}

// Cube faces as WebGPU samples them: forward and up per layer +X,-X,+Y,-Y,+Z,-Z.
// A camera with this forward/up sees the face mirrored horizontally (cube maps
// are left-handed), so each face is blitted with u -> 1 - u into its layer.
const CUBE_FACES = [[[1, 0, 0], [0, 1, 0]], [[-1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, -1]],
  [[0, -1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0]], [[0, 0, -1], [0, 1, 0]]];
const FLIP_BLIT_WGSL = `
@group(0) @binding(0) var face_texture: texture_2d<f32>;
struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> Out {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  var out: Out; out.position = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0); out.uv = vec2<f32>(p.x, 1.0 - p.y); return out;
}
@fragment fn fragment_main(input: Out) -> @location(0) vec4<f32> {
  let size = vec2<f32>(textureDimensions(face_texture));
  let texel = vec2<i32>(i32(size.x - 1.0 - floor(input.uv.x * size.x)), i32(floor(input.uv.y * size.y)));
  return textureLoad(face_texture, clamp(texel, vec2<i32>(0), vec2<i32>(size) - 1), 0);
}`;

/** Render the captured source scene into an rgba16float cube texture with the
 * same native scene bridge (a 90-degree camera per face at the capture
 * position, black clear unless the scene has a color background), then flip
 * each face into its cube layer. Upstream's optional sigma pre-blur is not
 * reproduced; the GGX filter below replaces PMREM's own filtering. */
async function renderSceneCapture(device, shape, three, signal) {
  const {scene, near, far, size, position} = shape.capture;
  const {createGpuThreeScene} = await import('./three_scene.mjs');
  // The bridge owns its own validation scopes across its asynchronous preparation.
  const bridge = await createGpuThreeScene(device, scene, {three, signal, maxBindings: 4096,
    renderer: {format: 'rgba16float', depthFormat: 'depth24plus', sampleCount: 1, maxDraws: 4096},
    ...(scene.fog ? {fog: {}} : {})});
  let cube = null, face = null, depth = null, thrown = null;
  // All capture work below is synchronous: push, issue, pop with no await between.
  device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
  try {
    cube = device.createTexture({label: 'f3d-environment-capture', size: [size, size, 6], dimension: '2d',
      format: 'rgba16float', mipLevelCount: 1, sampleCount: 1, usage: 4 | 16});
    face = device.createTexture({label: 'f3d-environment-face', size: [size, size, 1], dimension: '2d',
      format: 'rgba16float', mipLevelCount: 1, sampleCount: 1, usage: 4 | 16});
    depth = device.createTexture({label: 'f3d-environment-depth', size: [size, size, 1], dimension: '2d',
      format: 'depth24plus', mipLevelCount: 1, sampleCount: 1, usage: 16});
    const module = device.createShaderModule({label: 'f3d-environment-flip', code: FLIP_BLIT_WGSL});
    const pipeline = device.createRenderPipeline({label: 'f3d-environment-flip', layout: 'auto',
      vertex: {module, entryPoint: 'vertex_main'},
      fragment: {module, entryPoint: 'fragment_main', targets: [{format: 'rgba16float'}]}, primitive: {topology: 'triangle-list'}});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: face.createView()}]});
    const camera = new three.PerspectiveCamera(90, 1, near, far);
    const clear = scene.background?.isColor ? [scene.background.r, scene.background.g, scene.background.b, 1] : [0, 0, 0, 1];
    for (let layer = 0; layer < 6; layer++) {
      const [forward, up] = CUBE_FACES[layer];
      camera.position.copy(position); camera.up.set(...up);
      camera.lookAt(position.x + forward[0], position.y + forward[1], position.z + forward[2]);
      camera.updateMatrixWorld(true);
      bridge.render(camera, {colorView: face.createView(), depthView: depth.createView(), loadOp: 'clear', clearColor: clear});
      const encoder = device.createCommandEncoder({label: 'f3d-environment-flip'});
      const pass = encoder.beginRenderPass({colorAttachments: [{view: cube.createView({dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1}),
        loadOp: 'clear', storeOp: 'store', clearValue: {r: 0, g: 0, b: 0, a: 0}}]});
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
      device.queue.submit([encoder.finish()]);
    }
  } catch (error) { thrown = error; }
  const validation = device.popErrorScope(), oom = device.popErrorScope();
  try {
    const errors = await Promise.all([validation, oom]);
    if (thrown) throw thrown;
    if (errors.some(Boolean)) fail('DEVICE', errors.find(Boolean).message ?? 'Environment capture failed');
    await bridge.whenIdle();
    return cube;
  } catch (error) { cube?.destroy(); throw error; }
  finally { bridge.dispose(); face?.destroy(); depth?.destroy(); }
}

// Binary32 -> binary16, round-to-nearest/ties-to-even. Overflow rejects rather
// than clipping HDR energy. Binary16 inputs preserve every finite source bit.
function halfBits(value, scratch) {
  if (!Number.isFinite(value) || Math.abs(value) > 65504) fail('VALUE', 'HDR component exceeds finite half-float');
  scratch.f[0] = value;
  const bits = scratch.u[0], sign = (bits >>> 16) & 0x8000, exponent = ((bits >>> 23) & 255) - 127;
  let mantissa = bits & 0x7fffff;
  if (exponent < -25) return sign;
  let shift, result;
  if (exponent < -14) { mantissa |= 0x800000; shift = -exponent - 1; result = 0; }
  else { shift = 13; result = (exponent + 15) << 10; }
  let rounded = mantissa >>> shift;
  const remainder = mantissa & ((1 << shift) - 1), halfway = 1 << (shift - 1);
  if (remainder > halfway || (remainder === halfway && (rounded & 1))) rounded++;
  return sign | (result + rounded);
}
function capture(shape) {
  const {data, half, width, height, texture} = shape, pixels = new Uint16Array(width * height * 4);
  const f = new Float32Array(1), scratch = {f, u: new Uint32Array(f.buffer)};
  for (let y = 0; y < height; y++) {
    // r186 equirectUv uses asin(y)/PI+.5, while the core uses acos(y)/PI.
    // The core's north-first upload therefore reverses UNFLIPPED source rows.
    const row = texture.flipY ? y : height - 1 - y;
    for (let x = 0; x < width * 4; x++) {
      const v = data[row * width * 4 + x];
      if (half && (v & 0x7c00) === 0x7c00) fail('VALUE', 'HDR pixels contain infinity or NaN');
      pixels[y * width * 4 + x] = half ? v : halfBits(v, scratch);
    }
  }
  return pixels;
}

/** Copy the same bounded, oriented binary16 panorama for raw backgrounds.
 * No GPU allocation, filtering, source mutation or upload acknowledgement.
 * Consumers must preflight their own aggregate byte budget before this copy.
 */
export function captureThreeEnvironmentPixels(texture, three, options) {
  return capture(inspectThreeEnvironment(texture, three, options));
}

/** Own a filtered map, not a source Texture. maxBytes bounds transient panorama,
 * filter uniforms and output maps together; callers charge retained OLD maps
 * separately when replacing an owner. No sampling quality is silently reduced.
 */
export async function createGpuThreeEnvironment(device, texture, {
  three, maxBytes = 128 * 1024 * 1024, maxPixels = 16 * 1024 * 1024,
  size = 128, diffuseSize = 32, lutSize = 128, samples = 1024,
  maxSampleWork = 256 * 1024 * 1024, signal,
} = {}) {
  const shape = inspectThreeEnvironment(texture, three, {maxPixels});
  integer(maxBytes, 1, Number.MAX_SAFE_INTEGER, 'environment budget');
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' ||
      typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function'))
    fail('OPTIONS', 'Expected AbortSignal');
  if (typeof device?.createTexture !== 'function' || typeof device.queue?.writeTexture !== 'function' ||
      typeof device.lost?.then !== 'function') fail('DEVICE', 'Expected a WebGPU device');
  const limit = integer(device.limits?.maxTextureDimension2D, 1, 32768, 'device texture limit');
  if (shape.width > limit || shape.height > limit) fail('LIMIT', 'Panorama exceeds device texture size');
  const plan = planAnimationEnvironment({size, diffuseSize, lutSize, samples, maxSampleWork,
    maxTextureBytes: maxBytes, alignment: device.limits.minUniformBufferOffsetAlignment, maxDimension: limit});
  if (shape.sourceBytes + plan.textureBytes + plan.uniformBytes > maxBytes)
    fail('LIMIT', 'Panorama, filter uniforms and output maps exceed the old-plus-new budget');
  if (signal?.aborted) fail('ABORTED', 'Environment preparation was aborted');
  const pixels = shape.capture ? null : capture(shape), lifetime = new AbortController();
  let input = null, map = null, terminal = null, disposed = false, rejectStop;
  const stopped = new Promise((_, reject) => { rejectStop = reject; }); stopped.catch(() => {});
  function release() {
    texture.removeEventListener('dispose', onSourceDispose);
    signal?.removeEventListener('abort', onAbort);
    lifetime.abort(); map?.dispose(); map = null; input?.destroy(); input = null;
  }
  function stop(error) { if (!terminal && !disposed) { terminal = error; rejectStop(error); release(); } }
  function onAbort() { stop(new ThreeEnvironmentError('ABORTED', 'Source environment lifetime was aborted')); }
  function onSourceDispose() { stop(new ThreeEnvironmentError('SOURCE_DISPOSED', 'Source panorama was disposed')); }
  function live() {
    if (disposed) fail('DISPOSED', 'Source environment is disposed');
    if (terminal) throw terminal;
    if (map?.failed || map?.disposed) { stop(new ThreeEnvironmentError('DEVICE', 'Filtered environment failed')); throw terminal; }
  }
  function check() {
    live();
    if (!same(shape.signature, inspectThreeEnvironment(texture, three, {maxPixels}).signature))
      fail('PREPARE', 'Source panorama changed; call prepare() to filter its current version');
  }
  texture.addEventListener('dispose', onSourceDispose);
  signal?.addEventListener('abort', onAbort, {once: true});
  device.lost.then(info => stop(new ThreeEnvironmentError('DEVICE', info?.message ?? 'GPU device lost')), stop);
  try {
    if (signal?.aborted) onAbort(); check();
    let error, errors = [];
    if (shape.capture) {
      const capturing = renderSceneCapture(device, shape, three, lifetime.signal).then(value => {
        if (disposed || terminal) { value.destroy(); throw terminal ?? new ThreeEnvironmentError('DISPOSED', 'Environment is disposed'); }
        input = value; return value;
      });
      await Promise.race([capturing, stopped]);
    } else {
      device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
      try {
        input = device.createTexture({label: 'f3d-source-environment', size: [shape.width, shape.height, 1],
          dimension: '2d', format: 'rgba16float', mipLevelCount: 1, sampleCount: 1, usage: 2 | 4});
        device.queue.writeTexture({texture: input}, pixels, {bytesPerRow: shape.width * 8}, [shape.width, shape.height, 1]);
      } catch (cause) { error = cause; }
      const validation = device.popErrorScope(), oom = device.popErrorScope();
      errors = await Promise.race([Promise.all([validation, oom]), stopped]);
    }
    if (error) throw error;
    if (errors.some(Boolean)) fail('DEVICE', errors.find(Boolean).message ?? 'HDR upload failed');
    check();
    const constructing = createGpuAnimationEnvironment(device, input, {size, diffuseSize, lutSize, samples,
      maxTextureBytes: plan.textureBytes, maxSampleWork, signal: lifetime.signal}).then(value => {
      if (disposed || terminal) { value.dispose(); throw terminal ?? new ThreeEnvironmentError('DISPOSED', 'Environment is disposed'); }
      map = value; return value;
    });
    await Promise.race([constructing, stopped]); check();
    input.destroy(); input = null;
  } catch (error) { stop(error); throw error; }
  const owner = Object.freeze({source: texture, signature: shape.signature, plan,
    get allocatedBytes() { return (map?.textureBytes ?? 0) + (input ? shape.sourceBytes : 0); },
    get failed() { return terminal !== null || !!map?.failed; }, get disposed() { return disposed; },
    matches() { live(); return same(shape.signature, inspectThreeEnvironment(texture, three, {maxPixels}).signature); },
    check,
    sample(borrowedDevice) { live(); return map.sample(borrowedDevice); },
    async whenIdle() { live(); await Promise.race([map.whenIdle(), stopped]); live(); return owner; },
    dispose() { if (!disposed) { disposed = true; rejectStop(new ThreeEnvironmentError('DISPOSED', 'Source environment is disposed')); release(); } },
  });
  return owner;
}

/** Live source uniform values; no panorama filtering or pixel reads. */
export function threeEnvironmentDescriptor(owner, scene, three) {
  if (owner === null) return null;
  const intensity = scene.environmentIntensity, rotation = scene.environmentRotation;
  if (typeof intensity !== 'number' || intensity < 0 || !Number.isFinite(Math.fround(intensity)))
    fail('VALUE', 'Environment intensity must be finite, nonnegative f32');
  if (!rotation?.isEuler || !['XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY'].includes(rotation.order) ||
      [rotation.x, rotation.y, rotation.z].some(v => typeof v !== 'number' || !Number.isFinite(v)))
    fail('VALUE', 'Expected a finite source environment Euler rotation');
  const e = new three.Matrix4().makeRotationFromEuler(rotation).elements;
  // Transpose the orthonormal source rotation: world direction -> panorama.
  return {map: owner, intensity, rotation: [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]]};
}

/** Scene.environment applies to Standard materials, not Lambert/Phong/Toon.
 * Preserve global ordering by splitting only adjacent receiveEnvironment spans.
 * This composes with the shadow receiver wrapper; no additional scene traversal.
 */
export function withThreeEnvironmentReceivers(renderer, maxDraws = 1024) {
  integer(maxDraws, 1, Number.MAX_SAFE_INTEGER, 'draw capacity');
  let busy = false, terminal = null, drawCount = 0, drawCallCount = 0, colorPassCount = 0;
  function live() { if (terminal) throw terminal; if (renderer.disposed) fail('DISPOSED', 'Source color renderer is disposed'); }
  const owner = Object.freeze({
    addMesh(gpu, options) { live(); return renderer.addMesh(gpu, options); },
    get disposed() { return renderer.disposed; }, get failed() { return terminal !== null || renderer.failed; },
    get allocatedBytes() { return renderer.allocatedBytes; }, get bundleDiagnostics() { return renderer.bundleDiagnostics; },
    get drawCount() { return drawCount; }, get drawCallCount() { return drawCallCount; }, get colorPassCount() { return colorPassCount; },
    render(frame) {
      live(); if (busy) fail('REENTRANT', 'Environment color submission cannot be reentered');
      if (!frame || !Array.isArray(frame.draws) || frame.draws.length > maxDraws) fail('LIMIT', 'Source color draw list exceeds capacity');
      const spans = [];
      for (const input of frame.draws) {
        if (!input || typeof input !== 'object' || typeof input.receiveEnvironment !== 'boolean')
          fail('FRAME', 'Expected an explicit environment receiver flag');
        const {receiveEnvironment, ...draw} = input, enabled = receiveEnvironment && frame.environment != null;
        if (!spans.length || spans.at(-1).enabled !== enabled) spans.push({enabled, draws: []});
        spans.at(-1).draws.push(draw);
      }
      if (!spans.length) spans.push({enabled: false, draws: []});
      busy = true; let submitted = 0, calls = 0, passes = 0;
      try {
        for (const span of spans) {
          renderer.render({...frame, draws: span.draws, environment: span.enabled ? frame.environment : null,
            ...(submitted ? {loadOp: 'load', depthLoadOp: 'load',
              ...(frame.stencilLoadOp !== undefined ? {stencilLoadOp: 'load'} : {})} : {})});
          submitted++; calls += renderer.drawCallCount; passes += renderer.colorPassCount ?? 1;
        }
        drawCount = frame.draws.length; drawCallCount = calls; colorPassCount = passes; return owner;
      } catch (error) { if (submitted || renderer.failed) { terminal = error; renderer.dispose(); } throw error; }
      finally { busy = false; }
    },
    async whenIdle() { live(); await renderer.whenIdle(); live(); return owner; },
    dispose() { if (busy) fail('REENTRANT', 'Cannot dispose during color submission'); renderer.dispose(); },
  });
  return owner;
}
