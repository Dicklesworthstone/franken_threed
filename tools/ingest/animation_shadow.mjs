/**
 * Reusable projected depth shadows from existing GPU-deformed geometry.
 * Owns one depth32float texture and a depth-only animation renderer. Never owns
 * the device, deformers or alpha textures, and never samples/advances a pose.
 * Call render() after deformation and before a receiving color submission.
 * The light camera is explicit, column-major, with WebGPU clip depth 0..1.
 * Directional/spot projections are supported; point-light cubemaps, cascades,
 * automatic fitting and translucent shadow transmission are not synthesized.
 */
import {AnimationRenderError, createGpuAnimationRenderer} from './animation_render.mjs';
const fail = (message, code = 'ANIMATION_SHADOW_INPUT') => { throw new AnimationRenderError(code, message); };
const fields = (value, allowed) => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('Expected an options object');
  for (const key of Object.keys(value)) if (!allowed.includes(key)) fail(`Unsupported shadow field: ${key}`);
};
function matrix(value) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== 16) fail('Expected a 16-component matrix');
  if (ArrayBuffer.isView(value) && (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable)) fail('Matrix requires fixed unshared storage');
  const copy = Array.from(value);
  if (copy.some(v => typeof v !== 'number' || !Number.isFinite(Math.fround(v)))) fail('Matrix exceeds finite f32');
  return copy;
}

/** Create a bounded depth map. OPAQUE/MASK caster material options match addMesh(). */
export async function createGpuAnimationShadowMap(device, options = {}) {
  fields(options, ['width', 'height', 'maxBytes', 'maxDraws', 'maxMeshes', 'label']);
  const {width = 1024, height = width, maxBytes = 64 * 1024 * 1024,
    maxDraws = 1024, maxMeshes = 1024, label = 'f3d-animation-shadow'} = options;
  const maximum = device?.limits?.maxTextureDimension2D;
  if (!Number.isSafeInteger(maximum) || ![width, height].every(v => Number.isSafeInteger(v) && v >= 1 && v <= maximum) ||
      !Number.isSafeInteger(maxBytes) || maxBytes < 1 || typeof label !== 'string') fail('Invalid shadow extent or byte budget');
  const textureBytes = width * height * 4;
  if (!Number.isSafeInteger(textureBytes) || textureBytes >= maxBytes) fail('Depth map exceeds byte budget', 'ANIMATION_SHADOW_LIMIT');
  const renderer = await createGpuAnimationRenderer(device, {format: null, depthFormat: 'depth32float', sampleCount: 1,
    maxBytes: maxBytes - textureBytes, maxDraws, maxMeshes, label});
  let texture, view, sampler, disposed = false, terminal = null, busy = false, snapshot = null, dependencies = [];
  const owned = new WeakMap();
  function release() { renderer.dispose(); texture?.destroy(); texture = null; dependencies = []; snapshot = null; }
  function live() {
    if (disposed) fail('Shadow map has been disposed', 'ANIMATION_SHADOW_DISPOSED');
    if (terminal) throw terminal;
    if (renderer.failed) { terminal = new AnimationRenderError('ANIMATION_SHADOW_DEVICE', 'Shadow rendering failed'); release(); throw terminal; }
  }
  function exclusive(operation) {
    live(); if (busy) fail('Shadow operation cannot be reentered', 'ANIMATION_SHADOW_REENTRANT');
    busy = true; try { return operation(); } finally { busy = false; }
  }
  function current(dependency) {
    const {gpu, version, poseVersion, world} = dependency;
    if (gpu.disposed || gpu.failed || gpu.version !== version || gpu.poseVersion !== poseVersion) return false;
    if (world) {
      const now = gpu.worldMatrix;
      if (!now || now.length !== 16 || world.some((v, i) => v !== now[i])) return false;
    }
    return true;
  }
  try {
    device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory');
    let caught;
    try {
      texture = device.createTexture({label, size: {width, height, depthOrArrayLayers: 1}, dimension: '2d',
        format: 'depth32float', sampleCount: 1, mipLevelCount: 1, usage: 4 | 16});
      view = texture.createView({dimension: '2d', aspect: 'depth-only'});
      sampler = device.createSampler({label, compare: 'less-equal', minFilter: 'nearest', magFilter: 'nearest',
        addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
    } catch (error) { caught = error; }
    const pending = Promise.all([device.popErrorScope(), device.popErrorScope()]);
    const errors = await Promise.race([pending, device.lost.then(() => { fail('Device lost during shadow initialization', 'ANIMATION_SHADOW_DEVICE'); })]);
    if (caught) throw caught;
    if (errors.some(Boolean)) fail(errors.find(Boolean).message || 'Shadow allocation failed', 'ANIMATION_SHADOW_DEVICE');
    live();
  } catch (error) { disposed = true; release(); throw error; }
  device.lost.then(info => {
    terminal ??= new AnimationRenderError('ANIMATION_SHADOW_DEVICE', info?.message || 'Device lost');
    release();
  }).catch(() => {});

  const shadow = Object.freeze({width, height,
    get version() { return renderer.version; },
    get allocatedBytes() { return renderer.allocatedBytes + (texture ? textureBytes : 0); },
    get disposed() { return disposed; }, get failed() { return terminal !== null || renderer.failed; },
    async addMesh(gpu, material = {}) {
      live(); if (busy) fail('Cannot add a caster during a shadow operation', 'ANIMATION_SHADOW_REENTRANT');
      // The depth-only renderer validates and snapshots the same indices, alpha
      // maps, vertex colors and independent base-color UV transforms as color.
      const mesh = await renderer.addMesh(gpu, material);
      try { live(); } catch (error) { mesh.dispose(); throw error; }
      const handle = Object.freeze({vertexCount: mesh.vertexCount, indexCount: mesh.indexCount,
        get disposed() { return mesh.disposed || disposed; },
        dispose() {
          if (busy) fail('Cannot dispose a caster during a shadow operation', 'ANIMATION_SHADOW_REENTRANT');
          mesh.dispose();
        },
      });
      owned.set(handle, {mesh, gpu}); return handle;
    },
    render(frame) { return exclusive(() => {
      fields(frame, ['viewProjection', 'draws']);
      const viewProjection = matrix(frame.viewProjection), list = frame.draws;
      if (!Array.isArray(list) || list.length > maxDraws) fail('Invalid caster draw list');
      const nextDependencies = [], draws = list.map(value => {
        const input = owned.has(value) ? {mesh: value} : value;
        fields(input, ['mesh', 'worldMatrix', 'baseColor', 'first', 'count', 'uvTransform']);
        const record = owned.get(input.mesh);
        if (!record || record.mesh.disposed) fail('Caster does not belong to this live map');
        const {gpu, mesh} = record, world = matrix(input.worldMatrix ?? gpu.worldMatrix);
        nextDependencies.push({gpu, version: gpu.version, poseVersion: gpu.poseVersion,
          world: input.worldMatrix === undefined ? world : null});
        const draw = {mesh, worldMatrix: world};
        for (const key of ['baseColor', 'uvTransform']) if (input[key] !== undefined) {
          const values = input[key];
          if ((!Array.isArray(values) && !ArrayBuffer.isView(values)) ||
              (ArrayBuffer.isView(values) && (!(values.buffer instanceof ArrayBuffer) || values.buffer.resizable))) fail(`Invalid ${key} storage`);
          draw[key] = Array.from(values);
        }
        for (const key of ['first', 'count']) if (input[key] !== undefined) draw[key] = input[key];
        return draw;
      });
      if (!nextDependencies.every(current)) fail('Caster changed during shadow preparation', 'ANIMATION_SHADOW_STALE');
      renderer.render({depthView: view, viewProjection, draws, clearDepth: 1});
      dependencies = nextDependencies;
      snapshot = Object.freeze({view, sampler, viewProjection: Object.freeze(viewProjection), width, height, version: renderer.version});
      return shadow;
    }); },
    /** A same-device, current-pose snapshot for a subsequent receiver pass. */
    sample(borrowedDevice) { return exclusive(() => {
      if (borrowedDevice !== device) fail('Shadow and receiver must use the same GPU device');
      if (!snapshot) fail('Render the shadow map before sampling', 'ANIMATION_SHADOW_UNRENDERED');
      if (!dependencies.every(current)) fail('Caster pose changed; render the shadow map again', 'ANIMATION_SHADOW_STALE');
      return snapshot;
    }); },
    async whenIdle() {
      live();
      try { await renderer.whenIdle(); }
      catch (error) { terminal ??= error; release(); throw terminal; }
      live(); return shadow;
    },
    dispose() {
      if (busy) fail('Cannot dispose during a shadow operation', 'ANIMATION_SHADOW_REENTRANT');
      if (!disposed) { disposed = true; release(); }
    },
  });
  return shadow;
}
