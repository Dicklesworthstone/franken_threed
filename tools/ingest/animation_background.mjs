/** Native full-attachment background rendering over a borrowed GPU texture.
 * No depth attachment/test/write, geometry, readback, source upload, frame loop,
 * tone mapping or implicit output transfer. RGB remains linear; an sRGB view or
 * the existing HDR canvas compositor supplies the application's output policy.
 * Panoramas are north-first, repeat-U/clamp-V. Cubes use native WebGPU faces.
 * Screen UVs have (0,0) at bottom-left, before the explicit six-value transform.
 * Source texture/device/attachments stay caller-owned and must outlive use.
 */
export class AnimationBackgroundError extends Error {
  constructor(code, message) {
    super(`ANIMATION_BACKGROUND_${code}: ${message}`);
    this.name = 'AnimationBackgroundError'; this.code = `ANIMATION_BACKGROUND_${code}`;
  }
}
const fail = (code, message) => { throw new AnimationBackgroundError(code, message); };
const finite = (v, label) => {
  if (typeof v !== 'number' || !Number.isFinite(Math.fround(v))) fail('VALUE', `${label} must fit finite f32`);
  return v;
};
const tuple = (v, n, label) => {
  if ((!Array.isArray(v) && !ArrayBuffer.isView(v)) || v.length !== n) fail('VALUE', `Expected ${label}[${n}]`);
  return Array.from(v, x => finite(x, label));
};
const mappings = ['panorama', 'cube', 'screen'];
const formats = ['rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb', 'rgba16float'];
const identity = Object.freeze([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

export function animationBackgroundShader(mapping = 'panorama') {
  if (!mappings.includes(mapping)) fail('OPTIONS', 'Unknown background mapping');
  const coordinates = mapping === 'screen' ? `
  let uv = p.ndc * 0.5 + vec2<f32>(0.5);
  let q = info.uv0.xy * uv.x + info.uv1.xy * uv.y + info.uv2.xy;
  let color = textureSampleLevel(image, image_sampler, q, info.factors.y);` : `
  let q = info.direction_from_clip * vec4<f32>(p.ndc, 0.5, 1.0);
  let d = normalize(q.xyz / q.w);
  ${mapping === 'cube' ? 'let color = textureSampleLevel(image, image_sampler, d, info.factors.y);' : `
  let uv = vec2<f32>(atan2(d.z, d.x) / 6.283185307179586 + 0.5,
    acos(clamp(d.y, -1.0, 1.0)) / 3.141592653589793);
  let color = textureSampleLevel(image, image_sampler, uv, info.factors.y);`}`;
  return /* wgsl */ `
struct BackgroundInfo {
  direction_from_clip: mat4x4<f32>,
  uv0: vec4<f32>, uv1: vec4<f32>, uv2: vec4<f32>, factors: vec4<f32>,
}
@group(0) @binding(0) var<uniform> info: BackgroundInfo;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var image: texture_${mapping === 'cube' ? 'cube' : '2d'}<f32>;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) ndc: vec2<f32> }
@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> VertexOutput {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - vec2<f32>(1.0);
  var out: VertexOutput; out.position = vec4<f32>(p, 1.0, 1.0); out.ndc = p; return out;
}
@fragment fn fragment_main(p: VertexOutput) -> @location(0) vec4<f32> {
  ${coordinates}
  return vec4<f32>(color.rgb * info.factors.x, color.a);
}
`;
}

/** Copy/validate the entire frame before any GPU operation. directionFromClip
 * maps clip (x,y,0.5,1) to environment-space homogeneous directions. Camera
 * translation must be removed BEFORE projection inversion is composed. All
 * four corner rays must have finite nonzero directions and positive w.
 */
export function packAnimationBackgroundFrame(frame, mapping = 'panorama', mipLevelCount = 1) {
  if (!frame || typeof frame !== 'object' || !mappings.includes(mapping)) fail('FRAME', 'Expected a background frame');
  if (!Number.isSafeInteger(mipLevelCount) || mipLevelCount < 1 || mipLevelCount > 16) fail('VALUE', 'Invalid mip count');
  const matrix = tuple(frame.directionFromClip ?? (mapping === 'screen' ? identity : []), 16, 'direction matrix');
  const uv = tuple(frame.uvTransform ?? [1, 0, 0, 1, 0, 0], 6, 'UV transform');
  const intensity = finite(frame.intensity ?? 1, 'background intensity');
  const mipLevel = finite(frame.mipLevel ?? 0, 'mip level');
  if (intensity < 0 || mipLevel < 0 || mipLevel > mipLevelCount - 1) fail('VALUE', 'Intensity or mip level is out of range');
  const bytes = new Float32Array(32);
  bytes.set(matrix); bytes.set(uv.slice(0, 2), 16); bytes.set(uv.slice(2, 4), 20); bytes.set(uv.slice(4), 24);
  bytes[28] = intensity; bytes[29] = mipLevel;
  if (mapping !== 'screen') {
    // Use the packed values, not the pre-conversion doubles. A positive affine
    // w at every corner stays positive throughout this full-screen triangle's
    // visible rectangle. Reject a collapsed transform before submitting it.
    const m = bytes;
    const a = [m[0], m[1], m[2]], b = [m[4], m[5], m[6]], c = [m[8] * 0.5 + m[12], m[9] * 0.5 + m[13], m[10] * 0.5 + m[14]];
    const determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1]) + c[0] * (a[1] * b[2] - a[2] * b[1]);
    if (!Number.isFinite(determinant) || determinant === 0) fail('VALUE', 'Background ray plane must not pass through the origin');
    for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) {
      const q = Array.from({length: 4}, (_, r) => m[r] * x + m[r + 4] * y + m[r + 8] * 0.5 + m[r + 12]);
      const length = Math.hypot(q[0], q[1], q[2]);
      if (!(q[3] > 0) || !(length > 1e-20) || !Number.isFinite(Math.fround(length)) ||
          q.slice(0, 3).some(v => !Number.isFinite(Math.fround(v / q[3])))) fail('VALUE', 'Degenerate background ray transform');
    }
  }
  return bytes;
}

/** Explicit native background, independently usable by animation/model clients.
 * Each render immediately submits and snapshots its 128-byte uniform packet,
 * so consecutive camera/intensity changes keep queue ordering. sampleCount is
 * 1 or 4; multisample storage is ALWAYS stored for later scene color passes.
 * A resolveTarget is optional at 4x (for standalone final presentation), never
 * required for the background prefix of a later resolving scene pass.
 */
export async function createGpuAnimationBackground(device, source, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected options');
  for (const key of Object.keys(options)) if (!['format', 'sampleCount', 'mapping', 'sampler', 'label', 'signal'].includes(key))
    fail('OPTIONS', `Unknown background option: ${key}`);
  const {format = 'rgba8unorm', sampleCount = 1, mapping = 'panorama', sampler: suppliedSampler,
    label = 'f3d-background', signal} = options;
  if (!formats.includes(format) || ![1, 4].includes(sampleCount) || !mappings.includes(mapping) || typeof label !== 'string')
    fail('OPTIONS', 'Unsupported output format, sample count, mapping or label');
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' ||
      typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected AbortSignal');
  if (signal?.aborted) fail('ABORTED', 'Background creation was aborted');
  for (const key of ['createBuffer', 'createSampler', 'createBindGroupLayout', 'createPipelineLayout',
    'createShaderModule', 'createRenderPipelineAsync', 'createBindGroup', 'createCommandEncoder', 'pushErrorScope', 'popErrorScope'])
    if (typeof device?.[key] !== 'function') fail('DEVICE', `Missing WebGPU ${key}`);
  for (const key of ['writeBuffer', 'submit', 'onSubmittedWorkDone']) if (typeof device.queue?.[key] !== 'function')
    fail('DEVICE', `Missing GPU queue.${key}`);
  if (typeof device.lost?.then !== 'function' || !(device.limits?.maxBufferSize >= 128)) fail('DEVICE', 'Expected device loss and buffer limits');
  if (!source || source.dimension !== '2d' || source.sampleCount !== 1 || !formats.includes(source.format) ||
      !(Number.isInteger(source.usage) && (source.usage & 4)) || typeof source.createView !== 'function' ||
      !Number.isSafeInteger(source.width) || !Number.isSafeInteger(source.height) || source.width < 1 || source.height < 1 ||
      !Number.isSafeInteger(source.mipLevelCount) || source.mipLevelCount < 1 || source.mipLevelCount > 16 ||
      source.depthOrArrayLayers !== (mapping === 'cube' ? 6 : 1)) fail('SOURCE', 'Expected a filterable single-sample color texture');
  if ((mapping === 'panorama' && source.width !== 2 * source.height) || (mapping === 'cube' && source.width !== source.height))
    fail('SOURCE', 'Expected a 2:1 panorama or square native cubemap faces');
  if (suppliedSampler !== undefined && (!suppliedSampler || typeof suppliedSampler !== 'object')) fail('SOURCE', 'Expected a borrowed sampler');
  const levels = source.mipLevelCount;
  let buffer = null, pipeline = null, group = null, disposed = false, terminal = null, busy = false, frames = 0;
  let pending = Promise.resolve(), rejectStop;
  const stopped = new Promise((_, reject) => { rejectStop = reject; }); stopped.catch(() => {});
  function release() { signal?.removeEventListener('abort', onAbort); buffer?.destroy(); buffer = null; pipeline = null; group = null; }
  function stop(error) { if (!terminal && !disposed) { terminal = error; rejectStop(error); if (!busy) release(); } }
  function onAbort() { stop(new AnimationBackgroundError('ABORTED', 'Background lifetime was aborted')); }
  function live() { if (disposed) fail('DISPOSED', 'Background is disposed'); if (terminal) throw terminal; }
  // Both scopes are popped in the same synchronous turn as the operation. No
  // device-global error scope remains open across pipeline/queue awaits.
  function checked(operation) {
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let result, error;
    try { result = operation(); } catch (cause) { error = cause; }
    const validation = device.popErrorScope(), oom = device.popErrorScope();
    const complete = Promise.all([result, validation, oom]).then(([value, invalid, memory]) => {
      if (error) throw error;
      if (invalid || memory) fail('DEVICE', (invalid || memory).message ?? 'Native background operation failed');
      live(); return value;
    }).catch(cause => { stop(cause); throw cause; });
    complete.catch(() => {});
    if (error) { stop(error); throw error; }
    return complete;
  }
  signal?.addEventListener('abort', onAbort, {once: true});
  device.lost.then(info => stop(new AnimationBackgroundError('DEVICE', info?.message ?? 'GPU device lost')), stop);
  try {
    if (signal?.aborted) onAbort(); live();
    let layout;
    pipeline = await Promise.race([checked(() => {
      layout = device.createBindGroupLayout({entries: [
        {binding: 0, visibility: 2, buffer: {type: 'uniform', minBindingSize: 128}},
        {binding: 1, visibility: 2, sampler: {type: 'filtering'}},
        {binding: 2, visibility: 2, texture: {sampleType: 'float', viewDimension: mapping === 'cube' ? 'cube' : '2d'}},
      ]});
      const module = device.createShaderModule({label, code: animationBackgroundShader(mapping)});
      return device.createRenderPipelineAsync({label, layout: device.createPipelineLayout({bindGroupLayouts: [layout]}),
        vertex: {module, entryPoint: 'vertex_main'}, fragment: {module, entryPoint: 'fragment_main', targets: [{format}]},
        primitive: {topology: 'triangle-list'}, multisample: {count: sampleCount}});
    }), stopped]); live();
    await Promise.race([checked(() => {
      buffer = device.createBuffer({label: label + '/uniforms', size: 128, usage: 8 | 64});
      const sampler = suppliedSampler ?? device.createSampler({minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear',
        addressModeU: mapping === 'panorama' ? 'repeat' : 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
      const view = source.createView({dimension: mapping === 'cube' ? 'cube' : '2d', baseArrayLayer: 0,
        arrayLayerCount: mapping === 'cube' ? 6 : 1, baseMipLevel: 0, mipLevelCount: levels});
      group = device.createBindGroup({layout, entries: [{binding: 0, resource: {buffer, size: 128}},
        {binding: 1, resource: sampler}, {binding: 2, resource: view}]});
    }), stopped]); live();
  } catch (error) { stop(error); release(); throw error; }
  const renderer = Object.freeze({format, sampleCount, mapping,
    get disposed() { return disposed; }, get failed() { return terminal !== null; },
    get allocatedBytes() { return buffer ? 128 : 0; }, get drawCount() { return frames; },
    render(frame) {
      live(); if (busy) fail('REENTRANT', 'Background submission cannot be reentered');
      const packet = packAnimationBackgroundFrame(frame, mapping, levels);
      const {colorView, resolveTarget = null, loadOp = 'clear'} = frame;
      const clearValue = tuple(frame.clearColor ?? [0, 0, 0, 1], 4, 'clear color');
      if (!colorView || typeof colorView !== 'object' || !['clear', 'load'].includes(loadOp)) fail('FRAME', 'Expected a color attachment and load policy');
      if (resolveTarget !== null && (sampleCount !== 4 || !resolveTarget || typeof resolveTarget !== 'object'))
        fail('FRAME', 'Resolve requires a multisample output and a texture view');
      busy = true;
      try {
        const work = checked(() => {
          device.queue.writeBuffer(buffer, 0, packet);
          const encoder = device.createCommandEncoder({label});
          const pass = encoder.beginRenderPass({colorAttachments: [{view: colorView, loadOp, clearValue,
            storeOp: 'store', ...(resolveTarget ? {resolveTarget} : {})}]});
          pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
          device.queue.submit([encoder.finish()]);
        });
        pending = Promise.all([pending, work]).then(() => {}); pending.catch(() => {});
        live(); frames++; return renderer;
      } catch (error) { stop(error); throw error; }
      finally { busy = false; if (terminal || disposed) release(); }
    },
    async whenIdle() {
      live();
      try { await Promise.race([Promise.all([pending, device.queue.onSubmittedWorkDone()]), stopped]); live(); return renderer; }
      catch (error) { stop(error); throw error; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during background submission');
      if (!disposed) { disposed = true; rejectStop(new AnimationBackgroundError('DISPOSED', 'Background is disposed')); release(); }
    },
  });
  return renderer;
}
