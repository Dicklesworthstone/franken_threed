/**
 * Explicit unlit WebGPU draw submission for createGpuAnimationDeformer outputs.
 * This pass owns an aligned uniform arena and index buffers, not the supplied
 * device, deformers, attachments, camera, frame loop or source scene objects.
 * It is NOT an automatic Three.js renderer replacement: no lighting, textures,
 * tone mapping, implicit sorting or culling. RGB inputs are linear; an -srgb
 * attachment view supplies the display transfer function. Unsupported material
 * fields are rejected rather than silently rendered as unlit.
 *
 * await createGpuAnimationRenderer(device, {format, depthFormat, sampleCount});
 * await renderer.addMesh(deformer, {indices?, baseColor?, doubleSided?,
 *   alphaMode?: 'OPAQUE'|'MASK'|'BLEND', alphaCutoff?});
 * renderer.render({colorView, depthView, viewProjection, draws: [mesh, ...]});
 *
 * A draw may instead be {mesh, worldMatrix?, baseColor?, first?, count?}.
 * viewProjection maps world space to WebGPU clip space (depth 0..1). Each draw
 * snapshots a separate matrix/color uniform range; render() submits immediately.
 * Thus render(pose A); update(pose B); render(pose B) preserves both uses. A
 * recorded-but-unsubmitted external draw must still precede the next update.
 * BLEND preserves input order and disables depth writes; callers order transparent
 * draws. OPAQUE ignores input alpha; MASK discards below its threshold and writes
 * opaque alpha. Negative-determinant world transforms reverse front-face winding.
 *
 * Host validation finishes before GPU writes. Driver errors are terminal, not
 * rollbackable. version acknowledges submission, not completion: await whenIdle()
 * for cumulative draw/deformation validation, OOM and device-loss errors.
 */
export class AnimationRenderError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationRenderError'; this.code = code; }
}
const fail = (code, message) => { throw new AnimationRenderError(code, message); };
const COPY_DST = 8, INDEX = 16, UNIFORM = 64, VERTEX_STAGE = 1, FRAGMENT_STAGE = 2;
const UNIFORM_BYTES = 96;
export const ANIMATION_RENDER_WGSL = /* wgsl */`
struct DrawInfo { clip_from_local: mat4x4<f32>, color: vec4<f32>, options: vec4<f32> }
@group(0) @binding(0) var<uniform> draw_info: DrawInfo;
@vertex fn vertex_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
  return draw_info.clip_from_local * vec4<f32>(position, 1.0);
}
@fragment fn fragment_main() -> @location(0) vec4<f32> {
  if (draw_info.options.x >= 0.0 && draw_info.color.a < draw_info.options.x) { discard; }
  return vec4<f32>(draw_info.color.rgb, select(1.0, draw_info.color.a, draw_info.options.y > 0.0));
}
`;
function finite(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail('ANIMATION_RENDER_VALUE', `${label} must be finite`);
  return value;
}
function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail('ANIMATION_RENDER_RANGE', `Invalid ${label}`);
  return value;
}
function array(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length) fail('ANIMATION_RENDER_SHAPE', `Invalid ${label} shape`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail('ANIMATION_RENDER_STORAGE', `${label} must have fixed unshared storage`);
    try { new Uint8Array(value.buffer, 0, 0); } catch { fail('ANIMATION_RENDER_STORAGE', `${label} is detached`); }
  }
  for (let i = 0; i < length; i++) finite(value[i], label);
  return value;
}
function color(value) {
  array(value, 4, 'Linear RGBA color');
  for (let i = 0; i < 4; i++) if (value[i] < 0 || !Number.isFinite(Math.fround(value[i]))) fail('ANIMATION_RENDER_VALUE', 'Color must be nonnegative finite f32');
  if (value[3] > 1) fail('ANIMATION_RENDER_VALUE', 'Alpha must be in [0,1]');
  return value;
}
function keys(object, allowed, label) {
  if (!object || typeof object !== 'object' || Array.isArray(object)) fail('ANIMATION_RENDER_OPTIONS', `Invalid ${label}`);
  for (const key of Object.keys(object)) if (!allowed.includes(key)) fail('ANIMATION_RENDER_OPTIONS', `Unsupported ${label} field: ${key}`);
}
function scoped(device, operation) {
  device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory');
  let value, error;
  try { value = operation(); } catch (caught) { error = caught; }
  // Pop before awaiting anything: the device scope stack is shared by callers.
  const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(values => {
    if (error) throw error;
    const reported = values.find(Boolean);
    if (reported) fail('ANIMATION_RENDER_DEVICE', reported.message || 'WebGPU operation failed');
  });
  return {value, error, errors};
}
function deformerShape(gpu) {
  if (!gpu || !gpu.vertexBuffer || !Number.isSafeInteger(gpu.vertexCount) || gpu.vertexCount < 1 ||
      gpu.vertexLayout?.arrayStride !== 40 || gpu.vertexLayout?.stepMode !== 'vertex' ||
      !gpu.vertexLayout.attributes?.some(a => a.shaderLocation === 0 && a.offset === 0 && a.format === 'float32x3') ||
      typeof gpu.whenIdle !== 'function') fail('ANIMATION_RENDER_GEOMETRY', 'Expected an animation GPU deformer');
  if (gpu.disposed || gpu.failed) fail('ANIMATION_RENDER_GEOMETRY', 'GPU deformer is disposed or failed');
}

/** Create reusable unlit render pipelines; never initializes browser services. */
export async function createGpuAnimationRenderer(device, {
  format = 'rgba8unorm', depthFormat = 'depth24plus', sampleCount = 1,
  maxDraws = 1024, maxMeshes = 1024, maxBytes = 64 * 1024 * 1024, label = 'f3d-animation-draw',
} = {}) {
  if (!device?.queue || !device.limits || typeof device.createRenderPipelineAsync !== 'function' ||
      typeof device.lost?.then !== 'function') fail('ANIMATION_RENDER_DEVICE', 'Lend a live WebGPU device');
  if (!['rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb', 'rgba16float'].includes(format) ||
      ![null, 'depth24plus', 'depth32float', 'depth16unorm'].includes(depthFormat) ||
      ![1, 4].includes(sampleCount) || typeof label !== 'string') fail('ANIMATION_RENDER_OPTIONS', 'Unsupported attachment configuration');
  integer(maxDraws, 1, 65536, 'draw capacity'); integer(maxMeshes, 1, 65536, 'mesh capacity');
  integer(maxBytes, 1, Number.MAX_SAFE_INTEGER, 'byte budget');
  const limits = device.limits;
  const limit = (name, needed) => {
    if (!Number.isSafeInteger(limits[name]) || limits[name] < needed) fail('ANIMATION_RENDER_LIMIT', `Insufficient ${name}`);
  };
  const alignment = integer(limits.minUniformBufferOffsetAlignment, 4, 65536, 'uniform alignment');
  if ((alignment & (alignment - 1)) !== 0) fail('ANIMATION_RENDER_LIMIT', 'Uniform alignment must be a power of two');
  const stride = Math.ceil(UNIFORM_BYTES / alignment) * alignment, arenaBytes = stride * maxDraws;
  limit('maxBufferSize', arenaBytes); limit('maxUniformBufferBindingSize', UNIFORM_BYTES);
  limit('maxDynamicUniformBuffersPerPipelineLayout', 1); limit('maxBindGroups', 1);
  limit('maxUniformBuffersPerShaderStage', 1); limit('maxVertexBuffers', 1);
  limit('maxVertexAttributes', 1); limit('maxVertexBufferArrayStride', 40);
  if (arenaBytes > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Uniform arena exceeds byte budget');
  const staged = new Float32Array(arenaBytes / 4), commands = [];
  const records = new Set(), owned = new WeakMap(), buffers = new Map(), pipelines = new Map();
  let allocatedBytes = 0, pendingMeshes = 0, disposed = false, terminal = null, busy = false;
  let version = 0, drawCount = 0, completion = Promise.resolve(), uniformBuffer, bindGroup;
  function remember(buffer, bytes) { buffers.set(buffer, bytes); allocatedBytes += bytes; return buffer; }
  function forget(buffer) { if (buffers.has(buffer)) { allocatedBytes -= buffers.get(buffer); buffers.delete(buffer); buffer.destroy(); } }
  function release() { for (const buffer of buffers.keys()) forget(buffer); }
  const lost = device.lost.then(info => {
    terminal ??= new AnimationRenderError('ANIMATION_RENDER_LOST', info?.message || 'WebGPU device lost');
    release(); throw terminal;
  });
  lost.catch(() => {});
  function live() {
    if (disposed) fail('ANIMATION_RENDER_DISPOSED', 'Animation renderer has been disposed');
    if (terminal) throw terminal;
  }
  try {
    const initialized = scoped(device, () => {
      uniformBuffer = remember(device.createBuffer({label, size: arenaBytes, usage: UNIFORM | COPY_DST}), arenaBytes);
      const layout = device.createBindGroupLayout({label, entries: [{binding: 0, visibility: VERTEX_STAGE | FRAGMENT_STAGE,
        buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: UNIFORM_BYTES}}]});
      const pipelineLayout = device.createPipelineLayout({label, bindGroupLayouts: [layout]});
      bindGroup = device.createBindGroup({label, layout, entries: [{binding: 0, resource: {buffer: uniformBuffer, size: UNIFORM_BYTES}}]});
      const module = device.createShaderModule({label, code: ANIMATION_RENDER_WGSL});
      const created = [];
      for (const blend of [false, true]) for (const winding of ['ccw', 'cw', 'none']) {
        const key = `${blend}:${winding}`;
        created.push(device.createRenderPipelineAsync({label: `${label}/${key}`, layout: pipelineLayout,
          vertex: {module, entryPoint: 'vertex_main', buffers: [{arrayStride: 40, stepMode: 'vertex',
            attributes: [{shaderLocation: 0, offset: 0, format: 'float32x3'}]}]},
          fragment: {module, entryPoint: 'fragment_main', targets: [{format, ...(blend ? {blend: {
            color: {operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha'},
            alpha: {operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha'},
          }} : {})}]},
          primitive: {topology: 'triangle-list', cullMode: winding === 'none' ? 'none' : 'back', frontFace: winding === 'cw' ? 'cw' : 'ccw'},
          ...(depthFormat ? {depthStencil: {format: depthFormat, depthWriteEnabled: !blend, depthCompare: 'less-equal'}} : {}),
          multisample: {count: sampleCount},
        }).then(pipeline => pipelines.set(key, pipeline)));
      }
      return Promise.all(created);
    });
    await Promise.race([Promise.all([initialized.value, initialized.errors]), lost]); live();
  } catch (error) { disposed = true; release(); throw error; }

  async function addMesh(gpu, options = {}) {
    live(); if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot register a mesh during submission');
    keys(options, ['indices', 'baseColor', 'doubleSided', 'alphaMode', 'alphaCutoff'], 'unlit material/geometry');
    deformerShape(gpu);
    if (records.size + pendingMeshes >= maxMeshes) fail('ANIMATION_RENDER_LIMIT', 'Mesh capacity exceeded');
    const {indices = null, baseColor = [1, 1, 1, 1], doubleSided = false, alphaMode = 'OPAQUE', alphaCutoff = 0.5} = options;
    const rgba = Float64Array.from(color(baseColor));
    if (typeof doubleSided !== 'boolean' || !['OPAQUE', 'MASK', 'BLEND'].includes(alphaMode) ||
        finite(alphaCutoff, 'Alpha cutoff') < 0 || alphaCutoff > 1) fail('ANIMATION_RENDER_OPTIONS', 'Invalid unlit material');
    let data = null, indexBuffer = null, indexFormat = null;
    if (indices !== null) {
      integer(indices.length, 1, Math.floor(maxBytes / 2), 'index count'); array(indices, indices.length, 'Indices');
      let maximum = 0;
      for (const index of indices) maximum = Math.max(maximum, integer(index, 0, gpu.vertexCount - 1, 'vertex index'));
      indexFormat = maximum <= 65535 ? 'uint16' : 'uint32';
      const C = indexFormat === 'uint16' ? Uint16Array : Uint32Array;
      const bytes = Math.ceil(indices.length * C.BYTES_PER_ELEMENT / 4) * 4;
      limit('maxBufferSize', bytes);
      if (allocatedBytes + bytes > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Index buffers exceed byte budget');
      data = new C(bytes / C.BYTES_PER_ELEMENT); data.set(indices);
    }
    const extent = indices === null ? gpu.vertexCount : indices.length;
    pendingMeshes++;
    try {
      if (data) {
        const allocated = scoped(device, () => {
          indexBuffer = remember(device.createBuffer({label: `${label}/indices`, size: data.byteLength, usage: INDEX, mappedAtCreation: true}), data.byteLength);
          new Uint8Array(indexBuffer.getMappedRange()).set(new Uint8Array(data.buffer)); indexBuffer.unmap();
        });
        await Promise.race([allocated.errors, lost]);
      }
      live(); deformerShape(gpu);
      const record = {gpu, rgba, doubleSided, alphaMode, alphaCutoff, extent, indexBuffer, indexFormat, disposed: false};
      const mesh = Object.freeze({vertexCount: gpu.vertexCount, indexCount: indices === null ? 0 : extent,
        get disposed() { return record.disposed || disposed; },
        dispose() {
          if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot dispose a mesh during submission');
          if (!record.disposed) { record.disposed = true; records.delete(record); forget(indexBuffer); }
        },
      });
      owned.set(mesh, record); records.add(record); return mesh;
    } catch (error) { forget(indexBuffer); throw error; }
    finally { pendingMeshes--; }
  }
  function render(frame) {
    live(); if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Render submission cannot be reentered');
    busy = true;
    try {
      keys(frame, ['colorView', 'depthView', 'resolveTarget', 'viewProjection', 'draws', 'loadOp', 'depthLoadOp', 'clearColor', 'clearDepth', 'viewport', 'scissor'], 'frame');
      const {colorView, depthView, resolveTarget, viewProjection, draws, loadOp = 'clear', depthLoadOp = 'clear',
        clearColor = [0, 0, 0, 0], clearDepth = 1, viewport = null, scissor = null} = frame;
      if (!colorView || (depthFormat && !depthView) || (!depthFormat && depthView) || (sampleCount === 1 && resolveTarget)) fail('ANIMATION_RENDER_ATTACHMENT', 'Attachment configuration differs from pipeline');
      if (!['clear', 'load'].includes(loadOp) || !['clear', 'load'].includes(depthLoadOp)) fail('ANIMATION_RENDER_ATTACHMENT', 'Invalid load operation');
      color(clearColor); finite(clearDepth, 'Clear depth');
      if (clearDepth < 0 || clearDepth > 1) fail('ANIMATION_RENDER_RANGE', 'Clear depth must be in [0,1]');
      array(viewProjection, 16, 'View-projection matrix');
      if (!Array.isArray(draws) || draws.length > maxDraws) fail('ANIMATION_RENDER_LIMIT', 'Draw list exceeds capacity');
      if (viewport !== null) {
        array(viewport, 6, 'Viewport');
        if (viewport[0] < 0 || viewport[1] < 0 || viewport[2] <= 0 || viewport[3] <= 0 ||
            viewport[4] < 0 || viewport[5] > 1 || viewport[4] > viewport[5]) fail('ANIMATION_RENDER_RANGE', 'Invalid viewport');
      }
      if (scissor !== null) { array(scissor, 4, 'Scissor'); for (const v of scissor) integer(v, 0, 0xffffffff, 'scissor component'); }
      const dependencies = new Set();
      for (let i = 0; i < draws.length; i++) {
        const input = owned.has(draws[i]) ? {mesh: draws[i]} : draws[i];
        keys(input, ['mesh', 'worldMatrix', 'baseColor', 'first', 'count'], 'draw');
        const record = owned.get(input.mesh);
        if (!record || record.disposed) fail('ANIMATION_RENDER_MESH', 'Mesh is not live in this renderer');
        const gpu = record.gpu; deformerShape(gpu);
        const world = array(input.worldMatrix ?? gpu.worldMatrix, 16, 'World matrix');
        if (world[3] !== 0 || world[7] !== 0 || world[11] !== 0 || world[15] !== 1) fail('ANIMATION_RENDER_VALUE', 'World matrix must be affine');
        const rgba = color(input.baseColor ?? record.rgba), offset = i * stride / 4;
        for (let column = 0; column < 4; column++) for (let row = 0; row < 4; row++) {
          const value = viewProjection[row] * world[column * 4] + viewProjection[4 + row] * world[column * 4 + 1]
            + viewProjection[8 + row] * world[column * 4 + 2] + viewProjection[12 + row] * world[column * 4 + 3];
          if (!Number.isFinite(Math.fround(value))) fail('ANIMATION_RENDER_VALUE', 'Clip matrix overflows f32');
          staged[offset + column * 4 + row] = value;
        }
        staged.set(rgba, offset + 16); staged[offset + 20] = record.alphaMode === 'MASK' ? record.alphaCutoff : -1;
        staged[offset + 21] = record.alphaMode === 'BLEND' ? 1 : 0;
        const determinant = world[0] * (world[5] * world[10] - world[9] * world[6])
          - world[4] * (world[1] * world[10] - world[9] * world[2]) + world[8] * (world[1] * world[6] - world[5] * world[2]);
        finite(determinant, 'World determinant');
        const first = integer(input.first ?? 0, 0, record.extent, 'draw start');
        const count = integer(input.count ?? record.extent - first, 0, record.extent - first, 'draw count');
        const command = commands[i] ?? (commands[i] = {});
        Object.assign(command, {record, first, count, pipeline: pipelines.get(`${record.alphaMode === 'BLEND'}:${record.doubleSided ? 'none' : determinant < 0 ? 'cw' : 'ccw'}`)});
        dependencies.add(gpu);
      }
      const submitted = scoped(device, () => {
        const encoder = device.createCommandEncoder({label});
        const pass = encoder.beginRenderPass({label, colorAttachments: [{view: colorView, ...(resolveTarget ? {resolveTarget} : {}),
          loadOp, storeOp: 'store', clearValue: {r: clearColor[0], g: clearColor[1], b: clearColor[2], a: clearColor[3]}}],
          ...(depthFormat ? {depthStencilAttachment: {view: depthView, depthLoadOp, depthStoreOp: 'store', depthClearValue: clearDepth}} : {})});
        if (viewport) pass.setViewport(...viewport);
        if (scissor) pass.setScissorRect(...scissor);
        for (let i = 0; i < draws.length; i++) {
          const {record, first, count, pipeline} = commands[i];
          pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup, [i * stride]);
          pass.setVertexBuffer(0, record.gpu.vertexBuffer);
          if (record.indexBuffer) { pass.setIndexBuffer(record.indexBuffer, record.indexFormat); pass.drawIndexed(count, 1, first, 0, 0); }
          else pass.draw(count, 1, first, 0);
        }
        pass.end(); const command = encoder.finish();
        if (draws.length) device.queue.writeBuffer(uniformBuffer, 0, staged, 0, (draws.length - 1) * stride / 4 + UNIFORM_BYTES / 4);
        device.queue.submit([command]);
      });
      if (submitted.error) { submitted.errors.catch(() => {}); terminal ??= submitted.error; throw terminal; }
      let work;
      try { work = [completion, submitted.errors, device.queue.onSubmittedWorkDone(), ...[...dependencies].map(gpu => gpu.whenIdle())]; }
      catch (error) { submitted.errors.catch(() => {}); terminal ??= error; throw terminal; }
      completion = Promise.race([Promise.all(work), lost]).then(() => {
        if (terminal) throw terminal;
      }, error => { terminal ??= error; throw terminal; });
      completion.catch(() => {});
      version++; drawCount = draws.length; return renderer;
    } finally {
      for (const command of commands) command.record = null;
      busy = false;
    }
  }
  const renderer = Object.freeze({format, depthFormat, sampleCount, addMesh, render,
    get allocatedBytes() { return allocatedBytes; },
    get version() { return version; }, get drawCount() { return drawCount; }, get meshCount() { return records.size; },
    get disposed() { return disposed; }, get failed() { return terminal !== null; },
    async whenIdle() { live(); await Promise.race([completion, lost]); live(); return renderer; },
    dispose() {
      if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot dispose during submission');
      if (!disposed) { disposed = true; records.clear(); release(); }
    },
  });
  return renderer;
}
