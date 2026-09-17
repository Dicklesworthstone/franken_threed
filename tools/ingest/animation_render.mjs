/**
 * Explicit unlit WebGPU draw submission for createGpuAnimationDeformer outputs.
 * This pass owns an aligned uniform arena and index buffers, not the supplied
 * device, deformers, attachments, camera, frame loop or source scene objects.
 * It is NOT an automatic Three.js renderer replacement: no lighting,
 * tone mapping, implicit sorting or culling. RGB inputs are linear; an -srgb
 * attachment view supplies the display transfer function. Unsupported material
 * fields are rejected rather than silently rendered as unlit.
 *
 * await createGpuAnimationRenderer(device, {format, depthFormat, sampleCount});
 * await renderer.addMesh(deformer, {indices?, baseColor?, doubleSided?,
 *   alphaMode?: 'OPAQUE'|'MASK'|'BLEND', alphaCutoff?, texCoords?, vertexColors?,
 *   baseColorTexture?: {view, sampler}, uvTransform?});
 * UVs are packed XY; vertex colors are linear RGB or RGBA, decoded from any
 * normalized integer source. The texture is a borrowed filterable 2D float
 * view with straight alpha. Use an -srgb view for sRGB-encoded base-color data:
 * hardware sampling decodes RGB, not alpha. No flipY or color conversion is
 * guessed. The caller owns texture creation, mip levels and sampler settings.
 * UV/color arrays are copied and uploaded once, independently of deformation.
 * uvTransform=[a,b,c,d,tx,ty] maps (u,v) to (a*u+c*v+tx,b*u+d*v+ty).
 * A draw may override uvTransform; baseColor * vertexColor * sampledColor is
 * evaluated BEFORE alpha masking/blending. Plain meshes need no texture.
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
const COPY_DST = 8, INDEX = 16, VERTEX = 32, UNIFORM = 64, VERTEX_STAGE = 1, FRAGMENT_STAGE = 2;
const UNIFORM_BYTES = 128;
const UV_IDENTITY = Object.freeze([1, 0, 0, 1, 0, 0]);
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
function surfaceShader(textured) {
  return /* wgsl */`
struct DrawInfo { clip_from_local: mat4x4<f32>, color: vec4<f32>, options: vec4<f32>, uv_x: vec4<f32>, uv_y: vec4<f32> }
@group(0) @binding(0) var<uniform> draw_info: DrawInfo;
${textured ? '@group(1) @binding(0) var color_sampler: sampler;\n@group(1) @binding(1) var color_texture: texture_2d<f32>;' : ''}
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32> }
@vertex fn vertex_main(@location(0) position: vec3<f32>, @location(3) uv: vec2<f32>, @location(4) color: vec4<f32>) -> VertexOutput {
  var out: VertexOutput;
  out.position = draw_info.clip_from_local * vec4<f32>(position, 1.0);
  out.uv = vec2<f32>(dot(draw_info.uv_x.xyz, vec3<f32>(uv, 1.0)), dot(draw_info.uv_y.xyz, vec3<f32>(uv, 1.0)));
  out.color = color;
  return out;
}
@fragment fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let rgba = draw_info.color * input.color ${textured ? '* textureSample(color_texture, color_sampler, input.uv)' : ''};
  if (draw_info.options.x >= 0.0 && rgba.a < draw_info.options.x) { discard; }
  return vec4<f32>(rgba.rgb, select(1.0, rgba.a, draw_info.options.y > 0.0));
}
`;
}
function uvTransform(value) {
  array(value, 6, 'UV transform');
  for (const v of value) if (!Number.isFinite(Math.fround(v))) fail('ANIMATION_RENDER_VALUE', 'UV transform exceeds f32');
  return value;
}
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
  let version = 0, drawCount = 0, completion = Promise.resolve(), uniformBuffer, bindGroup, uniformLayout, textureLayout;
  const variants = new Map();
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
  function compilePipelines(variant) {
    const surface = variant !== 'plain', textured = variant === 'texture';
    if (surface) { limit('maxVertexBuffers', 2); limit('maxVertexAttributes', 5); }
    if (textured) {
      limit('maxBindGroups', 2); limit('maxSamplersPerShaderStage', 1); limit('maxSampledTexturesPerShaderStage', 1);
      textureLayout ??= device.createBindGroupLayout({label, entries: [
        {binding: 0, visibility: FRAGMENT_STAGE, sampler: {type: 'filtering'}},
        {binding: 1, visibility: FRAGMENT_STAGE, texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}},
      ]});
    }
    const pipelineLayout = device.createPipelineLayout({label, bindGroupLayouts: textured ? [uniformLayout, textureLayout] : [uniformLayout]});
    const module = device.createShaderModule({label, code: surface ? surfaceShader(textured) : ANIMATION_RENDER_WGSL});
    const vertexBuffers = [{arrayStride: 40, stepMode: 'vertex', attributes: [{shaderLocation: 0, offset: 0, format: 'float32x3'}]}];
    if (surface) vertexBuffers.push({arrayStride: 24, stepMode: 'vertex', attributes: [
      {shaderLocation: 3, offset: 0, format: 'float32x2'}, {shaderLocation: 4, offset: 8, format: 'float32x4'},
    ]});
    const created = [];
    for (const blend of [false, true]) for (const winding of ['ccw', 'cw', 'none']) {
      const key = `${variant}/${blend}:${winding}`;
      created.push(device.createRenderPipelineAsync({label: `${label}/${key}`, layout: pipelineLayout,
        vertex: {module, entryPoint: 'vertex_main', buffers: vertexBuffers},
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
  }
  function ensureVariant(variant) {
    if (!variants.has(variant)) {
      const built = scoped(device, () => compilePipelines(variant));
      const ready = Promise.race([Promise.all([built.value, built.errors]), lost]);
      variants.set(variant, ready);
      ready.catch(() => variants.delete(variant));
    }
    return variants.get(variant);
  }
  try {
    const initialized = scoped(device, () => {
      uniformBuffer = remember(device.createBuffer({label, size: arenaBytes, usage: UNIFORM | COPY_DST}), arenaBytes);
      uniformLayout = device.createBindGroupLayout({label, entries: [{binding: 0, visibility: VERTEX_STAGE | FRAGMENT_STAGE,
        buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: UNIFORM_BYTES}}]});
      bindGroup = device.createBindGroup({label, layout: uniformLayout, entries: [{binding: 0, resource: {buffer: uniformBuffer, size: UNIFORM_BYTES}}]});
      return compilePipelines('plain');
    });
    await Promise.race([Promise.all([initialized.value, initialized.errors]), lost]); live();
  } catch (error) { disposed = true; release(); throw error; }

  async function addMesh(gpu, options = {}) {
    live(); if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot register a mesh during submission');
    keys(options, ['indices', 'baseColor', 'doubleSided', 'alphaMode', 'alphaCutoff', 'texCoords', 'vertexColors', 'baseColorTexture', 'uvTransform'], 'unlit material/geometry');
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
    const {texCoords = null, vertexColors = null, baseColorTexture = null} = options;
    const transform = Float64Array.from(uvTransform(options.uvTransform ?? UV_IDENTITY));
    const variant = baseColorTexture !== null ? 'texture' : vertexColors !== null || texCoords !== null ? 'color' : 'plain';
    let texture = null, surfaceData = null, surfaceBuffer = null, textureGroup = null;
    if (baseColorTexture !== null) {
      keys(baseColorTexture, ['view', 'sampler'], 'base-color texture');
      const {view, sampler} = baseColorTexture;
      if (!view || typeof view !== 'object' || !sampler || typeof sampler !== 'object') fail('ANIMATION_RENDER_OPTIONS', 'Texture requires a borrowed view and sampler');
      texture = {view, sampler};
      if (texCoords === null) fail('ANIMATION_RENDER_GEOMETRY', 'Base-color texture requires UV coordinates');
    }
    if (variant !== 'plain') {
      limit('maxVertexBuffers', 2); limit('maxVertexAttributes', 5);
      if (texture) { limit('maxBindGroups', 2); limit('maxSamplersPerShaderStage', 1); limit('maxSampledTexturesPerShaderStage', 1); }
      const bytes = gpu.vertexCount * 24;
      limit('maxBufferSize', bytes);
      if (allocatedBytes + (data?.byteLength ?? 0) + bytes > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Surface/index buffers exceed byte budget');
      if (texCoords !== null) array(texCoords, gpu.vertexCount * 2, 'UV coordinates');
      let width = 0;
      if (vertexColors !== null) {
        width = vertexColors.length / gpu.vertexCount;
        if (width !== 3 && width !== 4) fail('ANIMATION_RENDER_SHAPE', 'Vertex colors require RGB or RGBA per vertex');
        array(vertexColors, gpu.vertexCount * width, 'Vertex colors');
      }
      surfaceData = new Float32Array(gpu.vertexCount * 6);
      for (let v = 0; v < gpu.vertexCount; v++) {
        for (let c = 0; c < 2; c++) surfaceData[v * 6 + c] = texCoords?.[v * 2 + c] ?? 0;
        for (let c = 0; c < 4; c++) {
          const value = c < width ? vertexColors[v * width + c] : 1;
          if (value < 0 || (c === 3 && value > 1)) fail('ANIMATION_RENDER_VALUE', 'Invalid linear vertex color');
          surfaceData[v * 6 + 2 + c] = value;
        }
      }
      for (const v of surfaceData) if (!Number.isFinite(v)) fail('ANIMATION_RENDER_VALUE', 'Surface attributes exceed f32');
    }
    const extent = indices === null ? gpu.vertexCount : indices.length;
    pendingMeshes++;
    try {
      if (data || surfaceData) {
        const ready = variant === 'plain' ? Promise.resolve() : ensureVariant(variant);
        const allocated = scoped(device, () => {
          for (const [values, usage, name] of [[data, INDEX, 'indices'], [surfaceData, VERTEX, 'surface']]) if (values) {
            const buffer = remember(device.createBuffer({label: `${label}/${name}`, size: values.byteLength, usage, mappedAtCreation: true}), values.byteLength);
            if (name === 'indices') indexBuffer = buffer; else surfaceBuffer = buffer;
            new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(values.buffer)); buffer.unmap();
          }
          if (texture) textureGroup = device.createBindGroup({label, layout: textureLayout, entries: [
            {binding: 0, resource: texture.sampler}, {binding: 1, resource: texture.view},
          ]});
        });
        await Promise.race([Promise.all([allocated.errors, ready]), lost]);
      }
      live(); deformerShape(gpu);
      const record = {gpu, rgba, doubleSided, alphaMode, alphaCutoff, extent, indexBuffer, indexFormat, variant, transform, surfaceBuffer, textureGroup, disposed: false};
      const mesh = Object.freeze({vertexCount: gpu.vertexCount, indexCount: indices === null ? 0 : extent,
        get disposed() { return record.disposed || disposed; },
        dispose() {
          if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot dispose a mesh during submission');
          if (!record.disposed) { record.disposed = true; records.delete(record); forget(indexBuffer); forget(surfaceBuffer); }
        },
      });
      owned.set(mesh, record); records.add(record); return mesh;
    } catch (error) { forget(indexBuffer); forget(surfaceBuffer); throw error; }
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
        keys(input, ['mesh', 'worldMatrix', 'baseColor', 'first', 'count', 'uvTransform'], 'draw');
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
        const uv = uvTransform(input.uvTransform ?? record.transform);
        staged.set([uv[0], uv[2], uv[4], 0, uv[1], uv[3], uv[5], 0], offset + 24);
        const determinant = world[0] * (world[5] * world[10] - world[9] * world[6])
          - world[4] * (world[1] * world[10] - world[9] * world[2]) + world[8] * (world[1] * world[6] - world[5] * world[2]);
        finite(determinant, 'World determinant');
        const first = integer(input.first ?? 0, 0, record.extent, 'draw start');
        const count = integer(input.count ?? record.extent - first, 0, record.extent - first, 'draw count');
        const command = commands[i] ?? (commands[i] = {});
        Object.assign(command, {record, first, count, pipeline: pipelines.get(`${record.variant}/${record.alphaMode === 'BLEND'}:${record.doubleSided ? 'none' : determinant < 0 ? 'cw' : 'ccw'}`)});
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
          if (record.surfaceBuffer) pass.setVertexBuffer(1, record.surfaceBuffer);
          if (record.textureGroup) pass.setBindGroup(1, record.textureGroup);
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
