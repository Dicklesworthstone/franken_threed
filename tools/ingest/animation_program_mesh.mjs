/** Compiled source programs (essl_wgsl.mjs output) as meshes of the core renderer.
 *
 * A program mesh owns two render pipelines (ccw/cw front faces, chosen per draw
 * from the object's determinant as WebGLRenderer does), its texture bind group
 * and a bind group over the renderer's shared program-uniform arena. Each
 * logical draw gets its own 256-byte-aligned arena slice written before the
 * frame's single submit, so two draws of one material in one submission never
 * share uniform bytes (the queue-write snapshot rule).
 *
 * Geometry comes from createGpuProgramGeometry(); locations the geometry lacks
 * read GL's constant default attribute (0, 0, 0, 1) from a stride-0 buffer.
 * Nothing here interprets source objects or uniform values.
 */
import {programGeometrySnapshot} from './gpu_buffer_geometry.mjs';

const VERTEX = 1, FRAGMENT = 2;
const UNIFORM_USAGE = 64 | 8; // UNIFORM | COPY_DST
const ALIGN = 256;

export function createProgramMeshes({device, format, depthFormat, sampleCount, maxDraws, label, fail, scoped, lost}) {
  let arena = null, constant = null, cursor = 0;
  function ensureArena() {
    if (arena) return arena;
    const size = Math.min(device.limits.maxBufferSize ?? 268435456, Math.max(65536, maxDraws * 1024));
    const buffer = device.createBuffer({label: `${label}/program-uniforms`, size, usage: UNIFORM_USAGE});
    arena = {buffer, size, staged: new Uint8Array(size)};
    constant = device.createBuffer({label: `${label}/program-default-attribute`, size: 16, usage: 32 | 8, mappedAtCreation: true});
    new Float32Array(constant.getMappedRange()).set([0, 0, 0, 1]);
    constant.unmap();
    return arena;
  }
  async function add(gpu, {program, textures = [], raster, topology = 'triangles', stripIndexFormat}) {
    const {reflection} = program;
    if (!reflection || typeof program.vertex !== 'string' || typeof program.fragment !== 'string') fail('ANIMATION_RENDER_OPTIONS', 'Expected a compiled program');
    if (textures.length !== reflection.textures.length) fail('ANIMATION_RENDER_TEXTURE', 'One texture binding per program sampler is required');
    if (reflection.uniformBufferSize > (device.limits.maxUniformBufferBindingSize ?? 65536)) fail('ANIMATION_RENDER_LIMIT', 'Program uniforms exceed the binding limit');
    if (reflection.outputs.length > 1 || (reflection.outputs[0] && reflection.outputs[0].location !== 0))
      fail('ANIMATION_RENDER_OPTIONS', 'Multiple program outputs need multiple render targets');
    const geometry = programGeometrySnapshot(gpu, device);
    const layouts = [...geometry.layouts];
    for (const location of geometry.channels.missing)
      layouts.push({arrayStride: 0, stepMode: 'vertex', attributes: [{shaderLocation: location, offset: 0, format: 'float32x4'}]});
    const a = ensureArena();
    const allocated = scoped(device, () => {
      const uniformLayout = device.createBindGroupLayout({label, entries: [{binding: 0, visibility: VERTEX | FRAGMENT,
        buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: reflection.uniformBufferSize}}]});
      const textureLayout = reflection.textures.length ? device.createBindGroupLayout({label, entries: reflection.textures.flatMap(t => [
        {binding: t.textureBinding, visibility: VERTEX | FRAGMENT, texture: {sampleType: t.sampleType, viewDimension: t.dimension}},
        {binding: t.samplerBinding, visibility: VERTEX | FRAGMENT, sampler: {type: t.comparison ? 'comparison' : t.sampleType === 'float' ? 'filtering' : 'non-filtering'}}])}) : null;
      const layout = device.createPipelineLayout({label, bindGroupLayouts: textureLayout ? [uniformLayout, textureLayout] : [uniformLayout]});
      const uniformGroup = device.createBindGroup({label, layout: uniformLayout, entries: [{binding: 0, resource: {buffer: a.buffer, size: reflection.uniformBufferSize}}]});
      const textureGroup = textureLayout ? device.createBindGroup({label, layout: textureLayout, entries: reflection.textures.flatMap((t, i) => [
        {binding: t.textureBinding, resource: textures[i].view}, {binding: t.samplerBinding, resource: textures[i].sampler}])}) : null;
      const vertexModule = device.createShaderModule({label: `${label}/vertex`, code: program.vertex});
      const fragmentModule = device.createShaderModule({label: `${label}/fragment`, code: program.fragment});
      const primitiveTopology = {triangles: 'triangle-list', lines: 'line-list', 'line-strip': 'line-strip', points: 'point-list'}[topology];
      if (!primitiveTopology) fail('ANIMATION_RENDER_OPTIONS', 'Invalid program topology');
      const pipeline = frontFace => device.createRenderPipelineAsync({label, layout,
        vertex: {module: vertexModule, entryPoint: 'f3d_vertex', buffers: layouts},
        fragment: {module: fragmentModule, entryPoint: 'f3d_fragment', targets: format === null ? [] : [{format,
          ...(raster.blend ? {blend: raster.blend} : {}), writeMask: raster.writeMask}]},
        primitive: {topology: primitiveTopology, cullMode: raster.cullMode, frontFace,
          ...(primitiveTopology === 'line-strip' && stripIndexFormat ? {stripIndexFormat} : {})},
        ...(depthFormat ? {depthStencil: {format: depthFormat, depthWriteEnabled: raster.depthWriteEnabled, depthCompare: raster.depthCompare,
          depthBias: raster.depthBias ?? 0, depthBiasSlopeScale: raster.depthBiasSlopeScale ?? 0}} : {}),
        multisample: {count: sampleCount}});
      const ready = Promise.all([pipeline(raster.frontFace), pipeline(raster.frontFace === 'ccw' ? 'cw' : 'ccw')]);
      ready.catch(() => {});
      return {uniformGroup, textureGroup, ready};
    });
    const [pipelines] = await Promise.race([Promise.all([allocated.value.ready, allocated.errors]), lost]);
    return {
      program: true, gpu, reflection, uniformGroup: allocated.value.uniformGroup, textureGroup: allocated.value.textureGroup,
      pipelines: {normal: pipelines[0], flipped: pipelines[1]}, geometrySignature: geometry.signature,
      blended: !!raster.blend, alphaMode: raster.blend ? 'BLEND' : 'OPAQUE', blendConstant: raster.blendConstant ?? null,
      surfaceBuffer: null, lit: false, raster: {}, disposed: false,
    };
  }
  /** Fill one command for a program draw; uniform bytes go to this frame's slice. */
  function stage(record, input, command) {
    const bytes = input.programUniforms;
    const size = record.reflection.uniformBufferSize;
    if (!(bytes instanceof Uint8Array) || bytes.byteLength !== size) fail('ANIMATION_RENDER_VALUE', 'Program draws need their packed uniform bytes');
    const geometry = programGeometrySnapshot(record.gpu, device);
    if (geometry.signature !== record.geometrySignature) fail('ANIMATION_RENDER_GEOMETRY', 'Program geometry layout changed; register it again');
    const offset = Math.ceil(cursor / ALIGN) * ALIGN;
    if (offset + size > arena.size) fail('ANIMATION_RENDER_LIMIT', 'Program uniform arena exhausted for this frame');
    arena.staged.set(bytes, offset);
    cursor = offset + size;
    const extent = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
    const range = geometry.drawRange;
    let first = Math.max(input.first ?? 0, range.first), end = Math.min((input.first ?? 0) + (input.count ?? extent), range.first + range.count, extent);
    const instanceCount = geometry.instanced ? Math.min(input.instanceCount ?? 1, geometry.instanceCapacity) : input.instanceCount ?? 1;
    Object.assign(command, {
      record, first, count: Math.max(0, end - first),
      pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal,
      program: {group: record.uniformGroup, offset},
      vertexBuffers: [...geometry.vertexBuffers, ...geometry.channels.missing.map(() => constant)],
      indexBuffer: geometry.indexBuffer, indexFormat: geometry.indexFormat,
      instanceCount, instanceBindGroup: null,
      blendConstant: record.blendConstant, stencilReference: null,
    });
    return geometry;
  }
  return {
    add, stage,
    begin() { cursor = 0; },
    write() { if (arena && cursor) device.queue.writeBuffer(arena.buffer, 0, arena.staged, 0, Math.ceil(cursor / 4) * 4); },
    dispose() { arena?.buffer.destroy(); constant?.destroy(); arena = null; constant = null; },
  };
}
