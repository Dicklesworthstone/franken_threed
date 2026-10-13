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
 * Indexed points use bounded, source-ordered quad runs without vertex copying;
 * maxPointDraws bounds their aggregate native draw expansion per frame.
 * Nothing here interprets source objects or uniform values.
 *
 * Provoking vertex: GL takes a flat varying from a primitive's LAST vertex,
 * WebGPU's @interpolate(flat) from its FIRST. Programs with flat varyings draw
 * triangle lists through a rotated index buffer (v2, v0, v1: same winding, GL's
 * provoking vertex first) and line lists through swapped pairs. The rotated
 * indices are lowered from resident upload history, not live CPU index arrays.
 * Flat line strips become ordered line-list pairs with each last vertex first.
 * Restart markers and draw-range boundaries never connect separate primitives.
 * Immutable index buffers are cached within a separate bounded index budget;
 * superseded buffers survive all current-frame uses and retire after a fence.
 */
import {programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {createAnimationColorTargetProfile} from './animation_color_target.mjs';
import {hasAnimationStencil, snapshotAnimationRaster} from './animation_raster.mjs';
import {createProgramIndexLowering} from './animation_program_indices.mjs';
import {snapshotProgramPointRuns} from './animation_program_points.mjs';

const VERTEX = 1, FRAGMENT = 2;
const UNIFORM_USAGE = 64 | 8; // UNIFORM | COPY_DST
const ALIGN = 256;

export function createProgramMeshes({device, format, depthFormat, sampleCount, maxDraws, label, fail, scoped, lost, maxIndexBytes = 64 * 1024 * 1024, maxPointDraws = 65536}) {
  const colorTarget = createAnimationColorTargetProfile(device, {format, sampleCount},
    message => fail('ANIMATION_RENDER_OPTIONS', message));
  if (!Number.isSafeInteger(maxPointDraws) || maxPointDraws < 1 || maxPointDraws > 0xffffffff)
    fail('ANIMATION_RENDER_LIMIT', 'Invalid native point draw budget');
  let pointDraws = 0;
  const indices = createProgramIndexLowering(device, {label, fail, maxBytes: maxIndexBytes, maxEntries: maxDraws});
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
    colorTarget.assertBlend(raster?.blend);
    // GL stencil testing has no effect without a stencil attachment. Otherwise
    // capture both faces, masks and reference before geometry work or awaiting
    // pipelines; caller mutations cannot change this registration's state.
    const {stencil, stencilReference} = snapshotAnimationRaster(
      hasAnimationStencil(depthFormat) && raster?.stencil != null
        ? {stencil: raster.stencil, stencilReference: raster.stencilReference}
        : {}, {format, depthFormat});
    const geometry = programGeometrySnapshot(gpu, device);
    const points = reflection.points === true;
    if (points && topology !== 'points') fail('ANIMATION_RENDER_OPTIONS', 'Point programs draw point primitives');
    if (points && (geometry.instanced || geometry.layouts.some(l => l.stepMode === 'instance')))
      fail('ANIMATION_RENDER_GEOMETRY', 'Instanced point sprites require a separate source-instance shader path');
    const provoking = !points && reflection.varyings?.some(v => v.flat) && topology !== 'points' ? topology : null;
    if (provoking && !['triangles', 'lines', 'line-strip'].includes(provoking))
      fail('ANIMATION_RENDER_OPTIONS', 'Unsupported flat-varying topology');
    // Point sprites: one instance per source vertex (6-vertex quad each).
    const layouts = points ? geometry.layouts.map(l => ({...l, stepMode: 'instance'})) : [...geometry.layouts];
    for (const location of geometry.channels.missing)
      layouts.push({arrayStride: 0, stepMode: 'vertex', attributes: [{shaderLocation: location, offset: 0, format: 'float32x4'}]});
    const a = ensureArena();
    const allocated = scoped(device, () => {
      const uniformLayout = device.createBindGroupLayout({label, entries: [{binding: 0, visibility: VERTEX | FRAGMENT,
        buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: reflection.uniformBufferSize}}]});
      // Float32 textures without float32-filterable bind as unfilterable (nearest only).
      const sampleTypeOf = (t, i) => t.sampleType === 'float' && textures[i].sampleType === 'unfilterable-float' ? 'unfilterable-float' : t.sampleType;
      const textureLayout = reflection.textures.length ? device.createBindGroupLayout({label, entries: reflection.textures.flatMap((t, i) => [
        {binding: t.textureBinding, visibility: VERTEX | FRAGMENT, texture: {sampleType: sampleTypeOf(t, i), viewDimension: t.dimension}},
        {binding: t.samplerBinding, visibility: VERTEX | FRAGMENT, sampler: {type: t.comparison ? 'comparison' : sampleTypeOf(t, i) === 'float' ? 'filtering' : 'non-filtering'}}])}) : null;
      const layout = device.createPipelineLayout({label, bindGroupLayouts: textureLayout ? [uniformLayout, textureLayout] : [uniformLayout]});
      const uniformGroup = device.createBindGroup({label, layout: uniformLayout, entries: [{binding: 0, resource: {buffer: a.buffer, size: reflection.uniformBufferSize}}]});
      const textureGroup = textureLayout ? device.createBindGroup({label, layout: textureLayout, entries: reflection.textures.flatMap((t, i) => [
        {binding: t.textureBinding, resource: textures[i].view}, {binding: t.samplerBinding, resource: textures[i].sampler}])}) : null;
      const vertexModule = device.createShaderModule({label: `${label}/vertex`, code: program.vertex});
      const fragmentModule = device.createShaderModule({label: `${label}/fragment`, code: program.fragment});
      const primitiveTopology = points ? 'triangle-list' : provoking === 'line-strip' ? 'line-list' : {triangles: 'triangle-list', lines: 'line-list', 'line-strip': 'line-strip', points: 'point-list'}[topology];
      if (!primitiveTopology) fail('ANIMATION_RENDER_OPTIONS', 'Invalid program topology');
      const pipeline = frontFace => device.createRenderPipelineAsync({label, layout,
        vertex: {module: vertexModule, entryPoint: 'f3d_vertex', buffers: layouts},
        fragment: {module: fragmentModule, entryPoint: 'f3d_fragment', targets: format === null ? [] : [{format,
          ...(raster.blend ? {blend: raster.blend} : {}), writeMask: raster.writeMask}]},
        primitive: {topology: primitiveTopology, cullMode: raster.cullMode, frontFace,
          ...(primitiveTopology === 'line-strip' && stripIndexFormat ? {stripIndexFormat} : {})},
        ...(depthFormat ? {depthStencil: {format: depthFormat, depthWriteEnabled: raster.depthWriteEnabled, depthCompare: raster.depthCompare,
          depthBias: raster.depthBias ?? 0, depthBiasSlopeScale: raster.depthBiasSlopeScale ?? 0, ...stencil}} : {}),
        multisample: {count: sampleCount, alphaToCoverageEnabled: raster.alphaToCoverage === true && sampleCount > 1 && format !== null}});
      const ready = Promise.all([pipeline(raster.frontFace), pipeline(raster.frontFace === 'ccw' ? 'cw' : 'ccw')]);
      ready.catch(() => {});
      return {uniformGroup, textureGroup, ready};
    });
    const [pipelines] = await Promise.race([Promise.all([allocated.value.ready, allocated.errors]), lost]);
    return {
      program: true, gpu, reflection, uniformGroup: allocated.value.uniformGroup, textureGroup: allocated.value.textureGroup,
      pipelines: {normal: pipelines[0], flipped: pipelines[1]}, geometrySignature: geometry.signature,
      points, provoking,
      blended: !!raster.blend, alphaMode: raster.blend ? 'BLEND' : 'OPAQUE', blendConstant: raster.blendConstant ?? null, stencilReference,
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
    // Capture indexed point order before consuming the uniform slice. Rejected
    // expansion/range requests leave both per-frame budgets available for retry.
    const pointPlan = record.points ? snapshotProgramPointRuns(geometry, input, maxPointDraws - pointDraws, fail) : null;
    const offset = Math.ceil(cursor / ALIGN) * ALIGN;
    if (offset + size > arena.size) fail('ANIMATION_RENDER_LIMIT', 'Program uniform arena exhausted for this frame');
    arena.staged.set(bytes, offset);
    cursor = offset + size;
    const extent = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
    const range = geometry.drawRange;
    let first = Math.max(input.first ?? 0, range.first), end = Math.min((input.first ?? 0) + (input.count ?? extent), range.first + range.count, extent);
    const instanceCount = geometry.instanced ? Math.min(input.instanceCount ?? 1, geometry.instanceCapacity) : input.instanceCount ?? 1;
    if (record.points) {
      // firstInstance indexes the source vertex and supplies gl_VertexID in
      // the point shader. Ordered runs retain duplicates and restart breaks.
      pointDraws += pointPlan.runs.length / 2;
      Object.assign(command, {record, first: 0, count: 6, firstInstance: geometry.indexBuffer ? 0 : first, pointCount: pointPlan.count,
        pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal, program: {group: record.uniformGroup, offset, pointRuns: pointPlan.runs},
        vertexBuffers: [...geometry.vertexBuffers, ...geometry.channels.missing.map(() => constant)], indexBuffer: null, indexFormat: null,
        instanceCount: pointPlan.count, instanceBindGroup: null, blendConstant: record.blendConstant, stencilReference: record.stencilReference});
      return geometry;
    }
    const rotated = record.provoking ? indices.prepare(record.gpu, geometry, record.provoking, first, Math.max(first, end)) : null;
    Object.assign(command, {
      record, first: rotated ? 0 : first, count: rotated ? rotated.count : Math.max(0, end - first),
      pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal,
      program: {group: record.uniformGroup, offset},
      vertexBuffers: [...geometry.vertexBuffers, ...geometry.channels.missing.map(() => constant)],
      indexBuffer: rotated ? rotated.buffer : geometry.indexBuffer, indexFormat: rotated?.buffer ? 'uint32' : geometry.indexFormat,
      instanceCount, firstInstance: 0, instanceBindGroup: null,
      blendConstant: record.blendConstant, stencilReference: record.stencilReference,
    });
    return geometry;
  }
  return {
    add, stage,
    begin() { cursor = 0; pointDraws = 0; indices.begin(); },
    write() { if (arena && cursor) device.queue.writeBuffer(arena.buffer, 0, arena.staged, 0, Math.ceil(cursor / 4) * 4); },
    dispose() { arena?.buffer.destroy(); constant?.destroy(); arena = null; constant = null; indices.dispose(); },
  };
}
