/** Compiled source programs (essl_wgsl.mjs output) as meshes of the core renderer.
 *
 * A program mesh owns two render pipelines (ccw/cw front faces, chosen per draw
 * from the object's determinant as WebGLRenderer does), its texture bind group
 * and bind groups over the renderer's bounded program-uniform pages. Each
 * logical draw gets its own device-aligned slice written before the
 * frame's single submit, so two draws of one material in one submission never
 * share uniform bytes (the queue-write snapshot rule). Registration prepares
 * pages for reflected block sizes, bounded by maxUniformBytes (default 64 MiB)
 * independently for GPU storage and CPU shadows. No guessed bytes-per-draw cap
 * or submission-time allocation is needed; earlier groups survive growth.
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
import {createProgramUniformArena} from './animation_program_uniforms.mjs';

const VERTEX = 1, FRAGMENT = 2;

export function createProgramMeshes({device, format, depthFormat, sampleCount, maxDraws, label, fail, scoped, lost, maxIndexBytes = 64 * 1024 * 1024, maxPointDraws = 65536, maxUniformBytes = 64 * 1024 * 1024}) {
  const colorTarget = createAnimationColorTargetProfile(device, {format, sampleCount},
    message => fail('ANIMATION_RENDER_OPTIONS', message));
  if (!Number.isSafeInteger(maxPointDraws) || maxPointDraws < 1 || maxPointDraws > 0xffffffff)
    fail('ANIMATION_RENDER_LIMIT', 'Invalid native point draw budget');
  let pointDraws = 0;
  const indices = createProgramIndexLowering(device, {label, fail, maxBytes: maxIndexBytes, maxEntries: maxDraws});
  const uniforms = createProgramUniformArena(device, {maxDraws, label, fail, scoped, lost, maxBytes: maxUniformBytes});
  let constant = null;
  async function add(gpu, {program, textures = [], raster, topology = 'triangles', stripIndexFormat}) {
    const source = program?.reflection, vertex = program?.vertex, fragment = program?.fragment;
    if (!source || typeof vertex !== 'string' || typeof fragment !== 'string') fail('ANIMATION_RENDER_OPTIONS', 'Expected a compiled program');
    // Uniform growth is asynchronous. Capture every subsequently consumed
    // program/binding/raster input before yielding, just as for pipeline creation.
    const reflection = {...source, textures: source.textures.map(t => ({...t})),
      outputs: source.outputs.map(o => ({...o})), varyings: source.varyings?.map(v => ({...v}))};
    textures = textures.map(t => ({...t}));
    raster = {...raster, ...(raster?.blend ? {blend: {color: {...raster.blend.color}, alpha: {...raster.blend.alpha}}} : {}),
      blendConstant: raster?.blendConstant == null ? null : (Array.isArray(raster.blendConstant) || ArrayBuffer.isView(raster.blendConstant) ? Array.from(raster.blendConstant) : {...raster.blendConstant})};
    if (textures.length !== reflection.textures.length) fail('ANIMATION_RENDER_TEXTURE', 'One texture binding per program sampler is required');
    uniforms.checkSize(reflection.uniformBufferSize);
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
    const uniformBinding = await uniforms.register(reflection.uniformBufferSize);
    uniforms.live();
    const allocated = scoped(device, () => {
      const uniformLayout = uniformBinding.layout;
      if (!constant) {
        constant = device.createBuffer({label: `${label}/program-default-attribute`, size: 16, usage: 32 | 8, mappedAtCreation: true});
        try { new Float32Array(constant.getMappedRange()).set([0, 0, 0, 1]); constant.unmap(); }
        catch (error) { constant.destroy(); constant = null; throw error; }
      }
      // Float32 textures without float32-filterable bind as unfilterable (nearest only).
      const sampleTypeOf = (t, i) => t.sampleType === 'float' && textures[i].sampleType === 'unfilterable-float' ? 'unfilterable-float' : t.sampleType;
      const textureLayout = reflection.textures.length ? device.createBindGroupLayout({label, entries: reflection.textures.flatMap((t, i) => [
        {binding: t.textureBinding, visibility: VERTEX | FRAGMENT, texture: {sampleType: sampleTypeOf(t, i), viewDimension: t.dimension}},
        {binding: t.samplerBinding, visibility: VERTEX | FRAGMENT, sampler: {type: t.comparison ? 'comparison' : sampleTypeOf(t, i) === 'float' ? 'filtering' : 'non-filtering'}}])}) : null;
      const layout = device.createPipelineLayout({label, bindGroupLayouts: textureLayout ? [uniformLayout, textureLayout] : [uniformLayout]});
      const textureGroup = textureLayout ? device.createBindGroup({label, layout: textureLayout, entries: reflection.textures.flatMap((t, i) => [
        {binding: t.textureBinding, resource: textures[i].view}, {binding: t.samplerBinding, resource: textures[i].sampler}])}) : null;
      const vertexModule = device.createShaderModule({label: `${label}/vertex`, code: vertex});
      const fragmentModule = device.createShaderModule({label: `${label}/fragment`, code: fragment});
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
      return {textureGroup, ready};
    });
    const [pipelines] = await Promise.race([Promise.all([allocated.value?.ready, allocated.errors]), uniforms.stopped]);
    if (allocated.error) throw allocated.error;
    uniforms.live();
    return {
      program: true, gpu, reflection, uniformBinding, textureGroup: allocated.value.textureGroup,
      pipelines: {normal: pipelines[0], flipped: pipelines[1]}, geometrySignature: geometry.signature,
      points, provoking, topology,
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
    const extent = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
    const range = geometry.drawRange;
    let first = Math.max(input.first ?? 0, range.first), end = Math.min((input.first ?? 0) + (input.count ?? extent), range.first + range.count, extent);
    const instanceCount = geometry.instanced ? Math.min(input.instanceCount ?? 1, geometry.instanceCapacity) : input.instanceCount ?? 1;
    if (record.points) {
      // firstInstance indexes the source vertex and supplies gl_VertexID in
      // the point shader. Ordered runs retain duplicates and restart breaks.
      const packet = uniforms.stage(record.uniformBinding, bytes);
      pointDraws += pointPlan.runs.length / 2;
      Object.assign(command, {record, first: 0, count: 6, firstInstance: geometry.indexBuffer ? 0 : first, pointCount: pointPlan.count,
        pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal, program: {...packet, pointRuns: pointPlan.runs},
        vertexBuffers: [...geometry.vertexBuffers, ...geometry.channels.missing.map(() => constant)], indexBuffer: null, indexFormat: null,
        instanceCount: pointPlan.count, instanceBindGroup: null, blendConstant: record.blendConstant, stencilReference: record.stencilReference});
      return geometry;
    }
    // WebGL2 restarts every indexed topology, but native WebGPU only restarts
    // strips. Smooth lists use the original buffer unless markers need removal.
    const lowerTopology = record.provoking ?? (geometry.indexBuffer && ['triangles', 'lines', 'points'].includes(record.topology) ? record.topology : null);
    const rotated = lowerTopology ? indices.prepare(record.gpu, geometry, lowerTopology, first, Math.max(first, end), record.provoking !== null) : null;
    const packet = uniforms.stage(record.uniformBinding, bytes);
    Object.assign(command, {
      record, first: rotated ? 0 : first, count: rotated ? rotated.count : Math.max(0, end - first),
      pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal,
      program: packet,
      vertexBuffers: [...geometry.vertexBuffers, ...geometry.channels.missing.map(() => constant)],
      indexBuffer: rotated ? rotated.buffer : geometry.indexBuffer, indexFormat: rotated?.buffer ? 'uint32' : geometry.indexFormat,
      instanceCount, firstInstance: 0, instanceBindGroup: null,
      blendConstant: record.blendConstant, stencilReference: record.stencilReference,
    });
    return geometry;
  }
  return {
    add, stage,
    begin() { uniforms.begin(); pointDraws = 0; indices.begin(); },
    write() { uniforms.write(); },
    get uniformDiagnostics() { return uniforms.diagnostics; },
    dispose() { uniforms.dispose(); constant?.destroy(); constant = null; indices.dispose(); },
  };
}
