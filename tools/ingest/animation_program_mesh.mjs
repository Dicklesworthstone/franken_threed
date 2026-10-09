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
 *
 * Provoking vertex: GL takes a flat varying from a primitive's LAST vertex,
 * WebGPU's @interpolate(flat) from its FIRST. Programs with flat varyings draw
 * triangle lists through a rotated index buffer (v2, v0, v1: same winding, GL's
 * provoking vertex first) and line lists through swapped pairs. The rotated
 * indices are rebuilt from the source index array when its version, identity or
 * count changes (the residency uploads on the same version rule). Line strips
 * with flat varyings fail explicitly.
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
    const points = reflection.points === true;
    if (points && topology !== 'points') fail('ANIMATION_RENDER_OPTIONS', 'Point programs draw point primitives');
    if (points && (geometry.indexBuffer || geometry.instanced)) fail('ANIMATION_RENDER_GEOMETRY', 'Indexed or instanced point sprites are not admitted yet');
    const provoking = !points && reflection.varyings?.some(v => v.flat) ? {triangles: 3, lines: 2}[topology] ?? 0 : 0;
    if (!points && !provoking && reflection.varyings?.some(v => v.flat) && topology !== 'points')
      fail('ANIMATION_RENDER_OPTIONS', 'Flat varyings on line strips need GL\'s last-vertex convention; not admitted yet');
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
      const primitiveTopology = points ? 'triangle-list' : {triangles: 'triangle-list', lines: 'line-list', 'line-strip': 'line-strip', points: 'point-list'}[topology];
      if (!primitiveTopology) fail('ANIMATION_RENDER_OPTIONS', 'Invalid program topology');
      const pipeline = frontFace => device.createRenderPipelineAsync({label, layout,
        vertex: {module: vertexModule, entryPoint: 'f3d_vertex', buffers: layouts},
        fragment: {module: fragmentModule, entryPoint: 'f3d_fragment', targets: format === null ? [] : [{format,
          ...(raster.blend ? {blend: raster.blend} : {}), writeMask: raster.writeMask}]},
        primitive: {topology: primitiveTopology, cullMode: raster.cullMode, frontFace,
          ...(primitiveTopology === 'line-strip' && stripIndexFormat ? {stripIndexFormat} : {})},
        ...(depthFormat ? {depthStencil: {format: depthFormat, depthWriteEnabled: raster.depthWriteEnabled, depthCompare: raster.depthCompare,
          depthBias: raster.depthBias ?? 0, depthBiasSlopeScale: raster.depthBiasSlopeScale ?? 0}} : {}),
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
    if (record.points) {
      // Point sprites: count points become instances of a 6-vertex quad.
      Object.assign(command, {record, first: 0, count: 6, firstInstance: first, pointCount: Math.max(0, end - first),
        pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal, program: {group: record.uniformGroup, offset},
        vertexBuffers: [...geometry.vertexBuffers, ...geometry.channels.missing.map(() => constant)], indexBuffer: null, indexFormat: null,
        instanceCount: Math.max(0, end - first), instanceBindGroup: null, blendConstant: record.blendConstant, stencilReference: null});
      return geometry;
    }
    const rotated = record.provoking ? provokingIndices(record, geometry, first, end) : null;
    Object.assign(command, {
      record, first, count: Math.max(0, end - first),
      pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal,
      program: {group: record.uniformGroup, offset},
      vertexBuffers: [...geometry.vertexBuffers, ...geometry.channels.missing.map(() => constant)],
      indexBuffer: rotated ?? geometry.indexBuffer, indexFormat: rotated ? 'uint32' : geometry.indexFormat,
      instanceCount, firstInstance: 0, instanceBindGroup: null,
      blendConstant: record.blendConstant, stencilReference: null,
    });
    return geometry;
  }
  // gpu residency -> Map(phase -> {key, buffer}); rebuilt only when the source changes.
  const rotations = new WeakMap(), rotationBuffers = new Set();
  function provokingIndices(record, geometry, first, end) {
    const n = record.provoking, phase = first % n, index = geometry.indexBuffer ? geometry.source?.index : null;
    if (geometry.indexBuffer && !index?.array) fail('ANIMATION_RENDER_GEOMETRY', 'Flat varyings need the source index array');
    const length = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
    let byPhase = rotations.get(record.gpu);
    if (!byPhase) rotations.set(record.gpu, byPhase = new Map());
    const key = geometry.indexBuffer ? [geometry.indexBuffer, index.version, length] : [null, -1, length];
    let entry = byPhase.get(phase);
    if (!entry || entry.key.some((v, i) => v !== key[i])) {
      const out = new Uint32Array(Math.max(1, length));
      for (let i = 0; i < length; i++) out[i] = index ? index.array[i] : i;
      // Rotate every whole primitive that starts at first + k*n.
      for (let s = phase; s + n <= length; s += n) {
        if (n === 3) { const a = out[s], b = out[s + 1], c = out[s + 2]; out[s] = c; out[s + 1] = a; out[s + 2] = b; }
        else { const a = out[s]; out[s] = out[s + 1]; out[s + 1] = a; }
      }
      if (entry) { entry.buffer.destroy(); rotationBuffers.delete(entry.buffer); }
      const buffer = device.createBuffer({label: `${label}/provoking-indices`, size: Math.max(4, out.byteLength), usage: 16 | 8});
      device.queue.writeBuffer(buffer, 0, out);
      rotationBuffers.add(buffer);
      byPhase.set(phase, entry = {key, buffer});
    }
    return entry.buffer;
  }
  return {
    add, stage,
    begin() { cursor = 0; },
    write() { if (arena && cursor) device.queue.writeBuffer(arena.buffer, 0, arena.staged, 0, Math.ceil(cursor / 4) * 4); },
    dispose() { arena?.buffer.destroy(); constant?.destroy(); arena = null; constant = null; for (const b of rotationBuffers) b.destroy(); rotationBuffers.clear(); },
  };
}
