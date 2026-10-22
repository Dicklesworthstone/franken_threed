/** F3D graphics pipelines for native WGSL node programs. Bindings follow the
 * pinned builder's groups directly, with one native uniform block per group.
 * Each logical draw gets independent aligned slices. No ESSL reflection, shader
 * rewriting, vertex-index rotation, upstream backend or hidden submission path.
 *
 * The core owns submission/validation and calls begin -> stage -> write ->
 * encode/submit. Existing program geometry owns attribute upload histories.
 * Native uniform arena pages stay stable until this owner is disposed/lost.
 */
import {programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {createProgramUniformArena} from './animation_program_uniforms.mjs';
import {createAnimationColorTargetProfile} from './animation_color_target.mjs';
import {hasAnimationStencil, snapshotAnimationRaster} from './animation_raster.mjs';

export function createThreeNodeProgramMeshes({device, format, depthFormat, sampleCount, maxDraws,
  label, fail, scoped, lost = device.lost, maxUniformBytes = 64 * 1024 * 1024, maxBindGroups = 4}) {
  const limit = message => fail('ANIMATION_RENDER_LIMIT', message);
  if (!Number.isSafeInteger(maxBindGroups) || maxBindGroups < 1 || maxBindGroups > (device.limits.maxBindGroups ?? 4) ||
      !Number.isSafeInteger(maxUniformBytes) || maxUniformBytes < 4 * maxBindGroups)
    limit('Invalid native node program uniform budget');
  const perGroupBytes = Math.floor(maxUniformBytes / maxBindGroups / 4) * 4;
  const arenas = new Map(), records = new Set();
  const colorTarget = createAnimationColorTargetProfile(device, {format, sampleCount},
    message => fail('ANIMATION_RENDER_OPTIONS', message));
  let stoppedError = null, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  function error(code, message) { try { fail(code, message); } catch (value) { return value; } }
  function stop(reason) {
    if (stoppedError) return;
    stoppedError = reason; rejectStopped(reason);
    for (const record of records) record.disposed = true;
    records.clear(); for (const arena of arenas.values()) arena.dispose(); arenas.clear();
  }
  function live() { if (stoppedError) throw stoppedError; }
  lost.then(info => stop(error('ANIMATION_RENDER_LOST', info?.message || 'WebGPU device lost')), stop).catch(() => {});
  function arenaAt(index) {
    live();
    let arena = arenas.get(index);
    if (!arena) {
      arena = createProgramUniformArena(device, {maxDraws, label: `${label}/node-group-${index}`,
        fail, scoped, lost, maxBytes: perGroupBytes});
      arenas.set(index, arena);
    }
    return arena;
  }
  async function add(gpu, {nodeProgram, raster, topology = 'triangles'}) {
    live();
    if (nodeProgram?.nodeProgram !== true || typeof nodeProgram.vertex !== 'string' ||
        typeof nodeProgram.fragment !== 'string' || !Array.isArray(nodeProgram.groups) ||
        nodeProgram.groups.length > maxBindGroups)
      fail('ANIMATION_RENDER_OPTIONS', 'Expected a native node graphics program');
    if (topology !== 'triangles') fail('ANIMATION_RENDER_OPTIONS', 'Native node graphics currently draw triangle meshes');
    const groups = nodeProgram.groups.map((group, index) => {
      if (group.index !== index) fail('ANIMATION_RENDER_OPTIONS', 'Native node bind groups must retain their contiguous source indices');
      const arena = arenaAt(index); arena.checkSize(group.byteLength);
      return {index, byteLength: group.byteLength, arena};
    });
    const vertex = nodeProgram.vertex, fragment = nodeProgram.fragment;
    if (!raster || !['ccw', 'cw'].includes(raster.frontFace) || !['none', 'front', 'back'].includes(raster.cullMode) ||
        !Number.isSafeInteger(raster.writeMask) || raster.writeMask < 0 || raster.writeMask > 15 ||
        (depthFormat && (typeof raster.depthWriteEnabled !== 'boolean' ||
          !['never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always'].includes(raster.depthCompare))))
      fail('ANIMATION_RENDER_OPTIONS', 'Invalid native node graphics raster state');
    const state = Object.freeze({frontFace: raster.frontFace, cullMode: raster.cullMode, writeMask: raster.writeMask,
      depthWriteEnabled: raster.depthWriteEnabled, depthCompare: raster.depthCompare,
      ...snapshotAnimationRaster(raster, {format, depthFormat})});
    colorTarget.assertBlend(state.blend);
    const geometry = programGeometrySnapshot(gpu, device);
    if (geometry.channels.missing.length || geometry.instanced || geometry.layouts.some(layout => layout.stepMode === 'instance'))
      fail('ANIMATION_RENDER_GEOMETRY', 'Native node graphics require complete ordinary vertex attributes');
    const geometrySignature = geometry.signature, layouts = geometry.layouts;
    // Capture all source-facing values before registrations or pipeline work
    // can yield. Older program records/groups survive arena growth unchanged.
    const uniformBindings = [];
    for (const group of groups) uniformBindings.push(await group.arena.register(group.byteLength));
    live();
    const issued = scoped(device, () => {
      const layout = device.createPipelineLayout({label, bindGroupLayouts: uniformBindings.map(binding => binding.layout)});
      const vertexModule = device.createShaderModule({label: `${label}/node-vertex`, code: vertex});
      const fragmentModule = device.createShaderModule({label: `${label}/node-fragment`, code: fragment});
      const stencil = hasAnimationStencil(depthFormat) ? state.stencil : {};
      const pipeline = frontFace => device.createRenderPipelineAsync({label, layout,
        vertex: {module: vertexModule, entryPoint: 'main', buffers: layouts},
        fragment: {module: fragmentModule, entryPoint: 'main', targets: format === null ? [] : [{format,
          ...(state.blend ? {blend: state.blend} : {}), writeMask: state.writeMask}]},
        primitive: {topology: 'triangle-list', cullMode: state.cullMode, frontFace},
        ...(depthFormat ? {depthStencil: {format: depthFormat, depthWriteEnabled: state.depthWriteEnabled,
          depthCompare: state.depthCompare, ...state.bias,
          ...stencil}} : {}),
        multisample: {count: sampleCount}});
      const ready = Promise.all([pipeline(state.frontFace), pipeline(state.frontFace === 'ccw' ? 'cw' : 'ccw')]);
      ready.catch(() => {}); return ready;
    });
    let pipelines;
    try {
      const ready = Promise.all([issued.value, issued.errors]);
      [pipelines] = await Promise.race([ready, stopped]);
      if (issued.error) throw issued.error;
      live();
    } catch (error) { issued.errors.catch(() => {}); throw error; }
    const record = {nodeProgram: true, gpu, groups, uniformBindings, geometrySignature,
      pipelines: {normal: pipelines[0], flipped: pipelines[1]},
      blended: !!state.blend, alphaMode: state.blend ? 'BLEND' : 'OPAQUE',
      blendConstant: state.blendConstant ?? null, stencilReference: state.stencilReference,
      surfaceBuffer: null, textureGroup: null, lit: false, raster: state, disposed: false};
    records.add(record); return record;
  }
  function stage(record, input, command) {
    live();
    if (!records.has(record) || record.disposed) fail('ANIMATION_RENDER_VALUE', 'Native node graphics record is not live');
    if (['first', 'count'].some(key => input[key] !== undefined && (!Number.isSafeInteger(input[key]) || input[key] < 0)) ||
        (input.frontFaceCW !== undefined && typeof input.frontFaceCW !== 'boolean') ||
        (input.nodeRangeCaptured !== undefined && typeof input.nodeRangeCaptured !== 'boolean') ||
        (input.nodeRangeCaptured === true && (input.first === undefined || input.count === undefined)))
      fail('ANIMATION_RENDER_VALUE', 'Native node draws require nonnegative integer ranges and boolean winding/capture flags');
    const bytes = input.nodeUniforms;
    if (!Array.isArray(bytes) || bytes.length !== record.groups.length ||
        bytes.some((value, index) => !(value instanceof Uint8Array) || value.byteLength !== record.groups[index].byteLength ||
          !(value.buffer instanceof ArrayBuffer) || value.buffer.resizable))
      fail('ANIMATION_RENDER_VALUE', 'Native node draws need one owned uniform byte packet per native group');
    const geometry = programGeometrySnapshot(record.gpu, device);
    if (geometry.signature !== record.geometrySignature)
      fail('ANIMATION_RENDER_GEOMETRY', 'Native node geometry layout changed; register it again');
    const extent = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
    // A facade request already selected its source range synchronously. Later
    // drawRange edits must not replace that history during pipeline preparation.
    // Actual geometry storage/extent remains upload-version governed as before.
    const range = input.nodeRangeCaptured ? {first: 0, count: extent} : geometry.drawRange;
    const first = Math.min(extent, Math.max(input.first ?? 0, range.first));
    const end = Math.min((input.first ?? 0) + (input.count ?? extent), range.first + range.count, extent);
    const groups = record.groups.map((group, index) => group.arena.stage(record.uniformBindings[index], bytes[index]));
    Object.assign(command, {record, first, count: Math.max(0, end - first),
      pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal,
      nodeProgram: {groups}, vertexBuffers: geometry.vertexBuffers,
      indexBuffer: geometry.indexBuffer, indexFormat: geometry.indexFormat,
      instanceCount: 1, instanceBindGroup: null, blendConstant: record.blendConstant, stencilReference: record.stencilReference});
    return geometry;
  }
  return Object.freeze({add, stage,
    begin() { live(); for (const arena of arenas.values()) arena.begin(); },
    write() { live(); for (const arena of arenas.values()) arena.write(); },
    remove(record) { if (records.delete(record)) record.disposed = true; },
    dispose() { stop(error('ANIMATION_RENDER_DISPOSED', 'Native node graphics owner is disposed')); },
    get uniformDiagnostics() {
      const values = [...arenas.values()].map(arena => arena.diagnostics);
      return Object.freeze({allocatedBytes: values.reduce((sum, v) => sum + v.allocatedBytes, 0),
        shadowBytes: values.reduce((sum, v) => sum + v.shadowBytes, 0), pages: values.reduce((sum, v) => sum + v.pages, 0),
        bindings: values.reduce((sum, v) => sum + v.bindings, 0), maxBytes: maxUniformBytes,
        groups: values.length, draws: Math.max(0, ...values.map(v => v.draws))});
    },
  });
}
