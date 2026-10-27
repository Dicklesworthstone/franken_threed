/** F3D graphics pipelines for native WGSL node programs. Bindings follow the
 * pinned builder's groups directly, with native uniform blocks packed per
 * group plus borrowed sampler/texture/read-only storage resources at their
 * source binding indices.
 * Each logical draw gets independent aligned slices. No ESSL reflection, shader
 * rewriting, vertex-index rotation, upstream backend or hidden submission path.
 *
 * The core owns submission/validation and calls begin -> stage -> prepare ->
 * encode -> write -> submit. stage only captures bytes/resource identities;
 * prepare creates mixed bind groups inside the core's GPU error scope, after
 * all logical draws pass host validation. Existing geometry/texture owners
 * retain upload histories and lifetime; no texture is allocated or uploaded here.
 * Native uniform arena pages stay stable until this owner is disposed/lost.
 */
import {programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {createProgramUniformArena} from './animation_program_uniforms.mjs';
import {createAnimationColorTargetProfile} from './animation_color_target.mjs';
import {hasAnimationStencil, snapshotAnimationRaster, snapshotAnimationBlendConstant} from './animation_raster.mjs';
import {inspectGpuRenderTargetColors} from './gpu_render_target.mjs';

export function createThreeNodeProgramMeshes({device, format, colorFormats = format === null ? [] : [format], depthFormat, sampleCount, maxDraws,
  label, fail, scoped, lost = device.lost, maxUniformBytes = 64 * 1024 * 1024, maxBindGroups = 4}) {
  const limit = message => fail('ANIMATION_RENDER_LIMIT', message);
  if (!Number.isSafeInteger(maxDraws) || maxDraws < 1 || maxDraws > 65536 ||
      !Number.isSafeInteger(maxBindGroups) || maxBindGroups < 1 || maxBindGroups > (device.limits.maxBindGroups ?? 4) ||
      !Number.isSafeInteger(maxUniformBytes) || maxUniformBytes < 4 * maxBindGroups)
    limit('Invalid native node program uniform budget');
  const perGroupBytes = Math.floor(maxUniformBytes / maxBindGroups / 4) * 4;
  const arenas = new Map(), records = new Set(), resourceGroups = new Map(), resourceIds = new WeakMap();
  const maxResourceGroups = maxDraws * maxBindGroups;
  let pendingGroups = [], stagedDraws = 0, nextResourceId = 0;
  if (!Array.isArray(colorFormats) || (colorFormats[0] ?? null) !== format)
    fail('ANIMATION_RENDER_OPTIONS', 'Native attachment formats must retain their primary format');
  colorFormats = Object.freeze(colorFormats.slice());
  if (colorFormats.length > 1) inspectGpuRenderTargetColors(device, colorFormats);
  const colorTargets = colorFormats.map(format => createAnimationColorTargetProfile(device, {format, sampleCount},
    message => fail('ANIMATION_RENDER_OPTIONS', message)));
  let stoppedError = null, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  function error(code, message) { try { fail(code, message); } catch (value) { return value; } }
  function stop(reason) {
    if (stoppedError) return;
    stoppedError = reason; rejectStopped(reason);
    for (const record of records) record.disposed = true;
    pendingGroups = []; resourceGroups.clear();
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
  function groupDescriptor(group, index) {
    if (group.index !== index) fail('ANIMATION_RENDER_OPTIONS', 'Native node bind groups must retain their contiguous source indices');
    const byteLength = group.byteLength;
    if (!Number.isSafeInteger(byteLength) || byteLength < 0 || byteLength % 4)
      fail('ANIMATION_RENDER_OPTIONS', 'Native node group uniform byte length must be a nonnegative multiple of four');
    const sourceEntries = group.entries ?? [{binding: 0, visibility: 3,
      buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: byteLength}}];
    const bindingLimit = device.limits.maxBindingsPerBindGroup ?? 1000;
    if (!Array.isArray(sourceEntries) || sourceEntries.length > bindingLimit)
      limit('Native node group exceeds the device binding limit');
    const seen = new Set(), uniformEntries = [];
    const entries = sourceEntries.map(entry => {
      if (!Number.isSafeInteger(entry.binding) || entry.binding < 0 || entry.binding >= bindingLimit || seen.has(entry.binding) ||
          !Number.isInteger(entry.visibility) || entry.visibility < 1 || entry.visibility > 7 ||
          ['buffer', 'sampler', 'texture'].filter(kind => entry[kind] != null).length !== 1)
        fail('ANIMATION_RENDER_OPTIONS', 'Invalid native node binding index, visibility or resource kind');
      seen.add(entry.binding);
      const base = {binding: entry.binding, visibility: entry.visibility};
      if (entry.buffer) {
        if (entry.buffer.type === 'read-only-storage') {
          const minBindingSize = entry.buffer.minBindingSize ?? 0;
          if ((entry.buffer.hasDynamicOffset !== undefined && entry.buffer.hasDynamicOffset !== false) ||
              !Number.isSafeInteger(minBindingSize) || minBindingSize < 0 || minBindingSize % 4)
            fail('ANIMATION_RENDER_OPTIONS', 'Native node storage requires a static read-only buffer layout');
          if (minBindingSize > (device.limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024))
            limit('Native node storage layout exceeds the device binding size limit');
          return {...base, buffer: {type: 'read-only-storage', hasDynamicOffset: false, minBindingSize}};
        }
        const size = entry.buffer.minBindingSize;
        if (entry.buffer.type !== 'uniform' || entry.buffer.hasDynamicOffset !== true ||
            !Number.isSafeInteger(size) || size < 4 || size % 4)
          fail('ANIMATION_RENDER_OPTIONS', 'Native node uniform bindings require exact dynamic buffer ranges');
        if (size > (device.limits.maxUniformBufferBindingSize ?? 65536))
          limit('Native node uniform binding exceeds the device binding size limit');
        const uniform = {...base, buffer: {...entry.buffer}};
        uniformEntries.push(uniform); return uniform;
      }
      if (entry.sampler) {
        if (!['filtering', 'non-filtering', 'comparison'].includes(entry.sampler.type))
          fail('ANIMATION_RENDER_OPTIONS', 'Unsupported native node sampler layout');
        return {...base, sampler: {...entry.sampler}};
      }
      if (!['float', 'unfilterable-float', 'depth', 'sint', 'uint'].includes(entry.texture.sampleType) ||
          !['1d', '2d', '2d-array', 'cube', 'cube-array', '3d'].includes(entry.texture.viewDimension) ||
          typeof entry.texture.multisampled !== 'boolean')
        fail('ANIMATION_RENDER_OPTIONS', 'Unsupported native node sampled texture layout');
      return {...base, texture: {...entry.texture}};
    });
    const planned = group.uniforms === undefined
      ? uniformEntries.map(entry => ({binding: entry.binding, offset: 0, byteLength})) : group.uniforms;
    if (!Array.isArray(planned) || planned.length !== uniformEntries.length ||
        (group.uniforms === undefined && uniformEntries.length > 1) ||
        Boolean(planned.length) !== (byteLength > 0))
      fail('ANIMATION_RENDER_OPTIONS', 'Native node uniform packets must describe every group binding');
    const alignment = device.limits.minUniformBufferOffsetAlignment ?? 256, uniformBindings = new Set();
    const uniforms = planned.map(member => {
      const entry = uniformEntries.find(value => value.binding === member?.binding);
      if (!entry || uniformBindings.has(member.binding) || !Number.isSafeInteger(member.offset) || member.offset < 0 ||
          member.offset % alignment || member.byteLength !== entry.buffer.minBindingSize ||
          !Number.isSafeInteger(member.offset + member.byteLength) || member.offset + member.byteLength > byteLength)
        fail('ANIMATION_RENDER_OPTIONS', 'Native node uniform members require aligned, non-overlapping packet ranges');
      uniformBindings.add(member.binding);
      return Object.freeze({binding: member.binding, offset: member.offset, byteLength: member.byteLength});
    });
    let end = 0;
    for (const member of [...uniforms].sort((a, b) => a.offset - b.offset)) {
      if (member.offset < end) fail('ANIMATION_RENDER_OPTIONS', 'Native node uniform packet members overlap');
      end = member.offset + member.byteLength;
    }
    if (end !== byteLength) fail('ANIMATION_RENDER_OPTIONS', 'Native node uniform packet length must end at its final binding');
    uniforms.sort((a, b) => a.binding - b.binding);
    const arena = uniforms.length ? arenaAt(index) : null;
    arena?.checkPacketSize(byteLength);
    return {index, byteLength, arena, uniforms: Object.freeze(uniforms), entries,
      resources: entries.filter(entry => entry.buffer?.type !== 'uniform'),
      arenaLayout: entries.length === 1 && uniforms[0]?.binding === 0 && uniforms[0].offset === 0 && entries[0].visibility === 3};
  }
  function snapshotResources(record, input) {
    const resources = input.nodeResources ?? [];
    if (!Array.isArray(resources) || resources.length !== record.resourceCount)
      fail('ANIMATION_RENDER_VALUE', 'Native node draws need one borrowed resource per sampler/texture/storage binding');
    const captured = record.groups.map(() => new Map());
    for (const binding of resources) {
      const group = record.groups[binding?.group];
      if (!Number.isSafeInteger(binding?.group) || !group || !Number.isSafeInteger(binding.binding) ||
          !group.resources.some(entry => entry.binding === binding.binding) || captured[binding.group].has(binding.binding) ||
          !binding.resource || typeof binding.resource !== 'object')
        fail('ANIMATION_RENDER_VALUE', 'Native node resource bindings must match their group layout exactly');
      const layout = group.resources.find(entry => entry.binding === binding.binding);
      let resource = binding.resource;
      if (layout.buffer) {
        const buffer = resource.buffer, offset = resource.offset ?? 0, size = resource.size ?? (buffer?.size - offset);
        if (!buffer || typeof buffer !== 'object' || !Number.isSafeInteger(buffer.size) || buffer.size < 4 ||
            !Number.isSafeInteger(buffer.usage) || !(buffer.usage & 128) ||
            !Number.isSafeInteger(offset) || offset < 0 || offset % (device.limits.minStorageBufferOffsetAlignment ?? 256) ||
            !Number.isSafeInteger(size) || size < 4 || size % 4 || size < layout.buffer.minBindingSize ||
            !Number.isSafeInteger(offset + size) || offset + size > buffer.size)
          fail('ANIMATION_RENDER_VALUE', 'Native node storage resources require an aligned, in-bounds STORAGE buffer range');
        if (size > (device.limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024))
          limit('Native node storage resource exceeds the device binding size limit');
        // A caller can reuse or mutate its resource wrapper after stage().
        // Keep the selected allocation/range, while borrowing its GPU contents.
        resource = Object.freeze({buffer, offset, size});
      }
      captured[binding.group].set(binding.binding, resource);
    }
    return record.groups.map((group, index) => group.resources.map(entry => ({binding: entry.binding, resource: captured[index].get(entry.binding)})));
  }
  function identity(resource) {
    let id = resourceIds.get(resource);
    if (id === undefined) { id = ++nextResourceId; resourceIds.set(resource, id); }
    return id;
  }
  function prepare() {
    live();
    for (const pending of pendingGroups) {
      const {record, index, packet, resources, output} = pending, layout = record.bindGroupLayouts[index];
      const key = [identity(layout), packet ? identity(packet.resource.buffer) : 0,
        ...resources.map(entry => {
          const resource = entry.resource;
          return record.groups[index].resources.find(value => value.binding === entry.binding).buffer
            ? `buffer:${identity(resource.buffer)}:${resource.offset}:${resource.size}` : identity(resource);
        })].join('/');
      let cached = resourceGroups.get(key);
      if (cached) resourceGroups.delete(key);
      else {
        const entries = [...resources];
        if (packet) for (const uniform of record.groups[index].uniforms)
          entries.push({binding: uniform.binding, resource: {buffer: packet.resource.buffer,
            offset: uniform.offset, size: uniform.byteLength}});
        entries.sort((a, b) => a.binding - b.binding);
        cached = {record, group: device.createBindGroup({label: `${label}/node-resources`, layout, entries})};
      }
      resourceGroups.set(key, cached);
      if (resourceGroups.size > maxResourceGroups) resourceGroups.delete(resourceGroups.keys().next().value);
      output.group = cached.group;
    }
    pendingGroups = [];
  }
  async function add(gpu, {nodeProgram, raster, topology = 'triangles'}) {
    live();
    if (nodeProgram?.nodeProgram !== true || typeof nodeProgram.vertex !== 'string' ||
        typeof nodeProgram.fragment !== 'string' || !Array.isArray(nodeProgram.groups) ||
        nodeProgram.groups.length > maxBindGroups)
      fail('ANIMATION_RENDER_OPTIONS', 'Expected a native node graphics program');
    const instanceCapacity = nodeProgram.instanceCapacity ?? Infinity;
    if (nodeProgram.instanceCapacity !== undefined &&
        (!Number.isSafeInteger(instanceCapacity) || instanceCapacity < 0 || instanceCapacity > 0xffffffff))
      fail('ANIMATION_RENDER_OPTIONS', 'Native node instance capacity must be a nonnegative 32-bit integer');
    if (!['triangles', 'lines'].includes(topology)) fail('ANIMATION_RENDER_OPTIONS', 'Native node graphics require triangle or line-list primitives');
    const groups = nodeProgram.groups.map(groupDescriptor);
    if (groups.reduce((count, group) => count + group.uniforms.length, 0) >
        (device.limits.maxDynamicUniformBuffersPerPipelineLayout ?? 8))
      limit('Native node bindings exceed maxDynamicUniformBuffersPerPipelineLayout');
    for (const stage of [1, 2, 4]) for (const [matches, name, fallback] of [
      [entry => entry.buffer?.type === 'uniform', 'maxUniformBuffersPerShaderStage', 12],
      [entry => entry.buffer?.type === 'read-only-storage', 'maxStorageBuffersPerShaderStage', 8],
      [entry => entry.sampler, 'maxSamplersPerShaderStage', 16],
      [entry => entry.texture, 'maxSampledTexturesPerShaderStage', 16]])
      if (groups.reduce((count, group) => count + group.entries.filter(entry => matches(entry) && (entry.visibility & stage)).length, 0) > (device.limits[name] ?? fallback))
        limit(`Native node bindings exceed ${name}`);
    const vertex = nodeProgram.vertex, fragment = nodeProgram.fragment;
    const colorCount = nodeProgram.colorCount ?? 1;
    if (!Number.isSafeInteger(colorCount) || colorCount < 1 || colorCount > Math.max(1, colorFormats.length))
      fail('ANIMATION_RENDER_OPTIONS', 'Native fragment outputs exceed the active color attachment layout');
    if (!raster || !['ccw', 'cw'].includes(raster.frontFace) || !['none', 'front', 'back'].includes(raster.cullMode) ||
        !Number.isSafeInteger(raster.writeMask) || raster.writeMask < 0 || raster.writeMask > 15 ||
        (raster.alphaToCoverageEnabled !== undefined && typeof raster.alphaToCoverageEnabled !== 'boolean') ||
        (depthFormat && (typeof raster.depthWriteEnabled !== 'boolean' ||
          !['never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always'].includes(raster.depthCompare))))
      fail('ANIMATION_RENDER_OPTIONS', 'Invalid native node graphics raster state');
    if (raster.colorBlends !== undefined && (!Array.isArray(raster.colorBlends) || raster.colorBlends.length !== colorFormats.length))
      fail('ANIMATION_RENDER_OPTIONS', 'Native blend states must match the color attachment layout');
    const colorStates = colorFormats.map((format, index) => {
      const blend = index >= colorCount ? null : raster.colorBlends === undefined ? raster.blend : raster.colorBlends[index];
      const state = snapshotAnimationRaster({...raster, blend, blendConstant: undefined}, {format, depthFormat});
      colorTargets[index].assertBlend(state.blend);
      return state;
    });
    const usesConstant = colorStates.some(state => state.usesConstant);
    const blendConstant = raster.colorBlends !== undefined && raster.blendConstant !== undefined
      ? raster.blendConstant === null ? null : snapshotAnimationBlendConstant(raster.blendConstant)
      : usesConstant ? snapshotAnimationBlendConstant(raster.blendConstant ?? [0, 0, 0, 0]) : null;
    const state = Object.freeze({frontFace: raster.frontFace, cullMode: raster.cullMode, writeMask: raster.writeMask,
      depthWriteEnabled: raster.depthWriteEnabled, depthCompare: raster.depthCompare,
      alphaToCoverageEnabled: raster.alphaToCoverageEnabled === true && sampleCount > 1,
      ...snapshotAnimationRaster({...raster, blendConstant: undefined}, {format, depthFormat}), usesConstant,
      blendConstant});
    const geometry = programGeometrySnapshot(gpu, device);
    if (geometry.channels.missing.length)
      fail('ANIMATION_RENDER_GEOMETRY', 'Native node graphics require complete source geometry attributes');
    const geometrySignature = geometry.signature, geometryInstanced = geometry.instanced, layouts = geometry.layouts;
    // Capture all source-facing values before registrations or pipeline work
    // can yield. Older program records/groups survive arena growth unchanged.
    const uniformBindings = [];
    for (const group of groups) uniformBindings.push(group.arena ? await (group.arenaLayout
      ? group.arena.register(group.byteLength) : group.arena.registerPacket(group.byteLength)) : null);
    live();
    let bindGroupLayouts;
    const issued = scoped(device, () => {
      bindGroupLayouts = groups.map((group, index) => group.arenaLayout ? uniformBindings[index].layout
        : device.createBindGroupLayout({label: `${label}/node-group-${index}`, entries: group.entries}));
      const layout = device.createPipelineLayout({label, bindGroupLayouts});
      const vertexModule = device.createShaderModule({label: `${label}/node-vertex`, code: vertex});
      const fragmentModule = device.createShaderModule({label: `${label}/node-fragment`, code: fragment});
      const stencil = hasAnimationStencil(depthFormat) ? state.stencil : {};
      const pipeline = frontFace => device.createRenderPipelineAsync({label, layout,
        vertex: {module: vertexModule, entryPoint: 'main', buffers: layouts},
        fragment: {module: fragmentModule, entryPoint: 'main', targets: colorFormats.map((format, index) => ({format,
          ...(colorStates[index].blend ? {blend: colorStates[index].blend} : {}),
          writeMask: index < colorCount ? state.writeMask : 0}))},
        primitive: {topology: topology === 'lines' ? 'line-list' : 'triangle-list', cullMode: state.cullMode, frontFace},
        ...(depthFormat ? {depthStencil: {format: depthFormat, depthWriteEnabled: state.depthWriteEnabled,
          depthCompare: state.depthCompare, ...state.bias,
          ...stencil}} : {}),
        multisample: {count: sampleCount, alphaToCoverageEnabled: state.alphaToCoverageEnabled}});
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
    const record = {nodeProgram: true, gpu, groups, uniformBindings, bindGroupLayouts, geometrySignature, geometryInstanced, instanceCapacity,
      resourceCount: groups.reduce((sum, group) => sum + group.resources.length, 0),
      pipelines: {normal: pipelines[0], flipped: pipelines[1]},
      blended: !!state.blend, alphaMode: state.blend ? 'BLEND' : 'OPAQUE',
      blendConstant: state.blendConstant ?? null, stencilReference: state.stencilReference,
      surfaceBuffer: null, textureGroup: null, lit: false, raster: state, disposed: false};
    records.add(record); return record;
  }
  function stage(record, input, command) {
    live();
    if (!records.has(record) || record.disposed) fail('ANIMATION_RENDER_VALUE', 'Native node graphics record is not live');
    if (stagedDraws >= maxDraws) limit('Native node draw capacity exhausted for this frame');
    if (['first', 'count'].some(key => input[key] !== undefined && (!Number.isSafeInteger(input[key]) || input[key] < 0)) ||
        (input.instanceCount !== undefined && (!Number.isSafeInteger(input.instanceCount) || input.instanceCount < 0 || input.instanceCount > 0xffffffff)) ||
        (input.firstInstance !== undefined && (!Number.isSafeInteger(input.firstInstance) || input.firstInstance < 0 || input.firstInstance > 0xffffffff)) ||
        (input.frontFaceCW !== undefined && typeof input.frontFaceCW !== 'boolean') ||
        (input.nodeRangeCaptured !== undefined && typeof input.nodeRangeCaptured !== 'boolean') ||
        (input.nodeRangeCaptured === true && (input.first === undefined || input.count === undefined)))
      fail('ANIMATION_RENDER_VALUE', 'Native node draws require nonnegative integer ranges and boolean winding/capture flags');
    const bytes = input.nodeUniforms;
    if (!Array.isArray(bytes) || bytes.length !== record.groups.length ||
        bytes.some((value, index) => !(value instanceof Uint8Array) || value.byteLength !== record.groups[index].byteLength ||
          !(value.buffer instanceof ArrayBuffer) || value.buffer.resizable))
      fail('ANIMATION_RENDER_VALUE', 'Native node draws need one owned uniform byte packet per native group');
    const resources = snapshotResources(record, input);
    const geometry = programGeometrySnapshot(record.gpu, device);
    if (geometry.signature !== record.geometrySignature || geometry.instanced !== record.geometryInstanced)
      fail('ANIMATION_RENDER_GEOMETRY', 'Native node geometry layout changed; register it again');
    const instanceCount = input.instanceCount ?? 1, firstInstance = input.firstInstance ?? 0;
    if (instanceCount + firstInstance > 0x100000000 ||
        instanceCount + firstInstance > Math.min(record.instanceCapacity, geometry.instanceCapacity ?? Infinity))
      fail('ANIMATION_RENDER_VALUE', 'Native node instance count exceeds the captured object or vertex stream capacity');
    const extent = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
    // A facade request already selected its source range synchronously. Later
    // drawRange edits must not replace that history during pipeline preparation.
    // Actual geometry storage/extent remains upload-version governed as before.
    const range = input.nodeRangeCaptured ? {first: 0, count: extent} : geometry.drawRange;
    const first = Math.min(extent, Math.max(input.first ?? 0, range.first));
    const end = Math.min((input.first ?? 0) + (input.count ?? extent), range.first + range.count, extent);
    const groups = record.groups.map((group, index) => {
      const packet = group.arena ? group.arena.stage(record.uniformBindings[index], bytes[index]) : null;
      const offsets = Object.freeze(group.uniforms.map(() => packet.offset));
      if (group.arenaLayout) return {...packet, offsets};
      const output = {group: null, offset: packet?.offset ?? null, offsets};
      pendingGroups.push({record, index, packet, resources: resources[index], output});
      return output;
    });
    stagedDraws++;
    Object.assign(command, {record, first, count: Math.max(0, end - first),
      pipeline: input.frontFaceCW ? record.pipelines.flipped : record.pipelines.normal,
      nodeProgram: {groups}, vertexBuffers: geometry.vertexBuffers,
      indexBuffer: geometry.indexBuffer, indexFormat: geometry.indexFormat,
      instanceCount, firstInstance, instanceBindGroup: null, blendConstant: record.blendConstant, stencilReference: record.stencilReference});
    return geometry;
  }
  return Object.freeze({add, stage, prepare,
    begin() { live(); pendingGroups = []; stagedDraws = 0; for (const arena of arenas.values()) arena.begin(); },
    write() { live(); for (const arena of arenas.values()) arena.write(); },
    remove(record) {
      if (records.delete(record)) record.disposed = true;
      pendingGroups = pendingGroups.filter(pending => pending.record !== record);
      for (const [key, value] of resourceGroups) if (value.record === record) resourceGroups.delete(key);
    },
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
