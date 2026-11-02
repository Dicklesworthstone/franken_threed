/** Internal draw encoding shared by direct passes and persistent render bundles.
 * Inputs are already admitted, source-ordered commands from animation_render.
 * Never inspect scene objects, CPU array values, or source version counters here.
 *
 * A cache belongs to ONE device, attachment layout and uniform/storage arena.
 * Its key contains every recorded pipeline, binding, offset and draw parameter.
 * Buffer contents, camera/light uniforms, viewport and attachments are NOT baked.
 * The caller still validates inputs, writes current data and submits each frame.
 * WebGPU validation/loss handling belongs to that caller's error scopes.
 */
import {hasAnimationStencil} from './animation_raster.mjs';
const blended = record => record.alphaMode === 'BLEND' || record.blended === true;
const nodeOffsets = binding => binding.offsets ?? (binding.offset === null ? [] : [binding.offset]);
function encodeAnimationNodeDraw(encoder, command) {
  encoder.setPipeline(command.pipeline);
  for (let group = 0; group < command.nodeProgram.groups.length; group++) {
    const binding = command.nodeProgram.groups[group];
    encoder.setBindGroup(group, binding.group, nodeOffsets(binding));
  }
  for (let slot = 0; slot < command.vertexBuffers.length; slot++) encoder.setVertexBuffer(slot, command.vertexBuffers[slot]);
  const indirect = command.nodeIndirect;
  if (command.indexBuffer) {
    encoder.setIndexBuffer(command.indexBuffer, command.indexFormat);
    if (indirect) for (const offset of indirect.offsets) encoder.drawIndexedIndirect(indirect.buffer, offset);
    else encoder.drawIndexed(command.count, command.instanceCount, command.first, 0, command.firstInstance ?? 0);
  } else if (indirect) {
    for (const offset of indirect.offsets) encoder.drawIndirect(indirect.buffer, offset);
  } else encoder.draw(command.count, command.instanceCount, command.first, command.firstInstance ?? 0);
  return indirect ? indirect.offsets.length : 1;
}
function compatible(a, b) {
  return !a.program && !b.program && !a.nodeProgram && !b.nodeProgram &&
    a.instanceCount == null && b.instanceCount == null &&
    !blended(b.record) && a.pipeline === b.pipeline &&
    a.first === b.first && a.count === b.count &&
    a.vertexBuffers.length === b.vertexBuffers.length &&
    a.vertexBuffers.every((buffer, i) => buffer === b.vertexBuffers[i]) &&
    a.indexBuffer === b.indexBuffer && a.indexFormat === b.indexFormat &&
    a.record.surfaceBuffer === b.record.surfaceBuffer &&
    a.record.textureGroup === b.record.textureGroup;
}

/** Execute one already-selected physical command/run. Used by the backend's
 * observable draw operation and the direct path, with identical GPU bindings. */
export function encodeAnimationDrawCommand(encoder, command, {bindGroup, lightGroup, stride, instancing}, index, instances = 1) {
  if (command.nodeProgram) return encodeAnimationNodeDraw(encoder, command);
  const {record, first, count, pipeline} = command;
  if (command.program) {
    encoder.setPipeline(pipeline);
    encoder.setBindGroup(0, command.program.group, [command.program.offset]);
    if (record.textureGroup) encoder.setBindGroup(1, record.textureGroup);
    for (let slot = 0; slot < command.vertexBuffers.length; slot++) encoder.setVertexBuffer(slot, command.vertexBuffers[slot]);
    const pointRuns = command.program.pointRuns;
    if (pointRuns != null) {
      for (let run = 0; run < pointRuns.length; run += 2) encoder.draw(6, pointRuns[run + 1], 0, pointRuns[run]);
      return pointRuns.length / 2;
    }
    if (command.indexBuffer) {
      encoder.setIndexBuffer(command.indexBuffer, command.indexFormat);
      encoder.drawIndexed(count, command.instanceCount, first, 0, 0);
    } else encoder.draw(count, command.instanceCount, first, command.firstInstance ?? 0);
    return 1;
  }
  const native = command.instanceCount != null;
  encoder.setPipeline(pipeline);
  encoder.setBindGroup(0, native ? command.instanceBindGroup : bindGroup,
    instancing && !native ? [] : [index * stride]);
  for (let slot = 0; slot < command.vertexBuffers.length; slot++) encoder.setVertexBuffer(slot, command.vertexBuffers[slot]);
  if (record.surfaceBuffer) encoder.setVertexBuffer(1, record.surfaceBuffer);
  if (record.textureGroup) encoder.setBindGroup(1, record.textureGroup);
  if (record.lit) encoder.setBindGroup(record.textureGroup ? 2 : 1, lightGroup);
  if (command.indexBuffer) {
    encoder.setIndexBuffer(command.indexBuffer, command.indexFormat);
    encoder.drawIndexed(count, native ? command.instanceCount : instances, first, 0, instancing && !native ? index : 0);
  } else encoder.draw(count, native ? command.instanceCount : instances, first, instancing && !native ? index : 0);
  return 1;
}

/** Fully bind every draw/run, including after bundles clear pass state. Physical
 * instancing decisions and original arena indices are independent of inspection. */
export function encodeAnimationDraws(encoder, commands, length, parameters) {
  const {instancing, start = 0, operations = null} = parameters;
  let calls = 0;
  const end = start + length;
  for (let i = start; i < end;) {
    const command = commands[i];
    // Empty native draws still run source updates but issue no GPU command.
    if (command.nodeProgram && ((!command.nodeIndirect && command.count === 0) ||
        command.instanceCount === 0 || command.nodeIndirect?.skip)) { i++; continue; }
    let instances = 1;
    if (!command.nodeProgram && !command.program && instancing && command.instanceCount == null && !blended(command.record))
      while (i + instances < end && compatible(command, commands[i + instances])) instances++;
    calls += operations && command.source
      ? operations.backend.draw(operations.backend._drawObject(commands, i, instances, parameters), encoder, command, parameters, i, instances)
      : encodeAnimationDrawCommand(encoder, command, parameters, i, instances);
    i += instances;
  }
  return calls;
}
function snapshot(command, lightGroup) {
  return {pipeline: command.pipeline, first: command.first, count: command.count,
    instanceCount: command.instanceCount, instanceBindGroup: command.instanceBindGroup,
    vertexBuffers: command.vertexBuffers.slice(), indexBuffer: command.indexBuffer,
    indexFormat: command.indexFormat, surfaceBuffer: command.record.surfaceBuffer,
    textureGroup: command.record.textureGroup, lit: command.record.lit,
    lightGroup: command.record.lit ? lightGroup : null,
    blend: blended(command.record),
    // Plans are immutable, owned snapshots; never retain a live index reader.
    pointRuns: command.program?.pointRuns ?? null,
    indirectBuffer: command.nodeIndirect?.buffer ?? null,
    indirectOffsets: command.nodeIndirect?.offsets ?? null,
    indirectSkip: command.nodeIndirect?.skip ?? false,
    nodeGroups: command.nodeProgram?.groups.map(binding => ({group: binding.group, offsets: [...nodeOffsets(binding)]})) ?? null,
    programGroup: command.program?.group ?? null, programOffset: command.program?.offset ?? null, firstInstance: command.firstInstance ?? 0};
}
function samePointRuns(a, b) {
  return a === b || (a != null && b != null && a.length === b.length && a.every((value, i) => value === b[i]));
}
function sameNodeGroups(a, b) {
  return a == null ? b == null : b != null && a.length === b.length &&
    a.every((binding, index) => binding.group === b[index].group && samePointRuns(binding.offsets, nodeOffsets(b[index])));
}
function matches(recorded, commands, length, lightGroup, start) {
  if (recorded.length !== length) return false;
  for (let i = 0; i < length; i++) {
    const a = recorded[i], b = commands[start + i], r = b.record;
    if (a.pipeline !== b.pipeline || a.first !== b.first || a.count !== b.count ||
        a.instanceCount !== b.instanceCount || a.instanceBindGroup !== b.instanceBindGroup ||
        a.indexBuffer !== b.indexBuffer || a.indexFormat !== b.indexFormat ||
        a.surfaceBuffer !== r.surfaceBuffer || a.textureGroup !== r.textureGroup ||
        a.lit !== r.lit || a.lightGroup !== (r.lit ? lightGroup : null) ||
        a.blend !== blended(r) || a.vertexBuffers.length !== b.vertexBuffers.length ||
        a.programGroup !== (b.program?.group ?? null) || a.programOffset !== (b.program?.offset ?? null) ||
        a.firstInstance !== (b.firstInstance ?? 0) ||
        a.indirectBuffer !== (b.nodeIndirect?.buffer ?? null) ||
        a.indirectSkip !== (b.nodeIndirect?.skip ?? false) ||
        !samePointRuns(a.indirectOffsets, b.nodeIndirect?.offsets ?? null) ||
        !sameNodeGroups(a.nodeGroups, b.nodeProgram?.groups) ||
        !samePointRuns(a.pointRuns, b.program?.pointRuns ?? null))
      return false;
    for (let slot = 0; slot < a.vertexBuffers.length; slot++)
      if (a.vertexBuffers[slot] !== b.vertexBuffers[slot]) return false;
  }
  return true;
}

/** Bounded MRU schedules, compared structurally without hashing/serializing the
 * frame or allocating identity IDs. At most maxBundles * maxDraws logical draw
 * descriptions are retained. Native command memory is implementation-owned and
 * cannot honestly be charged as an exact byte count. No GPU buffer is owned.
 *
 * clear() drops references before mesh/resource retirement; dispose() also
 * prohibits further execution. A device replacement needs a NEW cache/renderer.
 * Failed recording/execution never inserts an entry or retries the draw directly.
 */
export function createAnimationRenderBundleCache(device, {
  format, colorFormats = format === null ? [] : [format], depthFormat, sampleCount, bindGroup, stride, instancing = false,
  maxBundles = 4, maxDraws = 1024, label = 'f3d-draw-bundle',
}) {
  if (typeof device?.createRenderBundleEncoder !== 'function')
    throw new TypeError('Render bundles require a WebGPU bundle encoder');
  if (!Number.isSafeInteger(maxBundles) || maxBundles < 1 || maxBundles > 64 ||
      !Number.isSafeInteger(maxDraws) || maxDraws < 1 || maxDraws > 65536)
    throw new RangeError('Invalid render bundle cache capacity');
  const descriptor = {label, colorFormats: colorFormats.slice(), sampleCount,
    ...(depthFormat === null ? {} : {depthStencilFormat: depthFormat}),
    depthReadOnly: false, stencilReadOnly: !hasAnimationStencil(depthFormat)};
  let entries = [], disposed = false;
  let builds = 0, reuses = 0, executions = 0, evictions = 0, encodedDrawCalls = 0;
  const parameters = {bindGroup, stride, instancing, lightGroup: null};
  return Object.freeze({
    execute(pass, commands, length, lightGroup, start = 0, operations = null) {
      if (disposed) throw new Error('Render bundle cache is disposed');
      if (!Number.isSafeInteger(length) || length < 0 || length > maxDraws ||
          !Number.isSafeInteger(start) || start < 0 || start + length > maxDraws || start + length > commands.length)
        throw new RangeError('Render bundle draw list exceeds capacity');
      // Empty frames still run attachment load/store operations in the caller.
      // They do not allocate a bundle or disturb the pass's binding state.
      if (length === 0) return 0;
      let index = -1;
      for (let i = entries.length - 1; i >= 0; i--) {
        if (entries[i].start === start && matches(entries[i].commands, commands, length, lightGroup, start)) { index = i; break; }
      }
      if (index >= 0) {
        const entry = entries[index];
        if (operations) operations.backend.addBundle(operations.context, entry.bundle, pass);
        else pass.executeBundles([entry.bundle]);
        entries.splice(index, 1); entries.push(entry);
        reuses++; executions++;
        return entry.drawCalls;
      }
      parameters.lightGroup = lightGroup; parameters.start = start; parameters.operations = operations;
      let bundle, drawCalls;
      try {
        const encoder = operations ? operations.backend.beginBundle(operations.context, device, descriptor)
          : device.createRenderBundleEncoder(descriptor);
        drawCalls = encodeAnimationDraws(encoder, commands, length, parameters);
        bundle = operations ? operations.backend.finishBundle(operations.context, encoder, {label}) : encoder.finish({label});
      } finally { parameters.lightGroup = null; parameters.operations = null; }
      // Capture independent data, not the renderer's reused command objects.
      const recorded = [];
      for (let i = start; i < start + length; i++) recorded.push(snapshot(commands[i], lightGroup));
      if (operations) operations.backend.addBundle(operations.context, bundle, pass);
      else pass.executeBundles([bundle]);
      if (entries.length === maxBundles) { entries.shift(); evictions++; }
      entries.push({commands: recorded, bundle, drawCalls, start});
      builds++; executions++; encodedDrawCalls += drawCalls;
      return drawCalls;
    },
    clear() { entries = []; },
    dispose() { disposed = true; entries = []; },
    get diagnostics() {
      return Object.freeze({builds, reuses, executions, evictions, encodedDrawCalls,
        cachedBundles: entries.length,
        cachedDraws: entries.reduce((sum, entry) => sum + entry.commands.length, 0),
        maxBundles, maxDraws});
    },
  });
}

function samePassState(a, b) {
  const x = a.blendConstant, y = b.blendConstant;
  return (a.stencilReference ?? null) === (b.stencilReference ?? null) &&
    (x == null ? y == null : y != null && x.every((value, index) => value === y[index]));
}
/** Pass-only state is live, never recorded into a bundle. Contiguous runs retain
 * source order and original arena indices. Changing values reuses structurally
 * identical bundles; changing run boundaries creates bounded new schedules.
 */
export function encodeAnimationRenderSpans(pass, commands, length, parameters, cache = null) {
  let calls = 0, activeQuery = null;
  const queryIndices = parameters.queryIndices;
  for (let start = 0; start < length;) {
    const command = commands[start];
    const query = queryIndices?.[start] ?? null;
    if (query !== activeQuery) {
      if (activeQuery !== null) pass.endOcclusionQuery();
      if (query !== null) pass.beginOcclusionQuery(query);
      activeQuery = query;
    }
    let end = start + 1;
    while (end < length && samePassState(command, commands[end]) && (queryIndices?.[end] ?? null) === query) end++;
    if (command.blendConstant != null) pass.setBlendConstant(command.blendConstant);
    if (command.stencilReference != null) pass.setStencilReference(command.stencilReference);
    calls += cache ? cache.execute(pass, commands, end - start, parameters.lightGroup, start, parameters.operations)
      : encodeAnimationDraws(pass, commands, end - start, {...parameters, start});
    start = end;
  }
  if (activeQuery !== null) pass.endOcclusionQuery();
  return calls;
}
