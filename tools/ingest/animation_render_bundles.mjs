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
function compatible(a, b) {
  return a.instanceCount == null && b.instanceCount == null &&
    b.record.alphaMode !== 'BLEND' && a.pipeline === b.pipeline &&
    a.first === b.first && a.count === b.count &&
    a.vertexBuffers.length === b.vertexBuffers.length &&
    a.vertexBuffers.every((buffer, i) => buffer === b.vertexBuffers[i]) &&
    a.indexBuffer === b.indexBuffer && a.indexFormat === b.indexFormat &&
    a.record.surfaceBuffer === b.record.surfaceBuffer &&
    a.record.textureGroup === b.record.textureGroup;
}

/** Fully bind every draw/run, also after executeBundles([]), which clears pass
 * pipeline/binding state. No warm-state assumption crosses an encoder boundary.
 * Instancing never sorts or combines BLEND draws. The instance start/dynamic
 * offset remains the original logical draw's position in the current arena.
 */
export function encodeAnimationDraws(encoder, commands, length, {
  bindGroup, lightGroup, stride, instancing,
}) {
  let calls = 0;
  for (let i = 0; i < length;) {
    const command = commands[i], {record, first, count, pipeline} = command;
    const native = command.instanceCount != null;
    let instances = 1;
    if (instancing && !native && record.alphaMode !== 'BLEND') {
      while (i + instances < length && compatible(command, commands[i + instances])) instances++;
    }
    encoder.setPipeline(pipeline);
    encoder.setBindGroup(0, native ? command.instanceBindGroup : bindGroup,
      instancing && !native ? [] : [i * stride]);
    for (let slot = 0; slot < command.vertexBuffers.length; slot++)
      encoder.setVertexBuffer(slot, command.vertexBuffers[slot]);
    if (record.surfaceBuffer) encoder.setVertexBuffer(1, record.surfaceBuffer);
    if (record.textureGroup) encoder.setBindGroup(1, record.textureGroup);
    if (record.lit) encoder.setBindGroup(record.textureGroup ? 2 : 1, lightGroup);
    if (command.indexBuffer) {
      encoder.setIndexBuffer(command.indexBuffer, command.indexFormat);
      encoder.drawIndexed(count, native ? command.instanceCount : instances, first, 0, instancing && !native ? i : 0);
    } else encoder.draw(count, native ? command.instanceCount : instances, first, instancing && !native ? i : 0);
    calls++;
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
    blend: command.record.alphaMode === 'BLEND'};
}
function matches(recorded, commands, length, lightGroup) {
  if (recorded.length !== length) return false;
  for (let i = 0; i < length; i++) {
    const a = recorded[i], b = commands[i], r = b.record;
    if (a.pipeline !== b.pipeline || a.first !== b.first || a.count !== b.count ||
        a.instanceCount !== b.instanceCount || a.instanceBindGroup !== b.instanceBindGroup ||
        a.indexBuffer !== b.indexBuffer || a.indexFormat !== b.indexFormat ||
        a.surfaceBuffer !== r.surfaceBuffer || a.textureGroup !== r.textureGroup ||
        a.lit !== r.lit || a.lightGroup !== (r.lit ? lightGroup : null) ||
        a.blend !== (r.alphaMode === 'BLEND') || a.vertexBuffers.length !== b.vertexBuffers.length)
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
  format, depthFormat, sampleCount, bindGroup, stride, instancing = false,
  maxBundles = 4, maxDraws = 1024, label = 'f3d-draw-bundle',
}) {
  if (typeof device?.createRenderBundleEncoder !== 'function')
    throw new TypeError('Render bundles require a WebGPU bundle encoder');
  if (!Number.isSafeInteger(maxBundles) || maxBundles < 1 || maxBundles > 64 ||
      !Number.isSafeInteger(maxDraws) || maxDraws < 1 || maxDraws > 65536)
    throw new RangeError('Invalid render bundle cache capacity');
  const descriptor = {label, colorFormats: format === null ? [] : [format], sampleCount,
    ...(depthFormat === null ? {} : {depthStencilFormat: depthFormat}),
    depthReadOnly: false, stencilReadOnly: true};
  let entries = [], disposed = false;
  let builds = 0, reuses = 0, executions = 0, evictions = 0, encodedDrawCalls = 0;
  const parameters = {bindGroup, stride, instancing, lightGroup: null};
  return Object.freeze({
    execute(pass, commands, length, lightGroup) {
      if (disposed) throw new Error('Render bundle cache is disposed');
      if (!Number.isSafeInteger(length) || length < 0 || length > maxDraws || length > commands.length)
        throw new RangeError('Render bundle draw list exceeds capacity');
      // Empty frames still run attachment load/store operations in the caller.
      // They do not allocate a bundle or disturb the pass's binding state.
      if (length === 0) return 0;
      let index = -1;
      for (let i = entries.length - 1; i >= 0; i--) {
        if (matches(entries[i].commands, commands, length, lightGroup)) { index = i; break; }
      }
      if (index >= 0) {
        const entry = entries[index];
        pass.executeBundles([entry.bundle]);
        entries.splice(index, 1); entries.push(entry);
        reuses++; executions++;
        return entry.drawCalls;
      }
      parameters.lightGroup = lightGroup;
      let bundle, drawCalls;
      try {
        const encoder = device.createRenderBundleEncoder(descriptor);
        drawCalls = encodeAnimationDraws(encoder, commands, length, parameters);
        bundle = encoder.finish({label});
      } finally { parameters.lightGroup = null; }
      // Capture independent data, not the renderer's reused command objects.
      const recorded = [];
      for (let i = 0; i < length; i++) recorded.push(snapshot(commands[i], lightGroup));
      pass.executeBundles([bundle]);
      if (entries.length === maxBundles) { entries.shift(); evictions++; }
      entries.push({commands: recorded, bundle, drawCalls});
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
