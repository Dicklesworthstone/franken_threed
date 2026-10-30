/** Native r186 occlusion history. Query execution belongs to the owned WebGPU
 * passes; the public result belongs to the source render-context signature.
 * Only the PREVIOUS submitted query frame is mapped when its successor ends.
 */
export function createThreeOcclusionQueries({getDevice, maxContexts = 1024, maxQueries = 16384,
  maxPending = 64, fail = (code, message) => { throw new Error(`THREE_OCCLUSION_${code}: ${message}`); }} = {}) {
  if (typeof getDevice !== 'function' || [maxContexts, maxQueries, maxPending].some(value => !Number.isSafeInteger(value) || value < 1))
    fail('OPTIONS', 'Expected a device accessor and positive occlusion limits');
  const contexts = new Map(), ownedContexts = new WeakSet(), records = new Set(), pending = new Set();
  let disposed = false, terminal = null, watchedDevice = null;
  const stopped = new AbortController();
  const live = () => { if (terminal) throw terminal; if (disposed) fail('DISPOSED', 'Occlusion owner is disposed'); };
  function destroy(record) {
    if (!records.delete(record)) return;
    for (const part of record.parts) {
      part.querySet?.destroy(); part.resolveBuffer?.destroy(); part.readBuffer?.destroy();
    }
    record.parts.length = 0;
  }
  function stop(error) {
    terminal ??= error;
    stopped.abort(terminal);
    for (const record of records) destroy(record);
    contexts.clear();
  }
  async function wait(work) {
    if (stopped.signal.aborted) throw terminal;
    let abort;
    const cancelled = new Promise((_, reject) => {
      abort = () => reject(terminal); stopped.signal.addEventListener('abort', abort, {once: true});
    });
    try { return await Promise.race([work, cancelled]); }
    finally { stopped.signal.removeEventListener('abort', abort); }
  }
  function device() {
    live();
    const value = getDevice();
    if (!value) fail('DEVICE', 'Initialize the renderer before recording queries');
    if (watchedDevice && value !== watchedDevice) fail('DEVICE', 'Occlusion history cannot migrate between devices');
    if (!watchedDevice) {
      watchedDevice = value;
      value.lost.then(info => stop(new Error(`THREE_OCCLUSION_LOST: ${info?.message || 'WebGPU device lost'}`)), stop).catch(() => {});
    }
    return value;
  }
  function context(target, mrt = null, depth = 0) {
    live();
    // RenderContexts.get deliberately excludes scene/camera/target identity,
    // target dimensions, active cube face and mip level from this key.
    const key = `${target === null ? 'default' : [target.textures.length, target.texture.format,
      target.texture.type, target.samples, target.depthBuffer, target.stencilBuffer].join(':')}-${mrt?.id ?? 'default'}-${depth}`;
    if (!contexts.has(key)) {
      if (contexts.size >= maxContexts) fail('LIMIT', 'Occlusion render contexts exceed their capacity');
      const state = {occluded: undefined, latest: null};
      contexts.set(key, state); ownedContexts.add(state);
    }
    return contexts.get(key);
  }
  function mapPrevious(record, state) {
    const parts = record.parts.slice();
    const run = Promise.all(parts.map(async part => {
      await wait(part.readBuffer.mapAsync(1));
      if (disposed || terminal || !records.has(record)) return [];
      try {
        const values = new BigUint64Array(part.readBuffer.getMappedRange());
        return part.objects.filter((object, index) => object !== undefined && values[index] === 0n);
      } finally { part.readBuffer.unmap(); }
    })).then(results => {
      if (!disposed && !terminal && records.has(record)) state.occluded = new WeakSet(results.flat());
    }).catch(error => { if (!disposed && !terminal) stop(error); }).finally(() => {
      destroy(record); pending.delete(run);
    });
    pending.add(run);
  }
  function begin(state, {requested = false, arrayCamera = false} = {}) {
    live();
    if (!requested) return null;
    if (!state || !ownedContexts.has(state)) fail('CONTEXT', 'Expected an owned source render context');
    if (records.size >= maxPending) fail('LIMIT', 'Too many occlusion readbacks are pending');
    const d = device(), record = {parts: [], count: 0}; records.add(record);
    let completed = false;
    return Object.freeze({
      plan(objects, commands, length) {
        live();
        if (completed || !records.has(record)) fail('STALE', 'Occlusion render already completed');
        if (!Array.isArray(objects) || objects.length !== length) fail('DRAW', 'Occlusion identities must match the draw list');
        const indices = new Array(length).fill(null), queried = [];
        let last = null, active = null;
        for (let index = 0; index < length; index++) {
          const selected = objects[index], command = commands[index];
          const empty = command.nodeProgram && (command.instanceCount === 0 || command.nodeIndirect?.skip ||
            !command.nodeIndirect && command.count === 0);
          // Source bundles execute at finishRender, inside any last active
          // ordinary query. Their own occlusionTest flags do not open queries.
          if (!arrayCamera && !empty && !selected.bundle && selected.object !== last) {
            last = selected.object; active = null;
            if (selected.tested) { active = queried.length; queried.push(selected.object); }
          }
          indices[index] = active;
        }
        if (queried.length > 4096 || record.count + queried.length > maxQueries)
          fail('LIMIT', 'Occlusion queries exceed the render capacity');
        if (record.parts.length >= maxQueries + 1) fail('LIMIT', 'Occlusion pass parts exceed the render capacity');
        record.count += queried.length;
        // The source resolves a projected query even when its draw is empty or
        // ArrayCamera omits encoding. An unused slot preserves the real GPU
        // map boundary; Promise.all([]) would publish an empty set too early.
        if (!queried.length) queried.push(undefined);
        const part = {objects: queried, querySet: null, resolveBuffer: null, readBuffer: null};
        record.parts.push(part); // Publish each allocation before the next can fail.
        part.querySet = d.createQuerySet({label: 'f3d-occlusion', type: 'occlusion', count: queried.length});
        const size = queried.length * 8;
        part.resolveBuffer = d.createBuffer({label: 'f3d-occlusion-resolve', size, usage: 512 | 4});
        part.readBuffer = d.createBuffer({label: 'f3d-occlusion-readback', size, usage: 8 | 1});
        return Object.freeze({querySet: part.querySet, indices,
          resolve(encoder) {
            encoder.resolveQuerySet(part.querySet, 0, queried.length, part.resolveBuffer, 0);
            encoder.copyBufferToBuffer(part.resolveBuffer, 0, part.readBuffer, 0, size);
          }});
      },
      commit() {
        live();
        if (completed) fail('STALE', 'Occlusion render already completed');
        completed = true;
        const previous = state.latest;
        state.latest = record;
        if (previous) mapPrevious(previous, state);
      },
      cancel() { completed = true; destroy(record); },
    });
  }
  return Object.freeze({context, begin,
    async whenIdle() { live(); while (pending.size) await wait(Promise.all([...pending])); live(); },
    dispose() { if (disposed) return; disposed = true; stop(new Error('THREE_OCCLUSION_DISPOSED: Occlusion owner disposed')); },
    get pendingCount() { return records.size; },
  });
}
