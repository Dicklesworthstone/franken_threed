/** GL's last-provoking-vertex convention on first-provoking-vertex WebGPU.
 * Build immutable index buffers from synchronous resident-history readers.
 * Line strips become ordered line-list pairs. Lists restart their grouping at
 * each fixed-index marker and at the selected draw's first index. Vertex IDs,
 * triangle winding, primitive order and per-instance streams stay unchanged.
 */
export function createProgramIndexLowering(device, {label, fail, maxBytes = 64 * 1024 * 1024, maxEntries = 1024}) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 4 || !Number.isSafeInteger(maxEntries) || maxEntries < 1)
    fail('ANIMATION_RENDER_LIMIT', 'Invalid program index cache capacity');
  const maximum = Math.min(maxBytes, device.limits.maxBufferSize ?? maxBytes);
  const entries = new Set(), resources = new Set(), retired = new Set();
  let allocatedBytes = 0, frame = 0, disposed = false;
  function destroy(entry) {
    if (!resources.delete(entry)) return;
    allocatedBytes -= entry.size;
    entry.buffer.destroy();
  }
  function drain() {
    // Only older-frame entries may be fenced: the caller submits all prior
    // frames before begin(). Never destroy a buffer already staged this frame.
    const ready = [...retired].filter(e => e.frame < frame);
    if (!ready.length) return;
    for (const e of ready) retired.delete(e);
    const release = () => { for (const e of ready) destroy(e); };
    try { Promise.resolve(device.queue.onSubmittedWorkDone()).then(release, release).catch(() => {}); }
    catch { release(); }
  }
  function evict(entry) { entries.delete(entry); retired.add(entry); }
  function visit(geometry, topology, first, end, emit) {
    const read = geometry.indexBuffer ? geometry.readIndex : i => i;
    if (typeof read !== 'function') fail('ANIMATION_RENDER_GEOMETRY', 'Topology lowering needs resident index history');
    const restart = geometry.indexBuffer ? geometry.indexFormat === 'uint16' ? 65535 : 4294967295 : null;
    let n = 0, a = 0, b = 0, count = 0;
    const put = value => { if (emit) emit(count, value); count++; };
    for (let i = first; i < end; i++) {
      const v = read(i);
      if (v === restart) { n = 0; continue; }
      if (topology === 'line-strip') {
        if (n) { put(v); put(a); }
        a = v; n = 1;
      } else if (n === 0) { a = v; n = 1; }
      else if (topology === 'lines') { put(v); put(a); n = 0; }
      else if (n === 1) { b = v; n = 2; }
      else { put(v); put(a); put(b); n = 0; }
      if (count * 4 > maximum) fail('ANIMATION_RENDER_LIMIT', 'Lowered indices exceed the index buffer budget');
    }
    return count;
  }
  return Object.freeze({
    begin() { frame++; drain(); },
    prepare(gpu, geometry, topology, first, end) {
      if (disposed) fail('ANIMATION_RENDER_DISPOSED', 'Program index cache is disposed');
      const extent = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
      if (!['triangles', 'lines', 'line-strip'].includes(topology) ||
          !Number.isSafeInteger(first) || !Number.isSafeInteger(end) || first < 0 || end < first || end > extent)
        fail('ANIMATION_RENDER_GEOMETRY', 'Invalid lowered draw topology or range');
      const revision = geometry.indexRevision ?? 0;
      for (const e of entries) {
        if (e.gpu === gpu && e.topology === topology && e.first === first && e.end === end &&
            e.indexBuffer === geometry.indexBuffer && e.revision === revision && e.generation === geometry.generation) {
          e.frame = frame; entries.delete(e); entries.add(e); return e.result;
        }
      }
      // For implicit sequential vertices the exact size is known without a scan.
      if (!geometry.indexBuffer) {
        const n = end - first, count = topology === 'line-strip' ? Math.max(0, n - 1) * 2 : Math.floor(n / (topology === 'lines' ? 2 : 3)) * (topology === 'lines' ? 2 : 3);
        if (count * 4 > maximum) fail('ANIMATION_RENDER_LIMIT', 'Lowered indices exceed the index buffer budget');
      }
      const count = visit(geometry, topology, first, end);
      if (count === 0) return {buffer: null, count: 0};
      // Superseded generations and LRU entries are evicted without touching
      // buffers used by other draws in this frame. Retired bytes remain charged.
      for (const e of entries) if (e.gpu === gpu && (e.indexBuffer !== geometry.indexBuffer || e.revision !== revision || e.generation !== geometry.generation)) evict(e);
      while (entries.size >= maxEntries) {
        const candidate = [...entries].find(e => e.frame < frame);
        if (!candidate) fail('ANIMATION_RENDER_LIMIT', 'Program index cache entry budget exhausted for this frame');
        evict(candidate);
      }
      drain();
      const size = count * 4;
      if (allocatedBytes + size > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Program index budget includes unfinished retired buffers');
      const buffer = device.createBuffer({label: `${label}/lowered-indices`, size, usage: 16, mappedAtCreation: true});
      try {
        const out = new Uint32Array(buffer.getMappedRange());
        visit(geometry, topology, first, end, (i, value) => { out[i] = value; });
        buffer.unmap();
      } catch (error) { buffer.destroy(); throw error; }
      const result = Object.freeze({buffer, count});
      const entry = {gpu, topology, first, end, indexBuffer: geometry.indexBuffer, revision,
        generation: geometry.generation, frame, size, buffer, result};
      entries.add(entry); resources.add(entry); allocatedBytes += size;
      return result;
    },
    dispose() {
      disposed = true; entries.clear(); retired.clear();
      for (const e of resources) destroy(e);
    },
    get allocatedBytes() { return allocatedBytes; },
  });
}
