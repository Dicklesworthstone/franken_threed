/** One WebGL instance upload history per public renderer, shared by its scene
 * bridges and queued pass captures. A new destination must not turn an
 * unversioned CPU edit into an upload. Private pass streams never rejoin this
 * history: they carry the bytes accepted by one render call. */
export function createWebGLInstanceHistory({maxBytes = 128 * 1024 * 1024, maxAttributes = 512} = {}) {
  const histories = new Map(), privateAttributes = new WeakSet(), untrackedAttributes = new WeakSet();
  let bytes = 0, revision = 0;
  const fail = message => { throw Object.assign(new Error(`GEOMETRY_GPU_HISTORY: ${message}`), {code: 'GEOMETRY_GPU_HISTORY'}); };
  const eligible = entry => entry.owner?.isInstancedBufferAttribute === true &&
    [3, 16].includes(entry.owner.itemSize) && entry.source?.elementType === Float32Array &&
    !entry.conversion && !entry.borrowed && (entry.divisor ?? 1) === 1 && !privateAttributes.has(entry.owner) && !untrackedAttributes.has(entry.owner);
  const copy = source => {
    const array = new Float32Array(source.length);
    new Uint8Array(array.buffer).set(new Uint8Array(source.buffer, source.byteOffset, source.byteLength));
    return array;
  };
  function validate(attributes) {
    let additions = 0, count = 0;
    for (const attribute of new Set(attributes)) {
      if (untrackedAttributes.has(attribute)) fail('This instance stream retains an independent live upload history');
      const array = attribute.array, previous = histories.get(attribute);
      if (array == null && previous && Number.isSafeInteger(attribute.version) && attribute.version >= 0 && previous.version >= attribute.version) continue;
      if (!(array instanceof Float32Array) || !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable ||
          !Number.isSafeInteger(attribute.version) || attribute.version < 0) fail('Expected fixed Float32 instance storage and an upload version');
      try { new Uint8Array(array.buffer, 0, 0); } catch { fail('Detached instance storage'); }
      if (previous && previous.array.length !== array.length) fail('Replace an instance attribute when resizing its storage');
      if (!previous) { additions += array.byteLength; count++; }
      if (previous && attribute.version > previous.version) {
        if (!Array.isArray(attribute.updateRanges) || typeof attribute.clearUpdateRanges !== 'function') fail('Expected ordinary instance update ranges');
        for (const range of attribute.updateRanges) if (!Number.isSafeInteger(range.start) || !Number.isSafeInteger(range.count) ||
            range.start < 0 || range.count < 0 || range.start + range.count > array.length) fail('Invalid instance update range');
      }
    }
    if (bytes + additions > maxBytes || histories.size + count > maxAttributes) fail('Shared instance upload history capacity exceeded');
  }
  function upload(attribute, sink, callback, residentRevision = -1) {
    validate([attribute]);
    let previous = histories.get(attribute);
    if (previous && previous.version >= attribute.version) {
      if (residentRevision !== previous.revision) sink(previous.array);
      return previous;
    }
    const source = attribute.array, ranges = attribute.updateRanges;
    const partial = previous && previous.version !== -1 && ranges.length !== 0;
    const array = copy(partial ? previous.array : source);
    if (partial) {
      // r186 merges with +1, including the otherwise untouched element between
      // adjacent ranges. count=0 is WebGL2's upload-through-the-end operation.
      // Keep source range-object aliases: WebGLAttributes mutates the admitted
      // objects while merging, then clears the list before the upload callback.
      const merged = ranges.sort((a, b) => a.start - b.start);
      let last = 0;
      for (let i = 1; i < merged.length; i++) {
        const a = merged[last], b = merged[i];
        if (b.start <= a.start + a.count + 1) a.count = Math.max(a.count, b.start + b.count - a.start);
        else merged[++last] = b;
      }
      merged.length = last + 1;
      for (let i = 0; i <= last; i++) {
        const range = merged[i], end = range.count === 0 ? source.length : range.start + range.count;
        new Uint8Array(array.buffer, range.start * 4, (end - range.start) * 4)
          .set(new Uint8Array(source.buffer, source.byteOffset + range.start * 4, (end - range.start) * 4));
      }
    }
    sink(array);
    if (!previous) bytes += array.byteLength;
    const current = {array, version: previous?.version ?? -1, revision: ++revision};
    histories.set(attribute, current);
    if (partial) attribute.clearUpdateRanges();
    // As in WebGLAttributes, writes survive a throwing callback, but that
    // upload version is acknowledged only after the callback returns.
    callback?.();
    current.version = attribute.version;
    return current;
  }
  return {
    eligible, validate, upload,
    prepareRaw(entries) {
      // Optional capture cannot impose a new aggregate limit on ordinary live
      // scenes. Streams beyond this owner stay on their existing GPU path;
      // later required captures refuse their unavailable history explicitly.
      let available = maxBytes - bytes, slots = maxAttributes - histories.size;
      const selected = [];
      for (const entry of entries) if (eligible(entry)) {
        const attribute = entry.owner;
        if (!histories.has(attribute)) {
          if (entry.source.byteLength > available || slots === 0) { untrackedAttributes.add(attribute); continue; }
          available -= entry.source.byteLength; slots--;
        }
        selected.push(attribute);
      }
      validate(selected);
    },
    capture(attribute) { return upload(attribute, () => {}, () => attribute.onUploadCallback()).array; },
    markPrivate(attribute) { privateAttributes.add(attribute); },
    forget(attribute) { untrackedAttributes.delete(attribute); const old = histories.get(attribute); if (old) { bytes -= old.array.byteLength; histories.delete(attribute); } },
    dispose() { histories.clear(); bytes = 0; },
    get bytes() { return bytes; },
  };
}
