/** Bounded 3DS texture metadata discovery for the pinned r186 TDSLoader.
 * Only readFile -> readMeshData -> readMaterialEntry -> readMap is followed.
 * The original bytes, geometry, materials and parse/error behavior remain in
 * the retained loader. Unknown chunks are opaque, not recursively searched.
 */
const error = (code, message) => { throw Object.assign(new Error(`${code}: ${message}`), {code}); };
const MAX_BYTES = 64 * 1024 * 1024;
const ROOT_IDS = new Set([0x4d4d, 0x3daa, 0xc23d]);
const MAP_IDS = new Set([0xa200, 0xa230, 0xa210, 0xa204]);

export function tdsAssetUris(bytes, fail = error) {
  if (bytes.length > MAX_BYTES) fail('TDS_BUDGET', '3DS resource discovery exceeds 64 MiB');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const data = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const uris = new Set();
  let chunks = 0, stringBytes = 0;
  const chunkAt = offset => {
    // The source Chunk.readChunk catches truncated headers and ends that list.
    if (offset + 6 > data.length) return null;
    if (++chunks > 1000000) fail('TDS_BUDGET', '3DS resource discovery exceeds the chunk work bound');
    const size = view.getUint32(offset + 2, true);
    return {id: view.getUint16(offset, true), offset, size, end: offset + size};
  };
  const children = (parent, visit) => {
    let offset = parent.offset + 6;
    while (offset < parent.end) {
      const child = chunkAt(offset);
      if (!child) break;
      if (child.size === 0) fail('TDS_BUDGET', '3DS contains a non-advancing chunk');
      offset += child.size;
      visit(child);
    }
  };
  const root = chunkAt(0);
  if (!root || !ROOT_IDS.has(root.id)) return [];
  children(root, meshData => {
    if (meshData.id !== 0x3d3d) return;
    children(meshData, material => {
      if (material.id !== 0xafff) return;
      children(material, map => {
        if (!MAP_IDS.has(map.id)) return;
        children(map, name => {
          if (name.id !== 0xa300) return;
          const start = name.offset + 6;
          let end = start;
          // Chunk.readString uses byte-valued character codes, not UTF-8, and
          // reads until NUL in the entire buffer, even across the chunk end.
          // Keep that tolerance and every duplicate map-name request. A missing
          // terminator leaves the original runtime exception source-owned.
          while (end < data.length && data[end] !== 0) {
            if (++stringBytes > MAX_BYTES) fail('TDS_BUDGET', '3DS resource discovery exceeds the string work bound');
            end++;
          }
          if (end < data.length) uris.add(data.toString('latin1', start, end));
        });
      });
    });
  });
  return [...uris];
}
