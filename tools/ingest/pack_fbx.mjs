/** Read only FBX texture references. Geometry, animation, embedded images and
 * compressed arrays remain byte-identical for the retained r186 FBXLoader.
 * Binary nodes use absolute end offsets, with 32-bit headers before 7500 and
 * 64-bit headers thereafter. No geometry arrays are decoded or decompressed.
 */
const MAGIC = Buffer.from('Kaydara FBX Binary  \0');
const EMBEDDED = new Set(['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tga', 'webp']);
const FIELDS = new Set(['Filename', 'RelativeFilename', 'Content']);
const decoder = new TextDecoder();
const error = (code, message) => { throw Object.assign(new Error(`${code}: ${message}`), {code}); };

export function fbxAssetUris(bytes, fail = error) {
  if (bytes.length > 64 * 1024 * 1024) fail('FBX_BUDGET', 'FBX resource discovery exceeds 64 MiB');
  const state = {videos: Object.create(null), textures: new Set(), children: new Map()};
  let nodes = 0;
  const count = () => { if (++nodes > 1000000) fail('FBX_BUDGET', 'FBX node count exceeds the resource discovery bound'); };
  const connect = (from, to) => { if (!state.children.has(to)) state.children.set(to, from); };

  if (bytes.subarray(0, MAGIC.length).equals(MAGIC)) {
    if (bytes.length < 27) fail('FBX_HEADER', 'Truncated binary FBX header');
    const version = bytes.readUInt32LE(23), wide = version >= 7500, width = wide ? 8 : 4, headerSize = width * 3 + 1;
    if (version < 6400) fail('FBX_VERSION', 'Binary FBX versions below 6400 are not supported by the source loader');
    const check = (offset, size, end) => {
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || size < 0 || offset < 0 || offset > end - size)
        fail('FBX_NODE', 'Truncated or out-of-bounds FBX metadata');
    };
    const size = offset => {
      const value = wide ? Number(bytes.readBigUInt64LE(offset)) : bytes.readUInt32LE(offset);
      if (!Number.isSafeInteger(value)) fail('FBX_NODE', 'FBX offset or property count exceeds exact integer range');
      return value;
    };
    const string = (offset, length) => {
      const value = bytes.subarray(offset, offset + length), zero = value.indexOf(0);
      return decoder.decode(value.subarray(0, zero < 0 ? value.length : zero));
    };
    function header(offset, parentEnd) {
      check(offset, headerSize, parentEnd);
      const end = size(offset);
      if (end === 0) return null;
      const propertyCount = size(offset + width), nameLength = bytes[offset + width * 3];
      check(offset + headerSize, nameLength, end);
      if (end > parentEnd) fail('FBX_NODE', 'FBX child extends beyond its parent');
      count();
      return {name: string(offset + headerSize, nameLength), end, propertyCount, start: offset + headerSize + nameLength};
    }
    function properties(node) {
      if (node.propertyCount > 4096) fail('FBX_BUDGET', 'FBX metadata property count exceeds 4096');
      let offset = node.start;
      const values = [];
      for (let i = 0; i < node.propertyCount; i++) {
        check(offset, 1, node.end);
        const type = String.fromCharCode(bytes[offset++]);
        let value, length;
        const scalar = {C: 1, Y: 2, I: 4, F: 4, D: 8, L: 8}[type];
        if (scalar) {
          check(offset, scalar, node.end);
          if (type === 'C') value = (bytes[offset] & 1) === 1;
          else if (type === 'Y') value = bytes.readInt16LE(offset);
          else if (type === 'I') value = bytes.readInt32LE(offset);
          else if (type === 'F') value = bytes.readFloatLE(offset);
          else if (type === 'D') value = bytes.readDoubleLE(offset);
          else value = Number(bytes.readBigInt64LE(offset));
          offset += scalar;
        } else if (type === 'S' || type === 'R') {
          check(offset, 4, node.end); length = bytes.readUInt32LE(offset); offset += 4;
          check(offset, length, node.end);
          value = type === 'S' ? string(offset, length) : {rawBytes: length};
          offset += length;
        } else if ('bcdfil'.includes(type)) {
          check(offset, 12, node.end);
          const elements = bytes.readUInt32LE(offset), encoding = bytes.readUInt32LE(offset + 4);
          length = encoding === 0 ? elements * ({b: 1, c: 1, d: 8, f: 4, i: 4, l: 8}[type]) : bytes.readUInt32LE(offset + 8);
          offset += 12; check(offset, length, node.end); offset += length;
        } else fail('FBX_PROPERTY', `Unknown FBX metadata property type ${JSON.stringify(type)}`);
        values.push(value);
      }
      return {values, offset};
    }
    function children(node, visit) {
      let {offset} = properties(node);
      while (offset < node.end) {
        const child = header(offset, node.end);
        if (!child) break;
        visit(child); offset = child.end;
      }
    }
    const top = {start: 27, propertyCount: 0, end: bytes.length};
    children(top, node => {
      if (node.name === 'Objects') {
        state.videos = Object.create(null); state.textures.clear();
        children(node, object => {
          if (!['Video', 'Texture'].includes(object.name)) return;
          const id = properties(object).values[0];
          if (typeof id !== 'number') return;
          if (object.name === 'Texture') { state.textures.add(id); return; }
          if (Object.hasOwn(state.videos, id)) return; // BinaryParser keeps the first duplicate ID.
          const video = state.videos[id] = {};
          children(object, field => {
            if (!FIELDS.has(field.name)) return;
            const {values, offset} = properties(field);
            if (field.propertyCount === 1 && offset === field.end) video[field.name] = values[0];
          });
        });
      } else if (node.name === 'Connections') {
        state.children.clear();
        children(node, connection => {
          if (connection.name !== 'C') return;
          const values = properties(connection).values;
          if (values.length >= 3) connect(values[1], values[2]);
        });
      }
    });
  } else {
    const text = decoder.decode(bytes), version = Number(/FBXVersion: (\d+)/.exec(text)?.[1]);
    if (!(version >= 7000)) fail('FBX_VERSION', 'ASCII FBX requires a version of at least 7000');
    const lines = text.split(/[\r\n]+/), stack = [];
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i], depth = stack.length;
      if (/^[\s\t]*(?:;|$)/.test(line)) continue;
      const begin = line.match(new RegExp(`^\\t{${depth}}(\\w+):(.*){`));
      if (begin) {
        count();
        if (depth >= 128) fail('FBX_BUDGET', 'FBX nesting exceeds 128 nodes');
        const name = begin[1], id = parseInt(begin[2].split(',')[0].trim().replace(/^"/, '').replace(/"$/, ''));
        const node = {name};
        if (depth === 0 && name === 'Objects') { state.videos = Object.create(null); state.textures.clear(); }
        if (depth === 0 && name === 'Connections') state.children.clear();
        if (depth === 1 && stack[0].name === 'Objects' && !Number.isNaN(id)) {
          if (name === 'Video') node.video = state.videos[id] = {}; // TextParser replaces duplicates.
          if (name === 'Texture') state.textures.add(id);
        }
        stack.push(node); continue;
      }
      const property = line.match(new RegExp(`^\\t{${depth}}(\\w+):[\\s\\t\\r\\n](.*)`));
      if (property) {
        const name = property[1], node = stack.at(-1);
        let value = property[2].replace(/^"/, '').replace(/"$/, '').trim();
        if (node?.video && FIELDS.has(name)) {
          if (name === 'Content' && value === ',') value = (lines[i + 1] ?? '').replace(/"/g, '').replace(/,$/, '').trim();
          node.video[name] = value;
        } else if (depth === 1 && node?.name === 'Connections' && name === 'C') {
          const values = value.split(','); connect(parseInt(values[1]), parseInt(values[2]));
        }
      } else if (depth > 0 && new RegExp(`^\\t{${depth - 1}}}`).test(line)) stack.pop();
    }
  }

  const images = new Map(), embedded = new Set();
  for (const [id, video] of Object.entries(state.videos)) {
    const filename = video.RelativeFilename || video.Filename;
    if (typeof filename !== 'string') continue;
    images.set(parseInt(id), filename);
    const content = video.Content;
    if ((typeof content === 'string' ? content !== '' : content?.rawBytes > 0) &&
        EMBEDDED.has(filename.slice(filename.lastIndexOf('.') + 1).toLowerCase())) embedded.add(filename);
  }
  const uris = new Set();
  for (const texture of state.textures) {
    const filename = images.get(state.children.get(texture));
    if (filename === undefined || embedded.has(filename)) continue;
    const uri = filename.split('\\').pop();
    if (uri && !uri.startsWith('blob:') && !uri.startsWith('data:')) uris.add(uri);
  }
  return [...uris];
}
