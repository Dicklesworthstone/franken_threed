/** Small authored USDC fixtures, with literal LZ4 blocks and delta-coded tables.
 * Both r186 parsers consume these bytes; the packer does not use this writer.
 */
const u32 = value => { const b = Buffer.alloc(4); b.writeUInt32LE(value >>> 0); return b; };
const u64 = value => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b; };
const rep = (type, value, inline = false) => Buffer.concat([u32(value), u32((type << 16) | (inline ? 0x40000000 : 0))]);
const lz4 = bytes => {
  const lengths = [];
  if (bytes.length >= 15) {
    let left = bytes.length - 15;
    while (left >= 255) { lengths.push(255); left -= 255; }
    lengths.push(left);
  }
  return Buffer.concat([Buffer.from([0, Math.min(bytes.length, 15) << 4, ...lengths]), bytes]);
};
const integers = values => {
  const codes = Buffer.alloc(Math.ceil(values.length / 4), 255), deltas = Buffer.alloc(values.length * 4);
  values.forEach((value, i) => deltas.writeInt32LE((value - (values[i - 1] ?? 0)) | 0, i * 4));
  return lz4(Buffer.concat([u32(0), codes, deltas]));
};
const compressedArray = bytes => Buffer.concat([u64(bytes.length), bytes]);

export function usdcBytes(specs, {compressed = true} = {}) {
  const chunks = [Buffer.alloc(88)], sections = [], tokens = [""], strings = [];
  let offset = 88;
  const token = text => { let index = tokens.indexOf(text); if (index < 0) { index = tokens.length; tokens.push(text); } return index; };
  const append = bytes => { const at = offset; chunks.push(bytes); offset += bytes.length; return at; };
  const section = (name, bytes) => sections.push({name, start: append(bytes), size: bytes.length});
  const nodes = new Map([["/", {path: "/", children: []}]]);
  const node = key => {
    if (nodes.has(key)) return nodes.get(key);
    const dot = key.lastIndexOf("."), slash = key.lastIndexOf("/");
    const property = dot > slash, split = property ? dot : slash;
    const parent = node(key.slice(0, split) || "/");
    const result = {path: key, name: key.slice(split + 1), property, children: []};
    nodes.set(key, result); parent.children.push(result); return result;
  };
  Object.keys(specs).forEach(node);
  const paths = [];
  const visit = (current, sibling = false) => {
    current.index = paths.length; current.sibling = sibling; paths.push(current);
    current.token = current.name ? token(current.name) * (current.property ? -1 : 1) : 0;
    current.children.forEach((child, i) => visit(child, i + 1 < current.children.length));
    current.jump = current.children.length ? sibling ? paths.length - current.index : -1 : sibling ? 0 : -2;
  };
  visit(nodes.get("/"));
  const value = ({type, value: data}) => {
    if (type === "token" || type === "asset") return rep(type === "token" ? 11 : 12, token(data), true);
    if (type === "string") { const index = strings.length; strings.push(token(data)); return rep(10, index, true); }
    if (type === "float") { const b = Buffer.alloc(4); b.writeFloatLE(data); return rep(9, b.readUInt32LE(), true); }
    if (type === "paths") return rep(34, append(Buffer.concat([Buffer.from([32]), u64(data.length), ...data.map(key => u32(nodes.get(key).index))])));
    if (type === "times") {
      const doubles = Buffer.alloc(data.times.length * 8);
      data.times.forEach((time, i) => doubles.writeDoubleLE(time, i * 8));
      const times = rep(48, append(Buffer.concat([u64(data.times.length), doubles])));
      const values = data.values.map(value);
      return rep(46, append(Buffer.concat([u64(8), times, u64(8), u64(values.length), ...values])));
    }
    throw new Error("Unsupported USDC fixture value: " + type);
  };
  const fields = [], fieldSets = [], records = [];
  for (const [name, spec] of Object.entries(specs)) {
    records.push([nodes.get(name).index, fieldSets.length, spec.specType]);
    for (const [key, data] of Object.entries(spec.fields)) {
      const index = token(key), encoded = value(data);
      fieldSets.push(fields.length); fields.push({index, encoded});
    }
    fieldSets.push(-1);
  }
  const tokenBytes = Buffer.from(tokens.join("\0") + "\0");
  section("TOKENS", Buffer.concat([u64(tokens.length), u64(tokenBytes.length),
    ...(compressed ? [compressedArray(lz4(tokenBytes))] : [tokenBytes])]));
  section("STRINGS", Buffer.concat([u64(strings.length), ...strings.map(u32)]));
  section("FIELDS", compressed
    ? Buffer.concat([u64(fields.length), compressedArray(integers(fields.map(field => field.index))), compressedArray(lz4(Buffer.concat(fields.map(field => field.encoded))))])
    : Buffer.concat(fields.map(field => Buffer.concat([u32(field.index), field.encoded]))));
  section("FIELDSETS", compressed
    ? Buffer.concat([u64(fieldSets.length), compressedArray(integers(fieldSets))]) : Buffer.concat(fieldSets.map(u32)));
  if (compressed) {
    section("PATHS", Buffer.concat([u64(paths.length), u64(paths.length),
      ...[paths.map(p => p.index), paths.map(p => p.token), paths.map(p => p.jump)].map(values => compressedArray(integers(values)))]));
  } else {
    let at = offset + 8;
    const offsets = paths.map(p => { const value = at; at += p.children.length && p.sibling ? 17 : 9; return value; });
    section("PATHS", Buffer.concat([u64(paths.length), ...paths.map(p => Buffer.concat([
      u32(p.index), u32(Math.abs(p.token)), Buffer.from([(p.children.length ? 1 : 0) | (p.sibling ? 2 : 0) | (p.property ? 4 : 0)]),
      ...(p.children.length && p.sibling ? [u64(offsets[p.index + p.jump])] : []),
    ]))]));
  }
  section("SPECS", compressed
    ? Buffer.concat([u64(records.length), ...[0, 1, 2].map(i => compressedArray(integers(records.map(r => r[i]))))])
    : Buffer.concat(records.flatMap(record => record.map(u32))));
  const toc = offset;
  append(Buffer.concat([u64(sections.length), ...sections.map(s => {
    const name = Buffer.alloc(16); name.write(s.name);
    return Buffer.concat([name, u64(s.start), u64(s.size)]);
  })]));
  const result = Buffer.concat(chunks);
  result.write("PXR-USDC"); result[9] = compressed ? 8 : 3; result[10] = 0;
  result.writeBigUInt64LE(BigInt(toc), 16);
  return result;
}

export function usdcImageSpecs(image = "./paint%20%23.png?edition=2#surface") {
  const token = value => ({type: "token", value}), asset = value => ({type: "asset", value});
  const prim = typeName => ({specType: 6, fields: {typeName: token(typeName)}});
  const attribute = value => ({specType: 1, fields: {default: value}});
  const connection = target => ({specType: 1, fields: {connectionPaths: {type: "paths", value: [target]}}});
  return {
    "/": {specType: 7, fields: {upAxis: token("Y"), metersPerUnit: {type: "float", value: 1}}},
    "/Box": prim("Cube"),
    "/Box.size": attribute({type: "float", value: 3}),
    "/Box.material:binding": {specType: 8, fields: {targetPaths: {type: "paths", value: ["/Materials/Paint"]}}},
    "/Materials": prim("Scope"),
    "/Materials/Paint": prim("Material"),
    "/Materials/Paint.inputs:image": attribute(asset(image)),
    "/Materials/Paint.outputs:surface": connection("/Materials/Paint/Surface.outputs:surface"),
    "/Materials/Paint/Surface": prim("Shader"),
    "/Materials/Paint/Surface.info:id": attribute(token("UsdPreviewSurface")),
    "/Materials/Paint/Surface.inputs:diffuseColor": connection("/Materials/Paint/Image.outputs:rgb"),
    "/Materials/Paint/Surface.inputs:roughness": attribute({type: "float", value: 0.25}),
    "/Materials/Paint/Surface.outputs:surface": {specType: 1, fields: {}},
    "/Materials/Paint/Image": prim("Shader"),
    "/Materials/Paint/Image.info:id": attribute(token("UsdUVTexture")),
    "/Materials/Paint/Image.inputs:file": connection("/Materials/Paint.inputs:image"),
    "/Materials/Paint/Image.inputs:wrapS": attribute(token("clamp")),
    "/Materials/Paint/Image.inputs:wrapT": attribute(token("mirror")),
    "/Materials/Paint/Image.outputs:rgb": {specType: 1, fields: {}},
  };
}
