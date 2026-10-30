// Small authored FBX resource fixtures. Both encodings are decoded by the pinned
// FBXLoader in the packed execution tests; the candidate never emits model bytes.
import { deflateSync } from 'node:zlib';

export const fbxPixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==', 'base64');
const field = (name, value) => ({name, properties:[value]});
const node = (name, properties = [], children = []) => ({name, properties, children});

function videoNodes(records) {
  return records.map((record, index) => node('Video', [record.id ?? index + 10, 'Video::image', 'Clip'], [
    ...(record.filename !== undefined ? [field('Filename', record.filename)] : []),
    ...(record.relative !== undefined ? [field('RelativeFilename', record.relative)] : []),
    ...(record.content !== undefined ? [field('Content', record.content)] : []),
  ]));
}

export function fbxResources(records, {binary = false, version = 7400, triangle = false, numberType = 'L', connectVideos = true} = {}) {
  const objects = videoNodes(records), connections = [];
  if (triangle) {
    objects.push(node('Geometry', [100, 'Geometry::Triangle', 'Mesh'], [
      field('Vertices', new Float64Array([0,0,0,1,0,0,0,1,0])),
      field('PolygonVertexIndex', new Int32Array([0,1,-3])),
      node('LayerElementUV', [0], [field('MappingInformationType','ByPolygonVertex'),
        field('ReferenceInformationType','Direct'), field('UV',new Float64Array([0,0,1,0,0,1]))]),
    ]), node('Model',[101,'Model::Triangle','Mesh']),
    node('Material',[102,'Material::Paint',''],[field('ShadingModel','phong')]),
    node('Texture',[103,'Texture::Paint',''],[field('FileName','paint.png')]));
    for (const properties of [['OO',100,101],['OO',102,101],['OO',101,0],
      ['OP',103,102,'DiffuseColor'],['OO',records[0].id ?? 10,103]]) connections.push({name:'C',properties});
  } else if (connectVideos) {
    // Video metadata alone never requests an image. Give metadata-only cases
    // real Texture consumers, including the source's parseInt image-ID lookup.
    for (const [index,record] of records.entries()) {
      const textureId=1000+index;
      objects.push(node('Texture',[textureId,'Texture::resource','TextureVideoClip'],[field('FileName','resource.png')]));
      connections.push({name:'C',properties:['OO',parseInt(String(record.id??index+10)),textureId]});
    }
  }
  const roots = [node('FBXHeaderExtension',[],[field('FBXVersion',version)]),
    node('Objects',[],objects), node('Connections',[],connections)];
  if (!binary) {
    const quote = value => '"' + String(value) + '"';
    function text(n, depth = 0) {
      const tabs = '\t'.repeat(depth), value = n.properties[0];
      if (!n.children && ArrayBuffer.isView(value) && !Buffer.isBuffer(value))
        return `${tabs}${n.name}: *${value.length} {\n${tabs}\ta: ${[...value].join(',')}\n${tabs}}\n`;
      if (!n.children && Buffer.isBuffer(value))
        return `${tabs}${n.name}: ,\n${tabs}\t"${value.toString('base64')}"\n`;
      const properties = n.properties.map(value => typeof value === 'string' ? quote(value) : value).join(', ');
      if (!n.children) return `${tabs}${n.name}: ${properties}\n`;
      return `${tabs}${n.name}: ${properties} {\n${n.children.map(child=>text(child,depth+1)).join('')}${tabs}}\n`;
    }
    return Buffer.from('; FBX 7.4.0 project file\n' + roots.map(n=>text(n)).join(''));
  }
  const width = version >= 7500 ? 8 : 4, headerSize = width * 3 + 1;
  const u32 = value => {const bytes=Buffer.alloc(4); bytes.writeUInt32LE(value); return bytes;};
  function property(value) {
    if (typeof value === 'string') {const bytes=Buffer.from(value);return Buffer.concat([Buffer.from('S'),u32(bytes.length),bytes]);}
    if (Buffer.isBuffer(value)) return Buffer.concat([Buffer.from('R'),u32(value.length),value]);
    if (ArrayBuffer.isView(value)) {
      const bytes = Buffer.from(value.buffer,value.byteOffset,value.byteLength), compressed = deflateSync(bytes);
      return Buffer.concat([Buffer.from(value instanceof Float64Array ? 'd' : 'i'),u32(value.length),u32(1),u32(compressed.length),compressed]);
    }
    const [size, write] = {L:[8,'writeBigInt64LE'], I:[4,'writeInt32LE'],
      D:[8,'writeDoubleLE'], F:[4,'writeFloatLE'], Y:[2,'writeInt16LE']}[numberType];
    const bytes = Buffer.alloc(1+size);bytes[0]=numberType.charCodeAt(0);
    bytes[write](numberType==='L'?BigInt(value):value,1);return bytes;
  }
  function encode(n, offset) {
    const header=Buffer.alloc(headerSize), name=Buffer.from(n.name);
    const properties=Buffer.concat(n.properties.map((value,index)=>property(index===1&&typeof value==='string'
      ? value.replace(/^(\w+)::(.*)$/,'$2\0\x01$1') : value)));
    let cursor=offset+headerSize+name.length+properties.length;
    const children=[];
    if (n.children) {
      for (const child of n.children) {const bytes=encode(child,cursor);children.push(bytes);cursor+=bytes.length;}
      children.push(Buffer.alloc(headerSize));cursor+=headerSize;
    }
    const integer=(value,at)=>width===8?header.writeBigUInt64LE(BigInt(value),at):header.writeUInt32LE(value,at);
    integer(cursor,0);integer(n.properties.length,width);integer(properties.length,width*2);header[headerSize-1]=name.length;
    return Buffer.concat([header,name,properties,...children]);
  }
  const parts=[Buffer.from('Kaydara FBX Binary  \0\x1a\0'),u32(version)];
  let offset=27;
  for (const root of roots) {const bytes=encode(root,offset);parts.push(bytes);offset+=bytes.length;}
  return Buffer.concat([...parts,Buffer.alloc(headerSize+176)]);
}
