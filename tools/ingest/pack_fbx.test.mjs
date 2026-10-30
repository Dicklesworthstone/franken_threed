import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {fbxAssetUris} from './pack_fbx.mjs';

const string = value => { const b=Buffer.from(value), n=Buffer.alloc(4);n.writeUInt32LE(b.length);return Buffer.concat([Buffer.from('S'),n,b]); };
const id = value => { const b=Buffer.alloc(9);b[0]=76;b.writeBigInt64LE(BigInt(value),1);return b; };
const raw = value => { const b=Buffer.alloc(5);b[0]=82;b.writeUInt32LE(value.length,1);return Buffer.concat([b,value]); };
const node = (name,properties=[],children=[]) => ({name,properties,children});
const field = (name,value) => node(name,[string(value)]);
function binary(nodes,version=7400) {
  const wide=version>=7500,width=wide?8:4,headerSize=width*3+1;
  function emit(value,start) {
    const name=Buffer.from(value.name),props=Buffer.concat(value.properties),header=Buffer.alloc(headerSize);
    let offset=start+header.length+name.length+props.length;
    const children=value.children.map(child=>{const b=emit(child,offset);offset+=b.length;return b;});
    if(children.length){children.push(Buffer.alloc(headerSize));offset+=headerSize;}
    for(const [index,n] of [offset,value.properties.length,props.length].entries())
      if(wide)header.writeBigUInt64LE(BigInt(n),index*width);else header.writeUInt32LE(n,index*width);
    header[width*3]=name.length;
    return Buffer.concat([header,name,props,...children]);
  }
  const header=Buffer.alloc(27);header.write('Kaydara FBX Binary  \0\x1a\0');header.writeUInt32LE(version,23);
  let offset=27;
  return Buffer.concat([header,...nodes.map(value=>{const b=emit(value,offset);offset+=b.length;return b;}),Buffer.alloc(headerSize+176)]);
}
const video = (number,filename,extra=[]) => node('Video',[id(number),string('Video::image'),string('Clip')],
  [field('Filename',filename),...extra]);
const texture = number => node('Texture',[id(number),string('Texture::map'),string('TextureVideoClip')],
  [field('FileName','wrong-texture-name.png')]);
const connection = (from,to) => node('C',[string('OO'),id(from),id(to)]);

test('ASCII FBX follows connected Video paths, first children, embedded aliases and source duplicate rules',()=>{
  const bytes=Buffer.from(`; FBX 7.4.0 project file
FBXHeaderExtension:  {
\tFBXVersion: 7400
}
Objects:  {
\tVideo: 1, "Video::first", "Clip" {
\t\tFilename: "C:\\old\\paint.png"
\t\tRelativeFilename: "textures/paint%20%23.png?v=2#map"
\t}
\tVideo: 2, "Video::dead", "Clip" {
\t\tFilename: "unused.png"
\t}
\tVideo: 3, "Video::external-alias", "Clip" {
\t\tFilename: "shared.png"
\t}
\tVideo: 4, "Video::embedded-alias", "Clip" {
\t\tFilename: "shared.png"
\t\tContent: ,
\t\t "AQID"
\t}
\tVideo: 5, "Video::unsupported-embedded", "Clip" {
\t\tFilename: "C:\\author\\unknown.ext"
\t\tContent: "AQID"
\t}
\tVideo: 6, "Video::replaced", "Clip" {
\t\tFilename: "stale.png"
\t}
\tVideo: 6, "Video::replacement", "Clip" {
\t\tFilename: "C:\\author\\final.png"
\t}
\tTexture: 10, "Texture::diffuse", "TextureVideoClip" {
\t\tFileName: "not-the-resource.png"
\t}
\tTexture: 11, "Texture::embedded", "TextureVideoClip" {
\t}
\tTexture: 12, "Texture::missing-first-child", "TextureVideoClip" {
\t}
\tTexture: 13, "Texture::unsupported", "TextureVideoClip" {
\t}
\tTexture: 14, "Texture::replacement", "TextureVideoClip" {
\t}
}
Connections:  {
\tC: "OO",1,10
\tC: "OO",2,10
\tC: "OO",3,11
\tC: "OO",99,12
\tC: "OO",2,12
\tC: "OO",5,13
\tC: "OO",6,14
}
`);
  assert.deepEqual(fbxAssetUris(bytes),['textures/paint%20%23.png?v=2#map','unknown.ext','final.png']);
});

for(const version of [7400,7500])test(`binary FBX ${version} retains Video identity and skips embedded/dead resources`,()=>{
  const bytes=binary([node('Objects',[],[
    video(1,'C:\\ignored\\paint.png',[field('RelativeFilename','textures/paint.png')]),
    video(1,'later-duplicate.png'),video(2,'dead.png'),video(3,'shared.png'),
    video(4,'shared.png',[node('Content',[raw(Buffer.from([1,2,3]))])]),
    video(5,'still-external.png',[node('Content',[raw(Buffer.alloc(0))])]),
    texture(10),texture(11),texture(12),texture(13),
    // Resource inspection skips geometry payloads instead of decompressing.
    node('Geometry',[Buffer.from('opaque geometry payload')]),
  ]),node('Connections',[],[connection(1,10),connection(2,10),connection(3,11),
    connection(99,12),connection(2,12),connection(5,13)])],version);
  const original=Buffer.from(bytes);
  assert.deepEqual(fbxAssetUris(bytes),['textures/paint.png','still-external.png']);
  assert.deepEqual(bytes,original);
});

test('binary metadata rejects truncated lengths, parent overflows and imprecise 64-bit offsets',()=>{
  const valid=binary([node('Objects',[],[video(1,'paint.png'),texture(2)]),node('Connections',[],[connection(1,2)])],7500);
  assert.throws(()=>fbxAssetUris(valid.subarray(0,40)),{code:'FBX_NODE'});
  const overflow=Buffer.from(valid);overflow.writeBigUInt64LE(BigInt(valid.length+1),27);
  assert.throws(()=>fbxAssetUris(overflow),{code:'FBX_NODE'});
  const imprecise=Buffer.from(valid);imprecise.writeBigUInt64LE(9007199254740992n,27);
  assert.throws(()=>fbxAssetUris(imprecise),{code:'FBX_NODE'});
  const count=Buffer.from(valid);count.writeBigUInt64LE(4097n,35);
  assert.throws(()=>fbxAssetUris(count),{code:'FBX_BUDGET'});
});

test('real pinned binary and ASCII models expose only their retained-loader external resources',()=>{
  const root=new URL('../../upstream/three.js/examples/models/fbx/',import.meta.url);
  for(const [name,expected] of [['monkey.fbx',['monkey.fbm/UVTexture.png']],
    ['monkey_embedded_texture.fbx',[]],['RotationTest.fbx',[]],['ball_anims_asc_2018.fbx',[]],
    ['archer/ArcherRi01.fbx',['ArcherRi01.png','ArcherRi01_W.PNG']]])
    assert.deepEqual(fbxAssetUris(fs.readFileSync(new URL(name,root))),expected,name);
});
