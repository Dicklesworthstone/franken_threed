/** Numerical oracle: the immutable upstream source, with explicit host doubles.
 * No GPU/renderer/public Three.js class conformance is asserted by this harness.
 * The sole source instrumentation exposes the original private Float32 lists on
 * both sides. It neither changes nor reimplements the polygonization algorithm.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

export const MARCHING_CUBES_BLOB = '29a405be3eae30a7e2b1ff04827068921d31d5dc';
export function gitBlob(source) {
  const bytes = Buffer.from(source);
  return createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
}
export function readMarchingCubesOracle() {
  const root = process.env.F3D_THREE_ROOT ?? fileURLToPath(new URL('../../../upstream/three.js/', import.meta.url));
  const source = fs.readFileSync(path.join(root, 'examples/jsm/objects/MarchingCubes.js'), 'utf8');
  assert.equal(gitBlob(source), MARCHING_CUBES_BLOB, 'Require the pinned r186 oracle; never silently use another release');
  return source;
}

// These doubles provide only the cold allocation/publication surface used by
// MarchingCubes. The actual upstream source supplies ALL numerical behavior.
export const THREE_HOST = `
export class EventDispatcher {}
export class BufferAttribute {
 constructor(array,itemSize){this.array=array;this.itemSize=itemSize;this.version=0;}
 setUsage(value){this.usage=value;return this;}
 set needsUpdate(value){if(value===true)this.version++;}
}
export class BufferGeometry extends EventDispatcher {
 constructor(){super();this.attributes={};this.drawRange={start:0,count:Infinity};}
 setAttribute(name,value){this.attributes[name]=value;return this;}
 getAttribute(name){return this.attributes[name];}
 setDrawRange(start,count){this.drawRange.start=start;this.drawRange.count=count;}
}
export class Mesh extends EventDispatcher {constructor(geometry,material){super();this.geometry=geometry;this.material=material;}}
export class Material extends EventDispatcher {constructor(){super();this.flatShading=false;}}
export class Vector3 {constructor(x=0,y=0,z=0){this.x=x;this.y=y;this.z=z;}}
export class Sphere {constructor(center,radius){this.center=center;this.radius=radius;}}
export class Color {constructor(r=1,g=1,b=1){this.r=r;this.g=g;this.b=b;}}
export const DynamicDrawUsage=35048;
`;
export function exposeLists(source) {
  const anchor = '\t\tthis.init( resolution );';
  assert.equal(source.split(anchor).length, 2);
  return source.replace(anchor, `Object.defineProperty(this,'__f3dLists',{value:[vlist,nlist,clist]});\n${anchor}`);
}
export async function loadOracle({source = readMarchingCubesOracle(), host = THREE_HOST, files = {}} = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-marching-oracle-'));
  files = {...files, 'node_modules/three/package.json': '{"type":"module","exports":"./index.mjs"}',
    'node_modules/three/index.mjs': host, 'MarchingCubes.mjs': exposeLists(source)};
  for (const [name, content] of Object.entries(files)) {
    const file = path.join(root, name);
    fs.mkdirSync(path.dirname(file), {recursive:true});
    fs.writeFileSync(file, content, {flag:'wx'});
  }
  const module = await import(pathToFileURL(path.join(root, 'MarchingCubes.mjs')));
  const three = await import(pathToFileURL(path.join(root, 'node_modules/three/index.mjs')));
  return {...module, three, root};
}
export const outputNames = ['field','normal_cache','palette','positionArray','normalArray','uvArray','colorArray'];
export function sameArrays(actual, expected, label = '') {
  assert.equal(actual.length, expected.length, label);
  for (let i=0;i<actual.length;i++) assert.ok(Object.is(actual[i],expected[i]), `${label}[${i}]: ${actual[i]} versus ${expected[i]}`);
}
export function sameEffect(actual, expected) {
  assert.equal(actual.count,expected.count,'vertex count');
  for (const name of outputNames) if (actual[name] || expected[name]) sameArrays(actual[name],expected[name],name);
  for(let i=0;i<3;i++) sameArrays(actual.__f3dLists[i],expected.__f3dLists[i],`edge list ${i}`);
}
export function cubeField(effect, bits) {
  effect.isolation=0;
  effect.field.fill(2.375);
  const q=effect.size2+effect.size+1, y=effect.yd, z=effect.zd;
  const corners=[q,q+1,q+1+y,q+y,q+z,q+1+z,q+1+y+z,q+y+z];
  for(let i=0;i<8;i++) effect.field[corners[i]] = bits & (1<<i) ? -0.25-i*0.137 : 0.75+i*0.139;
  for(let i=0;i<effect.palette.length;i++)effect.palette[i]=((i*13)%71-23)/19;
}
