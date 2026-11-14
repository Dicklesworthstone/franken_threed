import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {registerHooks} from 'node:module';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {gzipSync} from 'node:zlib';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const root = new URL('../../upstream/three.js/', import.meta.url).href;
const core = root + 'build/three.core.js';
const loader = name => root + `examples/jsm/loaders/${name}.js`;
const dataUrl = bytes => 'data:application/octet-stream;base64,' + Buffer.from(bytes).toString('base64');

function plyBytes(x = 1) {
  const names = ['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3',
    'f_dc_0','f_dc_1','f_dc_2','opacity',...Array.from({length:9},(_,i)=>'f_rest_'+i)];
  const values = [x,2,3,Math.log(2),Math.log(3),Math.log(4),1,0,0,0,0,0,0,0,
    ...Array.from({length:9},(_,i)=>i/128)];
  return Buffer.from(['ply','format ascii 1.0','element vertex 1',
    ...names.map(name=>'property float '+name),'end_header',values.join(' '),''].join('\n'));
}

function splatBytes() {
  const bytes = Buffer.alloc(32);
  [1.5,-2,.25,2,3,4].forEach((value,i)=>bytes.writeFloatLE(value,i*4));
  bytes.set([10,20,30,40,255,128,128,128],24);
  return bytes;
}

function ksplatBytes() {
  // KSPLAT v0.1, one uncompressed section with one splat and first-band SH.
  const bytes = Buffer.alloc(4096+1024+44+9*4);
  bytes[1]=1;
  for (const offset of [4,8,12,16,4096,4100]) bytes.writeUInt32LE(1,offset);
  bytes.writeUInt16LE(1,4096+40);
  [1,2,3,2,3,4,1,0,0,0].forEach((value,i)=>bytes.writeFloatLE(value,5120+i*4));
  bytes.set([10,20,30,40],5120+40);
  for (let i=0;i<9;i++) bytes.writeFloatLE((i+1)/128,5120+44+i*4);
  return bytes;
}

function spzBytes(version) {
  // One fixed-point splat. v2 gzip and v4 Zstd retain distinct source decode
  // paths, including v4's asynchronous, embedded-Wasm decoder preparation.
  const positions=Buffer.alloc(9);
  [24,-32,4].forEach((value,i)=>positions.writeIntLE(value,i*3,3));
  const streams=[positions,Buffer.from([64]),Buffer.from([128,128,128]),Buffer.from([160,160,160]),
    version===4?Buffer.from([0,0,0,192]):Buffer.from([128,128,128]),Buffer.from([129,130,131,132,133,134,135,136,137])];
  const header=Buffer.alloc(version===4?32:16);
  header.writeUInt32LE(0x5053474e,0);header.writeUInt32LE(version,4);header.writeUInt32LE(1,8);
  header[12]=1;header[13]=4;
  if(version===2)return gzipSync(Buffer.concat([header,...streams]));
  // Each Zstd frame has a single raw, final block; decoding still runs the
  // exact upstream Zstd Wasm implementation, not a fixture replacement.
  const compressed=streams.map(bytes=>{
    const frame=Buffer.alloc(9+bytes.length);frame.set([0x28,0xb5,0x2f,0xfd,0x20,bytes.length]);
    frame.writeUIntLE((bytes.length<<3)|1,6,3);bytes.copy(frame,9);return frame;
  });
  header[15]=streams.length;header.writeUInt32LE(header.length,16);
  const table=Buffer.alloc(streams.length*16);
  compressed.forEach((bytes,i)=>{table.writeBigUInt64LE(BigInt(bytes.length),i*16);
    table.writeBigUInt64LE(BigInt(streams[i].length),i*16+8);});
  return Buffer.concat([header,table,...compressed]);
}

// Only document URL resolution is supplied by this Node host. Native fetch,
// streams, module evaluation, Wasm and the retained source decoders execute.
async function execute(url,imports,resources) {
  const saved=Object.fromEntries(['document','Request','ProgressEvent','fetch'].map(key=>[key,globalThis[key]]));
  const requests=[],nativeFetch=saved.fetch;
  const hooks=registerHooks({resolve(specifier,context,next){return next(imports[specifier]??(specifier==='three'?core:specifier),context);}});
  globalThis.document={URL:'https://splats.example/relocated/index.html',baseURI:'https://splats.example/relocated/'};
  globalThis.Request=class extends saved.Request {constructor(input,options){super(typeof input==='string'?new URL(input,document.baseURI):input,options);}};
  globalThis.ProgressEvent=class extends Event {constructor(type,details){super(type);Object.assign(this,details);}};
  globalThis.fetch=async(input,options)=>{
    const uri=input instanceof Request?input.url:String(input);requests.push(uri);
    if(uri.startsWith('data:'))return nativeFetch(input,options);
    const name=decodeURIComponent(new URL(uri).pathname).slice('/relocated/'.length);
    const bytes=typeof resources==='string'?fs.readFileSync(path.join(resources,name)):resources?.[name];
    if(!bytes)throw new Error('Unexpected external resource: '+uri);
    return nativeFetch(dataUrl(bytes),options);
  };
  let timer;
  try {
    const result=await Promise.race([(async()=>{const app=await import(url);return app.run();})(),
      new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Splat loader execution timed out')),15000);})]);
    return {result,requests};
  } finally {clearTimeout(timer);hooks.deregister();for(const [key,value] of Object.entries(saved)){
    if(value===undefined)delete globalThis[key];else globalThis[key]=value;
  }}
}

test('retained Gaussian splat formats decode from relocated and offline packages with native errors and identities',async()=>{
  const resources={'models/cloud #.ply':plyBytes(),'alternate/cloud.ply':plyBytes(7),
    'models/cloud.splat':splatBytes(),'models/cloud.ksplat':ksplatBytes(),
    'models/cloud.v2.spz':spzBytes(2),'models/cloud.v4.spz':spzBytes(4),
    'models/bad.ply':Buffer.from('not a ply'),'models/bad.splat':Buffer.from([1]),
    'models/bad.ksplat':Buffer.from([2]),'models/bad.spz':gzipSync(Buffer.from([3]))};
  const source=`import {BufferGeometry} from ${JSON.stringify(core)};
    import {GaussianSplatPLYLoader as PLY} from ${JSON.stringify(loader('GaussianSplatPLYLoader'))};
    import {SPLATLoader} from ${JSON.stringify(loader('SPLATLoader'))};
    import {SPZLoader} from ${JSON.stringify(loader('SPZLoader'))};
    import {KSPLATLoader} from ${JSON.stringify(loader('KSPLATLoader'))};
    const describe=g=>({native:g instanceof BufferGeometry,attributes:Object.fromEntries(Object.entries(g.attributes)
      .map(([name,a])=>[name,{type:a.array.constructor.name,itemSize:a.itemSize,count:a.count,normalized:a.normalized,values:Array.from(a.array)}])),
      min:g.boundingBox.min.toArray(),max:g.boundingBox.max.toArray(),center:g.boundingSphere.center.toArray(),radius:g.boundingSphere.radius});
    export async function run(){
      const sequence=[],progress=[];const ply=new PLY().setPath('models/');
      let finish,reject;const pending=new Promise((yes,no)=>{finish=yes;reject=no;});
      const returned=ply.load('cloud%20%23.ply?revision=2#points',g=>{sequence.push('loaded');finish(g);},
        event=>progress.push([event.type,event.loaded]),reject);
      sequence.push('returned');ply.setPath('alternate/');
      const first=await pending,second=await ply.loadAsync('cloud.ply');
      const third=await new PLY().setPath('models/').loadAsync('cloud%20%23.ply?revision=2#points');
      const firstInfo=describe(first),thirdInfo=describe(third);first.attributes.position.array[0]=999;
      const independent=first!==third&&first.attributes.position.array!==third.attributes.position.array&&third.attributes.position.getX(0)===1;
      const binary=await new SPLATLoader().setPath('models/').loadAsync('cloud.splat');
      const ksplat=new KSPLATLoader().setPath('models/');const sectioned=await ksplat.loadAsync('cloud.ksplat');
      const spz=new SPZLoader().setPath('models/');const compressed=await spz.loadAsync('cloud.v2.spz');
      const modern=await spz.loadAsync('cloud.v4.spz');
      let plyError,splatError,ksplatError,spzError;
      try{await new PLY().setPath('models/').loadAsync('bad.ply');}catch(e){plyError=[e.name,e.message];}
      try{await new SPLATLoader().setPath('models/').loadAsync('bad.splat');}catch(e){splatError=[e.name,e.message];}
      try{await ksplat.loadAsync('bad.ksplat');}catch(e){ksplatError=[e.name,e.message];}
      try{await spz.loadAsync('bad.spz');}catch(e){spzError=[e.name,e.message];}
      const parsed=new PLY().parse(${JSON.stringify(plyBytes(9).toString())});
      return {sequence,progress,returnedUndefined:returned===undefined,independent,first:firstInfo,second:describe(second),third:thirdInfo,
        binary:describe(binary),sectioned:describe(sectioned),compressed:describe(compressed),modern:describe(modern),parsed:describe(parsed),
        errors:{plyError,splatError,ksplatError,spzError}};
    }`;
  const base=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-splat-assets-'));
  const sourceDir=path.join(base,'source'),entry=path.join(sourceDir,'index.html');
  for(const [name,bytes] of Object.entries({'index.html':'<script type="module" src="app.mjs"></script>','app.mjs':source,...resources})){
    const file=path.join(sourceDir,name);fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,bytes);
  }
  const reference=await execute(pathToFileURL(path.join(sourceDir,'app.mjs')).href,{},resources);
  const networked=await buildApplication(entry,path.join(base,'networked'));
  const prepared=await buildApplication(entry,path.join(base,'prepared'),{prepareStandaloneLoaders:true});
  for(const output of [networked.outDir,prepared.outDir])for(const [name,bytes] of Object.entries(resources))
    assert.deepEqual(fs.readFileSync(path.join(output,name)),bytes,`${name} stays byte-identical`);
  const packedFile=path.join(base,'packed.html');packHtml(path.join(prepared.outDir,'index.html'),packedFile);
  const html=fs.readFileSync(packedFile,'utf8');
  const imports=JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const packedEntry=/<script type="module" src="([^"]+)"/.exec(html)[1];
  fs.renameSync(sourceDir,path.join(base,'source-unavailable'));
  fs.renameSync(prepared.outDir,path.join(base,'prepared-unavailable'));
  const relocated=path.join(base,'deployed');fs.renameSync(networked.outDir,relocated);
  const deployed=await execute(pathToFileURL(path.join(relocated,networked.entryFiles[0])).href,{},relocated);
  fs.renameSync(relocated,path.join(base,'deployed-unavailable'));
  const packed=await execute(packedEntry,imports,null);
  assert.deepEqual(deployed.result,reference.result,'relocated source parsers preserve all decoded buffers and lifecycle observations');
  assert.deepEqual(packed.result,reference.result,'offline source parsers preserve all decoded buffers and lifecycle observations');
  assert.deepEqual(packed.result.sequence,['returned','loaded']);
  assert.equal(packed.result.returnedUndefined,true);assert.equal(packed.result.independent,true);
  assert.deepEqual(packed.result.first.attributes.position.values,[1,2,3]);
  assert.deepEqual(packed.result.second.attributes.position.values,[7,2,3],'pending load keeps its captured path; later loads read the new path');
  assert.deepEqual(packed.result.parsed.attributes.position.values,[9,2,3],'runtime parse remains executable');
  assert.deepEqual(packed.result.binary.attributes.color.values,[10,20,30,40]);
  assert.deepEqual(packed.result.sectioned.attributes.covariance.values,[4,0,0,9,0,16]);
  for(const key of ['compressed','modern']){
    assert.deepEqual(packed.result[key].attributes.position.values,[1.5,-2,.25]);
    assert.deepEqual(packed.result[key].attributes.color.values,[128,128,128,64]);
    assert.equal(packed.result[key].attributes.sphericalHarmonics1.type,'Uint32Array');
  }
  for(const value of Object.values(packed.result.errors))assert.equal(value[0],'Error');
  assert.ok(packed.requests.length>=11);assert.ok(packed.requests.every(uri=>uri.startsWith('data:')),'only embedded model/decoder bytes are fetched');
  assert.ok(packed.requests.some(uri=>uri.startsWith('data:application/wasm;')),'SPZ v4 actually initializes the source Zstd Wasm');
});
