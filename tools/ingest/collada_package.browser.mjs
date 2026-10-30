// Native r186 Collada XML, image and animation behavior versus a packed page
// after its source/build directories and server become unavailable.
// CHROMIUM_PATH=/path/to/chrome PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ \
//   node tools/ingest/collada_package.browser.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import {createRequire} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const upstream = fileURLToPath(new URL('../../upstream/three.js/', import.meta.url));
const loader = new URL('../../upstream/three.js/examples/jsm/loaders/ColladaLoader.js', import.meta.url);
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-collada-package-'));
const source = path.join(root, 'source'), output = path.join(root, 'build'), packed = path.join(root, 'packed.html');
for (const directory of ['models', 'elf', 'textures', 'runtime']) fs.mkdirSync(path.join(source, directory), {recursive: true});
const copied = new Map([
  ['stormtrooper/stormtrooper.dae', 'models/trooper #.dae'], ['stormtrooper/Stormtrooper_D.jpg', 'models/Stormtrooper_D.jpg'],
  ['elf/elf.dae', 'elf/elf.dae'], ...['Face_tex_002_toObj.jpg', 'Body_tex_003.jpg', 'Hair_tex_001.jpg', 'ce.jpg'].map(name => ['elf/' + name, 'textures/' + name]),
  ['skin_and_morph.dae', 'models/morph.dae'],
]);
for (const [from, to] of copied) fs.copyFileSync(path.join(upstream, 'examples/models/collada', from), path.join(source, to));
const runtimeXml = `<COLLADA version="1.4.1"><asset><unit meter="0.01"/><up_axis>Z_UP</up_axis></asset>
  <library_images><image id="pixel"><init_from>pixel.tga</init_from></image></library_images>
  <library_effects><effect id="effect"><profile_COMMON>
    <newparam sid="surface"><surface type="2D"><init_from>pixel</init_from></surface></newparam>
    <newparam sid="sampler"><sampler2D><source>surface</source></sampler2D></newparam>
    <technique sid="common"><lambert><diffuse><texture texture="sampler"/></diffuse></lambert></technique>
  </profile_COMMON></effect></library_effects>
  <library_materials><material id="material"><instance_effect url="#effect"/></material></library_materials>
  <library_visual_scenes><visual_scene id="view"><node id="runtime" name="runtime"><translate>1 2 3</translate></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#view"/></scene></COLLADA>`;
const tga = Buffer.alloc(21); tga[2] = 2; tga.writeUInt16LE(1, 12); tga.writeUInt16LE(1, 14); tga[16] = 24; tga[20] = 255;
fs.writeFileSync(path.join(source, 'runtime/model.dae'), runtimeXml, {flag:'wx'});
fs.writeFileSync(path.join(source, 'runtime/pixel.tga'), tga, {flag:'wx'});
fs.writeFileSync(path.join(source, 'models/malformed.dae'), '<COLLADA><asset></COLLADA>', {flag:'wx'});

function application(specifier) {
  return `import {ColladaLoader} from ${JSON.stringify(specifier)};
import {AnimationMixer} from 'three';
function hash(array){let value=2166136261;for(const byte of new Uint8Array(array.buffer,array.byteOffset,array.byteLength))value=Math.imul(value^byte,16777619)>>>0;return value;}
function meshes(scene){const result=[];scene.traverse(object=>{if(object.isMesh)result.push(object)});return result;}
async function summarize(result){
  const scene=result.scene,nodes=[],objects=[],images=new Map();let vertices=0,triangles=0,skinned=0;
  scene.updateMatrixWorld(true);scene.traverse(object=>objects.push(object));
  const identities=new Map(objects.map((object,index)=>[object.uuid,index]));
  for(const object of objects)for(const material of [object.material??[]].flat())for(const key of ['map','normalMap','specularMap','lightMap','emissiveMap']){
    const image=material[key]?.image;if(!image||images.has(image))continue;
    if(image instanceof HTMLImageElement)await image.decode();
    if(image.data){images.set(image,{width:image.width,height:image.height,hash:hash(image.data),firstPixel:[...image.data.slice(0,4)]});continue;}
    const canvas=document.createElement('canvas');canvas.width=image.naturalWidth??image.width;canvas.height=image.naturalHeight??image.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
    images.set(image,{width:canvas.width,height:canvas.height,hash:hash(pixels)});
  }
  for(const object of objects){
    const node={type:object.type,name:object.name,children:object.children.length,matrix:[...object.matrixWorld.elements]};
    if(object.geometry){const g=object.geometry;vertices+=g.attributes.position.count;triangles+=(g.index?.count??g.attributes.position.count)/3;
      node.attributes=Object.fromEntries(Object.entries(g.attributes).map(([name,a])=>[name,{count:a.count,size:a.itemSize,normalized:a.normalized,hash:hash(a.array)}]));
      node.morph=Object.fromEntries(Object.entries(g.morphAttributes).map(([name,list])=>[name,list.map(a=>hash(a.array))]));
      node.morphInfluences=object.morphTargetInfluences;node.index=g.index?{count:g.index.count,hash:hash(g.index.array)}:null;
      node.materials=[object.material].flat().map(m=>({type:m.type,name:m.name,color:m.color?.toArray(),opacity:m.opacity,side:m.side,
        textures:Object.fromEntries(['map','normalMap','specularMap','lightMap','emissiveMap'].filter(k=>m[k]).map(k=>[k,
          {image:images.get(m[k].image),wrap:[m[k].wrapS,m[k].wrapT],repeat:m[k].repeat.toArray(),offset:m[k].offset.toArray(),colorSpace:m[k].colorSpace}]))}));
    }
    if(object.isSkinnedMesh){skinned++;object.skeleton.update();node.skin={bones:object.skeleton.bones.map(b=>identities.get(b.uuid)),
      inverses:object.skeleton.boneInverses.map(m=>m.toArray()),matrices:hash(object.skeleton.boneMatrices),bind:object.bindMatrix.toArray()};}
    nodes.push(node);
  }
  const animations=scene.animations.map(clip=>({name:clip.name,duration:clip.duration,tracks:clip.tracks.map(track=>{
    const split=track.name.indexOf('.'),id=track.name.slice(0,split);return {name:(identities.get(id)??id)+track.name.slice(split),
      type:track.ValueTypeName,interpolation:track.getInterpolation(),times:hash(track.times),values:hash(track.values)};})}));
  return JSON.parse(JSON.stringify({nodes,animations,meshes:meshes(scene).length,skinned,vertices,triangles,images:[...images.values()]}));
}
async function poses(result){const mixer=new AnimationMixer(result.scene);for(const clip of result.scene.animations)mixer.clipAction(clip).play();
  const frames=[];for(const time of [0,0.35,0.9]){mixer.setTime(time);frames.push(await summarize(result));}return frames;}
globalThis.__f3dColladaRun=async(runtimeText)=>{
  const trooper=await new ColladaLoader().setPath('models/').loadAsync('trooper%20%23.dae?edition=2#rig');
  const second=await new ColladaLoader().setPath('models/').loadAsync('trooper%20%23.dae?edition=2#rig');
  const sequence=[],elfLoader=new ColladaLoader().setResourcePath('before/');
  const pending=new Promise((resolve,reject)=>elfLoader.load('elf/elf.dae',value=>{sequence.push('loaded');resolve(value)},undefined,reject));
  elfLoader.setResourcePath('textures/');sequence.push('returned');const elf=await pending;
  const morph=await new ColladaLoader().setPath('models/').loadAsync('morph.dae');
  const small=await new ColladaLoader().loadAsync('runtime/model.dae');
  // This text is supplied after packing, not captured as a source asset.
  const parsed=new ColladaLoader().parse(runtimeText,'runtime/');
  const texture=small.library.materials.material.build.map,parsedTexture=parsed.library.materials.material.build.map;
  const a=meshes(trooper.scene)[0],b=meshes(second.scene)[0];
  const trooperFrames=await poses(trooper),secondFrames=await poses(second),morphFrames=await poses(morph);
  for(let i=0;i<100&&(!texture.image?.data||!parsedTexture.image?.data);i++)await new Promise(r=>setTimeout(r,10));
  return {trooperFrames,secondFrames,morphFrames,elf:await summarize(elf),sequence,
    independent:trooper.scene!==second.scene&&a.geometry!==b.geometry&&a.material!==b.material&&a.skeleton!==b.skeleton,
    runtime:{loaded:await summarize(small),parsed:await summarize(parsed),pixels:[...texture.image.data],parsedPixels:[...parsedTexture.image.data],
      type:texture.type,format:texture.format,flipY:texture.flipY,independent:texture!==parsedTexture},
    malformed:await new ColladaLoader().loadAsync('models/malformed.dae')};
};`;
}
fs.writeFileSync(path.join(source,'index.html'),'<script type="module" src="app.mjs"></script>',{flag:'wx'});
fs.writeFileSync(path.join(source,'app.mjs'),application(loader.href),{flag:'wx'});
await buildApplication(path.join(source,'index.html'),output,{prepareStandaloneLoaders:true});
for(const name of [...copied.values(),'runtime/model.dae','runtime/pixel.tga','models/malformed.dae'])
  assert.deepEqual(fs.readFileSync(path.join(output,name)),fs.readFileSync(path.join(source,name)),name);
packHtml(path.join(output,'index.html'),packed);

const {chromium}=createRequire(process.env.PLAYWRIGHT_MODULE_DIR??'/opt/node-tools/node_modules/')('playwright');
const server=http.createServer((request,response)=>{
  try{
    const pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname);
    if(pathname==='/'){response.setHeader('Content-Type','text/html');response.end('<script type="importmap">{"imports":{"three":"/upstream/build/three.module.js"}}</script><script type="module" src="/app.mjs"></script>');return;}
    if(pathname==='/app.mjs'){response.setHeader('Content-Type','text/javascript');response.end(application('/upstream/examples/jsm/loaders/ColladaLoader.js'));return;}
    if(pathname==='/favicon.ico'){response.writeHead(204).end();return;}
    const base=pathname.startsWith('/upstream/')?upstream:source;
    const relative=pathname.startsWith('/upstream/')?pathname.slice('/upstream/'.length):pathname.slice(1),filename=path.resolve(base,relative);
    if(!filename.startsWith(path.resolve(base)+path.sep)){response.writeHead(403).end();return;}
    response.setHeader('Content-Type',filename.endsWith('.js')?'text/javascript':'application/octet-stream');response.end(fs.readFileSync(filename));
  }catch{response.writeHead(404).end('missing');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let serverOpen=true;
async function capture(url,offline){
  const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,
    args:['--no-sandbox','--disable-gpu','--no-zygote','--single-process','--disable-dev-shm-usage']});
  try{
    const context=await browser.newContext({offline,userAgent:'OpenAI File Downloader, XaiImageApiFetch/1.0'}),page=await context.newPage();
    const errors=[],parseErrors=[],warnings=[],external=[];page.on('pageerror',error=>errors.push(String(error)));
    page.on('console',message=>{if(message.type()==='error'){
      if(message.text().startsWith('THREE.ColladaLoader: Failed to parse collada file.'))parseErrors.push(message.text());else errors.push(message.text());}
      if(message.type()==='warning')warnings.push(message.text());});
    page.on('request',request=>{if(offline&&/^https?:/.test(request.url()))external.push(request.url());});
    await page.goto(url);await page.waitForFunction(()=>typeof globalThis.__f3dColladaRun==='function');
    const result=await page.evaluate(text=>globalThis.__f3dColladaRun(text),runtimeXml.replace('name="runtime"','name="runtime-after-packing"'));
    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);assert.equal(parseErrors.length,1);
    return {result,parseErrors,warnings,browserVersion:browser.version()};
  }finally{await browser.close();}
}
try{
  const reference=await capture(`http://127.0.0.1:${server.address().port}/`,false);
  await new Promise(resolve=>server.close(resolve));serverOpen=false;
  fs.renameSync(source,path.join(root,'source-unavailable'));fs.renameSync(output,path.join(root,'build-unavailable'));
  const candidate=await capture(pathToFileURL(packed).href,true);
  assert.deepEqual(candidate.result,reference.result);assert.deepEqual(candidate.parseErrors,reference.parseErrors);
  assert.deepEqual(candidate.warnings,reference.warnings);
  const r=candidate.result;assert.equal(r.independent,true);assert.deepEqual(r.sequence,['returned','loaded']);
  assert.deepEqual(r.trooperFrames,r.secondFrames);assert.notDeepEqual(r.trooperFrames[0],r.trooperFrames[1]);
  assert.ok(r.trooperFrames[0].skinned>0&&r.trooperFrames[0].animations.length>0);assert.ok(r.elf.images.length>=3);
  // The pinned Collada parser does not implement morph target animation. Its
  // warning and partially parsed result must survive packaging unchanged.
  assert.ok(candidate.warnings.includes('THREE.ColladaLoader: Morph target animation not supported yet.'));
  assert.deepEqual(r.runtime.pixels,[255,0,0,255]);assert.deepEqual(r.runtime.parsedPixels,r.runtime.pixels);
  assert.equal(r.runtime.independent,true);assert.equal(r.malformed,null);
  console.log(JSON.stringify({status:'pass',browser:candidate.browserVersion,packed,
    trooper:{meshes:r.trooperFrames[0].meshes,skinned:r.trooperFrames[0].skinned,vertices:r.trooperFrames[0].vertices,triangles:r.trooperFrames[0].triangles,
      animations:r.trooperFrames[0].animations.length,poseCheckpoints:r.trooperFrames.length},
    elf:{meshes:r.elf.meshes,vertices:r.elf.vertices,images:r.elf.images},sourceMorphWarningPreserved:true,
    runtimeTgaPixels:r.runtime.pixels,exactSceneComparison:true,offline:true,externalRequests:0,browserErrors:0,expectedParseErrors:1}));
}finally{if(serverOpen)await new Promise(resolve=>server.close(resolve));}
