import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {registerHooks} from 'node:module';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {parse} from 'acorn';
import {findThreeLoaderRequests} from './three_loader_assets.mjs';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const root = new URL('../../upstream/three.js/', import.meta.url).href;
const core = root + 'build/three.core.js';
const loader = name => root + `examples/jsm/loaders/${name}.js`;
const resolveModule = specifier => specifier.startsWith('three/addons/')
  ? root + 'examples/jsm/' + specifier.slice('three/addons/'.length)
  : specifier === 'three' ? core : specifier === 'three/webgpu' ? root + 'build/three.webgpu.js'
  : specifier === 'three/tsl' ? root + 'build/three.tsl.js' : new URL(specifier, root + 'examples/').href;
const discover = source => findThreeLoaderRequests(parse(source, {ecmaVersion: 'latest', sourceType: 'module'}), {resolveModule});
const faces = ['px', 'nx', 'py', 'ny', 'pz', 'nz'];

test('pinned HDR cube lighting and IES spotlight examples retain their authored assets', () => {
  for (const example of ['webgl_tsl_clearcoat', 'webgl_materials_physical_clearcoat', 'webgpu_clearcoat',
    'webgpu_materialx_noise', 'webgpu_cubemap_dynamic', 'webgpu_cubemap_mix', 'webgpu_pmrem_cubemap',
    'webgl_pmrem_cubemap']) {
    const html = fs.readFileSync(new URL(`examples/${example}.html`, root), 'utf8');
    const source = /<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];
    const urls = discover(source).filter(request => request.loader === 'HDRCubeTextureLoader').map(request => request.url);
    const prefix = ['webgl_tsl_clearcoat', 'webgl_materials_physical_clearcoat', 'webgpu_clearcoat', 'webgpu_materialx_noise'].includes(example) ? '' : './';
    assert.deepEqual(urls, faces.map(face => prefix + 'textures/cube/pisaHDR/' + face + '.hdr'), example);
  }
  // This ninth HDR example exposes DefaultLoadingManager. Preserve the
  // existing explicit boundary instead of assuming its URL routing is closed.
  const managed = fs.readFileSync(new URL('examples/webgl_materials_envmaps_hdr.html', root), 'utf8');
  assert.deepEqual(discover(/<script type="module">([\s\S]*?)<\/script>/.exec(managed)[1]), []);
  const html = fs.readFileSync(new URL('examples/webgpu_lights_ies_spotlight.html', root), 'utf8');
  const source = /<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];
  assert.deepEqual(discover(source).filter(request => request.loader === 'IESLoader').map(request => request.url),
    ['007cfb11e343e2f42e3b476be4ab684e', '06b4cfdc8805709e767b5e2e904be8ad',
      '02a7562c650498ebb301153dbbf59207', '1a936937a49c63374e6d4fbed9252b29'].map(name => './ies/' + name + '.ies'));
});

test('lighting discovery preserves source datatype fields, aliases and conservative URL boundaries', () => {
  const imports = `import * as T from 'three';
    import {FloatType as Float, HalfFloatType} from 'three';
    import {HDRCubeTextureLoader as Cube, IESLoader as IES} from 'three/addons/Addons.js';
    import * as Addons from 'three/addons/Addons.js';`;
  const six = JSON.stringify(faces.map(face => face + '.hdr'));
  const requests = discover(imports + `
    const CubeAlias = Cube, cube = new CubeAlias().setPath('hdr/').setDataType(Float);
    const urls = ${six}; cube.load(urls); cube.loadAsync(urls);
    new Addons.HDRCubeTextureLoader().setPath('other/').load(${six});
    const ies = new IES().setPath('photometry/'), kind = HalfFloatType;
    ies.type = Float; ies.load('one%20%23.ies?q=1#beam');
    ies.type = kind; ies.loadAsync('two.ies');
    ies.type = choice ? T.UnsignedByteType : T.HalfFloatType; ies.load('three.ies');
    ies.type = 1015; const selectedType = ies.type; ies.load('four.ies');
    new Addons.IESLoader().load('namespace.ies');`);
  assert.deepEqual(requests.map(request => request.url), [
    ...faces.map(face => 'hdr/' + face + '.hdr'), ...faces.map(face => 'hdr/' + face + '.hdr'),
    ...faces.map(face => 'other/' + face + '.hdr'),
    'photometry/one%20%23.ies?q=1#beam', 'photometry/two.ies', 'photometry/three.ies', 'photometry/four.ies', 'namespace.ies']);

  for (const operation of [
    `new Cube().load('one.hdr');`, `new Cube().load(['one.hdr']);`,
    `const urls=${six}; urls[0]='changed.hdr'; new Cube().load(urls);`,
    `const urls=${six}; inspect(urls); new Cube().load(urls);`,
    `const cube=new Cube(); inspect(cube); cube.load(${six});`,
    `new Cube(manager).load(${six});`,
    `const ies=new IES(); ies.type=function(){this.setPath('moved/');}; ies.load('one.ies');`,
    `const ies=new IES(); ies.type(); ies.load('one.ies');`,
    `const ies=new IES(); const alias=ies; alias.setPath('moved/'); ies.load('one.ies');`,
    `const ies=new IES(); ies.type=externalType; ies.load('one.ies');`,
    `const ies=new IES(); ies.type+=1; ies.load('one.ies');`,
    `const ies=new IES(); ies.setDataType(Float); ies.load('one.ies');`,
  ]) assert.deepEqual(discover(imports + operation), [], operation);
  assert.deepEqual(discover(`import {IESLoader} from 'file:///foreign/examples/jsm/loaders/IESLoader.js';
    new IESLoader().load('one.ies');`), []);
  assert.deepEqual(discover(`import {IESLoader as IES} from 'three/addons/loaders/IESLoader.js';
    import {FloatType} from 'file:///foreign/three.core.js';
    const ies=new IES();ies.type=FloatType;ies.load('one.ies');`), []);
});

const dataUrl = (bytes, type = 'application/octet-stream') => `data:${type};base64,${Buffer.from(bytes).toString('base64')}`;
function fixture(files) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-lighting-assets-'));
  for (const [name, bytes] of Object.entries(files)) {
    const file = path.join(base, 'source', name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, bytes);
  }
  return {base, entry: path.join(base, 'source/index.html'), out: path.join(base, 'build'), packed: path.join(base, 'packed.html')};
}

// This host supplies document-relative Request resolution and embedded/local
// bytes. Native ESM, fetch, streams, and the exact retained parsers execute; no
// browser decoder or GPU pixel claim is made by these data/metadata checks.
async function execute(url, imports, resources, runtimeIES) {
  const saved = Object.fromEntries(['document', 'Request', 'ProgressEvent', 'fetch'].map(key => [key, globalThis[key]]));
  const requests = [], nativeFetch = saved.fetch;
  const hooks = registerHooks({resolve(specifier, context, next) { return next(imports[specifier] ?? (specifier === 'three' ? core : specifier), context); }});
  globalThis.document = {URL: 'https://lighting.example/moved/elsewhere/index.html', baseURI: 'https://lighting.example/moved/elsewhere/'};
  globalThis.Request = class extends saved.Request { constructor(input, options) { super(typeof input === 'string' ? new URL(input, document.baseURI) : input, options); } };
  globalThis.ProgressEvent = class extends Event { constructor(type, details) { super(type); Object.assign(this, details); } };
  globalThis.fetch = async (input, options) => {
    const request = input instanceof Request ? input.url : String(input); requests.push(request);
    if (request.startsWith('data:')) return nativeFetch(input, options);
    const name = decodeURIComponent(new URL(request).pathname).slice('/moved/elsewhere/'.length);
    if (!resources?.[name]) throw new Error('Unavailable resource: ' + request);
    return nativeFetch(dataUrl(resources[name]), options);
  };
  let timer;
  try {
    const result = await Promise.race([(async () => { const app = await import(url); return app.run(runtimeIES); })(),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Lighting loader execution timed out')), 15000); })]);
    return {result, requests};
  } finally { clearTimeout(timer); hooks.deregister(); for (const [key, value] of Object.entries(saved)) { if (value === undefined) delete globalThis[key]; else globalThis[key] = value; } }
}

test('original HDR cube and IES loaders preserve pixels, pending types and callbacks after relocation', async () => {
  const ies = fs.readFileSync(new URL('examples/ies/007cfb11e343e2f42e3b476be4ab684e.ies', root));
  const hdr = faces.map((_, index) => Buffer.concat([Buffer.from('#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n'),
    Buffer.from([255 - index * 24, 128 + index * 8, 32 + index * 16, 128])]));
  const resources = {...Object.fromEntries(faces.map((face, index) => ['lighting/cube/' + face + '.hdr', hdr[index]])),
    'lighting/ies/lamp #.ies': ies};
  const source = `import {FloatType, HalfFloatType, UnsignedByteType} from ${JSON.stringify(core)};
    import {HDRCubeTextureLoader} from ${JSON.stringify(loader('HDRCubeTextureLoader'))};
    import {IESLoader} from ${JSON.stringify(loader('IESLoader'))};
    const describe = texture => ({width:texture.image.width,height:texture.image.height,type:texture.type,format:texture.format,
      colorSpace:texture.colorSpace,minFilter:texture.minFilter,magFilter:texture.magFilter,generateMipmaps:texture.generateMipmaps,
      version:texture.version,array:texture.image.data.constructor.name,
      bytes:Array.from(new Uint8Array(texture.image.data.buffer,texture.image.data.byteOffset,texture.image.data.byteLength))});
    const cubeInfo = cube => ({type:cube.type,colorSpace:cube.colorSpace,generateMipmaps:cube.generateMipmaps,version:cube.version,
      faces:cube.images.map(describe)});
    export async function run(runtimeIES) {
      const sequence=[];
      const cubeLoader=new HDRCubeTextureLoader().setPath('lighting/cube/').setDataType(FloatType);
      const urls=${JSON.stringify(faces.map(face => face + '.hdr?revision=2#' + face))};
      let resolveLoad,rejectLoad;
      const pending=new Promise((resolve,reject)=>{resolveLoad=resolve;rejectLoad=reject;});
      const returned=cubeLoader.load(urls,value=>{sequence.push('loaded');resolveLoad(value);},undefined,rejectLoad);
      sequence.push('returned'); cubeLoader.setDataType(HalfFloatType);
      const first=await pending; cubeLoader.setDataType(FloatType);
      const second=await cubeLoader.loadAsync(urls);
      const iesLoader=new IESLoader().setPath('lighting/ies/');
      iesLoader.type=FloatType; const iesPending=iesLoader.loadAsync('lamp%20%23.ies?revision=2#beam');
      iesLoader.type=UnsignedByteType; const firstIES=await iesPending;
      iesLoader.type=HalfFloatType; const secondIES=await iesLoader.loadAsync('lamp%20%23.ies?revision=2#beam');
      const runtime=new IESLoader();runtime.type=FloatType;const parsed=runtime.parse(runtimeIES);
      let parseError;try{runtime.parse('malformed IES');}catch(error){parseError={name:error.name,message:error.message};}
      const missing=['not','present','.ies'].join('');
      let loadError;try{await new IESLoader().loadAsync(missing);}catch(error){loadError={name:error.name,message:error.message};}
      await Promise.resolve();
      return {sequence,returnIdentity:first===returned,independent:first!==second&&first.images.every((face,i)=>face!==second.images[i]),
        first:cubeInfo(first),second:cubeInfo(second),firstIES:describe(firstIES),secondIES:describe(secondIES),parsed:describe(parsed),
        parseError,loadError};
    }`;
  const f = fixture({'index.html':'<script type="module" src="app.mjs"></script>', 'app.mjs':source, ...resources});
  const runtimeIES = ies.toString().replace('330.8', '430.8');
  const reference = await execute(pathToFileURL(path.join(f.base, 'source/app.mjs')).href, {}, resources, runtimeIES);
  const networked = await buildApplication(f.entry, path.join(f.base, 'networked'));
  await buildApplication(f.entry, f.out, {prepareStandaloneLoaders:true});
  for (const output of [networked.outDir, f.out]) for (const [name, bytes] of Object.entries(resources))
    assert.deepEqual(fs.readFileSync(path.join(output, name)), bytes, name);
  packHtml(path.join(f.out, 'index.html'), f.packed);
  const html = fs.readFileSync(f.packed, 'utf8');
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const entry = /<script type="module" src="([^"]+)"/.exec(html)[1];
  fs.renameSync(path.join(f.base, 'source'), path.join(f.base, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.base, 'build-unavailable'));
  fs.renameSync(networked.outDir, path.join(f.base, 'networked-relocated'));
  const packed = await execute(entry, imports, null, runtimeIES);
  const summary = result => JSON.parse(JSON.stringify(result, (key, value) => key === 'bytes'
    ? {length:value.length,sha256:createHash('sha256').update(Uint8Array.from(value)).digest('hex')} : value));
  assert.deepEqual(summary(packed.result), summary(reference.result), 'retained metadata and decoded pixel bytes');
  assert.deepEqual(packed.result.sequence, ['returned', 'loaded']);
  assert.equal(packed.result.returnIdentity, true); assert.equal(packed.result.independent, true);
  assert.equal(packed.result.first.faces[0].array, 'Uint16Array', 'pending HDR decode reads the later source datatype');
  assert.equal(packed.result.first.type, 1015, 'cube metadata keeps the datatype present at load()');
  assert.equal(packed.result.second.faces[0].array, 'Float32Array');
  assert.equal(packed.result.firstIES.array, 'Uint8Array', 'pending IES parse reads its later .type value');
  assert.equal(packed.result.secondIES.array, 'Uint16Array');
  assert.equal(packed.result.firstIES.width, 180); assert.equal(packed.result.firstIES.height, 1);
  assert.equal(packed.result.firstIES.bytes.length, 360 * 180, 'preserve actual pinned IES dimensions and data length');
  assert.ok(packed.result.parseError); assert.match(packed.result.loadError.message, /notpresent\.ies/);
  assert.equal(packed.requests.filter(url => url.startsWith('data:')).length, 14);
  assert.deepEqual(packed.requests.filter(url => !url.startsWith('data:')), ['https://lighting.example/moved/elsewhere/notpresent.ies']);
});
