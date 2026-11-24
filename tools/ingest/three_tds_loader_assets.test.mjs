import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import {registerHooks} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {parse} from 'acorn';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';
import {tdsAssetUris} from './pack_tds.mjs';
import {collectThreeLoaderAssets, findThreeLoaderRequests} from './three_loader_assets.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url).href;
const core = upstream + 'build/three.core.js';
const tds = upstream + 'examples/jsm/loaders/TDSLoader.js';
const discover = source => findThreeLoaderRequests(parse(source, {ecmaVersion: 'latest', sourceType: 'module'}),
  {resolveModule: specifier => specifier === 'three' ? core : specifier});
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==', 'base64');
const names = ['paint%20%23.png?revision=2#diffuse', '/bump.png', 'opacity.png', 'specular.png'];
const mapIds = [0xa200, 0xa230, 0xa210, 0xa204];

function chunk(id, ...parts) {
  const body = Buffer.concat(parts), header = Buffer.alloc(6);
  header.writeUInt16LE(id); header.writeUInt32LE(body.length + 6, 2);
  return Buffer.concat([header, body]);
}
const string = value => Buffer.from(value + '\0', 'latin1');
const words = values => {const bytes = Buffer.alloc(values.length * 2); values.forEach((v, i) => bytes.writeUInt16LE(v, i * 2)); return bytes;};
const floats = values => {const bytes = Buffer.alloc(values.length * 4); values.forEach((v, i) => bytes.writeFloatLE(v, i * 4)); return bytes;};
function model(textureNames = names, extraMaterial = Buffer.alloc(0)) {
  return chunk(0x4d4d, chunk(0x3d3d,
    chunk(0xafff, chunk(0xa000, string('Paint')),
      ...textureNames.map((name, i) => chunk(mapIds[i], chunk(0xa300, string(name)),
        chunk(0xa354, floats([2])), chunk(0xa356, floats([3])), chunk(0xa358, floats([0.25])))), extraMaterial),
    chunk(0x4000, string('Triangle'), chunk(0x4100,
      chunk(0x4110, words([3]), floats([0, 0, 0, 1, 0, 0, 0, 1, 0])),
      chunk(0x4120, words([1, 0, 1, 2, 0]), chunk(0x4130, string('Paint'), words([1, 0]))),
      chunk(0x4140, words([3]), floats([0, 0, 1, 0, 0, 1]))))));
}
function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-tds-assets-'));
  for (const [name, bytes] of Object.entries(files)) {
    const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, bytes);
  }
  return root;
}

test('TDS imports and aliases close captured model bases and live pending resource paths', () => {
  const requests = discover(`import {TDSLoader as Source} from ${JSON.stringify(tds)};
    import * as Addons from ${JSON.stringify(upstream + 'examples/jsm/Addons.js')};
    const Loader=Source, pending=new Loader().setPath('models/'); pending.debug=false;
    pending.load('nested/first.3ds'); pending.setPath('later/'); pending.setResourcePath('textures/');
    pending.loadAsync('second.3ds'); pending.setResourcePath(''); pending.parse(runtimeBytes,'runtime/');
    new Addons.TDSLoader().loadAsync('nested/default.3ds');`);
  assert.deepEqual(requests.map(({url, texturePaths}) => [url, texturePaths]), [
    ['models/nested/first.3ds', ['models/', 'textures/']],
    ['later/second.3ds', ['later/', 'textures/']],
    ['nested/default.3ds', ['nested/']],
  ]);
  const opaque = discover(`import {TDSLoader} from ${JSON.stringify(tds)};
    import {TDSLoader as Foreign} from 'file:///unrelated/TDSLoader.js';
    new Foreign().load('foreign.3ds'); new TDSLoader(manager).load('managed.3ds');
    const escaped=new TDSLoader(); customize(escaped); escaped.load('escaped.3ds');
    const changed=new TDSLoader(); changed.parse=customParse; changed.load('changed.3ds');
    const debug=new TDSLoader(); debug.debug=callable; debug.load('callable.3ds');
    new TDSLoader().setPath(runtimePath).load('dynamic.3ds');
    const late=new TDSLoader(); late.load('pending.3ds'); late.setResourcePath(runtimePath);
    function shadow(TDSLoader){new TDSLoader().load('shadowed.3ds');}`);
  assert.deepEqual(opaque.map(({url, texturePaths}) => [url, texturePaths]), [['pending.3ds', null]]);
});

test('3DS metadata follows native map chunks, byte-valued names and duplicate requests', async () => {
  const extra = chunk(0xa200, chunk(0xa300, string('previous.png')), chunk(0xa300, string('caf\xe9.png')));
  const bytes = model(names, Buffer.concat([extra,
    chunk(0xaaaa, chunk(0xa200, chunk(0xa300, string('opaque.png')))),
    chunk(0xa220, chunk(0xa300, string('unsupported-reflection.png')))]));
  const before = Buffer.from(bytes), expected = [...names, 'previous.png', 'caf\xe9.png'];
  assert.deepEqual(tdsAssetUris(bytes), expected);
  const {TDSLoader} = await import(tds);
  const saved = globalThis.document, loaded = [];
  globalThis.document = {createElementNS() {return new class extends EventTarget {
    set src(url) {loaded.push(url); queueMicrotask(() => this.dispatchEvent(new Event('load')));}
  }();}};
  try {
    const group = new TDSLoader().setResourcePath('textures/').parse(Uint8Array.from(bytes).buffer, 'ignored/');
    assert.equal(group.children[0].name, 'Triangle');
    assert.deepEqual(loaded, expected.map(name => 'textures/' + name), 'the retained parser issues exactly the discovered image routes');
  } finally {if (saved === undefined) delete globalThis.document; else globalThis.document = saved;}
  assert.deepEqual(bytes, before);
  for (const signature of [0x3daa, 0xc23d]) {
    const variant = Buffer.from(bytes); variant.writeUInt16LE(signature);
    assert.deepEqual(tdsAssetUris(variant), expected);
  }
  const real = fs.readFileSync(new URL('examples/models/3ds/portalgun/portalgun.3ds', upstream));
  assert.deepEqual(tdsAssetUris(real), ['color.jpg']);
});

test('3DS chunk tolerance preserves partial metadata while work bounds remain explicit', () => {
  const name = chunk(0xa300, string('image.png'));
  name.writeUInt32LE(6, 2); // Native readString deliberately reads beyond this declared chunk end.
  const partial = chunk(0x4d4d, chunk(0x3d3d, chunk(0xafff, chunk(0xa200, name))));
  assert.deepEqual(tdsAssetUris(partial), ['image.png']);
  assert.deepEqual(tdsAssetUris(Buffer.from([0, 1])), []);
  assert.deepEqual(tdsAssetUris(chunk(0x1234, chunk(0x3d3d, chunk(0xafff, chunk(0xa200, name))))), []);
  const unterminated = chunk(0x4d4d, chunk(0x3d3d, chunk(0xafff, chunk(0xa200, chunk(0xa300, Buffer.from('missing-nul'))))));
  assert.deepEqual(tdsAssetUris(unterminated), []);
  assert.throws(() => tdsAssetUris({length: 64 * 1024 * 1024 + 1}), {code: 'TDS_BUDGET'});
  const stuck = chunk(0x4d4d, Buffer.alloc(6));
  assert.throws(() => tdsAssetUris(stuck), {code: 'TDS_BUDGET'});
});

test('3DS sidecars preserve rooted literal prefixes, encoded URLs and original model bytes', () => {
  const bytes = model(['paint%20%23.png?revision=2#surface', '/bump.png', 'https://assets.invalid/opacity.png', 'specular.png']);
  const root = fixture({'pages/index.html': '', 'models/nested/scene.3ds': bytes,
    'shared/paint #.png': pixel, 'shared/bump.png': pixel, 'shared/specular.png': pixel,
    'shared/https:/assets.invalid/opacity.png': pixel});
  const requests = discover(`import {TDSLoader} from ${JSON.stringify(tds)};
    new TDSLoader().setPath('/models/').setResourcePath('/shared/').loadAsync('nested/scene.3ds');`);
  const urls = [], assets = collectThreeLoaderAssets(requests, {documentRoot: root,
    documentUrl: pathToFileURL(path.join(root, 'pages/index.html')).href, onResource: (_name, url) => urls.push(url)});
  assert.deepEqual([...assets.keys()].sort(), ['models/nested/scene.3ds', 'shared/paint #.png', 'shared/bump.png',
    'shared/specular.png', 'shared/https:/assets.invalid/opacity.png'].sort());
  assert.deepEqual(assets.get('models/nested/scene.3ds').source, bytes);
  assert.ok(urls.some(url => url.endsWith('/shared//bump.png')));
  assert.ok(urls.some(url => url.endsWith('/shared/paint%20%23.png?revision=2#surface')));
});

// Native HTTP/data fetch and the pinned source parser execute. This recording
// host supplies document URL/import-map resolution and image completion only;
// byte equality is checked here, not browser JPEG/PNG decoding or rendering.
async function execute(url, imports, documentUrl, offline = false) {
  const saved = Object.fromEntries(['document', 'Request', 'ProgressEvent', 'fetch'].map(key => [key, globalThis[key]]));
  const nativeFetch = saved.fetch, requests = [], imageJobs = [], images = [];
  const hooks = registerHooks({resolve(specifier, context, next) {return next(imports[specifier] ?? specifier, context);}});
  globalThis.document = {URL: documentUrl, baseURI: new URL('site/', documentUrl).href,
    createElementNS(_namespace, tag) {assert.equal(tag, 'img'); return new ImageHost();}};
  globalThis.Request = class extends saved.Request {
    constructor(input, options) {super(typeof input === 'string' ? new URL(input, document.baseURI) : input, options);}
  };
  globalThis.ProgressEvent = class extends Event {constructor(type, details) {super(type); Object.assign(this, details);}};
  globalThis.fetch = async (input, options) => {
    const uri = input instanceof Request ? input.url : String(input); requests.push(uri);
    if (offline && !uri.startsWith('data:')) throw new Error('Unexpected external resource: ' + uri);
    return nativeFetch(input, options);
  };
  class ImageHost extends EventTarget {
    set src(url) {
      this._src = url;
      imageJobs.push((async () => {
        const bytes = Buffer.from(await (await fetch(new URL(url, document.baseURI))).arrayBuffer());
        images.push(hash(bytes)); this.byteLength = bytes.length; this.complete = true;
        this.naturalWidth = 1; this.naturalHeight = 1; this.dispatchEvent(new Event('load'));
      })());
    }
    get src() {return this._src;}
  }
  let timer;
  try {
    return await Promise.race([(async () => {
      const app = await import(url), observe = await app.run();
      await Promise.all(imageJobs);
      return {result: observe(), requests, images: images.sort()};
    })(), new Promise((_, reject) => {timer = setTimeout(() => reject(new Error('TDS loader execution timed out')), 15000);})]);
  } finally {
    clearTimeout(timer); hooks.deregister();
    for (const [key, value] of Object.entries(saved)) {if (value === undefined) delete globalThis[key]; else globalThis[key] = value;}
  }
}

test('textured 3DS applications preserve native models and material maps after relocation and offline packing', async () => {
  const portal = fs.readFileSync(new URL('examples/models/3ds/portalgun/portalgun.3ds', upstream));
  const color = fs.readFileSync(new URL('examples/models/3ds/portalgun/textures/color.jpg', upstream));
  const normal = fs.readFileSync(new URL('examples/models/3ds/portalgun/textures/normal.jpg', upstream));
  const resources = {'site/models/nested/scene #.3ds': model(), 'site/models/nested/default.3ds': model(),
    'site/models/invalid.3ds': Buffer.from([0, 1]), 'site/portal/portalgun.3ds': portal,
    'site/portal/textures/color.jpg': color, 'site/portal/textures/normal.jpg': normal};
  for (const prefix of ['site/textures/', 'site/models/nested/']) {
    for (const name of ['paint #.png', 'bump.png', 'opacity.png', 'specular.png']) resources[prefix + name] = pixel;
  }
  const source = `import {FileLoader,TextureLoader} from ${JSON.stringify(core)};
    import {TDSLoader} from ${JSON.stringify(tds)};
    const maps=['map','bumpMap','alphaMap','specularMap','normalMap'];
    const material=m=>({name:m.name,type:m.type,color:m.color.toArray(),opacity:m.opacity,
      maps:Object.fromEntries(maps.filter(key=>m[key]).map(key=>{const t=m[key];return [key,
        {repeat:t.repeat.toArray(),offset:t.offset.toArray(),wrap:[t.wrapS,t.wrapT],colorSpace:t.colorSpace,
          complete:t.image.complete,bytes:t.image.byteLength}];}))});
    const describe=group=>({type:group.type,scale:group.scale.toArray(),meshes:group.children.map(mesh=>({name:mesh.name,
      positions:[...mesh.geometry.attributes.position.array],uvs:[...mesh.geometry.attributes.uv.array],
      indices:[...mesh.geometry.index.array],groups:mesh.geometry.groups,
      materials:(Array.isArray(mesh.material)?mesh.material:[mesh.material]).map(material)}))});
    export async function run(){
      const sequence=[],progress=[],loader=new TDSLoader().setPath('models/');loader.debug=false;
      let returned;
      const pending=new Promise((resolve,reject)=>{returned=loader.load('nested/scene%20%23.3ds?edition=2#mesh',
        group=>{sequence.push('loaded');resolve(group);},event=>progress.push([event.type,event.loaded]),reject);
        sequence.push('returned');});
      loader.setPath('later/');loader.setResourcePath('textures/');
      const first=await pending;
      const second=await new TDSLoader().setPath('models/').setResourcePath('textures/').loadAsync('nested/scene%20%23.3ds?edition=2#mesh');
      const fallback=await new TDSLoader().loadAsync('models/nested/default.3ds');
      const portal=await new TDSLoader().setPath('portal/').setResourcePath('portal/textures/').loadAsync('portalgun.3ds');
      // Like the pinned example, authored normal maps use a separate TextureLoader.
      const normal=new TextureLoader().load('portal/textures/normal.jpg');
      portal.traverse(child=>{if(child.isMesh)child.material.normalMap=normal;});
      const raw=await new FileLoader().setResponseType('arraybuffer').loadAsync('models/nested/scene%20%23.3ds?edition=2#mesh');
      const parsed=new TDSLoader().setResourcePath('textures/').parse(raw,'ignored/');
      let failure;try{await new TDSLoader().loadAsync('models/invalid.3ds');}catch(error){failure={name:error.name,message:error.message};}
      return ()=>{
        const states=[first,second,fallback,parsed,portal].map(describe),a=first.children[0],b=second.children[0];
        a.geometry.attributes.position.setX(0,99);a.material.map.offset.x=99;
        const independent=first!==second&&a.geometry!==b.geometry&&a.material!==b.material&&a.material.map!==b.material.map&&
          b.geometry.attributes.position.getX(0)===0&&b.material.map.offset.x===0.25;
        return {sequence,progress,returnedUndefined:returned===undefined,states,independent,failure};
      };
    }`;
  const sourceDir = fixture({'index.html': '<base href="./site/"><script type="module" src="../app.mjs"></script>',
    'app.mjs': source, ...resources});
  const base = fixture({}), entry = path.join(sourceDir, 'index.html');
  let servedRoot = sourceDir;
  const server = http.createServer((request, response) => {
    try {
      const file = path.join(servedRoot, decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
      response.end(fs.readFileSync(file));
    } catch {response.writeHead(404); response.end('Missing resource');}
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const documentUrl = `http://127.0.0.1:${server.address().port}/index.html`;
    const reference = await execute(pathToFileURL(path.join(sourceDir, 'app.mjs')).href, {}, documentUrl);
    const networked = await buildApplication(entry, path.join(base, 'networked'));
    const prepared = await buildApplication(entry, path.join(base, 'prepared'), {prepareStandaloneLoaders: true});
    for (const output of [networked.outDir, prepared.outDir]) for (const [name, bytes] of Object.entries(resources))
      assert.deepEqual(fs.readFileSync(path.join(output, name)), bytes, name + ' stays byte-identical');
    const packedFile = path.join(base, 'packed.html');
    packHtml(path.join(prepared.outDir, 'index.html'), packedFile);
    const html = fs.readFileSync(packedFile, 'utf8');
    const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
    const packedEntry = /<script type="module" src="([^"]+)"/.exec(html)[1];
    fs.renameSync(sourceDir, sourceDir + '-unavailable');
    fs.renameSync(prepared.outDir, prepared.outDir + '-unavailable');
    const deployed = path.join(base, 'deployed'); fs.renameSync(networked.outDir, deployed); servedRoot = deployed;
    const relocated = await execute(pathToFileURL(path.join(deployed, networked.entryFiles[0])).href, {}, documentUrl);
    fs.renameSync(deployed, deployed + '-unavailable');
    const packed = await execute(packedEntry, imports, 'https://relocated.example/index.html', true);
    assert.deepEqual(relocated.result, reference.result, 'relocated retained parsers preserve source geometry, material maps, callbacks and errors');
    assert.deepEqual(packed.result, reference.result, 'offline retained parsers preserve source geometry, material maps, callbacks and errors');
    assert.deepEqual(packed.images, reference.images); assert.deepEqual(relocated.images, reference.images);
    assert.equal(packed.images.length, 18);
    assert.ok(packed.images.includes(hash(color))); assert.ok(packed.images.includes(hash(normal)));
    assert.deepEqual(packed.result.sequence, ['returned', 'loaded']);
    assert.equal(packed.result.returnedUndefined, true); assert.equal(packed.result.independent, true);
    assert.equal(packed.result.failure.name, 'RangeError');
    assert.deepEqual(packed.result.states[0].meshes[0].positions, [0, 0, 0, 1, 0, 0, 0, 1, 0]);
    assert.ok(packed.result.states[4].meshes.reduce((sum, mesh) => sum + mesh.positions.length / 3, 0) > 1000);
    assert.ok(packed.requests.every(uri => uri.startsWith('data:')), 'packed models and every material image use embedded bytes only');
    assert.ok(relocated.requests.some(uri => uri.includes('/site/textures/paint%20%23.png?revision=2')));
    assert.ok(relocated.requests.some(uri => uri.includes('/site/models/nested/paint%20%23.png?revision=2')));
  } finally {await new Promise(resolve => server.close(resolve));}
});
