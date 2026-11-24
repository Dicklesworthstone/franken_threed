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
import {vrmlAssetUris} from './pack_vrml.mjs';
import {collectThreeLoaderAssets, findThreeLoaderRequests} from './three_loader_assets.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url).href;
const core = upstream + 'build/three.core.js';
const vrml = upstream + 'examples/jsm/loaders/VRMLLoader.js';
const discover = source => findThreeLoaderRequests(parse(source, {ecmaVersion: 'latest', sourceType: 'module'}),
  {resolveModule: specifier => specifier === 'three' ? core : specifier});
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==', 'base64');
const scene = (image = 'paint%20%23.png?revision=2#surface') => `#VRML V2.0 utf8
WorldInfo { title "Retained scene" info ["source authored"] }
DEF Painted Shape {
  appearance Appearance {
    material Material { diffuseColor 0.2 0.4 0.6 shininess 0.25 transparency 0.1 }
    texture ImageTexture { url ["${image}" "unused-fallback.png"] repeatS FALSE repeatT TRUE }
    textureTransform TextureTransform { scale 2 3 translation 0.25 0.5 rotation 0.125 }
  }
  geometry IndexedFaceSet {
    coord Coordinate { point [0 0 0, 1 0 0, 0 1 0] }
    coordIndex [0 1 2 -1]
    texCoord TextureCoordinate { point [0 0, 1 0, 0 1] }
    solid FALSE
  }
}
Transform { translation 2 3 4 children [USE Painted] }
`;
function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-vrml-assets-'));
  for (const [name, bytes] of Object.entries(files)) {
    const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, bytes);
  }
  return root;
}

async function nativeImageRequests(text) {
  const {VRMLLoader} = await import(vrml);
  const saved = globalThis.document, requests = [];
  globalThis.document = {createElementNS() {return new class extends EventTarget {
    set src(url) {requests.push(url); queueMicrotask(() => this.dispatchEvent(new Event('load')));}
  }();}};
  try {
    const result = new VRMLLoader().setResourcePath('images/').parse(text, 'ignored/');
    await Promise.resolve();
    return {requests: [...new Set(requests)], result};
  } finally {if (saved === undefined) delete globalThis.document; else globalThis.document = saved;}
}

test('VRML aliases and addon imports retain captured defaults and live pending resource paths', () => {
  const requests = discover(`import {VRMLLoader as Source} from ${JSON.stringify(vrml)};
    import * as Addons from ${JSON.stringify(upstream + 'examples/jsm/Addons.js')};
    const Loader=Source, pending=new Loader().setPath('models/');
    pending.load('nested/first.wrl'); pending.setPath('later/'); pending.setResourcePath('textures/');
    pending.loadAsync('second.wrl'); pending.setResourcePath(''); pending.parse(runtimeText,'runtime/');
    new Addons.VRMLLoader().loadAsync('nested/default.wrl');`);
  assert.deepEqual(requests.map(({url, texturePaths}) => [url, texturePaths]), [
    ['models/nested/first.wrl', ['models/', 'textures/']],
    ['later/second.wrl', ['later/', 'textures/']],
    ['nested/default.wrl', ['nested/']],
  ]);
  const opaque = discover(`import {VRMLLoader} from ${JSON.stringify(vrml)};
    import {VRMLLoader as Foreign} from 'file:///unrelated/VRMLLoader.js';
    new Foreign().load('foreign.wrl');new VRMLLoader(manager).load('managed.wrl');
    const escaped=new VRMLLoader();customize(escaped);escaped.load('escaped.wrl');
    const changed=new VRMLLoader();changed.parse=customParse;changed.load('changed.wrl');
    new VRMLLoader().setPath(runtimePath).load('dynamic.wrl');
    const late=new VRMLLoader();late.load('pending.wrl');late.setResourcePath(runtimePath);
    function shadow(VRMLLoader){new VRMLLoader().load('shadowed.wrl');}`);
  assert.deepEqual(opaque.map(({url, texturePaths}) => [url, texturePaths]), [['pending.wrl', null]]);
});

test('VRML resource reachability matches native DEF/USE, first URLs and unsupported node boundaries', async () => {
  const text = String.raw`#VRML V2.0 utf8
    # ImageTexture { url "comment.png" }
    Inline { children [
      DEF Pick Appearance { texture ImageTexture { url "first-definition.png" } }
      DEF Pick Appearance { texture ImageTexture { url "last-definition.png" } }
      DEF Raw ImageTexture { url "raw-use-is-ignored.png" }
    ] }
    Collision {
      proxy Shape { appearance Appearance { texture ImageTexture { url "proxy.png" } } geometry Box {} }
      children [
        Shape { appearance USE Pick geometry Box {} }
        Shape { appearance Appearance { texture USE Raw } geometry Box {} }
      ]
    }
    ImageTexture { url ["a'postrophe.png" "unused-alternative.png"] url "later-field.png" }
    ImageTexture { url "dir\\tail.png" }
    ImageTexture { url "escaped\"quote.png" }
    ImageTexture { url "unicode\u0061.png" }
    ImageTexture { url ["" "empty-does-not-fallback.png"] }
    Background { backUrl ["background.png"] }
    MovieTexture { url "movie.mp4" }
    WorldInfo { info ["ImageTexture { url \"string.png\" }"] }
  `;
  const expected = ['last-definition.png', 'apostrophe.png', 'later-field.png',
    String.raw`dir\\tail.png`, String.raw`escaped\quote.png`, String.raw`unicode\u0061.png`];
  assert.deepEqual(vrmlAssetUris(Buffer.from(text)), expected);
  const native = await nativeImageRequests(text);
  assert.deepEqual(native.requests, expected.map(name => 'images/' + name));
  assert.equal(native.result.children[0].children.length, 2);
});

test('VRML image metadata matches real pinned geometry, cameras, PixelTexture and multiline fixtures', async () => {
  for (const name of ['meshWithTexture', 'house', 'camera', 'extrusion1', 'pixelTexture', 'multilineString']) {
    const bytes = fs.readFileSync(new URL('examples/models/vrml/' + name + '.wrl', upstream)), before = Buffer.from(bytes);
    const native = await nativeImageRequests(bytes.toString('utf8'));
    assert.deepEqual(vrmlAssetUris(bytes).map(uri => 'images/' + uri), native.requests, name);
    assert.deepEqual(bytes, before, name + ' bytes remain unchanged');
  }
});

test('VRML version and comment tokens stop at native Unicode line separators', async () => {
  for (const text of ['#VRML V2.0 utf8\u2028ImageTexture {url "line-separator.png"}',
    '#VRML V2.0 utf8\n# ignored image\u2029ImageTexture {url "line-separator.png"}']) {
    assert.deepEqual(vrmlAssetUris(Buffer.from(text)), ['line-separator.png']);
    assert.deepEqual((await nativeImageRequests(text)).requests, ['images/line-separator.png']);
  }
});

test('VRML metadata bounds are explicit and malformed bytes retain their source parser', () => {
  assert.throws(() => vrmlAssetUris({length: 64 * 1024 * 1024 + 1}), {code: 'VRML_BUDGET'});
  assert.throws(() => vrmlAssetUris(Buffer.from('#VRML V2.0 utf8\n' + 'Group { children ['.repeat(130) + '] }'.repeat(130))), {code: 'VRML_BUDGET'});
  assert.throws(() => vrmlAssetUris(Buffer.from('#VRML V2.0 utf8\nImageTexture {url "unterminated}')), {code: 'VRML_PARSE'});
  assert.throws(() => vrmlAssetUris(Buffer.from('#VRML V2.0 utf8\nGroup {children [')), {code: 'VRML_PARSE'});
  assert.deepEqual(vrmlAssetUris(Buffer.from('#VRML V1.0 ascii\nImageTexture {url "old.png"}')), []);
  const root = fixture({'index.html': '', 'invalid.wrl': '#VRML V2.0 utf8\nGroup { children ['});
  const assets = collectThreeLoaderAssets(discover(`import {VRMLLoader} from ${JSON.stringify(vrml)};new VRMLLoader().load('invalid.wrl');`),
    {documentUrl: pathToFileURL(path.join(root, 'index.html')).href});
  assert.equal(assets.get('invalid.wrl').source.toString(), '#VRML V2.0 utf8\nGroup { children [');
});

test('VRML sidecars retain literal root prefixes and encoded query/fragment identities', () => {
  const text = scene('/paint%20%23.png?revision=2#surface');
  const root = fixture({'pages/index.html': '', 'models/nested/scene.wrl': text, 'shared/paint #.png': pixel,
    'paint #.png': 'wrong root', 'shared/unused-fallback.png': 'not loaded'});
  const requests = discover(`import {VRMLLoader} from ${JSON.stringify(vrml)};
    new VRMLLoader().setPath('/models/').setResourcePath('/shared/').loadAsync('nested/scene.wrl');`);
  const urls = [], assets = collectThreeLoaderAssets(requests, {documentRoot: root,
    documentUrl: pathToFileURL(path.join(root, 'pages/index.html')).href, onResource: (_name, url) => urls.push(url)});
  assert.deepEqual([...assets.keys()].sort(), ['models/nested/scene.wrl', 'shared/paint #.png']);
  assert.equal(assets.get('models/nested/scene.wrl').source.toString(), text);
  assert.deepEqual(assets.get('shared/paint #.png').source, pixel);
  assert.ok(urls.some(url => url.endsWith('/shared//paint%20%23.png?revision=2#surface')));
});

// Native HTTP/data fetch and retained VRML/Chevrotain parsing execute. This
// existing recording-host pattern supplies document/import-map and image host
// behavior; image bytes are verified, not browser pixel decoding or rendering.
async function execute(url, imports, documentUrl, runtimeText, offline = false) {
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
      const job = (async () => {
        const response = await fetch(new URL(url, document.baseURI));
        if (!response.ok) throw new Error('Missing image: ' + url);
        const bytes = Buffer.from(await response.arrayBuffer());
        images.push(hash(bytes)); this.byteLength = bytes.length; this.complete = true;
        this.naturalWidth = 1; this.naturalHeight = 1; this.dispatchEvent(new Event('load'));
      })();
      imageJobs.push(job); job.catch(() => {});
    }
    get src() {return this._src;}
  }
  let timer;
  try {
    return await Promise.race([(async () => {
      const app = await import(url), observe = await app.run(runtimeText);
      await Promise.all(imageJobs);
      return {result: observe(), requests, images: images.sort()};
    })(), new Promise((_, reject) => {timer = setTimeout(() => reject(new Error('VRML loader execution timed out')), 20000);})]);
  } finally {
    clearTimeout(timer); hooks.deregister();
    for (const [key, value] of Object.entries(saved)) {if (value === undefined) delete globalThis[key]; else globalThis[key] = value;}
  }
}

test('VRML scenes, cameras and textures preserve native behavior after relocation and offline packing', async () => {
  const resources = {'site/models/nested/scene #.wrl': Buffer.from(scene()), 'site/models/nested/default.wrl': Buffer.from(scene()),
    'site/models/invalid.wrl': Buffer.from('#VRML V1.0 ascii\nGroup {}'),
    'site/textures/paint #.png': pixel, 'site/models/nested/paint #.png': pixel};
  for (const name of ['meshWithTexture.wrl', 'camera.wrl', 'house.wrl', 'extrusion1.wrl', 'pixelTexture.wrl', 'map.gif'])
    resources['site/oracle/' + name] = fs.readFileSync(new URL('examples/models/vrml/' + name, upstream));
  const source = `import {VRMLLoader} from ${JSON.stringify(vrml)};
    const describe=scene=>{scene.updateMatrixWorld(true);const nodes=[];scene.traverse(node=>{
      const item={type:node.type,name:node.name,children:node.children.length,matrix:node.matrixWorld.toArray()};
      if(node.isCamera)item.camera={near:node.near,far:node.far,projection:node.projectionMatrix.toArray()};
      if(node.geometry)item.geometry={attributes:Object.fromEntries(Object.entries(node.geometry.attributes).map(([name,a])=>
        [name,{size:a.itemSize,normalized:a.normalized,values:[...a.array]}])),index:node.geometry.index?[...node.geometry.index.array]:null};
      if(node.material)item.material={type:node.material.type,color:node.material.color.toArray(),opacity:node.material.opacity,
        side:node.material.side,shininess:node.material.shininess,map:node.material.map?{
          repeat:node.material.map.repeat.toArray(),offset:node.material.map.offset.toArray(),rotation:node.material.map.rotation,
          wrap:[node.material.map.wrapS,node.material.map.wrapT],colorSpace:node.material.map.colorSpace,
          image:node.material.map.image.data?[...node.material.map.image.data]:{complete:node.material.map.image.complete,bytes:node.material.map.image.byteLength}}:null};
      nodes.push(item);
    });return {nodes,worldInfo:scene.userData.worldInfo};};
    export async function run(runtimeText){
      const sequence=[],progress=[],loader=new VRMLLoader().setPath('models/');let returned;
      const pending=new Promise((resolve,reject)=>{returned=loader.load('nested/scene%20%23.wrl?edition=2#scene',
        scene=>{sequence.push('loaded');resolve(scene);},event=>progress.push([event.type,event.loaded]),reject);sequence.push('returned');});
      loader.setPath('later/');loader.setResourcePath('textures/');
      const first=await pending;
      const second=await new VRMLLoader().setPath('models/').setResourcePath('textures/').loadAsync('nested/scene%20%23.wrl?edition=2#scene');
      const fallback=await new VRMLLoader().loadAsync('models/nested/default.wrl');
      const model=await new VRMLLoader().setPath('oracle/').loadAsync('meshWithTexture.wrl');
      const cameras=await new VRMLLoader().setPath('oracle/').loadAsync('camera.wrl');
      const house=await new VRMLLoader().setPath('oracle/').loadAsync('house.wrl');
      const extrusion=await new VRMLLoader().setPath('oracle/').loadAsync('extrusion1.wrl');
      const pixels=await new VRMLLoader().setPath('oracle/').loadAsync('pixelTexture.wrl');
      const parsed=new VRMLLoader().parse(runtimeText,'textures/');
      let failure;try{await new VRMLLoader().loadAsync('models/invalid.wrl');}catch(error){failure={name:error.name,message:error.message};}
      return ()=>{
        const states=[first,second,fallback,model,cameras,house,extrusion,pixels,parsed].map(describe),a=first.children[0],b=second.children[0];
        const use=first.children[1].children[0],clonedUse=use!==a&&use.geometry===a.geometry&&use.material===a.material;
        a.geometry.attributes.position.setX(0,99);a.material.map.offset.x=99;
        const independent=first!==second&&a.geometry!==b.geometry&&a.material!==b.material&&a.material.map!==b.material.map&&
          b.geometry.attributes.position.getX(0)===0&&b.material.map.offset.x!==99;
        return {sequence,progress,returnedUndefined:returned===undefined,states,clonedUse,independent,failure};
      };
    }`;
  const sourceDir = fixture({'index.html': '<base href="./site/"><script type="module" src="../app.mjs"></script>',
    'app.mjs': source, ...resources}), base = fixture({}), entry = path.join(sourceDir, 'index.html');
  const runtimeText = scene().replace('Retained scene', 'Runtime input').replace('1 0 0, 0 1 0', '2 0 0, 0 2 0');
  let servedRoot = sourceDir;
  const server = http.createServer((request, response) => {
    try {response.end(fs.readFileSync(path.join(servedRoot, decodeURIComponent(new URL(request.url, 'http://localhost').pathname))));}
    catch {response.writeHead(404); response.end('Missing resource');}
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const documentUrl = `http://127.0.0.1:${server.address().port}/index.html`;
    const reference = await execute(pathToFileURL(path.join(sourceDir, 'app.mjs')).href, {}, documentUrl, runtimeText);
    const networked = await buildApplication(entry, path.join(base, 'networked'));
    const prepared = await buildApplication(entry, path.join(base, 'prepared'), {prepareStandaloneLoaders: true});
    for (const output of [networked.outDir, prepared.outDir]) for (const [name, bytes] of Object.entries(resources))
      assert.deepEqual(fs.readFileSync(path.join(output, name)), bytes, name + ' stays byte-identical');
    const packedFile = path.join(base, 'packed.html'); packHtml(path.join(prepared.outDir, 'index.html'), packedFile);
    const html = fs.readFileSync(packedFile, 'utf8');
    const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
    const packedEntry = /<script type="module" src="([^"]+)"/.exec(html)[1];
    fs.renameSync(sourceDir, sourceDir + '-unavailable'); fs.renameSync(prepared.outDir, prepared.outDir + '-unavailable');
    const deployed = path.join(base, 'deployed'); fs.renameSync(networked.outDir, deployed); servedRoot = deployed;
    const relocated = await execute(pathToFileURL(path.join(deployed, networked.entryFiles[0])).href, {}, documentUrl, runtimeText);
    fs.renameSync(deployed, deployed + '-unavailable');
    const packed = await execute(packedEntry, imports, 'https://relocated.example/index.html', runtimeText, true);
    assert.deepEqual(relocated.result, reference.result, 'relocated geometry, cameras, materials, aliases, callbacks and native errors match source');
    assert.deepEqual(packed.result, reference.result, 'offline retained parser results match source with original/build assets unavailable');
    assert.deepEqual(packed.images, reference.images); assert.deepEqual(relocated.images, reference.images);
    assert.equal(packed.images.length, 5);
    assert.ok(packed.images.includes(hash(resources['site/oracle/map.gif'])));
    assert.deepEqual(packed.result.sequence, ['returned', 'loaded']);
    assert.equal(packed.result.returnedUndefined, true); assert.equal(packed.result.independent, true); assert.equal(packed.result.clonedUse, true);
    assert.deepEqual(packed.result.failure, {name: 'Error', message: 'THREE.VRMLLexer: Version of VRML asset not supported.'});
    assert.ok(packed.result.states[4].nodes.some(node => node.type === 'PerspectiveCamera'));
    assert.ok(packed.result.states[4].nodes.some(node => node.type === 'OrthographicCamera'));
    assert.ok(packed.result.states[5].nodes.length > 10);
    assert.ok(packed.requests.every(uri => uri.startsWith('data:')));
    assert.ok(relocated.requests.some(uri => uri.includes('/site/textures/paint%20%23.png?revision=2')));
    assert.ok(relocated.requests.some(uri => uri.includes('/site/models/nested/paint%20%23.png?revision=2')));
  } finally {await new Promise(resolve => server.close(resolve));}
});

