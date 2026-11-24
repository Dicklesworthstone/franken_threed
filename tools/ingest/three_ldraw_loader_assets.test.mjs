import assert from 'node:assert/strict';
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
import {collectLDrawPartAssets, ldrawPartCandidates, ldrawPartReferences} from './pack_ldraw.mjs';
import {collectThreeLoaderAssets, findThreeLoaderRequests} from './three_loader_assets.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url).href;
const core = upstream + 'build/three.core.js';
const three = upstream + 'build/three.module.js';
const ldraw = upstream + 'examples/jsm/loaders/LDrawLoader.js';
const conditional = upstream + 'examples/jsm/materials/LDrawConditionalLineMaterial.js';
const discover = source => findThreeLoaderRequests(parse(source, {ecmaVersion: 'latest', sourceType: 'module'}),
  {resolveModule: specifier => specifier === 'three' ? core : specifier});
const part = (name, x = 0) => `1 16 ${x} 0 0 1 0 0 0 1 0 0 0 1 ${name}\n`;
const triangle = '0 !LDRAW_ORG Primitive\n0 BFC CERTIFY CCW\n3 16 0 0 0 1 0 0 0 1 0\n';
function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-ldraw-assets-'));
  for (const [name, bytes] of Object.entries(files)) {
    const file = path.join(root, name); fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, bytes);
  }
  return root;
}

test('LDraw library discovery closes live pending paths and confined file maps without guessing opaque owners', () => {
  const requests = discover(`import {LDrawLoader} from ${JSON.stringify(ldraw)};
    const loader=new LDrawLoader().setPartsLibraryPath('before/').setFileMap({'Alias.dat':'first.dat'});
    loader.setPath('models/'); loader.loadAsync('scene.ldr');
    loader.setPartsLibraryPath(choice?'after/':'fallback/'); loader.setFileMap({'Alias.dat':'last.dat'});
    loader.load('second.ldr'); loader.preloadMaterials('palette.ldr');
    const dynamic=new LDrawLoader(); dynamic.load('dynamic.ldr'); dynamic.setPartsLibraryPath(runtimePath);
    const alias={'Alias.dat':'part.dat'}, opaque=new LDrawLoader().setFileMap(alias); opaque.load('opaque.ldr');
    const getter=new LDrawLoader().setFileMap({get part(){return runtimeName;}});getter.load('getter.ldr');
    const prototype=new LDrawLoader().setFileMap({__proto__: runtimeMap});prototype.load('prototype.ldr');`);
  for (const request of requests.slice(0, 2)) {
    assert.deepEqual(request.partPaths, ['before/', 'after/', 'fallback/']);
    assert.deepEqual(request.partFileMaps, [{'Alias.dat': ['first.dat']}, {'Alias.dat': ['last.dat']}]);
  }
  assert.equal(requests[2].url, 'models/palette.ldr');
  assert.equal(requests[2].partPaths, undefined, 'material preload does not load model parts');
  assert.deepEqual(requests.slice(3).map(({url, partPaths}) => [url, partPaths]),
    [['dynamic.ldr', null], ['opaque.ldr', null], ['getter.ldr', null], ['prototype.ldr', null]]);
});

test('LDraw metadata and search order match the pinned parser, embedded cache and all twelve failed attempts', async () => {
  const hooks = registerHooks({resolve(specifier, context, next) {return next(specifier === 'three' ? three : specifier, context);}});
  const saved = {fetch: globalThis.fetch, Request: globalThis.Request};
  try {
    const {LDrawLoader} = await import(ldraw);
    const text = '0 FILE Scene.ldr\r\n' + part('Alias.dat') + part('s\\detail.dat') + part('48\\curve.dat') +
      part('MiXeD/part%20%23.DAT?rev=2#part') + '0 FILE Internal.DAT\n0 !LDRAW_ORG Part\n' + part('Nested.dat') +
      '0 FILE nested.dat\n' + triangle;
    const map = {'Alias.dat': 'Internal.DAT'}, cache = new LDrawLoader().setFileMap(map).partsCache.parseCache;
    const parsed = cache.parse(text), all = [parsed, ...Object.values(cache._cache)];
    const native = [...new Set(all.flatMap(value => value.subobjects.map(object => object.fileName)))]
      .filter(name => !(name.toLowerCase() in cache._cache));
    assert.deepEqual(ldrawPartReferences(Buffer.from(text), [{'Alias.dat': ['Internal.DAT']}]).sort(), native.sort());
    assert.deepEqual(native.sort(), ['parts/s/detail.dat', 'p/48/curve.dat', 'MiXeD/part%20%23.DAT?rev=2#part'].sort());
    const attempts = [];
    globalThis.Request = class extends saved.Request {
      constructor(input, options) {super(new URL(input, 'https://source.example/'), options);}
    };
    globalThis.fetch = async request => {attempts.push(new URL(request.url).pathname.slice(1)); throw new Error('Missing part');};
    await assert.rejects(cache.fetchData('MiXeD/Part.DAT'), {message: 'THREE.LDrawLoader: Subobject "mixed/part.dat" could not be loaded.'});
    assert.deepEqual(attempts, ldrawPartCandidates('MiXeD/Part.DAT'));
    assert.equal(attempts.length, 12);
  } finally {hooks.deregister(); Object.assign(globalThis, saved);}
});

test('LDraw recursive closure preserves bytes, encoded routes, standardized folders and MPD-local identities', () => {
  const files = {
    'models/scene #.ldr': '0 !LDRAW_ORG Model\n' + part('Alias.DAT'),
    'library/parts/assembly.dat': '0 !LDRAW_ORG Part\n' + part('s\\detail.dat') + part('48\\curve.dat') +
      part('Branch/Part.DAT') + part('Mapped.dat') + part('Internal.DAT') + '0 FILE internal.dat\n' + triangle,
    'library/parts/s/detail.dat': triangle,
    'library/p/48/curve.dat': triangle,
    'library/branch/branch/part.dat': triangle,
    'library/custom #.dat': triangle,
    'library/parts/internal.dat': 'must stay outside the MPD cache closure',
  };
  const root = fixture({'pages/index.html': '', ...files});
  const requests = discover(`import {LDrawLoader} from ${JSON.stringify(ldraw)};
    new LDrawLoader().setPath('/models/').setPartsLibraryPath('/library/')
      .setFileMap({'Alias.DAT':'Assembly.DAT','Mapped.dat':'custom%20%23.dat?rev=2#part'}).load('scene%20%23.ldr?version=2#model');`);
  const logical = [], assets = collectThreeLoaderAssets(requests, {documentRoot: root,
    documentUrl: pathToFileURL(path.join(root, 'pages/index.html')).href, onResource: (_name, url) => logical.push(url)});
  assert.deepEqual([...assets.keys()].sort(), Object.keys(files).filter(name => !name.endsWith('/internal.dat')).sort());
  for (const [name, asset] of assets) assert.equal(asset.source.toString('utf8'), files[name]);
  assert.ok(logical.some(url => url.endsWith('/library/custom%20%23.dat?rev=2#part')));
  assert.ok(logical.some(url => url.endsWith('/models/scene%20%23.ldr?version=2#model')));
  assert.ok(!logical.some(url => url.endsWith('/library/parts/internal.dat')));
});

test('LDraw repeated and cyclic metadata is bounded without modifying malformed source bytes', () => {
  assert.throws(() => ldrawPartReferences({length: 64 * 1024 * 1024 + 1}), {code: 'LDRAW_BUDGET'});
  const first = {originalFileName: 'a.dat', source: Buffer.from(part('b.dat'))};
  const second = {originalFileName: 'b.dat', source: Buffer.from(part('a.dat'))};
  let reads = 0;
  collectLDrawPartAssets(first, {partPaths: [''], partFileMaps: [{}]}, name => {
    reads++; return name === 'parts/a.dat' ? first : name === 'parts/b.dat' ? second : null;
  });
  assert.equal(reads, 24, 'each physical file is inspected once even if references form a cycle');
  const invalid = Buffer.from('6 invalid\n');
  assert.deepEqual(ldrawPartReferences(invalid), []); assert.equal(invalid.toString(), '6 invalid\n');
});

// The pinned source, native HTTP/data fetch, Request and stream handling run.
// Only the browser document base, import map and progress-event host are supplied.
// Offline misses reject on the existing host route, so source retries still run.
async function execute(entry, imports, documentUrl, offline = false) {
  const saved = Object.fromEntries(['document', 'Request', 'ProgressEvent', 'fetch'].map(key => [key, globalThis[key]]));
  const hooks = registerHooks({resolve(specifier, context, next) {
    return next(imports[specifier] ?? (specifier === 'three' ? three : specifier), context);
  }});
  const successful = [], attempted = [], events = [], nativeFetch = saved.fetch;
  globalThis.document = {URL: documentUrl, baseURI: new URL('./', documentUrl).href};
  globalThis.Request = class extends saved.Request {
    constructor(input, options) {super(typeof input === 'string' ? new URL(input, document.baseURI) : input, options);}
  };
  globalThis.ProgressEvent = class extends Event {constructor(type, details) {super(type); Object.assign(this, details);}};
  globalThis.fetch = async (input, options) => {
    const uri = input instanceof Request ? input.url : String(input); attempted.push(uri);
    if (offline && !uri.startsWith('data:')) throw new Error('Offline missing resource');
    const response = await nativeFetch(input, options);
    if (response.ok) successful.push(uri);
    return response;
  };
  let manager;
  try {
    const app = await import(entry); manager = app.THREE.DefaultLoadingManager;
    manager.onProgress = url => events.push(['end', url]); manager.onError = url => events.push(['error', url]);
    return {result: await app.run(), successful, attempted, events};
  } finally {
    if (manager) {manager.onProgress = undefined; manager.onError = undefined;}
    hooks.deregister();
    for (const [key, value] of Object.entries(saved)) {if (value === undefined) delete globalThis[key]; else globalThis[key] = value;}
  }
}

test('unpacked LDraw assemblies run from relocated builds and packed bytes with native geometry, caches, retries and errors', async () => {
  const resources = {
    'models/scene #.ldr': '0 !LDRAW_ORG Model\n0 !CATEGORY Vehicles\n' + part('Alias.DAT') + '0 STEP\n' + part('Alias.DAT', 5),
    'library/parts/assembly.dat': '0 !LDRAW_ORG Part\n' + part('s\\detail.dat') + part('48\\curve.dat', 2) +
      part('Branch/Part.DAT', 4) + part('Mapped.dat', 6) + part('Internal.DAT', 8) + '0 FILE internal.dat\n' + triangle,
    'library/parts/s/detail.dat': triangle,
    'library/p/48/curve.dat': triangle,
    'library/branch/branch/part.dat': triangle,
    'library/custom #.dat': triangle,
    'models/missing.ldr': '0 !LDRAW_ORG Model\n' + part('Missing.DAT'),
    'models/invalid.ldr': '6 invalid\n',
  };
  const map = `{'Alias.DAT':'Assembly.DAT','Mapped.dat':'custom%20%23.dat?revision=2#part'}`;
  const source = `import * as THREE from ${JSON.stringify(core)}; export {THREE};
    import {LDrawLoader} from ${JSON.stringify(ldraw)};
    import {LDrawConditionalLineMaterial} from ${JSON.stringify(conditional)};
    function describe(group){const meshes=[];group.traverse(value=>{if(value.isMesh)meshes.push({
      positions:[...value.geometry.attributes.position.array],normals:[...value.geometry.attributes.normal.array],
      material:value.material.name,color:value.material.color.getHexString()});});
      return {meshes,steps:group.userData.numBuildingSteps,category:group.userData.category,
        children:group.children.map(value=>({name:value.name,position:value.position.toArray(),step:value.userData.buildingStep}))};}
    export async function run(){
      const order=[],loader=new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial)
        .setPath('models/').setPartsLibraryPath('library/').setFileMap(${map});
      let returned;const first=await new Promise((resolve,reject)=>{returned=loader.load('scene%20%23.ldr?revision=2#model',
        value=>{order.push('loaded');resolve(value);},undefined,reject);order.push('returned');});
      const late=new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial)
        .setPath('models/').setPartsLibraryPath('before/').setFileMap({'Alias.DAT':'wrong.dat'});
      const pending=late.loadAsync('scene%20%23.ldr?revision=2#model');
      late.setPartsLibraryPath('library/');late.setFileMap(${map});const second=await pending;
      const failures=[];
      try{await new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial).setPartsLibraryPath('library/')
        .loadAsync('models/missing.ldr');}catch(error){failures.push(error.message);}
      try{await new LDrawLoader().setConditionalLineMaterial(LDrawConditionalLineMaterial).loadAsync('models/invalid.ldr');}
      catch(error){failures.push(error.message);}
      const states=[describe(first),describe(second)];let a,b;
      first.traverse(value=>{if(!a&&value.isMesh)a=value;});second.traverse(value=>{if(!b&&value.isMesh)b=value;});
      const before=b.geometry.attributes.position.getX(0);a.geometry.attributes.position.setX(0,99);
      return {states,order,undefinedReturn:returned===undefined,failures,
        independent:a!==b&&a.geometry!==b.geometry&&b.geometry.attributes.position.getX(0)===before};
    }`;
  const sourceDir = fixture({'index.html': '<script type="module" src="app.mjs"></script>', 'app.mjs': source, ...resources});
  const out = fixture({}); let servedRoot = sourceDir;
  const server = http.createServer((request, response) => {
    try {response.end(fs.readFileSync(path.join(servedRoot, decodeURIComponent(new URL(request.url, 'http://localhost').pathname))));}
    catch {response.writeHead(404); response.end('Missing resource');}
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const documentUrl = `http://127.0.0.1:${server.address().port}/index.html`;
    const reference = await execute(pathToFileURL(path.join(sourceDir, 'app.mjs')).href, {}, documentUrl);
    const networked = await buildApplication(path.join(sourceDir, 'index.html'), path.join(out, 'networked'));
    const prepared = await buildApplication(path.join(sourceDir, 'index.html'), path.join(out, 'prepared'), {prepareStandaloneLoaders: true});
    for (const build of [networked, prepared]) for (const [name, bytes] of Object.entries(resources))
      assert.equal(fs.readFileSync(path.join(build.outDir, name), 'utf8'), bytes, name + ' remains byte-identical');
    const packedPath = path.join(out, 'packed.html'); packHtml(path.join(prepared.outDir, 'index.html'), packedPath);
    const html = fs.readFileSync(packedPath, 'utf8');
    const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
    const packedEntry = /<script type="module" src="([^"]+)"/.exec(html)[1];
    fs.renameSync(sourceDir, sourceDir + '-unavailable'); fs.renameSync(prepared.outDir, prepared.outDir + '-unavailable');
    const deployed = path.join(out, 'deployed'); fs.renameSync(networked.outDir, deployed); servedRoot = deployed;
    const relocated = await execute(pathToFileURL(path.join(deployed, networked.entryFiles[0])).href, {}, documentUrl);
    fs.renameSync(deployed, deployed + '-unavailable');
    const packed = await execute(packedEntry, imports, 'https://relocated.example/index.html', true);
    assert.deepEqual(relocated.result, reference.result); assert.deepEqual(packed.result, reference.result);
    // Independent sibling parts complete concurrently. Compare each logical
    // request's lifecycle, not the transport-dependent sibling interleaving.
    const lifecycle = events => {
      const byUrl = new Map();
      for (const [kind, url] of events) {if (!byUrl.has(url)) byUrl.set(url, []); byUrl.get(url).push(kind);}
      return [...byUrl].sort(([a], [b]) => a.localeCompare(b));
    };
    assert.deepEqual(lifecycle(relocated.events), lifecycle(reference.events));
    assert.deepEqual(lifecycle(packed.events), lifecycle(reference.events));
    assert.deepEqual(reference.result.order, ['returned', 'loaded']);
    assert.equal(reference.result.undefinedReturn, true); assert.equal(reference.result.independent, true);
    assert.equal(reference.result.states[0].meshes.length, 2);
    assert.equal(reference.result.states[0].meshes[0].positions.length, 45);
    assert.equal(reference.result.states[0].steps, 2);
    assert.deepEqual(reference.result.failures, ['THREE.LDrawLoader: Subobject "missing.dat" could not be loaded.',
      'THREE.LDrawLoader: Unknown line type "6" at line 1.']);
    assert.ok(packed.successful.length >= 12);
    assert.ok(packed.successful.every(uri => uri.startsWith('data:')), 'all successful responses use embedded bytes');
    assert.ok(packed.events.some(([kind, url]) => kind === 'error' && url === 'library/parts/Assembly.DAT'));
    assert.ok(packed.events.some(([kind, url]) => kind === 'end' && url === 'library/parts/assembly.dat'));
    assert.ok(packed.events.some(([kind, url]) => kind === 'end' && url === 'library/branch/branch/part.dat'));
  } finally {await new Promise(resolve => server.close(resolve));}
});
