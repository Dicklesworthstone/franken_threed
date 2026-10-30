import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { parse } from "acorn";
import { buildApplication } from "./build_application.mjs";
import { packHtml } from "./pack_html.mjs";
import { collectThreeLoaderAssets, findThreeLoaderRequests } from "./three_loader_assets.mjs";

const rootUrl = new URL("../../upstream/three.js/", import.meta.url).href;
const coreUrl = rootUrl + "build/three.core.js";
const gltfUrl = rootUrl + "examples/jsm/loaders/GLTFLoader.js";
const dracoUrl = rootUrl + "examples/jsm/loaders/DRACOLoader.js";
const ktx2Url = rootUrl + "examples/jsm/loaders/KTX2Loader.js";
const mtlUrl = rootUrl + "examples/jsm/loaders/MTLLoader.js";
const objUrl = rootUrl + "examples/jsm/loaders/OBJLoader.js";
const discover = (code, options = {}) => findThreeLoaderRequests(
  parse(code, { ecmaVersion: "latest", sourceType: "module" }),
  { resolveModule: (specifier) => specifier === "three" ? coreUrl : specifier, ...options },
);

test("discovers H2-style const cube-face arrays and literal textures through pinned namespace identity", () => {
  const requests = discover(`import * as THREE from 'three';
    function generateMaterials() {
      const path = 'textures/cube/SwedishRoyalCastle/', format = '.jpg';
      const urls = [path+'px'+format,path+'nx'+format,path+'py'+format,path+'ny'+format,path+'pz'+format,path+'nz'+format];
      const loader = new THREE.CubeTextureLoader();
      loader.load(urls); loader.load(urls);
      new THREE.TextureLoader().load('textures/uv_grid_opengl.jpg');
    }`);
  assert.deepEqual([...new Set(requests.map((request) => request.url))], [
    ...["px", "nx", "py", "ny", "pz", "nz"].map((face) => `textures/cube/SwedishRoyalCastle/${face}.jpg`),
    "textures/uv_grid_opengl.jpg",
  ]);
});

test("follows named constructor aliases, finite string choices, templates and straight-line loader settings", () => {
  const requests = discover(`import { FileLoader as SourceLoader } from 'three';
    const Loader = SourceLoader;
    function load(which) {
      const loader = new Loader();
      loader.setPath('assets/');
      const extension = which ? '.bin' : '.dat';
      loader.loadAsync(\`file\${extension}\`);
      loader.setPath('discarded/').setCrossOrigin('anonymous').setPath('other/');
      loader.load('final.txt');
      new Loader().setPath('chained/').load('third.txt');
    }`);
  assert.deepEqual(requests.map((request) => request.url), [
    "assets/file.bin", "assets/file.dat", "other/final.txt", "chained/third.txt",
  ]);
});

test("does not infer resources from foreign names, shadowed imports, mutable arrays or escaping loaders", () => {
  const requests = discover(`import * as THREE from 'three';
    import { TextureLoader } from 'file:///unrelated/build/three.core.js';
    new TextureLoader().load('foreign.png');
    function shadow(THREE) { new THREE.TextureLoader().load('parameter.png'); }
    function hoisted() { new THREE.TextureLoader().load('var.png'); var THREE; }
    { const THREE = globalThis.other; new THREE.TextureLoader().load('block.png'); }
    const loader = new THREE.TextureLoader(); useElsewhere(loader); loader.load('escaped.png');
    const faces=['a','b','c','d','e','f']; faces.reverse(); new THREE.CubeTextureLoader().load(faces);
    const conditional = new THREE.TextureLoader();
    if (globalThis.enabled) conditional.setPath('maybe/'); conditional.load('conditional.png');
    const argument = new THREE.TextureLoader();
    argument.load(argument.setPath('args/') ? 'yes.png' : 'no.png');
    const nestedArgument = new THREE.TextureLoader();
    nestedArgument.setCrossOrigin(nestedArgument.setPath('nested-args/') ? 'anonymous' : '').load('nested.png');
    const escapedSetting = new THREE.TextureLoader();
    useElsewhere(escapedSetting.setCrossOrigin('anonymous')); escapedSetting.load('escaped-setting.png');
    const escapedPath = new THREE.TextureLoader();
    useElsewhere(escapedPath.setPath('escape/')); escapedPath.load('escaped-path.png');
    const repeated = new THREE.TextureLoader();
    for (let i=0;i<2;i++) { repeated.load('repeated.png'); repeated.setPath('next-iteration/'); }
    new THREE.TextureLoader(globalThis.manager).load('managed.png');
    new THREE.TextureLoader().load('real.png');`);
  assert.deepEqual(requests.map((request) => request.url), ["real.png"]);
});

test("fluent GLTFLoader path and resource settings retain their final values", () => {
  const requests = discover(`import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
    const loader = new GLTFLoader();
    loader.setPath('first/').setPath('models/').setResourcePath('old/').setResourcePath('textures/');
    loader.load('scene.gltf');`);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "models/scene.gltf");
  assert.equal(requests[0].resourcePath, "textures/");
});

test("pinned single-file addons retain finite paths, decoder settings and import identities", () => {
  const addons = rootUrl + "examples/jsm/Addons.js";
  const requests = discover(`import { HDRLoader as SourceHDR, RGBELoader, EXRLoader, FontLoader, STLLoader, PLYLoader, OBJLoader } from ${JSON.stringify(addons)};
    import * as HDR from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/HDRLoader.js')};
    import * as Addons from ${JSON.stringify(addons)};
    const Loader = SourceHDR;
    const radiance = new Loader(); radiance.setDataType(floatType).setPath('env/');
    radiance.loadAsync(useDay ? 'day.hdr' : 'night.hdr');
    new RGBELoader().setDataType(halfType).load('legacy.hdr');
    new EXRLoader().setPart(1).setOutputFormat(rgba).setDataType(floatType).setPath('exr/').loadAsync('sky.exr');
    const font = new FontLoader(); font.parse(inMemoryFont); font.load('font.json');
    new STLLoader().setPath('geometry/').load('mesh.stl');
    const ply = new PLYLoader();
    ply.setPropertyNameMapping({diffuse_red:'red'}); ply.setCustomPropertyNameMapping({custom:['alpha']}); ply.load('mesh.ply');
    new OBJLoader().loadAsync('mesh.obj');
    new HDR.HDRLoader().load('namespace.hdr');
    new Addons.FontLoader().load('namespace-font.json');`);
  assert.deepEqual(requests.map(({url}) => url), ['env/day.hdr', 'env/night.hdr', 'legacy.hdr',
    'exr/sky.exr', 'font.json', 'geometry/mesh.stl', 'mesh.ply', 'mesh.obj', 'namespace.hdr', 'namespace-font.json']);
});

test("addon asset discovery preserves opaque loaders and shadows while OBJ material setters retain its model URL", () => {
  const addon = name => JSON.stringify(rootUrl + `examples/jsm/loaders/${name}.js`);
  const requests = discover(`import { HDRLoader } from ${addon('HDRLoader')};
    import { OBJLoader } from ${addon('OBJLoader')};
    import { FBXLoader } from ${addon('FBXLoader')};
    import { HDRLoader as Foreign } from 'file:///other/examples/jsm/loaders/HDRLoader.js';
    import * as Wrong from ${addon('FontLoader')};
    function shadow(HDRLoader) { new HDRLoader().load('shadow.hdr'); }
    new Foreign().load('foreign.hdr'); new Wrong.HDRLoader().load('wrong-export.hdr');
    new HDRLoader(manager).load('managed.hdr');
    const escaped = new HDRLoader(); modify(escaped); escaped.load('escape.hdr');
    const changed = new HDRLoader(); changed.setPath(runtimePath); changed.load('dynamic.hdr');
    const nested = new HDRLoader(); nested.setDataType(nested.setPath('nested/') && floatType).load('nested.hdr');
    new OBJLoader().setMaterials(materials).load('materials.obj');
    new FBXLoader().load('sidecars.fbx');
    new HDRLoader().load('known.hdr');`);
  assert.deepEqual(requests.map(({url}) => url), ['materials.obj', 'known.hdr']);
});

test("MTL and configured OBJ loaders retain finite model paths, material settings and namespace identity", () => {
  const requests = discover(`import { MTLLoader as Materials } from ${JSON.stringify(mtlUrl)};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import { OBJLoader } from ${JSON.stringify(objUrl)};
    const loader = new Materials().setMaterialOptions({side:doubleSide,normalizeRGB:true});
    loader.setPath('materials/');
    loader.loadAsync(which ? 'day.mtl' : 'night.mtl');
    loader.parse(runtimeText, 'runtime/');
    new Addons.MTLLoader().setResourcePath('textures/').load('other/library.mtl');
    const object = new OBJLoader(); object.setMaterials(materials); object.setPath('models/');
    object.loadAsync('mesh.obj');
    new OBJLoader().setMaterials(materials).load('other.obj');`);
  assert.deepEqual(requests.map(({url}) => url), ['materials/day.mtl', 'materials/night.mtl',
    'other/library.mtl', 'models/mesh.obj', 'other.obj']);
  assert.deepEqual(requests.slice(0, 3).map(({texturePaths}) => texturePaths),
    [['materials/'], ['materials/'], ['textures/']]);
});

test("MTL pending resource resets retain each request's captured default path rather than a sticky base", () => {
  const requests = discover(`import { MTLLoader } from ${JSON.stringify(mtlUrl)};
    const loader = new MTLLoader();
    loader.load('first/library.mtl'); loader.loadAsync('second/library.mtl');
    loader.setResourcePath('shared/');
    loader.setPath('models/'); loader.load('nested/library.mtl');
    loader.setResourcePath('');
    loader.load(runtimeURL);
    const pending = new MTLLoader().setResourcePath('before/');
    pending.load('default/library.mtl'); pending.setResourcePath('');
    const changed = new MTLLoader(); changed.load('known/library.mtl'); changed.setResourcePath(runtimePath);
    new MTLLoader(manager).load('managed.mtl');
    const escaped = new MTLLoader(); customize(escaped); escaped.load('escaped.mtl');`);
  assert.deepEqual(requests.map(({url}) => url), ['first/library.mtl', 'second/library.mtl',
    'models/nested/library.mtl', 'default/library.mtl', 'known/library.mtl']);
  assert.deepEqual(requests.map(({texturePaths}) => texturePaths), [['first/', 'shared/'],
    ['second/', 'shared/'], ['models/', 'shared/'], ['before/', 'default/'], null]);
});

test("only the exact pinned MTL manager fallback is exempt from application manager escape detection", () => {
  const source = fs.readFileSync(new URL(mtlUrl), 'utf8');
  for (const [moduleUrl, code, expected] of [[mtlUrl, source, false],
    [mtlUrl, source + '\n// customized\n', true], ['file:///other/MTLLoader.js', source, true]]) {
    let exposed = false;
    const requests = discover(code, {moduleUrl, source: code, onDefaultManagerExposure: () => { exposed = true; }});
    assert.equal(exposed, expected);
    assert.deepEqual(requests, []);
  }
});

test("ObjectLoader retains sticky image bases and finite resets visible to pending loads", () => {
  const requests = discover(`import { ObjectLoader } from 'three';
    const loader = new ObjectLoader();
    loader.load('first/scene.json'); loader.loadAsync('second/scene.json');
    loader.setResourcePath('shared/'); loader.load('third/scene.json');
    loader.setResourcePath(''); loader.setPath('models/'); loader.load('nested/scene.json');
    new ObjectLoader().setPath('other/').load('nested/scene.json');`);
  assert.deepEqual(requests.map(({url}) => url), ['first/scene.json', 'second/scene.json',
    'third/scene.json', 'models/nested/scene.json', 'other/nested/scene.json']);
  const pendingBases = new Set(['first/', 'second/', 'shared/', 'models/', '']);
  for (const request of requests.slice(0, 4)) assert.deepEqual(new Set(request.imagePaths), pendingBases);
  assert.deepEqual(requests[4].imagePaths, ['other/']);
  const callbacks = discover(`import { ObjectLoader } from 'three'; const loader = new ObjectLoader();
    export const first = () => loader.loadAsync('a/scene.json');
    export const second = () => loader.loadAsync('b/scene.json');`);
  assert.ok(callbacks.every(request => request.imagePaths.join(',') === 'a/,b/'));
  const pending = discover(`import { ObjectLoader } from 'three';
    const loader = new ObjectLoader(); loader.loadAsync('models/scene.json'); loader.setResourcePath('late/');
    const reset = new ObjectLoader().setResourcePath('before/'); reset.load('models/scene.json'); reset.setResourcePath('');`);
  assert.deepEqual(new Set(pending[0].imagePaths), new Set(['models/', 'late/']));
  assert.deepEqual(new Set(pending[1].imagePaths), new Set(['before/', '']));
});

test("ObjectLoader unknown sticky bases and custom managers retain their host routes", () => {
  const requests = discover(`import { ObjectLoader } from 'three';
    const unknown = new ObjectLoader(); unknown.load(runtimeURL); unknown.load('known/scene.json');
    const fixed = new ObjectLoader().setPath('fixed/'); fixed.load(runtimeURL); fixed.load('scene.json');
    new ObjectLoader(manager).load('managed.json');
    const escaped = new ObjectLoader(); customize(escaped); escaped.load('escaped.json');`);
  assert.deepEqual(requests.map(({url, imagePaths}) => [url, imagePaths]), [
    ['known/scene.json', null], ['fixed/scene.json', ['fixed/']],
  ]);
  const pending = discover(`import { ObjectLoader } from 'three';
    const loader = new ObjectLoader(); loader.loadAsync('scene.json'); loader.setResourcePath(runtimePath);
    const changed = new ObjectLoader(); changed.load('scene.json'); changed.setPath(runtimePath); changed.setResourcePath(runtimeImages);`);
  assert.equal(pending.length, 2);
  assert.ok(pending.every(request => request.imagePaths === null), 'unknown later resets remain host-owned');
});

test("configured GLTF loaders retain model roots and source codec paths through fluent and separate setters", () => {
  const requests = discover(`import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
    import { DRACOLoader as Draco } from ${JSON.stringify(dracoUrl)};
    import { KTX2Loader } from ${JSON.stringify(ktx2Url)};
    const draco = new Draco();
    draco.setDecoderPath('discarded/').setWorkerLimit(2);
    const ktx = new KTX2Loader().setTranscoderPath('codecs/basis/').detectSupport(renderer);
    const loader = new GLTFLoader();
    loader.setDRACOLoader(draco).setKTX2Loader(ktx).setMeshoptDecoder(decoder);
    draco.setDecoderPath('codecs/draco/');
    loader.setPath('models/');
    loader.load('scene.gltf');
    new GLTFLoader().setMeshoptDecoder(decoder).setPath('other/').loadAsync('scene.glb');`);
  assert.deepEqual(requests.map(request => request.url), [
    "models/scene.gltf", "codecs/draco/draco_wasm_wrapper.js", "codecs/draco/draco_decoder.wasm",
    "codecs/draco/draco_decoder.js", "codecs/basis/basis_transcoder.js", "codecs/basis/basis_transcoder.wasm",
    "other/scene.glb",
  ]);
});

test("Draco packaging includes both source JS and Wasm branches, explicit URL dictionaries and preload", () => {
  for (const mode of ["js", "wasm"]) {
    const requests = discover(`import { DRACOLoader } from ${JSON.stringify(dracoUrl)};
      const loader = new DRACOLoader().setDecoderPath('codec-prefix-').setDecoderConfig({type:${JSON.stringify(mode)}});
      loader.setPath('models/'); loader.loadAsync('mesh.drc'); loader.dispose();`);
    assert.deepEqual(requests.map(request => request.url), [
      "models/mesh.drc", "codec-prefix-draco_wasm_wrapper.js", "codec-prefix-draco_decoder.wasm", "codec-prefix-draco_decoder.js",
    ], "decoder paths use the source setter's literal prefix, not URL directory resolution");
  }
  const requests = discover(`import { DRACOLoader } from ${JSON.stringify(dracoUrl)};
    const version = globalThis.next ? 'v2' : 'v1';
    new DRACOLoader().setDecoderPath({js:'codecs/'+version+'.js',wasm:'codecs/'+version+'.wasm'}).preload();
    new DRACOLoader().setDecoderPath('').preload();`);
  assert.deepEqual(requests.map(request => request.url), [
    "codecs/v2.js", "codecs/v1.js", "codecs/v2.wasm", "codecs/v1.wasm",
    "draco_wasm_wrapper.js", "draco_decoder.wasm", "draco_decoder.js",
  ]);
});

test("direct KTX2 requests and parse-time codecs retain paths while empty KTX2 paths use module defaults", () => {
  const requests = discover(`import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
    import { DRACOLoader } from ${JSON.stringify(dracoUrl)};
    import { KTX2Loader } from ${JSON.stringify(ktx2Url)};
    const ktx = new KTX2Loader();
    ktx.setTranscoderPath('basis/').setWorkerLimit(1).detectSupport(renderer);
    ktx.setPath('textures/'); ktx.loadAsync('albedo.ktx2'); ktx.dispose();
    new KTX2Loader().setTranscoderPath('').load('default.ktx2');
    new GLTFLoader().setDRACOLoader(new DRACOLoader().setDecoderPath('draco/')).parseAsync(data, 'models/');`);
  assert.deepEqual(requests.map(request => request.url), [
    "textures/albedo.ktx2", "basis/basis_transcoder.js", "basis/basis_transcoder.wasm", "default.ktx2",
    "draco/draco_wasm_wrapper.js", "draco/draco_decoder.wasm", "draco/draco_decoder.js",
  ]);
});

test("custom managers, unproved receivers, dynamic codec paths and escaping codecs retain their host routes", () => {
  const requests = discover(`import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
    import { DRACOLoader } from ${JSON.stringify(dracoUrl)};
    import { KTX2Loader } from ${JSON.stringify(ktx2Url)};
    import { DRACOLoader as Foreign } from 'file:///other/DRACOLoader.js';
    new Foreign().setDecoderPath('foreign/').load('foreign.drc');
    new DRACOLoader(manager).setDecoderPath('managed/').load('managed.drc');
    new GLTFLoader(manager).setDRACOLoader(new DRACOLoader().setDecoderPath('unused/')).load('managed.gltf');
    const escaped = new DRACOLoader().setDecoderPath('escaped/');
    arbitrary.setDRACOLoader(escaped); new GLTFLoader().setDRACOLoader(escaped).load('escaped.gltf');
    const conditional = new DRACOLoader();
    if (enabled) conditional.setDecoderPath('conditional/');
    new GLTFLoader().setDRACOLoader(conditional).load('conditional.gltf');
    new GLTFLoader().setDRACOLoader(new DRACOLoader().setDecoderPath(runtimePath)).load('dynamic.gltf');
    const dictionary = {js:'aliased.js',wasm:'aliased.wasm'};
    mutate(dictionary);
    new GLTFLoader().setDRACOLoader(new DRACOLoader().setDecoderPath(dictionary)).load('dictionary.gltf');
    const ktx = new KTX2Loader().setTranscoderPath('escaped-basis/');
    mutate(ktx); new GLTFLoader().setKTX2Loader(ktx).load('basis.gltf');
    const custom = new GLTFLoader(); custom.register(plugin); custom.load('plugin.gltf');`);
  assert.deepEqual(requests.map(request => request.url), [
    "escaped.gltf", "conditional.gltf", "dynamic.gltf", "dictionary.gltf", "basis.gltf",
  ], "a source GLTF codec setter does not change the model URL, even when the codec itself is unproved");
});

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-loader-assets-"));
  for (const [name, content] of Object.entries({ "package.json": '{"type":"module"}', ...files })) {
    const target = path.join(root, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, content);
  }
  return root;
}

test("MTL texture closure matches source map overrides, option parsing, encoded URLs and document bases", () => {
  const mtl = `# No material owns this first map\nmap_Kd unused.png
newmtl replaced
map_Kd discarded.png
newmtl painted
map_Kd stale.png
MAP_KD -o 0.25 0.5 0 -s 2 3 1 paint%20%23.png?version=2#face
map_Ks specular.png
map_Ke glow.png
norm normal.png
map_bump -bm 0.75 bump.png
bump ignored-bump.png
disp -mm -0.5 2 height.png
map_d alpha.png
map_Ka ignored-ambient.png
newmtl replaced
map_Kd replacement.png
newmtl remote
map_Kd https://example.test/live.png
newmtl unknown-option
map_Kd -clamp on literal.png
newmtl slash
map_Kd /slash.png
`;
  const images = ['paint #.png', 'specular.png', 'glow.png', 'normal.png', 'bump.png', 'height.png',
    'alpha.png', 'replacement.png', '-clamp on literal.png', 'slash.png'];
  const root = fixture({'index.html': '', 'site/models/nested/library.mtl': mtl,
    ...Object.fromEntries(images.map(name => [`site/models/${name}`, name])),
    ...Object.fromEntries(['unused.png','discarded.png','stale.png','ignored-bump.png','ignored-ambient.png']
      .map(name => [`site/models/${name}`, 'must not be loaded'])),
    'site/models/nested/paint #.png': 'wrong nested base',
  });
  const requests = discover(`import { MTLLoader } from ${JSON.stringify(mtlUrl)};
    new MTLLoader().setPath('models/').load('nested/library.mtl');`);
  const resources = [], assets = collectThreeLoaderAssets(requests, {
    documentUrl: pathToFileURL(path.join(root, 'index.html')).href,
    documentBaseUrl: pathToFileURL(path.join(root, 'site/')).href,
    onResource: (name, url) => resources.push({name, url}),
  });
  assert.deepEqual([...assets.keys()].sort(), ['site/models/nested/library.mtl',
    ...images.map(name => `site/models/${name}`)].sort());
  assert.equal(assets.get('site/models/nested/library.mtl').source.toString(), mtl);
  assert.ok(resources.some(({url}) => url.endsWith('/models/paint%20%23.png?version=2#face')));
});

test("ObjectLoader collects encoded image and cube sidecars without changing JSON, data images or host URLs", () => {
  const scene = JSON.stringify({metadata: {type: 'Object'}, images: [
    {uuid: 'paint', url: 'paint%20%23.png?version=2#surface'},
    {uuid: 'cube', url: ['faces/px.png', 'faces/nx.png', {data: [0, 1, 2, 255], width: 1, height: 1, type: 'Uint8Array'}]},
    {uuid: 'embedded', url: 'data:image/png;base64,AA=='},
    {uuid: 'remote', url: 'https://example.test/live.png'},
    {uuid: 'relative', url: '/slash.png'},
  ]});
  const root = fixture({'index.html': '', 'site/models/nested/scene.json': scene,
    'site/models/paint #.png': 'paint', 'site/models/faces/px.png': 'px', 'site/models/faces/nx.png': 'nx',
    'site/models/slash.png': 'slash', 'site/models/nested/paint #.png': 'wrong model-directory base',
    'site/shared/paint #.png': 'override', 'site/shared/faces/px.png': 'shared-px',
    'site/shared/faces/nx.png': 'shared-nx', 'site/shared/slash.png': 'shared-slash',
    'site/models/bad.json': '{invalid source JSON'});
  const requests = discover(`import { ObjectLoader } from 'three';
    new ObjectLoader().setPath('models/').load('nested/scene.json');
    new ObjectLoader().setPath('models/').setResourcePath('shared/').loadAsync('nested/scene.json');
    new ObjectLoader().setPath('models/').load('bad.json');`);
  const resources = [], assets = collectThreeLoaderAssets(requests, {
    documentUrl: pathToFileURL(path.join(root, 'index.html')).href,
    documentBaseUrl: pathToFileURL(path.join(root, 'site/')).href,
    fail: (code, message) => { throw new Error(`${code}: ${message}`); },
    onResource: (name, url) => resources.push({name, url}),
  });
  assert.deepEqual([...assets.keys()].sort(), ['site/models/nested/scene.json', 'site/models/paint #.png',
    'site/models/faces/px.png', 'site/models/faces/nx.png', 'site/models/slash.png',
    'site/shared/paint #.png', 'site/shared/faces/px.png', 'site/shared/faces/nx.png',
    'site/shared/slash.png', 'site/models/bad.json'].sort());
  assert.equal(assets.get('site/models/nested/scene.json').source.toString(), scene);
  assert.equal(assets.get('site/models/bad.json').source.toString(), '{invalid source JSON');
  assert.ok(resources.some(({url}) => url.endsWith('/models/paint%20%23.png?version=2#surface')));
});

test("built FileLoader and GLTFLoader fetch original document-base URLs and external geometry bytes", async () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const model = JSON.stringify({
    asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    buffers: [{ uri: "geometry%20%23.bin?cache=1#positions", byteLength: positions.byteLength }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
  });
  const root = fixture({
    "index.html": '<base href="./site/"><link rel="icon" href="resources/message%20%23.txt"><script type="module" src="../modules/app.mjs"></script>',
    "modules/app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)};
      import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
      import { DRACOLoader } from ${JSON.stringify(dracoUrl)};
      import { KTX2Loader } from ${JSON.stringify(ktx2Url)};
      export async function load() {
        const loader = new FileLoader(); loader.setPath('resources/');
        const text = await loader.loadAsync('message%20%23.txt?edition=2#part');
        const draco = new DRACOLoader().setDecoderPath('codecs/draco/');
        const ktx = new KTX2Loader(); ktx.setTranscoderPath('codecs/basis/');
        const gltf = await new GLTFLoader().setDRACOLoader(draco).setKTX2Loader(ktx)
          .setPath('models/').setResourcePath('shared/').loadAsync('triangle.gltf?version=3#scene');
        return { text, positions: [...gltf.scene.children[0].geometry.attributes.position.array] };
      }`,
    "site/resources/message #.txt": "document-relative bytes",
    "modules/resources/message #.txt": "wrong module-relative bytes",
    "site/models/triangle.gltf": model,
    "site/shared/geometry #.bin": Buffer.from(positions.buffer),
  });
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  assert.equal(fs.readFileSync(path.join(built.outDir, "site/models/triangle.gltf"), "utf8"), model);
  const seen = [];
  const server = http.createServer((request, response) => {
    seen.push(request.url);
    try { response.end(fs.readFileSync(path.join(built.outDir, decodeURIComponent(new URL(request.url, "http://localhost").pathname)))); }
    catch { response.writeHead(404); response.end("missing resource"); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const documentBase = `http://127.0.0.1:${server.address().port}/site/`;
  const OriginalRequest = globalThis.Request, OriginalProgressEvent = globalThis.ProgressEvent;
  // Node's Request lacks the browser document-base resolution used by the real
  // loaders. Supply only that host primitive; all loading/parsing code is built
  // from the unmodified pinned upstream implementation and uses real HTTP.
  globalThis.Request = class extends OriginalRequest {
    constructor(input, options) { super(typeof input === "string" ? new URL(input, documentBase) : input, options); }
  };
  globalThis.ProgressEvent = class extends Event {
    constructor(type, details) { super(type); Object.assign(this, details); }
  };
  try {
    const application = await import(pathToFileURL(path.join(built.outDir, built.entryFiles[0])).href);
    assert.deepEqual(await application.load(), { text: "document-relative bytes", positions: [...positions] });
    assert.deepEqual(seen, [
      "/site/resources/message%20%23.txt?edition=2",
      "/site/models/triangle.gltf?version=3",
      "/site/shared/geometry%20%23.bin?cache=1",
    ]);
  } finally {
    globalThis.Request = OriginalRequest;
    if (OriginalProgressEvent === undefined) delete globalThis.ProgressEvent;
    else globalThis.ProgressEvent = OriginalProgressEvent;
    await new Promise((resolve) => server.close(resolve));
  }
});

test("configured source codec bytes and glTF sidecars retain document paths through build and standalone packing", async () => {
  const codecFiles = Object.fromEntries([
    ...["draco_wasm_wrapper.js", "draco_decoder.wasm", "draco_decoder.js"].map(name => ["draco", name]),
    ...["basis_transcoder.js", "basis_transcoder.wasm"].map(name => ["basis", name]),
  ].map(([kind, name]) => [`site/codecs/${kind}/${name}`, fs.readFileSync(new URL(`examples/jsm/libs/${kind}/${name}`, rootUrl))]));
  const model = JSON.stringify({ asset: { version: "2.0" },
    buffers: [{ uri: "geometry.bin", byteLength: 4 }], images: [{ uri: "albedo.ktx2" }] });
  const resources = {
    ...codecFiles,
    "site/models/scene.gltf": Buffer.from(model),
    "site/shared/geometry.bin": Buffer.from([1, 2, 3, 4]),
    "site/shared/albedo.ktx2": fs.readFileSync(new URL("examples/textures/ktx2/2d_etc1s.ktx2", rootUrl)),
  };
  for (const prepareStandaloneLoaders of [false, true]) {
    // The standalone packer's existing contract excludes base elements. Prove
    // authored document-base URLs in the networked build and the same bytes at
    // explicit relative paths in the supported standalone document shape.
    const prefix = prepareStandaloneLoaders ? "site/" : "";
    const root = fixture({
      "index.html": prepareStandaloneLoaders ? '<script type="module" src="modules/app.mjs"></script>'
        : '<base href="./site/"><script type="module" src="../modules/app.mjs"></script>',
      "modules/app.mjs": `import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
      import { DRACOLoader } from ${JSON.stringify(dracoUrl)};
      import { KTX2Loader } from ${JSON.stringify(ktx2Url)};
      import { MeshoptDecoder } from ${JSON.stringify(rootUrl + "examples/jsm/libs/meshopt_decoder.module.js")};
      export async function load(renderer) {
        const draco = new DRACOLoader().setDecoderPath('${prefix}codecs/draco/').setDecoderConfig({type:'js'});
        const ktx = new KTX2Loader(); ktx.setTranscoderPath('${prefix}codecs/basis/'); ktx.detectSupport(renderer);
        const loader = new GLTFLoader(); loader.setDRACOLoader(draco); loader.setKTX2Loader(ktx);
        loader.setMeshoptDecoder(MeshoptDecoder);
        loader.setPath('${prefix}models/').setResourcePath('${prefix}shared/');
        return loader.loadAsync('scene.gltf');
      }`,
      ...resources,
    });
    const built = await buildApplication(path.join(root, "index.html"),
      path.join(root, prepareStandaloneLoaders ? "standalone" : "networked"), { prepareStandaloneLoaders });
    for (const [name, bytes] of Object.entries(resources)) {
      assert.deepEqual(fs.readFileSync(path.join(built.outDir, name)), bytes, `authored ${name} is byte-identical`);
    }
    if (!prepareStandaloneLoaders) continue;
    const helper = fs.readFileSync(path.join(built.outDir, "f3d-loader-resources.mjs"), "utf8");
    for (const [name, bytes] of Object.entries(resources)) {
      assert.ok(helper.includes(bytes.toString("base64")), `${name} enters the existing embedded resource resolver`);
    }
    const packed = path.join(root, "packed.html");
    packHtml(path.join(built.outDir, built.htmlFile), packed);
    const html = fs.readFileSync(packed, "utf8");
    const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
    const embedded = Object.values(imports).map(url => Buffer.from(url.split(",")[1].split("#")[0], "base64").toString());
    assert.ok(embedded.includes(helper), "the exact native codec-worker/resource runtime survives packing");
  }
});

test("retains host-owned URLs and custom manager routes while packaging local finite resources", async () => {
  const root = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)};
      export function load(manager) {
        new FileLoader().load('https://example.invalid/data.bin');
        new FileLoader().load('/host/data.bin');
        new FileLoader().load('data:text/plain,inline');
        new FileLoader(manager).load('logical-name');
        new FileLoader().load('runtime-generated.txt');
        new FileLoader().load('local.txt');
      }`,
    "local.txt": "local bytes",
  });
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  assert.equal(fs.readFileSync(path.join(built.outDir, "local.txt"), "utf8"), "local bytes");
  const code = fs.readFileSync(path.join(built.outDir, built.entryFiles[0]), "utf8");
  for (const url of ["https://example.invalid/data.bin", "/host/data.bin", "data:text/plain,inline", "logical-name", "runtime-generated.txt", "local.txt"]) assert.ok(code.includes(url));
});

test("a shared DefaultLoadingManager route in another module prevents false local asset reads and collisions", async () => {
  const root = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import './manager.mjs'; import { FileLoader } from ${JSON.stringify(coreUrl)};
      import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
      export function load() { new FileLoader().load('app.js'); new GLTFLoader().load('logical.gltf'); }`,
    "manager.mjs": `import { DefaultLoadingManager as manager } from ${JSON.stringify(coreUrl)};
      manager.setURLModifier(() => 'data:text/plain,managed');`,
    "app.js": "unrelated local bytes must not replace the bundle",
    "logical.gltf": "this is not the logical manager response",
  });
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, "dist"));
  assert.equal(built.emittedFiles.includes("logical.gltf"), false);
  const code = fs.readFileSync(path.join(built.outDir, built.entryFiles[0]), "utf8");
  assert.ok(code.includes("data:text/plain,managed"));
  assert.ok(code.includes("logical.gltf"));
  assert.ok(!code.includes("unrelated local bytes"));
});

test("loader asset names cannot overwrite bundled JavaScript or the HTML entry", async () => {
  for (const url of ["app.js", "index.html"]) {
    const root = fixture({
      "index.html": '<script type="module" src="app.mjs"></script>',
      "app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)}; new FileLoader().load(${JSON.stringify(url)});`,
      "app.js": "authored asset bytes",
    });
    await assert.rejects(buildApplication(path.join(root, "index.html"), path.join(root, "dist")), /collides with (?:an emitted bundle file|the emitted HTML entry)/);
    assert.equal(fs.existsSync(path.join(root, "dist")), false);
  }
});

test("ES module libraries retain relative loader URLs until a document host is known", async () => {
  const root = fixture({
    "app.mjs": `import { FileLoader } from ${JSON.stringify(coreUrl)}; export const load=()=>new FileLoader().loadAsync('host.txt');`,
    "host.txt": "belongs to one possible host",
  });
  const built = await buildApplication(path.join(root, "app.mjs"), path.join(root, "dist"));
  assert.equal(built.emittedFiles.some((file) => path.basename(file) === "host.txt"), false);
});
