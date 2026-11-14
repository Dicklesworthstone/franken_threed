import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import * as nodeModule from "node:module";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { parse } from "acorn";
import { buildApplication } from "./build_application.mjs";
import { packHtml } from "./pack_html.mjs";
import { collectThreeLoaderAssets, findThreeLoaderRequests } from "./three_loader_assets.mjs";
import { fbxResources, fbxPixel } from "./fixtures/fbx_resources.mjs";
import { colladaAssetUris } from "./pack_collada.mjs";
import { prepareUsdAssetUris } from "./pack_usd.mjs";

const rootUrl = new URL("../../upstream/three.js/", import.meta.url).href;
const coreUrl = rootUrl + "build/three.core.js";
const gltfUrl = rootUrl + "examples/jsm/loaders/GLTFLoader.js";
const dracoUrl = rootUrl + "examples/jsm/loaders/DRACOLoader.js";
const ktx2Url = rootUrl + "examples/jsm/loaders/KTX2Loader.js";
const ultraHdrUrl = rootUrl + "examples/jsm/loaders/UltraHDRLoader.js";
const mtlUrl = rootUrl + "examples/jsm/loaders/MTLLoader.js";
const objUrl = rootUrl + "examples/jsm/loaders/OBJLoader.js";
const fbxUrl = rootUrl + "examples/jsm/loaders/FBXLoader.js";
const colladaUrl = rootUrl + "examples/jsm/loaders/ColladaLoader.js";
const threeMfUrl = rootUrl + "examples/jsm/loaders/3MFLoader.js";
const rhinoUrl = rootUrl + "examples/jsm/loaders/3DMLoader.js";
const ldrawUrl = rootUrl + "examples/jsm/loaders/LDrawLoader.js";
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

test("private resource tables close loader files, model sidecars, cube faces and runtime codecs", () => {
  const requests = discover(`import { FileLoader, CubeTextureLoader } from 'three';
    import { GLTFLoader } from ${JSON.stringify(gltfUrl)};
    import { DRACOLoader } from ${JSON.stringify(dracoUrl)};
    import { KTX2Loader } from ${JSON.stringify(ktx2Url)};
    const paths={files:'assets/',models:'models/',sidecars:'shared/',draco:'draco/',basis:'basis/'}, alias=paths;
    const files=['one.txt','two.txt'], index=choose?0:1;
    new FileLoader().setPath((observe(),alias.files)).loadAsync(files[index]);
    const models={primary:'first.gltf',alternate:'second.gltf'}, choice=choose?'primary':'alternate';
    const draco=new DRACOLoader().setDecoderPath(paths.draco);
    const basis=new KTX2Loader().setTranscoderPath(paths.basis).detectSupport(renderer);
    new GLTFLoader().setDRACOLoader(draco).setKTX2Loader(basis).setPath(paths.models)
      .setResourcePath(paths.sidecars).loadAsync(models[choice]);
    const faces={px:'px.png',nx:'nx.png',py:'py.png',ny:'ny.png',pz:'pz.png',nz:'nz.png'};
    new CubeTextureLoader().load([faces.px,faces.nx,faces.py,faces.ny,faces.pz,faces.nz]);`);
  assert.deepEqual(requests.map(({url}) => url), [
    'assets/one.txt', 'assets/two.txt', 'models/first.gltf', 'models/second.gltf',
    'draco/draco_wasm_wrapper.js', 'draco/draco_decoder.wasm', 'draco/draco_decoder.js',
    'basis/basis_transcoder.js', 'basis/basis_transcoder.wasm',
    'px.png', 'nx.png', 'py.png', 'ny.png', 'pz.png', 'nz.png',
  ]);
  assert.deepEqual(requests.filter(({loader}) => loader === 'GLTFLoader').map(({resourcePath}) => resourcePath),
    ['shared/', 'shared/']);
});

test("loader resource tables keep mutations, escapes, shadowing and unknown selections on the host route", () => {
  for (const tail of [
    "routes.entry='changed.txt';", "const alias=routes;alias.entry='changed.txt';",
    "consume(routes);", "export {routes};", "routes.missing;", "routes[unknown];",
    "Object.defineProperty(routes,'entry',{get:opaque});", "routes.entry();", "eval(code);",
  ]) {
    assert.deepEqual(discover(`import { FileLoader } from 'three';
      const routes={entry:'file.txt'}; new FileLoader().load(routes.entry); ${tail}`), [], tail);
  }
  for (const source of [
    "const routes={get entry(){return 'file.txt'}};new FileLoader().load(routes.entry);",
    "const routes={entry:'file.txt'};function load(routes){new FileLoader().load(routes.entry)}",
    "const routes=['first.txt',,'third.txt'];new FileLoader().load(routes[0]);",
    "const routes={entry:'file.txt'};new FileLoader().load(routes[flag?'entry':'missing']);",
    "const routes={base:'assets/'};consume(routes);new FileLoader().setPath(routes.base).load('file.txt');",
    "const routes={entry:'file.txt'};const loader=new FileLoader();consume(loader);loader.load(routes.entry);",
    "const routes={entry:'file.txt'};new FileLoader(manager).load(routes.entry);",
  ]) assert.deepEqual(discover(`import { FileLoader } from 'three';${source}`), [], source);
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

test("volume, voxel and color-grading loaders retain finite files and live source options", () => {
  const requests = discover(`import { FloatType, UnsignedByteType } from 'three';
    import { NRRDLoader as Volume, VOXLoader, LUTCubeLoader, LUT3dlLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import * as Voxels from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/VOXLoader.js')};
    const volume=new Volume().setPath('volumes/'); volume.setSegmentation(false);
    volume.loadAsync('raw%20%23.nrrd?version=2#scan'); volume.segmentation=true;
    new Volume().load('second.nrrd');
    new VOXLoader().setPath('models/').loadAsync(choice?'one.vox':'two.vox');
    new Voxels.VOXLoader().load('namespace.vox');
    const cube=new LUTCubeLoader().setType(FloatType).setPath('luts/');
    cube.loadAsync('grade.CUBE'); cube.type=UnsignedByteType;
    const film=new LUT3dlLoader(); film.type=FloatType; film.setPath('luts/');
    film.load('grade.3dl'); film.setType(UnsignedByteType); film.parse(runtimeText);`);
  assert.deepEqual(requests.map(({url}) => url), ['volumes/raw%20%23.nrrd?version=2#scan',
    'second.nrrd', 'models/one.vox', 'models/two.vox', 'namespace.vox', 'luts/grade.CUBE', 'luts/grade.3dl']);
});

test("volume and LUT discovery preserves custom managers, escaped loaders and opaque configuration", () => {
  const requests = discover(`import { NRRDLoader, VOXLoader, LUTCubeLoader, LUT3dlLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import { NRRDLoader as Foreign } from 'file:///foreign/NRRDLoader.js';
    new Foreign().load('foreign.nrrd'); new VOXLoader(manager).load('managed.vox');
    function shadow(NRRDLoader){new NRRDLoader().load('shadow.nrrd');}
    const escaped=new LUTCubeLoader(); customize(escaped); escaped.load('escaped.CUBE');
    const changed=new NRRDLoader(); changed.segmentation=opaque; changed.load('opaque.nrrd');
    const method=new NRRDLoader(); method.segmentation=()=>true; method.load('method.nrrd');
    const lut=new LUT3dlLoader(); lut.type=runtimeType; lut.load('opaque.3dl');
    new LUTCubeLoader().setPath(runtimePath).load('dynamic.CUBE');`);
  assert.deepEqual(requests, []);
});

test("Gaussian splat loaders retain finite models, source identities and decoding calls", () => {
  const requests = discover(`import { GaussianSplatPLYLoader as PLY, SPLATLoader, SPZLoader, KSPLATLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import * as Splat from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/SPLATLoader.js')};
    const paths={models:'models/',alternate:'other/'}, models={preview:'preview.ply',full:'full.ply'};
    const Loader=PLY, ply=new Loader().setPath(paths.models);
    ply.loadAsync(models[preview?'preview':'full']); ply.parse(runtimePLY);
    new Splat.SPLATLoader().setPath('binary/').load('cloud%20%23.splat?version=2#points');
    const spz=new SPZLoader(); spz.setPath('compressed/');
    spz.loadAsync(modern?'model.v4.spz':'model.v3.spz'); spz.parse(runtimeBytes,onLoad,onError);
    const ksplat=new KSPLATLoader().setPath(paths.alternate);
    ksplat.parse(runtimeKSplat); ksplat.load('cloud.ksplat');
    new SPLATLoader().loadAsync('direct.splat');`);
  assert.deepEqual(requests.map(({url})=>url), ['models/preview.ply','models/full.ply',
    'binary/cloud%20%23.splat?version=2#points','compressed/model.v4.spz','compressed/model.v3.spz',
    'other/cloud.ksplat','direct.splat']);
});

test("Gaussian splat discovery keeps custom managers, escaped loaders and mutable paths source-owned", () => {
  for (const name of ['GaussianSplatPLYLoader','SPLATLoader','SPZLoader','KSPLATLoader']) {
    const imported=`import {${name} as Loader} from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/' + name + '.js')};`;
    for (const operation of [
      "new Loader(manager).load('managed');",
      "const loader=new Loader();customize(loader);loader.load('escaped');",
      "const loader=new Loader();loader.parse=custom;loader.load('changed');",
      "const loader=new Loader();loader.setPath(runtimePath).load('dynamic');",
      "const paths={entry:'first'};paths.entry='second';new Loader().load(paths.entry);",
      "function run(Loader){new Loader().load('shadowed');}",
    ]) assert.deepEqual(discover(imported+operation),[],name+': '+operation);
    assert.deepEqual(discover(`import {${name}} from 'file:///foreign/${name}.js';new ${name}().load('foreign');`),[]);
  }
});

test("H8's original UltraHDR environment request enters the application resource closure", () => {
  const html = fs.readFileSync(new URL("examples/webgpu_compute_particles_fluid.html", rootUrl), "utf8");
  const source = /<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];
  const requests = discover(source, {resolveModule(specifier) {
    if (specifier.startsWith("three/addons/")) return rootUrl + "examples/jsm/" + specifier.slice("three/addons/".length);
    if (specifier === "three/tsl") return rootUrl + "build/three.tsl.js";
    return rootUrl + "build/three.webgpu.js";
  }});
  assert.deepEqual(requests.filter(({loader}) => loader === "UltraHDRLoader").map(({url}) => url),
    ["textures/equirectangular/royal_esplanade_2k.hdr.jpg"]);
});

test("UltraHDR discovery keeps source data-type settings, aliases, finite choices and opaque routes", () => {
  const requests = discover(`import { UltraHDRLoader as Source } from ${JSON.stringify(ultraHdrUrl)};
    import * as HDR from ${JSON.stringify(ultraHdrUrl)};
    import { UltraHDRLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import { UltraHDRLoader as Foreign } from 'file:///other/examples/jsm/loaders/UltraHDRLoader.js';
    const Loader = Source, loader = new Loader().setPath('env/');
    loader.setDataType(floatType); loader.parse(runtimeBytes, onLoad);
    loader.loadAsync(day ? 'day%20%23.hdr.jpg?version=2#sky' : 'night.hdr.jpg');
    loader.setDataType(halfType); loader.setPath('next/'); loader.load('sky.hdr.jpg');
    new HDR.UltraHDRLoader().load('namespace.hdr.jpg');
    new UltraHDRLoader().loadAsync('aggregate.hdr.jpg');
    new Foreign().load('foreign.hdr.jpg');
    function shadow(Source) { new Source().load('shadow.hdr.jpg'); }
    new Source(manager).load('managed.hdr.jpg');
    const escaped = new Source(); customize(escaped); escaped.load('escaped.hdr.jpg');
    new Source().setPath(runtimePath).load('dynamic.hdr.jpg');`);
  assert.deepEqual(requests.map(({url}) => url), ['env/day%20%23.hdr.jpg?version=2#sky',
    'env/night.hdr.jpg', 'next/sky.hdr.jpg', 'namespace.hdr.jpg', 'aggregate.hdr.jpg']);
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
  assert.deepEqual(requests.map(({url}) => url), ['materials.obj', 'sidecars.fbx', 'known.hdr']);
});

test("point, motion, toolpath and molecule loaders preserve finite files and source boolean options", () => {
  const requests = discover(`import { PCDLoader as Points, VTKLoader, XYZLoader, BVHLoader, GCodeLoader, PDBLoader }
      from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import * as PCD from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/PCDLoader.js')};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    const Loader = Points, cloud = new Loader();
    cloud.littleEndian = false; const endian = cloud.littleEndian;
    cloud.setPath('points/').setRequestHeader({Accept:'application/octet-stream'}).setWithCredentials(true);
    cloud.loadAsync(binary ? 'cloud%20%23.pcd?version=2#points' : 'cloud-ascii.pcd');
    cloud.littleEndian = true;
    const motion = new BVHLoader(); motion.animateBonePositions = false;
    motion.animateBoneRotations = true; if (motion.animateBoneRotations) motion.parse(inMemoryMotion);
    motion.setPath('motion/').load('walking.bvh');
    const machine = new GCodeLoader(); machine.splitLayer = true;
    machine.load('paths/print.gcode'); machine.splitLayer = false;
    new VTKLoader().setPath('geometry/').loadAsync('surface.vtk');
    new XYZLoader().load('points/cloud.xyz');
    new PDBLoader().setPath('molecules/').load('ethanol.pdb');
    new PCD.PCDLoader().load('namespace.pcd');
    new Addons.BVHLoader().load('namespace.bvh');`);
  assert.deepEqual(requests.map(({loader, url}) => [loader, url]), [
    ['PCDLoader', 'points/cloud%20%23.pcd?version=2#points'], ['PCDLoader', 'points/cloud-ascii.pcd'],
    ['BVHLoader', 'motion/walking.bvh'], ['GCodeLoader', 'paths/print.gcode'],
    ['VTKLoader', 'geometry/surface.vtk'], ['XYZLoader', 'points/cloud.xyz'],
    ['PDBLoader', 'molecules/ethanol.pdb'], ['PCDLoader', 'namespace.pcd'], ['BVHLoader', 'namespace.bvh'],
  ]);
});

test("single-file loader configuration does not admit callable, unknown or foreign state", () => {
  const requests = discover(`import { PCDLoader, VTKLoader, XYZLoader, BVHLoader, GCodeLoader, PDBLoader }
      from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import { PCDLoader as Foreign } from 'file:///foreign/PCDLoader.js';
    import * as Wrong from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/BVHLoader.js')};
    new Foreign().load('foreign.pcd'); new Wrong.PCDLoader().load('wrong-export.pcd');
    function shadow(VTKLoader) { new VTKLoader().load('shadow.vtk'); }
    new PDBLoader(manager).load('managed.pdb');
    const custom = new PCDLoader(); custom.littleEndian = function(){this.setPath('other/');};
    custom.littleEndian(); custom.load('custom.pcd');
    const motion = new BVHLoader(); motion.animateBonePositions = runtimeOption; motion.load('dynamic.bvh');
    const called = new BVHLoader(); called.animateBoneRotations(); called.load('called.bvh');
    const destructured = new GCodeLoader(); ({flag:destructured.splitLayer}=runtimeOptions);
    destructured.load('destructured.gcode');
    const wrongOption = new XYZLoader(); wrongOption.littleEndian = false; wrongOption.load('wrong-option.xyz');
    const escaped = new PDBLoader(); customize(escaped); escaped.load('escaped.pdb');
    new VTKLoader().setPath(runtimePath).load('dynamic.vtk');
    new PCDLoader().load('known.pcd');`);
  assert.deepEqual(requests.map(({url}) => url), ['known.pcd']);
});

test("3MF discovery follows its actual source filename, aggregate exports and finite archive choices", () => {
  const requests = discover(`import { ThreeMFLoader as Source } from ${JSON.stringify(threeMfUrl)};
    import * as SourceModule from ${JSON.stringify(threeMfUrl)};
    import { ThreeMFLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    const Loader = Source, loader = new Loader();
    loader.parse(inMemoryArchive); loader.setPath('models/');
    loader.loadAsync(choice ? 'truck%20%23.3mf?edition=2#model' : 'colors.3mf');
    new SourceModule.ThreeMFLoader().load('module.3mf');
    new ThreeMFLoader().load('aggregate.3mf');
    new Addons.ThreeMFLoader().loadAsync('namespace.3mf');`);
  assert.deepEqual(requests.map(({loader, url}) => [loader, url]), [
    ['ThreeMFLoader', 'models/truck%20%23.3mf?edition=2#model'], ['ThreeMFLoader', 'models/colors.3mf'],
    ['ThreeMFLoader', 'module.3mf'], ['ThreeMFLoader', 'aggregate.3mf'], ['ThreeMFLoader', 'namespace.3mf'],
  ]);
});

test("LDraw discovery preserves packed models, finite material preloads and source configuration", () => {
  const requests = discover(`import { LDrawLoader as Source } from ${JSON.stringify(ldrawUrl)};
    import * as Module from ${JSON.stringify(ldrawUrl)};
    import { LDrawLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    const Loader=Source, loader=new Loader();
    loader.setConditionalLineMaterial(ConditionalMaterial).setPartsLibraryPath('library/')
      .setFileMap({'old.dat':'part.dat'}).setMaterials([]).addMaterials([]).clearMaterials().addDefaultMaterials();
    loader.smoothNormals=false; const smooth=loader.smoothNormals;
    loader.setPath('palettes/'); await loader.preloadMaterials(choice?'colors%20%23.ldr?version=2#palette':'fallback.ldr');
    loader.setPath('models/'); loader.loadAsync(choice?'car%20%23.mpd?edition=2#model':'scene.mpd');
    loader.smoothNormals=true; loader.parse(runtimeModel,onLoad,onError);
    new Module.LDrawLoader().load('module.mpd');
    new LDrawLoader().load('aggregate.mpd');
    new Addons.LDrawLoader().preloadMaterials('namespace.ldr');`);
  assert.deepEqual(requests.map(({loader,url}) => [loader,url]), [
    ['LDrawLoader','palettes/colors%20%23.ldr?version=2#palette'], ['LDrawLoader','palettes/fallback.ldr'],
    ['LDrawLoader','models/car%20%23.mpd?edition=2#model'], ['LDrawLoader','models/scene.mpd'],
    ['LDrawLoader','module.mpd'], ['LDrawLoader','aggregate.mpd'], ['LDrawLoader','namespace.ldr'],
  ]);
});

test("LDraw unknown mutations, foreign identities and dynamic URLs retain their host route", () => {
  const requests = discover(`import { LDrawLoader } from ${JSON.stringify(ldrawUrl)};
    import { LDrawLoader as Foreign } from 'file:///foreign/LDrawLoader.js';
    import * as Wrong from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/OBJLoader.js')};
    function shadow(LDrawLoader){new LDrawLoader().load('shadow.mpd');}
    new Foreign().load('foreign.mpd'); new Wrong.LDrawLoader().load('wrong.mpd');
    new LDrawLoader(manager).load('managed.mpd');
    const custom=new LDrawLoader();custom.smoothNormals=()=>custom.setPath('other/');custom.load('custom.mpd');
    const dynamic=new LDrawLoader();dynamic.smoothNormals=option;dynamic.load('dynamic.mpd');
    const escaped=new LDrawLoader();customize(escaped);escaped.load('escaped.mpd');
    new LDrawLoader().setPath(runtimePath).preloadMaterials('dynamic.ldr');
    new LDrawLoader().preloadMaterials(runtimePalette);
    new LDrawLoader().loadAsync('known.mpd');`);
  assert.deepEqual(requests.map(({url}) => url), ['known.mpd']);
});

test("3MF foreign files, shadows, custom extensions and escaping loaders retain their host route", () => {
  const requests = discover(`import { ThreeMFLoader } from ${JSON.stringify(threeMfUrl)};
    import { ThreeMFLoader as Invented } from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/ThreeMFLoader.js')};
    import { ThreeMFLoader as Foreign } from 'file:///foreign/3MFLoader.js';
    function shadow(ThreeMFLoader) { new ThreeMFLoader().load('shadow.3mf'); }
    new Invented().load('invented.3mf'); new Foreign().load('foreign.3mf');
    new ThreeMFLoader(manager).load('managed.3mf');
    const custom = new ThreeMFLoader(); custom.addExtension(extension); custom.load('custom.3mf');
    const escaped = new ThreeMFLoader(); change(escaped); escaped.load('escaped.3mf');
    new ThreeMFLoader().setPath(dynamicPath).load('dynamic.3mf');
    new ThreeMFLoader().load('known.3mf');`);
  assert.deepEqual(requests.map(({url}) => url), ['known.3mf']);
});

test("USD archive discovery retains source aliases, addon exports, finite paths and runtime parse", () => {
  const requests = discover(`import { USDLoader as Source } from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/USDLoader.js')};
    import * as Legacy from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/USDZLoader.js')};
    import { USDLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    const Constructor=Source, loader=new Constructor(); loader.parse(runtimeBytes, '', onLoad, onError);
    loader.setPath('models/'); loader.loadAsync(choice ? 'scene%20%23.usdz?edition=2#root' : 'other.usdz');
    new Legacy.USDZLoader().setPath('legacy/').load('scene.usdz');
    new USDLoader().load('aggregate.usdz'); new Addons.USDZLoader().loadAsync('namespace.usdz');`);
  assert.deepEqual(requests.map(({loader, url}) => [loader, url]), [
    ['USDLoader', 'models/scene%20%23.usdz?edition=2#root'], ['USDLoader', 'models/other.usdz'],
    ['USDZLoader', 'legacy/scene.usdz'], ['USDLoader', 'aggregate.usdz'], ['USDZLoader', 'namespace.usdz'],
  ]);
});

test("USD discovery keeps custom managers, modified receivers and unknown paths on the host route", () => {
  const requests = discover(`import { USDLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/USDLoader.js')};
    import { USDZLoader } from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/USDZLoader.js')};
    import { USDLoader as Foreign } from 'file:///other/USDLoader.js';
    import { USDLoader as Wrong } from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/USDZLoader.js')};
    new Foreign().load('foreign.usdz'); new Wrong().load('wrong.usdz');
    new USDLoader(manager).load('managed.usdz');
    const escaped=new USDLoader(); customize(escaped); escaped.load('escaped.usdz');
    const changed=new USDLoader(); changed.parse=customParse; changed.load('changed.usdz');
    new USDZLoader().setPath(runtimePath).load('dynamic.usdz');
    function shadow(USDLoader) { new USDLoader().load('shadow.usdz'); }
    new USDLoader().load('known.usdz');`);
  assert.deepEqual(requests.map(({url}) => url), ['known.usdz']);
});

test("FBX loader discovery retains animation settings, captured defaults and pending resource changes", () => {
  const requests = discover(`import { FBXLoader } from ${JSON.stringify(fbxUrl)};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    const loader = new FBXLoader(); loader.trimAnimationClips = true; const trimming = loader.trimAnimationClips;
    loader.setPath('models/'); loader.loadAsync('nested/animated.fbx');
    loader.setResourcePath('textures/'); loader.load('second.fbx');
    loader.setResourcePath('');
    new Addons.FBXLoader().load('other/model.fbx');
    const late = new FBXLoader(); late.load('before/model.fbx'); late.setResourcePath(runtimePath);
    new FBXLoader(manager).load('managed.fbx');
    const escaping = new FBXLoader(); customize(escaping); escaping.load('escaped.fbx');
    const changed = new FBXLoader(); changed.path = runtimePath; changed.load('changed.fbx');`);
  assert.deepEqual(requests.map(({url,texturePaths}) => [url,texturePaths]), [
    ['models/nested/animated.fbx',['models/','textures/']], ['models/second.fbx',['models/','textures/']],
    ['other/model.fbx',['other/']], ['before/model.fbx',null],
  ]);
});

test("FBX animation option methods and unknown mutations cannot masquerade as boolean settings", () => {
  const requests = discover(`import { FBXLoader } from ${JSON.stringify(fbxUrl)};
    const loader = new FBXLoader();
    loader.trimAnimationClips = function(){this.setPath('other/');};
    loader.trimAnimationClips(); loader.load('mesh.fbx');
    const indirect = new FBXLoader(); indirect.trimAnimationClips = runtimeOption; indirect.load('unknown.fbx');
    const destructured = new FBXLoader(); ({flag:destructured.trimAnimationClips}=runtimeOptions);
    destructured.load('destructured.fbx');
    new FBXLoader().load('ordinary.fbx');`);
  assert.deepEqual(requests.map(({url}) => url), ['ordinary.fbx']);
});

test("Collada discovery retains captured default paths, pending resource resets and original runtime parse", () => {
  const requests = discover(`import { ColladaLoader as Source } from ${JSON.stringify(colladaUrl)};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    const loader = new Source(); loader.parse(runtimeText, 'runtime/');
    loader.load('first/model.dae'); loader.loadAsync('second/model.dae');
    loader.setPath('models/'); loader.load('nested/model.dae'); loader.setResourcePath('shared/');
    const reset = new Source().setResourcePath('before/'); reset.load('fallback/model.dae'); reset.setResourcePath('');
    new Addons.ColladaLoader().setResourcePath('textures/').loadAsync('namespace.dae');
    const late = new Source(); late.load('host.dae'); late.setResourcePath(runtimePath);
    new Source(manager).load('managed.dae'); const escaped = new Source(); customize(escaped); escaped.load('escaped.dae');
    const changed = new Source(); changed.path = runtimePath; changed.load('changed.dae');`);
  assert.deepEqual(requests.map(({url,texturePaths}) => [url,texturePaths]), [
    ['first/model.dae',['first/','shared/']], ['second/model.dae',['second/','shared/']],
    ['models/nested/model.dae',['models/','shared/']], ['fallback/model.dae',['before/','fallback/']],
    ['namespace.dae',['textures/']], ['host.dae',null],
  ]);
});

test("Collada image metadata follows direct source libraries, XML text and duplicate identities", () => {
  const text = `<?xml version="1.0"?><COLLADA><extra><library_images>
      <image id="fake"><init_from>nested-library.png</init_from></image></library_images></extra>
    <library_images>
      <!-- <image id="fake"><init_from>comment.png</init_from></image> -->
      <image id="same"><init_from>replaced.png</init_from></image>
      <image id="same"><init_from>paint&#32;%23.png?variant=2&amp;channel=1#map</init_from></image>
      <image id="data"><init_from><![CDATA[data:image/png;base64,A&B]]></init_from></image>
      <image id="literal"><init_from> folder/<extra>nested</extra> image.tga </init_from><init_from>second.png</init_from></image>
      <image id="unicode"><init_from>caf&#233;&#x1F343;.png</init_from></image>
    </library_images>
    <library_images><image id="later"><init_from>ignored.png</init_from></image></library_images>
    <library_effects><effect><surface><init_from>sampler-id</init_from></surface></effect></library_effects>
  </COLLADA>`;
  assert.deepEqual(colladaAssetUris(Buffer.from(text)), [
    'paint %23.png?variant=2&channel=1#map', 'data:image/png;base64,A&B', ' folder/nested image.tga ', 'café🍃.png',
  ]);
  for (const invalid of ['<COLLADA><library_images></COLLADA>',
    '<COLLADA><library_images><image id="x"><init_from>&unknown;</init_from></image></library_images></COLLADA>'])
    assert.throws(() => colladaAssetUris(Buffer.from(invalid)), {code:'COLLADA_XML'});
  assert.throws(() => colladaAssetUris(Buffer.from('<!DOCTYPE COLLADA [<!ENTITY custom "image.png">]><COLLADA/>')),
    {code:'COLLADA_XML_UNSUPPORTED'});
  assert.throws(() => colladaAssetUris(Buffer.from('<COLLADA>' + '<a>'.repeat(128))), {code:'COLLADA_BUDGET'});
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

test("FBX pending resources preserve captured file bases, live resets and opaque loader boundaries", () => {
  const requests = discover(`import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import { FBXLoader as Source } from ${JSON.stringify(fbxUrl)};
    const loader = new Source(); loader.load('first/scene.fbx'); loader.loadAsync('second/scene.fbx');
    loader.setPath('models/'); loader.load('nested/scene.fbx'); loader.setResourcePath('shared/');
    const reset = new Source().setResourcePath('before/'); reset.load('fallback/scene.fbx'); reset.setResourcePath('');
    new Addons.FBXLoader().setResourcePath('textures/').loadAsync('namespace.fbx');
    const dynamic = new Source(); dynamic.load('host.fbx'); dynamic.setResourcePath(runtimePath);
    new Source(manager).load('managed.fbx'); const escaped = new Source(); change(escaped); escaped.load('escaped.fbx');`);
  assert.deepEqual(requests.map(({url,texturePaths})=>[url,texturePaths]), [
    ['first/scene.fbx',['first/','shared/']], ['second/scene.fbx',['second/','shared/']],
    ['models/nested/scene.fbx',['models/','shared/']], ['fallback/scene.fbx',['before/','fallback/']],
    ['namespace.fbx',['textures/']], ['host.fbx',null],
  ]);
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

test("Rhino3DM closes authored and default decoder libraries for loads and runtime parse inputs", () => {
  const requests = discover(`import { Rhino3dmLoader as CAD } from ${JSON.stringify(rhinoUrl)};
    import * as Models from ${JSON.stringify(rhinoUrl)};
    import * as Addons from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    const loader = new CAD().setLibraryPath('codec-prefix-').setWorkerLimit(1).setSubdivisionLevel(2);
    loader.setPath('models/'); loader.loadAsync('cad%20%23.3dm?version=2#assembly');
    loader.parse(bytes, onLoad, onError); loader.parseAsync(otherBytes); loader.dispose();
    new Models.Rhino3dmLoader().parseAsync(runtimeBytes);
    const version = next ? 'v2/' : 'v1/';
    new Addons.Rhino3dmLoader().setLibraryPath('codecs/'+version).decodeObjects(input, '');`);
  assert.deepEqual(requests.map(request => request.url), [
    'models/cad%20%23.3dm?version=2#assembly', 'codec-prefix-rhino3dm.js', 'codec-prefix-rhino3dm.wasm',
    'rhino3dm.js', 'rhino3dm.wasm', 'codecs/v2/rhino3dm.js', 'codecs/v2/rhino3dm.wasm',
    'codecs/v1/rhino3dm.js', 'codecs/v1/rhino3dm.wasm',
  ]);
});

test("Rhino3DM includes finite library changes while loads are pending and keeps unknown libraries on the host route", () => {
  const requests = discover(`import { Rhino3dmLoader } from ${JSON.stringify(rhinoUrl)};
    const loader = new Rhino3dmLoader().setLibraryPath('before/');
    const pending = loader.loadAsync('models/pending.3dm');
    loader.setLibraryPath('after/');
    loader.parseAsync(otherBytes);
    const unknown = new Rhino3dmLoader().setLibraryPath('known/');
    unknown.load('models/unknown.3dm'); unknown.setLibraryPath(runtimePath);
    new Rhino3dmLoader().setLibraryPath('unused/');`);
  assert.deepEqual(requests.map(request => request.url), [
    'models/pending.3dm', 'models/unknown.3dm',
    'before/rhino3dm.js', 'before/rhino3dm.wasm', 'after/rhino3dm.js', 'after/rhino3dm.wasm',
  ]);
});

test("Rhino3DM custom managers, altered receivers and dynamic paths do not gain asset closure", () => {
  const requests = discover(`import { Rhino3dmLoader } from ${JSON.stringify(rhinoUrl)};
    import { Rhino3dmLoader as Foreign } from 'file:///foreign/3DMLoader.js';
    import { Rhino3dmLoader as Invented } from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/Rhino3dmLoader.js')};
    new Foreign().load('foreign.3dm'); new Invented().load('invented.3dm');
    new Rhino3dmLoader(manager).setLibraryPath('managed/').load('managed.3dm');
    new Rhino3dmLoader().setLibraryPath(runtimePath).parseAsync(bytes);
    const escaped = new Rhino3dmLoader(); customize(escaped); escaped.load('escaped.3dm');
    const changed = new Rhino3dmLoader(); changed.libraryPath='custom/'; changed.load('changed.3dm');
    const conditional = new Rhino3dmLoader();
    if (choose) conditional.setLibraryPath('conditional/'); conditional.load('conditional.3dm');
    function shadow(Rhino3dmLoader) { new Rhino3dmLoader().load('shadow.3dm'); }`);
  assert.deepEqual(requests, []);
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

test("UltraHDR bytes survive relocated networked and packed application builds", async () => {
  const bytes = fs.readFileSync(new URL("examples/textures/equirectangular/royal_esplanade_2k.hdr.jpg", rootUrl));
  const logicalUrl = "textures/equirectangular/sky%20%23.hdr.jpg?version=2#environment";
  const filename = "textures/equirectangular/sky #.hdr.jpg";
  const root = fixture({
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": `import { FloatType } from ${JSON.stringify(coreUrl)};
      import { UltraHDRLoader } from ${JSON.stringify(ultraHdrUrl)};
      export async function load() {
        const hdrLoader = new UltraHDRLoader().setPath('textures/equirectangular/').setDataType(FloatType);
        return hdrLoader.loadAsync('sky%20%23.hdr.jpg?version=2#environment');
      }`,
    [filename]: bytes,
  });
  const deployment = fixture({});
  const networked = await buildApplication(path.join(root, "index.html"), path.join(deployment, "networked"));
  const standalone = await buildApplication(path.join(root, "index.html"), path.join(deployment, "prepared"),
    {prepareStandaloneLoaders: true});
  fs.renameSync(root, root + "-source-unavailable");
  for (const output of [networked, standalone]) {
    assert.deepEqual(fs.readFileSync(path.join(output.outDir, filename)), bytes);
  }
  const packedFile = path.join(deployment, "relocated.html");
  packHtml(path.join(standalone.outDir, standalone.htmlFile), packedFile);
  const resources = fs.readFileSync(path.join(standalone.outDir, "f3d-loader-resources.mjs"), "utf8");
  assert.ok(resources.includes(logicalUrl.slice(0, logicalUrl.indexOf("#"))));
  const html = fs.readFileSync(packedFile, "utf8");
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const embedded = Object.values(imports).find(url =>
    Buffer.from(url.split(",")[1].split("#")[0], "base64").toString() === resources);
  assert.ok(embedded, "the source loader's finite resource resolver survives packing");
  fs.renameSync(standalone.outDir, standalone.outDir + "-not-used-by-packed-page");

  const requests = [], server = http.createServer((request, response) => {
    requests.push(request.url);
    try {
      const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
      response.setHeader("Content-Type", "image/jpeg");
      response.end(fs.readFileSync(path.join(networked.outDir, pathname)));
    } catch { response.writeHead(404); response.end("missing resource"); }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}`, originalDocument = globalThis.document;
  try {
    const response = await fetch(new URL(logicalUrl, origin + "/index.html"));
    assert.equal(response.status, 200);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), bytes);
    assert.deepEqual(requests, ["/textures/equirectangular/sky%20%23.hdr.jpg?version=2"]);
    globalThis.document = {URL: origin + "/moved/to/a/different/depth/relocated.html",
      baseURI: origin + "/moved/to/a/different/depth/relocated.html"};
    const packedResources = await import(embedded);
    const resolved = packedResources.resolveThreeLoaderResource(logicalUrl);
    assert.match(resolved, /^data:image\/jpeg;/);
    assert.ok(resolved.endsWith("#environment"));
    assert.deepEqual(Buffer.from(await (await fetch(resolved)).arrayBuffer()), bytes);
    assert.equal(requests.length, 1, "the relocated packed texture must not reach the server");
  } finally {
    if (originalDocument === undefined) delete globalThis.document; else globalThis.document = originalDocument;
    server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  }
});

test("Collada assets preserve source path concatenation, TGA bytes, URL identities and pending bases", () => {
  const model = Buffer.from(`<COLLADA><library_images>
    <image id="paint"><init_from>paint%20%23.png?version=2&amp;quality=1#map</init_from></image>
    <image id="tga"><init_from>normal.tga</init_from></image>
    <image id="literal"><init_from>https://example.test/literal.png</init_from></image>
    <image id="outside"><init_from>../../../../outside.png</init_from></image>
  </library_images></COLLADA>`);
  const names = ['paint #.png','normal.tga','https:/example.test/literal.png'];
  const root = fixture({'index.html':'','site/models/nested/model.dae':model,
    ...Object.fromEntries(['models','textures'].flatMap(base => names.map(name => [`site/${base}/${name}`,fbxPixel])))});
  const requests = discover(`import {ColladaLoader} from ${JSON.stringify(colladaUrl)};
    const loader=new ColladaLoader().setPath('models/');loader.load('nested/model.dae');loader.setResourcePath('textures/');`);
  const logical=[], options={documentUrl:pathToFileURL(path.join(root,'index.html')).href,
    documentBaseUrl:pathToFileURL(root+'/site/').href,onResource:(name,url)=>logical.push(url)};
  const assets=collectThreeLoaderAssets(requests,options);
  assert.deepEqual([...assets.keys()].sort(), ['site/models/nested/model.dae',
    ...['models','textures'].flatMap(base=>names.map(name=>`site/${base}/${name}`))].sort());
  assert.deepEqual(assets.get('site/models/nested/model.dae').source,model);
  assert.ok(logical.some(url=>url.endsWith('paint%20%23.png?version=2&quality=1#map')));
  for (const name of [...assets.keys()].filter(name=>!name.endsWith('.dae'))) assert.deepEqual(assets.get(name).source,fbxPixel);
});

test("Collada builds retain textured animated model bytes and native XML parse dependencies", async () => {
  const base=new URL('../../upstream/three.js/examples/models/collada/stormtrooper/', import.meta.url);
  const model=fs.readFileSync(new URL('stormtrooper.dae',base)), image=fs.readFileSync(new URL('Stormtrooper_D.jpg',base));
  const root=fixture({'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':`import {ColladaLoader} from ${JSON.stringify(colladaUrl)};
      export const load=()=>new ColladaLoader().setPath('models/').loadAsync('trooper%20%23.dae?version=2#rig');`,
    'models/trooper #.dae':model, 'models/Stormtrooper_D.jpg':image});
  const output=path.join(root,'dist');
  await buildApplication(path.join(root,'index.html'),output,{prepareStandaloneLoaders:true});
  for (const name of ['trooper #.dae','Stormtrooper_D.jpg'])
    assert.deepEqual(fs.readFileSync(path.join(output,'models',name)),fs.readFileSync(path.join(root,'models',name)));
  const resources=fs.readFileSync(path.join(output,'f3d-loader-resources.mjs'),'utf8');
  assert.ok(resources.includes('models/trooper%20%23.dae?version=2'));
  assert.ok(resources.includes(image.toString('base64')));
  const packed=path.join(root,'packed.html');packHtml(path.join(output,'index.html'),packed);
  assert.ok(fs.statSync(packed).size>model.length+image.length);
});

test("valid Collada DTD image entities cannot silently produce an incomplete standalone package", async () => {
  const model = `<?xml version="1.0"?><!DOCTYPE COLLADA [<!ENTITY image "pixel.png">]>
    <COLLADA version="1.4.1"><asset/><library_images><image id="pixel"><init_from>&image;</init_from></image></library_images>
    <library_visual_scenes><visual_scene id="view"/></library_visual_scenes><scene><instance_visual_scene url="#view"/></scene></COLLADA>`;
  const root = fixture({'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':`import {ColladaLoader} from ${JSON.stringify(colladaUrl)};
      export const load=()=>new ColladaLoader().loadAsync('model.dae');`,
    'model.dae':model, 'pixel.png':fbxPixel});
  const requests=discover(fs.readFileSync(path.join(root,'app.mjs'),'utf8'));
  assert.throws(()=>collectThreeLoaderAssets(requests,{
    documentUrl:pathToFileURL(path.join(root,'index.html')).href,documentBaseUrl:pathToFileURL(root+'/').href,
  }),{code:'COLLADA_XML_UNSUPPORTED'});
  await assert.rejects(buildApplication(path.join(root,'index.html'),path.join(root,'dist'),{prepareStandaloneLoaders:true}),
    /COLLADA_XML_UNSUPPORTED/);
  // The input remains intact for the caller to choose a networked deployment.
  assert.equal(fs.readFileSync(path.join(root,'model.dae'),'utf8'),model);
});

test("3MF application builds preserve complete source ZIP archives and their logical URL variants", async () => {
  const archive = fs.readFileSync(new URL('../../upstream/three.js/examples/models/3mf/facecolors.3mf', import.meta.url));
  const root = fixture({
    'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import { ThreeMFLoader } from ${JSON.stringify(threeMfUrl)};
      export const load=()=>new ThreeMFLoader().setPath('models/').loadAsync('colors%20%23.3mf?edition=2#model');`,
    'models/colors #.3mf': archive,
  });
  const output = path.join(root, 'dist');
  await buildApplication(path.join(root, 'index.html'), output, {prepareStandaloneLoaders: true});
  assert.deepEqual(fs.readFileSync(path.join(output, 'models/colors #.3mf')), archive);
  const resources = fs.readFileSync(path.join(output, 'f3d-loader-resources.mjs'), 'utf8');
  assert.ok(resources.includes('models/colors%20%23.3mf?edition=2'));
  assert.ok(resources.includes(archive.toString('base64')));
  const packed = path.join(root, 'packed.html');
  packHtml(path.join(output, 'index.html'), packed);
  assert.ok(fs.statSync(packed).size > archive.length);
});

test("Rhino3DM builds preserve model, decoder JS and Wasm bytes and pack the source worker", async () => {
  const model = fs.readFileSync(new URL('../../upstream/three.js/examples/models/3dm/Rhino_Logo.3dm', import.meta.url));
  const wrapper = '/* Original codec response: </script> */\nvar rhino3dm = function () {};\n';
  const wasm = Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]);
  const root = fixture({
    'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import { Rhino3dmLoader } from ${JSON.stringify(rhinoUrl)};
      export const load=()=>new Rhino3dmLoader().setLibraryPath('codecs/').setPath('models/')
        .loadAsync('cad%20%23.3dm?version=2#assembly');`,
    'models/cad #.3dm': model, 'codecs/rhino3dm.js': wrapper, 'codecs/rhino3dm.wasm': wasm,
  });
  const output = path.join(root, 'dist');
  await buildApplication(path.join(root, 'index.html'), output, {prepareStandaloneLoaders: true});
  for (const name of ['models/cad #.3dm', 'codecs/rhino3dm.js', 'codecs/rhino3dm.wasm'])
    assert.deepEqual(fs.readFileSync(path.join(output, name)), fs.readFileSync(path.join(root, name)));
  const resources = fs.readFileSync(path.join(output, 'f3d-loader-resources.mjs'), 'utf8');
  assert.ok(resources.includes('models/cad%20%23.3dm?version=2'));
  for (const bytes of [model, Buffer.from(wrapper), wasm]) assert.ok(resources.includes(bytes.toString('base64')));
  const packed = path.join(root, 'packed.html');
  packHtml(path.join(output, 'index.html'), packed);
  assert.ok(fs.statSync(packed).size > model.length);
});

for (const encoding of [{binary:false}, {binary:true,version:7400}, {binary:true,version:7500}]) {
  test(`FBX ${encoding.binary ? encoding.version : 'ASCII'} closes texture metadata without touching model or embedded bytes`, () => {
    const model = fbxResources([
      {relative:'C:\\export\\paint%20%23.png?variant=2#map', filename:'discarded.png'},
      {filename:'faces/front.png', relative:''},
      {relative:'embedded.png',content:fbxPixel}, {relative:'embedded.png'},
      {relative:'empty.png',content:Buffer.alloc(0)},
      {relative:'fallback.unknown',content:fbxPixel},
      {relative:'data:image/png;base64,'+fbxPixel.toString('base64')},
      {relative:'blob:https://example.test/image'},
      {relative:'../escape.png'},
    ],encoding);
    const images=['paint #.png','faces/front.png','empty.png','fallback.unknown'];
    const root=fixture({'index.html':'','site/models/nested/scene.fbx':model,
      ...Object.fromEntries([...images,'embedded.png','discarded.png'].map(name=>['site/textures/'+name,fbxPixel]))});
    const requests=discover(`import {FBXLoader} from ${JSON.stringify(fbxUrl)};
      new FBXLoader().setPath('models/').setResourcePath('textures/').load('nested/scene.fbx');`);
    const logical=[],assets=collectThreeLoaderAssets(requests,{
      documentUrl:pathToFileURL(path.join(root,'index.html')).href,
      documentBaseUrl:pathToFileURL(path.join(root,'site/')).href,
      onResource:(name,url)=>logical.push(url),
    });
    assert.deepEqual([...assets.keys()].sort(),['site/models/nested/scene.fbx',...images.map(name=>'site/textures/'+name)].sort());
    assert.deepEqual(assets.get('site/models/nested/scene.fbx').source,model);
    assert.ok(logical.some(url=>url.endsWith('paint%20%23.png?variant=2#map')));
  });
}

test('FBX binary image records retain every source numeric ID encoding',()=>{
  for (const version of [7400,7500]) for (const [numberType,id] of [['L',10],['I',-10],['D',10.5],['F',10.5],['Y',-10]]) {
    const model=fbxResources([{id,relative:'paint.png'}],{binary:true,version,numberType});
    const root=fixture({'index.html':'','scene.fbx':model,'paint.png':fbxPixel});
    const requests=discover(`import {FBXLoader} from ${JSON.stringify(fbxUrl)};new FBXLoader().load('scene.fbx');`);
    const options={documentUrl:pathToFileURL(path.join(root,'index.html')).href,documentBaseUrl:pathToFileURL(root+'/').href};
    const assets=collectThreeLoaderAssets(requests,options);
    assert.deepEqual([...assets.keys()],['scene.fbx','paint.png'],`${version} ${numberType}`);
    assert.deepEqual(assets.get('scene.fbx').source,model);
  }
});

test('FBX duplicate video IDs retain each source parser rule and malformed offsets preserve original bytes',()=>{
  for (const binary of [false,true]) {
    const model=fbxResources([{id:10,relative:'first.png'},{id:10,relative:'last.png'}],{binary});
    const root=fixture({'index.html':'','scene.fbx':model,'first.png':fbxPixel,'last.png':fbxPixel});
    const requests=discover(`import {FBXLoader} from ${JSON.stringify(fbxUrl)};new FBXLoader().load('scene.fbx');`);
    const options={documentUrl:pathToFileURL(path.join(root,'index.html')).href,documentBaseUrl:pathToFileURL(root+'/').href};
    assert.deepEqual([...collectThreeLoaderAssets(requests,options).keys()],['scene.fbx',binary?'first.png':'last.png']);
    if (binary) {
      model.writeUInt32LE(0xffffffff,27);fs.writeFileSync(path.join(root,'scene.fbx'),model);
      const assets=collectThreeLoaderAssets(requests,options);
      assert.deepEqual([...assets.keys()],['scene.fbx']);assert.deepEqual(assets.get('scene.fbx').source,model);
    }
  }
});

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

test("rooted glTF sidecars preserve pinned LoaderUtils path concatenation", async () => {
  const {LoaderUtils} = await import(rootUrl + "src/loaders/LoaderUtils.js");
  assert.equal(LoaderUtils.resolveURL("/geometry.bin", "/shared/"), "/shared//geometry.bin");
  const geometry = Buffer.from([1, 2, 3, 4]);
  const model = Buffer.from(JSON.stringify({asset: {version: "2.0"}, buffers: [{uri: "/geometry.bin", byteLength: geometry.length}]}));
  const root = fixture({"pages/demo.html": "", "models/scene.gltf": model,
    "shared/geometry.bin": geometry, "geometry.bin": "wrong origin-root resolution"});
  const requests = discover(`import {GLTFLoader} from ${JSON.stringify(gltfUrl)};
    new GLTFLoader().setPath('/models/').setResourcePath('/shared/').loadAsync('scene.gltf');`);
  const logical = [], prefix = pathToFileURL(root + path.sep).href;
  const assets = collectThreeLoaderAssets(requests, {documentRoot: root,
    documentUrl: pathToFileURL(path.join(root, "pages/demo.html")).href,
    onResource: (_name, url) => logical.push(url.slice(prefix.length)),
  });
  assert.deepEqual([...assets.keys()].sort(), ["models/scene.gltf", "shared/geometry.bin"]);
  assert.deepEqual(logical, ["models/scene.gltf", "shared//geometry.bin"]);
  assert.deepEqual(assets.get("models/scene.gltf").source, model);
  assert.deepEqual(assets.get("shared/geometry.bin").source, geometry);
});

test("rooted nested-page loaders retain model sidecars and relative resources after site and standalone relocation", async t => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const model = JSON.stringify({
    asset: {version: "2.0"}, scene: 0, scenes: [{nodes: [0]}], nodes: [{mesh: 0}],
    meshes: [{primitives: [{attributes: {POSITION: 0}}]}],
    buffers: [{uri: "geometry%20%23.bin?cache=1#positions", byteLength: positions.byteLength}],
    bufferViews: [{buffer: 0, byteOffset: 0, byteLength: positions.byteLength}],
    accessors: [{bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0]}],
  });
  const resources = {
    "resources/message #.txt": "origin-root bytes", "pages/resources/local.txt": "nested document bytes",
    "models/triangle.gltf": model, "shared/geometry #.bin": Buffer.from(positions.buffer),
  };
  const root = fixture({"pages/demo.html": '<script type="module" src="/modules/app.mjs"></script>', ...resources,
    "modules/app.mjs": `import {FileLoader} from ${JSON.stringify(coreUrl)};
      import {GLTFLoader} from ${JSON.stringify(gltfUrl)};
      export async function load() {
        const text = await new FileLoader().setPath('/resources/').loadAsync('message%20%23.txt?edition=2#part');
        const relative = await new FileLoader().setPath('resources/').loadAsync('local.txt?edition=4');
        const gltf = await new GLTFLoader().setPath('/models/').setResourcePath('/shared/').loadAsync('triangle.gltf?version=3#scene');
        return {text, relative, positions: [...gltf.scene.children[0].geometry.attributes.position.array]};
      }`,
  });
  const entry = path.join(root, "pages/demo.html");
  const built = await buildApplication(entry, path.join(root, "dist"), {documentRoot: root});
  const prepared = await buildApplication(entry, path.join(root, "prepared"), {documentRoot: root, prepareStandaloneLoaders: true});
  for (const output of [built, prepared]) {
    assert.equal(output.htmlFile, "pages/demo.html");
    for (const [name, bytes] of Object.entries(resources))
      assert.deepEqual(fs.readFileSync(path.join(output.outDir, name)), Buffer.from(bytes), name);
  }
  const deployment = fixture({}), networked = path.join(deployment, "networked"), standalone = path.join(deployment, "standalone");
  fs.renameSync(built.outDir, networked); fs.renameSync(prepared.outDir, standalone);
  fs.renameSync(root, root + "-source-unavailable");
  const packedFile = path.join(deployment, "scene.html");
  packHtml(path.join(standalone, prepared.htmlFile), packedFile, {rootDir: standalone, documentRoot: standalone});
  const seen = [], server = http.createServer((request, response) => {
    seen.push(request.url);
    try { response.end(fs.readFileSync(path.join(networked, decodeURIComponent(new URL(request.url, "http://localhost").pathname)))); }
    catch { response.writeHead(404); response.end("missing resource"); }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const OriginalRequest = globalThis.Request, OriginalProgressEvent = globalThis.ProgressEvent, originalDocument = globalThis.document;
  globalThis.document = {URL: origin + "/pages/demo.html", baseURI: origin + "/pages/demo.html"};
  globalThis.Request = class extends OriginalRequest {
    constructor(input, options) { super(typeof input === "string" ? new URL(input, document.baseURI) : input, options); }
  };
  globalThis.ProgressEvent = class extends Event {
    constructor(type, details) { super(type); Object.assign(this, details); }
  };
  try {
    const expected = {text: "origin-root bytes", relative: "nested document bytes", positions: [...positions]};
    const application = await import(pathToFileURL(path.join(networked, built.entryFiles[0])).href);
    assert.deepEqual(await application.load(), expected);
    assert.deepEqual(seen, ["/resources/message%20%23.txt?edition=2", "/pages/resources/local.txt?edition=4",
      "/models/triangle.gltf?version=3", "/shared/geometry%20%23.bin?cache=1"]);
    const embedded = await import(pathToFileURL(path.join(standalone, prepared.entryFiles[0])).href);
    assert.deepEqual(await embedded.load(), expected);
    assert.equal(seen.length, 4, "Prepared root and document-relative loader resources must be embedded");
    await t.test("packed loaders preserve the original relative resource base at a different document depth", {
      skip: typeof nodeModule.registerHooks !== "function",
    }, async () => {
      const html = fs.readFileSync(packedFile, "utf8");
      const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
      const moduleUrl = /<script[^>]*type="module"[^>]*src="([^"]+)"/.exec(html)[1];
      const hooks = nodeModule.registerHooks({resolve(specifier, context, next) { return next(imports[specifier] ?? specifier, context); }});
      document.URL = document.baseURI = origin + "/relocated/to/a/different/depth/scene.html";
      try {
        const packed = await import(moduleUrl);
        assert.deepEqual(await packed.load(), expected);
        assert.equal(seen.length, 4, "Packing must not fall back to the old site or a new relative HTTP path");
      } finally { hooks.deregister(); }
    });
  } finally {
    globalThis.Request = OriginalRequest;
    if (OriginalProgressEvent === undefined) delete globalThis.ProgressEvent; else globalThis.ProgressEvent = OriginalProgressEvent;
    if (originalDocument === undefined) delete globalThis.document; else globalThis.document = originalDocument;
    server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
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

test("DDS, KTX and TGA discovery closes source texture files and immutable compressed cube faces", () => {
  const requests = discover(`import { DDSLoader as SourceDDS, KTXLoader, TGALoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import * as DDS from ${JSON.stringify(rootUrl + 'examples/jsm/loaders/DDSLoader.js')};
    const Loader = SourceDDS, faces = ['px.dds','nx.dds','py.dds','ny.dds','pz.dds','nz.dds'];
    const dds = new Loader().setPath('dds/'); dds.parse(runtimeBytes, false); dds.loadAsync('surface%20%23.dds?v=1#mip');
    dds.load(faces);
    new DDS.DDSLoader().load('single-cube.dds');
    new KTXLoader().setPath('ktx/').loadAsync(['px.ktx','nx.ktx','py.ktx','ny.ktx','pz.ktx','nz.ktx']);
    new KTXLoader().load(day ? 'day.ktx' : 'night.ktx');
    const tga = new TGALoader(); tga.parse(runtimeBytes); tga.setPath('tga/').loadAsync('paint.tga');`);
  assert.deepEqual(requests.map(({url}) => url), ['dds/surface%20%23.dds?v=1#mip',
    ...['px', 'nx', 'py', 'ny', 'pz', 'nz'].map(face => `dds/${face}.dds`), 'single-cube.dds',
    ...['px', 'nx', 'py', 'ny', 'pz', 'nz'].map(face => `ktx/${face}.ktx`),
    'day.ktx', 'night.ktx', 'tga/paint.tga']);
});

test("compressed texture discovery preserves mutable faces, foreign loaders and opaque instance behavior", () => {
  const requests = discover(`import { DDSLoader, KTXLoader, TGALoader } from ${JSON.stringify(rootUrl + 'examples/jsm/Addons.js')};
    import { DDSLoader as Foreign } from 'file:///other/examples/jsm/loaders/DDSLoader.js';
    const faces = ['a','b','c','d','e','f']; faces.reverse(); new DDSLoader().load(faces);
    const escaped = ['a','b','c','d','e','f']; expose(escaped); new KTXLoader().loadAsync(escaped);
    new DDSLoader().load(['a','b','c','d','e']);
    new TGALoader().load(['a','b','c','d','e','f']);
    new Foreign().load('foreign.dds');
    const changed = new KTXLoader(); changed.parse = otherParser; changed.load('changed.ktx');
    const owned = new DDSLoader(manager); owned.load('managed.dds');
    new DDSLoader().setPath('safe/').load('known.dds');`);
  assert.deepEqual(requests.map(({url}) => url), ['safe/known.dds']);
});

test("loose USDA sidecars use the captured source base, ignoring resourcePath and later path changes", async () => {
  const usdUrl = rootUrl + "examples/jsm/loaders/USDLoader.js";
  const model = Buffer.from('#usda 1.0\ndef Material "Paint" {\n def Shader "Image" {\n' +
    '  uniform token info:id = "UsdUVTexture"\n  asset inputs:file = @"./paint%20%23.png?edition=2#map"@\n }\n}');
  const root = fixture({"index.html": "", "site/models/nested/scene.usda": model,
    "site/flatmodel.usda": model, "site/models/paint #.png": fbxPixel,
    "site/models/nested/paint #.png": fbxPixel, "site/flat/paint #.png": fbxPixel,
    "site/ignored/paint #.png": fbxPixel, "site/later/paint #.png": fbxPixel});
  const requests = discover('import {USDLoader} from ' + JSON.stringify(usdUrl) + ';' +
    "const pending = new USDLoader().setPath('models/').setResourcePath('ignored/');" +
    "pending.load('nested/scene.usda'); pending.setPath('later/');" +
    "new USDLoader().loadAsync('models/nested/scene.usda');" +
    "new USDLoader().setPath('flat').load('model.usda');");
  const logical = [], assets = collectThreeLoaderAssets(requests, {
    documentUrl: pathToFileURL(path.join(root, "index.html")).href,
    documentBaseUrl: pathToFileURL(root + "/site/").href,
    usdAssetUris: await prepareUsdAssetUris(requests, {strict: true}),
    onResource: (_name, url) => logical.push(url),
  });
  assert.deepEqual([...assets.keys()].sort(), [
    "site/flatmodel.usda", "site/flat/paint #.png", "site/models/nested/scene.usda",
    "site/models/paint #.png", "site/models/nested/paint #.png",
  ].sort());
  assert.ok(logical.some(url => url.endsWith("models/paint%20%23.png?edition=2#map")));
  assert.ok(logical.some(url => url.endsWith("flat/paint%20%23.png?edition=2#map")));
  assert.deepEqual(assets.get("site/models/nested/scene.usda").source, model);
});

test("loose USDA images keep LoaderUtils root-relative concatenation and external host URLs", async () => {
  const model = Buffer.from('#usda 1.0\ndef Shader "Image" {\n token info:id = "UsdUVTexture"\n' +
    ' asset inputs:file = @/paint.png@\n}\ndef Shader "Remote" {\n token info:id = "arnold:image"\n' +
    ' string inputs:filename = "https://images.example/remote.png"\n}');
  const root = fixture({"pages/demo.html": "", "models/scene.usda": model, "models/paint.png": fbxPixel,
    "paint.png": fbxPixel});
  const requests = discover('import {USDLoader} from ' + JSON.stringify(rootUrl + "examples/jsm/loaders/USDLoader.js") + ';' +
    "new USDLoader().setPath('/models/').loadAsync('scene.usda');");
  const logical = [], assets = collectThreeLoaderAssets(requests, {
    documentRoot: root, documentUrl: pathToFileURL(path.join(root, "pages/demo.html")).href,
    usdAssetUris: await prepareUsdAssetUris(requests, {strict: true}),
    onResource: (_name, url) => logical.push(url),
  });
  assert.deepEqual([...assets.keys()].sort(), ["models/paint.png", "models/scene.usda"]);
  assert.ok(logical.some(url => url.endsWith("/models//paint.png")));
});
