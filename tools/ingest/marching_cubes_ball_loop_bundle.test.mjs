/** The original H2 update loop passes through the production Rollup plugin, with real
 * Three objects and executable Wasm. Texture loaders stop at their resource
 * boundary; this suite checks public numeric state, not rendered pixels. */
import assert from 'node:assert/strict';
import {before, after, test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import * as acorn from 'acorn';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {buildApplication} from './build_application.mjs';
import {OBJExporter} from '../../upstream/three.js/examples/jsm/exporters/OBJExporter.js';

const upstream = fileURLToPath(new URL('../../upstream/three.js/', import.meta.url));
let root, built, actual, expected;

async function applicationSource(diagnostics) {
  const html = await fs.readFile(path.join(upstream, 'examples/webgl_marchingcubes.html'), 'utf8');
  const source = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
  const ast = acorn.parse(source, {ecmaVersion: 'latest', sourceType: 'module'});
  const functions = ast.body.filter(node => node.type === 'FunctionDeclaration' &&
    ['generateMaterials', 'createShaderMaterial', 'updateCubes'].includes(node.id.name));
  assert.equal(functions.length, 3);
  return `import * as Library from 'three';
import {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
export {Color, ColorManagement} from 'three';
import {ToonShader1,ToonShader2,ToonShaderHatching,ToonShaderDotted} from 'three/addons/shaders/ToonShader.js';
${diagnostics ? `export {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};` : ''}
// Only asset loading is replaced. The functions below are copied verbatim
// from the pinned application, including all ball formulas and color branches.
const THREE = {
  Color: Library.Color, UniformsUtils: Library.UniformsUtils,
  DirectionalLight: Library.DirectionalLight, AmbientLight: Library.AmbientLight,
  MeshStandardMaterial: Library.MeshStandardMaterial, MeshLambertMaterial: Library.MeshLambertMaterial,
  MeshPhongMaterial: Library.MeshPhongMaterial, ShaderMaterial: Library.ShaderMaterial,
  CubeRefractionMapping: Library.CubeRefractionMapping, RepeatWrapping: Library.RepeatWrapping,
  SRGBColorSpace: Library.SRGBColorSpace,
  CubeTextureLoader: class { load() { return new Library.CubeTexture(); } },
  TextureLoader: class { load() { return new Library.Texture(); } }
};
const light = new THREE.DirectionalLight(0xffffff, 3);
light.position.set(.5, .5, 1);
const ambientLight = new THREE.AmbientLight(0x323232, 3);
let current_material = 'shiny';
${functions.map(node => source.slice(node.start, node.end)).join('\n')}
export function create(resolution = 14) {
  const materials = generateMaterials();
  const effect = new MarchingCubes(resolution, materials.shiny, true, true, 20000);
  effect.name = 'h2-ball-loop';
  return {effect, materials};
}
export function select(application, name) {
  current_material = name;
  application.effect.material = application.materials[name];
  application.effect.enableUvs = name === 'textured';
  application.effect.enableColors = name === 'colors' || name === 'multiColors';
}
export function update(application, time, count, floor = false, wallx = false, wallz = false) {
  return updateCubes(application.effect, time, count, floor, wallx, wallz);
}
`;
}

async function execute(directory, entry) {
  return import(pathToFileURL(path.join(directory, entry)).href);
}
before(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), 'f3d-ball-loop-build-'));
  for (const [name, specializeNumeric] of [['actual', true], ['expected', false]]) {
    const entry = path.join(root, `${name}.mjs`);
    await fs.writeFile(entry, await applicationSource(specializeNumeric), {flag: 'wx'});
    const plugin = specializeNumeric ? marchingCubesRollupPlugin() : null;
    const bundle = await rollup({input: entry, plugins: [{name: 'pinned-h2-resource-boundary',
      resolveId(id) {
        if (id === 'three') return path.join(upstream, 'build/three.core.js');
        if (id.startsWith('three/addons/')) return path.join(upstream, 'examples/jsm', id.slice('three/addons/'.length));
        return null;
      },
    }, ...(plugin ? [plugin] : [])]});
    const outDir = path.join(root, name);
    let output;
    try { output = await bundle.write({dir: outDir, format: 'es', entryFileNames: 'index.mjs'}); }
    finally { await bundle.close(); }
    const build = {outDir, entryFiles: output.output.filter(file => file.type === 'chunk' && file.isEntry).map(file => file.fileName),
      report: plugin?.api.getReport()};
    await fs.writeFile(path.join(build.outDir, 'package.json'), '{"type":"module"}', {flag: 'wx'});
    if (specializeNumeric) {
      built = build;
      // Execute a moved copy of the whole product output, never a test-only
      // invocation of the recognizer or adapter.
      const relocated = path.join(root, 'relocated');
      await fs.cp(build.outDir, relocated, {recursive: true, errorOnExist: true});
      actual = await execute(relocated, build.entryFiles[0]);
    } else expected = await execute(build.outDir, build.entryFiles[0]);
  }
});
after(async () => { if (root) await fs.rm(root, {recursive: true, force: true}); });

function equal(a, b) {
  assert.equal(a.count, b.count, 'public vertex count');
  assert.deepEqual(a.geometry.drawRange, b.geometry.drawRange);
  for (const key of ['field', 'normal_cache', 'palette', 'positionArray', 'normalArray', 'uvArray', 'colorArray'])
    assert.deepEqual(a[key], b[key], key);
  for (const [name, key] of [['position', 'positionArray'], ['normal', 'normalArray'], ['uv', 'uvArray'], ['color', 'colorArray']]) {
    const attribute = a.geometry.getAttribute(name), reference = b.geometry.getAttribute(name);
    assert.equal(attribute.array, a[key], `${name} keeps the public array identity`);
    assert.equal(attribute.version, reference.version, `${name} upload version`);
  }
}
function pair(resolution = 14) { return [actual, expected].map(module => module.create(resolution)); }
function update(a, b, ...args) {
  assert.equal(actual.update(a, ...args), undefined);
  assert.equal(expected.update(b, ...args), undefined);
  equal(a.effect, b.effect);
}

test('production Rollup builds automatically batch the original loop and relocate executable output', () => {
  const report = built.report;
  assert.equal(report.compiledBallLoops, 1);
  assert.ok(report.modules.some(module => module.compiledBallBatch === true));
  assert.equal(report.accelerationClaim, false);
  const [a, b] = pair();
  update(a, b, .125, 50);
  const diagnostics = actual.diagnostics(a.effect);
  assert.equal(diagnostics.ballLoops.wasmCalls, 1);
  assert.equal(diagnostics.ballLoops.balls, 50);
  assert.equal(diagnostics.ballLoops.fallbackCalls, 0);
  assert.equal(diagnostics.fieldKernels.addBall?.wasmCalls ?? 0, 0,
    'one batch replaces fifty independent field transfers');
  assert.equal(diagnostics.wasmCalls, 1, 'polygonization still publishes synchronously');
  assert.ok(diagnostics.ballLoops.kernel.copiedBytes > 0);
});

test('all thirteen original H2 material choices retain exact public geometry and one ball transaction', () => {
  const [a, b] = pair(), methods = [a.effect.addBall, a.effect.update];
  const arrays = [a.effect.field, a.effect.palette, a.effect.positionArray, a.effect.normalArray];
  const names = Object.keys(a.materials);
  assert.equal(names.length, 13);
  for (const [index, name] of names.entries()) {
    actual.select(a, name); expected.select(b, name);
    update(a, b, index / 13, [1, 4, 17, 50][index % 4], index % 2 === 0, index % 3 === 0, index % 4 === 0);
    assert.equal(actual.diagnostics(a.effect).ballLoops.wasmCalls, index + 1, name);
  }
  assert.equal(actual.diagnostics(a.effect).ballLoops.fallbackCalls, 0);
  assert.equal(a.effect.addBall, methods[0]); assert.equal(a.effect.update, methods[1]);
  for (const [index, key] of ['field', 'palette', 'positionArray', 'normalArray'].entries())
    assert.equal(a.effect[key], arrays[index]);
});

test('resolution, isolation, color and wall changes remain synchronously visible to exporters', () => {
  const [a, b] = pair(), exporter = new OBJExporter();
  for (const [index, resolution] of [8, 18, 14].entries()) {
    for (const application of [a, b]) {
      application.effect.init(resolution);
      application.effect.isolation = 60 + 15 * index;
      application.effect.position.set(2, -3, 4);
      application.effect.scale.set(3, 2, -1);
    }
    const name = index === 1 ? 'multiColors' : 'textured';
    actual.select(a, name); expected.select(b, name);
    update(a, b, .375 * index, 12 + index, true, index === 1, index === 2);
    a.effect.updateMatrixWorld(true); b.effect.updateMatrixWorld(true);
    assert.equal(exporter.parse(a.effect), exporter.parse(b.effect));
  }
  assert.equal(actual.diagnostics(a.effect).ballLoops.wasmCalls, 3);
});

test('an observed public addBall override runs once per ball and can return to the batch', () => {
  const [a, b] = pair(8), trace = [[], []], methods = [a.effect.addBall, b.effect.addBall];
  for (const [index, application] of [a, b].entries()) application.effect.addBall = function (...args) {
    trace[index].push(args.slice(0, 5));
    return methods[index].apply(this, args);
  };
  update(a, b, .25, 7, true);
  assert.deepEqual(trace[0], trace[1]); assert.equal(trace[0].length, 7);
  assert.equal(actual.diagnostics(a.effect).ballLoops.wasmCalls, 0);
  for (const [index, application] of [a, b].entries()) application.effect.addBall = methods[index];
  update(a, b, .5, 7, true);
  assert.equal(actual.diagnostics(a.effect).ballLoops.wasmCalls, 1);
});

test('Color, color management, Math and field observers keep the original source effect counts', () => {
  for (const mode of ['set', 'setRGB', 'colorSpaceToWorking', 'sin', 'abs', 'field']) {
    const [a, b] = pair(8), counts = [0, 0], restores = [];
    let side = 0;
    const replace = (object, key, index) => {
      const original = Object.getOwnPropertyDescriptor(object, key);
      const replacement = key === 'field'
        ? {get() { counts[index]++; return original.value; }}
        : {value: function (...args) { counts[index ?? side]++; return original.value.apply(this, args); }, writable: true};
      Object.defineProperty(object, key, {configurable: true, enumerable: original.enumerable, ...replacement});
      restores.push(() => Object.defineProperty(object, key, original));
    };
    if (mode === 'sin' || mode === 'abs') replace(Math, mode, null);
    else for (const [index, module] of [actual, expected].entries()) {
      const object = mode === 'field' ? [a, b][index].effect :
        mode === 'colorSpaceToWorking' ? module.ColorManagement : module.Color.prototype;
      replace(object, mode, index);
    }
    try {
      actual.update(a, .25, 5, true); side = 1;
      expected.update(b, .25, 5, true);
    } finally { for (const restore of restores.reverse()) restore(); }
    assert.ok(counts[0] > 0, `${mode} observer actually runs`);
    assert.equal(counts[0], counts[1], `${mode} effects execute exactly as often as the original`);
    equal(a.effect, b.effect);
    assert.equal(actual.diagnostics(a.effect).ballLoops.wasmCalls, 0, mode);
    assert.equal(actual.diagnostics(a.effect).ballLoops.fallbackCalls, 1, mode);
    update(a, b, .5, 5, true);
    assert.equal(actual.diagnostics(a.effect).ballLoops.wasmCalls, 1, `${mode} can recover`);
  }
});

test('relocated builds preserve the complete original update when WebAssembly is unavailable', async () => {
  const directory = path.join(root, 'no-wasm');
  await fs.cp(built.outDir, directory, {recursive: true, errorOnExist: true});
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'WebAssembly');
  let unavailable, a;
  const b = expected.create(8); expected.select(b, 'multiColors');
  try {
    Object.defineProperty(globalThis, 'WebAssembly', {...descriptor, value: undefined});
    unavailable = await execute(directory, built.entryFiles[0]);
    a = unavailable.create(8); unavailable.select(a, 'multiColors');
    assert.equal(unavailable.update(a, .625, 9, true, false, true), undefined);
  } finally { Object.defineProperty(globalThis, 'WebAssembly', descriptor); }
  expected.update(b, .625, 9, true, false, true); equal(a.effect, b.effect);
  assert.equal(unavailable.diagnostics(a.effect).ballLoops.wasmCalls, 0);
  assert.equal(unavailable.diagnostics(a.effect).ballLoops.fallbackCalls, 1);
});

test('the complete application build wires the ball batch through generic numeric specialization', async () => {
  const entry = path.join(root, 'application.mjs');
  await fs.writeFile(entry, await applicationSource(true), {flag: 'wx'});
  const build = await buildApplication(entry, path.join(root, 'application-output'), {
    packageRootUrl: pathToFileURL(upstream).href, specializeNumeric: true,
  });
  const metadata = path.join(build.outDir, 'package.json');
  try { await fs.writeFile(metadata, '{"type":"module"}', {flag: 'wx'}); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    assert.equal(JSON.parse(await fs.readFile(metadata, 'utf8')).type, 'module');
  }
  const relocated = path.join(root, 'application-relocated');
  await fs.cp(build.outDir, relocated, {recursive: true, errorOnExist: true});
  const module = await execute(relocated, build.entryFiles[0]);
  const a = module.create(), b = expected.create();
  module.select(a, 'multiColors'); expected.select(b, 'multiColors');
  module.update(a, .375, 50, true, true, false);
  expected.update(b, .375, 50, true, true, false);
  equal(a.effect, b.effect);
  assert.equal(build.numericSpecialization.libraryKernels.marchingCubes.compiledBallLoops, 1);
  assert.equal(module.diagnostics(a.effect).ballLoops.wasmCalls, 1);
  assert.equal(module.diagnostics(a.effect).ballLoops.balls, 50);
  assert.equal(module.diagnostics(a.effect).ballLoops.fallbackCalls, 0);
});
