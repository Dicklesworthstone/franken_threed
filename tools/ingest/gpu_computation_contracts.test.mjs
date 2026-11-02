import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath, pathToFileURL} from 'node:url';
import test from 'node:test';
import {buildModuleGraph} from './module_graph.mjs';
import {evaluateGraphRoutes} from './route_bridge.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';
import {transformThreeGpuComputation} from './gpu_computation_contracts.mjs';
import {registerThreeGpuComputation, supportsThreeGpuComputation} from './three_gpu_computation.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const three = new URL('build/three.module.js', upstream).href;
const addon = new URL('examples/jsm/misc/GPUComputationRenderer.js', upstream).href;
const source = fs.readFileSync(new URL(addon), 'utf8');
const environment = {generalWebGPUAvailable: true, generalWebGLAvailable: true,
  hostCapabilities: {hasWebGPU: true, hasWebGL: true}};
const header = `import * as THREE from 'three'; import {GPUComputationRenderer} from 'compute';
const renderer = new THREE.WebGLRenderer(); const gpu = new GPUComputationRenderer(8, 4, renderer);\n`;
function fixture(body = '', {imports = {}, modules = {}, prefix = header} = {}) {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-gpu-computation-proof-'));
  for (const [name, code] of Object.entries(modules)) {
    fs.mkdirSync(path.dirname(path.join(dir, name)), {recursive: true});
    fs.writeFileSync(path.join(dir, name), code);
  }
  const entry = path.join(dir, 'index.html');
  fs.writeFileSync(entry, '<script type="importmap">' + JSON.stringify({imports: {three, compute: addon, ...imports}}) +
    '</script><script type="module">' + prefix + body + '</script>');
  return {dir, entry};
}
const route = graph => evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route;

test('private compute capability does not fabricate observable WebGL limits', () => {
  let reads = 0;
  const exact = {get capabilities() { reads++; return {maxVertexTextures: 0}; }};
  assert.equal(supportsThreeGpuComputation(exact), false);
  assert.equal(reads, 1);
  for (const value of [1, 16, undefined, null, false])
    assert.equal(supportsThreeGpuComputation({capabilities: {maxVertexTextures: value}}), true);
  const error = new Error('source capability getter');
  const native = {isF3DRenderer: true, get capabilities() { throw error; }};
  assert.throws(() => supportsThreeGpuComputation(native), caught => caught === error,
    'public flags do not establish native renderer identity');
  registerThreeGpuComputation(native);
  assert.equal(supportsThreeGpuComputation(native), true);
});

test('pinned computation retains closed source methods, variables, aliases and fluent configuration', async () => {
  const f = fixture(`const local = gpu.setDataType(THREE.HalfFloatType);
const initial = local.createTexture();
const position = local.addVariable('position', 'void main(){gl_FragColor=vec4(1.0);}', initial);
position.material.uniforms.time = {value: 1};
local.setVariableDependencies(position, [position]);
const error = local.init(); local.compute();
const current = local.getCurrentRenderTarget(position), previous = local.getAlternateRenderTarget(position);
const texture = current.texture; const variables = local.variables; const index = local.currentTextureIndex;
local.currentTextureIndex = 0;
local.renderTexture(initial, previous); local.doRenderTarget(position.material, current);
local.addResolutionDefine(position.material); local.createRenderTarget(); local.createShaderMaterial('void main(){}');
const originalClass = local instanceof GPUComputationRenderer; local.dispose();`);
  const graph = await buildModuleGraph(f.entry);
  assert.equal(route(graph), 'general-webgpu');
  const record = graph.modules[addon].gpu_computation_transform;
  assert.equal(record.name, 'GPUComputationRenderer.js');
  const transformed = transformThreeGpuComputation(source, record);
  assert.ok(transformed.includes('!__f3dSupportsGpuComputation(renderer)'));
  assert.ok(transformed.includes('class GPUComputationRenderer'));
  assert.throws(() => transformThreeGpuComputation(source + '\n', record), /source changed/);
});

test('namespace construction and a local factory keep native renderer ownership', async () => {
  const f = fixture('gpu.compute();', {prefix: `import * as THREE from 'three'; import * as G from 'compute';
function make(renderer) { return new G.GPUComputationRenderer(8, 4, renderer).setDataType(THREE.FloatType); }
const renderer = new THREE.WebGLRenderer(); const gpu = make(renderer);\n`});
  assert.equal(route(await buildModuleGraph(f.entry)), 'general-webgpu');
});

test('escaped instances, replaced methods, constructors and reflection keep exact ownership', async () => {
  for (const body of [
    'globalThis.compute = gpu;', 'globalThis.plugin(gpu);', 'gpu.compute = globalThis.plugin;',
    'const method = gpu.compute; method();', 'gpu.variables = globalThis.opaque;',
    'GPUComputationRenderer.prototype.compute = globalThis.plugin;',
    'Object.defineProperty(gpu, "compute", {value: globalThis.plugin});',
    'globalThis.context.createTexture();',
  ]) {
    const f = fixture(body);
    assert.equal(route(await buildModuleGraph(f.entry)), 'exact-backend', body);
  }
});

test('changed addon or resolved dependency bytes cannot authorize native lowering', async () => {
  for (const changed of ['addon', 'pass', 'core']) {
    const originalPass = fs.readFileSync(new URL('examples/jsm/postprocessing/Pass.js', upstream), 'utf8');
    const f = fixture('', {imports: {compute: './misc/renamed.js', three: './three.module.js'}, modules: {
      'misc/renamed.js': source + (changed === 'addon' ? '\n// changed' : ''),
      'postprocessing/Pass.js': originalPass + (changed === 'pass' ? '\n// changed' : ''),
      'three.module.js': fs.readFileSync(new URL(three), 'utf8'),
      'three.core.js': fs.readFileSync(new URL('build/three.core.js', upstream), 'utf8') + (changed === 'core' ? '\n// changed' : ''),
    }});
    const graph = await buildModuleGraph(f.entry);
    assert.equal(route(graph), 'exact-backend', changed);
    assert.equal(Object.values(graph.modules).some(mod => mod.gpu_computation_transform), false);
  }
});

test('identical relocated source is admitted by its dependency bytes; open extensions stay exact', async () => {
  const f = fixture('gpu.init();', {imports: {compute: './misc/renamed.js'}, modules: {
    'misc/renamed.js': source,
    'postprocessing/Pass.js': fs.readFileSync(new URL('examples/jsm/postprocessing/Pass.js', upstream), 'utf8'),
  }});
  const graph = await buildModuleGraph(f.entry);
  assert.equal(route(graph), 'general-webgpu');
  assert.ok(graph.modules[pathToFileURL(path.join(f.dir, 'misc/renamed.js')).href].gpu_computation_transform);
  const open = fixture('import(location.hash.slice(1)); gpu.compute();');
  const openGraph = await buildModuleGraph(open.entry);
  assert.equal(route(openGraph), 'exact-backend');
  assert.equal(Object.values(openGraph.modules).some(mod => mod.gpu_computation_transform), false);
});
