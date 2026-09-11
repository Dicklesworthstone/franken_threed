/**
 * @file tools/compat-facade/facade.test.mjs
 * Unit test suite for Three.js r186 export routing and compatibility facades (f3d-04.2).
 *
 * Verifies:
 * 1. Enumerates every pinned r186 package export entry (root ESM/CJS, webgpu, tsl, addons, src).
 * 2. Export surface extraction accurately resolves named and default exports across modules.
 * 3. Every generated ESM facade re-exports exactly the upstream export names (Acorn parse-only, no Rollup).
 * 4. Root CommonJS facade synchronously re-exports without Promises or GPU/DOM access.
 * 5. CPU-only entry points (math, geometry, exporters) have zero import paths to GPU/DOM/Wasm.
 * 6. Attestations truthfully record retained upstream ownership without GPU acceleration claims.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

import {
  FACADE_NO_CLAIM_ATTESTATION,
  categorizeExportEntry,
  enumeratePackageExportEntries,
  extractModuleExportSurface,
  generateFacadeModule,
  parseFacadeExportSurface,
  buildFacadeModuleMap,
} from './index.mjs';

test('Positive: enumeratePackageExportEntries covers all 1778 pinned r186 package export entries', () => {
  const entries = enumeratePackageExportEntries();
  assert.equal(entries.length, 1778, 'Must enumerate all 1778 reconciled export entries');

  const categories = {};
  for (const e of entries) {
    categories[e.category] = (categories[e.category] || 0) + 1;
  }

  assert.equal(categories.root_esm, 1, 'Root ESM entry must exist');
  assert.equal(categories.root_cjs, 1, 'Root CJS entry must exist');
  assert.equal(categories.webgpu, 1, 'three/webgpu entry must exist');
  assert.equal(categories.tsl, 1, 'three/tsl entry must exist');
  assert.equal(categories.addons, 1, 'three/addons root entry must exist');
  assert.equal(categories.addons_wildcard, 494, 'three/addons/* wildcard code exports must be 494');
  assert.equal(categories.src_wildcard, 753, 'three/src/* wildcard code exports must be 753');
  assert.equal(categories.examples_jsm_wildcard, 494, 'examples/jsm/* wildcard code exports must be 494');
  assert.equal(categories.asset, 32, 'Asset entries (fonts, wasm, text) must be 32');

  // Verify key canonical paths
  const rootEsm = entries.find(e => e.exportKey === '.' && e.condition === 'import');
  assert.ok(rootEsm);
  assert.equal(rootEsm.target, 'build/three.module.js');

  const rootCjs = entries.find(e => e.exportKey === '.' && e.condition === 'require');
  assert.ok(rootCjs);
  assert.equal(rootCjs.target, 'build/three.cjs');

  const webgpu = entries.find(e => e.exportKey === './webgpu');
  assert.ok(webgpu);
  assert.equal(webgpu.target, 'build/three.webgpu.js');

  const tsl = entries.find(e => e.exportKey === './tsl');
  assert.ok(tsl);
  assert.equal(tsl.target, 'build/three.tsl.js');

  const addons = entries.find(e => e.exportKey === './addons');
  assert.ok(addons);
  assert.equal(addons.target, 'examples/jsm/Addons.js');
});

test('Positive: extractModuleExportSurface extracts accurate export surface for core bundles', () => {
  const cache = new Map();

  // 1. three.module.js (root ESM)
  const threeModule = extractModuleExportSurface('upstream/three.js/build/three.module.js', { cache });
  assert.equal(threeModule.hasDefault, false);
  assert.equal(threeModule.named.length, 444, 'three.module.js must export exactly 444 symbols');
  assert.ok(threeModule.named.includes('Mesh'));
  assert.ok(threeModule.named.includes('BoxGeometry'));
  assert.ok(threeModule.named.includes('WebGLRenderer'));
  assert.ok(threeModule.named.includes('Vector3'));

  // 2. three.webgpu.js
  const threeWebgpu = extractModuleExportSurface('upstream/three.js/build/three.webgpu.js', { cache });
  assert.equal(threeWebgpu.hasDefault, false);
  assert.equal(threeWebgpu.named.length, 635, 'three.webgpu.js must export exactly 635 symbols');
  assert.ok(threeWebgpu.named.includes('WebGPURenderer'));

  // 3. three.tsl.js
  const threeTsl = extractModuleExportSurface('upstream/three.js/build/three.tsl.js', { cache });
  assert.equal(threeTsl.hasDefault, false);
  assert.equal(threeTsl.named.length, 682, 'three.tsl.js must export exactly 682 symbols');

  // 4. Addons.js
  const addons = extractModuleExportSurface('upstream/three.js/examples/jsm/Addons.js', { cache });
  assert.equal(addons.hasDefault, false);
  assert.equal(addons.named.length, 330, 'Addons.js must aggregate 330 addon exports');
  assert.ok(addons.named.includes('OrbitControls'));
  assert.ok(addons.named.includes('GLTFExporter'));

  // 5. Individual addon module with named-only export
  const orbitControls = extractModuleExportSurface('upstream/three.js/examples/jsm/controls/OrbitControls.js', { cache });
  assert.equal(orbitControls.hasDefault, false);
  assert.deepEqual(orbitControls.named, ['OrbitControls']);

  // 6. Individual addon module with default export
  const webglCapability = extractModuleExportSurface('upstream/three.js/examples/jsm/capabilities/WebGL.js', { cache });
  assert.equal(webglCapability.hasDefault, true);
});

test('Core: Every generated ESM facade re-exports exactly the upstream export names (Acorn parse-only, no Rollup)', () => {
  const result = buildFacadeModuleMap();

  assert.equal(result.summary.esm_entries, 1745);
  assert.equal(result.summary.cjs_entries, 1);
  assert.equal(result.summary.asset_entries, 32);

  let verifiedCount = 0;
  const failures = [];

  for (const [key, item] of result.map.entries()) {
    if (item.moduleType !== 'esm') continue;

    // Parse the generated facade module using Acorn (parse-only, no bundler)
    const parsed = parseFacadeExportSurface(item.facadeSource, 'esm');

    // 1. Assert named exports match exactly
    const expectedNamed = new Set(item.exportSurface.named);
    const actualNamed = new Set(parsed.named);

    if (expectedNamed.size !== actualNamed.size) {
      failures.push({ key, reason: `Named export count mismatch: expected ${expectedNamed.size}, got ${actualNamed.size}` });
      continue;
    }

    let namesMatch = true;
    for (const name of expectedNamed) {
      if (!actualNamed.has(name)) {
        failures.push({ key, reason: `Missing named export: ${name}` });
        namesMatch = false;
        break;
      }
    }
    if (!namesMatch) continue;

    // 2. Assert default export presence matches exactly
    if (parsed.hasDefault !== item.exportSurface.hasDefault) {
      failures.push({ key, reason: `Default export mismatch: expected ${item.exportSurface.hasDefault}, got ${parsed.hasDefault}` });
      continue;
    }

    // 3. Assert wildcard export is present
    if (!parsed.hasWildcard) {
      failures.push({ key, reason: 'Missing export * wildcard declaration' });
      continue;
    }

    verifiedCount++;
  }

  assert.equal(failures.length, 0, `All ESM facades must match export surface exactly. Failures: ${JSON.stringify(failures.slice(0, 3))}`);
  assert.equal(verifiedCount, 1745, 'All 1745 ESM facades verified successfully');
});

test('Core: Root CommonJS facade synchronously re-exports without Promises or GPU access', () => {
  const result = buildFacadeModuleMap();
  const cjsEntry = result.map.get('.#require');

  assert.ok(cjsEntry, 'Root CJS entry must exist');
  assert.equal(cjsEntry.moduleType, 'cjs');
  assert.equal(cjsEntry.condition, 'require');
  assert.equal(cjsEntry.target, 'build/three.cjs');

  const parsed = parseFacadeExportSurface(cjsEntry.facadeSource, 'cjs');
  assert.equal(parsed.isCJS, true);
  assert.equal(parsed.assignsModuleExports, true, 'Must assign to module.exports');
  assert.equal(parsed.callsRequire, true, 'Must call require(...) synchronously');

  // Verify no Promise, async, or GPU references in source
  assert.ok(!cjsEntry.facadeSource.includes('Promise'), 'CJS facade must not create or return Promises');
  assert.ok(!cjsEntry.facadeSource.includes('async'), 'CJS facade must be synchronous');
  assert.ok(!cjsEntry.facadeSource.includes('navigator.gpu'), 'CJS facade must not reference navigator.gpu');
  assert.ok(!cjsEntry.facadeSource.includes('window'), 'CJS facade must not reference window');
  assert.ok(!cjsEntry.facadeSource.includes('document'), 'CJS facade must not reference document');
});

test('Environment Gating: CPU-only entries have zero GPU/DOM/Wasm import paths', () => {
  const result = buildFacadeModuleMap();

  const cpuEntries = [
    './src/math/Vector3.js#default',
    './src/math/Matrix4.js#default',
    './src/math/Quaternion.js#default',
    './src/geometries/BoxGeometry.js#default',
    './addons/curves/CurveExtras.js#default',
  ];

  for (const key of cpuEntries) {
    const item = result.map.get(key);
    assert.ok(item, `CPU entry ${key} must exist in facade map`);
    assert.ok(item.facadeSource, `Facade source for ${key} must be generated`);

    // Verify facade imports strictly its retained upstream target
    assert.ok(
      !item.facadeSource.includes('@franken_threed/gpu'),
      `CPU entry ${key} must not import GPU runtime`
    );
    assert.ok(
      !item.facadeSource.includes('.wasm'),
      `CPU entry ${key} must not import Wasm modules`
    );
    assert.ok(
      !item.facadeSource.includes('navigator.gpu'),
      `CPU entry ${key} must not touch navigator.gpu`
    );
    assert.ok(
      !item.facadeSource.includes('document'),
      `CPU entry ${key} must not touch DOM document`
    );
  }
});

test('Attestations: Every facade contains truthful no-claim notice without GPU acceleration claims', () => {
  const result = buildFacadeModuleMap();

  for (const [key, item] of result.map.entries()) {
    assert.equal(
      item.noClaimAttestation,
      FACADE_NO_CLAIM_ATTESTATION,
      `Item ${key} must carry standard no-claim attestation`
    );

    if (item.facadeSource) {
      assert.ok(
        item.facadeSource.includes('does not claim GPU acceleration'),
        `Facade source for ${key} must include no-claim notice`
      );
      assert.ok(
        item.facadeSource.includes('Retained upstream JS execution'),
        `Facade source for ${key} must state retained ownership`
      );
    }
  }
});

test('Integrity: All 1778 enumerated target files actually exist on disk in upstream/three.js', () => {
  const entries = enumeratePackageExportEntries();

  let missingCount = 0;
  for (const e of entries) {
    if (!fs.existsSync(e.retainedAbsolutePath)) {
      missingCount++;
    }
  }

  assert.equal(missingCount, 0, 'Zero missing target files on disk');
});

test('Emitted Facades: emitFacadeFiles successfully writes valid Acorn-parseable facades to disk', async () => {
  const { emitFacadeFiles } = await import('./index.mjs');
  const tempDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'f3d_facade_emit_'));

  try {
    const outcome = await emitFacadeFiles(tempDir);
    assert.equal(outcome.count, 1746, 'Must emit 1746 code facade files (1745 ESM + 1 CJS)');
    assert.ok(outcome.emittedFiles.length === 1746);

    // Spot-check key emitted files
    const acorn = await import('acorn');

    const rootEsmFile = path.resolve(tempDir, 'index.js');
    assert.ok(fs.existsSync(rootEsmFile), 'Root ESM facade file must exist');
    const rootEsmCode = fs.readFileSync(rootEsmFile, 'utf-8');
    const parsedEsm = acorn.parse(rootEsmCode, { ecmaVersion: 'latest', sourceType: 'module' });
    assert.ok(parsedEsm.body.length > 0);

    const rootCjsFile = path.resolve(tempDir, 'index.cjs');
    assert.ok(fs.existsSync(rootCjsFile), 'Root CJS facade file must exist');
    const rootCjsCode = fs.readFileSync(rootCjsFile, 'utf-8');
    const parsedCjs = acorn.parse(rootCjsCode, { ecmaVersion: 'latest', sourceType: 'script' });
    assert.ok(parsedCjs.body.length > 0);

    const webgpuFile = path.resolve(tempDir, 'webgpu.js');
    assert.ok(fs.existsSync(webgpuFile));

    const tslFile = path.resolve(tempDir, 'tsl.js');
    assert.ok(fs.existsSync(tslFile));

    const orbitFile = path.resolve(tempDir, 'addons/controls/OrbitControls.js');
    assert.ok(fs.existsSync(orbitFile));
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('Node Execution: require() CJS and import() root ESM, webgpu, tsl, and addons facades with strict guards', async () => {
  const { buildFacadeModuleMap, emitFacadeFiles } = await import('./index.mjs');
  const { createRequire } = await import('node:module');
  const { register } = await import('node:module');
  const { pathToFileURL } = await import('node:url');

  // Register loader hook for https: URLs used by upstream TTFLoader in Addons.js
  const hookCode = `
    export async function load(url, context, nextLoad) {
      if (url.startsWith('https:')) {
        return {
          format: 'module',
          source: 'export default {}; export const createFile = () => {}; export const DataStream = () => {};',
          shortCircuit: true
        };
      }
      return nextLoad(url, context);
    }
  `;
  register('data:text/javascript,' + encodeURIComponent(hookCode));

  const tempDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'f3d_facade_node_'));
  try {
    await emitFacadeFiles(tempDir);
    const moduleMapResult = buildFacadeModuleMap();

    // 1. require() root CJS facade in Node
    const require = createRequire(import.meta.url);
    const rootCjsPath = path.resolve(tempDir, 'index.cjs');
    const cjsThree = require(rootCjsPath);

    assert.ok(cjsThree && typeof cjsThree === 'object', 'Root CJS require must return an object');
    assert.equal(typeof cjsThree.Vector3, 'function', 'CJS export must include Vector3 constructor');
    assert.equal(typeof cjsThree.Mesh, 'function', 'CJS export must include Mesh constructor');
    assert.equal(typeof cjsThree.WebGLRenderer, 'function', 'CJS export must include WebGLRenderer constructor');
    assert.ok(!(cjsThree instanceof Promise), 'CJS entry must be strictly synchronous, never a Promise');

    // Assert CJS exports match the map
    const rootEsmMapItem = moduleMapResult.map.get('.#import');
    const cjsKeys = Object.keys(cjsThree);
    assert.equal(cjsKeys.length, 444, 'CJS export keys count must match the 444 mapped symbols');
    for (const name of rootEsmMapItem.exportSurface.named) {
      assert.ok(name in cjsThree, `CJS export missing mapped symbol: ${name}`);
    }

    // 2. import() root ESM facade in Node
    const rootEsmPath = path.resolve(tempDir, 'index.js');
    const esmThree = await import(pathToFileURL(rootEsmPath).href);
    assert.equal(typeof esmThree.Vector3, 'function');
    assert.equal(typeof esmThree.Mesh, 'function');
    const actualEsmKeys = Object.keys(esmThree).filter(k => k !== 'default').sort();
    assert.deepEqual(actualEsmKeys, rootEsmMapItem.exportSurface.named, 'Root ESM export names must match map');

    // 3. import() three/webgpu facade in Node
    const webgpuMapItem = moduleMapResult.map.get('./webgpu#default');
    const webgpuPath = path.resolve(tempDir, 'webgpu.js');
    const webgpuThree = await import(pathToFileURL(webgpuPath).href);
    assert.equal(typeof webgpuThree.WebGPURenderer, 'function');
    const actualWebgpuKeys = Object.keys(webgpuThree).filter(k => k !== 'default').sort();
    assert.deepEqual(actualWebgpuKeys, webgpuMapItem.exportSurface.named, 'WebGPU export names must match map');

    // 4. import() three/tsl facade in Node
    const tslMapItem = moduleMapResult.map.get('./tsl#default');
    const tslPath = path.resolve(tempDir, 'tsl.js');
    const tslThree = await import(pathToFileURL(tslPath).href);
    assert.equal(typeof tslThree.Fn, 'function');
    const actualTslKeys = Object.keys(tslThree).filter(k => k !== 'default').sort();
    assert.deepEqual(actualTslKeys, tslMapItem.exportSurface.named, 'TSL export names must match map');

    // 5. import() three/addons facade in Node
    const addonsMapItem = moduleMapResult.map.get('./addons#default');
    const addonsPath = path.resolve(tempDir, 'addons.js');
    const addonsThree = await import(pathToFileURL(addonsPath).href);
    assert.equal(typeof addonsThree.OrbitControls, 'function');
    assert.equal(typeof addonsThree.GLTFExporter, 'function');
    const actualAddonsKeys = Object.keys(addonsThree).filter(k => k !== 'default').sort();
    assert.deepEqual(actualAddonsKeys, addonsMapItem.exportSurface.named, 'Addons export names must match map');

    // 6. Strict Environment Gating with throwing getters
    // Guard window, document, navigator.gpu, and WebAssembly instantiation
    const origWindowDesc = Object.getOwnPropertyDescriptor(globalThis, 'window');
    const origDocDesc = Object.getOwnPropertyDescriptor(globalThis, 'document');
    const origNavGpuDesc = typeof globalThis.navigator !== 'undefined'
      ? Object.getOwnPropertyDescriptor(globalThis.navigator, 'gpu')
      : undefined;
    const origWebAssemblyInstantiate = WebAssembly.instantiate;
    const origWebAssemblyInstantiateStreaming = WebAssembly.instantiateStreaming;

    Object.defineProperty(globalThis, 'window', {
      get() {
        throw new Error('ILLEGAL_ACCESS: global window accessed during CPU-only module execution');
      },
      configurable: true,
    });

    Object.defineProperty(globalThis, 'document', {
      get() {
        throw new Error('ILLEGAL_ACCESS: global document accessed during CPU-only module execution');
      },
      configurable: true,
    });

    if (typeof globalThis.navigator !== 'undefined') {
      Object.defineProperty(globalThis.navigator, 'gpu', {
        get() {
          throw new Error('ILLEGAL_ACCESS: navigator.gpu accessed during CPU-only module execution');
        },
        configurable: true,
      });
    }

    WebAssembly.instantiate = () => {
      throw new Error('ILLEGAL_ACCESS: WebAssembly.instantiate called during CPU-only module execution');
    };
    WebAssembly.instantiateStreaming = () => {
      throw new Error('ILLEGAL_ACCESS: WebAssembly.instantiateStreaming called during CPU-only module execution');
    };

    try {
      // Import and execute CPU math/geometry/exporter facades under active throwing guards
      const vector3Path = path.resolve(tempDir, 'src/math/Vector3.js');
      const { Vector3 } = await import(pathToFileURL(vector3Path).href);
      const v1 = new Vector3(10, 20, 30);
      const v2 = new Vector3(1, 2, 3);
      v1.add(v2);
      assert.equal(v1.x, 11);
      assert.equal(v1.y, 22);
      assert.equal(v1.z, 33);

      const matrix4Path = path.resolve(tempDir, 'src/math/Matrix4.js');
      const { Matrix4 } = await import(pathToFileURL(matrix4Path).href);
      const m = new Matrix4();
      m.makeTranslation(5, 10, 15);
      assert.equal(m.elements[12], 5);
      assert.equal(m.elements[13], 10);
      assert.equal(m.elements[14], 15);

      const boxGeoPath = path.resolve(tempDir, 'src/geometries/BoxGeometry.js');
      const { BoxGeometry } = await import(pathToFileURL(boxGeoPath).href);
      const box = new BoxGeometry(2, 4, 6);
      assert.equal(box.parameters.width, 2);
      assert.equal(box.parameters.height, 4);
      assert.equal(box.parameters.depth, 6);

      const gltfPath = path.resolve(tempDir, 'addons/exporters/GLTFExporter.js');
      const { GLTFExporter } = await import(pathToFileURL(gltfPath).href);
      const exporter = new GLTFExporter();
      assert.ok(exporter && typeof exporter.parse === 'function');
    } finally {
      // Restore original global descriptors
      if (origWindowDesc) Object.defineProperty(globalThis, 'window', origWindowDesc);
      else delete globalThis.window;

      if (origDocDesc) Object.defineProperty(globalThis, 'document', origDocDesc);
      else delete globalThis.document;

      if (typeof globalThis.navigator !== 'undefined') {
        if (origNavGpuDesc) Object.defineProperty(globalThis.navigator, 'gpu', origNavGpuDesc);
        else delete globalThis.navigator.gpu;
      }

      WebAssembly.instantiate = origWebAssemblyInstantiate;
      WebAssembly.instantiateStreaming = origWebAssemblyInstantiateStreaming;
    }
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});


