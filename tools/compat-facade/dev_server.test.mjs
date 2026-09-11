/**
 * @file tools/compat-facade/dev_server.test.mjs
 * Test suite for FrankenThreeD development server and H1 compatibility routing (Plan §3.4, §5.1, §5.12).
 *
 * Verifies bead 6mv.7 requirements:
 * 1. Import map transformation dynamically routes three, three/webgpu, three/tsl, three/addons/* to facade.
 * 2. Original upstream application script, CSS links, HTML structure, and controls are 100% preserved.
 * 3. HTTP server serves H1 demo (examples/webgpu_performance_renderbundle.html) with correct headers.
 * 4. Compatibility facade endpoints (/compat-facade/*) serve valid ES modules without Rollup.
 * 5. Addons mapping routes /compat-facade/addons/* directly to examples/jsm/*.
 * 6. Both WebGPU and WebGL modes retain genuine WebGPURenderer (backend=webgl uses WebGLBackend, NOT legacy WebGLRenderer).
 * 7. Router records decisions externally without mutating constructed instance object shapes.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as acorn from 'acorn';

import {
  createDevImportMap,
  transformHtmlImportMap,
  FACADE_NO_CLAIM_ATTESTATION,
} from './index.mjs';

import {
  createCompatDevServer,
  startDevServer,
  generateRoutedWebGPUSource,
} from './dev_server.mjs';

import { ExecutionRoute } from '../compat/index.mjs';
import { parseHtmlEntries } from '../ingest/html_parser.mjs';
import { resolveModuleSpecifier } from '../ingest/resolver.mjs';
import { analyzeModuleAst } from '../ingest/ast_analyzer.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../..');
const H1_PATH = path.resolve(REPO_ROOT, 'upstream/three.js/examples/webgpu_performance_renderbundle.html');

test('Import Map: createDevImportMap produces correct package routing entries', () => {
  // Default routing when no source import map is provided
  const map = createDevImportMap();
  assert.equal(map.imports['three'], '/compat-facade/three.js');
  assert.equal(map.imports['three/webgpu'], '/compat-facade/webgpu.js');
  assert.equal(map.imports['three/tsl'], '/compat-facade/tsl.js');
  assert.equal(map.imports['three/addons/'], '/compat-facade/addons/');

  const baseMap = createDevImportMap({ baseUrl: 'http://localhost:8080' });
  assert.equal(baseMap.imports['three'], 'http://localhost:8080/compat-facade/three.js');
  assert.equal(baseMap.imports['three/webgpu'], 'http://localhost:8080/compat-facade/webgpu.js');

  // Source import map preservation (H1 maps both 'three' and 'three/webgpu' to three.webgpu.js)
  const h1SourceMap = {
    imports: {
      three: '../build/three.webgpu.js',
      'three/webgpu': '../build/three.webgpu.js',
      'three/tsl': '../build/three.tsl.js',
      'three/addons/': './jsm/',
    },
  };
  const h1DevMap = createDevImportMap({ baseUrl: 'http://127.0.0.1:8080', sourceImportMap: h1SourceMap });
  assert.equal(h1DevMap.imports['three'], 'http://127.0.0.1:8080/compat-facade/webgpu.js', 'H1 three must route to webgpu facade');
  assert.equal(h1DevMap.imports['three/webgpu'], 'http://127.0.0.1:8080/compat-facade/webgpu.js', 'H1 three/webgpu must route to webgpu facade');

  // Explicit routeThreeToWebgpu option without sourceImportMap
  const explicitWebgpuMap = createDevImportMap({ baseUrl: 'http://127.0.0.1:8080', routeThreeToWebgpu: true });
  assert.equal(explicitWebgpuMap.imports['three'], 'http://127.0.0.1:8080/compat-facade/webgpu.js');
  assert.equal(explicitWebgpuMap.imports['three/webgpu'], 'http://127.0.0.1:8080/compat-facade/webgpu.js');
});

test('HTML Transformation: transformHtmlImportMap preserves H1 application code while routing import map', () => {
  const originalHtml = fs.readFileSync(H1_PATH, 'utf-8');
  const transformedHtml = transformHtmlImportMap(originalHtml);

  // 1. Import map must point to compatibility facade, binding both three and three/webgpu to the same singleton
  assert.ok(transformedHtml.includes('"three": "/compat-facade/webgpu.js"'), 'Import map must preserve H1 alias mapping three to webgpu facade');
  assert.ok(transformedHtml.includes('"three/webgpu": "/compat-facade/webgpu.js"'), 'Import map must route three/webgpu to webgpu facade');
  assert.ok(transformedHtml.includes('"three/tsl": "/compat-facade/tsl.js"'), 'Import map must route three/tsl to tsl facade');
  assert.ok(transformedHtml.includes('"three/addons/": "/compat-facade/addons/"'), 'Import map must route three/addons/ to addons facade');

  // 2. Upstream application script must be 100% preserved
  assert.ok(transformedHtml.includes("import * as THREE from 'three/webgpu';"), 'App import from three/webgpu preserved');
  assert.ok(transformedHtml.includes("import { Inspector } from 'three/addons/inspector/Inspector.js';"), 'Inspector import preserved');
  assert.ok(transformedHtml.includes("import { OrbitControls } from 'three/addons/controls/OrbitControls.js';"), 'OrbitControls import preserved');
  assert.ok(transformedHtml.includes("renderer = new THREE.WebGPURenderer( { antialias: true, forceWebGL: ! api.webgpu } );"), 'Renderer construction preserved');
  assert.ok(transformedHtml.includes("function randomizeMatrix( matrix )"), 'randomizeMatrix function preserved');
  assert.ok(transformedHtml.includes("renderer.inspector = new Inspector();"), 'Inspector setup preserved');
  assert.ok(transformedHtml.includes("gui.add( api, 'renderBundle' ).name( 'render bundle' ).onChange( reload );"), 'GUI controls preserved');

  // 3. Document body outside the import map must match verbatim
  const origParts = originalHtml.split(/<script\s+type=["']importmap["']>[\s\S]*?<\/script>/i);
  const transParts = transformedHtml.split(/<script\s+type=["']importmap["']>[\s\S]*?<\/script>/i);
  assert.equal(origParts[0], transParts[0], 'HTML before importmap must be identical');
  assert.equal(origParts[1], transParts[1], 'HTML after importmap must be identical');
});

test('Dev Server: Lifecycle, HTTP responses, and H1 delivery', async () => {
  const { server, port, url, close } = await startDevServer({ port: 0 });

  try {
    assert.ok(port > 0, `Server must bind to valid port, got ${port}`);
    assert.ok(url.startsWith('http://127.0.0.1:'), `URL must be 127.0.0.1, got ${url}`);

    // 1. GET /examples/webgpu_performance_renderbundle.html
    const h1Res = await fetch(`${url}/examples/webgpu_performance_renderbundle.html`);
    assert.equal(h1Res.status, 200);
    assert.ok(h1Res.headers.get('content-type').includes('text/html'));
    assert.equal(h1Res.headers.get('x-frankenthreed-facade'), 'development-mode');
    assert.equal(h1Res.headers.get('x-frankenthreed-attestation'), FACADE_NO_CLAIM_ATTESTATION);

    const h1Body = await h1Res.text();
    assert.ok(h1Body.includes('/compat-facade/webgpu.js'));
    assert.ok(h1Body.includes('THREE.WebGPURenderer'));

    // 2. GET / (convenience alias)
    const rootRes = await fetch(`${url}/`);
    assert.equal(rootRes.status, 200);
    const rootBody = await rootRes.text();
    assert.equal(rootBody, h1Body);

    // 3. GET /example.css
    const cssRes = await fetch(`${url}/example.css`);
    assert.equal(cssRes.status, 200);
    assert.ok(cssRes.headers.get('content-type').includes('text/css'));
    const cssBody = await cssRes.text();
    assert.ok(cssBody.length > 0);

    // 4. GET /compat-facade/webgpu.js
    const webgpuRes = await fetch(`${url}/compat-facade/webgpu.js`);
    assert.equal(webgpuRes.status, 200);
    assert.ok(webgpuRes.headers.get('content-type').includes('javascript'));
    const webgpuSource = await webgpuRes.text();

    // Verify valid Acorn parsing (ES module syntax)
    const ast = acorn.parse(webgpuSource, { ecmaVersion: 'latest', sourceType: 'module' });
    assert.equal(ast.type, 'Program');
    assert.ok(webgpuSource.includes('WebGPURenderer'));
    assert.ok(webgpuSource.includes('RendererConstructionRouter'));
    assert.ok(webgpuSource.includes('tools/compat/route_types.mjs'), 'Must directly import route_types.mjs');
    assert.ok(webgpuSource.includes('tools/compat/construction_adapter.mjs'), 'Must directly import construction_adapter.mjs');
    assert.ok(!webgpuSource.includes('tools/compat/index.mjs'), 'Must not import barrel index.mjs');
    assert.ok(!webgpuSource.includes('three.module.js'), 'Must not import three.module.js');

    // 5. GET /compat-facade/three.js
    const threeRes = await fetch(`${url}/compat-facade/three.js`);
    assert.equal(threeRes.status, 200);
    const threeSource = await threeRes.text();
    assert.ok(threeSource.includes('build/three.module.js'));

    // 6. GET /compat-facade/tsl.js
    const tslRes = await fetch(`${url}/compat-facade/tsl.js`);
    assert.equal(tslRes.status, 200);
    const tslSource = await tslRes.text();
    assert.ok(tslSource.includes('build/three.tsl.js'));

    // 7. GET /compat-facade/addons/inspector/Inspector.js
    const inspectorRes = await fetch(`${url}/compat-facade/addons/inspector/Inspector.js`);
    assert.equal(inspectorRes.status, 200);
    const inspectorSource = await inspectorRes.text();
    assert.ok(inspectorSource.includes('class Inspector'));

    // 8. GET /compat-facade/addons/controls/OrbitControls.js
    const controlsRes = await fetch(`${url}/compat-facade/addons/controls/OrbitControls.js`);
    assert.equal(controlsRes.status, 200);
    const controlsSource = await controlsRes.text();
    assert.ok(controlsSource.includes('class OrbitControls'));

    // 9. 404 for nonexistent addon
    const notFoundRes = await fetch(`${url}/compat-facade/addons/nonexistent/FakeAddon.js`);
    assert.equal(notFoundRes.status, 404);

    // 10. Request Regression 1: Assets navigated from /examples/ page URL (avoiding doubled examples/examples/)
    // Both /examples/textures/uv_grid_opengl.jpg and /textures/uv_grid_opengl.jpg must resolve with 200 OK
    const textureResWithPrefix = await fetch(`${url}/examples/textures/uv_grid_opengl.jpg`);
    assert.equal(textureResWithPrefix.status, 200, '/examples/textures/uv_grid_opengl.jpg must return 200');
    assert.ok(textureResWithPrefix.headers.get('content-type').includes('image/jpeg'));
    const textureBytes = await textureResWithPrefix.arrayBuffer();
    assert.ok(textureBytes.byteLength > 0);

    const textureResNoPrefix = await fetch(`${url}/textures/uv_grid_opengl.jpg`);
    assert.equal(textureResNoPrefix.status, 200, '/textures/uv_grid_opengl.jpg must return 200');

    const examplesJsmRes = await fetch(`${url}/examples/jsm/controls/OrbitControls.js`);
    assert.equal(examplesJsmRes.status, 200, '/examples/jsm/controls/OrbitControls.js must return 200');
    const jsmSource = await examplesJsmRes.text();
    assert.ok(jsmSource.includes('class OrbitControls'));

    // 11. Request Regression 2: Static alias for /build/* (source-relative ../build/* from examples)
    const buildWebgpuRes = await fetch(`${url}/build/three.webgpu.js`);
    assert.equal(buildWebgpuRes.status, 200, '/build/three.webgpu.js must return 200');
    assert.ok(buildWebgpuRes.headers.get('content-type').includes('javascript'));
    const buildWebgpuSource = await buildWebgpuRes.text();
    assert.ok(buildWebgpuSource.includes('WebGPURenderer'));

    const buildModuleRes = await fetch(`${url}/build/three.module.js`);
    assert.equal(buildModuleRes.status, 200, '/build/three.module.js must return 200');

    const buildCoreRes = await fetch(`${url}/build/three.core.js`);
    assert.equal(buildCoreRes.status, 200, '/build/three.core.js must return 200');

    // 12. Favicon: returns 204 No Content to eliminate browser console errors
    const faviconRes = await fetch(`${url}/favicon.ico`);
    assert.equal(faviconRes.status, 204, '/favicon.ico must return 204 No Content');

    // 13. Request Regression 3: Inspector bare import targets resolve to served 200 modules (Topaz 6953)
    const lut3DFacadeRes = await fetch(`${url}/compat-facade/addons/tsl/display/Lut3DNode.js`);
    assert.equal(lut3DFacadeRes.status, 200, '/compat-facade/addons/tsl/display/Lut3DNode.js must return 200');
    assert.ok(lut3DFacadeRes.headers.get('content-type').includes('javascript'));
    const lut3DFacadeSource = await lut3DFacadeRes.text();
    assert.ok(lut3DFacadeSource.includes('class Lut3DNode'));
    assert.ok(lut3DFacadeSource.includes('lut3D'));

    const tslFacadeRes = await fetch(`${url}/compat-facade/tsl.js`);
    assert.equal(tslFacadeRes.status, 200, '/compat-facade/tsl.js must return 200');
    assert.ok(tslFacadeRes.headers.get('content-type').includes('javascript'));
    const tslFacadeSource = await tslFacadeRes.text();
    assert.ok(tslFacadeSource.includes('build/three.tsl.js'));
  } finally {
    await close();
  }
});

test('Exact Backend & WebGPU Mode: Genuine WebGPURenderer retained in both modes', async () => {
  // Test the construction routing behavior matching H1:
  // new THREE.WebGPURenderer({ antialias: true, forceWebGL: !api.webgpu })
  const { WebGPURenderer: PinnedWebGPURenderer } = await import(
    path.resolve(REPO_ROOT, 'upstream/three.js/build/three.webgpu.js')
  );
  const { RendererConstructionRouter, ExecutionRoute } = await import(
    path.resolve(REPO_ROOT, 'tools/compat/index.mjs')
  );

  const router = new RendererConstructionRouter({
    implementations: {
      [ExecutionRoute.EXACT_BACKEND]: {
        WebGPURenderer: PinnedWebGPURenderer,
      },
      [ExecutionRoute.GENERAL_WEBGPU]: {
        WebGPURenderer: PinnedWebGPURenderer,
      },
      [ExecutionRoute.RETAINED_UPSTREAM]: {
        WebGPURenderer: PinnedWebGPURenderer,
      },
    },
  });

  // Mock canvas
  const canvas1 = { getContext: () => null, addEventListener: () => {} };
  const canvas2 = { getContext: () => null, addEventListener: () => {} };

  // Mode 1: api.webgpu = true -> forceWebGL = false (WebGPU mode)
  const webgpuInstance = router.routeAndConstruct({
    constructorFn: PinnedWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: canvas1, antialias: true, forceWebGL: false },
    sourceSpan: 'webgpu_performance_renderbundle.html:188:13',
  });

  assert.ok(webgpuInstance instanceof PinnedWebGPURenderer, 'Must construct genuine WebGPURenderer');
  assert.equal(webgpuInstance.backend.constructor.name, 'WebGPUBackend', 'Default mode must select WebGPUBackend');

  // Mode 2: api.webgpu = false -> forceWebGL = true (WebGL fallback mode via ?backend=webgl)
  const webglInstance = router.routeAndConstruct({
    constructorFn: PinnedWebGPURenderer,
    constructorName: 'WebGPURenderer',
    options: { canvas: canvas2, antialias: true, forceWebGL: true },
    sourceSpan: 'webgpu_performance_renderbundle.html:188:13',
  });

  assert.ok(webglInstance instanceof PinnedWebGPURenderer, 'Must construct genuine WebGPURenderer');
  assert.equal(webglInstance.backend.constructor.name, 'WebGLBackend', 'forceWebGL must select WebGLBackend, NOT legacy WebGLRenderer');

  // Verify external decision log
  const log = router.getDecisionLog();
  assert.equal(log.length, 2);
  assert.equal(log[0].site, 'WebGPURenderer');
  assert.equal(log[0].span, 'webgpu_performance_renderbundle.html:188:13');
  assert.equal(log[1].site, 'WebGPURenderer');
  assert.equal(log[1].route, ExecutionRoute.EXACT_BACKEND);

  // Invariant: no enumerable __f3d_* own properties on instances
  const webgpuOwnKeys = Object.keys(webgpuInstance);
  const webglOwnKeys = Object.keys(webglInstance);
  assert.ok(!webgpuOwnKeys.some(k => k.startsWith('__f3d_')), 'No enumerable __f3d own properties on WebGPU instance');
  assert.ok(!webglOwnKeys.some(k => k.startsWith('__f3d_')), 'No enumerable __f3d own properties on WebGL instance');
});

test('Generated WebGPU Facade Module: Evaluation, Proxy constructor adapter, instanceof, subclassing, and new.target', async () => {
  // 1. Generate the actual module source as emitted for runtime execution
  const generatedSource = generateRoutedWebGPUSource({
    importBase: 'file://' + REPO_ROOT,
  });

  // Verify that there is NO prototype assignment in the emitted source (would throw TypeError on ES class)
  assert.ok(
    !generatedSource.includes('WebGPURenderer.prototype ='),
    'Generated source must not assign to WebGPURenderer.prototype (non-writable ES class prototype)'
  );

  // Require direct routing module imports and absence of gratuitous three.module load
  assert.ok(
    generatedSource.includes('/tools/compat/route_types.mjs'),
    'Generated source must directly import route_types.mjs'
  );
  assert.ok(
    generatedSource.includes('/tools/compat/construction_adapter.mjs'),
    'Generated source must directly import construction_adapter.mjs'
  );
  assert.ok(
    !generatedSource.includes('/tools/compat/index.mjs'),
    'Generated source must not import barrel tools/compat/index.mjs (prevents gratuitous WebGL three.module.js load)'
  );
  assert.ok(
    !generatedSource.includes('three.module.js'),
    'Generated source must not import three.module.js (preserves RNG sequence parity with reference)'
  );

  // 2. Dynamically import the actual generated module via data: URL
  const dataUri = 'data:text/javascript;base64,' + Buffer.from(generatedSource).toString('base64');
  const facadeModule = await import(dataUri);

  const { WebGPURenderer: UpstreamWebGPURenderer } = await import(
    path.resolve(REPO_ROOT, 'upstream/three.js/build/three.webgpu.js')
  );

  // 3. Verify module exports and constructor shape
  assert.equal(typeof facadeModule.WebGPURenderer, 'function', 'WebGPURenderer must be exported as a constructor');
  assert.equal(
    facadeModule.WebGPURenderer.prototype,
    UpstreamWebGPURenderer.prototype,
    'Exported Proxy constructor must share upstream prototype'
  );

  // 4. Test WebGPU mode (forceWebGL: false)
  const canvas1 = { getContext: () => null, addEventListener: () => {} };
  const rendererWebGPU = new facadeModule.WebGPURenderer({
    canvas: canvas1,
    antialias: true,
    forceWebGL: false,
  });

  assert.ok(rendererWebGPU instanceof facadeModule.WebGPURenderer, 'Instance must be instanceof facade WebGPURenderer');
  assert.ok(rendererWebGPU instanceof UpstreamWebGPURenderer, 'Instance must be instanceof Upstream WebGPURenderer');
  // Honest constructor identity: Proxy preserves pristine native prototype, so instance.constructor identifies UpstreamWebGPURenderer
  assert.equal(rendererWebGPU.constructor, UpstreamWebGPURenderer, 'Direct instance constructor references pristine upstream class');
  assert.equal(rendererWebGPU.backend.constructor.name, 'WebGPUBackend', 'Default mode selects WebGPUBackend');
  assert.ok(
    !Object.keys(rendererWebGPU).some(k => k.startsWith('__f3d_')),
    'Instance must not have enumerable __f3d_* own properties'
  );

  // 5. Test WebGL mode (forceWebGL: true) - genuine WebGPURenderer with WebGLBackend
  const canvas2 = { getContext: () => null, addEventListener: () => {} };
  const rendererWebGL = new facadeModule.WebGPURenderer({
    canvas: canvas2,
    antialias: true,
    forceWebGL: true,
  });

  assert.ok(rendererWebGL instanceof facadeModule.WebGPURenderer, 'WebGL instance must be instanceof facade WebGPURenderer');
  assert.ok(rendererWebGL instanceof UpstreamWebGPURenderer, 'WebGL instance must be instanceof Upstream WebGPURenderer');
  assert.equal(rendererWebGL.constructor, UpstreamWebGPURenderer, 'Direct WebGL instance constructor references pristine upstream class');
  assert.equal(rendererWebGL.backend.constructor.name, 'WebGLBackend', 'forceWebGL selects WebGLBackend (never legacy WebGLRenderer)');

  // 6. Test Subclassing with new.target and prototype chain preservation
  class CustomAppRenderer extends facadeModule.WebGPURenderer {
    constructor(options) {
      super(options);
      this.customFlag = 'app-custom-renderer';
    }

    customMethod() {
      return 999;
    }
  }

  const canvas3 = { getContext: () => null, addEventListener: () => {} };
  const derivedInstance = new CustomAppRenderer({
    canvas: canvas3,
    antialias: true,
    forceWebGL: false,
  });

  assert.ok(derivedInstance instanceof CustomAppRenderer, 'Subclass instance must be instanceof CustomAppRenderer');
  assert.ok(derivedInstance instanceof facadeModule.WebGPURenderer, 'Subclass instance must be instanceof facade WebGPURenderer');
  assert.ok(derivedInstance instanceof UpstreamWebGPURenderer, 'Subclass instance must be instanceof Upstream WebGPURenderer');
  assert.equal(derivedInstance.constructor, CustomAppRenderer, 'Subclass instance constructor identity must equal CustomAppRenderer');
  assert.equal(derivedInstance.customFlag, 'app-custom-renderer', 'Subclass own property preserved');
  assert.equal(derivedInstance.customMethod(), 999, 'Subclass method callable');
  assert.equal(derivedInstance.backend.constructor.name, 'WebGPUBackend', 'Subclass backend created');

  // 7. Test router singleton tracking
  assert.ok(facadeModule.router, 'Facade module must export router instance');
  const decisions = facadeModule.router.getDecisionLog();
  assert.ok(decisions.length >= 3, 'Router must have logged decisions for all constructions');
  assert.equal(decisions[0].site, 'WebGPURenderer');
  assert.equal(decisions[0].span, 'webgpu_performance_renderbundle.html:188:13');

  // 8. Test canvas exposure route locking and getter preservation (Plan §3.4, §6.9, Bead 6mv.5)
  // Auto-created canvas is registered post-construction via descriptor inspection without invoking
  // observable getters on subclasses or factories, and preserves throwing getter contracts.
  const origDoc = globalThis.document;
  try {
    globalThis.document = {
      createElementNS: () => ({
        style: {},
        addEventListener: () => {},
        removeEventListener: () => {},
        getContext: () => null,
        setAttribute: () => {},
      }),
      createElement: () => ({
        style: {},
        addEventListener: () => {},
        removeEventListener: () => {},
        getContext: () => null,
        setAttribute: () => {},
      }),
    };

    // 8a. Observable getter counting subclass: construction must NOT invoke getter
    let subclassGetterCount = 0;
    class ObservableSubclass extends facadeModule.WebGPURenderer {
      get domElement() {
        subclassGetterCount++;
        return super.domElement;
      }
    }

    const sub = new ObservableSubclass({ antialias: true, forceWebGL: false });
    assert.equal(subclassGetterCount, 0, 'Construction must NOT invoke observable domElement getter on subclass');

    const exposedDomElement = sub.domElement;
    assert.equal(subclassGetterCount, 1, 'Accessing sub.domElement invokes getter exactly once');
    assert.ok(exposedDomElement && typeof exposedDomElement === 'object', 'Exposed canvas must be an object');
    assert.equal(
      facadeModule.router.getCanvasLock(exposedDomElement)?.route,
      'retained-upstream',
      'Exposed canvas must be locked to route upon construction'
    );

    // Conflicting route attempt on exposed canvas must throw RouteLockError
    assert.throws(
      () => {
        new facadeModule.WebGPURenderer({
          canvas: exposedDomElement,
          forceWebGL: true,
        });
      },
      {
        name: 'RouteLockError',
      },
      'Constructing conflicting route on exposed canvas must throw RouteLockError'
    );

    // 8b. Throwing getter subclass: construction does not invoke getter; caller access faithfully throws
    class ThrowingGetterSubclass extends facadeModule.WebGPURenderer {
      get domElement() {
        throw new Error('user intentional throw');
      }
    }

    const throwingSub = new ThrowingGetterSubclass({ antialias: true, forceWebGL: false });
    assert.ok(throwingSub, 'Construction must succeed without invoking throwing getter');
    assert.throws(
      () => { throwingSub.domElement; },
      /user intentional throw/,
      'Caller accessing throwingSub.domElement must faithfully receive original throw'
    );
  } finally {
    if (origDoc) globalThis.document = origDoc;
    else delete globalThis.document;
  }
});

test('Served Module Graph Closure: Ingest tools resolve served H1 and Inspector reachable static imports with zero unresolved specifiers', async () => {
  const { server, port, url, close } = await startDevServer({ port: 0 });

  try {
    const pageUrl = `${url}/examples/webgpu_performance_renderbundle.html`;
    const pageRes = await fetch(pageUrl);
    assert.equal(pageRes.status, 200, 'Served H1 HTML must return 200 OK');
    const pageHtml = await pageRes.text();

    // 1. Parse served HTML entries using ingest tool (extracting served import map and inline script)
    const { importMap, moduleScripts } = parseHtmlEntries(pageHtml, pageUrl);
    assert.ok(moduleScripts.length > 0, 'H1 must contain at least one <script type="module"> entry');
    assert.ok(importMap.imports['three'], 'Served import map must define three');
    assert.ok(importMap.imports['three/webgpu'], 'Served import map must define three/webgpu');

    // 2. Traversal queue starting with H1 page inline script and Inspector entry point
    const visited = new Set();
    const queue = [];
    const fetchedModules = [];
    const unresolvedErrors = [];
    const discoveredAssetReferences = [];

    // Queue H1 inline module scripts
    for (const script of moduleScripts) {
      queue.push({
        isInline: true,
        content: script.inlineContent,
        url: script.id,
        referrerUrl: pageUrl,
      });
    }

    // Queue Inspector addon entry explicitly to ensure full reachable coverage
    const inspectorSpecifier = 'three/addons/inspector/Inspector.js';
    const inspectorUrl = resolveModuleSpecifier(inspectorSpecifier, pageUrl, importMap, {
      mapBaseUrl: pageUrl,
    });
    queue.push({
      isInline: false,
      url: inspectorUrl,
      referrerUrl: pageUrl,
    });

    // 3. Breadth-first traversal of the served module graph (parse-only, no Rollup)
    while (queue.length > 0) {
      const current = queue.shift();
      if (visited.has(current.url)) continue;
      visited.add(current.url);

      let code;
      if (current.isInline) {
        code = current.content;
      } else {
        const res = await fetch(current.url);
        assert.equal(
          res.status,
          200,
          `Module "${current.url}" (imported by "${current.referrerUrl}") must return HTTP 200`
        );
        code = await res.text();
        fetchedModules.push({
          url: current.url,
          pathname: new URL(current.url).pathname,
          status: res.status,
        });
      }

      // Parse-only AST analysis using ingest tool
      const analysis = analyzeModuleAst(code, current.url);

      // Collect asset references surfaced by ast_analyzer (e.g. new URL(..., import.meta.url))
      if (analysis.assetReferences && analysis.assetReferences.length > 0) {
        for (const assetRef of analysis.assetReferences) {
          discoveredAssetReferences.push({
            specifier: assetRef.specifier,
            referrerUrl: current.url,
            sourceSpan: assetRef.sourceSpan,
          });

          // Resolve and fetch asset reference to verify it serves with HTTP 200
          try {
            const resolvedAssetUrl = new URL(assetRef.specifier, current.url).href;
            const assetRes = await fetch(resolvedAssetUrl);
            assert.equal(
              assetRes.status,
              200,
              `Asset "${assetRef.specifier}" (referenced by "${current.url}") must return HTTP 200`
            );
          } catch (err) {
            unresolvedErrors.push({
              specifier: assetRef.specifier,
              referrerUrl: current.url,
              error: `Asset fetch error: ${err.message}`,
            });
          }
        }
      }

      // Collect all static import specifiers, export-from re-export specifiers, and literal dynamic imports
      const specifiersToResolve = [
        ...analysis.staticImports.map(i => i.specifier),
        ...analysis.staticExports.filter(e => e.specifier).map(e => e.specifier),
        ...analysis.dynamicImports
          .filter(d => d.classification === 'literal' && d.specifier)
          .map(d => d.specifier),
      ];

      for (const specifier of specifiersToResolve) {
        try {
          const resolvedUrl = resolveModuleSpecifier(specifier, current.url, importMap, {
            mapBaseUrl: pageUrl,
          });
          if (!visited.has(resolvedUrl)) {
            queue.push({
              isInline: false,
              url: resolvedUrl,
              referrerUrl: current.url,
            });
          }
        } catch (err) {
          unresolvedErrors.push({
            specifier,
            referrerUrl: current.url,
            error: err.message,
          });
        }
      }
    }

    // 4. Assert zero unresolved specifiers and verify served graph closes
    assert.equal(
      unresolvedErrors.length,
      0,
      `All static specifiers must resolve cleanly without errors: ${JSON.stringify(unresolvedErrors)}`
    );

    // Verify significant reachable graph closure (facade, Inspector, OrbitControls, TSL, WebGPU, and Three core)
    assert.ok(fetchedModules.length >= 25, `Expected at least 25 fetched modules, got ${fetchedModules.length}`);

    // Verify key modules are present in the fetched set
    const fetchedPaths = new Set(fetchedModules.map(m => m.pathname));
    assert.ok(fetchedPaths.has('/compat-facade/webgpu.js'), 'Must fetch /compat-facade/webgpu.js');
    assert.ok(fetchedPaths.has('/compat-facade/tsl.js'), 'Must fetch /compat-facade/tsl.js');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/Inspector.js'), 'Must fetch Inspector.js');
    assert.ok(fetchedPaths.has('/compat-facade/addons/controls/OrbitControls.js'), 'Must fetch OrbitControls.js');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/RendererInspector.js'), 'Must fetch RendererInspector.js');
    assert.ok(fetchedPaths.has('/upstream/three.js/build/three.webgpu.js'), 'Must fetch upstream three.webgpu.js');

    // Routing modules: direct routing modules present, gratuitous WebGL barrel absent
    assert.ok(fetchedPaths.has('/tools/compat/route_types.mjs'), 'Must fetch route_types.mjs');
    assert.ok(fetchedPaths.has('/tools/compat/construction_adapter.mjs'), 'Must fetch construction_adapter.mjs');
    assert.ok(fetchedPaths.has('/tools/compat/connected_groups.mjs'), 'Must fetch connected_groups.mjs');
    assert.ok(fetchedPaths.has('/tools/compat/route_decider.mjs'), 'Must fetch route_decider.mjs');
    assert.ok(!fetchedPaths.has('/tools/compat/exact_backend.mjs'), 'Must NOT fetch exact_backend.mjs (prevents three.module load)');
    assert.ok(!fetchedPaths.has('/tools/compat/index.mjs'), 'Must NOT fetch barrel index.mjs');
    assert.ok(!fetchedPaths.has('/tools/compat/route_report.mjs'), 'Must NOT fetch route_report.mjs');
    assert.ok(!fetchedPaths.has('/upstream/three.js/build/three.module.js'), 'Must NOT fetch three.module.js');

    // Inspector tabs
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/tabs/Performance.js'), 'Must fetch Performance tab');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/tabs/Memory.js'), 'Must fetch Memory tab');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/tabs/Console.js'), 'Must fetch Console tab');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/tabs/Parameters.js'), 'Must fetch Parameters tab');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/tabs/Settings.js'), 'Must fetch Settings tab');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/tabs/Viewer.js'), 'Must fetch Viewer tab');
    assert.ok(fetchedPaths.has('/compat-facade/addons/inspector/tabs/Timeline.js'), 'Must fetch Timeline tab');

    // 5. Asset References: Verify asset references surfaced by ingest analysis
    // H1 and Inspector modules make zero static asset references via new URL(..., import.meta.url)
    // (H1 uses procedural geometry/materials without textures; Inspector's lone new URL in Settings.js:283 is dynamic)
    assert.equal(
      discoveredAssetReferences.length,
      0,
      `Expected 0 static asset references in served H1 and Inspector, got ${discoveredAssetReferences.length}`
    );

    // Verify H1 HTML-level asset reference: <link rel="stylesheet" href="example.css">
    const cssRes = await fetch(`${url}/examples/example.css`);
    assert.equal(cssRes.status, 200, 'H1 stylesheet /examples/example.css must return 200');
    assert.ok(cssRes.headers.get('content-type').includes('text/css'));
    const cssContent = await cssRes.text();
    // example.css inlines its SVG logo via data URI, making 0 external asset requests
    assert.ok(cssContent.includes('data:image/svg+xml'), 'example.css must inline logo via data URI');

    // Verify static texture asset endpoint (e.g. uv_grid_opengl.jpg) resolves with 200 OK
    const textureRes = await fetch(`${url}/examples/textures/uv_grid_opengl.jpg`);
    assert.equal(textureRes.status, 200, 'Texture asset /examples/textures/uv_grid_opengl.jpg must return 200');
    assert.ok(textureRes.headers.get('content-type').includes('image/jpeg'));
  } finally {
    await close();
  }
});

test('Regression (Topaz 6953): Inspector bare import three/addons/tsl/display/Lut3DNode.js and three/tsl resolve to served 200 modules', async () => {
  const { server, port, url, close } = await startDevServer({ port: 0 });

  try {
    const pageUrl = `${url}/examples/webgpu_performance_renderbundle.html`;
    const pageRes = await fetch(pageUrl);
    assert.equal(pageRes.status, 200, 'Served H1 HTML must return 200 OK');
    const pageHtml = await pageRes.text();

    const { importMap } = parseHtmlEntries(pageHtml, pageUrl);

    // 1. Resolve Inspector bare import three/addons/tsl/display/Lut3DNode.js via served import map
    const lut3DSpecifier = 'three/addons/tsl/display/Lut3DNode.js';
    const lut3DUrl = resolveModuleSpecifier(lut3DSpecifier, pageUrl, importMap, {
      mapBaseUrl: pageUrl,
    });
    assert.equal(
      lut3DUrl,
      `${url}/compat-facade/addons/tsl/display/Lut3DNode.js`,
      'Bare import three/addons/tsl/display/Lut3DNode.js must resolve to /compat-facade/addons/tsl/display/Lut3DNode.js'
    );

    // Real GET request verifying HTTP 200 and module contents
    const lut3DRes = await fetch(lut3DUrl);
    assert.equal(lut3DRes.status, 200, 'Resolved Lut3DNode.js URL must return HTTP 200');
    assert.ok(lut3DRes.headers.get('content-type').includes('javascript'));
    const lut3DSource = await lut3DRes.text();
    assert.ok(lut3DSource.includes('class Lut3DNode'), 'Lut3DNode.js must contain class Lut3DNode');
    assert.ok(lut3DSource.includes('lut3D'), 'Lut3DNode.js must export lut3D function');

    // 2. Resolve Inspector bare import three/tsl via served import map
    const tslSpecifier = 'three/tsl';
    const tslUrl = resolveModuleSpecifier(tslSpecifier, pageUrl, importMap, {
      mapBaseUrl: pageUrl,
    });
    assert.equal(
      tslUrl,
      `${url}/compat-facade/tsl.js`,
      'Bare import three/tsl must resolve to /compat-facade/tsl.js'
    );

    // Real GET request verifying HTTP 200 and module contents
    const tslRes = await fetch(tslUrl);
    assert.equal(tslRes.status, 200, 'Resolved three/tsl URL must return HTTP 200');
    assert.ok(tslRes.headers.get('content-type').includes('javascript'));
    const tslSource = await tslRes.text();
    assert.ok(tslSource.includes('build/three.tsl.js'), 'TSL facade must re-export from build/three.tsl.js');
  } finally {
    await close();
  }
});

test('H1 Query Branches & Verbatim Script Preservation: backend=webgl, renderBundle=false, and count=1000 serve byte-identical HTML preserving original inline module script', async () => {
  const { server, port, url, close } = await startDevServer({ port: 0 });

  try {
    const basePath = '/examples/webgpu_performance_renderbundle.html';
    const defaultUrl = `${url}${basePath}`;

    // 1. Fetch default page
    const defaultRes = await fetch(defaultUrl);
    assert.equal(defaultRes.status, 200, 'Default H1 page must return 200 OK');
    const defaultHtml = await defaultRes.text();
    const defaultBuffer = Buffer.from(defaultHtml, 'utf-8');

    // 2. Fetch H1 reload query branches
    const queryBranches = [
      '?backend=webgl',
      '?renderBundle=false',
      '?count=1000',
      '?backend=webgl&renderBundle=false&count=1000',
    ];

    for (const query of queryBranches) {
      const queryUrl = `${url}${basePath}${query}`;
      const queryRes = await fetch(queryUrl);
      assert.equal(queryRes.status, 200, `Query URL "${queryUrl}" must return 200 OK`);

      const queryHtml = await queryRes.text();
      const queryBuffer = Buffer.from(queryHtml, 'utf-8');

      // Assert strict string equality and byte equality across all query strings
      assert.strictEqual(
        queryHtml,
        defaultHtml,
        `Served HTML for query "${query}" must be string-identical to default HTML`
      );
      assert.ok(
        defaultBuffer.equals(queryBuffer),
        `Served HTML for query "${query}" must be byte-identical to default HTML`
      );
      assert.equal(
        Buffer.compare(defaultBuffer, queryBuffer),
        0,
        `Buffer.compare must return 0 for query "${query}"`
      );
    }

    // 3. Confirm served HTML preserves the original inline module script verbatim by comparing against upstream
    const upstreamPath = path.resolve(REPO_ROOT, 'upstream/three.js/examples/webgpu_performance_renderbundle.html');
    const upstreamHtml = fs.readFileSync(upstreamPath, 'utf-8');

    // Extract inline module scripts using parseHtmlEntries
    const upstreamEntries = parseHtmlEntries(upstreamHtml, 'file://' + upstreamPath);
    const servedEntries = parseHtmlEntries(defaultHtml, defaultUrl);

    assert.equal(
      servedEntries.moduleScripts.length,
      upstreamEntries.moduleScripts.length,
      'Module script count must match upstream'
    );
    assert.ok(servedEntries.moduleScripts.length > 0, 'Must have at least one module script');

    const upstreamInlineContent = upstreamEntries.moduleScripts[0].inlineContent;
    const servedInlineContent = servedEntries.moduleScripts[0].inlineContent;

    // Strict verbatim equality
    assert.strictEqual(
      servedInlineContent,
      upstreamInlineContent,
      'Served inline module script must match upstream script verbatim'
    );
    assert.ok(
      Buffer.from(servedInlineContent, 'utf-8').equals(Buffer.from(upstreamInlineContent, 'utf-8')),
      'Served inline module script bytes must match upstream script bytes exactly'
    );

    // Also assert that the entire HTML outside the <script type="importmap"> is byte-for-byte identical
    const importMapRegex = /<script\s+type=["']importmap["']>([\s\S]*?)<\/script>/i;
    const upstreamStripped = upstreamHtml.replace(importMapRegex, '');
    const servedStripped = defaultHtml.replace(importMapRegex, '');

    assert.strictEqual(
      servedStripped,
      upstreamStripped,
      'Served HTML body must be byte-for-byte identical to upstream HTML except for the rewritten import map'
    );
  } finally {
    await close();
  }
});

test('Honesty Guard (Plan §5.1, Mails 6869/6879): Actual generated facade router implementations contain exactly exact-backend and retained-upstream, pristine prototype constructor, and zero enumerable __f3d_ properties', async () => {
  // 1. Generate runtime source emitted by facade dev server and evaluate via fresh data URI
  const generatedSource = generateRoutedWebGPUSource({
    importBase: 'file://' + REPO_ROOT,
  }) + `\n// nonce: honesty-guard-${Date.now()}`;
  const dataUri = 'data:text/javascript;base64,' + Buffer.from(generatedSource).toString('base64');
  const facadeModule = await import(dataUri);

  const { WebGPURenderer: UpstreamWebGPURenderer } = await import(
    path.resolve(REPO_ROOT, 'upstream/three.js/build/three.webgpu.js')
  );

  // 2. Assert router implementations table contains exactly exact-backend and retained-upstream
  assert.ok(facadeModule.router, 'Facade module must export router instance');
  const implementations = facadeModule.router.implementations;
  assert.ok(implementations, 'Router must have implementations table');

  const registeredRoutes = Object.keys(implementations).sort();
  const expectedRoutes = [ExecutionRoute.EXACT_BACKEND, ExecutionRoute.RETAINED_UPSTREAM].sort();

  assert.deepEqual(
    registeredRoutes,
    expectedRoutes,
    `Implementations table must contain exactly [${expectedRoutes.join(', ')}], got [${registeredRoutes.join(', ')}]`
  );

  // Explicit negative assertions against unadmitted / unbuilt acceleration claims
  assert.equal(
    implementations[ExecutionRoute.SPECIALIZED_WEBGPU],
    undefined,
    'Router must NOT register specialized-webgpu (Plan §5.1: no fake acceleration claim)'
  );
  assert.equal(
    implementations[ExecutionRoute.GENERAL_WEBGPU],
    undefined,
    'Router must NOT register general-webgpu (Plan §5.1: no unverified general claim)'
  );

  // 3. Assert upstream WebGPURenderer.prototype.constructor is unchanged after module evaluation
  assert.strictEqual(
    UpstreamWebGPURenderer.prototype.constructor,
    UpstreamWebGPURenderer,
    'Upstream WebGPURenderer.prototype.constructor must remain pristine and reference itself'
  );
  assert.notStrictEqual(
    UpstreamWebGPURenderer.prototype.constructor,
    facadeModule.WebGPURenderer,
    'Upstream prototype.constructor must NOT be mutated to the facade Proxy'
  );

  // 4. Assert no enumerable __f3d_ properties exist on constructed instances
  // Branch A: Default WebGPU mode
  const canvasWebGPU = { getContext: () => null, addEventListener: () => {} };
  const instanceWebGPU = new facadeModule.WebGPURenderer({
    canvas: canvasWebGPU,
    forceWebGL: false,
  });

  const f3dKeysWebGPU = Object.keys(instanceWebGPU).filter(k => k.startsWith('__f3d_'));
  assert.deepEqual(
    f3dKeysWebGPU,
    [],
    `Constructed WebGPU instance must have 0 enumerable __f3d_ properties, found: ${f3dKeysWebGPU.join(', ')}`
  );

  const f3dOwnPropsWebGPU = Object.getOwnPropertyNames(instanceWebGPU).filter(k => k.startsWith('__f3d_'));
  assert.deepEqual(
    f3dOwnPropsWebGPU,
    [],
    `Constructed WebGPU instance must have 0 own __f3d_ properties, found: ${f3dOwnPropsWebGPU.join(', ')}`
  );

  const forInKeysWebGPU = [];
  for (const key in instanceWebGPU) {
    if (key.startsWith('__f3d_')) forInKeysWebGPU.push(key);
  }
  assert.deepEqual(forInKeysWebGPU, [], 'for..in loop on WebGPU instance must yield 0 __f3d_ properties');

  // Branch B: forceWebGL mode
  const canvasWebGL = { getContext: () => null, addEventListener: () => {} };
  const instanceWebGL = new facadeModule.WebGPURenderer({
    canvas: canvasWebGL,
    forceWebGL: true,
  });

  const f3dKeysWebGL = Object.keys(instanceWebGL).filter(k => k.startsWith('__f3d_'));
  assert.deepEqual(
    f3dKeysWebGL,
    [],
    `Constructed WebGL instance must have 0 enumerable __f3d_ properties, found: ${f3dKeysWebGL.join(', ')}`
  );

  const f3dOwnPropsWebGL = Object.getOwnPropertyNames(instanceWebGL).filter(k => k.startsWith('__f3d_'));
  assert.deepEqual(
    f3dOwnPropsWebGL,
    [],
    `Constructed WebGL instance must have 0 own __f3d_ properties, found: ${f3dOwnPropsWebGL.join(', ')}`
  );

  const forInKeysWebGL = [];
  for (const key in instanceWebGL) {
    if (key.startsWith('__f3d_')) forInKeysWebGL.push(key);
  }
  assert.deepEqual(forInKeysWebGL, [], 'for..in loop on WebGL instance must yield 0 __f3d_ properties');
});

test('Served H1 HTML Import Map: Contains exactly intended routed specifiers (three, three/webgpu, three/tsl, three/addons/) and nothing else', async () => {
  const { server, port, url, close } = await startDevServer({ port: 0 });

  try {
    const pageUrl = `${url}/examples/webgpu_performance_renderbundle.html`;
    const res = await fetch(pageUrl);
    assert.equal(res.status, 200, 'Served H1 HTML must return HTTP 200');
    const html = await res.text();

    // 1. Extract the raw <script type="importmap"> block from served HTML
    const importMapRegex = /<script\s+type=["']importmap["']>([\s\S]*?)<\/script>/i;
    const match = importMapRegex.exec(html);
    assert.ok(match, 'Served H1 HTML must contain <script type="importmap"> tag');

    const parsedImportMap = JSON.parse(match[1]);
    assert.ok(parsedImportMap && typeof parsedImportMap === 'object', 'Import map must parse to an object');
    assert.ok(parsedImportMap.imports && typeof parsedImportMap.imports === 'object', 'Import map must have imports object');

    // 2. Assert exact routed specifiers: strictly three, three/webgpu, three/tsl, three/addons/ and nothing else
    const expectedSpecifiers = ['three', 'three/webgpu', 'three/tsl', 'three/addons/'].sort();
    const actualSpecifiers = Object.keys(parsedImportMap.imports).sort();

    assert.deepEqual(
      actualSpecifiers,
      expectedSpecifiers,
      `Served H1 import map must contain exactly [${expectedSpecifiers.join(', ')}], got [${actualSpecifiers.join(', ')}]`
    );
    assert.equal(actualSpecifiers.length, 4, 'Import map must contain exactly 4 routed specifiers');

    // Assert no scopes or unexpected top-level import map properties
    const topLevelKeys = Object.keys(parsedImportMap).sort();
    assert.deepEqual(topLevelKeys, ['imports'], 'Import map must not contain unexpected keys like scopes');

    // 3. Assert exact target mappings point to compatibility facade endpoints
    assert.equal(
      parsedImportMap.imports['three'],
      '/compat-facade/webgpu.js',
      '"three" specifier must route to /compat-facade/webgpu.js'
    );
    assert.equal(
      parsedImportMap.imports['three/webgpu'],
      '/compat-facade/webgpu.js',
      '"three/webgpu" specifier must route to /compat-facade/webgpu.js'
    );
    assert.equal(
      parsedImportMap.imports['three/tsl'],
      '/compat-facade/tsl.js',
      '"three/tsl" specifier must route to /compat-facade/tsl.js'
    );
    assert.equal(
      parsedImportMap.imports['three/addons/'],
      '/compat-facade/addons/',
      '"three/addons/" prefix specifier must route to /compat-facade/addons/'
    );

    // 4. Verify also against root URL (/) which serves H1 as default entry
    const rootRes = await fetch(`${url}/`);
    assert.equal(rootRes.status, 200, 'Served root (/) must return HTTP 200');
    const rootHtml = await rootRes.text();
    const rootMatch = importMapRegex.exec(rootHtml);
    assert.ok(rootMatch, 'Served root HTML must contain <script type="importmap"> tag');
    const rootImportMap = JSON.parse(rootMatch[1]);
    const rootSpecifiers = Object.keys(rootImportMap.imports).sort();
    assert.deepEqual(
      rootSpecifiers,
      expectedSpecifiers,
      'Root entry import map must also contain exactly the 4 intended specifiers'
    );
  } finally {
    await close();
  }
});

test('Dynamic Import Closure: Dynamic import() sites reachable from served H1 and Inspector resolve with HTTP 200 and JavaScript content-type', async () => {
  const { server, port, url, close } = await startDevServer({ port: 0 });

  try {
    const pageUrl = `${url}/examples/webgpu_performance_renderbundle.html`;
    const pageRes = await fetch(pageUrl);
    assert.equal(pageRes.status, 200, 'Served H1 HTML must return 200 OK');
    const pageHtml = await pageRes.text();

    const { importMap, moduleScripts } = parseHtmlEntries(pageHtml, pageUrl);

    // 1. Traverse the served module graph starting from H1 inline script and Inspector entry point
    const visited = new Set();
    const queue = [];

    for (const script of moduleScripts) {
      queue.push({ isInline: true, content: script.inlineContent, url: script.id, referrerUrl: pageUrl });
    }

    const inspectorSpecifier = 'three/addons/inspector/Inspector.js';
    const inspectorUrl = resolveModuleSpecifier(inspectorSpecifier, pageUrl, importMap, {
      mapBaseUrl: pageUrl,
    });
    queue.push({ isInline: false, url: inspectorUrl, referrerUrl: pageUrl });

    const dynamicImportSites = [];
    const discoveredDynamicTargets = [];

    while (queue.length > 0) {
      const current = queue.shift();
      if (visited.has(current.url)) continue;
      visited.add(current.url);

      let code;
      if (current.isInline) {
        code = current.content;
      } else {
        const res = await fetch(current.url);
        assert.equal(res.status, 200, `Module ${current.url} must return 200`);
        code = await res.text();
      }

      // AST analysis using Acorn
      const analysis = analyzeModuleAst(code, current.url);

      // Collect all dynamic import sites in this module
      if (analysis.dynamicImports && analysis.dynamicImports.length > 0) {
        for (const di of analysis.dynamicImports) {
          dynamicImportSites.push({
            referrerUrl: current.url,
            classification: di.classification,
            specifier: di.specifier,
            sourceSpan: di.sourceSpan,
            unresolved: di.unresolved,
          });

          // If literal, resolve target URL directly
          if (di.classification === 'literal' && di.specifier) {
            const targetUrl = resolveModuleSpecifier(di.specifier, current.url, importMap, {
              mapBaseUrl: pageUrl,
            });
            discoveredDynamicTargets.push({
              sourceSite: `${new URL(current.url).pathname}:${di.sourceSpan.start.line}`,
              specifier: di.specifier,
              targetUrl,
              classification: 'literal',
            });
          }
        }
      }

      // In Settings.js (inspector/tabs/Settings.js), import(extUrl) is a variable dynamic import
      // that loads inspector extensions. Extract declared extension URLs from module code.
      if (current.url.includes('/inspector/tabs/Settings.js')) {
        const extensionUrlMatches = [...code.matchAll(/url:\s*['"]([^'"]+)['"]/g)].map(m => m[1]);
        assert.ok(
          extensionUrlMatches.length >= 2,
          'Settings.js must declare at least 2 extension URLs (Color Grading and TSL Graph)'
        );

        for (const extRelUrl of extensionUrlMatches) {
          const resolvedExtUrl = new URL(extRelUrl, current.url).href;
          discoveredDynamicTargets.push({
            sourceSite: `${new URL(current.url).pathname}:285`,
            specifier: extRelUrl,
            targetUrl: resolvedExtUrl,
            classification: 'extension_target',
          });
        }
      }

      // Continue static traversal to discover all reachable modules
      const specifiers = [
        ...analysis.staticImports.map(i => i.specifier),
        ...analysis.staticExports.filter(e => e.specifier).map(e => e.specifier),
      ];

      for (const specifier of specifiers) {
        const resolved = resolveModuleSpecifier(specifier, current.url, importMap, {
          mapBaseUrl: pageUrl,
        });
        if (!visited.has(resolved)) {
          queue.push({ isInline: false, url: resolved, referrerUrl: current.url });
        }
      }
    }

    // 2. Assert enumeration census of dynamic import sites
    // With direct routing imports (no WebGL barrel), exact_backend.mjs is not loaded into H1 WebGPU graph.
    // Dynamic import site in Settings.js:285 -> import(extUrl) [variable/nonliteral] is discovered.
    assert.ok(
      dynamicImportSites.length >= 1,
      `Reachable graph must contain dynamic import() sites, found: ${dynamicImportSites.length}`
    );

    // Verify absence of gratuitous exact_backend dynamic import sites in WebGPU route
    assert.ok(
      !dynamicImportSites.some(s => s.referrerUrl.includes('/tools/compat/exact_backend.mjs')),
      'WebGPU facade must not load exact_backend.mjs or its dynamic imports'
    );

    const settingsSite = dynamicImportSites.find(s => s.referrerUrl.includes('/inspector/tabs/Settings.js'));
    assert.ok(settingsSite, 'Must discover dynamic import() site in Settings.js');
    assert.equal(
      settingsSite.classification,
      'nonliteral',
      'Settings.js dynamic import must be honestly classified as nonliteral'
    );
    assert.equal(
      settingsSite.unresolved,
      true,
      'Settings.js variable dynamic import must be marked unresolved in AST analysis'
    );

    // 3. Fetch each dynamic target through the dev server and assert 200 + JavaScript content-type
    // (Settings.js declares 2 extension targets: ColorGrading and TSLGraph)
    assert.ok(
      discoveredDynamicTargets.length >= 2,
      `Must discover at least 2 dynamic targets (found ${discoveredDynamicTargets.length})`
    );

    for (const target of discoveredDynamicTargets) {
      const res = await fetch(target.targetUrl);
      assert.equal(
        res.status,
        200,
        `Dynamic import target "${target.targetUrl}" (from ${target.sourceSite}) must return HTTP 200`
      );

      const contentType = res.headers.get('content-type') || '';
      assert.ok(
        contentType.includes('application/javascript') || contentType.includes('text/javascript'),
        `Target "${target.targetUrl}" must return JavaScript content-type, got "${contentType}"`
      );

      const body = await res.text();
      assert.ok(body.length > 0, `Target "${target.targetUrl}" body must not be empty`);

      // Verify each target parses as valid ES module JavaScript
      assert.doesNotThrow(
        () => acorn.parse(body, { ecmaVersion: 'latest', sourceType: 'module' }),
        `Target "${target.targetUrl}" must be valid parseable ES module`
      );
    }
  } finally {
    await close();
  }
});

test('Module Isolation Guard: Generated WebGPU facade uses direct routing imports without loading three.module.js or invoking Math.random()', async () => {
  const generatedSource = generateRoutedWebGPUSource({
    importBase: 'file://' + REPO_ROOT,
  });

  // Verify explicit direct import paths in source
  assert.ok(
    generatedSource.includes('tools/compat/route_types.mjs'),
    'Must import route_types.mjs directly'
  );
  assert.ok(
    generatedSource.includes('tools/compat/construction_adapter.mjs'),
    'Must import construction_adapter.mjs directly'
  );
  assert.ok(
    !generatedSource.includes('tools/compat/index.mjs'),
    'Barrel import tools/compat/index.mjs must be absent'
  );
  assert.ok(
    !generatedSource.includes('three.module.js'),
    'three.module.js must be absent to prevent RNG consumption shift'
  );

  // Verify direct imports do not execute Math.random() calls (no core singletons instantiated)
  let rngCalls = 0;
  const origRandom = Math.random;
  Math.random = () => {
    rngCalls++;
    return origRandom();
  };
  try {
    await import(path.resolve(REPO_ROOT, 'tools/compat/route_types.mjs'));
    await import(path.resolve(REPO_ROOT, 'tools/compat/construction_adapter.mjs'));
    assert.equal(rngCalls, 0, 'Direct routing imports must invoke Math.random() 0 times');
  } finally {
    Math.random = origRandom;
  }
});
