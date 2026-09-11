/**
 * @file tools/compat-facade/dev_server.mjs
 * Development server for FrankenThreeD compatibility facade routing (Plan §3.4, §4.7, §5.1, §5.12).
 *
 * Implements bead f3d-04-module-routing-exact-boundaries-6mv.7:
 * - Serves original upstream H1 (examples/webgpu_performance_renderbundle.html) through compatibility facade.
 * - Dynamically rewrites <script type="importmap"> to route three, three/webgpu, three/tsl, three/addons/*.
 * - Preserves the entire upstream application script, GUI controls, animation loop, and all assets byte-for-byte.
 * - Serves genuine upstream WebGPURenderer for both WebGPU and WebGL modes (backend=webgl uses WebGLBackend, NOT legacy WebGLRenderer).
 * - Exposes external construction routing decisions via __f3d_router__ without mutating object instances.
 * - Zero external dependencies; uses node:http, node:fs, node:path.
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  FACADE_NO_CLAIM_ATTESTATION,
  transformHtmlImportMap,
  createDevImportMap,
} from './index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../..');

export const DEFAULT_DEV_PORT = 8080;
export const DEFAULT_DEV_HOST = '127.0.0.1';

export const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.hdr': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json',
};

/**
 * Generates the source for the routed `three/webgpu` compatibility facade.
 * Re-exports the complete export surface from upstream/three.js/build/three.webgpu.js,
 * while routing WebGPURenderer construction through RendererConstructionRouter.
 *
 * Uses a Proxy constructor adapter over the genuine upstream WebGPURenderer
 * preserving prototype chain, instanceof, subclassing, and new.target semantics
 * without illegal assignment to read-only class prototype.
 *
 * @param {Object} [options]
 * @param {string} [options.importBase=''] Optional base URL or directory prefix for imports
 * @returns {string} JavaScript module source
 */
export function generateRoutedWebGPUSource(options = {}) {
  const importBase = (options.importBase || '').replace(/\/+$/, '');
  const webgpuModulePath = `${importBase}/upstream/three.js/build/three.webgpu.js`;
  const compatModulePath = `${importBase}/tools/compat/index.mjs`;

  return `/**
 * FrankenThreeD Compatibility Facade: three/webgpu
 * Pinned Three.js r186 retained export surface with construction routing (Plan §3.4, §5.1).
 *
 * Attestation: ${FACADE_NO_CLAIM_ATTESTATION}
 */

import * as UpstreamThreeWebGPU from '${webgpuModulePath}';
import {
  RendererConstructionRouter,
  ExecutionRoute,
} from '${compatModulePath}';

// 1. Re-export all members from upstream three.webgpu.js
export * from '${webgpuModulePath}';

// 2. Global router singleton recording construction decisions
export const router = new RendererConstructionRouter({
  implementations: {
    [ExecutionRoute.EXACT_BACKEND]: {
      WebGPURenderer: UpstreamThreeWebGPU.WebGPURenderer,
      default: UpstreamThreeWebGPU.WebGPURenderer,
    },
    [ExecutionRoute.RETAINED_UPSTREAM]: {
      WebGPURenderer: UpstreamThreeWebGPU.WebGPURenderer,
      default: UpstreamThreeWebGPU.WebGPURenderer,
    },
  },
});

if (typeof window !== 'undefined') {
  window.__f3d_router__ = router;
  window.__f3d_h1_facade_active__ = true;
}

// 3. Routed WebGPURenderer constructor proxy preserving genuine upstream prototype,
// instanceof checks, subclassing, and new.target semantics without mutating native prototypes.
export const WebGPURenderer = new Proxy(UpstreamThreeWebGPU.WebGPURenderer, {
  construct(target, args, newTarget) {
    const params = args[0] || {};

    // Subclassing support: if called via a derived class constructor, preserve new.target via Reflect.construct without mutating prototypes
    if (newTarget !== WebGPURenderer && newTarget !== target) {
      return router.routeAndConstruct({
        constructorFn: function (opts) {
          return Reflect.construct(target, args, newTarget);
        },
        constructorName: 'WebGPURenderer',
        options: params,
        sourceSpan: 'webgpu_performance_renderbundle.html:188:13',
      });
    }

    // Direct new WebGPURenderer(...) construction:
    // Both WebGPU mode and forceWebGL mode construct genuine WebGPURenderer.
    // When forceWebGL: true, WebGPURenderer instantiates WebGLBackend (never legacy WebGLRenderer).
    return router.routeAndConstruct({
      constructorFn: target,
      constructorName: 'WebGPURenderer',
      options: params,
      sourceSpan: 'webgpu_performance_renderbundle.html:188:13',
    });
  },
});

// Keep the retained prototype pristine. This development routing proxy preserves
// instanceof, but direct instances still identify the upstream constructor.
`;
}

/**
 * Generates the source for the routed `three` root compatibility facade.
 *
 * @param {Object} [options]
 * @param {string} [options.importBase='']
 * @returns {string} JavaScript module source
 */
export function generateRoutedThreeRootSource(options = {}) {
  const importBase = (options.importBase || '').replace(/\/+$/, '');
  const modulePath = `${importBase}/upstream/three.js/build/three.module.js`;
  return `/**
 * FrankenThreeD Compatibility Facade: three (root ESM)
 * Pinned Three.js r186 retained export surface (Plan §3.4, §5.1).
 *
 * Attestation: ${FACADE_NO_CLAIM_ATTESTATION}
 */

export * from '${modulePath}';
`;
}

/**
 * Generates the source for the routed `three/tsl` compatibility facade.
 *
 * @param {Object} [options]
 * @param {string} [options.importBase='']
 * @returns {string} JavaScript module source
 */
export function generateRoutedTslSource(options = {}) {
  const importBase = (options.importBase || '').replace(/\/+$/, '');
  const modulePath = `${importBase}/upstream/three.js/build/three.tsl.js`;
  return `/**
 * FrankenThreeD Compatibility Facade: three/tsl
 * Pinned Three.js r186 retained export surface (Plan §3.4, §5.1).
 *
 * Attestation: ${FACADE_NO_CLAIM_ATTESTATION}
 */

export * from '${modulePath}';
`;
}

/**
 * Creates the FrankenThreeD development server instance.
 *
 * @param {Object} [options]
 * @param {string} [options.repoRoot] Root repository directory (defaults to repo root)
 * @param {string} [options.h1RelativePath] Relative path to H1 html file
 * @returns {http.Server} Configured Node HTTP server
 */
export function createCompatDevServer(options = {}) {
  const repoRoot = options.repoRoot || REPO_ROOT;
  const h1RelPath = options.h1RelativePath || 'upstream/three.js/examples/webgpu_performance_renderbundle.html';
  const h1AbsPath = path.resolve(repoRoot, h1RelPath);

  const server = http.createServer((req, res) => {
    // Enable CORS and headers required for WebGPU / SharedArrayBuffer
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', '*');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const urlObj = new URL(req.url, 'http://127.0.0.1');
    const pathname = decodeURIComponent(urlObj.pathname);

    // 1. Root & H1 Example Page
    if (
      pathname === '/' ||
      pathname === '/examples/webgpu_performance_renderbundle.html' ||
      pathname === '/webgpu_performance_renderbundle.html'
    ) {
      if (!fs.existsSync(h1AbsPath)) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end(`H1 demo file not found at: ${h1AbsPath}`);
        return;
      }

      try {
        const originalHtml = fs.readFileSync(h1AbsPath, 'utf-8');
        const transformedHtml = transformHtmlImportMap(originalHtml);

        res.writeHead(200, {
          'Content-Type': 'text/html; charset=utf-8',
          'X-FrankenThreeD-Facade': 'development-mode',
          'X-FrankenThreeD-Attestation': FACADE_NO_CLAIM_ATTESTATION,
        });
        res.end(transformedHtml);
        return;
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end(`Error transforming H1 HTML: ${err.message}`);
        return;
      }
    }

    // 1b. Favicon: return 204 No Content to eliminate error-level browser log entries
    if (pathname === '/favicon.ico') {
      res.writeHead(204, {
        'Content-Type': 'image/x-icon',
        'Cache-Control': 'public, max-age=86400',
      });
      res.end();
      return;
    }

    // 2. Compatibility Facade Endpoints
    if (pathname === '/compat-facade/webgpu.js') {
      const source = generateRoutedWebGPUSource();
      res.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'X-FrankenThreeD-Attestation': FACADE_NO_CLAIM_ATTESTATION,
      });
      res.end(source);
      return;
    }

    if (pathname === '/compat-facade/three.js') {
      const source = generateRoutedThreeRootSource();
      res.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'X-FrankenThreeD-Attestation': FACADE_NO_CLAIM_ATTESTATION,
      });
      res.end(source);
      return;
    }

    if (pathname === '/compat-facade/tsl.js') {
      const source = generateRoutedTslSource();
      res.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'X-FrankenThreeD-Attestation': FACADE_NO_CLAIM_ATTESTATION,
      });
      res.end(source);
      return;
    }

    // 3. /compat-facade/addons/* -> upstream/three.js/examples/jsm/*
    if (pathname.startsWith('/compat-facade/addons/')) {
      const subpath = pathname.slice('/compat-facade/addons/'.length);
      const targetPath = path.resolve(repoRoot, 'upstream/three.js/examples/jsm', subpath);
      if (fs.existsSync(targetPath) && fs.statSync(targetPath).isFile()) {
        const ext = path.extname(targetPath).toLowerCase();
        const contentType = MIME_TYPES[ext] || 'application/javascript; charset=utf-8';
        res.writeHead(200, {
          'Content-Type': contentType,
          'X-FrankenThreeD-Attestation': FACADE_NO_CLAIM_ATTESTATION,
        });
        fs.createReadStream(targetPath).pipe(res);
        return;
      }
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end(`Addon not found: ${subpath}`);
      return;
    }

    // 4. Convenience route for H1 local relative assets (example.css, jsm/*)
    if (pathname === '/example.css' || pathname === '/examples/example.css') {
      const cssPath = path.resolve(repoRoot, 'upstream/three.js/examples/example.css');
      if (fs.existsSync(cssPath)) {
        res.writeHead(200, { 'Content-Type': 'text/css; charset=utf-8' });
        fs.createReadStream(cssPath).pipe(res);
        return;
      }
    }

    // 4b. Static alias for /build/* -> upstream/three.js/build/* (source-relative ../build/* from examples)
    if (pathname.startsWith('/build/')) {
      const subpath = pathname.slice('/build/'.length);
      const targetPath = path.resolve(repoRoot, 'upstream/three.js/build', subpath);
      if (fs.existsSync(targetPath) && fs.statSync(targetPath).isFile()) {
        const ext = path.extname(targetPath).toLowerCase();
        const contentType = MIME_TYPES[ext] || 'application/javascript; charset=utf-8';
        res.writeHead(200, {
          'Content-Type': contentType,
          'X-FrankenThreeD-Attestation': FACADE_NO_CLAIM_ATTESTATION,
        });
        fs.createReadStream(targetPath).pipe(res);
        return;
      }
    }

    // 5. Static file serving from repo root (upstream, tools, tests)
    const sanitizedRelPath = pathname.replace(/^\/+/, '');
    const candidatePath = path.resolve(repoRoot, sanitizedRelPath);

    // Security check: ensure path stays within repoRoot
    if (!candidatePath.startsWith(repoRoot)) {
      res.writeHead(403, { 'Content-Type': 'text/plain' });
      res.end('Access denied');
      return;
    }

    if (fs.existsSync(candidatePath) && fs.statSync(candidatePath).isFile()) {
      const ext = path.extname(candidatePath).toLowerCase();
      const contentType = MIME_TYPES[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': contentType });
      fs.createReadStream(candidatePath).pipe(res);
      return;
    }

    // 6. Upstream examples fallback (e.g. textures, fonts, sounds, jsm)
    // Strip leading 'examples/' if present to avoid doubled examples/examples/ resolution
    const examplesRelSubpath = sanitizedRelPath.replace(/^examples\//, '');
    const examplesFallback = path.resolve(repoRoot, 'upstream/three.js/examples', examplesRelSubpath);
    if (fs.existsSync(examplesFallback) && fs.statSync(examplesFallback).isFile()) {
      const ext = path.extname(examplesFallback).toLowerCase();
      const contentType = MIME_TYPES[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': contentType });
      fs.createReadStream(examplesFallback).pipe(res);
      return;
    }

    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end(`File not found: ${pathname}`);
  });

  return server;
}

/**
 * Starts the dev server on the specified port and host.
 *
 * @param {Object} [options]
 * @param {number} [options.port] Port to listen on (default: 8080, or 0 for ephemeral)
 * @param {string} [options.host] Host to bind (default: 127.0.0.1)
 * @returns {Promise<{ server: http.Server, port: number, host: string, url: string, close: Function }>}
 */
export function startDevServer(options = {}) {
  const port = options.port !== undefined ? options.port : DEFAULT_DEV_PORT;
  const host = options.host || DEFAULT_DEV_HOST;
  const server = createCompatDevServer(options);

  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(port, host, () => {
      const boundAddress = server.address();
      const actualPort = boundAddress.port;
      const url = `http://${host}:${actualPort}`;
      resolve({
        server,
        port: actualPort,
        host,
        url,
        close: () =>
          new Promise(res => {
            if (typeof server.closeAllConnections === 'function') {
              server.closeAllConnections();
            }
            server.close(res);
          }),
      });
    });
  });
}

// Direct CLI launch
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  let port = DEFAULT_DEV_PORT;
  let host = DEFAULT_DEV_HOST;

  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i] === '--port' && process.argv[i + 1]) {
      port = parseInt(process.argv[i + 1], 10);
      i++;
    } else if (process.argv[i] === '--host' && process.argv[i + 1]) {
      host = process.argv[i + 1];
      i++;
    }
  }

  startDevServer({ port, host })
    .then(({ port: actualPort, url }) => {
      console.log(`[f3d-dev-server] FrankenThreeD Compatibility Dev Server running at: ${url}/`);
      console.log(`[f3d-dev-server] H1 Demo (WebGPU): ${url}/examples/webgpu_performance_renderbundle.html`);
      console.log(`[f3d-dev-server] H1 Demo (WebGL):  ${url}/examples/webgpu_performance_renderbundle.html?backend=webgl`);
      console.log(`[f3d-dev-server] Pinned Three.js r186 retained compatibility active.`);
      console.log(`[f3d-dev-server] Press Ctrl+C to terminate.`);
    })
    .catch(err => {
      console.error('[f3d-dev-server] Failed to start server:', err);
      process.exit(1);
    });
}
