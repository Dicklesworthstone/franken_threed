import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { startDevServer } from './dev_server.mjs';

// Retain temporary fixtures for inspection; never touch the pinned checkout.
function fixture() {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-http-safety-'));
  const repoRoot = path.join(parent, 'repo');
  const files = {
    'repo/upstream/three.js/examples/webgpu_performance_renderbundle.html': '<!doctype html><script nonce="keep-nonce" type="importmap">{"imports":{"three":"../build/three.webgpu.js"},"scopes":{"/cpu/":{"three":"../build/three.module.js"}}}</script><script type="module">window.applicationIsUnchanged = true;</script>',
    'repo/upstream/three.js/examples/jsm/controls/Example.js': 'export const control = 1;',
    'repo/upstream/three.js/examples/example.css': 'body { margin: 0; }',
    'repo/upstream/three.js/examples/textures/example.bin': 'TEXTURE',
    'repo/upstream/three.js/build/three.module.js': 'export const value = 1;',
    'repo/tools/example.mjs': 'export const tool = 1;',
    'repo/private.txt': 'REPOSITORY_FILE_OUTSIDE_ALIAS',
    'repo-private/outside.txt': 'SYNTHETIC_OUTSIDE_SECRET',
  };
  for (const [name, content] of Object.entries(files)) {
    const target = path.join(parent, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, content);
  }
  return { parent, repoRoot };
}
function request(port, target, method = 'GET') {
  return new Promise((resolve, reject) => {
    // Do not use fetch: raw request paths exercise encoded traversal before URL normalization.
    const req = http.request({ host: '127.0.0.1', port, path: target, method, agent: false }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('error', reject);
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString() }));
    });
    req.setTimeout(5000, () => req.destroy(new Error('HTTP request timed out')));
    req.on('error', reject);
    req.end();
  });
}
async function withServer(run, options = {}) {
  const files = fixture();
  const server = await startDevServer({ ...files, port: 0, ...options });
  try { await run(server.port, files); } finally { await server.close(); }
}

test('real HTTP H1 routing preserves scopes, CSP nonce and application bytes', async () => {
  await withServer(async (port) => {
    const res = await request(port, '/');
    assert.equal(res.status, 200);
    assert.ok(res.body.includes('<script nonce="keep-nonce" type="importmap">'));
    assert.ok(res.body.endsWith('<script type="module">window.applicationIsUnchanged = true;</script>'));
    const map = JSON.parse(/<script[^>]*>([\s\S]*?)<\/script>/.exec(res.body)[1]);
    assert.equal(map.imports.three, '/compat-facade/webgpu.js');
    assert.equal(map.scopes['/cpu/'].three, '/compat-facade/three.js');
    assert.equal(res.headers['cross-origin-embedder-policy'], 'require-corp');
  });
});

test('real HTTP aliases, repository files and example fallback serve original bytes', async () => {
  await withServer(async (port) => {
    for (const [url, expected] of [
      ['/compat-facade/addons/controls/Example.js', 'export const control = 1;'],
      ['/build/three.module.js?revision=1', 'export const value = 1;'],
      ['/tools/example.mjs', 'export const tool = 1;'],
      ['/textures/example.bin', 'TEXTURE'], ['/examples/textures/example.bin', 'TEXTURE'],
      ['/example.css', 'body { margin: 0; }'], ['/examples/example.css', 'body { margin: 0; }'],
    ]) {
      const res = await request(port, url);
      assert.equal(res.status, 200, url);
      assert.equal(res.body, expected, url);
    }
  });
});

test('encoded traversal cannot escape addon/build aliases', async () => {
  await withServer(async (port) => {
    for (const target of [
      '/compat-facade/addons/..%2f..%2f..%2f..%2fprivate.txt',
      '/build/..%2f..%2f..%2fprivate.txt',
      '/compat-facade/addons/..%2f..%2f..%2f..%2f..%2frepo-private%2foutside.txt',
      '/build/..%2f..%2f..%2f..%2frepo-private%2foutside.txt',
    ]) {
      const res = await request(port, target);
      assert.equal(res.status, 403, target);
      assert.equal(res.body, 'Access denied');
    }
  });
});

test('repository containment rejects a sibling directory sharing its name prefix', async () => {
  await withServer(async (port) => {
    const res = await request(port, '/..%2frepo-private%2foutside.txt');
    assert.equal(res.status, 403);
    assert.equal(res.body, 'Access denied');
  });
});

test('malformed URL encodings are contained and the server remains available', async () => {
  await withServer(async (port) => {
    for (const target of ['/%', '/%FF', '/%E0%A4%A', '/%00', '/%5c']) {
      assert.equal((await request(port, target)).status, 400, target);
      assert.equal((await request(port, '/tools/example.mjs')).status, 200);
    }
  });
});

test('external symlinks cannot leak through any static route', async () => {
  await withServer(async (port, { parent, repoRoot }) => {
    const outside = path.join(parent, 'repo-private/outside.txt');
    const links = [
      ['outside.txt', '/outside.txt'],
      ['upstream/three.js/examples/jsm/outside.txt', '/compat-facade/addons/outside.txt'],
      ['upstream/three.js/build/outside.txt', '/build/outside.txt'],
      ['upstream/three.js/examples/textures/outside.txt', '/textures/outside.txt'],
    ];
    for (const [name, target] of links) {
      fs.symlinkSync(outside, path.join(repoRoot, name));
      const res = await request(port, target);
      assert.equal(res.status, 403, target);
      assert.equal(res.body, 'Access denied');
    }
    fs.symlinkSync(path.join(parent, 'repo-private'), path.join(repoRoot, 'external-dir'), 'dir');
    assert.equal((await request(port, '/external-dir/outside.txt')).status, 403);
    fs.symlinkSync(path.join(repoRoot, 'private.txt'), path.join(repoRoot, 'upstream/three.js/build/alias-escape.txt'));
    assert.equal((await request(port, '/build/alias-escape.txt')).status, 403);
  });
});

test('in-root symlinks, relative repository roots and generated endpoints still work', async () => {
  const files = fixture();
  fs.symlinkSync(path.join(files.repoRoot, 'tools/example.mjs'), path.join(files.repoRoot, 'tool-link.mjs'));
  const server = await startDevServer({repoRoot: path.relative(process.cwd(), files.repoRoot), port: 0});
  try {
    assert.equal((await request(server.port, '/tool-link.mjs')).body, 'export const tool = 1;');
    for (const endpoint of ['three', 'webgpu', 'tsl']) {
      const res = await request(server.port, `/compat-facade/${endpoint}.js`);
      assert.equal(res.status, 200);
      assert.ok(res.body.includes('export * from'));
    }
  } finally { await server.close(); }
});

test('missing paths and directories return 404 without disrupting subsequent requests', async () => {
  await withServer(async (port) => {
    for (const target of ['/missing.bin', '/compat-facade/addons/missing.js', '/build/missing.js', '/tools/', '/textures/']) {
      assert.equal((await request(port, target)).status, 404, target);
    }
    assert.equal((await request(port, '/')).status, 200);
  });
});

test('HTTP methods are read-only and HEAD returns matching headers without a body', async () => {
  await withServer(async (port) => {
    for (const target of ['/', '/compat-facade/webgpu.js', '/build/three.module.js', '/tools/example.mjs']) {
      const head = await request(port, target, 'HEAD');
      const get = await request(port, target);
      assert.equal(head.status, get.status);
      assert.equal(head.headers['content-type'], get.headers['content-type']);
      assert.equal(head.body, '');
    }
    assert.equal((await request(port, '/', 'OPTIONS')).status, 204);
    for (const method of ['POST', 'PUT', 'DELETE']) {
      const res = await request(port, '/', method);
      assert.equal(res.status, 405, method);
      assert.equal(res.headers.allow, 'GET, HEAD, OPTIONS');
    }
  });
});

test('unsafe configured H1 paths and symlinks cannot leave the repository', async () => {
  await withServer(async (port) => {
    assert.equal((await request(port, '/')).status, 403);
  }, {h1RelativePath:'../repo-private/outside.txt'});
  const files = fixture();
  fs.symlinkSync(path.join(files.parent, 'repo-private/outside.txt'), path.join(files.repoRoot, 'outside.html'));
  const server = await startDevServer({...files, port:0, h1RelativePath:'outside.html'});
  try { assert.equal((await request(server.port, '/')).status, 403); } finally { await server.close(); }
});

test('HTML transformation failures return 500 without disabling static assets', async () => {
  await withServer(async (port, {repoRoot}) => {
    fs.writeFileSync(path.join(repoRoot, 'upstream/three.js/examples/webgpu_performance_renderbundle.html'), '<script type="importmap">{bad}</script>');
    assert.equal((await request(port, '/')).status, 500);
    assert.equal((await request(port, '/build/three.module.js')).status, 200);
  });
});
