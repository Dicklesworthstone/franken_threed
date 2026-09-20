import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { emitPortablePackage, portablePackageManifest } from './portable-package.mjs';

const cli = fileURLToPath(new URL('./portable-package.mjs', import.meta.url));
function fixture(t, overrides = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-portable-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, 'source');
  const put = (name, contents) => {
    const file = path.join(source, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, contents);
  };
  const manifest = {
    name: 'three', version: '0.186.0', type: 'module', license: 'MIT',
    main: './build/root.cjs', module: './build/root.js',
    exports: {
      '.': { import: './build/root.js', require: './build/root.cjs' },
      './webgpu': './build/root.js',
      './addons/*': './examples/jsm/*',
      './private/*': null,
      './codec.wasm': './assets/codec.wasm',
    },
    imports: { '#value': './src/value.js' },
    dependencies: { 'external-runtime': '^1.0.0' },
    scripts: { postinstall: 'throw new Error("must not execute")' },
    ...overrides,
  };
  put('package.json', JSON.stringify(manifest));
  put('build/root.js', 'export { value } from "#value"; export class Mesh {}');
  put('build/root.cjs', 'module.exports = { value: 42, synchronous: true };');
  put('src/value.js', 'export const value = 42;');
  put('examples/jsm/Loader.js', 'import { Mesh } from "three"; export { Mesh }; export default Mesh; export const worker = new URL("../workers/decoder.js", import.meta.url);');
  put('examples/workers/decoder.js', 'export const decoder = true;');
  put('assets/codec.wasm', Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  put('LICENSE', 'MIT fixture license');
  put('.git/config', 'not a runtime asset');
  put('node_modules/not-packaged/index.js', 'not a runtime asset');
  return { root, source, put, manifest, out: path.join(root, 'node_modules/@franken/three') };
}

function run(root, source, type = 'module') {
  const result = spawnSync(process.execPath, [`--input-type=${type}`, '-e', source], { cwd: root, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result.stdout.trim();
}

test('relocated package executes ESM, CJS, self-references and worker URLs without the source', (t) => {
  const f = fixture(t);
  const result = emitPortablePackage(f.out, { packageDir: f.source });
  fs.rmSync(f.source, { recursive: true });
  assert.equal(result.manifest.name, '@franken/three');
  assert.equal(run(f.root, `
    import assert from 'node:assert/strict';
    import fs from 'node:fs';
    import { Mesh, value } from '@franken/three';
    import { Mesh as GPU } from '@franken/three/webgpu';
    import DefaultMesh, { Mesh as Addon, worker } from '@franken/three/addons/Loader.js';
    assert.equal(Mesh, GPU); assert.equal(Mesh, Addon); assert.equal(Mesh, DefaultMesh);
    assert.equal(value, 42); assert.match(fs.readFileSync(worker, 'utf8'), /decoder/);
    console.log('esm-ok');
  `), 'esm-ok');
  assert.equal(run(f.root, `const assert = require('node:assert/strict'); const api = require('@franken/three'); assert.equal(api.value,42); assert.equal(api.synchronous,true); assert.equal(api.then,undefined); console.log('cjs-ok');`, 'commonjs'), 'cjs-ok');
});

test('preserves binary assets, private export blockers and original nested package metadata', (t) => {
  const f = fixture(t);
  emitPortablePackage(f.out, { packageDir: f.source });
  fs.rmSync(f.source, { recursive: true });
  run(f.root, `
    const assert = require('node:assert/strict'); const fs = require('node:fs');
    assert.deepEqual([...fs.readFileSync(require.resolve('@franken/three/codec.wasm'))], [0,97,115,109,1,0,0,0]);
    assert.throws(() => require.resolve('@franken/three/private/secret.js'), { code: 'ERR_PACKAGE_PATH_NOT_EXPORTED' });
  `, 'commonjs');
  assert.equal(JSON.parse(fs.readFileSync(path.join(f.out, '_retained/package.json'))).name, 'three');
  assert.equal(fs.readFileSync(path.join(f.out, 'LICENSE'), 'utf8'), 'MIT fixture license');
});

test('retains conditional order, fallback arrays, wildcard targets and runtime dependencies', () => {
  const source = { exports: { '.': { browser: './web.js', node: { import: './esm.js', require: './cjs.cjs' }, default: ['./fallback.js', null] }, './x/*': './x/*', './blocked': null }, dependencies: { x: '1' }, scripts: { install: 'bad' }, devDependencies: { dev: '1' } };
  const result = portablePackageManifest(source);
  assert.deepEqual(Object.keys(result.exports['.']), ['browser', 'node', 'default']);
  assert.deepEqual(result.exports['.'].default, ['./_retained/fallback.js', null]);
  assert.equal(result.exports['.'].node.require, './_retained/cjs.cjs');
  assert.equal(result.exports['./x/*'], './_retained/x/*');
  assert.equal(result.exports['./blocked'], null);
  assert.deepEqual(result.dependencies, { x: '1' });
  assert.equal(result.scripts, undefined); assert.equal(result.devDependencies, undefined);
  assert.equal(source.exports['./x/*'], './x/*');
});

test('rewrites legacy bundler, declaration, bin and browser fields without touching external dependencies', () => {
  const result = portablePackageManifest({ exports: './index.js', main: 'index.cjs', types: './types/index.d.ts', bin: { tool: 'bin/tool.js' }, browser: { './index.js': './web.js', fs: false }, imports: { '#local': './src/local.js', '#external': 'external' } });
  assert.equal(result.main, './_retained/index.cjs');
  assert.equal(result.types, './_retained/types/index.d.ts');
  assert.equal(result.bin.tool, './_retained/bin/tool.js');
  assert.deepEqual(result.browser, { './_retained/index.js': './_retained/web.js', fs: false });
  assert.deepEqual(result.imports, { '#local': './_retained/src/local.js', '#external': 'external' });
});

test('does not copy checkout metadata or installed dependencies', (t) => {
  const f = fixture(t);
  const result = emitPortablePackage(f.out, { packageDir: f.source });
  assert.ok(result.files.includes('_retained/examples/workers/decoder.js'));
  assert.equal(fs.existsSync(path.join(f.out, '_retained/.git')), false);
  assert.equal(fs.existsSync(path.join(f.out, '_retained/node_modules')), false);
  assert.equal(result.files.length, result.count);
});

test('rejects traversal, malformed targets and unsupported package contracts', () => {
  for (const target of ['../escape.js', './a/../../escape.js', './%2e%2e/escape.js', '/tmp/x.js', './a\\b.js', './node_modules/x.js', './a//b.js']) {
    assert.throws(() => portablePackageManifest({ exports: target }), /target/);
  }
  assert.throws(() => portablePackageManifest({}), /exports contract/);
  assert.throws(() => portablePackageManifest({ exports: './x.js' }, { packageName: '../bad' }), /package name/);
});

test('refuses existing destinations without modifying any files', (t) => {
  const f = fixture(t);
  fs.mkdirSync(f.out, { recursive: true });
  fs.writeFileSync(path.join(f.out, 'keep'), 'original');
  assert.throws(() => emitPortablePackage(f.out, { packageDir: f.source }), /already exists/);
  assert.equal(fs.readFileSync(path.join(f.out, 'keep'), 'utf8'), 'original');
});

test('refuses recursive output even through a symlinked parent', (t) => {
  const f = fixture(t);
  assert.throws(() => emitPortablePackage(path.join(f.source, 'dist'), { packageDir: f.source }), /outside/);
  fs.symlinkSync(f.source, path.join(f.root, 'alias'), 'dir');
  assert.throws(() => emitPortablePackage(path.join(f.root, 'alias/dist'), { packageDir: f.source }), /outside/);
  assert.equal(fs.existsSync(path.join(f.source, 'dist')), false);
});

test('fails before publishing symlinked runtime assets', (t) => {
  const f = fixture(t);
  fs.symlinkSync(path.join(f.source, 'src/value.js'), path.join(f.source, 'linked.js'));
  assert.throws(() => emitPortablePackage(f.out, { packageDir: f.source }), /symlinks/);
  assert.equal(fs.existsSync(f.out), false);
});

test('CLI produces a working package and reports retained execution honestly', (t) => {
  const f = fixture(t);
  const result = spawnSync(process.execPath, [cli, '--out', f.out, '--package-dir', f.source], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.execution, 'retained-upstream-js'); assert.equal(report.accelerated, false);
  assert.ok(report.retainedFiles > 0);
  const invalid = spawnSync(process.execPath, [cli, '--out'], { encoding: 'utf8' });
  assert.equal(invalid.status, 1); assert.match(invalid.stderr, /Usage/);
});

test('relocates tree-shaking globs and versioned declaration paths', () => {
  const source = {
    exports: './index.js',
    sideEffects: ['./src/nodes/**/*', '*.css', '**/*.wasm'],
    typesVersions: { '>=5.0': { '*': ['types/*', 'fallback/*'] } },
  };
  const result = portablePackageManifest(source);
  assert.deepEqual(result.sideEffects, ['./_retained/src/nodes/**/*', './_retained/**/*.css', './_retained/**/*.wasm']);
  assert.deepEqual(result.typesVersions, { '>=5.0': { '*': ['./_retained/types/*', './_retained/fallback/*'] } });
  assert.deepEqual(source.sideEffects, ['./src/nodes/**/*', '*.css', '**/*.wasm']);
  for (const value of [true, false]) assert.equal(portablePackageManifest({ exports: './index.js', sideEffects: value }).sideEffects, value);
  assert.throws(() => portablePackageManifest({ exports: './index.js', sideEffects: 'invalid' }), /sideEffects/);
  assert.throws(() => portablePackageManifest({ exports: './index.js', sideEffects: ['!src/*'] }), /sideEffects/);
});

test('excludes nested packaging controls while leaving source files untouched', (t) => {
  const f = fixture(t);
  for (const name of ['.npmignore', '.gitignore', '.npmrc']) f.put(`assets/${name}`, 'original config');
  f.put('examples/vendor/node_modules/excluded.js', 'not retained');
  const result = emitPortablePackage(f.out, { packageDir: f.source });
  for (const name of ['.npmignore', '.gitignore', '.npmrc']) {
    assert.equal(fs.existsSync(path.join(f.out, '_retained/assets', name)), false);
    assert.equal(fs.readFileSync(path.join(f.source, 'assets', name), 'utf8'), 'original config');
  }
  assert.equal(fs.existsSync(path.join(f.out, '_retained/examples/vendor/node_modules')), false);
  assert.ok(result.files.includes('_retained/assets/codec.wasm'));
});
