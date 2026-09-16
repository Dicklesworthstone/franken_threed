import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildNumericKernel } from './numeric_kernel_build.mjs';

const source = `export function integrate(position, velocity, dt) {
  for (let i = 0; i < position.length; i++) {
    position[i] += velocity[i] * dt;
    velocity[i] -= dt;
  }
}`;
const parameterTypes = ['f64[]', 'f64[]', 'f64'];
const cli = fileURLToPath(new URL('./cli.mjs', import.meta.url));

function fixture(t, text = source) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-kernel-build-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const entry = path.join(root, 'update.mjs');
  fs.writeFileSync(entry, text);
  const out = path.join(root, 'output');
  return { root, entry, out };
}

async function load(out) { return import(pathToFileURL(path.join(out, 'kernel.mjs')).href); }
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

function command(entry, out, extra = []) {
  return spawnSync(process.execPath, [cli, '--entry', entry, '--build-kernel', out,
    '--parameter-types', 'f64[],f64[],f64', ...extra], { encoding: 'utf8', timeout: 10000 });
}

test('builds a relocatable executable package with exact binary provenance', async t => {
  const { root, entry, out } = fixture(t);
  const result = buildNumericKernel(entry, out, { parameterTypes });
  assert.deepEqual(fs.readdirSync(out).sort(), result.emittedFiles.slice().sort());
  const manifest = JSON.parse(fs.readFileSync(path.join(out, 'kernel.json'), 'utf8'));
  const binary = fs.readFileSync(path.join(out, 'kernel.wasm'));
  assert.equal(WebAssembly.validate(binary), true);
  assert.equal(manifest.wasmSha256, digest(binary));
  assert.equal(manifest.sourceSha256, digest(source));
  assert.equal(manifest.accelerationClaim, false);
  assert.equal(manifest.kernel.automaticRouteAdmission, false);
  assert.equal(fs.readFileSync(path.join(out, 'retained.mjs'), 'utf8'), `${source}\nexport default integrate;\n`);
  assert.ok(!fs.readFileSync(path.join(out, 'kernel.mjs'), 'utf8').includes(root));

  const moved = path.join(root, 'relocated');
  fs.renameSync(out, moved);
  const { createKernel, retained } = await load(moved);
  const engine = createKernel();
  const position = new Float64Array([1, 2, 3]);
  const velocity = new Float64Array([5, -2, 7]);
  const expectedPosition = position.slice(), expectedVelocity = velocity.slice();
  for (let frame = 0; frame < 120; frame++) {
    engine.run(position, velocity, 1 / 60);
    retained(expectedPosition, expectedVelocity, 1 / 60);
  }
  assert.deepEqual(position, expectedPosition);
  assert.deepEqual(velocity, expectedVelocity);
  assert.equal(engine.diagnostics.wasmCalls, 120);
  assert.equal(engine.diagnostics.fallbackCalls, 0);
});

test('generated package preserves ordinary Array updates through retained fallback', async t => {
  const { entry, out } = fixture(t);
  buildNumericKernel(entry, out, { parameterTypes });
  const { createKernel } = await load(out);
  const engine = createKernel();
  const a = [1, 2], b = [3, 4];
  engine.run(a, b, 0.5);
  assert.deepEqual(a, [2.5, 4]);
  assert.deepEqual(b, [2.5, 3.5]);
  assert.equal(engine.diagnostics.fallbackCalls, 1);
  assert.equal(engine.diagnostics.lastGuardFailure, 'KERNEL_ARRAY_TYPE');
});

test('package imports and executes retained source when Wasm is unavailable', async t => {
  const { entry, out } = fixture(t);
  buildNumericKernel(entry, out, { parameterTypes });
  const prior = Object.getOwnPropertyDescriptor(globalThis, 'WebAssembly');
  Object.defineProperty(globalThis, 'WebAssembly', { configurable: true, value: undefined });
  try {
    const { createKernel } = await load(out);
    const engine = createKernel();
    const a = new Float64Array([1]), b = new Float64Array([2]);
    engine.run(a, b, 0.25);
    assert.equal(a[0], 1.5);
    assert.equal(b[0], 1.75);
    assert.equal(engine.diagnostics.wasmCalls, 0);
    assert.equal(engine.diagnostics.fallbackCalls, 1);
    assert.equal(engine.diagnostics.lastGuardFailure, 'KERNEL_WASM_UNAVAILABLE');
    assert.equal(engine.manifest.functionName, 'integrate');
    engine.dispose(); engine.dispose();
    assert.throws(() => engine.run(a, b, 1), /KERNEL_DISPOSED/);
  } finally { Object.defineProperty(globalThis, 'WebAssembly', prior); }
});

test('policy-denied Wasm construction does not suppress the original exception semantics', async t => {
  const { entry, out } = fixture(t);
  buildNumericKernel(entry, out, { parameterTypes });
  const { createKernel } = await load(out);
  const prior = Object.getOwnPropertyDescriptor(globalThis, 'WebAssembly');
  Object.defineProperty(globalThis, 'WebAssembly', {
    configurable: true, value: { Module: class { constructor() { throw new Error('blocked by host policy'); } } },
  });
  try {
    const engine = createKernel();
    const failure = new Error('original read failure');
    const position = { get length() { throw failure; } };
    assert.throws(() => engine.run(position, [], 1), error => error === failure);
    assert.equal(engine.diagnostics.fallbackCalls, 1);
    assert.equal(engine.diagnostics.lastGuardFailure, 'KERNEL_INITIALIZATION_FAILED');
  } finally { Object.defineProperty(globalThis, 'WebAssembly', prior); }
});

test('build is deterministic and accepts plain declarations, including a retained-named function', async t => {
  const { root, entry, out } = fixture(t, 'function retained(x) { for(let i=0; i<x.length; i++) x[i]*=2; } // trailing comment');
  const result = buildNumericKernel(entry, out, { parameterTypes: ['f64[]'] });
  const other = path.join(root, 'other');
  buildNumericKernel(entry, other, { parameterTypes: ['f64[]'] });
  for (const name of result.emittedFiles) {
    assert.deepEqual(fs.readFileSync(path.join(out, name)), fs.readFileSync(path.join(other, name)));
  }
  const { createKernel } = await load(out);
  const x = new Float64Array([2, 3]);
  createKernel().run(x);
  assert.deepEqual([...x], [4, 6]);
});

test('source refusal happens before output and cannot execute build-time source effects', t => {
  const { entry, out } = fixture(t, 'throw new Error("BUILD_SOURCE_EXECUTED");');
  assert.throws(() => buildNumericKernel(entry, out, { parameterTypes }), /KERNEL_NOT_CLOSED/);
  assert.equal(fs.existsSync(out), false);
});

test('fresh-output enforcement preserves existing directories, files and symlink targets', t => {
  const { root, entry, out } = fixture(t);
  fs.mkdirSync(out);
  fs.writeFileSync(path.join(out, 'keep.txt'), 'caller-owned');
  assert.throws(() => buildNumericKernel(entry, out, { parameterTypes }), { code: 'EEXIST' });
  assert.deepEqual(fs.readdirSync(out), ['keep.txt']);
  assert.equal(fs.readFileSync(path.join(out, 'keep.txt'), 'utf8'), 'caller-owned');
  const link = path.join(root, 'linked');
  fs.symlinkSync(out, link, 'dir');
  assert.throws(() => buildNumericKernel(entry, link, { parameterTypes }), { code: 'EEXIST' });
  assert.throws(() => buildNumericKernel(entry, entry, { parameterTypes }), { code: 'EEXIST' });
  assert.equal(fs.readFileSync(entry, 'utf8'), source);
});

test('existing ingestion CLI emits and runs a real Wasm package without loading Rollup', async t => {
  const { entry, out } = fixture(t);
  const result = command(entry, out, ['--max-memory-pages', '2']);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Numeric kernel package emitted/);
  const { createKernel } = await load(out);
  const engine = createKernel();
  assert.equal(engine.manifest.maxMemoryPages, 2);
  const x = new Float64Array([2]), v = new Float64Array([5]);
  engine.run(x, v, 0.5);
  assert.equal(x[0], 4.5);
  assert.equal(engine.diagnostics.wasmCalls, 1);
});

test('CLI rejects missing signatures, conflicting modes and invalid budgets before output', t => {
  const { entry, out } = fixture(t);
  const common = [cli, '--entry', entry, '--build-kernel', out];
  for (const extra of [[], ['--parameter-types', 'f64[]'],
    ['--parameter-types', 'f64[],f64[],f64', '--max-memory-pages', 'oops'],
    ['--parameter-types', 'f64[],f64[],f64', '--max-memory-pages', '0'],
    ['--parameter-types', 'f64[],f64[],f64', '--build-app', out],
    ['--parameter-types', 'f64[],f64[],f64', '--output', path.join(out, 'kernel.mjs')],
    ['--parameter-types', 'f64[],f64[],f64', '--package-root', '.']]) {
    const result = spawnSync(process.execPath, [...common, ...extra], { encoding: 'utf8', timeout: 10000 });
    assert.equal(result.status, 1, result.stderr);
    assert.equal(fs.existsSync(out), false);
  }
  const missingMode = spawnSync(process.execPath, [cli, '--entry', entry, '--parameter-types', 'f64[]'], { encoding: 'utf8' });
  assert.equal(missingMode.status, 1);
  assert.match(missingMode.stderr, /require --build-kernel/);
});

test('CLI help remains available without eager compiler or bundler instantiation', () => {
  const result = spawnSync(process.execPath, [cli, '--help'], { encoding: 'utf8', timeout: 10000 });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /--build-app/);
  assert.match(result.stdout, /--build-kernel/);
  assert.match(result.stdout, /--parameter-types/);
});
