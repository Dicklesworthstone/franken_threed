/**
 * Plugin lifecycle/asset tests independent of the Rollup dependency. Chunk
 * descriptors model the public plugin contract; actual linking is covered by
 * numeric_application.test.mjs. Generated modules execute native Wasm here.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { numericKernelRollupPlugin } from './numeric_rollup.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';

const CODE = `function step(a, dt) { for (let i = 0; i < a.length; i++) a[i] += dt; }
export function tick(a, dt) { return step(a, dt); }
export { step };`;
const sha256 = value => crypto.createHash('sha256').update(value).digest('hex');
function lifecycle(options = {}) {
  const plugin = numericKernelRollupPlugin(options);
  const bundle = {};
  const context = { emitFile(asset) { bundle[asset.fileName] = { ...asset }; return asset.fileName; } };
  plugin.renderStart();
  function chunk(code = CODE, fileName = 'app.mjs', output = { format: 'es', sourcemap: false }) {
    const record = {
      type: 'chunk', name: 'app', fileName, preliminaryFileName: fileName,
      moduleIds: ['file:///source/update.mjs', 'file:///source/app.mjs'],
      modules: {}, imports: [], importedBindings: {}, code,
    };
    const result = plugin.renderChunk.call(context, code, record, output);
    if (result) record.code = result.code;
    bundle[fileName] = record;
    return { record, result };
  }
  function finish() { plugin.generateBundle.call(context, {}, bundle); return plugin.api.getReport(); }
  return { plugin, bundle, context, chunk, finish };
}
async function materialize(bundle, entry) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-plugin-assets-'));
  for (const [name, item] of Object.entries(bundle)) {
    const dest = path.join(dir, name); fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, item.type === 'chunk' ? item.code : item.source);
  }
  return { dir, module: await import(pathToFileURL(path.join(dir, entry))) };
}

test('lifecycle emits self-contained hashed runtime assets and executable native Wasm', async t => {
  const native = WebAssembly.Instance; let calls = 0;
  t.after(() => { WebAssembly.Instance = native; });
  WebAssembly.Instance = function (...args) {
    const instance = Reflect.construct(native, args);
    return { exports: { memory: instance.exports.memory, run(...xs) { calls++; return instance.exports.run(...xs); } } };
  };
  const run = lifecycle(); const { record } = run.chunk(); const report = run.finish();
  assert.equal(report.compiledKernels, 1); assert.equal(report.rewrittenCalls, 1);
  assert.equal(report.accelerated, false); assert.equal(report.runtimeAssets.length, 2);
  for (const asset of report.runtimeAssets) {
    assert.equal(sha256(run.bundle[asset.fileName].source), asset.sha256);
    assert.ok(asset.fileName.includes(asset.sha256.slice(0, 20)));
  }
  assert.equal(record.imports.length, 1);
  assert.deepEqual(record.importedBindings[record.imports[0]], ['createNumericDispatch', 'dispatchNumericCall']);
  const { module } = await materialize(run.bundle, 'app.mjs');
  const actual = new Float64Array([1, 2]), expected = actual.slice();
  for (let i = 0; i < 300; i++) {
    module.tick(actual, 1 / 60); module.step(expected, 1 / 60); assert.deepEqual(actual, expected);
  }
  assert.equal(calls, 300);
  assert.deepEqual(JSON.parse(run.bundle[report.reportFile].source), report);
});

test('nested and dynamically imported chunks share assets and resolve relative runtime imports', async () => {
  const run = lifecycle(); run.chunk(CODE, 'chunks/deep/update.mjs');
  run.chunk(`export const load = () => import('./chunks/deep/update.mjs');`, 'entry.mjs');
  run.chunk(CODE, 'other.mjs');
  const report = run.finish(); assert.equal(report.compiledKernels, 2); assert.equal(report.runtimeAssets.length, 2);
  assert.match(run.bundle['chunks/deep/update.mjs'].code, /from "\.\.\/\.\.\/f3d-runtime\//);
  const { dir, module } = await materialize(run.bundle, 'entry.mjs');
  const update = await module.load(); const a = new Float64Array([1]); update.tick(a, 3); assert.equal(a[0], 4);
  const relocated = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-relocated-'));
  fs.cpSync(dir, relocated, { recursive: true });
  const moved = await import(pathToFileURL(path.join(relocated, 'entry.mjs')));
  (await moved.load()).tick(a, 2); assert.equal(a[0], 6);
});

test('final reports follow Rollup filename hash substitutions', () => {
  const run = lifecycle(); const preliminary = 'app-!~{001}~.mjs';
  const { record } = run.chunk(CODE, preliminary);
  delete run.bundle[preliminary]; record.fileName = 'app-finalhash.mjs'; run.bundle[record.fileName] = record;
  const report = run.finish(); assert.equal(report.units[0].fileName, record.fileName);
  assert.equal(report.units[0].inputSha256, sha256(CODE));
  assert.equal(report.units[0].coordinateSpace, 'renderChunk-before-specialization-and-hash-substitution');
});

test('refused and uncalled kernels emit no runtime and keep code untouched', () => {
  const run = lifecycle(); const code = CODE.replace('a[i] += dt', 'a[i] = Math.sin(a[i])');
  assert.equal(run.chunk(code).result, null);
  const report = run.finish(); assert.equal(report.compiledKernels, 0); assert.deepEqual(report.runtimeAssets, []);
  assert.equal(run.bundle['app.mjs'].code, code);
  assert.equal(report.units[0].candidates[0].reason, 'KERNEL_NOT_CLOSED');
});

test('policy-blocked Wasm in a finished generated module keeps original update semantics', async t => {
  const run = lifecycle(); run.chunk(); run.finish();
  const native = globalThis.WebAssembly; t.after(() => { globalThis.WebAssembly = native; });
  globalThis.WebAssembly = undefined;
  const { module } = await materialize(run.bundle, 'app.mjs');
  const a = new Float64Array([1]); module.tick(a, 2); module.tick(a, 2); assert.equal(a[0], 5);
});

test('a new output generation resets kernel/asset/report state', () => {
  const run = lifecycle(); run.chunk(); assert.equal(run.finish().compiledKernels, 1);
  run.plugin.renderStart(); for (const key of Object.keys(run.bundle)) delete run.bundle[key];
  run.chunk('export const empty = true;');
  const report = run.finish(); assert.equal(report.compiledKernels, 0); assert.deepEqual(report.runtimeAssets, []);
});

test('source map and non-ES contracts are retained rather than silently changed', () => {
  for (const output of [{ format: 'cjs' }, { format: 'es', sourcemap: true }]) {
    const run = lifecycle(); assert.equal(run.chunk(CODE, 'app.mjs', output).result, null);
    const report = run.finish(); assert.equal(report.compiledKernels, 0); assert.deepEqual(report.runtimeAssets, []);
    assert.equal(report.units[0].refusal.code, output.format === 'cjs' ? 'NON_ES_OUTPUT' : 'SOURCE_MAP_SPECIALIZATION_UNAVAILABLE');
  }
});

test('runtime and report collisions fail before silently shipping incompatible assets', () => {
  let run = lifecycle(); run.chunk(); const fileName = Object.keys(run.bundle).find(name => name.includes('numeric-kernel'));
  run.bundle[fileName].source = 'wrong runtime'; assert.throws(() => run.finish(), /runtime asset collision/);
  run = lifecycle(); run.chunk(); run.bundle['f3d-numeric-specialization.json'] = { type: 'asset', source: '{}' };
  assert.throws(() => run.finish(), /report asset collision/);
});

test('runtime resolution is lazy and configuration mistakes are rejected', () => {
  let calls = 0;
  const runtimeModule = () => { calls++; return './runtime.mjs'; };
  assert.equal(specializeNumericModule('export const empty = true;', { runtimeModule }).changed, false);
  assert.equal(calls, 0);
  assert.equal(specializeNumericModule(CODE, { runtimeModule }).changed, true); assert.equal(calls, 1);
  for (const invalid of [null, true, 'yes', [], { maxKernels: 0 }, { maxMemoryPages: NaN }, { misspelled: true }]) {
    assert.throws(() => numericKernelRollupPlugin(invalid));
  }
  assert.throws(() => specializeNumericModule(CODE, { runtimeModule: () => null }), /specifier/);
});
