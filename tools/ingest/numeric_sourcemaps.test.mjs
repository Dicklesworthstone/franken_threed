/** Original-source mappings and real native execution in mapped application builds. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {SourceMap} from 'node:module';
import {test} from 'node:test';
import {pathToFileURL} from 'node:url';
import MagicString from 'magic-string';
import {rollup} from 'rollup';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {numericKernelRollupPlugin} from './numeric_rollup.mjs';

function position(source, text) {
  const offset = source.indexOf(text);
  assert.notEqual(offset, -1, text);
  const lines = source.slice(0, offset).split('\n');
  return [lines.length - 1, lines.at(-1).length];
}
function mapsTo(result, source, text, fileName) {
  const entry = new SourceMap(JSON.parse(result.map.toString()))
    .findEntry(...position(result.code, text));
  assert.equal(entry.originalSource.split('/').at(-1), fileName);
  assert.deepEqual([entry.originalLine, entry.originalColumn], position(source, text), text);
}

test('standalone maps preserve hashbangs, directives, original functions and later statements', () => {
  const source = `#!/usr/bin/env node
"use strict";
function step(a, dt) {
  for (let i = 0; i < a.length; i++) a[i] += dt;
}
export function tick(a, dt) { step(a, dt); return a[0]; }
`;
  const options = {sourceName: 'source.mjs'};
  const result = specializeNumericModule(source, {...options, sourcemap: true});
  assert.equal(result.report.compiledKernels, 1);
  assert.equal(result.code, specializeNumericModule(source, options).code);
  assert.ok(result.code.startsWith('#!/usr/bin/env node\n"use strict";'));
  for (const text of ['function step', 'a[i] += dt', 'return a[0]'])
    mapsTo(result, source, text, 'source.mjs');
  assert.deepEqual(result.map.sourcesContent, [source]);
  assert.throws(() => specializeNumericModule(source, {sourcemap: 'yes'}), /sourcemap/);
});

test('retained array and scalar-publication loop fallbacks keep exact original coordinates', () => {
  const source = `export function tick(a, dt, events) {
  events.push("before");
  for (let i = 0; i < a.length; i++) {
    a[i] += dt;
  }
  events.push("middle");
  let total = 0;
  for (let j = 0; j < a.length; j++) {
    total += a[j];
  }
  events.push(total);
  return total;
}`;
  const result = specializeNumericModule(source, {loopIslands: true, sourcemap: true, sourceName: 'loops.mjs'});
  assert.equal(result.report.loopIslands.compiledKernels, 2);
  assert.ok(result.report.loopIslands.candidates.some(item => item.scalarOutputs?.includes('total')));
  for (const text of ['a[i] += dt', 'total += a[j]', 'events.push("middle")', 'return total'])
    mapsTo(result, source, text, 'loops.mjs');
  // Generated registrations/guards have no fictitious source statement.
  const map = new SourceMap(JSON.parse(result.map.toString()));
  assert.equal(map.findEntry(...position(result.code, 'import {')).originalSource, undefined);
});

test('adjacent independently guarded loops preserve their execution and source-map boundaries', async t => {
  const source = `let gain=1;const scale=x=>x*gain;export const out=new Float64Array([2]);` +
    `for(let i=0;i<1;i++)gain+=1;for(let i=0;i<out.length;i++)out[i]=scale(out[i]);export{gain};`;
  const native = globalThis.WebAssembly;
  t.after(() => { globalThis.WebAssembly = native; });
  for (const available of [true, false]) {
    globalThis.WebAssembly = available ? native : undefined;
    const result = specializeNumericModule(source, {loopIslands: true, sourcemap: true,
      sourceName: 'adjacent.mjs', runtimeModule: new URL('./numeric_dispatch.mjs', import.meta.url).href});
    assert.equal(result.report.loopIslands.compiledKernels, 2);
    const url = code => 'data:text/javascript;base64,' + Buffer.from(code).toString('base64') + '#' + available;
    const actual = await import(url(result.code)), expected = await import(url(source));
    assert.deepEqual([...actual.out], [...expected.out]);
    assert.equal(actual.gain, expected.gain);
    for (const text of ['gain+=1', 'out[i]=scale(out[i])', 'export{gain}'])
      mapsTo(result, source, text, 'adjacent.mjs');
  }
});

for (const sourcemap of [true, 'inline', 'hidden']) {
  test(`Rollup ${sourcemap === true ? 'external' : sourcemap} maps compose through split native chunks`, async t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-numeric-maps-'));
    const sourceDir = path.join(root, 'src'), outDir = path.join(root, 'dist');
    fs.mkdirSync(sourceDir);
    const producer = `export function step(a, delta) {
  for (let i = 0; i < a.length; i++) a[i] += delta;
}`;
    const consumer = `import {step} from './producer.mjs';
export function tick(a, delta) { step(a, delta); return a[0]; }`;
    const entryPath = path.join(sourceDir, 'entry.mjs');
    fs.writeFileSync(entryPath, consumer);
    fs.writeFileSync(path.join(sourceDir, 'producer.mjs'), producer);
    const native = WebAssembly.Instance;
    let calls = 0;
    t.after(() => { WebAssembly.Instance = native; });
    WebAssembly.Instance = function (...args) {
      const instance = Reflect.construct(native, args);
      return {exports: {memory: instance.exports.memory, run(...values) {
        calls++; return instance.exports.run(...values);
      }}};
    };
    const plugin = numericKernelRollupPlugin();
    const bundle = await rollup({input: entryPath, plugins: [{
      name: 'mapped-input',
      transform(code) {
        const text = new MagicString(code).prepend('// upstream transform\n');
        return {code: text.toString(), map: text.generateMap({hires: true})};
      },
    }, plugin]});
    try {
      const result = await bundle.write({dir: outDir, format: 'es', sourcemap,
        preserveModules: true, entryFileNames: 'chunks/[name]-[hash].mjs', plugins: [{
          name: 'mapped-output',
          renderChunk(code) {
            const text = new MagicString(code).prepend('// downstream output\n');
            return {code: text.toString(), map: text.generateMap({hires: true})};
          },
        }]});
      const entry = result.output.find(item => item.type === 'chunk' && item.facadeModuleId === entryPath);
      const update = result.output.find(item => item.type === 'chunk' && item.facadeModuleId?.endsWith('producer.mjs'));
      mapsTo(entry, consumer, 'return a[0]', 'entry.mjs');
      mapsTo(update, producer, 'a[i] += delta', 'producer.mjs');
      assert.deepEqual(entry.map.sourcesContent, [consumer]);
      assert.equal(plugin.api.getReport().compiledKernels, 1);
      assert.equal(plugin.api.getReport().runtimeAssets.length, 2);
      assert.equal(entry.code.includes('sourceMappingURL='), sourcemap !== 'hidden');
      assert.equal(result.output.some(item => item.fileName === entry.fileName + '.map'), sourcemap !== 'inline');
      const app = await import(pathToFileURL(path.join(outDir, entry.fileName)));
      const data = new Float64Array([1, 2]);
      assert.equal(app.tick(data, 3), 4);
      assert.deepEqual([...data], [4, 5]);
      assert.equal(calls, 1, 'source maps must not silently disable native execution');
    } finally { await bundle.close(); }
  });
}
