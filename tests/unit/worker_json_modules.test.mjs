import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Worker} from 'node:worker_threads';
import test from 'node:test';
import {buildWorkerGraph, moduleWorkerRollupPlugin} from '../../tools/ingest/module_workers.mjs';

function fixture(t, sources, workerType = 'module') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-worker-json-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  function write(name, bytes) {
    const dest = path.join(root, name); fs.mkdirSync(path.dirname(dest), {recursive: true});
    fs.writeFileSync(dest, bytes); return pathToFileURL(dest).href;
  }
  for (const [name, bytes] of Object.entries(sources)) write('source/' + name, bytes);
  const entry = pathToFileURL(path.join(root, 'source/entry.mjs')).href;
  const build = options => buildWorkerGraph(entry, {workerType, ...options});
  function emit(graph) {
    for (const file of graph.files) write('output/' + file.fileName, file.source);
    return pathToFileURL(path.join(root, 'output', graph.entryFile)).href;
  }
  async function run(url) {
    const worker = new Worker(new URL('../fixtures/workers/native_host.mjs', import.meta.url), {
      workerData: {entry: url, workerType}, execArgv: ['--experimental-vm-modules'],
    });
    let timer;
    try {
      return await new Promise((resolve, reject) => {
        timer = setTimeout(() => reject(new Error('Worker did not report a result')), 8000);
        worker.once('message', resolve); worker.once('error', reject);
        worker.once('exit', code => reject(new Error(`Worker exited before a result: ${code}`)));
      });
    } finally { clearTimeout(timer); await worker.terminate(); }
  }
  async function compare(expected) {
    assert.deepEqual(await run(entry), expected, 'original worker');
    const graph = build(), emitted = emit(graph);
    fs.renameSync(path.join(root, 'source'), path.join(root, 'source-offline'));
    assert.deepEqual(await run(emitted), expected, 'emitted worker without source tree');
    return graph;
  }
  return {root, entry, build, emit, run, compare, write};
}
const codeIs = name => error => error.code === 'F3D_WORKER_' + name;
const jsonFiles = graph => graph.files.filter(file => file.fileName.endsWith('.json'));

for (const declaration of [
  `import value from './config.json' with {type:'json'};`,
  `import * as ns from './config.json' with {'type':'json'};`,
  `export {default as config} from './config.json' with {type:'json'};`,
  `export * as config from './config.json' with {type:'json'};`,
  `export * from './config.json' with {type:'json'};`,
  `import './config.json' with {type:'json'};`,
]) test(`packages typed dependency: ${declaration}`, t => {
  const bytes = Buffer.from(' { "answer": 42 }\n');
  const f = fixture(t, {'entry.mjs': declaration, 'config.json': bytes}), graph = f.build();
  assert.equal(graph.moduleCount, 2); assert.equal(graph.assetCount, 0);
  assert.deepEqual(jsonFiles(graph)[0].source, bytes);
  assert.match(graph.files[0].source, /with \{(?:'type'|type):'json'\}/);
  assert.deepEqual(graph, f.build());
});

test('static, lazy, namespace and reexport paths share native JSON identity and mutations', async t => {
  const f = fixture(t, {
    'entry.mjs': `import value from './config.json' with {type:'json'};
      import * as ns from './config.json' with {type:'json'};
      import {config, namespace} from './barrel.mjs';
      value.counter++; const lazy = await import('./config.json',{with:{type:'json'}});
      postMessage([value===ns.default,value===config,value===namespace.default,ns===lazy,lazy.default.counter,
        Object.keys(ns),Object.getPrototypeOf(ns)===null,Object.isExtensible(ns)]);`,
    'barrel.mjs': `export {default as config} from './config.json' with {type:'json'};
      export * as namespace from './config.json' with {type:'json'};`,
    'config.json': '{"counter":41}',
  });
  const graph = await f.compare([true, true, true, true, 42, ['default'], true, false]);
  assert.equal(jsonFiles(graph).length, 1);
});

test('JSON values are host-parsed rather than JavaScript object literals', async t => {
  const f = fixture(t, {
    'entry.mjs': `import config from './config.json' with {type:'json'};
      postMessage([Object.hasOwn(config,'__proto__'),Object.getPrototypeOf(config)===Object.prototype,
        config.__proto__.safe,Object.is(config.zero,-0),config.repeated,config.text]);`,
    'config.json': '\ufeff{"__proto__":{"safe":42},"zero":-0,"repeated":1,"repeated":2,"text":"λ\\u2028end"}\n',
  });
  await f.compare([true, true, 42, true, 2, 'λ\u2028end']);
});
for (const value of [null, false, 42, 'hello', [1, {answer: 42}]])
  test(`native JSON top-level value ${JSON.stringify(value)}`, async t => {
    const f = fixture(t, {'entry.mjs': `import value from './value.json' with {type:'json'}; postMessage(value);`,
      'value.json': JSON.stringify(value)});
    await f.compare(value);
  });

test('conditional JSON imports keep selector and options effects in order, without parsing unselected input', async t => {
  const f = fixture(t, {
    'entry.mjs': `const effects=[];const ns=await import((effects.push('select'),true)?'./good.json':'./bad.json',
      (effects.push('options'),{with:{type:(effects.push('type'),'json')},ignored:effects.push('ignored')}));
      postMessage([ns.default.answer,effects]);`,
    'good.json': '{"answer":42}', 'bad.json': 'not JSON and never evaluated',
  });
  const graph = await f.compare([42, ['select', 'options', 'type', 'ignored']]);
  assert.equal(jsonFiles(graph).length, 2);
});

test('malformed JSON remains a lazy native rejection after synchronous effects', async t => {
  const f = fixture(t, {
    'entry.mjs': `const effects=['before'];const pending=import('./bad.json',{with:{type:'json'}});
      effects.push('after');try{await pending;}catch(error){effects.push(error.name);}postMessage(effects);`,
    'bad.json': '{"answer":',
  });
  await f.compare(['before', 'after', 'SyntaxError']);
});

test('a throwing option expression runs exactly once and prevents JSON evaluation', async t => {
  const f = fixture(t, {
    'entry.mjs': `let effects=0;try{await import('./bad.json', (effects++,(()=>{throw new Error('options');})(),
      {with:{type:'json'}}));}catch(error){postMessage([effects,error.message]);}`,
    'bad.json': 'INVALID',
  });
  await f.compare([1, 'options']);
});

test('query and fragment variants preserve independent mutable JSON module instances', async t => {
  const f = fixture(t, {
    'entry.mjs': `import a from './state.json?one#x' with {type:'json'};
      import b from './state.json?two#x' with {type:'json'};
      import c from './state.json?one#y' with {type:'json'};
      a.value++;const same=await import('./state.json?one#x',{with:{type:'json'}});
      postMessage([a===b,a===c,a===same.default,a.value,b.value,c.value]);`,
    'state.json': '{"value":0}',
  });
  const graph = await f.compare([false, false, true, 1, 0, 0]);
  assert.equal(jsonFiles(graph).length, 3); assert.equal(graph.watchFiles.length, 2);
});

test('equal JSON bytes from different source URLs do not share mutable objects', async t => {
  const f = fixture(t, {
    'entry.mjs': `import a from './a.json' with {type:'json'};import b from './b.json' with {type:'json'};
      a.value=42;postMessage([a===b,a.value,b.value]);`,
    'a.json': '{"value":0}', 'b.json': '{"value":0}',
  });
  await f.compare([false, 42, 0]);
});

test('classic importScripts helpers dynamically import JSON relative to their own script', async t => {
  const f = fixture(t, {
    'entry.mjs': `importScripts('./lib/load.js');load().then(postMessage);`,
    'lib/load.js': `self.load=async()=>{const ns=await import('./settings.json',{with:{type:'json'}});return ns.default.answer;};`,
    'lib/settings.json': '{"answer":42}', 'settings.json': '{"answer":-1}',
  }, 'classic');
  await f.compare(42);
});

test('mixed worker realms independently instantiate the same imported JSON resource', async t => {
  const f = fixture(t, {
    'entry.mjs': `const a=new Worker(new URL('./a.mjs',import.meta.url),{type:'module'});
      const b=new Worker(new URL('./b.js',import.meta.url));
      const values=await Promise.all([new Promise(r=>a.onmessage=e=>r(e.data)),new Promise(r=>b.onmessage=e=>r(e.data))]);
      postMessage(values);a.terminate();b.terminate();`,
    'a.mjs': `import data from './settings.json' with {type:'json'};data.value++;postMessage(data.value);`,
    'b.js': `import('./settings.json',{with:{type:'json'}}).then(ns=>{ns.default.value+=2;postMessage(ns.default.value);});`,
    'settings.json': '{"value":40}',
  });
  const graph = await f.compare([41, 42]);
  assert.equal(jsonFiles(graph).length, 2);
});

test('fetch and JSON import preserve bytes without conflating a response and a module object', async t => {
  const bytes = ' { "answer" : 42 }\n';
  const f = fixture(t, {
    'entry.mjs': `import value from './config.json' with {type:'json'};
      const response=await fetch('./config.json');postMessage([value.answer,await response.text()]);`,
    'config.json': bytes,
  });
  const graph = await f.compare([42, bytes]); assert.equal(graph.assetCount, 1);
});

test('unknown static attributes retain native rejection rather than being discarded', async t => {
  const f = fixture(t, {
    'entry.mjs': `try{await import('./config.json',{with:{type:'json',unsupported:'x'}});}
      catch(error){postMessage([error.name,error.code]);}`,
    'config.json': '{"answer":42}',
  });
  const expected = await f.run(f.entry); assert.equal(expected[1], 'ERR_IMPORT_ATTRIBUTE_UNSUPPORTED');
  assert.deepEqual(await f.run(f.emit(f.build())), expected);
});

test('JSON exports remain default-only, leaving invalid named imports to the native linker', async t => {
  const f = fixture(t, {'entry.mjs': `import {answer} from './config.json' with {type:'json'};postMessage(answer);`,
    'config.json': '{"answer":42}'});
  await assert.rejects(f.run(f.entry), {name:'SyntaxError'});
  await assert.rejects(f.run(f.emit(f.build())), {name:'SyntaxError'});
});

test('JSON payload is never scanned as JavaScript or rewritten as URL-bearing source', t => {
  const bytes = Buffer.from('{"source":"new Worker(./missing.js)","file":"../missing","x":"importScripts(./bad)"}');
  const f = fixture(t, {'entry.mjs':`import value from './config.json' with {type:'json'};`,'config.json':bytes});
  assert.deepEqual(jsonFiles(f.build())[0].source, bytes);
});

test('opaque byte payloads, including invalid UTF-8, are copied without changing native JSON decoding', t => {
  const bytes = Buffer.from([0xef,0xbb,0xbf,0x22,0xff,0x22,0x0a]);
  const f = fixture(t, {'entry.mjs':`import value from './config.json' with {type:'json'};`,'config.json':bytes});
  assert.deepEqual(jsonFiles(f.build())[0].source, bytes);
});

for (const options of ['options', '{...options}', "{get with(){return attributes}}", "{['with']:{type:'json'}}",
  "{__proto__:options}", '{with:attributes}', "{with:{...attributes,type:'json'}}", "{with:{type:kind}}",
  "{with:{type:'json',type:'json'}}", "{with:{type:'css'}}"])
  test(`unproved local import type fails before any file emission: ${options}`, t => {
    const f = fixture(t, {'entry.mjs': `import('./config.json',${options});`, 'config.json':'{}'});
    const emitted = [], plugin = moduleWorkerRollupPlugin();
    assert.throws(() => plugin.transform.call({emitFile(file){emitted.push(file);}},
      `new Worker(new URL('./entry.mjs',import.meta.url),{type:'module'});`, f.entry), codeIs('IMPORT_ATTRIBUTES'));
    assert.equal(emitted.length, 0);
  });

test('remote typed imports stay host-owned even with runtime attributes', t => {
  const source = `import('https://example.invalid/config.json',options);`;
  const f = fixture(t, {'entry.mjs':source}), graph = f.build();
  assert.equal(graph.files[0].source, 'import("https://example.invalid/config.json",options);');
  assert.equal(graph.moduleCount, 1);
});

test('JSON-looking filenames do not silently supply a missing type attribute', t => {
  const f = fixture(t, {'entry.mjs':`import './config.json';`,'config.json':'{"answer":42}'});
  assert.throws(() => f.build(), codeIs('PARSE'));
});

test('module goal is part of identity even when JavaScript and JSON use one physical input', t => {
  const f = fixture(t, {'entry.mjs':`import './value';import value from './value' with {type:'json'};`,'value':'42'});
  const graph = f.build(); assert.equal(graph.moduleCount, 3); assert.equal(graph.watchFiles.length, 2);
  assert.equal(jsonFiles(graph).length, 1); assert.equal(new Set(graph.files.map(file=>file.fileName)).size, 3);
});

test('missing JSON dependencies fail before any graph is emitted', t => {
  const f = fixture(t, {'entry.mjs':`import './missing.json' with {type:'json'};`});
  assert.throws(() => f.build(), codeIs('RESOURCE'));
});

test('JSON modules participate in source/output budgets, watches and content invalidation', t => {
  const f = fixture(t, {'entry.mjs':`import value from './config.json' with {type:'json'};`,'config.json':'{"value":1}'});
  assert.throws(() => f.build({maxModules:1}), codeIs('BUDGET'));
  assert.throws(() => f.build({maxBytes:55}), codeIs('BUDGET'));
  const first = f.build(); assert.equal(first.watchFiles.length, 2);
  assert.equal(first.outputBytes, first.files.reduce((total,file)=>total+Buffer.byteLength(file.source),0));
  f.write('source/config.json','{"value":2}'); assert.notEqual(first.entryFile, f.build().entryFile);
});

test('nested SharedWorker addresses include JSON bytes and not unrelated parent source', t => {
  const f = fixture(t, {'entry.mjs':`new SharedWorker(new URL('./shared.mjs',import.meta.url),{type:'module'});`,
    'shared.mjs':`import value from './config.json' with {type:'json'};`,'config.json':'{"value":1}'});
  const first = f.build();
  f.write('source/entry.mjs',`const unrelated=1;new SharedWorker(new URL('./shared.mjs',import.meta.url),{type:'module'});`);
  const second = f.build(); assert.equal(first.entries[1].fileName,second.entries[1].fileName);
  f.write('source/config.json','{"value":2}');assert.notEqual(second.entries[1].fileName,f.build().entries[1].fileName);
});

test('raw-copy worker baseline fails whereas the typed package executes without its source tree', async t => {
  const code = `import value from './config.json' with {type:'json'};postMessage(value.answer);`;
  const f = fixture(t, {'entry.mjs':code,'config.json':'{"answer":42}'});
  const raw = f.write('raw/entry.mjs',code);
  await assert.rejects(f.run(raw),{code:'ERR_MODULE_NOT_FOUND'});
  await f.compare(42);
});
