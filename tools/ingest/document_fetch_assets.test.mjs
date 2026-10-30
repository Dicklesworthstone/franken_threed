import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {Worker} from 'node:worker_threads';
import {analyzeDocumentFetch, collectDocumentFetchAssets} from './document_fetch_assets.mjs';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const data = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const decode = url => Buffer.from(url.slice(url.indexOf(',') + 1).split('#')[0], 'base64').toString();
function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-document-fetch-'));
  const source = path.join(root, 'source'), out = path.join(root, 'out');
  for (const [name, bytes] of Object.entries({'package.json': '{"type":"module"}', ...files})) {
    const filename = path.join(source, name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, bytes, {flag: 'wx'});
  }
  return {root, source, out, entry: path.join(source, 'index.html'), packed: path.join(root, 'packed.html')};
}
function application(html, directory) {
  const imports = JSON.parse(/<script type="importmap">(.*?)<\/script>/s.exec(html)?.[1] ?? '{}').imports;
  const entries = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
  const module = entries.find(([, attrs]) => /type="module"/.test(attrs));
  const url = /src="([^"]+)"/.exec(module[1])[1];
  const scripts = entries.filter(([, attrs]) => !/type="(?:module|importmap)"/.test(attrs)).map(([, attrs, body]) => {
    const src = /src="([^"]+)"/.exec(attrs)?.[1];
    return !src ? body : src.startsWith('data:') ? decode(src) : fs.readFileSync(path.join(directory, src), 'utf8');
  });
  return {imports, scripts, url: url.startsWith('data:') ? url : pathToFileURL(path.join(directory, url)).href};
}
const runner = new URL(data(`
  import vm from 'node:vm';
  import {registerHooks} from 'node:module';
  import {parentPort,workerData} from 'node:worker_threads';
  registerHooks({resolve(specifier,context,next){return next(workerData.imports?.[specifier]??specifier,context);}});
  // Only the networked reference needs Node's missing document URL base.
  // Packed execution uses the actual native fetch with no wrapper or replies.
  if(workerData.base){const nativeFetch=fetch;globalThis.fetch=(input,options)=>nativeFetch(
    typeof input==='string'?new URL(input,workerData.base):input,options);}
  for(const source of workerData.scripts)vm.runInThisContext(source);
  const app=await import(workerData.url);
  parentPort.postMessage(await app.run());
`));
async function execute(input) {
  const thread = new Worker(runner, {workerData: input});
  let timer;
  try {
    return await new Promise((resolve, reject) => {
      let result, received = false;
      timer = setTimeout(() => reject(new Error('Document fetch execution timed out')), 15000);
      thread.once('message', value => { result = value; received = true; });
      thread.once('error', reject);
      thread.once('exit', code => code === 0 && received ? resolve(result) : reject(new Error('Worker exit: ' + code)));
    });
  } finally { clearTimeout(timer); await thread.terminate(); }
}
async function host(directory) {
  const types = {'.json': 'application/json', '.wasm': 'application/wasm', '.wav': 'audio/wav', '.css': 'text/css'};
  const server = http.createServer((req, res) => {
    const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    try {
      const bytes = fs.readFileSync(path.join(directory, name));
      res.writeHead(200, {'Content-Type': types[path.extname(name)] ?? 'text/javascript'}); res.end(bytes);
    } catch { res.writeHead(404); res.end('missing'); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return {base: `http://127.0.0.1:${server.address().port}/`, close: () => new Promise(resolve => server.close(resolve))};
}

test('finite GET discovery respects lexical fetch bindings, host writes and opaque Request options', () => {
  const found = source => analyzeDocumentFetch(source).sites.map(({candidates, admitted}) => ({candidates, admitted}));
  assert.deepEqual(found(`const name=choice?'left':'right';const uri='data/'+name+'.json';
    fetch((effect(),uri),(effect(),{method:(effect(),'get'),signal:controller.signal,credentials:'include',integrity:'hash'}));`),
  [{candidates: ['data/left.json', 'data/right.json'], admitted: true}]);
  assert.deepEqual(found(`function local(fetch){fetch('missing.json')}
    function defaults(value=fetch('default.json')){var fetch;}
    fetch('real.json');`), [
    {candidates: ['default.json'], admitted: true}, {candidates: ['real.json'], admitted: true},
  ]);
  for (const source of [
    `import {fetch} from './shim.mjs'; fetch('missing.json');`,
    `import {fetch as request} from './shim.mjs'; request('missing.json');`,
    `const request=fetch;request('missing.json');`,
    `fetch(new Request('config.json'));`, `fetch(runtimeURL);`,
  ]) assert.deepEqual(found(source), []);
  for (const source of [
    `globalThis.fetch=shim;fetch('config.json');`,
    `Object.defineProperty(globalThis,'fetch',{value:shim});fetch('config.json');`,
    `Object.prototype.method='POST';fetch('config.json',{});`,
    `fetch('config.json',{method:'POST'});`, `fetch('config.json',options);`,
    `fetch('config.json',{get method(){return 'GET'}});`, `fetch('config.json',{...options});`,
  ]) assert.equal(found(source)[0].admitted, false, source);
  const {sites} = analyzeDocumentFetch(`fetch('remote.json')`);
  assert.equal(collectDocumentFetchAssets(sites, {
    documentUrl: import.meta.url, documentBaseUrl: 'https://example.invalid/app/',
  }).size, 0, 'a relative fetch against a remote document base stays network-owned');
});

test('native packed fetch preserves bodies, raw resources, selectors, aborts, integrity and streaming Wasm offline', async () => {
  const json = '{"value":1,"value":42,"zero":-0}';
  const integrity = 'sha256-' + createHash('sha256').update(json).digest('base64');
  const wav = Buffer.alloc(52);
  wav.write('RIFF'); wav.writeUInt32LE(44, 4); wav.write('WAVEfmt ', 8); wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22); wav.writeUInt32LE(8000, 24);
  wav.writeUInt32LE(16000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(8, 40); wav.writeInt16LE(8192, 44); wav.writeInt16LE(-8192, 46);
  const f = fixture({
    'index.html': '<script src="./classic.js"></script><script>globalThis.inlineLoad=()=>fetch("raw/body.css").then(r=>r.text());</script><script type="module" src="./src/main.mjs"></script>',
    'src/main.mjs': `import {load} from './nested/load.mjs'; import {fetch as projectFetch} from './shim.mjs';
      export async function run(){const log=[];
        const left=await load(true,log),right=await load(false,log);
        const wasm=await WebAssembly.instantiateStreaming(fetch('kernel.wasm'));
        const audio=await fetch('sounds/tone.wav');const audioType=audio.headers.get('content-type');
        const audioBytes=[...new Uint8Array(await audio.arrayBuffer())];
        const reason='source-abort',controller=new AbortController();controller.abort(reason);
        let aborted=false,badBody=false,badIntegrity=false;
        try{await fetch('data/left.json',{signal:controller.signal})}catch(e){aborted=e===reason}
        try{await fetch('data/left.json',{method:'GET',body:'invalid'})}catch(e){badBody=e instanceof TypeError}
        try{await fetch('data/left.json',{integrity:'sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA='})}catch(e){badIntegrity=e instanceof TypeError}
        return {left,right,log,wasm:wasm.instance.exports.run(),audioType,audioBytes,aborted,badBody,badIntegrity,
          raw:[await globalThis.classicLoad(),await globalThis.inlineLoad(),await(await fetch('raw/model.gltf')).text()],
          custom:projectFetch('not-an-asset')};}
      globalThis.__f3dFetchRun=run;`,
    'src/nested/load.mjs': `export async function load(left,log){
      const name=left?'left':'right',uri=\`data/\${name}.json?version=1#selected\`;
      const controller=new AbortController();const r=await globalThis.fetch((log.push('url'),uri),
        (log.push('options'),{method:(log.push('method'),'get'),signal:(log.push('signal'),controller.signal),credentials:'omit',integrity:${JSON.stringify(integrity)}}));
      const copy=r.clone(),before=r.bodyUsed,reader=r.body.getReader(),parts=[];
      for(;;){const {done,value}=await reader.read();if(done)break;parts.push(...value)}
      const parsed=await copy.json();return {uri,before,after:r.bodyUsed,value:parsed.value,zero:Object.is(parsed.zero,-0),text:new TextDecoder().decode(new Uint8Array(parts))};}`,
    'src/shim.mjs': `export const fetch=value=>'custom:'+value;`,
    'classic.js': `globalThis.classicLoad=()=>fetch('raw/body.js').then(r=>r.text());`,
    'data/left.json': json, 'data/right.json': json,
    'kernel.wasm': Buffer.from('0061736d010000000105016000017f030201000707010372756e00000a06010400412a0b', 'hex'),
    'sounds/tone.wav': wav,
    'raw/body.js': 'this is raw response data, not JavaScript;',
    'raw/body.css': '@import "not-a-dependency.css"; body{background:url(missing.png)}',
    'raw/model.gltf': '{"asset":{"version":"2.0"},"buffers":[{"uri":"not-a-dependency.bin"}]}',
  });
  const build = await buildApplication(f.entry, f.out);
  for (const name of ['data/left.json', 'data/right.json', 'kernel.wasm', 'sounds/tone.wav', 'raw/body.js', 'raw/body.css', 'raw/model.gltf'])
    assert.deepEqual(fs.readFileSync(path.join(f.out, name)), fs.readFileSync(path.join(f.source, name)));
  const referenceHost = await host(f.source), builtHost = await host(f.out);
  let expected;
  try {
    expected = await execute({...application(fs.readFileSync(f.entry, 'utf8'), f.source), base: referenceHost.base});
    assert.deepEqual(await execute({...application(fs.readFileSync(path.join(f.out, build.htmlFile), 'utf8'), f.out), base: builtHost.base}), expected);
  } finally { await Promise.all([referenceHost.close(), builtHost.close()]); }
  const packed = packHtml(path.join(f.out, build.htmlFile), f.packed);
  const emitted = application(fs.readFileSync(f.packed, 'utf8'), f.out);
  fs.renameSync(f.source, path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'build-unavailable'));
  const actual = await execute(emitted);
  assert.deepEqual(actual, expected);
  assert.equal(actual.wasm, 42); assert.equal(actual.audioType, 'audio/wav');
  assert.ok(actual.aborted && actual.badBody && actual.badIntegrity);
  assert.deepEqual(actual.log, ['url', 'options', 'method', 'signal', 'url', 'options', 'method', 'signal']);
  assert.equal(actual.left.value, 42); assert.equal(actual.right.value, 42);
  assert.equal(actual.left.uri, 'data/left.json?version=1#selected');
  assert.ok(packed.outputBytes < 100000);
});

test('document fetch bases survive project transforms and retained scripts; response data stays raw', async () => {
  const f = fixture({
    'index.html': '<base href="./assets/"><script src="../classic.js"></script><script type="module" src="../src/main.mjs"></script>',
    'src/main.mjs': `export const load=()=>fetch('__FRONTEND_RESOURCE__');`,
    'classic.js': `globalThis.loadRaw=()=>fetch('body.js');`,
    'assets/settings.json': '{"base":"document"}',
    'assets/body.js': 'fetched source bytes must not run through the frontend',
  });
  const seen = [];
  await buildApplication(f.entry, f.out, {transformModule(source, {url}) {
    seen.push(url);return source.includes('__FRONTEND_RESOURCE__') ? source.replace('__FRONTEND_RESOURCE__', 'settings.json?mode=1#v') : null;
  }});
  for (const name of ['assets/settings.json', 'assets/body.js'])
    assert.deepEqual(fs.readFileSync(path.join(f.out, name)), fs.readFileSync(path.join(f.source, name)));
  assert.ok(!seen.some(url => url.endsWith('/assets/body.js')));
});

test('packing refuses opaque finite requests and cross-module fetch replacement before publishing', () => {
  for (const source of [
    `fetch('config.json',{get method(){return 'GET'}});`,
    `fetch('config.json',{method:'POST'});`, `fetch('https://example.invalid/config.json');`,
    `import './shim.mjs';fetch('config.json');`,
  ]) {
    const f = fixture({'index.html': '<script type="module" src="main.mjs"></script>',
      'main.mjs': source, 'shim.mjs': 'globalThis.fetch=()=>Promise.resolve();', 'config.json': '{}'});
    assert.throws(() => packHtml(f.entry, f.packed), {code: 'FETCH_URL'});
    assert.equal(fs.existsSync(f.packed), false);
  }
});
