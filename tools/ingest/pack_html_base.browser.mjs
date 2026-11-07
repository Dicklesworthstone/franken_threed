/** Local document bases through source, emitted and offline standalone pages. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const browser = await chromium.launch({headless: true,
  executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking']});
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-packed-base-'));
const mime = {'.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.css': 'text/css', '.svg': 'image/svg+xml'};
try {
  for (const [index, href] of ['../assets/', '../assets/current.html?edition=2&amp;view=all#selected', '/assets/'].entries()) {
    const root = path.join(directory, String(index)), source = path.join(root, 'source'), built = path.join(root, 'built');
    const bases = `<base target="_self"><base href="${href}"><base href="https://ignored.invalid/">`;
    const files = {
      'pages/index.html': `<!doctype html><html><head>${bases}
        <script type="importmap">{"imports":{"shared":"./shared.mjs"}}</script>
        <link rel="stylesheet" href="./main.css"><link rel="modulepreload" href="../scripts/main.mjs">
        <style>#title{background-image:url('./pixel.svg')}</style></head><body>
        <h1 id="title">Document base</h1><img id="image" src="./pixel.svg">
        <a id="next" href="./next.html?x=1#go">Next</a>
        <script src="./classic.js"></script><script type="module" src="../scripts/main.mjs"></script>
        <script type="module">import {state} from 'shared'; import {value} from './inline.mjs';
          globalThis.inlineResult={state,value,bytes:await (await fetch(new URL('./bytes.bin',import.meta.url))).text()};</script>
        </body></html>`,
      'assets/shared.mjs': 'export const state={value:42};',
      'assets/inline.mjs': 'export const value=17;',
      'assets/bytes.bin': 'module-relative bytes',
      'assets/settings.json': '{"value":"document-base fetch"}',
      'assets/classic.js': `globalThis.classicResult={base:document.baseURI};
        globalThis.workerResult=new Promise((resolve,reject)=>{
          const worker=new Worker('./worker.mjs',{type:'module'});
          worker.onmessage=event=>{worker.terminate();resolve(event.data)};worker.onerror=reject;
        });`,
      'assets/worker.mjs': `import {value} from './worker-value.mjs';self.postMessage(value);`,
      'assets/worker-value.mjs': `export const value='native worker';`,
      'scripts/main.mjs': `import {state} from 'shared';
        const settings=await (await fetch('./settings.json')).json();
        const worker=await globalThis.workerResult;
        await document.querySelector('#image').decode();
        while(!globalThis.inlineResult)await new Promise(resolve=>setTimeout(resolve,0));
        globalThis.result={base:document.baseURI,classic:globalThis.classicResult,
          bases:[...document.querySelectorAll('base')].map(node=>[node.getAttribute('href'),node.getAttribute('target')]),
          next:document.querySelector('#next').href,settings,worker,
          inline:globalThis.inlineResult.value,bytes:globalThis.inlineResult.bytes,
          sameModule:state===globalThis.inlineResult.state,color:getComputedStyle(document.querySelector('#title')).color,
          imageWidth:document.querySelector('#image').naturalWidth};`,
      'assets/main.css': `@import './theme.css';body{background-image:url('./pixel.svg')}`,
      'assets/theme.css': '#title{color:rgb(12,34,56)}',
      'assets/pixel.svg': '<svg xmlns="http://www.w3.org/2000/svg" width="3" height="2"><rect width="3" height="2" fill="red"/></svg>',
    };
    for (const [name, contents] of Object.entries(files)) {
      const file = path.join(source, name);fs.mkdirSync(path.dirname(file), {recursive: true});fs.writeFileSync(file, contents);
    }
    const application = await buildApplication(path.join(source, 'pages/index.html'), built, {documentRoot: source});
    const packed = path.join(root, 'standalone.html');
    packHtml(path.join(built, application.htmlFile), packed, {rootDir: built, documentRoot: built});
    let serving = source, standalone = false;
    const subresources = [];
    const server = http.createServer((request, response) => {
      const url = new URL(request.url, 'http://localhost');
      if (url.pathname === '/favicon.ico') {response.writeHead(204).end();return;}
      if (standalone) {
        if (url.pathname === '/pages/index.html') {response.setHeader('Content-Type', 'text/html');response.end(fs.readFileSync(packed));return;}
        subresources.push(url.pathname);response.writeHead(410).end();return;
      }
      const filename = path.resolve(serving, '.' + decodeURIComponent(url.pathname));
      if (!filename.startsWith(serving + path.sep) || !fs.existsSync(filename)) {response.writeHead(404).end();return;}
      response.setHeader('Content-Type', mime[path.extname(filename)] ?? 'application/octet-stream');
      response.end(fs.readFileSync(filename));
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    async function capture() {
      const context = await browser.newContext({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'});
      try {
        const page = await context.newPage(), errors = [], external = [];
        page.on('pageerror', error => errors.push(String(error)));
        page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
        await context.route('**/*', route => {
          const url = route.request().url();
          if (url.startsWith(origin + '/') || /^(data|blob):/.test(url)) return route.continue();
          external.push(url);return route.abort();
        });
        await page.goto(origin + '/pages/index.html?navigation=retained#start');
        await page.waitForFunction(() => globalThis.result, undefined, {timeout: 15000});
        const result = await page.evaluate(() => globalThis.result);
        assert.deepEqual(errors, []);assert.deepEqual(external, []);
        assert.equal(result.color, 'rgb(12, 34, 56)');assert.equal(result.imageWidth, 3);
        assert.equal(result.sameModule, true);assert.equal(result.worker, 'native worker');
        assert.deepEqual(result.settings, {value: 'document-base fetch'});
        return result;
      } finally {await context.close();}
    }
    try {
      const expected = await capture();
      serving = built;assert.deepEqual(await capture(), expected, 'emitted application preserves source behavior');
      fs.renameSync(source, source + '-unavailable');fs.renameSync(built, built + '-unavailable');standalone = true;
      assert.deepEqual(await capture(), expected, 'standalone application preserves source base and resource behavior');
      assert.deepEqual(subresources, [], 'the packed document makes no network subresource requests');
      console.log(JSON.stringify({case: href, status: 'pass', browser: browser.version(),
        sourceEmittedStandalone: true, networkSubresources: 0, nativeWorker: true}));
    } finally {await new Promise(resolve => server.close(resolve));}
  }
} finally {await browser.close();}
