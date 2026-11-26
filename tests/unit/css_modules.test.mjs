import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import test from 'node:test';
import {analyzeModuleAst} from '../../tools/ingest/ast_analyzer.mjs';
import {buildModuleGraph} from '../../tools/ingest/module_graph.mjs';
import {buildApplication} from '../../tools/ingest/build_application.mjs';
import {packHtml} from '../../tools/ingest/pack_html.mjs';
import {cssResources} from '../../tools/ingest/css_resources.mjs';

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-css-modules-'));
  for (const [name, content] of Object.entries(files)) {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, content);
  }
  const out = path.join(root, 'out'), url = name => pathToFileURL(path.join(root, name)).href;
  return {root, out, url,
    graph: (entry = 'app.mjs', options = {}) => buildModuleGraph(path.join(root, entry), options),
    build: (entry = 'app.mjs', options = {}) => buildApplication(path.join(root, entry), out, options),
    pack(entry = path.join(root, 'index.html'), options = {}) {
      const output = path.join(root, 'packed.html'), report = packHtml(entry, output, options);
      const html = fs.readFileSync(output, 'utf8');
      const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
      return {report, html, imports};
    }};
}
const sheets = built => built.emittedFiles.filter(name => /(?:^|\/)f3d-css-[0-9a-f]+\.css$/.test(name));
const decode = url => Buffer.from(url.slice(url.indexOf(',') + 1).split('#')[0], 'base64').toString('utf8');
const sheetUrls = packed => Object.values(packed.imports).filter(url => url.startsWith('data:text/css;'));

test('graph follows static, reexported and finite lazy CSS imports as native data', async () => {
  const f = fixture({
    'app.mjs': `import theme from './theme.css' with {type:'css'};
      export {theme}; export {default as again} from './theme.css' with {type:'css'};
      export * as styles from './theme.css' with {'type':'css'};
      export const load=flag=>import(flag?'./theme.css':'./lazy.css',{with:{type:'css'}});`,
    'theme.css': 'body { color: red; }', 'lazy.css': '.incomplete { color:',
  });
  const graph = await f.graph(), entry = graph.modules[f.url('app.mjs')];
  const nodes = Object.values(graph.modules).filter(node => node.module_type === 'css');
  assert.equal(nodes.length, 2);
  assert.equal(entry.static_imports.length, 3);
  assert.equal(new Set(entry.static_imports.map(edge => edge.resolved_id)).size, 1);
  assert.equal(entry.static_imports[0].module_type, 'css');
  assert.equal(entry.dynamic_imports[0].claims_closure, true);
  for (const node of nodes) {
    assert.deepEqual(node.static_imports, []);
    assert.deepEqual(node.renderer_construction_sites, []);
    assert.equal(node.static_exports[0].type, 'default');
  }
});

test('CSS, JSON and JavaScript goals keep distinct graph identities at one source URL', async () => {
  const f = fixture({'app.mjs': `import './value.js';
    import data from './value.js' with {type:'json'};
    import sheet from './value.js' with {type:'css'}; export {data,sheet};`, 'value.js': '42'});
  const graph = await f.graph(), nodes = Object.values(graph.modules).filter(node => node.source_url === f.url('value.js'));
  assert.deepEqual(nodes.map(node => node.module_type).sort(), ['css', 'javascript', 'json']);
  assert.equal(new Set(nodes.map(node => node.id)).size, 3);
});

test('CSS sources never pass through the JavaScript frontend', async () => {
  const f = fixture({'app.mjs': `import sheet from './theme.css' with {type:'css'}; export {sheet};`,
    'theme.css': 'body { color: red; }'});
  const seen = [];
  const transformModule = (source, {url}) => { seen.push(url); return source; };
  await f.graph('app.mjs', {transformModule});
  await f.build('app.mjs', {transformModule});
  assert.ok(seen.length > 0);
  assert.ok(seen.every(url => !url.endsWith('.css')));
});

test('static, barrel and dynamic imports select one emitted native CSS URL', async () => {
  const source = '\ufeff/* literal bytes */ body { color: red; }\n';
  const f = fixture({
    'app.mjs': `import sheet from './theme.css' with {type:'css'};
      import * as namespace from './theme.css' with {type:'css'};
      import {again} from './barrel.mjs'; export {sheet,namespace,again};
      export const load=()=>import('./theme.css',{with:{type:'css'}});`,
    'barrel.mjs': `export {default as again} from './theme.css' with {type:'css'};`, 'theme.css': source,
  });
  const built = await f.build();
  assert.equal(sheets(built).length, 1);
  assert.equal(fs.readFileSync(path.join(f.out, sheets(built)[0]), 'utf8'), source);
  const code = built.chunks.map(chunk => chunk.code).join('\n');
  const references = [...code.matchAll(/(?:\.\/)?assets\/f3d-css-[0-9a-f]+\.css/g)].map(match => match[0]);
  assert.ok(references.length >= 2);
  assert.equal(new Set(references).size, 1);
  assert.match(code, /with\s*:\s*\{\s*type:\s*'css'/);
  assert.match(code, /with\s*\{\s*type:\s*'css'/);
  assert.doesNotMatch(code, /CSSStyleSheet|replaceSync|adoptedStyleSheets/);
});

test('same CSS bytes at different URLs and query/fragment variants keep separate sheet identities', async () => {
  const f = fixture({'app.mjs': `import a from './theme.css?one#x' with {type:'css'};
    import b from './theme.css?two#x' with {type:'css'};
    import c from './theme.css?one#y' with {type:'css'};
    import d from './copy.css' with {type:'css'}; export {a,b,c,d};`,
    'theme.css': 'body{}', 'copy.css': 'body{}'});
  const built = await f.build();
  assert.equal(sheets(built).length, 4);
  assert.equal(new Set(sheets(built)).size, 4);
});

test('CSS module assets resolve from the stylesheet and retain URL suffixes and image-set descriptors', async () => {
  const f = fixture({'app.mjs': `export {default} from './styles/theme.css' with {type:'css'};`,
    'styles/theme.css': `@font-face{src:url('../fonts/font.woff2?v=1')}
      body{background:image-set('../images/a.png' 1x, url('../images/b.png?x#y') 2x);
        mask:url('../images/icon.svg#mask'); content:'url(missing.png)'}`,
    'fonts/font.woff2': 'font-bytes', 'images/a.png': 'a-bytes', 'images/b.png': 'b-bytes',
    'images/icon.svg': '<svg xmlns="http://www.w3.org/2000/svg"><mask id="mask"/></svg>'});
  const built = await f.build(), sheetPath = path.join(f.out, sheets(built)[0]);
  const css = fs.readFileSync(sheetPath, 'utf8'), resources = cssResources(css);
  assert.equal(resources.length, 4);
  assert.match(css, / 1x,/); assert.match(css, / 2x/); assert.match(css, /content:'url\(missing.png\)'/);
  assert.deepEqual(resources.map(resource => {
    const url = new URL(resource.url, pathToFileURL(sheetPath));
    return [fs.readFileSync(fileURLToPath(url), 'utf8'), url.search + url.hash];
  }), [['font-bytes', '?v=1'], ['a-bytes', ''], ['b-bytes', '?x#y'],
    ['<svg xmlns="http://www.w3.org/2000/svg"><mask id="mask"/></svg>', '#mask']]);
});

test('CSS module @import stays source-owned without following ignored URLs or supports() resources', async () => {
  const source = `@import './missing.css' layer(theme) supports(background:url('missing.png')) screen;
    @import url('./also-missing.css') supports(font-family:'a;b');
    @import/**/url('./comment-separated.css');
    @im\\70ort url('./escaped-keyword.css');
    body{background:url('./present.png')}`;
  assert.deepEqual(cssResources(source, {ignoreImports: true}).map(resource => resource.url), ['./present.png']);
  const f = fixture({'index.html': '<script type="module" src="./app.mjs"></script>',
    'app.mjs': `export {default} from './theme.css' with {type:'css'};`,
    'theme.css': source, 'present.png': 'present'});
  const built = await f.build('index.html');
  const emitted = fs.readFileSync(path.join(f.out, sheets(built)[0]), 'utf8');
  assert.equal(emitted.slice(0, emitted.indexOf('body')), source.slice(0, source.indexOf('body')));
  const packed = f.pack(path.join(f.out, 'index.html'));
  const css = decode(sheetUrls(packed)[0]);
  assert.match(css, /@import '\.\/missing.css'/);
  assert.match(css, /supports\(background:url\('missing.png'\)\)/);
  assert.equal(decode(cssResources(css, {ignoreImports: true})[0].url), 'present');
});

test('normal and direct standalone builds embed CSS modules with static, lazy and shared imports', async () => {
  for (const buildFirst of [false, true]) {
    const f = fixture({'index.html': '<script type="module" src="./app.mjs"></script>',
      'app.mjs': `import theme from './styles/theme.css' with {type:'css'};
        export {theme}; export {default as again} from './styles/theme.css' with {type:'css'};
        export const load=flag=>import(flag?'./styles/theme.css':'./styles/lazy.css',{with:{type:'css'}});`,
      'styles/theme.css': 'body{background:url(../pixel.png)}', 'styles/lazy.css': 'body{color:blue}',
      'pixel.png': 'pixel-bytes'});
    if (buildFirst) await f.build('index.html');
    const packed = f.pack(path.join(buildFirst ? f.out : f.root, 'index.html'));
    assert.equal(sheetUrls(packed).length, 2);
    assert.equal(packed.report.moduleCount, 3);
    const css = sheetUrls(packed).map(decode).join('\n');
    assert.equal(decode(cssResources(css)[0].url), 'pixel-bytes');
    assert.match(css, /color:blue/);
    const js = Object.values(packed.imports).filter(url => url.startsWith('data:text/javascript;')).map(decode).join('\n');
    assert.match(js, /type\s*:\s*['"]css['"]/);
    assert.doesNotMatch(js, /CSSStyleSheet|replaceSync/);
  }
});

test('retained classic imports and bundled modules share a CSS sheet and its original relative asset tree', async () => {
  const classic = `globalThis.loadSheet=()=>import('./styles/theme.css',{with:{type:'css'}});`;
  const f = fixture({'index.html': '<script src="./classic.js"></script><script type="module" src="./app.mjs"></script>',
    'classic.js': classic, 'app.mjs': `export {default} from './styles/theme.css' with {type:'css'};`,
    'styles/theme.css': "@import './missing.css'; body{background:url('../pixel.png')}", 'pixel.png': 'pixel'});
  const built = await f.build('index.html');
  assert.deepEqual(built.emittedFiles.filter(name => name.endsWith('.css')), ['styles/theme.css']);
  assert.equal(fs.readFileSync(path.join(f.out, 'classic.js'), 'utf8'), classic);
  assert.match(built.chunks[0].code, /\.\/styles\/theme.css/);
  assert.equal(fs.readFileSync(path.join(f.out, 'pixel.png'), 'utf8'), 'pixel');
  assert.equal(sheetUrls(f.pack(path.join(f.out, 'index.html'))).length, 1);
});

test('ordinary stylesheet use still follows @import when the same file is also a native CSS module', async () => {
  const f = fixture({'index.html': '<link rel="stylesheet" href="./theme.css"><script>import("./theme.css",{with:{type:"css"}})</script>',
    'theme.css': '@import "./child.css"; body{color:red}', 'child.css': 'body{background:green}'});
  const built = await f.build('index.html');
  assert.ok(built.emittedFiles.includes('child.css'));
  const packed = f.pack(path.join(f.out, 'index.html'));
  assert.match(packed.html, /data:text\/css/);
  assert.match(decode(sheetUrls(packed)[0]), /@import "\.\/child.css"/);
});

test('CSS modules honor import maps and documentRoot asset URLs', async () => {
  const f = fixture({'pages/index.html': `<script type="importmap">{"imports":{"theme":"/styles/theme.css"}}</script>
    <script type="module">import sheet from 'theme' with {type:'css'}; globalThis.sheet=sheet;</script>`,
    'styles/theme.css': 'body{background:url(/images/pixel.png)}', 'images/pixel.png': 'root-pixel'});
  const built = await f.build('pages/index.html', {documentRoot: f.root});
  const sheetPath = path.join(f.out, sheets(built)[0]);
  const resource = cssResources(fs.readFileSync(sheetPath, 'utf8'))[0];
  assert.equal(fs.readFileSync(fileURLToPath(new URL(resource.url, pathToFileURL(sheetPath))), 'utf8'), 'root-pixel');
  assert.equal(sheetUrls(f.pack(path.join(f.out, 'pages/index.html'), {rootDir: f.out, documentRoot: f.out})).length, 1);
});

test('external CSS remains an attributed native import and standalone refuses open resources', async () => {
  const f = fixture({'index.html': '<script type="module" src="./app.mjs"></script>',
    'app.mjs': `export {default} from 'https://example.invalid/theme.css' with {type:'css'};`});
  const built = await f.build('index.html');
  assert.match(built.chunks[0].code, /https:\/\/example.invalid\/theme.css/);
  assert.match(built.chunks[0].code, /with\s*\{\s*type:\s*'css'/);
  assert.throws(() => f.pack(path.join(f.out, 'index.html')), error => error.code === 'EXTERNAL_RESOURCE');
});

test('wrong MIME and missing CSS attributes remain native failures instead of gaining a sheet polyfill', async () => {
  const f = fixture({'app.mjs': `export const effects=[];
    export const css=()=>import('./text.js',(effects.push('options'),{with:{type:'css'}}));
    export const untyped=()=>import('./theme.css');`,
    'text.js': 'not CSS or JavaScript {', 'theme.css': 'body{}'});
  const built = await f.build(), app = await import(f.url('out/' + built.entryFiles[0]));
  await assert.rejects(app.css());
  assert.deepEqual(app.effects, ['options']);
  await assert.rejects(app.untyped());
  assert.ok(built.emittedFiles.some(name => /f3d-css-[0-9a-f]+\.js$/.test(name)));
  assert.ok(built.emittedFiles.some(name => /f3d-css-[0-9a-f]+\.css$/.test(name)));
  assert.doesNotMatch(built.chunks[0].code, /CSSStyleSheet|replaceSync/);
});

test('dynamic CSS options keep selector order, attribute effects and unknown dictionaries unresolved', async () => {
  const f = fixture({'app.mjs': `export const effects=[];
    export const load=flag=>import((effects.push('selector'),flag?'./a.css':'./b.css'),
      (effects.push('options'),{with:{type:(effects.push('type'),'css')}}));
    export const unknown=options=>import('./a.css',options);`, 'a.css': 'body{}', 'b.css': 'body{}'});
  const graph = await f.graph(), requests = graph.modules[f.url('app.mjs')].dynamic_imports;
  assert.equal(requests[0].claims_closure, true);
  assert.equal(requests[1].module_type, 'unknown');
  assert.equal(requests[1].claims_closure, false);
  const built = await f.build(), app = await import(f.url('out/' + built.entryFiles[0]));
  await assert.rejects(app.load(true));
  assert.deepEqual(app.effects, ['selector', 'options', 'type']);
  const effects = [];
  await assert.rejects(app.unknown({get with() {effects.push('with'); return {type:'css'};}}));
  assert.deepEqual(effects, ['with']);
});

test('AST records CSS goals without discarding extra native import attributes', () => {
  const analysis = analyzeModuleAst(`import sheet from './theme.css' with {type:'css',unsupported:'x'};
    export const load=()=>import('./theme.css',{with:{type:'css'}});`, 'file:///app.mjs');
  assert.equal(analysis.staticImports[0].module_type, 'css');
  assert.deepEqual(analysis.staticImports[0].import_attributes, {type:'css',unsupported:'x'});
  assert.equal(analysis.dynamicImports[0].module_type, 'css');
});

test('a missing CSS type rejects lazily without fetching stylesheet dependencies', async () => {
  for (const classic of [false, true]) {
    const source = `globalThis.loadUntypedSheet=()=>import('./theme.css');`;
    const f = fixture({'index.html': classic ? `<script>${source}</script>` : '<script type="module" src="./app.mjs"></script>',
      'app.mjs': source + 'export const load=globalThis.loadUntypedSheet;',
      'theme.css': "body{background:url('./never-requested.png')}"});
    const built = await f.build('index.html');
    if (!classic) {
      const app = await import(f.url('out/' + built.entryFiles[0]));
      await assert.rejects(app.load(), {code:'ERR_UNKNOWN_FILE_EXTENSION'});
    }
    const packed = f.pack(path.join(f.out, 'index.html'));
    assert.equal(packed.report.assetCount, 0);
    assert.match(decode(sheetUrls(packed)[0]), /never-requested.png/);
  }
});

test('a static untyped request cannot inherit the attributes of a valid CSS import', async () => {
  const f = fixture({'app.mjs': `import sheet from './theme.css' with {type:'css'};
    import './theme.css'; export {sheet};`, 'theme.css': 'body{}'});
  const built = await f.build();
  const analysis = analyzeModuleAst(built.chunks[0].code, 'file:///emitted.mjs');
  const requests = [...analysis.staticImports, ...analysis.staticExports.filter(item => item.specifier)];
  assert.deepEqual(requests.map(item => item.module_type).sort(), ['css', 'javascript']);
  assert.equal(new Set(requests.map(item => item.specifier)).size, 2);
  assert.equal(sheets(built).length, 2);
});

test('a static CSS request cannot inherit successful JSON or JavaScript attributes', async () => {
  for (const [name, contents, attribute] of [['data.json', '{"value":1}', " with {type:'json'}"],
    ['data.mjs', 'export default {value:1};', '']]) {
    for (const [alias, retained] of [[false, false], [true, false], [true, true]]) {
      const f = fixture({'app.mjs': `import good from './${name}'${attribute};
        import bad from './${alias ? './' : ''}${name}' with {type:'css'}; export {good,bad};`,
        'index.html': `<script>globalThis.loadNative=()=>import('./${name}'${attribute ? ",{with:{type:'json'}}" : ''})</script>
          <script type="module" src="./app.mjs"></script>`, [name]: contents});
      const built = await f.build(retained ? 'index.html' : 'app.mjs');
      assert.match(built.chunks[0].code, /with\s*\{\s*type:\s*'css'/);
      await assert.rejects(import(f.url('out/' + built.entryFiles[0])), {code:'ERR_IMPORT_ATTRIBUTE_UNSUPPORTED'});
    }
  }
});
