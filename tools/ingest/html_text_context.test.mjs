import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {buildApplication, extractRelativeAssetUrls, rewriteHtmlForBuild} from './build_application.mjs';
import {parseHtmlEntries, stripHtmlComments, stripScriptAndStyleBodies} from './html_parser.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {packHtml} from './pack_html.mjs';

function fixture(html, files = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-html-text-'));
  fs.writeFileSync(path.join(root, 'index.html'), html);
  fs.writeFileSync(path.join(root, 'package.json'), '{"type":"module"}');
  for (const [name, source] of Object.entries(files)) fs.writeFileSync(path.join(root, name), source);
  return {root, entry: path.join(root, 'index.html'), out: path.join(root, 'dist')};
}

const displayed = `<base href="https://display-only.invalid/">
<script type="importmap">this is displayed, not JSON</script>
<script type="module" src="./display-only.mjs"></script>
<script>new Worker('./display-only-worker.js'); import('./display-only-lazy.mjs');</script>
<link rel="stylesheet" href="./display-only.css"><img src="./display-only.png">
<style>body { background: url('./display-only-style.png') }</style>`;
const active = '<script type="module">export const answer = 42;</script>';

for (const name of ['textarea', 'title', 'xmp', 'noembed', 'noframes']) {
  test(`${name} text stays outside discovery, ordinary builds and standalone modules`, async () => {
    const inert = `<${name} data-description="display > source">${displayed}</${name}>`;
    const html = `<!doctype html>\n${inert}\n${active}`;
    const f = fixture(html), url = pathToFileURL(f.entry).href;
    const parsed = parseHtmlEntries(html, url);
    assert.equal(parsed.baseHref, null);
    assert.deepEqual(parsed.importMap, {imports: {}, scopes: {}});
    assert.equal(parsed.moduleScripts.length, 1);
    assert.equal(parsed.moduleScripts[0].inlineContent, 'export const answer = 42;');
    assert.equal(parsed.moduleScripts[0].startOffset, html.indexOf('export const answer'));
    assert.equal(parsed.moduleScripts[0].startLine, html.slice(0, parsed.moduleScripts[0].startOffset).split('\n').length);
    assert.deepEqual(extractRelativeAssetUrls(html), []);
    const graph = await buildModuleGraph(f.entry);
    assert.equal(Object.keys(graph.modules).length, 1);
    const built = await buildApplication(f.entry, f.out);
    assert.equal(built.entryFiles.length, 1);
    const emitted = fs.readFileSync(path.join(f.out, 'index.html'), 'utf8');
    assert.ok(emitted.includes(inert), 'displayed contents retain every source byte');
    assert.equal((await import(pathToFileURL(path.join(f.out, built.entryFiles[0])))).answer, 42);
    const packed = path.join(f.root, 'packed.html');
    const result = packHtml(path.join(f.out, 'index.html'), packed);
    assert.equal(result.moduleCount, 1);
    assert.ok(fs.readFileSync(packed, 'utf8').includes(inert));
  });
}

for (const name of ['div', 'code-editor.v1', 'code-editor_v1', 'code-écran'])
test(`quoted ${name} display markup cannot introduce scripts, bases, assets or workers`, async () => {
  const inert = `<${name} data-example='<script type="module" src="./display-only.mjs"></script><base href="https://wrong.invalid/"><img src="./display-only.png">'>Example</${name}>`;
  const f = fixture(inert + active);
  const result = await buildApplication(f.entry, f.out);
  assert.equal(result.entryFiles.length, 1);
  assert.ok(fs.readFileSync(path.join(f.out, 'index.html'), 'utf8').includes(inert));
  const packed = path.join(f.root, 'packed.html');
  assert.equal(packHtml(path.join(f.out, 'index.html'), packed).moduleCount, 1);
  assert.ok(fs.readFileSync(packed, 'utf8').includes(inert));
});

test('text end tags need the exact name and keep following module source offsets', () => {
  const inert = `<TeXtArEa>first </textarea-other>${displayed}</tExTaReA \t>`;
  const html = inert + active, parsed = parseHtmlEntries(html, 'file:///app/index.html');
  assert.equal(parsed.moduleScripts.length, 1);
  assert.equal(parsed.moduleScripts[0].startOffset, html.indexOf('export const answer'));
  const output = rewriteHtmlForBuild(html, ['main.mjs']);
  assert.equal(output, inert + '<script type="module" src="./main.mjs"></script>');
});

for (const name of ['textarea', 'title', 'xmp', 'plaintext']) {
  test(`unterminated ${name} consumes the remaining display text without rewriting it`, async () => {
    const inert = `<${name}>${displayed}${name === 'plaintext' ? '</plaintext>' + displayed : ''}`;
    const f = fixture(active + inert);
    const result = await buildApplication(f.entry, f.out);
    assert.equal(result.entryFiles.length, 1);
    assert.ok(fs.readFileSync(path.join(f.out, 'index.html'), 'utf8').endsWith(inert));
    const packed = path.join(f.root, 'packed.html');
    assert.equal(packHtml(path.join(f.out, 'index.html'), packed).moduleCount, 1);
    assert.ok(fs.readFileSync(packed, 'utf8').endsWith(inert));
  });
}

test('text element style attributes still have real resource dependencies', async () => {
  const html = `<textarea style="background-image:url('./paper.svg')">${displayed}</textarea>${active}`;
  const image = '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>';
  const f = fixture(html, {'paper.svg': image});
  assert.deepEqual(extractRelativeAssetUrls(html), ['./paper.svg']);
  await buildApplication(f.entry, f.out);
  assert.equal(fs.readFileSync(path.join(f.out, 'paper.svg'), 'utf8'), image);
  const packed = path.join(f.root, 'packed.html');
  packHtml(path.join(f.out, 'index.html'), packed);
  const output = fs.readFileSync(packed, 'utf8');
  const packedImage = /<textarea style="background-image:url\(&quot;(data:image\/svg\+xml[^&]+)&quot;\)/.exec(output)?.[1];
  assert.ok(packedImage);
  assert.equal(Buffer.from(packedImage.split(',')[1], 'base64').toString(), image);
  assert.ok(output.includes(displayed));
});

test('scripting-enabled noscript and iframe fallback text never becomes entry code', () => {
  const html = `<noscript>${displayed}</noscript><iframe>${displayed}</iframe>${active}`;
  assert.equal(parseHtmlEntries(html, 'file:///app/index.html').moduleScripts.length, 1);
  assert.deepEqual(extractRelativeAssetUrls(html), []);
});

test('displayed comments stay text while actual comments retain source coordinates', () => {
  const html = '<!-- actual\ncomment --><textarea><!-- displayed --></textarea><script>const comment = "<!-- source -->";</script>';
  const stripped = stripHtmlComments(html);
  assert.equal(stripped.length, html.length);
  assert.ok(stripped.includes('<textarea><!-- displayed --></textarea>'));
  assert.ok(stripped.includes('const comment = "<!-- source -->";'));
  assert.ok(!stripped.includes('actual'));
  const masked = stripScriptAndStyleBodies(html);
  assert.equal(masked.length, html.length);
  assert.equal(masked.split('\n').length, html.split('\n').length);
  assert.ok(!masked.includes('displayed'));
});
