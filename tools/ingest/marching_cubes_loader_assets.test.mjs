/** Real numeric application builds with pinned Three loaders. Image requests
 * stop at the DOM boundary; this checks resource bytes and source effects. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {parse} from 'acorn';
import {buildApplication} from './build_application.mjs';
import {f3dRollupPlugin} from './bundler.mjs';
import {packHtml} from './pack_html.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const faces = ['px', 'nx', 'py', 'ny', 'pz', 'nz'];
const array = prefix => '[' + faces.map(face => JSON.stringify(`${prefix}${face}.jpg`)).join(',') + ']';

for (const [prepareStandaloneLoaders, modifiedLoader] of [[false, false], [true, false], [true, true]])
  test(modifiedLoader ? 'numeric builds retain the host route of a frontend-modified loader'
    : `numeric ball-loop builds retain cube-loader resources after relocation${prepareStandaloneLoaders ? ' and standalone packing' : ''}`, async t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-ball-loader-assets-'));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const source = path.join(root, 'source'), output = path.join(root, 'output');
    fs.mkdirSync(path.join(source, 'assets/sky'), {recursive: true});
    fs.mkdirSync(path.join(source, 'host'));
    for (const prefix of ['mutable-', 'escaped-', 'opaque-']) for (const face of faces)
      fs.writeFileSync(path.join(source, `host/${prefix}${face}.jpg`), `unresolved ${prefix}${face}`);
    for (const name of ['changed', 'escaped-change']) fs.writeFileSync(path.join(source, `host/${name}.jpg`), name);
    const images = faces.map(face => fs.readFileSync(new URL(`examples/textures/cube/SwedishRoyalCastle/${face}.jpg`, upstream)));
    for (const [i, face] of faces.entries()) fs.writeFileSync(path.join(source, `assets/sky/${face}.jpg`), images[i]);
    fs.writeFileSync(path.join(source, 'index.html'), '<script type="module" src="./app.mjs"></script>');
    fs.writeFileSync(path.join(source, 'app.mjs'), `
import {CubeTextureLoader} from ${JSON.stringify(new URL('build/three.core.js', upstream).href)};
export const effects = [];
export function load() {
  const path = 'assets/sky/', extension = '.jpg';
  const urls = [${faces.map(face => `(effects.push(${JSON.stringify(face)}), path + ${JSON.stringify(face)} + extension)`).join(',')}];
  return new CubeTextureLoader().load(urls);
}
export function mutable() {
  const urls = ${array('host/mutable-')}; urls[0] = 'host/changed.jpg';
  return new CubeTextureLoader().load(urls);
}
export function escaped(change) {
  const urls = ${array('host/escaped-')}; change(urls);
  return new CubeTextureLoader().load(urls);
}
export function opaque(identity) {
  return new CubeTextureLoader().load(identity(${array('host/opaque-')}));
}
export function update(owner, count, time) {
  for (let i = 0; i < count; i++) {
    const x = Math.sin(i + time) * .2 + .5;
    owner.addBall(x, .5, .5, .8, 12);
  }
}
`);
    const built = await buildApplication(path.join(source, 'index.html'), output,
      {specializeNumeric: true, prepareStandaloneLoaders,
        ...(modifiedLoader ? {transformModule(code, {url}) {
          return url === new URL('build/three.core.js', upstream).href
            ? code.replace('image.src = url;', 'image.src = String( url );') : null;
        }} : {})});
    assert.equal(built.numericSpecialization.libraryKernels.marchingCubes.compiledBallLoops, 1);
    fs.writeFileSync(path.join(output, 'package.json'), '{"type":"module"}', {flag: 'wx'});
    fs.renameSync(source, source + '-unavailable');
    const relocated = path.join(root, 'relocated'); fs.renameSync(output, relocated);
    for (const [i, face] of faces.entries())
      assert.deepEqual(fs.readFileSync(path.join(relocated, `assets/sky/${face}.jpg`)), images[i]);
    assert.equal(fs.existsSync(path.join(relocated, 'host')), false, 'mutable, escaping and arbitrary helper arrays stay host-owned');

    const previousDocument = globalThis.document, requests = [];
    globalThis.document = {URL: 'https://example.test/index.html', baseURI: 'https://example.test/index.html',
      createElementNS(_namespace, tag) {
        assert.equal(tag, 'img');
        return {addEventListener() {}, removeEventListener() {}, set src(value) { requests.push(value); }};
      }};
    try {
      const application = await import(pathToFileURL(path.join(relocated, built.entryFiles[0])).href);
      application.load(); assert.deepEqual(application.effects, faces);
      assert.equal(requests.length, 6);
      for (const [i, request] of requests.entries()) {
        if (prepareStandaloneLoaders && !modifiedLoader) {
          assert.match(request, /^data:image\/jpeg[^,]*;base64,/);
          assert.deepEqual(Buffer.from(request.slice(request.indexOf(',') + 1), 'base64'), images[i]);
        } else assert.equal(request, `assets/sky/${faces[i]}.jpg`);
      }
      application.mutable();
      assert.equal(requests[6], 'host/changed.jpg', 'array mutation still controls the actual loader request');
      application.escaped(urls => { urls[0] = 'host/escaped-change.jpg'; });
      assert.equal(requests[12], 'host/escaped-change.jpg');
      let opaqueCalls = 0;
      application.opaque(urls => { opaqueCalls++; return urls; });
      assert.equal(opaqueCalls, 1); assert.equal(requests[18], 'host/opaque-px.jpg');
    } finally {
      if (previousDocument === undefined) delete globalThis.document; else globalThis.document = previousDocument;
    }
    if (prepareStandaloneLoaders && !modifiedLoader) {
      const resolver = fs.readFileSync(path.join(relocated, 'f3d-loader-resources.mjs'), 'utf8');
      for (const image of images) assert.ok(resolver.includes(image.toString('base64')));
      const packed = path.join(root, 'packed.html'); packHtml(path.join(relocated, built.htmlFile), packed);
      const html = fs.readFileSync(packed, 'utf8');
      const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
      assert.ok(Object.values(imports).some(url =>
        Buffer.from(url.split(',')[1].split('#')[0], 'base64').toString() === resolver),
      'the working loader resource resolver survives standalone packing byte for byte');
    }
  });

for (const mode of ['matching compiler output', 'intervening transform', 'changed lexical owner'])
  test(`caller-forged ${mode} cannot authorize standalone hook relocation`, () => {
    const id = new URL('src/loaders/ImageLoader.js', upstream).href, before = fs.readFileSync(new URL(id), 'utf8');
    const after = '// compiler-owned header\n' + (mode === 'changed lexical owner'
      ? before.replace('image.src = url;', 'image.src = String( url );') : before);
    const current = mode === 'intervening transform' ? after + '\n// later transform\n' : after;
    const plugin = f3dRollupPlugin({mapBaseUrl: id, documentUrl: new URL('examples/index.html', upstream).href,
      prepareStandaloneLoaders: true, _resourceSourceSnapshots: new Map([[id, {before, after}]])});
    const result = plugin.transform.call({parse: code => parse(code, {ecmaVersion: 'latest', sourceType: 'module'})}, current, id);
    assert.equal(result, null, 'no source-pin bypass or stale source-span edits');
  });
