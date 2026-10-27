import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath, pathToFileURL} from 'node:url';
import test from 'node:test';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {buildApplication} from './build_application.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const settingsUrl = new URL('examples/jsm/inspector/tabs/Settings.js', upstream);
const settings = fs.readFileSync(settingsUrl, 'utf8');
const extensions = ['../extensions/color-grading/ColorGrading.js', '../extensions/tsl-graph/TSLGraphEditor.js'];

test('exact Inspector source discovers its registry at any filename without claiming runtime closure', () => {
  const [request] = analyzeModuleAst(settings, 'file:///renamed/navigation.mjs').dynamicImports;
  assert.equal(request.classification, 'finite_set');
  assert.deepEqual(request.candidates, extensions);
  assert.equal(request.runtime_fallback, true); assert.equal(request.unresolved, true);
  assert.equal(settings.slice(request.url_argument_span.start, request.url_argument_span.end), 'extension.url');
  for (const changed of [settings + '\n// changed source\n', settings.replace('return _extensions;', 'return arbitraryExtensions;')]) {
    const [unverified] = analyzeModuleAst(changed, settingsUrl.href).dynamicImports;
    assert.equal(unverified.classification, 'nonliteral');
    assert.equal(unverified.runtime_fallback, undefined);
  }
});

test('the real Inspector graph includes both builtin modules and their static dependencies, retaining unknown imports', async () => {
  const graph = await buildModuleGraph(fileURLToPath(settingsUrl));
  const request = graph.modules[settingsUrl.href].dynamic_imports[0];
  assert.deepEqual(request.resolved_ids, extensions.map(relative => new URL(relative, settingsUrl).href));
  assert.equal(request.unresolved, true);
  assert.equal(request.claims_closure, false); assert.equal(request.claimsClosure, false);
  assert.equal(request.runtime_fallback, true);
  assert.ok(Object.keys(graph.modules).some(url => url.endsWith('/color-grading/ColorWheel.js')));
  assert.ok(Object.keys(graph.modules).some(url => url.endsWith('/tsl-graph/TSLGraphLoader.js')));
});

function fixture() {
  const root = fs.mkdtempSync(path.join(tmpdir(), 'f3d-inspector-import-'));
  const write = (name, code) => {
    const target = path.join(root, name); fs.mkdirSync(path.dirname(target), {recursive: true});
    fs.writeFileSync(target, code); return target;
  };
  write('package.json', '{"type":"module"}');
  write('src/tabs/RenamedSettings.js', settings);
  write('src/three.js', 'export class WebGPURenderer { async init() {} } export class WebGLBackend {} export const Node = {};');
  write('src/Inspector.js', 'export const getItem = () => ({}), setItem = () => {};');
  write('src/tabs/Parameters.js', `export class Parameters {
    createGroup() { return { add: () => { const control = { checkbox: {}, domElement: {style: {setProperty() {}, removeProperty() {}}},
      name() {return this;}, info() {return this;}, onChange(fn) {this.change = fn; return this;} }; return control; } }; }
    dispatchEvent(event) { this.events.push(event.type); }
  }`);
  write('src/state.js', 'export const state = {evaluations: [], constructors: [], initialized: [], removed: []};');
  for (const [folder, name] of [['color-grading', 'ColorGrading'], ['tsl-graph', 'TSLGraphEditor']]) {
    write(`src/extensions/${folder}/${name}.js`, `import {state} from '../../state.js';
      state.evaluations.push(${JSON.stringify(name)});
      export class ${name} {
        constructor() { this.name = ${JSON.stringify(name)}; this.asset = new URL('./ramp.txt', import.meta.url).href; state.constructors.push(this.name); }
        init(inspector) { state.initialized.push(this.name); this.sameState = inspector.state === state; }
      }`);
    write(`src/extensions/${folder}/ramp.txt`, `${name} relative asset`);
  }
  write('src/main.js', `import {Settings} from './tabs/RenamedSettings.js'; import {state} from './state.js';
    const settings = new Settings(); settings.events = [];
    settings.inspector = {state, addTab(tab) {tab.init(this);}, removeTab(tab) {state.removed.push(tab.name);} };
    settings.init(); await Promise.resolve();
    export {state}; export const records = settings.extensions, events = settings.events;
    export const select = (name, active) => settings.setActiveExtension(name, active);
    export const load = record => settings._loadExtension(settings.inspector, record);
  `);
  const entry = write('index.html', '<script type="importmap">{"imports":{"three/webgpu":"./src/three.js"}}</script>' +
    '<script type="module" src="./src/main.js"></script>');
  return {root, entry};
}

test('emitted registry imports stay lazy and share source identity, initialize tabs and load relative child assets', async () => {
  const f = fixture(), out = path.join(f.root, 'dist');
  const result = await buildApplication(f.entry, out);
  assert.ok(result.chunks.some(chunk => chunk.fileName.startsWith('ColorGrading-')));
  assert.ok(result.chunks.some(chunk => chunk.fileName.startsWith('TSLGraphEditor-')));
  const api = await import(pathToFileURL(path.join(out, result.entryFiles[0])).href);
  assert.deepEqual(api.state.evaluations, [], 'loading Inspector does not activate an extension');
  for (const [label, name] of [['Color Grading', 'ColorGrading'], ['TSL Graph', 'TSLGraphEditor']]) {
    const before = api.state.initialized.length;
    const pending = api.select(label, true);
    assert.equal(api.state.initialized.length, before, 'native asynchronous import boundary remains');
    await pending;
    const record = api.records[label];
    assert.equal(record.active, true); assert.equal(record.loaded, true);
    assert.equal(record.tab.sameState, true, 'the extension and application share their original module singleton');
    assert.equal(fs.readFileSync(fileURLToPath(record.tab.asset), 'utf8'), `${name} relative asset`);
    assert.ok(fileURLToPath(record.tab.asset).startsWith(out + path.sep));
    await api.select(label, true);
    assert.equal(api.state.constructors.filter(value => value === name).length, 1, 'active tabs are not re-created');
    await api.select(label, false); assert.equal(record.loaded, false); assert.equal(record.tab, null);
    await api.select(label, true);
    assert.equal(api.state.evaluations.filter(value => value === name).length, 1, 'reopening keeps native module identity');
    assert.equal(api.state.constructors.filter(value => value === name).length, 2);
  }
  assert.deepEqual(api.events, ['extensionadded', 'extensionremoved', 'extensionadded', 'extensionadded', 'extensionremoved', 'extensionadded']);
});

test('unknown runtime extension URLs and errors retain native import semantics and evaluate the selector once', async () => {
  const f = fixture(), result = await buildApplication(f.entry, path.join(f.root, 'dist'));
  const api = await import(pathToFileURL(path.join(f.root, 'dist', result.entryFiles[0])).href);
  const code = 'export class RuntimeExtension { init(inspector) { this.observed = inspector.state; } }';
  const url = 'data:text/javascript,' + encodeURIComponent(code);
  let reads = 0;
  const record = {name: 'runtime', active: false, loaded: false, ui: {domElement: {style: {setProperty() {}}}, checkbox: {}}};
  Object.defineProperty(record, 'url', {get() {reads++; return url;}});
  await api.load(record);
  assert.equal(reads, 1); assert.equal(record.loaded, true); assert.equal(record.tab.observed, api.state);
  const failure = new Error('extension URL getter');
  const throwing = {...record, active: false, get url() {throw failure;}};
  await assert.rejects(api.load(throwing), error => error === failure);
  await assert.rejects(api.load({...record, active: false, url: 'data:text/javascript,invalid%20module%20syntax'}), SyntaxError);
});
