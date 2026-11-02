import assert from "node:assert/strict";
import test from "node:test";
import { analyzeModuleAst } from "./ast_analyzer.mjs";
import { decideRendererRoute } from "../compat/route_decider.mjs";

const moduleUrl = "file:///app/main.js";
function analyze(body, constructor = "WebGPURenderer") {
  return analyzeModuleAst("new " + constructor + "({forceWebGL:false});\n" + body, moduleUrl);
}
function decision(analysis) {
  const site = analysis.rendererConstructionSites[0];
  assert.ok(site, "the source constructor must be recognized");
  return decideRendererRoute({
    constructorName: site.constructorName,
    options: {forceWebGL: site.forceWebGL, forceWebGLUnresolved: site.forceWebGLUnresolved},
    analysis: analysis.routingFacts,
    hostCapabilities: {hasWebGPU: true, hasWebGL: true},
    generalWebGPUAvailable: true,
    generalWebGLAvailable: true,
  });
}

test("computed and optional native-context calls retain exact renderer ownership", () => {
  for (const constructor of ["WebGPURenderer", "WebGLRenderer"]) {
    for (const body of [
      "canvas['getContext']('webgl2');",
      "canvas[\`getContext\`]('webgpu');",
      "canvas['get' + 'Context']('webgl');",
      "canvas[\`get\${'Context'}\`]('experimental-webgl');",
      "canvas[choose() ? 'getContext' : 'unused']('webgl2');",
      "canvas?.['getContext']?.('webgl2');",
      "renderer['getContext']();",
    ]) {
      const result = analyze(body, constructor);
      assert.equal(result.routingFacts.hasNativeContextAccess, true, body);
      assert.equal(decision(result).route, "exact-backend", body);
    }
  }
});

test("computed raw GL methods prevent WebGPU substitution", () => {
  for (const name of ["getExtension", "getParameter", "createBuffer", "bindBuffer", "createTexture", "bindTexture"]) {
    for (const body of [
      "gl[" + JSON.stringify(name) + "](value);",
      "gl[choose() ? " + JSON.stringify(name) + " : 'unused'](value);",
    ]) {
      const result = analyze(body);
      assert.equal(result.routingFacts.hasOpaqueGLEscapes, true, body);
      assert.equal(decision(result).route, "exact-backend", body);
      assert.equal(result.routingFacts.escapes[0].method, name);
    }
  }
});

test("borrowed context accessors and GL methods retain their exact native owner", () => {
  for (const body of [
    "const acquire = renderer['getContext']; acquire.call(renderer);",
    "const acquire = canvas.getContext.bind(canvas); acquire('webgl2');",
    "const extension = gl[\`getExtension\`]; extension.call(gl, 'EXT_color_buffer_float');",
    "Reflect.apply(gl['getParameter'], gl, [123]);",
    "(0, renderer.getContext)();",
  ]) {
    const result = analyze(body);
    assert.equal(decision(result).route, "exact-backend", body);
    assert.equal(result.routingFacts.escapes.length, 1, "one source access, not duplicate call/read facts");
  }
});

test("finite non-native context selections still admit ordinary applications", () => {
  for (const body of [
    "canvas.getContext('2d');",
    "canvas['getContext'](\`2d\`);",
    "canvas['get' + 'Context']('bitmap' + 'renderer');",
    "canvas.getContext(choose() ? '2d' : 'bitmaprenderer');",
    "canvas?.[\`getContext\`]?.(choose() ? '2d' : 'bitmaprenderer');",
    "object['render'](scene, camera);",
  ]) {
    const result = analyze(body);
    assert.equal(result.routingFacts.hasNativeContextAccess, false, body);
    assert.equal(result.routingFacts.hasOpaqueGLEscapes, false, body);
    assert.equal(decision(result).route, "general-webgpu", body);
  }
});

test("unknown context arguments remain unresolved and finite native alternatives stay exact", () => {
  for (const body of [
    "canvas['getContext'](kind);",
    "canvas['getContext'](\`\${kind}\`);",
    "canvas['getContext'](...args);",
  ]) {
    const result = analyze(body);
    assert.equal(result.routingFacts.hasUnresolvedContextAccess, true, body);
    assert.equal(decision(result).route, "exact-backend", body);
  }
  const choice = analyze("canvas['getContext'](choose() ? '2d' : 'webgl2');");
  assert.equal(choice.routingFacts.hasOpaqueGLEscapes, true);
  assert.equal(decision(choice).route, "exact-backend");
});

test("static computed renderer constructors remain eligible when their options are known", () => {
  for (const constructor of ["THREE['WebGPURenderer']", "THREE[\`WebGPURenderer\`]",
    "THREE['Web' + 'GPU' + 'Renderer']"]) {
    const result = analyze("", constructor);
    assert.equal(result.rendererConstructionSites.length, 1, constructor);
    assert.equal(result.rendererConstructionSites[0].constructorName, "WebGPURenderer");
    assert.equal(decision(result).route, "general-webgpu", constructor);
  }
});

test("constructor options preserve spread order and explicit computed keys", () => {
  for (const [options, expected, opaqueCanvas = false] of [
    ["{['forceWebGL']:true}", true],
    ["{[\`forceWebGL\`]:false}", false],
    ["{forceWebGL:false,...options}", "unresolved", true],
    ["{...options,forceWebGL:false}", false, true],
    ["{forceWebGL:true,...{antialias:true}}", true],
    ["{forceWebGL:true,...{forceWebGL:false}}", false],
    ["{...{...options},forceWebGL:true}", true],
    ["{forceWebGL:false,[key]:1}", "unresolved", true],
    ["{[key]:1,forceWebGL:false}", false, true],
    ["{[choose()?'forceWebGL':'alpha']:true}", "unresolved"],
    ["{forceWebGL:true,[choose()?'alpha':'antialias']:true}", true],
    ["(effect(),{forceWebGL:true})", true],
    ["options", "unresolved", true],
    ["...options", "unresolved", true],
    ["{get forceWebGL(){ throw new Error('must not run'); }}", "unresolved"],
    ["{forceWebGL(){}}", "unresolved"],
    ["{forceWebGL:true,...'text'}", true],
    ["{...null,forceWebGL:true}", true],
  ]) {
    const result = analyzeModuleAst("new WebGPURenderer(" + options + ");", moduleUrl);
    const site = result.rendererConstructionSites[0];
    assert.equal(site.forceWebGL, expected, options);
    assert.equal(site.forceWebGLUnresolved, expected === "unresolved", options);
    const route = decision(result);
    assert.equal(route.route, expected === true || opaqueCanvas ? "exact-backend" : "general-webgpu", options);
    assert.equal(route.reasons.includes("unresolved-force-webgl"), expected === "unresolved", options);
  }
});

test("explicit prototypes and object spreads preserve inherited versus own options", () => {
  for (const [options, expected] of [
    ["{__proto__:{forceWebGL:true}}", true],
    ["{'__proto__':{forceWebGL:true}}", true],
    ["{__proto__:options}", "unresolved"],
    ["{__proto__:null}", false],
    ["{forceWebGL:false,__proto__:options}", false],
    ["{__proto__:options,forceWebGL:false}", false],
    ["{...{__proto__:{forceWebGL:true}}}", false],
    ["{['__proto__']:{forceWebGL:true}}", false],
    ["{forceWebGL:true,...{__proto__:{forceWebGL:false}}}", true],
    ["{...{__proto__:null,forceWebGL:false}}", false],
  ]) {
    const result = analyzeModuleAst("new WebGPURenderer(" + options + ");", moduleUrl);
    assert.equal(result.rendererConstructionSites[0].forceWebGL, expected, options);
  }
});

test("deep or opaque constructor options remain bounded runtime decisions", () => {
  const options = "{...".repeat(40) + "unknown" + "}".repeat(40);
  const result = analyzeModuleAst("new WebGPURenderer(" + options + ");", moduleUrl);
  assert.equal(result.rendererConstructionSites[0].forceWebGL, "unresolved");
  assert.equal(result.rendererConstructionSites[0].forceWebGLUnresolved, true);
});

test("native-access diagnostics retain exact source spans without evaluating selector effects", () => {
  const code = "new WebGPURenderer();\ncanvas[choose() ? 'getContext' : 'unused']('webgl2');";
  const result = analyzeModuleAst(code, moduleUrl);
  const span = result.routingFacts.escapes[0].source_span;
  assert.equal(span.start.line, 2);
  assert.equal(code.slice(span.start.offset, span.end.offset),
    "canvas[choose() ? 'getContext' : 'unused']('webgl2')");
  assert.equal(result.routingFacts.escapes.length, 1);
  const borrowed = "new WebGPURenderer(); const get = renderer['getContext'];";
  const access = analyzeModuleAst(borrowed, moduleUrl).routingFacts.escapes[0];
  assert.equal(borrowed.slice(access.source_span.start.offset, access.source_span.end.offset),
    "renderer['getContext']");
});

test("renderer and canvas aliases retain their native owner at opaque observation boundaries", () => {
  for (const body of [
    "plugin(renderer);",
    "new Plugin(renderer);",
    "globalThis.renderer = renderer;",
    "export { renderer };",
    "export default renderer;",
    "const alias = renderer; plugin(alias);",
    "const alias = renderer.setMRT(layout); plugin(alias);",
    "let alias; function capture() { alias = renderer; } plugin(alias);",
    "const canvas = renderer.domElement; plugin(canvas);",
    "const { domElement: canvas } = renderer; plugin(canvas);",
    "const items = [renderer];",
    "const holder = { renderer };",
    "const copied = { ...renderer };",
    "export const create = () => renderer;",
    "function expose() { return renderer; }",
    "renderer[method](scene, camera);",
    "renderer.customPlugin();",
    "renderer.render = plugin; renderer.render(scene, camera);",
    "const alias = renderer; alias['render'] = plugin; renderer.render(scene, camera);",
    "renderer.setMRT = plugin; renderer.setMRT(layout);",
    "const { getContext: acquire } = renderer; acquire();",
    "const backend = renderer.backend;",
    "const constructor = renderer.constructor;",
    "document.body.appendChild = plugin; document.body.appendChild(renderer.domElement);",
    "const document = {body:{appendChild:plugin}}; document.body.appendChild(renderer.domElement);",
    "Reflect.get(renderer, method)();",
    "eval(source);",
  ]) {
    const source = "const renderer = new WebGPURenderer();\n" + body;
    const result = analyzeModuleAst(source, moduleUrl);
    assert.equal(decision(result).route, "exact-backend", body);
    const escape = result.routingFacts.escapes.find(item => item.type === "renderer_resource_escape");
    assert.ok(escape, body);
    assert.equal(escape.source_span.start.line, 2, body);
    assert.ok(source.slice(escape.source_span.start.offset, escape.source_span.end.offset), body);
  }
});

test("constructor aliases, returned factories, subclasses and later exported assignments stay conservative", () => {
  for (const source of [
    "import { WebGPURenderer as Renderer } from 'three/webgpu'; const renderer = new Renderer(); plugin(renderer);",
    "const Renderer = THREE.WebGPURenderer; const renderer = new Renderer(); export { renderer };",
    "const { WebGPURenderer: Renderer } = THREE; const renderer = new Renderer(); plugin(renderer);",
    "const renderer = new WebGPURenderer(); class Custom extends THREE.WebGPURenderer {}",
    "export let renderer; renderer = new WebGPURenderer();",
    "export const make = () => new WebGPURenderer();",
    "import * as THREE from 'three/webgpu'; const renderer = new THREE.WebGPURenderer(); export { THREE };",
    "import * as engine from 'mapped-engine'; const Renderer = engine.WebGPURenderer; const renderer = new Renderer(); export default engine;",
    "const renderer = new WebGPURenderer(); export * from './engine-api.mjs';",
    "import * as engine from 'mapped-engine'; const renderer = new WebGPURenderer(); plugin(engine);",
    "const Renderer = THREE.WebGPURenderer; Renderer.prototype.render = plugin; const renderer = new Renderer(); renderer.render(scene,camera);",
  ]) {
    const result = analyzeModuleAst(source, moduleUrl);
    assert.equal(result.rendererConstructionSites[0].constructorName, "WebGPURenderer", source);
    assert.equal(decision(result).route, "exact-backend", source);
  }
});

test("constructor-supplied local canvases are tracked before the renderer is bound", () => {
  const source = `const canvas = document.createElement('canvas');
const alias = canvas;
plugin(alias);
const renderer = new WebGPURenderer({canvas});`;
  const result = analyzeModuleAst(source, moduleUrl);
  assert.equal(decision(result).route, "exact-backend");
  assert.ok(result.routingFacts.escapes.some(item => item.reason === "opaque-call-argument"));
});

test("externally supplied canvas identities and opaque options cannot authorize a new context owner", () => {
  for (const source of [
    "const renderer = new WebGPURenderer({canvas: host.canvas});",
    "const canvas = host.canvas; const renderer = new WebGPURenderer({canvas});",
    "import {canvas} from './host.mjs'; const renderer = new WebGPURenderer({canvas});",
    "function setup(canvas) { const renderer = new WebGPURenderer({canvas}); renderer.render(scene,camera); }",
    "let canvas = document.createElement('canvas'); canvas = host.canvas; const renderer = new WebGPURenderer({canvas});",
    "const renderer = new WebGPURenderer(options);",
    "document.createElement = plugin; const canvas = document.createElement('canvas'); const renderer = new WebGPURenderer({canvas});",
    "const renderer = new WebGPURenderer(); let container = document.body; container = host; container.appendChild(renderer.domElement);",
  ]) {
    const result = analyzeModuleAst(source, moduleUrl);
    assert.equal(decision(result).route, "exact-backend", source);
  }
});

test("closed render calls, lexical aliases and ordinary DOM attachment remain eligible", () => {
  for (const source of [
    "const renderer = new WebGPURenderer(); renderer.render(scene, camera);",
    "const renderer = new WebGPURenderer(); renderer.setMRT(layout).render(scene, camera);",
    "const renderer = new WebGPURenderer(); const alias = renderer; alias.render(scene, camera);",
    "const Renderer = THREE.WebGPURenderer; const renderer = new Renderer(); renderer.render(scene, camera);",
    "import { WebGPURenderer as Renderer } from 'three/webgpu'; const renderer = new Renderer(); renderer.render(scene, camera);",
    "const renderer = new WebGPURenderer(); document.body.appendChild(renderer.domElement);",
    "let container; container = document.getElementById('container'); const renderer = new WebGPURenderer(); const {domElement: canvas} = renderer; container.appendChild(canvas);",
    "const canvas = document.createElement('canvas'); const renderer = new WebGPURenderer({canvas}); document.body.appendChild(canvas);",
    "const renderer = new WebGPURenderer(); export const size = renderer.getPixelRatio(); export const draw = () => renderer.render(scene, camera);",
    "const renderer = new WebGPURenderer(); { const renderer = ordinary; plugin(renderer); } renderer.render(scene, camera);",
    "const renderer = new WebGPURenderer(); function helper(renderer) { plugin(renderer); } renderer.render(scene, camera);",
    "const renderer = new WebGPURenderer(); try {} catch (renderer) { plugin(renderer); } renderer.render(scene, camera);",
    "const renderer = new WebGPURenderer(); for (let renderer of ordinary) plugin(renderer); renderer.render(scene, camera);",
  ]) {
    const result = analyzeModuleAst(source, moduleUrl);
    assert.equal(result.routingFacts.hasOpaqueGLEscapes, false, source);
    assert.equal(decision(result).route, "general-webgpu", source);
  }
});

test("closed local factories retain renderer ownership through calls, returns and canvas parameters", () => {
  for (const source of [
    "function make() { return new WebGPURenderer(); } const renderer = make(); renderer.render(scene,camera);",
    "const make = () => new WebGPURenderer(); make().render(scene,camera);",
    "const make = function factory() { const value = new WebGPURenderer(); return value; }; const renderer=make(); renderer.render(scene,camera);",
    "const renderer=(() => new WebGPURenderer())(); renderer.render(scene,camera);",
    "function make() { const first=new WebGPURenderer(), second=new WebGPURenderer(); return flag?first:second; } make().render(scene,camera);",
    "const original=new WebGPURenderer(); function identity(value) { return value; } const renderer=identity(original); renderer.render(scene,camera);",
    "function create(canvas) { return new WebGPURenderer({canvas}); } const canvas=document.createElement('canvas'); const renderer=create(canvas); document.body.appendChild(renderer.domElement);",
    "const canvas=document.createElement('canvas'); function inner(canvas) { return new WebGPURenderer({canvas}); } function outer(canvas) { return inner(canvas); } outer(canvas).render(scene,camera);",
    "function canvas() { return document.createElement('canvas'); } function make(element) { return new WebGPURenderer({canvas:element}); } make(canvas()).render(scene,camera);",
    "function make() { const canvas=document.createElement('canvas'); return new WebGPURenderer({canvas}); } export const draw=()=>make().render(scene,camera);",
  ]) {
    const result = analyzeModuleAst(source, moduleUrl);
    assert.equal(result.routingFacts.hasOpaqueGLEscapes, false, JSON.stringify(result.routingFacts.escapes) + '\n' + source);
    assert.equal(decision(result).route, "general-webgpu", source);
  }
});

test("factory results remain resources at exports, opaque calls, containers, throws and native reads", () => {
  for (const use of [
    "plugin(make());", "const renderer=make(); plugin(renderer);", "export const renderer=make();",
    "export default make();", "const renderer=make(); renderer.getContext();", "const renderer=make(); plugin(renderer.domElement);",
    "const renderer=make(); renderer.backend.device;", "throw make();", "const box={renderer:make()};",
    "const list=[make()];", "const renderer=make(); renderer.render=plugin;",
    "function leak() { return make(); } export {leak};", "register(() => make());",
  ]) {
    const source = "function make() { return new WebGPURenderer(); } " + use;
    const result = analyzeModuleAst(source, moduleUrl);
    assert.equal(decision(result).route, "exact-backend", source);
  }
});

test("escaped, reassigned, recursive, async and dynamically selected factories stay conservative", () => {
  for (const source of [
    "export function make() { return new WebGPURenderer(); } make().render(scene,camera);",
    "export const make=()=>new WebGPURenderer(); make().render(scene,camera);",
    "function make() { return new WebGPURenderer(); } export {make}; make().render(scene,camera);",
    "function make() { return new WebGPURenderer(); } export {make as default}; make().render(scene,camera);",
    "function make() { return new WebGPURenderer(); } export default make; make().render(scene,camera);",
    "function make() { return new WebGPURenderer(); } plugin(make); make().render(scene,camera);",
    "function make() { return new WebGPURenderer(); } const alias=make; alias().render(scene,camera);",
    "let make=()=>new WebGPURenderer(); make=plugin; make().render(scene,camera);",
    "function make() { return new WebGPURenderer(); } make.call(null).render(scene,camera);",
    "function make() { return new WebGPURenderer(); } new make().render(scene,camera);",
    "function make() { if(flag) return make(); return new WebGPURenderer(); } make().render(scene,camera);",
    "function first() { return second(); } function second() { if(flag) return first(); return new WebGPURenderer(); } first().render(scene,camera);",
    "async function make() { return new WebGPURenderer(); } (await make()).render(scene,camera);",
    "function* make() { yield new WebGPURenderer(); } make().next();",
    "const make=()=>flag?new WebGPURenderer():unknown(); make().render(scene,camera);",
    "function make() { if(flag) return unknown(); return new WebGPURenderer(); } make().render(scene,camera);",
    "function make() { let value=new WebGPURenderer(); if(flag) value=unknown(); return value; } make().render(scene,camera);",
    "function make() { let value=new WebGPURenderer(); if(flag) value=unknown(); return value.domElement; } document.body.appendChild(make());",
    "function make(...args) { return new WebGPURenderer(); } make().render(scene,camera);",
    "function make() { return new WebGPURenderer(); } make(...args).render(scene,camera);",
  ]) assert.equal(decision(analyzeModuleAst(source, moduleUrl)).route, "exact-backend", source);
});

test("factory canvas provenance includes every call and preserves earlier aliases and opaque observations", () => {
  for (const setup of [
    "const canvas=host.canvas; make(canvas);",
    "const canvas=document.createElement('canvas'); plugin(canvas); make(canvas);",
    "const canvas=document.createElement('canvas'); const alias=canvas; plugin(canvas); make(alias);",
    "const canvas=document.createElement('canvas'); make(canvas); make(host.canvas);",
    "const canvas=document.createElement('canvas'); function getCanvas(){return canvas;} plugin(canvas); make(getCanvas());",
    "const canvas=document.createElement('canvas'); make(canvas); export {canvas};",
    "const canvas=document.createElement('canvas'); function nested(canvas){return make(canvas);} plugin(canvas); nested(canvas);",
  ]) {
    const source = "function make(canvas) { return new WebGPURenderer({canvas}); } " + setup;
    assert.equal(decision(analyzeModuleAst(source, moduleUrl)).route, "exact-backend", source);
  }
  const source = "const canvas=document.createElement('canvas'); const alias=canvas; plugin(canvas); new WebGPURenderer({canvas:alias});";
  assert.equal(decision(analyzeModuleAst(source, moduleUrl)).route, "exact-backend");
});
