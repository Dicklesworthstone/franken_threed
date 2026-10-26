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
  for (const [options, expected] of [
    ["{['forceWebGL']:true}", true],
    ["{[\`forceWebGL\`]:false}", false],
    ["{forceWebGL:false,...options}", "unresolved"],
    ["{...options,forceWebGL:false}", false],
    ["{forceWebGL:true,...{antialias:true}}", true],
    ["{forceWebGL:true,...{forceWebGL:false}}", false],
    ["{...{...options},forceWebGL:true}", true],
    ["{forceWebGL:false,[key]:1}", "unresolved"],
    ["{[key]:1,forceWebGL:false}", false],
    ["{[choose()?'forceWebGL':'alpha']:true}", "unresolved"],
    ["{forceWebGL:true,[choose()?'alpha':'antialias']:true}", true],
    ["(effect(),{forceWebGL:true})", true],
    ["options", "unresolved"],
    ["...options", "unresolved"],
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
    assert.equal(route.route, expected === true ? "exact-backend" : "general-webgpu", options);
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
