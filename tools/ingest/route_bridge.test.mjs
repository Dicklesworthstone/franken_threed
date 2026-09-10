/**
 * @file route_bridge.test.mjs
 * Unit test suite for route_bridge.mjs (f3d-04.1 -> f3d-04.4 bridge).
 * Exercises static fact extraction and routing input preparation without bundling.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  extractGraphRoutingFacts,
  prepareRouteInputs,
  evaluateGraphRoutes,
  buildModuleGraph,
  analyzeModuleAst
} from "./index.mjs";
import { decideRendererRoute, ExecutionRoute, EscapeReason } from "../compat/index.mjs";

test("Positive: extractGraphRoutingFacts aggregates application-level escapes and ignores 2D canvas", () => {
  const fakeBundle = {
    modules: {
      "file:///app/main.js": {
        id: "file:///app/main.js",
        renderer_construction_sites: [
          {
            constructor_name: "WebGPURenderer",
            has_force_webgl: false,
            canvas_option: "c1",
            source_span: "file:///app/main.js:10:5"
          }
        ],
        routing_facts: {
          has_opaque_gl_escapes: false,
          has_native_context_access: false,
          escapes: []
        }
      },
      "file:///app/stats_hud.js": {
        id: "file:///app/stats_hud.js",
        routing_facts: {
          has_opaque_gl_escapes: false,
          has_native_context_access: false, // 2D canvas getContext does not trigger native context access
          escapes: []
        }
      },
      "file:///three/build/three.webgpu.js": {
        id: "file:///three/build/three.webgpu.js",
        routing_facts: {
          has_opaque_gl_escapes: true, // internal Three.js fallback
          has_native_context_access: true,
          escapes: [{ type: "opaque_gl_method_call", method: "getParameter" }]
        }
      }
    }
  };

  const facts = extractGraphRoutingFacts(fakeBundle);
  assert.equal(facts.hasOpaqueGLEscapes, false, "Internal library escapes must be filtered out by default");
  assert.equal(facts.hasNativeContextAccess, false);
  assert.equal(facts.constructionSites.length, 1);
  assert.equal(facts.constructionSites[0].constructorName, "WebGPURenderer");
});

test("Positive: Application-level GL escapes propagate across graph to all construction sites", () => {
  const fakeBundle = {
    modules: {
      "file:///app/main.js": {
        id: "file:///app/main.js",
        renderer_construction_sites: [
          {
            constructor_name: "WebGPURenderer",
            has_force_webgl: false,
            canvas_option: null,
            source_span: "file:///app/main.js:5:1"
          }
        ],
        routing_facts: { has_opaque_gl_escapes: false, has_native_context_access: false }
      },
      "file:///app/custom_shader.js": {
        id: "file:///app/custom_shader.js",
        routing_facts: {
          has_opaque_gl_escapes: true,
          has_native_context_access: false,
          escapes: [{ type: "opaque_gl_method_call", method: "getExtension" }]
        }
      }
    }
  };

  const facts = extractGraphRoutingFacts(fakeBundle);
  assert.equal(facts.hasOpaqueGLEscapes, true);
  assert.equal(facts.constructionSites[0].analysis.hasOpaqueGLEscapes, true);

  const decisions = evaluateGraphRoutes(fakeBundle, decideRendererRoute, {
    hostCapabilities: { hasWebGPU: true, hasWebGL: true }
  });
  assert.equal(decisions[0].decision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(decisions[0].decision.reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));
});

test("Positive: Fallback input generated when no renderer construction site exists statically", () => {
  const fakeBundle = {
    entry_path: "file:///app/headless.js",
    modules: {
      "file:///app/headless.js": {
        id: "file:///app/headless.js",
        routing_facts: { has_opaque_gl_escapes: false, has_native_context_access: false }
      }
    }
  };

  const inputs = prepareRouteInputs(fakeBundle, { constructorName: "WebGPURenderer" });
  assert.equal(inputs.length, 1);
  assert.equal(inputs[0].constructorName, "WebGPURenderer");
  assert.equal(inputs[0].analysis.hasOpaqueGLEscapes, false);
});

test("Positive: Real H1 and H2 graph route evaluations execute without bundling", async () => {
  // Ingest H1 (AST parse + import map resolution only, no Rollup)
  const h1 = await buildModuleGraph("upstream/three.js/examples/webgpu_performance_renderbundle.html");
  const h1Inputs = prepareRouteInputs(h1);
  assert.equal(h1Inputs.length, 1);
  assert.equal(h1Inputs[0].constructorName, "WebGPURenderer");
  assert.equal(h1Inputs[0].options.forceWebGL, "unresolved", "H1 !api.webgpu expression must be unresolved");
  assert.equal(h1Inputs[0].options.forceWebGLUnresolved, true);

  // Ingest H2
  const h2 = await buildModuleGraph("upstream/three.js/examples/webgl_marchingcubes.html");
  const h2Decision = evaluateGraphRoutes(h2, decideRendererRoute, {
    hostCapabilities: { hasWebGPU: true, hasWebGL: true }
  });
  assert.equal(h2Decision[0].decision.route, ExecutionRoute.EXACT_BACKEND);
  assert.equal(h2Decision[0].constructorName, "WebGLRenderer");
  assert.ok(h2Decision[0].decision.reasons.includes(EscapeReason.EXPLICIT_SOURCE_SELECTION));
});

test("Regression (a): getContext with non-literal argument is classified as unresolved native-context access with source span, never non-native", () => {
  const codeVar = `
    const ctxType = "webgl";
    const gl = canvas.getContext(ctxType);
  `;
  const resultVar = analyzeModuleAst(codeVar, "test_var.js");
  assert.equal(resultVar.routingFacts.hasNativeContextAccess, true, "Must be classified as native-context access");
  assert.equal(resultVar.routingFacts.hasUnresolvedContextAccess, true, "Must be classified as unresolved context access");
  assert.ok(resultVar.routingFacts.escapes.length > 0);
  const escapeVar = resultVar.routingFacts.escapes.find(e => e.type === "unresolved_native_context_access");
  assert.ok(escapeVar, "Must record an unresolved_native_context_access escape");
  assert.equal(escapeVar.classification, "nonliteral");
  assert.equal(escapeVar.unresolved, true);
  assert.ok(escapeVar.sourceSpan, "Must include a source span");

  const codeTmpl = `
    const gl = canvas.getContext(\`\${mode}\`);
  `;
  const resultTmpl = analyzeModuleAst(codeTmpl, "test_tmpl.js");
  assert.equal(resultTmpl.routingFacts.hasNativeContextAccess, true);
  assert.equal(resultTmpl.routingFacts.hasUnresolvedContextAccess, true);
  const escapeTmpl = resultTmpl.routingFacts.escapes.find(e => e.type === "unresolved_native_context_access");
  assert.ok(escapeTmpl);
  assert.equal(escapeTmpl.classification, "nonliteral");
  assert.equal(escapeTmpl.unresolved, true);
});

test("Regression (b): forceWebGL with non-literal value is classified as unresolved, not false", () => {
  const codeUnary = `
    const renderer = new THREE.WebGPURenderer({ forceWebGL: !api.webgpu });
  `;
  const resultUnary = analyzeModuleAst(codeUnary, "test_unary.js");
  const siteUnary = resultUnary.rendererConstructionSites[0];
  assert.equal(siteUnary.forceWebGL, "unresolved", "Must be 'unresolved', never false");
  assert.equal(siteUnary.hasForceWebGL, "unresolved", "hasForceWebGL must be 'unresolved'");
  assert.equal(siteUnary.forceWebGLUnresolved, true, "forceWebGLUnresolved must be true");

  const codeVar = `
    const useWebGL = true;
    const renderer = new THREE.WebGPURenderer({ forceWebGL: useWebGL });
  `;
  const resultVar = analyzeModuleAst(codeVar, "test_var.js");
  const siteVar = resultVar.rendererConstructionSites[0];
  assert.equal(siteVar.forceWebGL, "unresolved", "Must be 'unresolved', never false");
  assert.equal(siteVar.hasForceWebGL, "unresolved");
  assert.equal(siteVar.forceWebGLUnresolved, true);

  // Literal false must remain false
  const codeFalse = `
    const renderer = new THREE.WebGPURenderer({ forceWebGL: false });
  `;
  const resultFalse = analyzeModuleAst(codeFalse, "test_false.js");
  const siteFalse = resultFalse.rendererConstructionSites[0];
  assert.equal(siteFalse.forceWebGL, false);
  assert.equal(siteFalse.hasForceWebGL, false);
  assert.equal(siteFalse.forceWebGLUnresolved, false);

  // Literal true must remain true
  const codeTrue = `
    const renderer = new THREE.WebGPURenderer({ forceWebGL: true });
  `;
  const resultTrue = analyzeModuleAst(codeTrue, "test_true.js");
  const siteTrue = resultTrue.rendererConstructionSites[0];
  assert.equal(siteTrue.forceWebGL, true);
  assert.equal(siteTrue.hasForceWebGL, true);
  assert.equal(siteTrue.forceWebGLUnresolved, false);
});

test("Regression (c): JSON bundle serialization round-trip preserves unresolved context access, escapes, forceWebGL facts, and summary counters", async () => {
  // 1. Ingest real H1 and verify JSON serialization round-trip
  const h1 = await buildModuleGraph("upstream/three.js/examples/webgpu_performance_renderbundle.html");
  const jsonStrH1 = JSON.stringify(h1, null, 2);
  const roundTrippedH1 = JSON.parse(jsonStrH1);

  assert.equal(roundTrippedH1.schema_version, "1.0.0");
  assert.equal(roundTrippedH1.summary.total_unresolved_force_webgl, 1);
  assert.equal(roundTrippedH1.summary.totalUnresolvedForceWebGL, 1);

  const h1Facts = extractGraphRoutingFacts(roundTrippedH1);
  assert.equal(h1Facts.constructionSites.length, 1);
  const h1Site = h1Facts.constructionSites[0];
  assert.equal(h1Site.constructorName, "WebGPURenderer");
  assert.equal(h1Site.options.forceWebGL, "unresolved");
  assert.equal(h1Site.options.force_webgl, "unresolved");
  assert.equal(h1Site.options.forceWebGLUnresolved, true);
  assert.equal(h1Site.options.force_webgl_unresolved, true);
  assert.equal(h1Site.analysis.hasUnresolvedContextAccess, false);

  // 2. Verify synthetic module bundle containing both non-literal getContext and non-literal forceWebGL
  const testCode = `
    const ctxType = "webgl";
    const gl = canvas.getContext(ctxType);
    const renderer = new THREE.WebGPURenderer({ forceWebGL: !api.webgpu, canvas });
  `;
  const analysis = analyzeModuleAst(testCode, "synthetic_unresolved.js");

  const syntheticBundle = {
    schema_version: "1.0.0",
    entry_type: "module",
    entry_path: "/app/main.js",
    root_entries: ["file:///app/main.js"],
    import_map: null,
    modules: {
      "file:///app/main.js": {
        id: "file:///app/main.js",
        content_hash: "sha256:test1234",
        is_inline: false,
        source_path: "/app/main.js",
        duplicate_content_with: [],
        static_imports: [],
        static_exports: [],
        dynamic_imports: [],
        asset_references: [],
        classes: [],
        prototype_writes: [],
        has_top_level_side_effects: true,
        has_live_bindings: false,
        mutable_exported_bindings: [],
        renderer_construction_sites: analysis.renderer_construction_sites,
        rendererConstructionSites: analysis.renderer_construction_sites,
        routing_facts: analysis.routing_facts,
        routingFacts: analysis.routing_facts
      }
    },
    cycles: [],
    summary: {
      total_modules: 1,
      total_static_imports: 0,
      total_dynamic_imports: 0,
      unresolved_dynamic_imports: 0,
      cycles_count: 0,
      identical_content_pairs: 0,
      total_renderer_construction_sites: 1,
      total_unresolved_native_context_access: 1,
      totalUnresolvedNativeContextAccess: 1,
      total_unresolved_force_webgl: 1,
      totalUnresolvedForceWebGL: 1
    }
  };

  const jsonStr = JSON.stringify(syntheticBundle, null, 2);
  const roundTripped = JSON.parse(jsonStr);

  // Assert schema version and summary counts survived serialization
  assert.equal(roundTripped.schema_version, "1.0.0");
  assert.equal(roundTripped.summary.total_unresolved_native_context_access, 1);
  assert.equal(roundTripped.summary.totalUnresolvedNativeContextAccess, 1);
  assert.equal(roundTripped.summary.total_unresolved_force_webgl, 1);
  assert.equal(roundTripped.summary.totalUnresolvedForceWebGL, 1);

  // Assert node-level routing facts survived serialization (both camelCase and snake_case)
  const node = roundTripped.modules["file:///app/main.js"];
  assert.equal(node.routing_facts.hasUnresolvedContextAccess, true);
  assert.equal(node.routing_facts.has_unresolved_context_access, true);
  assert.equal(node.routingFacts.hasUnresolvedContextAccess, true);
  assert.equal(node.routing_facts.hasNativeContextAccess, true);
  assert.equal(node.routing_facts.has_native_context_access, true);

  // Assert unresolved_native_context_access escape survived with span and metadata
  const unresEscape = node.routing_facts.escapes.find(e => e.type === "unresolved_native_context_access");
  assert.ok(unresEscape, "Must preserve unresolved_native_context_access escape in JSON");
  assert.equal(unresEscape.classification, "nonliteral");
  assert.equal(unresEscape.unresolved, true);
  assert.ok(unresEscape.source_span || unresEscape.sourceSpan);

  // Assert renderer construction site options survived serialization
  const site = node.renderer_construction_sites[0];
  assert.equal(site.forceWebGL, "unresolved");
  assert.equal(site.force_webgl, "unresolved");
  assert.equal(site.hasForceWebGL, "unresolved");
  assert.equal(site.has_force_webgl, "unresolved");
  assert.equal(site.forceWebGLUnresolved, true);
  assert.equal(site.force_webgl_unresolved, true);

  // Assert extracted graph routing facts from round-tripped bundle
  const extractedFacts = extractGraphRoutingFacts(roundTripped);
  assert.equal(extractedFacts.hasUnresolvedContextAccess, true);
  assert.equal(extractedFacts.has_unresolved_context_access, true);
  assert.equal(extractedFacts.hasNativeContextAccess, true);
  assert.ok(extractedFacts.escapes.some(e => e.type === "unresolved_native_context_access"));

  assert.equal(extractedFacts.constructionSites.length, 1);
  const extractedSite = extractedFacts.constructionSites[0];
  assert.equal(extractedSite.options.forceWebGL, "unresolved");
  assert.equal(extractedSite.options.force_webgl, "unresolved");
  assert.equal(extractedSite.options.forceWebGLUnresolved, true);
  assert.equal(extractedSite.options.force_webgl_unresolved, true);
  assert.equal(extractedSite.analysis.hasUnresolvedContextAccess, true);
  assert.equal(extractedSite.analysis.has_unresolved_context_access, true);
});



