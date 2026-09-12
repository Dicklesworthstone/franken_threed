/**
 * @file route_bridge.test.mjs
 * Unit test suite for route_bridge.mjs (f3d-04.1 -> f3d-04.4 bridge).
 * Exercises static fact extraction and routing input preparation without bundling.
 */

import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

import {
  isInternalLibraryModule,
  extractGraphRoutingFacts,
  prepareRouteInputs,
  evaluateGraphRoutes,
  buildModuleGraph,
  analyzeModuleAst
} from "./index.mjs";
import { decideRendererRoute, ExecutionRoute, EscapeReason } from "../compat/index.mjs";

test("Positive: extractGraphRoutingFacts aggregates application-level escapes and ignores 2D canvas", () => {
  const fakeBundle = {
    packageRootUrl: "file:///three",
    package_root_url: "file:///three",
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

test("Positive & Unit: isInternalLibraryModule canonical identity checks, lookalike rejection, and URL normalization", () => {
  const internalWebgpuPath = path.resolve("upstream/three.js/build/three.webgpu.js");
  const internalWebgpuUrl = pathToFileURL(internalWebgpuPath).href;
  const internalModulePath = path.resolve("upstream/three.js/build/three.module.js");
  const internalCjsPath = path.resolve("upstream/three.js/build/three.cjs");
  const internalTslPath = path.resolve("upstream/three.js/build/three.tsl.js");
  const internalCorePath = path.resolve("upstream/three.js/build/three.core.js");
  const internalNodesPath = path.resolve("upstream/three.js/build/three.webgpu.nodes.js");

  // 1. Exact existing build filenames recognized under default pinned root
  assert.equal(isInternalLibraryModule(internalWebgpuPath), true, "Internal webgpu build file must be recognized");
  assert.equal(isInternalLibraryModule(internalWebgpuUrl), true, "Internal webgpu file URL must be recognized");
  assert.equal(isInternalLibraryModule(internalModulePath), true, "Internal module build file must be recognized");
  assert.equal(isInternalLibraryModule(internalCjsPath), true, "Internal cjs build file must be recognized");
  assert.equal(isInternalLibraryModule(internalTslPath), true, "Internal tsl build file must be recognized");
  assert.equal(isInternalLibraryModule(internalCorePath), true, "Internal core build file must be recognized");
  assert.equal(isInternalLibraryModule(internalNodesPath), true, "Internal webgpu.nodes build file must be recognized");

  // URL search query and hash fragments cleared and stripped
  assert.equal(isInternalLibraryModule(`${internalWebgpuUrl}?v=0.186.0#header`), true, "URL query/hash must be stripped");

  // 2. Broad src/ exemption removed: src/ files are NOT admitted build files
  const internalSrcPath = path.resolve("upstream/three.js/src/renderers/WebGLRenderer.js");
  assert.equal(isInternalLibraryModule(internalSrcPath), false, "Broad src/ exemption removed");

  // 3. Path traversal attack root/src/../../app.js must NOT be trusted
  const traversalPath = path.resolve("upstream/three.js/src/../../app.js");
  const defaultRoot = new URL("../../upstream/three.js/", import.meta.url).href;
  const traversalUrl = new URL("src/../../app.js", defaultRoot).href;
  assert.equal(isInternalLibraryModule(traversalPath), false, "root/src/../../app.js path must NOT be trusted");
  assert.equal(isInternalLibraryModule(traversalUrl), false, "root/src/../../app.js URL must NOT be trusted");
  assert.equal(isInternalLibraryModule("file:///three/src/../../app.js", "file:///three"), false, "fixture root/src/../../app.js must NOT be trusted");

  // 4. Invented legacy build names rejected
  assert.equal(isInternalLibraryModule(path.resolve("upstream/three.js/build/three.js")), false, "Legacy three.js rejected");
  assert.equal(isInternalLibraryModule(path.resolve("upstream/three.js/build/three.mjs")), false, "Legacy three.mjs rejected");
  assert.equal(isInternalLibraryModule(path.resolve("upstream/three.js/build/three.core.min.js")), false, "Legacy min.js rejected");

  // 5. Lookalike build files under upstream root rejected
  const lookalikeUnderUpstream = path.resolve("upstream/three.js/build/three.custom.js");
  assert.equal(isInternalLibraryModule(lookalikeUnderUpstream), false, "three.custom.js under upstream is not admitted");

  // 6. Addon files under upstream root are NOT internal library modules
  const addonPath = path.resolve("upstream/three.js/examples/jsm/postprocessing/EffectComposer.js");
  assert.equal(isInternalLibraryModule(addonPath), false, "Addon files are application/addon code, not internal library");

  // 7. External application lookalikes rejected
  assert.equal(isInternalLibraryModule("/app/build/three.custom.js"), false, "App lookalike /app/build/three.custom.js rejected");
  assert.equal(isInternalLibraryModule("file:///app/build/three.custom.js"), false, "App lookalike file URL rejected");
  assert.equal(isInternalLibraryModule("/app/build/three.webgpu.js"), false, "App file with standard name outside package root rejected");
  assert.equal(isInternalLibraryModule("file:///app/build/three.webgpu.js"), false, "App file with standard name outside package root rejected");

  // 8. Explicit packageRootUrl for fixtures
  assert.equal(isInternalLibraryModule("file:///three/build/three.webgpu.js", "file:///three"), true);
  assert.equal(isInternalLibraryModule("file:///three/build/three.webgpu.js?bundle=1#entry", "file:///three"), true);
  assert.equal(isInternalLibraryModule("file:///three/build/three.custom.js", "file:///three"), false);
  assert.equal(isInternalLibraryModule("file:///other/build/three.webgpu.js", "file:///three"), false);

  // 9. Invalid inputs
  assert.equal(isInternalLibraryModule(null), false);
  assert.equal(isInternalLibraryModule(""), false);
  assert.equal(isInternalLibraryModule(12345), false);
});

test("f3d-04.4 regression: analyzeModuleAst -> extractGraphRoutingFacts -> evaluateGraphRoutes isolates pinned internal modules and forces exact route for app lookalike", () => {
  // Application main entry constructing WebGPURenderer
  const mainCode = `
    import { WebGPURenderer } from "three/webgpu";
    const renderer = new WebGPURenderer({ forceWebGL: false });
  `;
  const mainAst = analyzeModuleAst(mainCode, "/app/main.js");

  // Pinned internal Three.js module containing library WebGL fallback
  const internalCode = `
    export function internalFallbackHelper(gl) {
      return gl.getParameter(0x1F00);
    }
  `;
  const internalPath = path.resolve("upstream/three.js/build/three.webgpu.js");
  const internalAst = analyzeModuleAst(internalCode, internalPath);
  assert.equal(internalAst.routing_facts.has_opaque_gl_escapes, true, "Internal module has getParameter");

  // App lookalike module attempting to disguise itself as a Three.js build file
  const lookalikeCode = `
    export function appLookalikeHelper(canvas) {
      const gl = canvas.getContext("webgl");
      return gl.getParameter(0x1F00);
    }
  `;
  const lookalikePath = "/app/build/three.custom.js";
  const lookalikeAst = analyzeModuleAst(lookalikeCode, lookalikePath);
  assert.equal(lookalikeAst.routing_facts.has_opaque_gl_escapes, true, "Lookalike module has getParameter");
  assert.equal(lookalikeAst.routing_facts.has_native_context_access, true, "Lookalike module has getContext");

  // Scenario 1: Bundle with main + pinned internal module only
  // Internal library escapes must be filtered out; does NOT force exact backend
  const bundleInternalOnly = {
    entry_path: "/app/main.js",
    modules: {
      "/app/main.js": {
        id: "/app/main.js",
        renderer_construction_sites: mainAst.renderer_construction_sites,
        routing_facts: mainAst.routing_facts,
      },
      [internalPath]: {
        id: internalPath,
        renderer_construction_sites: internalAst.renderer_construction_sites,
        routing_facts: internalAst.routing_facts,
      },
    },
  };

  const factsInternal = extractGraphRoutingFacts(bundleInternalOnly);
  assert.equal(factsInternal.hasOpaqueGLEscapes, false, "Internal library escapes must be filtered out");
  assert.equal(factsInternal.hasNativeContextAccess, false, "Internal library context access must be filtered out");
  assert.equal(factsInternal.escapes.length, 0, "No application escapes recorded from internal module");

  const decisionsInternal = evaluateGraphRoutes(bundleInternalOnly, decideRendererRoute, {
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(decisionsInternal.length, 1);
  assert.equal(decisionsInternal[0].decision.route, ExecutionRoute.SPECIALIZED_WEBGPU, "Must route to SPECIALIZED_WEBGPU");
  assert.ok(!decisionsInternal[0].decision.reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));
  assert.ok(!decisionsInternal[0].decision.reasons.includes(EscapeReason.NATIVE_CONTEXT_ACCESS));

  // Scenario 2: Bundle with main + app lookalike (/app/build/three.custom.js)
  // Lookalike is NOT trusted and its escapes force EXACT_BACKEND
  const bundleLookalike = {
    entry_path: "/app/main.js",
    modules: {
      "/app/main.js": {
        id: "/app/main.js",
        renderer_construction_sites: mainAst.renderer_construction_sites,
        routing_facts: mainAst.routing_facts,
      },
      [lookalikePath]: {
        id: lookalikePath,
        renderer_construction_sites: lookalikeAst.renderer_construction_sites,
        routing_facts: lookalikeAst.routing_facts,
      },
    },
  };

  const factsLookalike = extractGraphRoutingFacts(bundleLookalike);
  assert.equal(factsLookalike.hasOpaqueGLEscapes, true, "App lookalike escape must NOT be filtered out");
  assert.equal(factsLookalike.hasNativeContextAccess, true, "App lookalike context access must NOT be filtered out");
  assert.ok(factsLookalike.escapes.some(e => e.type === "opaque_gl_method_call"), "Must record opaque_gl_method_call");
  assert.ok(factsLookalike.escapes.some(e => e.type === "webgl_context_acquisition"), "Must record webgl_context_acquisition");

  const decisionsLookalike = evaluateGraphRoutes(bundleLookalike, decideRendererRoute, {
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(decisionsLookalike.length, 1);
  assert.equal(decisionsLookalike[0].decision.route, ExecutionRoute.EXACT_BACKEND, "Lookalike escape forces EXACT_BACKEND");
  assert.ok(decisionsLookalike[0].decision.reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));

  // Scenario 3: Combined bundle (main + pinned internal + app lookalike)
  // App lookalike forces exact route even when valid internal library modules are also in the graph
  const bundleCombined = {
    entry_path: "/app/main.js",
    modules: {
      "/app/main.js": {
        id: "/app/main.js",
        renderer_construction_sites: mainAst.renderer_construction_sites,
        routing_facts: mainAst.routing_facts,
      },
      [internalPath]: {
        id: internalPath,
        renderer_construction_sites: internalAst.renderer_construction_sites,
        routing_facts: internalAst.routing_facts,
      },
      [lookalikePath]: {
        id: lookalikePath,
        renderer_construction_sites: lookalikeAst.renderer_construction_sites,
        routing_facts: lookalikeAst.routing_facts,
      },
    },
  };

  const factsCombined = extractGraphRoutingFacts(bundleCombined);
  assert.equal(factsCombined.hasOpaqueGLEscapes, true);
  assert.equal(factsCombined.hasNativeContextAccess, true);
  // Escapes must only come from the lookalike, not the internal module
  assert.ok(factsCombined.escapes.every(e => e.moduleId === lookalikePath));

  const decisionsCombined = evaluateGraphRoutes(bundleCombined, decideRendererRoute, {
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(decisionsCombined.length, 1);
  assert.equal(decisionsCombined[0].decision.route, ExecutionRoute.EXACT_BACKEND);
  assert.ok(decisionsCombined[0].decision.reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));

  // Scenario 4: Traversal lookalike root/src/../../app.js with GL escape
  // Path traversal attempting to escape via src/ is NOT trusted and forces EXACT_BACKEND
  const traversalLookalikePath = path.resolve("upstream/three.js/src/../../app.js");
  const traversalCode = `
    export function traversalEscape(gl) {
      return gl.getParameter(0x1F00);
    }
  `;
  const traversalAst = analyzeModuleAst(traversalCode, traversalLookalikePath);
  assert.equal(traversalAst.routing_facts.has_opaque_gl_escapes, true);

  const bundleTraversal = {
    entry_path: "/app/main.js",
    modules: {
      "/app/main.js": {
        id: "/app/main.js",
        renderer_construction_sites: mainAst.renderer_construction_sites,
        routing_facts: mainAst.routing_facts,
      },
      [traversalLookalikePath]: {
        id: traversalLookalikePath,
        renderer_construction_sites: traversalAst.renderer_construction_sites,
        routing_facts: traversalAst.routing_facts,
      },
    },
  };

  const factsTraversal = extractGraphRoutingFacts(bundleTraversal);
  assert.equal(factsTraversal.hasOpaqueGLEscapes, true, "Traversal lookalike escapes must NOT be filtered out");

  const decisionsTraversal = evaluateGraphRoutes(bundleTraversal, decideRendererRoute, {
    hostCapabilities: { hasWebGPU: true, hasWebGL: true },
    specializationAvailable: true,
  });
  assert.equal(decisionsTraversal.length, 1);
  assert.equal(decisionsTraversal[0].decision.route, ExecutionRoute.EXACT_BACKEND, "Traversal lookalike forces EXACT_BACKEND");
  assert.ok(decisionsTraversal[0].decision.reasons.includes(EscapeReason.OPAQUE_GL_ESCAPE));
});

