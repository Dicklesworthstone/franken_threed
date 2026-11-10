/** Close renderer ownership across ordinary static ESM helpers using the same
 * lexical proof as local helpers. Rollup only links a private analysis copy;
 * it never executes source or changes the emitted application. Entry exports,
 * live binding writes and opaque consumers still face the existing analyzer.
 */
import {rollup} from 'rollup';
import {createHash} from 'node:crypto';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {isInternalLibraryModule} from './route_bridge.mjs';
import {verifiedRendererConsumerContracts} from './renderer_consumer_contracts.mjs';

// Immutable r186 build bytes, also recorded in evidence/01.1/build_sha256.txt.
// A path or a matching export name cannot establish a library boundary.
const BUILDS = new Map([
  ['three.core.js', '9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6'],
  ['three.module.js', '9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea'],
  ['three.webgpu.js', 'a78400fa1d359e81fcc12b84ed0a89a7fbecb4b8373e28ae8f9d2b11f6ea13bc'],
  ['three.tsl.js', '6e6b92d8ce467818944f73815a96abadc6392fb34de577d479a23810b6aeeed0'],
  ['three.webgpu.nodes.js', 'a6a3a3b59fff0496783cd77162f1733141af31d4b596a5a77004b4f1790d53f6'],
]);
const constructionProfile = sites => sites.map(site => JSON.stringify([
  site.constructor_name, site.force_webgl, site.force_webgl_unresolved, Boolean(site.canvas_option),
])).sort();

/** Recompute only resource-escape facts after proving the complete static
 * application region. Unknown code, dynamic entry points and proof limits keep
 * their original per-module diagnostics and exact construction route.
 */
export async function closeRendererModuleOwnership(modules, sourceInputs, roots,
  {packageRootUrl, externalModules = []} = {}) {
  if (roots.length !== 1 || externalModules.size || externalModules.length) return false;
  const library = new Set(), application = new Map();
  const consumers = verifiedRendererConsumerContracts(modules, {externalModules}).retainedSources ?? new Map();
  let sourceBytes = 0, hasBoundary = false;
  for (const [id, mod] of modules) {
    if (isInternalLibraryModule(id, packageRootUrl)) {
      const hash = BUILDS.get(new URL(id).pathname.split('/').at(-1));
      if (!hash || mod.content_hash !== 'sha256:' + hash) return false;
      library.add(id);
      continue;
    }
    if (consumers.has(id)) continue;
    const input = sourceInputs.get(id), facts = mod.routing_facts;
    if (!input || mod.content_hash !== 'sha256:' + createHash('sha256').update(input.code).digest('hex') ||
        mod.analysis_complete === false || mod.module_type !== 'javascript' ||
        mod.dynamic_imports.length || mod.postprocessing_transform || mod.gpu_computation_transform ||
        facts.has_native_context_access || facts.has_unresolved_context_access ||
        facts.escapes.some(escape => escape.type !== 'renderer_resource_escape')) return false;
    hasBoundary ||= facts.has_opaque_gl_escapes;
    sourceBytes += Buffer.byteLength(input.code);
    if (sourceBytes > 2 * 1024 * 1024 || application.size >= 128) return false;
    application.set(id, mod);
  }
  if (!hasBoundary || !library.size || application.size < 2 || !application.has(roots[0])) return false;
  // Every externalized library's own dependencies must retain their admitted
  // identities. Import-map redirects of a build's core invalidate this proof.
  for (const id of library) {
    const mod = modules.get(id);
    if (mod.dynamic_imports.length || mod.static_imports.some(request => !library.has(request.resolved_id) ||
        new URL(request.specifier, id).href !== request.resolved_id)) return false;
  }
  // A source fingerprint does not close an imported dependency. Consumer
  // contracts verify the exact addon and resolved bases; every dependency must
  // also be inside this proof's admitted boundary, never arbitrary app code.
  for (const id of consumers.keys()) {
    const mod = modules.get(id), input = sourceInputs.get(id);
    if (!input || mod.content_hash !== 'sha256:' + createHash('sha256').update(input.code).digest('hex') ||
        mod.analysis_complete === false || mod.module_type !== 'javascript' ||
        mod.dynamic_imports.length || mod.postprocessing_transform || mod.gpu_computation_transform ||
        mod.static_imports.some(request => !library.has(request.resolved_id) && !consumers.has(request.resolved_id))) return false;
  }
  const boundary = id => library.has(id) || consumers.has(id);
  const originalSites = [...application.values()].flatMap(mod => mod.renderer_construction_sites);
  if (!originalSites.length) return false;
  const originalProfile = constructionProfile(originalSites);
  let linked;
  try {
    linked = await rollup({
      input: roots[0], treeshake: false, preserveEntrySignatures: 'strict',
      onwarn(warning) {
        // Eval, cycles, missing exports and namespace conflicts need separate
        // binding proofs. None can establish ownership in this bounded slice.
        throw new Error(warning.message);
      },
      plugins: [{
        name: 'f3d-renderer-ownership',
        resolveId(specifier, importer) {
          if (!importer && specifier === roots[0]) return specifier;
          const requests = application.get(importer)?.static_imports
            .filter(request => request.specifier === specifier) ?? [];
          const target = requests[0]?.resolved_id;
          if (!target || requests.some(request => request.resolved_id !== target || request.module_type !== 'javascript') ||
              !boundary(target) && !application.has(target)) throw new Error('Unclosed renderer analysis import');
          return {id: target, external: boundary(target)};
        },
        load(id) {
          if (!application.has(id)) throw new Error('Unclosed renderer analysis module');
          return sourceInputs.get(id).code;
        },
      }],
    });
    // Keep exact external IDs in this private copy so named and namespace
    // imports cannot acquire another module's verified consumer role.
    const generated = await linked.generate({format: 'es', compact: false, paths: id => id});
    if (generated.output.length !== 1 || generated.output[0].type !== 'chunk') return false;
    const chunk = generated.output[0];
    if (Object.keys(chunk.modules).length !== application.size ||
        [...application.keys()].some(id => !Object.hasOwn(chunk.modules, id)) ||
        Buffer.byteLength(chunk.code) > 2 * 1024 * 1024) return false;
    const imports = analyzeModuleAst(chunk.code, roots[0]);
    const contract = {bindings: new Map(), namespaces: new Map(), reexports: new Set(),
      inspectors: [...consumers.values()].some(source => source.role === 'renderer-inspector')};
    const role = (id, name) => {
      const source = consumers.get(id);
      if (source) return source.name === name ? source.role : null;
      if (!library.has(id)) return null;
      return name === 'WebGPURenderer' ? 'webgpu' : name === 'WebGLRenderer' ? 'webgl' :
        ['Controls', 'EventDispatcher', 'Renderer', 'CanvasTarget', 'InspectorBase'].includes(name) ? 'base' : null;
    };
    for (const request of imports.staticImports) {
      if (!boundary(request.specifier)) return false;
      for (const binding of request.imported_bindings) {
        if (binding.type === 'namespace') contract.namespaces.set(binding.local, consumers.get(request.specifier)?.role ?? 'base');
        else if (role(request.specifier, binding.imported)) contract.bindings.set(binding.local, role(request.specifier, binding.imported));
      }
    }
    for (const exp of imports.staticExports) if (exp.specifier) {
      if (!boundary(exp.specifier)) return false;
      for (const binding of exp.specifiers ?? []) if (role(exp.specifier, binding.local))
        contract.reexports.add(`${exp.specifier}\0${binding.local}`);
    }
    // Graph-mode checks also cover indirect canvas observations through DOM
    // callbacks and imported constructor identities. The weaker standalone
    // analyzer mode must never discharge an imported resource boundary.
    const analysis = analyzeModuleAst(chunk.code, roots[0], {}, contract);
    const facts = analysis.routing_facts;
    if (facts.has_opaque_gl_escapes || facts.has_native_context_access || facts.has_unresolved_context_access ||
        facts.escapes.length || analysis.dynamicImports.length ||
        JSON.stringify(constructionProfile(analysis.renderer_construction_sites)) !== JSON.stringify(originalProfile)) return false;
    // Keep the original construction sites and spans: they also select exact
    // source calls in mixed-backend output. No generated offset enters emission.
    for (const mod of application.values()) {
      mod.routing_facts = mod.routingFacts = {...mod.routing_facts,
        has_opaque_gl_escapes: false, hasOpaqueGLEscapes: false, escapes: []};
    }
    return true;
  } catch {
    // This is an optimization proof, not a replacement parser/build path.
    // Any unsupported linking shape keeps the functioning source component.
    return false;
  } finally { await linked?.close(); }
}
