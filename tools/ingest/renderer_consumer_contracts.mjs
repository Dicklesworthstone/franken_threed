/**
 * Source-verified ownership summary for retained OrbitControls (r186,
 * 148ef33ecb6d2502ff796d4554abd1549c95d519). The constructor keeps its canvas
 * in the returned instance. It is NOT a nonescaping-call exemption: callers
 * must track that instance, its domElement, inherited hooks and callbacks.
 *
 * Resolve the actual base implementation too. Identical addon bytes can import
 * a different Controls implementation through an import map. Paths and export
 * names alone never establish this contract. Build hashes are also recorded in
 * evidence/01.1/build_sha256.txt; the addon hash is from the pinned source tree.
 */
const ORBIT = 'sha256:3d79d07ecb686b4e5d93232eedab255331c1beef711e13164eaa1f68655a5f2b';
const CORE = 'sha256:9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const BUILDS = new Set([
  'sha256:9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea',
  'sha256:a78400fa1d359e81fcc12b84ed0a89a7fbecb4b8373e28ae8f9d2b11f6ea13bc',
]);
const BASE_EXPORTS = new Set(['Controls', 'EventDispatcher']);

/** Return contracts only for a completely analyzed graph and verified imports.
 * The analyzer's default remains conservative when this proof is unavailable.
 * A contract includes all application modules so another module cannot expose
 * or replace the protected superclass while the constructing module looks safe.
 */
export function verifiedRendererConsumerContracts(modules, {externalModules = []} = {}) {
  const contracts = new Map();
  if (externalModules.size || externalModules.length || [...modules.values()].some(mod =>
    mod.dynamic_imports?.some(request => request.unresolved || request.claims_closure !== true))) return contracts;

  const cores = new Set([...modules].filter(([, mod]) => mod.content_hash === CORE).map(([id]) => id));
  const bases = new Set(cores);
  for (const [id, mod] of modules) {
    if (BUILDS.has(mod.content_hash) && mod.static_imports.length > 0 &&
        mod.static_imports.every(request => cores.has(request.resolved_id))) bases.add(id);
  }
  const orbits = new Set();
  for (const [id, mod] of modules) {
    if (mod.content_hash === ORBIT && mod.static_imports.length === 1 &&
        bases.has(mod.static_imports[0].resolved_id)) orbits.add(id);
  }
  if (!orbits.size) return contracts;

  const role = (id, name) => orbits.has(id) && name === 'OrbitControls' ? 'orbit' :
    bases.has(id) && BASE_EXPORTS.has(name) ? 'base' :
    bases.has(id) && name === 'WebGPURenderer' ? 'webgpu' :
    bases.has(id) && name === 'WebGLRenderer' ? 'webgl' : null;
  for (const [id, mod] of modules) {
    // These exact implementations are the audited contract, not its callers.
    if (bases.has(id) || orbits.has(id)) continue;
    const bindings = new Map(), namespaces = new Map(), reexports = new Set();
    for (const request of mod.static_imports) {
      for (const binding of request.imported_bindings ?? []) {
        if (binding.type === 'namespace') {
          if (orbits.has(request.resolved_id)) namespaces.set(binding.local, 'orbit');
          else if (bases.has(request.resolved_id)) namespaces.set(binding.local, 'base');
        } else {
          const kind = role(request.resolved_id, binding.imported);
          if (kind) bindings.set(binding.local, kind);
        }
      }
    }
    for (const exp of mod.static_exports) {
      if (!exp.specifier) continue;
      for (const binding of exp.specifiers ?? []) {
        if (role(exp.resolved_id, binding.local)) reexports.add(`${exp.specifier}\0${binding.local}`);
      }
    }
    contracts.set(id, {bindings, namespaces, reexports});
  }
  return contracts;
}
