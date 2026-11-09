/**
 * Source-verified ownership summaries for retained camera controls, inspectors
 * and standard postprocessing passes (r186, 148ef33ecb6d2502ff796d4554abd1549c95d519).
 * These objects retain their canvas or renderer. Callers must track the
 * returned instances, inherited methods and callbacks; this is not a generic
 * exemption for imported calls.
 *
 * Resolve the actual base implementation too. Identical addon bytes can import
 * a different Controls implementation through an import map. Paths and export
 * names alone never establish this contract. Build hashes are also recorded in
 * evidence/01.1/build_sha256.txt; the addon hash is from the pinned source tree.
 */
import {verifiedPostProcessingSources} from './postprocessing_contracts.mjs';
import {verifiedGpuComputationSources} from './gpu_computation_contracts.mjs';

const ORBIT = 'sha256:3d79d07ecb686b4e5d93232eedab255331c1beef711e13164eaa1f68655a5f2b';
const FIRST_PERSON = 'sha256:4f6c4bf60c8956a1201c966de1592f4f035be47b4f6fe97879a782898945992c';
const MAP = 'sha256:5a686152c498914389d1503b60121f484d06f15cb4f7405c15fb6b8ddcc1bdb6';
const CONTROL_SOURCES = new Map([
  [ORBIT, {name: 'OrbitControls', role: 'orbit'}],
  [FIRST_PERSON, {name: 'FirstPersonControls', role: 'first-person'}],
  ['sha256:cce95baddf589c16fea901b70f2038eca4557357937e1b03f8c76fe7ffe7457f', {name: 'TrackballControls', role: 'trackball'}],
  ['sha256:8063858389f2ccd5cd6f0968562752dbab4b21a03c7a6e1dcbb5cc746fc06373', {name: 'FlyControls', role: 'fly'}],
]);
const CORE = 'sha256:9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6';
const WEBGPU = 'sha256:a78400fa1d359e81fcc12b84ed0a89a7fbecb4b8373e28ae8f9d2b11f6ea13bc';
const BUILDS = new Set([
  'sha256:9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea',
  WEBGPU,
]);
const BASE_EXPORTS = new Set(['Controls', 'EventDispatcher', 'Renderer', 'CanvasTarget', 'InspectorBase']);
const RENDERER_INSPECTOR = 'sha256:4252223601805bb32df911328e1626c3161284b5211d95ad399865d880dbeeae';
const TSL = 'sha256:6e6b92d8ce467818944f73815a96abadc6392fb34de577d479a23810b6aeeed0';

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
  const controls = new Map();
  for (const [id, mod] of modules) {
    const source = CONTROL_SOURCES.get(mod.content_hash);
    if (source && mod.static_imports.length === 1 && bases.has(mod.static_imports[0].resolved_id))
      controls.set(id, source);
  }
  // MapControls inherits the whole Orbit implementation. Its own fingerprint
  // cannot establish a contract when either resolved dependency was replaced.
  for (const [id, mod] of modules) {
    if (mod.content_hash === MAP && mod.static_imports.length === 2 &&
        mod.static_imports.every(request => request.specifier === 'three' ? bases.has(request.resolved_id) :
          request.specifier === './OrbitControls.js' && controls.get(request.resolved_id)?.role === 'orbit'))
      controls.set(id, {name: 'MapControls', role: 'map'});
  }
  const passes = verifiedPostProcessingSources(modules, bases);
  contracts.postprocessingSources = passes;
  const computations = verifiedGpuComputationSources(modules, bases, passes);
  contracts.gpuComputationSources = computations;
  const inspectors = new Set();
  for (const [id, mod] of modules) {
    if (mod.content_hash !== RENDERER_INSPECTOR || mod.static_imports.length !== 2) continue;
    const renderer = mod.static_imports.find(request => request.specifier === 'three/webgpu')?.resolved_id;
    const tsl = modules.get(mod.static_imports.find(request => request.specifier === 'three/tsl')?.resolved_id);
    // The inspector and its TSL helpers must share the same pinned renderer
    // instance and core. A renamed addon is fine; redirected dependencies are not.
    if (bases.has(renderer) && modules.get(renderer)?.content_hash === WEBGPU && tsl?.content_hash === TSL && tsl.static_imports.length === 1 &&
        tsl.static_imports[0].resolved_id === renderer) inspectors.add(id);
  }
  if (!controls.size && !passes.size && !computations.size && !inspectors.size) return contracts;

  const role = (id, name) => inspectors.has(id) && name === 'RendererInspector' ? 'renderer-inspector' :
    computations.has(id) && name === 'GPUComputationRenderer' ? 'gpu-computation' :
    passes.get(id)?.exports[name] ?? (controls.get(id)?.name === name ? controls.get(id).role :
    bases.has(id) && BASE_EXPORTS.has(name) ? 'base' :
    bases.has(id) && name === 'WebGPURenderer' ? 'webgpu' :
    bases.has(id) && name === 'WebGLRenderer' ? 'webgl' : null);
  for (const [id, mod] of modules) {
    // These exact implementations are the audited contract, not its callers.
    if (bases.has(id) || controls.has(id) || passes.has(id) || computations.has(id) || inspectors.has(id)) continue;
    const bindings = new Map(), namespaces = new Map(), reexports = new Set();
    for (const request of mod.static_imports) {
      for (const binding of request.imported_bindings ?? []) {
        if (binding.type === 'namespace') {
          if (controls.has(request.resolved_id)) namespaces.set(binding.local, controls.get(request.resolved_id).role);
          else if (computations.has(request.resolved_id)) namespaces.set(binding.local, 'gpu-computation');
          else if (inspectors.has(request.resolved_id)) namespaces.set(binding.local, 'renderer-inspector');
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
    contracts.set(id, {bindings, namespaces, reexports, inspectors: inspectors.size > 0});
  }
  return contracts;
}
