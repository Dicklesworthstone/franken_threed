/** Finite r186 GPUComputationRenderer lowering. The original addon continues
 * to own its public objects, synchronous API, dependency graph and ping-pong
 * targets. Only its backend capability guard needs a portable operation.
 */
import {createHash} from 'node:crypto';

const SOURCE = 'sha256:1a580add8398e969a63336ed99a8d3ef33705b059132c0ee9914c83b688177a6';
const PASS = 'sha256:444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5';

export function verifiedGpuComputationSources(modules, bases, passes) {
  const result = new Map();
  for (const [id, mod] of modules) {
    if (mod.content_hash !== SOURCE || mod.dynamic_imports?.length || mod.static_imports.length !== 2) continue;
    if (!mod.static_imports.every(request => request.specifier === 'three' ? bases.has(request.resolved_id) :
      request.specifier === '../postprocessing/Pass.js' &&
      modules.get(request.resolved_id)?.content_hash === PASS && passes.has(request.resolved_id))) continue;
    result.set(id, {name: 'GPUComputationRenderer.js', sourceHash: SOURCE});
  }
  return result;
}

export function transformThreeGpuComputation(code, record,
  runtime = new URL('./three_gpu_computation.mjs', import.meta.url).href) {
  if (!record) return null;
  if (record.sourceHash !== SOURCE || record.name !== 'GPUComputationRenderer.js' ||
      'sha256:' + createHash('sha256').update(code).digest('hex') !== SOURCE)
    throw new Error('Pinned GPUComputationRenderer source changed before capability lowering');
  // Exact source identity proves there is one guard and no conflicting binding.
  // A boolean operation avoids inventing a WebGL limit from GPUDevice limits.
  return `import {supportsThreeGpuComputation as __f3dSupportsGpuComputation} from ${JSON.stringify(runtime)};\n` +
    code.replace('renderer.capabilities.maxVertexTextures === 0', '!__f3dSupportsGpuComputation(renderer)');
}
