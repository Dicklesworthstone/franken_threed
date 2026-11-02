/** Compiler-private portability guard for the pinned GPUComputationRenderer.
 * Native renderers support vertex texture inputs on every admitted WebGPU
 * device. Keep that fact separate from public, source-observable GL limits.
 */
const nativeRenderers = new WeakSet();

export function registerThreeGpuComputation(renderer) {
  nativeRenderers.add(renderer);
}

export function supportsThreeGpuComputation(renderer) {
  // The source guard tests strict numeric zero, not truthiness or positivity.
  // Preserve its access count and thrown errors for actual upstream renderers.
  return nativeRenderers.has(renderer) || renderer.capabilities.maxVertexTextures !== 0;
}
