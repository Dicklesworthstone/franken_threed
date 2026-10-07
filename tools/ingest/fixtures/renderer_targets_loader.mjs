// Test-only module substitution at the facade's GPU/scene factory boundary.
// All target residency, frame routing and readback code remains the real code.
const mocked = new Set(['./gpu_canvas_renderer.mjs', './gpu_hdr_canvas.mjs', './three_scene.mjs',
  './three_program.mjs', './three_program_pmrem.mjs', './three_program_shadows.mjs']);
export function resolve(specifier, context, nextResolve) {
  if (context.parentURL?.endsWith('/three_renderer.mjs') && mocked.has(specifier))
    return {url: new URL('./renderer_targets_fixture.mjs', import.meta.url).href, shortCircuit: true};
  return nextResolve(specifier, context);
}
