// Exercise the real scene bridge; replace only its GPU/deformation dependencies.
const dependencies = new Set(['./animation_render.mjs', './gpu_buffer_geometry.mjs', './three_textures.mjs', './three_deformation.mjs']);
export function resolve(specifier, context, nextResolve) {
  if (context.parentURL?.endsWith('/three_scene.mjs') && dependencies.has(specifier))
    return {url: new URL('./scene_program_fixture.mjs', import.meta.url).href, shortCircuit: true};
  return nextResolve(specifier, context);
}
