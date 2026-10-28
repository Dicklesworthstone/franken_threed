// Use the existing exact source/candidate offscreen comparison for native
// lines, GPU-produced point clouds and pixel-sized instanced point sprites.
// Pass individual case names to bound software-GPU memory use.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_primitives.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativePrimitiveScenarios';
if (process.argv.length === 2) process.argv.push('native_tsl_lines', 'native_tsl_point_storage',
  'native_tsl_point_sprites', 'native_tsl_point_sprites_orthographic');
await import('./sprites.mjs');
