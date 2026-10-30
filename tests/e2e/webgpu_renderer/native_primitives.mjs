// Use the existing exact source/candidate offscreen comparison for native
// thin/wide lines, GPU-produced point clouds, point sprites and indirect geometry.
// Pass individual case names to bound software-GPU memory use.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_primitives.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativePrimitiveScenarios';
if (process.argv.length === 2) process.argv.push('native_tsl_lines', 'native_tsl_point_storage',
  'native_tsl_point_sprites', 'native_tsl_point_sprites_orthographic',
  'native_indirect_compute', 'native_indirect_host_upload',
  'native_indirect_vertex_alias', 'native_indirect_vertex_alias_bundle',
  'native_wide_lines', 'native_wide_lines_msaa');
await import('./sprites.mjs');
