// Reuse the pinned source/candidate pixel runner and its native GPU error scopes.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_storage_reads.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'storageReadScenarios';
if (process.argv.length === 2) process.argv.push('native_sampled_copy_data',
  'native_sampled_copy_image', 'native_sampled_copy_aliases', 'native_sampled_copy_bundle', 'native_sampled_copy_history',
  'native_sampled_copy_authored_mips', 'native_sampled_copy_generated_mips',
  'native_sampled_copy_storage', 'native_sampled_copy_storage_array_mip', 'native_sampled_copy_storage_volume_mip',
  'native_sampled_copy_target', 'native_sampled_copy_target_mip', 'native_sampled_copy_target_mrt_bundle');
await import('./sprites.mjs');
