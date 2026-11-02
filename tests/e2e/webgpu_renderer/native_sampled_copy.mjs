// Reuse the pinned source/candidate pixel runner and its native GPU error scopes.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_storage_reads.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'storageReadScenarios';
if (process.argv.length === 2) process.argv.push('native_sampled_copy_data',
  'native_sampled_copy_image', 'native_sampled_copy_aliases', 'native_sampled_copy_bundle', 'native_sampled_copy_history');
await import('./sprites.mjs');
