// Reuse the source/candidate offscreen comparison and its unchanged oracle.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_storage_matrices.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'storageMatrixScenarios';
if (process.argv.length === 2) process.argv.push('native_storage_mat2', 'native_storage_mat3');
await import('./sprites.mjs');
