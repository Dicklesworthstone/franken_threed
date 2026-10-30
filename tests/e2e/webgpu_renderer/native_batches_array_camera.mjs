// Reuse the exact source/candidate offscreen comparison for per-view batches.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_batches.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativeBatchScenarios';
if (process.argv.length === 2) process.argv.push('native_batches_array_camera', 'native_batches_array_camera_msaa');
await import('./sprites.mjs');
