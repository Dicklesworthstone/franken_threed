// Source/candidate pixels and GPU readbacks for synchronous object-hook compute.
// Reuse the unchanged oracle and GPU validation scopes; no performance claim.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_callback_compute.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'callbackComputeScenarios';
if (process.argv.length === 2) process.argv.push('native_callback_compute_before', 'native_callback_compute_interleaved');
await import('./sprites.mjs');
