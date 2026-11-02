// Reuse the pinned source/candidate runner, unchanged pixel oracle and GPU scopes.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_atomic_structs.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'atomicStructScenarios';
if (process.argv.length === 2) process.argv.push('native_atomic_structs', 'native_atomic_structs_float_backing');
await import('./sprites.mjs');
