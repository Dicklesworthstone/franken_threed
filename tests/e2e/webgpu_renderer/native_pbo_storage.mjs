// PBO fallback hints must preserve native compute-to-graphics storage.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_pbo_storage.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'pboStorageScenarios';
if (process.argv.length === 2) process.argv.push('native_pbo_cloth', 'native_pbo_instances');
await import('./sprites.mjs');
