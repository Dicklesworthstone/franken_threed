// Source Basic/Lambert/Phong environment lighting through the native renderer.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_basic_environment.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'basicEnvironmentScenarios';
if (process.argv.length === 2) process.argv.push('native_basic_environment_cube',
  'native_basic_environment_panorama', 'native_basic_environment_skinning');
await import('./sprites.mjs');
