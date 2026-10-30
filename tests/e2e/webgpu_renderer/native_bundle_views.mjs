// Source BundleGroup recording with viewport-less ArrayCamera children.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_bundle_views.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativeBundleViewScenarios';
if (process.argv.length === 2) process.argv.push('native_bundle_array_camera', 'native_bundle_batch_array_camera');
await import('./sprites.mjs');
