// Exact pixels also encode query observations made by source draw callbacks.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_occlusion.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativeOcclusionScenarios';
if (process.argv.length === 2) process.argv.push('native_occlusion_groups', 'native_occlusion_batch_shared_targets', 'native_occlusion_array_camera');
await import('./sprites.mjs');
