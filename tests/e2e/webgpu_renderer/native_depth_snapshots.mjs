// Raw native depth snapshots, including source multisample loads.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_depth_snapshots.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativeDepthSnapshotScenarios';
if (process.argv.length === 2) process.argv.push('native_viewport_depth', 'native_viewport_depth_msaa',
  'native_viewport_depth_array_camera', 'native_public_depth_float');
await import('./sprites.mjs');
