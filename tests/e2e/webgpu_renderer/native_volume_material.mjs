// Pinned VolumeNodeMaterial raymarching through the owned native renderer.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_volume_material.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'volumeMaterialScenarios';
if (process.argv.length === 2) process.argv.push('native_volume_emission',
  'native_volume_scattering', 'native_volume_depth');
await import('./sprites.mjs');
