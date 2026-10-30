// Pinned MeshSSSNodeMaterial through the owned native WebGPU renderer.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_sss_material.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'sssMaterialScenarios';
if (process.argv.length === 2) process.argv.push('native_sss_thickness',
  'native_sss_transmission', 'native_sss_shadow_environment');
await import('./sprites.mjs');
