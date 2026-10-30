// Automatic geometryNode compute through the owned native WebGPU renderer.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_geometry_compute.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'geometryComputeScenarios';
if (process.argv.length === 2) process.argv.push('native_geometry_compute_fn', 'native_geometry_compute_shared');
await import('./sprites.mjs');
