// Actual pinned GPU probe baking and spatial SH lighting on source/candidate.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_light_probe_grid.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'lightProbeGridScenarios';
if (process.argv.length === 2) process.argv.push('native_probe_grid_bounce', 'native_probe_grid_ranged');
await import('./sprites.mjs');
