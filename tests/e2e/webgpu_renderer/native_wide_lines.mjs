// Source Line2/LineSegments2 geometry and shaders on the owned WebGPU backend.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_wide_lines.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'wideLineScenarios';
if (process.argv.length === 2) process.argv.push('native_wide_line_screen',
  'native_wide_line_world_dashes', 'native_wide_line_transparent');
await import('./sprites.mjs');
