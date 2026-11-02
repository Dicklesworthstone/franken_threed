// Native signed/unsigned generated streams, using the established source GPU comparison.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_integer_attributes.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'integerAttributeScenarios';
if (process.argv.length === 2) process.argv.push('native_integer_vertices', 'native_integer_repeated_vertices');
await import('./sprites.mjs');
