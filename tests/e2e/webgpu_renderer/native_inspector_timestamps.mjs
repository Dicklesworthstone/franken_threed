// Real query availability and Inspector attribution, never a timing comparison.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_inspector_timestamps.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativeInspectorTimestampScenarios';
if (process.argv.length === 2) process.argv.push('native_inspector_timestamps');
await import('./sprites.mjs');
