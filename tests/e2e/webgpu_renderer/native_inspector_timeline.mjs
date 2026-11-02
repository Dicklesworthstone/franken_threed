// Unchanged pinned Timeline controls observe actual native GPU operations.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_inspector_timeline.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativeInspectorTimelineScenarios';
if (process.argv.length === 2) process.argv.push('native_inspector_timeline');
await import('./sprites.mjs');
