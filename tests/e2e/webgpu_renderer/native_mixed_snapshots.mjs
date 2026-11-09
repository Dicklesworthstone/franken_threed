// Mixed ordinary/authored draws retain accepted inputs and source update order.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_mixed_snapshots.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'nativeMixedSnapshotScenarios';
if (process.argv.length === 2) process.argv.push('native_mixed_accepted_inputs', 'native_mixed_source_updates');
await import('./sprites.mjs');
