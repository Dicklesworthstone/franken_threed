// Reuse the exact offscreen comparison for storage texture mip/layer reads.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_storage_reads.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'storageReadScenarios';
if (process.argv.length === 2) process.argv.push('native_storage_read_mips',
  'native_storage_read_array_mips', 'native_storage_read_uint_mips');
await import('./sprites.mjs');
