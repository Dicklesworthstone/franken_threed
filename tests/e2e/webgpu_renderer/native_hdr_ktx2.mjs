// Retained KTX2/Basis HDR decoding through the owned native WebGPU renderer.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_hdr_ktx2.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'hdrKtx2Scenarios';
if (process.argv.length === 2) process.argv.push('native_hdr_ktx2');
await import('./sprites.mjs');
