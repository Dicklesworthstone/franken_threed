// Pinned cascaded shadow maps through the owned native WebGPU renderer.
process.env.F3D_NATIVE_SCENARIO_MODULE = '/tests/e2e/webgpu_renderer/scenarios_native_csm_shadows.js';
process.env.F3D_NATIVE_SCENARIO_EXPORT = 'csmShadowScenarios';
if (process.argv.length === 2) process.argv.push('native_csm_pcf', 'native_csm_fade_vsm', 'native_csm_orthographic_basic');
await import('./sprites.mjs');
