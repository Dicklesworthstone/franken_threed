/** Build-time only. Reuse the numeric compiler; do not ship a second executor. */
import { compileNumericKernel } from './numeric_kernel.mjs';
import { triangulateMarchingCubes } from './marching_cubes_numeric.mjs';

export const MARCHING_CUBES_PARAMETERS = Object.freeze([
  ...Array(10).fill('f32[]'), 'i32[]', 'i32[]', ...Array(10).fill('f64'),
]);

/**
 * Compile one whole-volume transaction. The source uses runtime grid/material
 * data, not a recorded scene, filename, resolution, isovalue or triangle count.
 * The independent-storage contract belongs to the caller; leave the generic
 * host's preserveAliasing option false. Overflow/fuel exhaustion must use the
 * complete original update, never a truncated mesh or partly published cache.
 */
export function compileMarchingCubesKernel({maxMemoryPages = 2048, maxIterations = 100000000} = {}) {
  return compileNumericKernel(triangulateMarchingCubes.toString(), {
    parameterTypes: [...MARCHING_CUBES_PARAMETERS],
    sourceName: 'f3d:marching-cubes-r186',
    generalControl: true,
    maxMemoryPages,
    maxIterations,
  });
}
