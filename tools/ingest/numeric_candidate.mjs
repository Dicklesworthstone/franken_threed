/** Choose the prefix ABI first; use checked full-view indexing only when needed. */
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';

export function compileNumericCandidate(source, options = {}) {
  // Explicit callers can force either mode. Integer topology requires v7, but
  // neither type guessing nor a failed prefix proof relaxes the source closure.
  const integerAbi = Array.isArray(options.parameterTypes) &&
    options.parameterTypes.some(type => type === 'u16[]' || type === 'u32[]');
  if (options.checkedIndexing !== undefined || integerAbi) {
    return compileNumericKernel(source, { ...options, checkedIndexing: options.checkedIndexing === undefined ? true : options.checkedIndexing });
  }
  try { return compileNumericKernel(source, options); }
  catch (error) {
    if (!(error instanceof NumericKernelCompileError) || error.code !== 'KERNEL_NOT_CLOSED') throw error;
    try { return compileNumericKernel(source, { ...options, checkedIndexing: true }); }
    catch (checkedError) {
      if (!(checkedError instanceof NumericKernelCompileError)) throw checkedError;
      // Unsupported applications retain their original prefix-path diagnostic.
      throw error;
    }
  }
}
