/** Prefer legacy ABIs/unrolling, then checked structured and general control. */
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';
import { expandNumericFixedLoops } from './numeric_fixed_loops.mjs';

function compileDirect(source, options) {
  // Explicit callers can force either mode. Integer topology requires v7, but
  // neither type guessing nor a failed prefix proof relaxes the source closure.
  const integerAbi = Array.isArray(options.parameterTypes) &&
    options.parameterTypes.some(type => type === 'u16[]' || type === 'u32[]');
  if (options.checkedIndexing !== undefined || integerAbi || options.structuredLoops === true) {
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

function compileLegacyCandidate(source, options) {
  try { return compileDirect(source, options); }
  catch (error) {
    if (!(error instanceof NumericKernelCompileError) || error.code !== 'KERNEL_NOT_CLOSED') throw error;
    const expanded = expandNumericFixedLoops(source);
    if (!expanded.changed) throw error;
    let artifact;
    const sourceName = String(options.sourceName ?? '<numeric-kernel>');
    try {
      artifact = compileDirect(expanded.source, { ...options, sourceName: sourceName + '#fixed-loop-expansion' });
    } catch (expandedError) {
      if (!(expandedError instanceof NumericKernelCompileError)) throw expandedError;
      // Expansion is never partial feature admission: every generated statement
      // must still satisfy numeric closure, bounds and original storage guards.
      throw error;
    }
    for (const loop of expanded.loops) { Object.freeze(loop.sourceSpan); Object.freeze(loop); }
    // Keep source coordinate spaces explicit. Runtime ABI spans refer to the
    // expanded source; these build-time spans refer to the supplied original.
    const fixedLoops = Object.freeze({ sourceName, expandedSourceName: artifact.manifest.sourceName,
      expandedIterations: expanded.expandedIterations, loops: Object.freeze(expanded.loops) });
    return Object.freeze({ ...artifact, fixedLoops });
  }
}

function compileStructuredCandidate(source, options) {
  try { return compileLegacyCandidate(source, options); }
  catch (error) {
    if (!(error instanceof NumericKernelCompileError) || error.code !== 'KERNEL_NOT_CLOSED' ||
        options.structuredLoops === false || options.checkedIndexing === false) throw error;
    try {
      // Compile the ORIGINAL source, not a partly expanded program. Every
      // access is checked, every nested body shares the embedded work cap,
      // and the existing dispatch host retains the whole original on a miss.
      // Existing successful candidates keep their bytecode and route priority.
      return compileNumericKernel(source, { ...options, checkedIndexing: true, structuredLoops: true });
    } catch (structuredError) {
      if (!(structuredError instanceof NumericKernelCompileError)) throw structuredError;
      throw error;
    }
  }
}

/**
 * General control is the last AOT route, never a substitute for source closure.
 * Keep successful legacy bytecode/route priority and explicit opt-outs intact.
 * A forced generalControl call skips unrolling and uses the original source.
 */
export function compileNumericCandidate(source, options = {}) {
  if (options.generalControl === true) return compileNumericKernel(source, options);
  try { return compileStructuredCandidate(source, options); }
  catch (error) {
    if (!(error instanceof NumericKernelCompileError) || error.code !== 'KERNEL_NOT_CLOSED' ||
        options.generalControl === false || options.checkedIndexing === false ||
        options.structuredLoops === false) throw error;
    try {
      return compileNumericKernel(source, {
        ...options, generalControl: true, checkedIndexing: true, structuredLoops: true,
      });
    } catch (controlError) {
      if (!(controlError instanceof NumericKernelCompileError)) throw controlError;
      // Preserve the original diagnostic for unsupported source; do not emit a
      // partially compiled program or turn an application feature into an error.
      throw error;
    }
  }
}
