/**
 * Emit an explicit numeric update kernel as a relocatable browser/Node package.
 * Compilation never evaluates the source. The original function remains an
 * independently importable fallback; no application call sites or identities
 * are rewritten and no automatic specialization or acceleration is claimed.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { compileNumericKernel } from './numeric_kernel.mjs';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

function loaderSource(artifact) {
  // Embed the binary once in the module graph: loading works without fetch(),
  // browser import-map changes, a Node Buffer dependency, or an async frame API.
  // kernel.wasm is also emitted for hosts that prefer explicit binary loading.
  const base64 = Buffer.from(artifact.wasm).toString('base64');
  return `/** Generated numeric-kernel package: explicit opt-in, no speedup claim. */
import { instantiateNumericKernel, NumericKernelGuardError } from './runtime.mjs';
import retained from './retained.mjs';
export { retained };
const manifest = ${JSON.stringify(artifact.manifest, null, 2)};
for (const parameter of manifest.parameters) {
  if (parameter.access) Object.freeze(parameter.access);
  Object.freeze(parameter);
}
Object.freeze(manifest.parameters);
Object.freeze(manifest.sourceSpan);
Object.freeze(manifest);

/** Instantiate once, then use synchronous run(...args) for every update. */
export function createKernel() {
  try {
    const binary = globalThis.atob(${JSON.stringify(base64)});
    const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
    return instantiateNumericKernel(bytes, { fallback: retained });
  } catch (error) {
    // Wasm may be unavailable or forbidden by host policy. This does not
    // disable the application's original numeric update function.
    const initializationFailure = error.code || 'KERNEL_INITIALIZATION_FAILED';
    let calls = 0;
    let disposed = false;
    return Object.freeze({
      manifest,
      run(...args) {
        if (disposed) throw new NumericKernelGuardError('KERNEL_DISPOSED', 'The numeric kernel has been disposed');
        calls++;
        return Reflect.apply(retained, this, args);
      },
      get diagnostics() {
        return Object.freeze({
          wasmCalls: 0, fallbackCalls: calls, copiedBytes: 0,
          lastGuardFailure: initializationFailure, memoryBytes: 0, disposed,
        });
      },
      dispose() { disposed = true; },
    });
  }
}
`;
}

/**
 * Build a single closed function file into a fresh directory. All compile and
 * source-read errors occur before creating output. Existing directories/files
 * and leaf symlinks are never overwritten. A failed filesystem write is
 * reported rather than deleting or replacing anything owned by the caller.
 *
 * @param {string} entry Path to a function declaration, optionally exported.
 * @param {string} outDir Fresh destination directory.
 * @param {{parameterTypes: string[], maxMemoryPages?: number}} options
 */
export function buildNumericKernel(entry, outDir, options = {}) {
  const sourcePath = path.resolve(entry);
  const destination = path.resolve(outDir);
  const source = fs.readFileSync(sourcePath, 'utf8');
  const artifact = compileNumericKernel(source, {
    parameterTypes: options.parameterTypes,
    maxMemoryPages: options.maxMemoryPages,
    sourceName: path.basename(sourcePath),
  });
  const runtime = fs.readFileSync(new URL('./numeric_kernel_runtime.mjs', import.meta.url), 'utf8');
  const files = new Map([
    ['kernel.wasm', artifact.wasm],
    ['runtime.mjs', runtime],
    // The compiler admits no default export, so this cannot create a duplicate.
    // Keep source bytes, comments, directives and original named exports intact.
    ['retained.mjs', `${source}\nexport default ${artifact.manifest.functionName};\n`],
    ['kernel.mjs', loaderSource(artifact)],
  ]);
  const manifest = {
    format: 'f3d-numeric-kernel-package-v1',
    entry: 'kernel.mjs',
    binary: 'kernel.wasm',
    sourceSha256: sha256(source),
    wasmSha256: sha256(artifact.wasm),
    kernel: artifact.manifest,
    execution: 'explicit-guarded-wasm-with-retained-javascript-fallback',
    accelerationClaim: false,
    emittedFiles: [...files.keys(), 'kernel.json'],
  };
  files.set('kernel.json', `${JSON.stringify(manifest, null, 2)}\n`);

  fs.mkdirSync(path.dirname(destination), { recursive: true });
  // Atomic creation of the leaf reserves the destination; recursive:true here
  // would wrongly accept a pre-existing application directory or symlink.
  fs.mkdirSync(destination);
  for (const [name, contents] of files) {
    fs.writeFileSync(path.join(destination, name), contents, { flag: 'wx' });
  }
  return { ...manifest, outDir: destination };
}
