/**
 * Emit an explicit numeric update kernel as a relocatable browser/Node package.
 * Compilation never evaluates the source. The original function remains an
 * independently importable fallback; no application call sites or identities
 * are rewritten and no automatic specialization or acceleration is claimed.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

/** Standalone files may contain helpers, but never top-level effects/imports. */
function compileFile(source, sourceName, options) {
  const refuse = (message, code = 'KERNEL_NOT_CLOSED') => { throw new NumericKernelCompileError(code, message); };
  if (source.length > 1048576) refuse('Numeric source file exceeds the 1 MiB limit', 'INVALID_KERNEL_SOURCE');
  let ast;
  try { ast = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module' }); }
  catch (error) { refuse(error.message, 'INVALID_KERNEL_SOURCE'); }
  const functions = ast.body.map(statement => statement.type === 'ExportNamedDeclaration' ? statement.declaration : statement);
  if (!functions.length || functions.some(fn => fn?.type !== 'FunctionDeclaration' || !fn.id)) {
    refuse('Numeric package source must contain only named function declarations and optional named exports');
  }
  // Preserve the original single-function contract and diagnostics exactly.
  // In particular, mutation of a same-named scalar parameter is not rebinding
  // the function, and there are no external helper bindings to prove here.
  if (functions.length === 1 && (options.functionName === undefined || options.functionName === functions[0].id.name)) {
    return { artifact: compileNumericKernel(source, {
      parameterTypes: options.parameterTypes, maxMemoryPages: options.maxMemoryPages, sourceName,
    }), selection: null };
  }
  // A function-only module can still export a setter that rebinds a helper.
  // Exclude every syntactically mutable binding, including destructuring and
  // writes in unused functions; external callers may invoke those later.
  const mutations = new Set();
  function assigned(node) {
    if (!node) return;
    if (node.type === 'Identifier') mutations.add(node.name);
    else if (node.type === 'RestElement') assigned(node.argument);
    else if (node.type === 'AssignmentPattern') assigned(node.left);
    else if (node.type === 'ArrayPattern') node.elements.forEach(assigned);
    else if (node.type === 'ObjectPattern') node.properties.forEach(item => assigned(item.type === 'RestElement' ? item.argument : item.value));
  }
  walk.full(ast, node => {
    if (node.type === 'AssignmentExpression') assigned(node.left);
    if (node.type === 'UpdateExpression') assigned(node.argument);
    if (node.type === 'VariableDeclarator' && node.init) assigned(node.id);
    if (node.type === 'ForInStatement' || node.type === 'ForOfStatement') assigned(node.left);
    if (node.type === 'CallExpression' && node.callee.type === 'Identifier' && node.callee.name === 'eval') {
      refuse('Direct eval prevents immutable numeric helper closure');
    }
  });
  const roots = options.functionName === undefined
    ? functions.filter(fn => fn.body.body.some(statement => statement.type === 'ForStatement'))
    : functions.filter(fn => fn.id.name === options.functionName);
  if (roots.length !== 1) refuse('Select one counted-loop entry using functionName when the source is ambiguous');
  const fn = roots[0];
  if (mutations.has(fn.id.name)) refuse('Numeric entry binding must be immutable');
  const single = functions.length === 1;
  const artifact = compileNumericKernel(single ? source : source.slice(fn.start, fn.end), {
    parameterTypes: options.parameterTypes,
    maxMemoryPages: options.maxMemoryPages,
    sourceName: single ? sourceName : `${sourceName}:${fn.id.name}`,
    helperSources: new Map(functions.filter(helper => !mutations.has(helper.id.name))
      .map(helper => [helper.id.name, source.slice(helper.start, helper.end)])),
  });
  return { artifact, selection: single ? null : {
    name: fn.id.name, sourceSpan: { start: fn.start, end: fn.end },
    scalarHelpers: artifact.helpers.map(helper => {
      const declaration = functions.find(fn => fn.id.name === helper.name);
      return { ...helper, sourceSpan: { start: declaration.start, end: declaration.end } };
    }),
  } };
}

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
 * Build a closed loop and its scalar helper declarations into a fresh directory. All compile and
 * source-read errors occur before creating output. Existing directories/files
 * and leaf symlinks are never overwritten. A failed filesystem write is
 * reported rather than deleting or replacing anything owned by the caller.
 *
 * @param {string} entry Function-only source file; named exports are preserved.
 * @param {string} outDir Fresh destination directory.
 * @param {{parameterTypes: string[], maxMemoryPages?: number, functionName?: string}} options
 */
export function buildNumericKernel(entry, outDir, options = {}) {
  const sourcePath = path.resolve(entry);
  const destination = path.resolve(outDir);
  const source = fs.readFileSync(sourcePath, 'utf8');
  const { artifact, selection } = compileFile(source, path.basename(sourcePath), options);
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
    ...(selection ? { selectedFunction: selection } : {}),
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
