/**
 * Browser/Node host for compileNumericKernel's explicit, closed f64-array ABI.
 * No parser, DOM, GPU, eval, scheduler, or eager WebAssembly instantiation.
 *
 * Validate all arguments before effects; pack independent array prefixes into
 * private memory, execute one Wasm call, then publish only declared outputs.
 * Guard failure invokes the caller's original function exactly once (or throws).
 * This is a copying baseline, not a zero-copy or measured-speedup claim.
 */
const SECTION = 'f3d.numeric-kernel';
const PAGE_BYTES = 65536;
const MAX_BYTES = 1024 * 1024 * 1024;
const F64Array = Float64Array;
const U8Array = Uint8Array;
const apply = Reflect.apply;
const descriptor = Object.getOwnPropertyDescriptor;
const prototype = Object.getPrototypeOf;
const typedPrototype = prototype(F64Array.prototype);
const typedLength = descriptor(typedPrototype, 'length').get;
const typedBuffer = descriptor(typedPrototype, 'buffer').get;
const typedOffset = descriptor(typedPrototype, 'byteOffset').get;
const typedTag = descriptor(typedPrototype, Symbol.toStringTag).get;
const typedSet = typedPrototype.set;
const bufferLength = descriptor(ArrayBuffer.prototype, 'byteLength').get;
const resizable = descriptor(ArrayBuffer.prototype, 'resizable')?.get;
const immutable = descriptor(ArrayBuffer.prototype, 'immutable')?.get;
const littleEndian = new U8Array(new Uint16Array([1]).buffer)[0] === 1;

export class NumericKernelGuardError extends Error {
  constructor(code, message, options) {
    super(`${code}: ${message}`, options);
    this.name = 'NumericKernelGuardError';
    this.code = code;
  }
}

function refuse(code, message) { throw new NumericKernelGuardError(code, message); }

function readManifest(module, wasm) {
  const sections = wasm.Module.customSections(module, SECTION);
  if (sections.length !== 1 || sections[0].byteLength > 65536) {
    refuse('KERNEL_ABI_MISMATCH', 'Expected exactly one bounded numeric-kernel ABI section');
  }
  let manifest;
  try { manifest = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(sections[0])); }
  catch (cause) { throw new NumericKernelGuardError('KERNEL_ABI_MISMATCH', 'Invalid ABI metadata', { cause }); }
  if (!manifest || manifest.version !== 1 || manifest.kind !== 'closed-f64-loop' ||
      manifest.numericSemantics !== 'f64-operator-order' || manifest.automaticRouteAdmission !== false ||
      !Number.isInteger(manifest.maxMemoryPages) || manifest.maxMemoryPages < 1 ||
      manifest.maxMemoryPages > MAX_BYTES / PAGE_BYTES || !Array.isArray(manifest.parameters) ||
      manifest.parameters.length === 0 || manifest.parameters.length > 64 ||
      !Number.isInteger(manifest.boundParameter) || manifest.boundParameter < 0 ||
      manifest.boundParameter >= manifest.parameters.length) {
    refuse('KERNEL_ABI_MISMATCH', 'Unsupported numeric-kernel ABI');
  }
  const names = new Set();
  let writes = 0;
  for (const param of manifest.parameters) {
    if (!param || typeof param.name !== 'string' || names.has(param.name) ||
        !['f64[]', 'f64'].includes(param.type) || typeof param.read !== 'boolean' ||
        typeof param.write !== 'boolean' || (param.type === 'f64' && (param.write || param.read))) {
      refuse('KERNEL_ABI_MISMATCH', 'Invalid parameter descriptor');
    }
    names.add(param.name);
    if (param.write) writes++;
    Object.freeze(param);
  }
  if (!writes || manifest.parameters[manifest.boundParameter].type !== 'f64[]') {
    refuse('KERNEL_ABI_MISMATCH', 'Missing output or invalid array loop bound');
  }
  Object.freeze(manifest.parameters);
  if (manifest.sourceSpan) Object.freeze(manifest.sourceSpan);
  return Object.freeze(manifest);
}

function arrayInfo(value, name, checkLength) {
  // Native slot access rejects proxies without running their traps or user getters.
  if (apply(typedTag, value, []) !== 'Float64Array' || prototype(value) !== F64Array.prototype) {
    refuse('KERNEL_ARRAY_TYPE', `${name} must be a genuine, non-subclass Float64Array`);
  }
  if (checkLength && (descriptor(value, 'length') || descriptor(F64Array.prototype, 'length') ||
      prototype(F64Array.prototype) !== typedPrototype || descriptor(typedPrototype, 'length')?.get !== typedLength)) {
    refuse('KERNEL_MUTABLE_LENGTH', `${name}.length no longer has intrinsic semantics`);
  }
  const buffer = apply(typedBuffer, value, []);
  try {
    // ArrayBuffer's own getter rejects SharedArrayBuffer without property access.
    apply(bufferLength, buffer, []);
    if ((resizable && apply(resizable, buffer, [])) || (immutable && apply(immutable, buffer, []))) {
      refuse('KERNEL_ARRAY_OWNERSHIP', `${name} requires a mutable fixed-length unshared buffer`);
    }
    new U8Array(buffer, 0, 0); // Detached zero-length buffers must not pass as empty arrays.
  } catch (cause) {
    if (cause instanceof NumericKernelGuardError) throw cause;
    throw new NumericKernelGuardError('KERNEL_ARRAY_OWNERSHIP', `${name} has shared or detached storage`, { cause });
  }
  return { buffer, offset: apply(typedOffset, value, []), length: apply(typedLength, value, []) };
}

/**
 * Instantiate a trusted compiler-produced artifact once and reuse it across
 * updates. A custom section is ABI metadata, NOT authentication of arbitrary
 * Wasm supplied by another party. The Wasm module receives no host imports.
 *
 * fallback must be the original function for conservative execution. It is
 * called with the same arguments and receiver; return values and exceptions
 * propagate unchanged. run() itself remains synchronous.
 */
export function instantiateNumericKernel(bytes, { fallback = null, maxMemoryBytes = null } = {}) {
  if (fallback !== null && typeof fallback !== 'function') throw new TypeError('fallback must be a function or null');
  if (maxMemoryBytes !== null && (!Number.isSafeInteger(maxMemoryBytes) ||
      maxMemoryBytes < PAGE_BYTES || maxMemoryBytes > MAX_BYTES)) {
    throw new RangeError('maxMemoryBytes must be between 65536 and 1073741824');
  }
  const wasm = globalThis.WebAssembly;
  if (!wasm) refuse('KERNEL_WASM_UNAVAILABLE', 'WebAssembly is unavailable in this host');
  const module = new wasm.Module(bytes);
  const manifest = readManifest(module, wasm);
  const exports = wasm.Module.exports(module);
  if (wasm.Module.imports(module).length !== 0 || exports.length !== 2 ||
      !exports.some(item => item.name === 'run' && item.kind === 'function') ||
      !exports.some(item => item.name === 'memory' && item.kind === 'memory')) {
    refuse('KERNEL_ABI_MISMATCH', 'Kernel must export only run and memory and have no imports');
  }
  const limit = Math.min(maxMemoryBytes ?? MAX_BYTES, manifest.maxMemoryPages * PAGE_BYTES);
  let { memory, run: execute } = new wasm.Instance(module).exports;
  if (memory.buffer.byteLength > limit) refuse('KERNEL_MEMORY_LIMIT', 'Initial Wasm memory exceeds the configured limit');
  let disposed = false;
  const stats = { wasmCalls: 0, fallbackCalls: 0, copiedBytes: 0, lastGuardFailure: null };

  function prepare(args) {
    if (args.length !== manifest.parameters.length) refuse('KERNEL_ARGUMENT_COUNT', 'Argument count differs from the ABI');
    if (!littleEndian) refuse('KERNEL_ENDIANNESS', 'This packing ABI requires a little-endian host');
    const records = [];
    const values = [...args];
    for (let i = 0; i < manifest.parameters.length; i++) {
      const param = manifest.parameters[i];
      if (param.type === 'f64') {
        if (typeof args[i] !== 'number') refuse('KERNEL_SCALAR_TYPE', `${param.name} must be a number without coercion`);
      } else {
        records.push({ ...arrayInfo(args[i], param.name, i === manifest.boundParameter), param, index: i });
      }
    }
    const count = records.find(record => record.index === manifest.boundParameter).length;
    const bytesPerArray = count * 8;
    const requiredBytes = bytesPerArray * records.length;
    if (!Number.isSafeInteger(requiredBytes) || requiredBytes > limit) {
      refuse('KERNEL_MEMORY_LIMIT', `Packed update requires ${requiredBytes} bytes; limit is ${limit}`);
    }
    for (const record of records) {
      if (record.length < count) refuse('KERNEL_ARRAY_LENGTH', `${record.param.name} is shorter than the loop bound`);
    }
    // At most 64 parameter views: no per-element guard scan or scene traversal.
    // Only accessed prefixes matter; overlapping unused tails are harmless.
    for (let i = 0; i < records.length; i++) {
      for (let j = 0; j < i; j++) {
        const a = records[i], b = records[j];
        if (count && a.buffer === b.buffer && a.offset < b.offset + bytesPerArray && b.offset < a.offset + bytesPerArray) {
          refuse('KERNEL_ARRAY_ALIAS', `${a.param.name} and ${b.param.name} have overlapping accessed storage`);
        }
      }
    }
    const pages = Math.max(1, Math.ceil(requiredBytes / PAGE_BYTES));
    if (pages * PAGE_BYTES > limit) {
      refuse('KERNEL_MEMORY_LIMIT', 'Page-rounded Wasm allocation exceeds the configured limit');
    }
    const currentPages = memory.buffer.byteLength / PAGE_BYTES;
    if (pages > currentPages) memory.grow(pages - currentPages);
    // Acquire every view AFTER possible memory.grow; never retain detached views.
    const scratchBuffer = memory.buffer;
    records.forEach((record, slot) => {
      const ptr = slot * bytesPerArray;
      values[record.index] = ptr;
      record.source = new F64Array(record.buffer, record.offset, count);
      record.scratch = new F64Array(scratchBuffer, ptr, count);
    });
    return { records, values, count, bytesPerArray };
  }

  function run(...args) {
    if (disposed) refuse('KERNEL_DISPOSED', 'The numeric kernel has been disposed');
    let prepared;
    try {
      prepared = prepare(args);
      for (const record of prepared.records) {
        if (record.param.read) apply(typedSet, record.scratch, [record.source]);
      }
      execute(...prepared.values, prepared.count);
    } catch (cause) {
      const error = cause instanceof NumericKernelGuardError ? cause :
        new NumericKernelGuardError('KERNEL_EXECUTION_FAILED', 'Wasm preparation/execution failed before publication', { cause });
      stats.lastGuardFailure = error.code;
      if (!fallback) throw error;
      stats.fallbackCalls++;
      return apply(fallback, this, args);
    }
    // No user code, memory growth, coercion, or allocation between validated
    // execution and publication. Never rerun fallback after publishing outputs.
    for (const record of prepared.records) {
      if (record.param.write) apply(typedSet, record.source, [record.scratch]);
    }
    stats.wasmCalls++;
    stats.lastGuardFailure = null;
    for (const record of prepared.records) {
      stats.copiedBytes += prepared.bytesPerArray * (Number(record.param.read) + Number(record.param.write));
    }
  }

  return Object.freeze({
    manifest,
    run,
    get diagnostics() { return Object.freeze({ ...stats, memoryBytes: memory?.buffer.byteLength ?? 0, disposed }); },
    dispose() { disposed = true; memory = null; execute = null; },
  });
}
