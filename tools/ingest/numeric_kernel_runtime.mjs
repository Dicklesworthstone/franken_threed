/**
 * Browser/Node host for compileNumericKernel's explicit, closed numeric-array ABI.
 * No parser, DOM, GPU, eval, scheduler, or eager WebAssembly instantiation.
 *
 * Validate all arguments before effects; pack accessed storage into private
 * memory, execute one Wasm call, then publish only declared outputs. Aliases
 * require the explicit ordered-storage option; independent packing is default.
 * Guard failure invokes the caller's original function exactly once (or throws).
 * This is a copying baseline, not a zero-copy or measured-speedup claim.
 */
const SECTION = 'f3d.numeric-kernel';
const PAGE_BYTES = 65536;
const MAX_BYTES = 1024 * 1024 * 1024;
const F64Array = Float64Array;
const F32Array = Float32Array;
const U8Array = Uint8Array;
const U16Array = Uint16Array;
const U32Array = Uint32Array;
const apply = Reflect.apply;
const descriptor = Object.getOwnPropertyDescriptor;
const prototype = Object.getPrototypeOf;
const hasOwn = Object.hasOwn;
const functionSource = Function.prototype.toString;
const mathHost = globalThis;
const MATH_SEMANTICS = 'f64-operator-order+guarded-math-v1';
const MATH_NAMES = Object.freeze(['abs', 'ceil', 'clz32', 'floor', 'fround', 'imul', 'max', 'min', 'round', 'sign', 'sqrt', 'trunc']);
function dataValue(object, key) {
  const entry = descriptor(object, key);
  return entry && hasOwn(entry, 'value') ? entry.value : undefined;
}
// Like the typed-array primordials below, these are captured at trusted module
// bootstrap, before application code runs. This is not a sandbox for a realm
// whose platform objects were replaced with proxies before runtime loading.
// Data descriptors avoid invoking getters, even when Math was already patched.
const mathObject = dataValue(mathHost, 'Math');
const mathMethods = new Map(MATH_NAMES.map(name => {
  const value = mathObject && typeof mathObject === 'object' ? dataValue(mathObject, name) : undefined;
  // Fail closed for pre-bootstrap ordinary replacements, bound functions and
  // callable proxies. An unfamiliar native source format merely retains JS.
  const native = typeof value === 'function' &&
    apply(functionSource, value, []) === `function ${name}() { [native code] }`;
  return [name, native ? value : null];
}));
// Memory sizing must not call application-overridable Math methods, including
// while deciding whether an intrinsic-using function needs its original path.
const maximum = (a, b) => a > b ? a : b;
const align = (value, unit) => value + (unit - value % unit) % unit;
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


// v7 has one length argument for each array, in declaration order. Full-view
// packing plus checked element addresses supports indirect geometry without
// pretending that the iteration domain proves the destination's bounds.
function indexedManifest(manifest) {
  const arrays = [], names = new Set();
  if (manifest.kind !== 'closed-indexed-numeric' ||
      !['f64-operator-order', MATH_SEMANTICS].includes(manifest.numericSemantics) ||
      manifest.indexSemantics !== 'checked-integer-full-view-v1' ||
      manifest.automaticRouteAdmission !== false || manifest.iterationSemantics !== 'ordered' ||
      !['f64', 'void'].includes(manifest.resultType) ||
      !Number.isInteger(manifest.maxMemoryPages) || manifest.maxMemoryPages < 1 ||
      manifest.maxMemoryPages > 16384 ||
      !Array.isArray(manifest.parameters) || !manifest.parameters.length || manifest.parameters.length > 64 ||
      manifest.boundParameter !== undefined || manifest.boundParameters !== undefined || manifest.loopStride !== undefined) {
    refuse('KERNEL_ABI_MISMATCH', 'Invalid checked-index numeric contract');
  }
  let writes = false;
  manifest.parameters.forEach((param, index) => {
    if (!param || typeof param.name !== 'string' || names.has(param.name) ||
        !['f64', 'f32[]', 'f64[]', 'u16[]', 'u32[]'].includes(param.type) ||
        typeof param.read !== 'boolean' || typeof param.write !== 'boolean' || param.access !== undefined ||
        (param.type === 'f64' && (param.read || param.write)) ||
        (param.write && (!param.read || !['f32[]', 'f64[]'].includes(param.type)))) {
      refuse('KERNEL_ABI_MISMATCH', 'Invalid checked-index parameter');
    }
    names.add(param.name);
    if (param.type !== 'f64') arrays.push(index);
    writes ||= param.write;
    Object.freeze(param);
  });
  const lengths = manifest.lengthParameters;
  if (!arrays.length || !Array.isArray(lengths) || lengths.length !== arrays.length ||
      arrays.some((index, i) => lengths[i] !== index) ||
      !Array.isArray(manifest.loops) || !manifest.loops.length || manifest.loops.length > 16 ||
      manifest.loops.some(pass => !pass || !arrays.includes(pass.boundParameter) ||
        !Number.isInteger(pass.loopStride) || pass.loopStride < 1 || pass.loopStride > 16) ||
      (!writes && manifest.resultType !== 'f64')) {
    refuse('KERNEL_ABI_MISMATCH', 'Invalid checked-index lengths, loops or output');
  }
  Object.freeze(lengths);
  manifest.loops.forEach(Object.freeze);
  Object.freeze(manifest.loops);
  Object.freeze(manifest.parameters);
  if (manifest.sourceSpan) Object.freeze(manifest.sourceSpan);
  return Object.freeze(manifest);
}

function readManifest(module, wasm) {
  const sections = wasm.Module.customSections(module, SECTION);
  if (sections.length !== 1 || sections[0].byteLength > 65536) {
    refuse('KERNEL_ABI_MISMATCH', 'Expected exactly one bounded numeric-kernel ABI section');
  }
  let manifest;
  try { manifest = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(sections[0])); }
  catch (cause) { throw new NumericKernelGuardError('KERNEL_ABI_MISMATCH', 'Invalid ABI metadata', { cause }); }
  const guardedMath = manifest?.numericSemantics === MATH_SEMANTICS;
  if (guardedMath ? !Array.isArray(manifest.mathIntrinsics) ||
      !manifest.mathIntrinsics.length || manifest.mathIntrinsics.length > MATH_NAMES.length ||
      new Set(manifest.mathIntrinsics).size !== manifest.mathIntrinsics.length ||
      manifest.mathIntrinsics.some(name => !MATH_NAMES.includes(name)) : manifest?.mathIntrinsics !== undefined) {
    refuse('KERNEL_ABI_MISMATCH', 'Invalid guarded Math requirements');
  }
  if (guardedMath) Object.freeze(manifest.mathIntrinsics);
  if (manifest?.version === 7) return indexedManifest(manifest);
  const pipeline = manifest?.version === 6 && manifest.kind === 'closed-numeric-pipeline';
  const float32Abi = pipeline || ([2, 3, 4, 5].includes(manifest?.version) && manifest.kind === 'closed-numeric-loop');
  if (!manifest || (!float32Abi && (manifest.version !== 1 || manifest.kind !== 'closed-f64-loop')) ||
      (!guardedMath && manifest.numericSemantics !== 'f64-operator-order') || manifest.automaticRouteAdmission !== false ||
      !Number.isInteger(manifest.maxMemoryPages) || manifest.maxMemoryPages < 1 ||
      manifest.maxMemoryPages > MAX_BYTES / PAGE_BYTES || !Array.isArray(manifest.parameters) ||
      manifest.parameters.length === 0 || manifest.parameters.length > 64 ||
      (!pipeline && (!Number.isInteger(manifest.boundParameter) || manifest.boundParameter < 0 ||
      manifest.boundParameter >= manifest.parameters.length))) {
    refuse('KERNEL_ABI_MISMATCH', 'Unsupported numeric-kernel ABI');
  }
  if (!pipeline && manifest.version >= 3 && (!Number.isInteger(manifest.loopStride) ||
      manifest.loopStride < (manifest.version === 3 ? 2 : 1) || manifest.loopStride > 16)) {
    refuse('KERNEL_ABI_MISMATCH', 'Invalid fixed-stride loop ABI');
  }
  if (manifest.version >= 5 && (!['f64', 'void'].includes(manifest.resultType) ||
      manifest.iterationSemantics !== 'ordered')) {
    refuse('KERNEL_ABI_MISMATCH', 'Invalid ordered-loop result ABI');
  }
  if (pipeline) {
    const bounds = manifest.boundParameters;
    if (!Array.isArray(bounds) || bounds.length < 1 || bounds.length > 16 ||
        new Set(bounds).size !== bounds.length || bounds.some(index => !Number.isInteger(index) ||
          index < 0 || index >= manifest.parameters.length ||
          !['f32[]', 'f64[]'].includes(manifest.parameters[index]?.type)) ||
        !Array.isArray(manifest.loops) || manifest.loops.length < 2 || manifest.loops.length > 16 ||
        manifest.loops.some(pass => !pass || !bounds.includes(pass.boundParameter) ||
          !Number.isInteger(pass.loopStride) || pass.loopStride < 1 || pass.loopStride > 16) ||
        manifest.boundParameter !== undefined || manifest.loopStride !== undefined) {
      refuse('KERNEL_ABI_MISMATCH', 'Invalid ordered-pipeline loop descriptors');
    }
    // Counts are appended to the entry signature in first-use order. Refuse
    // reordered/unused descriptors rather than silently changing the call ABI.
    const used = [...new Set(manifest.loops.map(pass => pass.boundParameter))];
    if (used.length !== bounds.length || used.some((index, i) => index !== bounds[i])) {
      refuse('KERNEL_ABI_MISMATCH', 'Pipeline count arguments are not in first-use order');
    }
    Object.freeze(bounds);
    manifest.loops.forEach(Object.freeze);
    Object.freeze(manifest.loops);
  }
  const names = new Set();
  let writes = 0;
  for (const param of manifest.parameters) {
    if (!param || typeof param.name !== 'string' || names.has(param.name) ||
        !(float32Abi ? ['f32[]', 'f64[]', 'f64'] : ['f64[]', 'f64']).includes(param.type) || typeof param.read !== 'boolean' ||
        typeof param.write !== 'boolean' || (param.type === 'f64' && (param.write || param.read))) {
      refuse('KERNEL_ABI_MISMATCH', 'Invalid parameter descriptor');
    }
    names.add(param.name);
    if (manifest.version >= 4) {
      const access = param.access;
      if (param.type === 'f64' ? access !== undefined :
          !access || typeof access.indexed !== 'boolean' || !Number.isInteger(access.minimumLength) ||
          access.minimumLength < 0 || access.minimumLength > 65536 ||
          (param.write && (!access.indexed || access.minimumLength !== 0)) ||
          (access.minimumLength > 0 && !param.read) ||
          (access.indexed && !param.read && !param.write)) {
        refuse('KERNEL_ABI_MISMATCH', 'Invalid streamed/uniform access extent');
      }
      if (pipeline && access) {
        if (!Array.isArray(access.loopBounds) || access.loopBounds.length > 16 ||
            new Set(access.loopBounds).size !== access.loopBounds.length ||
            access.loopBounds.some(index => !manifest.boundParameters.includes(index)) ||
            access.indexed !== (access.loopBounds.length > 0) || (param.write && !param.read)) {
          refuse('KERNEL_ABI_MISMATCH', 'Invalid pipeline array access union');
        }
        Object.freeze(access.loopBounds);
      }
      if (access) Object.freeze(access);
    }
    if (param.write) writes++;
    Object.freeze(param);
  }
  if ((!writes && !(manifest.version >= 5 && manifest.resultType === 'f64')) ||
      (!pipeline && manifest.parameters[manifest.boundParameter].type === 'f64')) {
    refuse('KERNEL_ABI_MISMATCH', 'Missing output or invalid array loop bound');
  }
  Object.freeze(manifest.parameters);
  if (manifest.sourceSpan) Object.freeze(manifest.sourceSpan);
  return Object.freeze(manifest);
}

function arrayInfo(value, param, checkLength) {
  const { name } = param;
  const ArrayType = param.type === 'u16[]' ? U16Array : param.type === 'u32[]' ? U32Array
    : param.type === 'f32[]' ? F32Array : F64Array;
  const tag = param.type === 'u16[]' ? 'Uint16Array' : param.type === 'u32[]' ? 'Uint32Array'
    : param.type === 'f32[]' ? 'Float32Array' : 'Float64Array';
  // Native slot access rejects proxies without running their traps or user getters.
  if (apply(typedTag, value, []) !== tag || prototype(value) !== ArrayType.prototype) {
    refuse('KERNEL_ARRAY_TYPE', `${name} must be a genuine, non-subclass ${tag}`);
  }
  if (checkLength && (descriptor(value, 'length') || descriptor(ArrayType.prototype, 'length') ||
      prototype(ArrayType.prototype) !== typedPrototype || descriptor(typedPrototype, 'length')?.get !== typedLength)) {
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
  return { buffer, offset: apply(typedOffset, value, []), length: apply(typedLength, value, []),
    ArrayType, elementBytes: param.type === 'u16[]' ? 2 : param.type === 'f64[]' ? 8 : 4 };
}

/**
 * Plan already native-slot-guarded intervals, not whole backing buffers. At most
 * 64 views, no per-element scan. Keep this helper inside the self-contained
 * runtime: generated applications also load this source as an emitted asset.
 * The producer must establish ordered loads/stores without no-alias rewrites.
 */
/** Set each record's ptr and return disjoint byte-copy spans. */
function planNumericStorage(records, refuse) {
  const buffers = new Map();
  for (const record of records) {
    record.ptr = 0;
    if (!record.byteLength || (!record.param.read && !record.param.write)) continue;
    if (!buffers.has(record.buffer)) buffers.set(record.buffer, []);
    buffers.get(record.buffer).push(record);
  }
  const inputs = [], outputs = [];
  let requiredBytes = 0;
  for (const [buffer, views] of buffers) {
    views.sort((a, b) => a.offset - b.offset || a.index - b.index);
    const regions = [];
    for (const view of views) {
      let region = regions.at(-1);
      // Adjacent views need not share an allocation; separate components also
      // avoid copying enormous unused gaps in application-owned buffers.
      if (!region || view.offset >= region.end) {
        region = { start: view.offset, end: view.offset + view.byteLength, views: [] };
        regions.push(region);
      } else region.end = maximum(region.end, view.offset + view.byteLength);
      region.views.push(view);
    }
    for (const region of regions) {
      const writing = region.views.some(view => view.param.write);
      // Cross-type writes expose floating-point bit representations to another
      // numeric type. That requires a stronger proof than Number/store ordering
      // (notably for NaN payloads); keep the original function for those cases.
      if (writing && region.views.some(view => view.param.type !== region.views[0].param.type)) {
        refuse('KERNEL_ARRAY_ALIAS', 'Overlapping mixed-type writes require original JavaScript');
      }
      // Retain offset modulo this component's largest element alignment. Mixed
      // read-only views may start at a 2-byte/4-byte offset before a Float64 view.
      const alignment = region.views.reduce((unit, view) => maximum(unit, view.elementBytes), 1);
      const ptr = align(requiredBytes, alignment) + region.start % alignment;
      requiredBytes = ptr + region.end - region.start;
      for (const view of region.views) view.ptr = ptr + view.offset - region.start;
      // Include write-only bytes: another alias may read them, and copying raw
      // storage retains untouched NaN payloads, signed zeros and record channels.
      inputs.push({ buffer, offset: region.start, ptr, byteLength: region.end - region.start });
      let output = null;
      for (const view of region.views) {
        if (!view.param.write) continue;
        const end = view.offset + view.byteLength;
        if (output && view.offset <= output.offset + output.byteLength) {
          output.byteLength = maximum(output.offset + output.byteLength, end) - output.offset;
        } else {
          output = { buffer, offset: view.offset, ptr: view.ptr, byteLength: view.byteLength };
          outputs.push(output);
        }
      }
    }
  }
  return { requiredBytes, inputs, outputs };
}

/**
 * Instantiate a trusted compiler-produced artifact once and reuse it across
 * updates. A custom section is ABI metadata, NOT authentication of arbitrary
 * Wasm supplied by another party. The Wasm module receives no host imports.
 *
 * fallback must be the original function for conservative execution. It is
 * called with the same arguments and receiver; return values and exceptions
 * propagate unchanged. run() itself remains synchronous.
 * For guarded Math artifacts, resolveMath must be a compiler-produced, effect-
 * free closure `() => Math` in the original function/helper lexical environment.
 * It is not a user callback. Omission conservatively refuses native execution.
 * Global/property descriptors are checked without calling application getters.
 * preserveAliasing opts into shared scratch intervals for same-type aliases.
 * Enable it only for compiler products that preserve ordered memory operations
 * without assuming disjoint parameters. Mixed-type writable overlaps still
 * retain JavaScript. The default keeps the existing independent-prefix ABI.
 */
export function instantiateNumericKernel(bytes, {
  fallback = null, maxMemoryBytes = null, resolveMath = null, preserveAliasing = false,
} = {}) {
  if (typeof preserveAliasing !== 'boolean') throw new TypeError('preserveAliasing must be a boolean');
  if (fallback !== null && typeof fallback !== 'function') throw new TypeError('fallback must be a function or null');
  if (resolveMath !== null && typeof resolveMath !== 'function') throw new TypeError('resolveMath must be a function or null');
  if (maxMemoryBytes !== null && (!Number.isSafeInteger(maxMemoryBytes) ||
      maxMemoryBytes < PAGE_BYTES || maxMemoryBytes > MAX_BYTES)) {
    throw new RangeError('maxMemoryBytes must be between 65536 and 1073741824');
  }
  const wasm = globalThis.WebAssembly;
  if (!wasm) refuse('KERNEL_WASM_UNAVAILABLE', 'WebAssembly is unavailable in this host');
  const module = new wasm.Module(bytes);
  const manifest = readManifest(module, wasm);
  const pipeline = manifest.version === 6;
  const checkedIndexing = manifest.version === 7;
  const boundParameters = checkedIndexing ? manifest.lengthParameters
    : pipeline ? manifest.boundParameters : [manifest.boundParameter];
  const boundSet = new Set(boundParameters);
  const passes = pipeline || checkedIndexing ? manifest.loops : [{ boundParameter: manifest.boundParameter, loopStride: manifest.loopStride ?? 1 }];
  const exports = wasm.Module.exports(module);
  if (wasm.Module.imports(module).length !== 0 || exports.length !== 2 ||
      !exports.some(item => item.name === 'run' && item.kind === 'function') ||
      !exports.some(item => item.name === 'memory' && item.kind === 'memory')) {
    refuse('KERNEL_ABI_MISMATCH', 'Kernel must export only run and memory and have no imports');
  }
  const requestedLimit = maxMemoryBytes ?? MAX_BYTES;
  const declaredLimit = manifest.maxMemoryPages * PAGE_BYTES;
  const limit = requestedLimit < declaredLimit ? requestedLimit : declaredLimit;
  let { memory, run: execute } = new wasm.Instance(module).exports;
  if (memory.buffer.byteLength > limit) refuse('KERNEL_MEMORY_LIMIT', 'Initial Wasm memory exceeds the configured limit');
  let disposed = false;
  const stats = { wasmCalls: 0, fallbackCalls: 0, copiedBytes: 0, lastGuardFailure: null };

  function checkMath() {
    if (!manifest.mathIntrinsics) return;
    if (!resolveMath || !mathObject || dataValue(mathHost, 'Math') !== mathObject) {
      refuse('KERNEL_MATH_BINDING', 'The original Math binding is not available');
    }
    // Compare identity BEFORE inspecting the resolved object: replacement
    // proxies must never receive extra traps during guards.
    let binding;
    try { binding = resolveMath(); }
    catch { refuse('KERNEL_MATH_BINDING', 'The lexical Math binding is uninitialized or unavailable'); }
    if (binding !== mathObject || manifest.mathIntrinsics.some(name =>
        !mathMethods.get(name) || dataValue(mathObject, name) !== mathMethods.get(name))) {
      refuse('KERNEL_MATH_BINDING', 'A required Math binding or method has changed');
    }
  }

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
        records.push({ ...arrayInfo(args[i], param, boundSet.has(i)), param, index: i });
      }
    }
    const lengths = new Map(records.map(record => [record.index, record.length]));
    const counts = boundParameters.map(index => lengths.get(index));
    if (counts.some(count => count > 0xfffffff0)) refuse('KERNEL_LOOP_EXTENT', 'Loop extent exceeds the non-wrapping i32 range');
    for (const pass of passes) {
      if (lengths.get(pass.boundParameter) % pass.loopStride !== 0) {
        refuse('KERNEL_LOOP_EXTENT', 'Incomplete final record requires original JavaScript bounds semantics');
      }
    }
    let requiredBytes = 0;
    for (const record of records) {
      const access = record.param.access;
      // Pack each parameter once, covering the union of every pass's accessed
      // prefix. Later passes see earlier writes in the same private memory.
      // An indirect view's accessed extent is unknown before execution. Preserve
      // the full view, including untouched scatter destinations and tails.
      record.count = checkedIndexing ? (record.param.read || record.param.write ? record.length : 0)
        : pipeline ? access.minimumLength
        : manifest.version >= 4 ? maximum(access.indexed ? counts[0] : 0, access.minimumLength) : counts[0];
      if (pipeline) for (const index of access.loopBounds) record.count = maximum(record.count, lengths.get(index));
      record.byteLength = record.count * record.elementBytes;
      record.ptr = align(requiredBytes, record.elementBytes);
      requiredBytes = record.ptr + record.byteLength;
    }
    if (!preserveAliasing && (!Number.isSafeInteger(requiredBytes) || requiredBytes > limit)) {
      refuse('KERNEL_MEMORY_LIMIT', `Packed update requires ${requiredBytes} bytes; limit is ${limit}`);
    }
    for (const record of records) {
      if (record.length < record.count) refuse('KERNEL_ARRAY_LENGTH', `${record.param.name} is shorter than its accessed extent`);
    }
    // At most 64 parameter views: no per-element guard scan or scene traversal.
    // Only accessed prefixes matter; overlapping unused tails are harmless.
    for (let i = 0; !preserveAliasing && i < records.length; i++) {
      for (let j = 0; j < i; j++) {
        const a = records[i], b = records[j];
        // Ordered reductions can read the same input twice (e.g. dot(a, a)).
        // Unshared fixed buffers cannot change between the two private copies.
        if ((manifest.version < 5 || a.param.write || b.param.write) &&
            a.byteLength && b.byteLength && a.buffer === b.buffer &&
            a.offset < b.offset + b.byteLength && b.offset < a.offset + a.byteLength) {
          refuse('KERNEL_ARRAY_ALIAS', `${a.param.name} and ${b.param.name} have overlapping accessed storage`);
        }
      }
    }
    const storage = preserveAliasing ? planNumericStorage(records, refuse) : null;
    if (storage) requiredBytes = storage.requiredBytes;
    if (!Number.isSafeInteger(requiredBytes) || requiredBytes > limit) {
      refuse('KERNEL_MEMORY_LIMIT', `Packed update requires ${requiredBytes} bytes; limit is ${limit}`);
    }
    const pages = maximum(1, align(requiredBytes, PAGE_BYTES) / PAGE_BYTES);
    if (pages * PAGE_BYTES > limit) {
      refuse('KERNEL_MEMORY_LIMIT', 'Page-rounded Wasm allocation exceeds the configured limit');
    }
    const currentPages = memory.buffer.byteLength / PAGE_BYTES;
    if (pages > currentPages) memory.grow(pages - currentPages);
    // Acquire every view AFTER possible memory.grow; never retain detached views.
    const scratchBuffer = memory.buffer;
    for (const record of records) values[record.index] = record.ptr;
    const inputs = [], outputs = [];
    let copiedBytes = 0;
    if (storage) {
      for (const [spans, destination, reverse] of [
        [storage.inputs, inputs, false], [storage.outputs, outputs, true],
      ]) for (const span of spans) {
        const source = new U8Array(span.buffer, span.offset, span.byteLength);
        const scratch = new U8Array(scratchBuffer, span.ptr, span.byteLength);
        destination.push({ target: reverse ? source : scratch, args: [reverse ? scratch : source] });
        copiedBytes += span.byteLength;
      }
    } else {
      for (const record of records) {
        const source = new record.ArrayType(record.buffer, record.offset, record.count);
        const scratch = new record.ArrayType(scratchBuffer, record.ptr, record.count);
        if (record.param.read) { inputs.push({ target: scratch, args: [source] }); copiedBytes += record.byteLength; }
        if (record.param.write) { outputs.push({ target: source, args: [scratch] }); copiedBytes += record.byteLength; }
      }
    }
    return { inputs, outputs, copiedBytes, values, counts };
  }

  function run(...args) {
    if (disposed) refuse('KERNEL_DISPOSED', 'The numeric kernel has been disposed');
    let prepared;
    let result;
    try {
      checkMath();
      prepared = prepare(args);
      for (const copy of prepared.inputs) apply(typedSet, copy.target, copy.args);
      // Revalidate after preparation (including a possible host memory.grow).
      // The import-free Wasm call cannot rebind Math before publication.
      checkMath();
      result = execute(...prepared.values, ...prepared.counts);
      if (manifest.version >= 5 && manifest.resultType === 'f64' && typeof result !== 'number') {
        refuse('KERNEL_ABI_MISMATCH', 'Numeric return was not produced before publication');
      }
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
    for (const copy of prepared.outputs) apply(typedSet, copy.target, copy.args);
    stats.wasmCalls++;
    stats.lastGuardFailure = null;
    stats.copiedBytes += prepared.copiedBytes;
    if (manifest.version >= 5 && manifest.resultType === 'f64') return result;
  }

  return Object.freeze({
    manifest,
    run,
    get diagnostics() { return Object.freeze({ ...stats, memoryBytes: memory?.buffer.byteLength ?? 0, disposed }); },
    dispose() { disposed = true; memory = null; execute = null; },
  });
}
