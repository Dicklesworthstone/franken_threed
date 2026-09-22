/**
 * Static host for application call-site specialization. This module is kept in
 * its own ESM scope so application bindings cannot shadow the runtime's native
 * globals. No Wasm is instantiated until an admitted target is actually called.
 */
import { instantiateNumericKernel } from "./numeric_kernel_runtime.mjs";

const apply = Reflect.apply;
const hasOwn = Object.hasOwn;
const U8Array = Uint8Array;
const ARRAY_TAGS = Object.freeze({
  "f32[]": "Float32Array", "f64[]": "Float64Array",
  "i8[]": "Int8Array", "u8[]": "Uint8Array", "u8c[]": "Uint8ClampedArray",
  "i16[]": "Int16Array", "u16[]": "Uint16Array",
  "i32[]": "Int32Array", "u32[]": "Uint32Array",
});
const typedTag = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Float64Array.prototype),
  Symbol.toStringTag,
).get;
const records = new WeakMap();

function state(target, bytes, parameterTypes = null, resolveMath = null, preserveAliasing = false) {
  return {
    target,
    bytes,
    parameterTypes,
    resolveMath,
    preserveAliasing,
    attempted: false,
    kernel: null,
    initializationFailure: null,
    retainedCalls: 0,
  };
}

/**
 * Create a private dispatch token without changing the original function.
 * Alternatives are trusted ahead-of-time compiler products, never JIT source.
 * Their native slot checks select an ABI; the kernel still performs every
 * ownership, shape, alias, length and scalar guard before executing.
 * resolveMath is an optional compiler-produced `() => Math` in the target's
 * lexical environment; creating a token must not evaluate that live binding.
 * preserveAliasing is a producer assertion for ALL variants: their loads/stores
 * preserve source order without optimizations assuming disjoint parameters.
 * Storage identity and mixed-type overlap guards still run on each invocation.
 */
export function createNumericDispatch(
  target, bytes, alternatives = [], resolveMath = null, preserveAliasing = false,
) {
  if (typeof preserveAliasing !== "boolean")
    throw new TypeError("preserveAliasing must be a boolean");
  if (typeof target !== "function")
    throw new TypeError("Numeric dispatch target must be a function");
  if (resolveMath !== null && typeof resolveMath !== "function")
    throw new TypeError("resolveMath must be a function or null");
  if (!Array.isArray(alternatives) || alternatives.length > 16)
    throw new TypeError("Expected at most 16 numeric dispatch alternatives");
  const variants = alternatives.map((variant) => {
    if (
      !variant ||
      !Array.isArray(variant.parameterTypes) ||
      variant.parameterTypes.length > 64 ||
      variant.parameterTypes.some(
        (type) => type !== "f64" && !hasOwn(ARRAY_TAGS, type),
      )
    ) {
      throw new TypeError("Invalid numeric dispatch alternative ABI");
    }
    return state(target, variant.bytes, [...variant.parameterTypes], resolveMath, preserveAliasing);
  });
  const token = Object.freeze({});
  const primary = state(target, bytes, null, resolveMath, preserveAliasing);
  records.set(token, { primary, variants, selected: primary, identityMisses: 0 });
  return token;
}

function matches(types, args) {
  if (types.length !== args.length) return false;
  try {
    return types.every((type, index) =>
      type === "f64"
        ? typeof args[index] === "number"
        : apply(typedTag, args[index], []) === ARRAY_TAGS[type],
    );
  } catch {
    return false;
  }
}

/**
 * callee and args have already been evaluated in source order. Identity misses
 * (including shadowed bindings) and pre-initialization calls in ESM cycles must
 * invoke that actual callee, never the statically analyzed function by name.
 */
export function dispatchNumericCall(token, callee, args) {
  const root = records.get(token);
  if (!root || root.primary.target !== callee) {
    if (root) root.identityMisses++;
    return apply(callee, undefined, args);
  }
  const record =
    root.variants.find((variant) => matches(variant.parameterTypes, args)) ?? root.primary;
  root.selected = record;
  if (!record.attempted) {
    // Set before initialization: host policy hooks may reenter the application.
    record.attempted = true;
    try {
      const kernel = instantiateNumericKernel(new U8Array(record.bytes), {
        fallback: record.target,
        resolveMath: record.resolveMath,
        preserveAliasing: record.preserveAliasing,
      });
      if (
        record.parameterTypes &&
        (kernel.manifest.parameters.length !== record.parameterTypes.length ||
          kernel.manifest.parameters.some((param, i) => param.type !== record.parameterTypes[i]))
      ) {
        kernel.dispose();
        throw new TypeError("Dispatch alternative does not match its embedded Wasm ABI");
      }
      record.kernel = kernel;
    } catch {
      // A policy hook can throw any value, including objects with hostile getters.
      record.initializationFailure = "KERNEL_INITIALIZATION_FAILED";
    }
    record.bytes = null;
  }
  if (record.kernel) return apply(record.kernel.run, undefined, args);
  record.retainedCalls++;
  return apply(callee, undefined, args);
}

/** Diagnostics are opt-in; generated applications do not gain new exports. */
export function numericDispatchDiagnostics(token) {
  const root = records.get(token);
  if (!root) throw new TypeError("Unknown numeric dispatch token");
  const all = [root.primary, ...root.variants];
  return Object.freeze({
    initialized: all.some((record) => record.attempted),
    initializationFailure: root.selected.initializationFailure,
    retainedCalls: all.reduce((count, record) => count + record.retainedCalls, 0),
    identityMisses: root.identityMisses,
    kernel: root.selected.kernel?.diagnostics ?? null,
    variants: Object.freeze(
      all.map((record) =>
        Object.freeze({
          initialized: record.attempted,
          kernel: record.kernel?.diagnostics ?? null,
        }),
      ),
    ),
  });
}

// One registry per emitted dispatcher ESM instance, shared by all linked chunks.
// Keys are the actual original functions, not export names, URLs or properties.
// Neither registration nor lookup reads application object properties or keeps
// otherwise unreachable functions alive. Private dispatch tokens stay private.
const sharedTargets = new WeakMap();

/**
 * Compiler-only producer entry point. The target/bytecode pairing must satisfy
 * the same closure proof as createNumericDispatch; the registry is not a trust
 * boundary or a way to authenticate third-party Wasm. Registration is lazy and
 * preserves the target's exports, descriptors, name, length and identity.
 *
 * The first registration owns cross-module calls. Re-exporting a target never
 * needs another registration, and accidentally registering it twice must not
 * reset an already initialized kernel or switch budgets with evaluation order.
 */
export function registerNumericDispatch(...args) {
  const token = createNumericDispatch(...args);
  const target = args[0];
  if (!sharedTargets.has(target)) sharedTargets.set(target, token);
  return token;
}

/**
 * Direct imported calls have an undefined receiver. Evaluate their callee and
 * arguments BEFORE entering here, just as for local dispatch. Live rebindings,
 * early ESM-cycle calls, uncompiled dependencies and shadowed names use the
 * actual callee. A WeakMap miss does not invoke getters or proxy traps.
 */
export function dispatchImportedNumericCall(callee, args) {
  return dispatchNumericCall(sharedTargets.get(callee), callee, args);
}

/** Opt-in diagnostics without exposing registrations or changing app exports. */
export function importedNumericDispatchDiagnostics(target) {
  const token = sharedTargets.get(target);
  return token ? numericDispatchDiagnostics(token) : null;
}
