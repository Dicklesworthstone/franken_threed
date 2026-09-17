/**
 * Static host for application call-site specialization. This module is kept in
 * its own ESM scope so application bindings cannot shadow the runtime's native
 * globals. No Wasm is instantiated until an admitted target is actually called.
 */
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

const apply = Reflect.apply;
const U8Array = Uint8Array;
const typedTag = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(Float64Array.prototype), Symbol.toStringTag).get;
const records = new WeakMap();

function state(target, bytes, parameterTypes = null, resolveMath = null) {
  return { target, bytes, parameterTypes, resolveMath, attempted: false, kernel: null,
    initializationFailure: null, retainedCalls: 0 };
}

/**
 * Create a private dispatch token without changing the original function.
 * Alternatives are trusted ahead-of-time compiler products, never JIT source.
 * Their native slot checks select an ABI; the kernel still performs every
 * ownership, shape, alias, length and scalar guard before executing.
 * resolveMath is an optional compiler-produced `() => Math` in the target's
 * lexical environment; creating a token must not evaluate that live binding.
 */
export function createNumericDispatch(target, bytes, alternatives = [], resolveMath = null) {
  if (typeof target !== 'function') throw new TypeError('Numeric dispatch target must be a function');
  if (resolveMath !== null && typeof resolveMath !== 'function') throw new TypeError('resolveMath must be a function or null');
  if (!Array.isArray(alternatives) || alternatives.length > 16) throw new TypeError('Expected at most 16 numeric dispatch alternatives');
  const variants = alternatives.map(variant => {
    if (!variant || !Array.isArray(variant.parameterTypes) || variant.parameterTypes.length > 64 ||
        variant.parameterTypes.some(type => !['f32[]', 'f64[]', 'u16[]', 'u32[]', 'f64'].includes(type))) {
      throw new TypeError('Invalid numeric dispatch alternative ABI');
    }
    return state(target, variant.bytes, [...variant.parameterTypes], resolveMath);
  });
  const token = Object.freeze({});
  const primary = state(target, bytes, null, resolveMath);
  records.set(token, { primary, variants, selected: primary, identityMisses: 0 });
  return token;
}

function matches(types, args) {
  if (types.length !== args.length) return false;
  try {
    return types.every((type, index) => type === 'f64'
      ? typeof args[index] === 'number'
      : apply(typedTag, args[index], []) === (type === 'u16[]' ? 'Uint16Array' : type === 'u32[]' ? 'Uint32Array'
        : type === 'f32[]' ? 'Float32Array' : 'Float64Array'));
  } catch { return false; }
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
  const record = root.variants.find(variant => matches(variant.parameterTypes, args)) ?? root.primary;
  root.selected = record;
  if (!record.attempted) {
    // Set before initialization: host policy hooks may reenter the application.
    record.attempted = true;
    try {
      const kernel = instantiateNumericKernel(new U8Array(record.bytes), { fallback: record.target, resolveMath: record.resolveMath });
      if (record.parameterTypes && (kernel.manifest.parameters.length !== record.parameterTypes.length ||
          kernel.manifest.parameters.some((param, i) => param.type !== record.parameterTypes[i]))) {
        kernel.dispose();
        throw new TypeError('Dispatch alternative does not match its embedded Wasm ABI');
      }
      record.kernel = kernel;
    } catch {
      // A policy hook can throw any value, including objects with hostile getters.
      record.initializationFailure = 'KERNEL_INITIALIZATION_FAILED';
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
  if (!root) throw new TypeError('Unknown numeric dispatch token');
  const all = [root.primary, ...root.variants];
  return Object.freeze({
    initialized: all.some(record => record.attempted),
    initializationFailure: root.selected.initializationFailure,
    retainedCalls: all.reduce((count, record) => count + record.retainedCalls, 0),
    identityMisses: root.identityMisses,
    kernel: root.selected.kernel?.diagnostics ?? null,
    variants: Object.freeze(all.map(record => Object.freeze({
      initialized: record.attempted, kernel: record.kernel?.diagnostics ?? null,
    }))),
  });
}
