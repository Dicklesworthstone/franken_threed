/**
 * Static host for application call-site specialization. This module is kept in
 * its own ESM scope so application bindings cannot shadow the runtime's native
 * globals. No Wasm is instantiated until an admitted target is actually called.
 */
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';

const apply = Reflect.apply;
const U8Array = Uint8Array;
const records = new WeakMap();

/** Create a private dispatch token without changing the original function. */
export function createNumericDispatch(target, bytes) {
  if (typeof target !== 'function') throw new TypeError('Numeric dispatch target must be a function');
  const token = Object.freeze({});
  records.set(token, {
    target, bytes, attempted: false, kernel: null, initializationFailure: null,
    retainedCalls: 0, identityMisses: 0,
  });
  return token;
}

/**
 * callee and args have already been evaluated in source order. Identity misses
 * (including shadowed bindings) and pre-initialization calls in ESM cycles must
 * invoke that actual callee, never the statically analyzed function by name.
 */
export function dispatchNumericCall(token, callee, args) {
  const record = records.get(token);
  if (!record || record.target !== callee) {
    if (record) record.identityMisses++;
    return apply(callee, undefined, args);
  }
  if (!record.attempted) {
    // Set before initialization: host policy hooks may reenter the application.
    record.attempted = true;
    try {
      record.kernel = instantiateNumericKernel(new U8Array(record.bytes), { fallback: record.target });
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
  const record = records.get(token);
  if (!record) throw new TypeError('Unknown numeric dispatch token');
  return Object.freeze({
    initialized: record.attempted,
    initializationFailure: record.initializationFailure,
    retainedCalls: record.retainedCalls,
    identityMisses: record.identityMisses,
    kernel: record.kernel?.diagnostics ?? null,
  });
}
