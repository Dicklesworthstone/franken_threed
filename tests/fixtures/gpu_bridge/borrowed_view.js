/**
 * borrowed_view.js - Borrowed Wasm Memory Packet View (§6.6, §13.1, §13.4, §16.7, tmt.5)
 *
 * NOTE: This is a measurement sub-variant valid only between f3d_borrow_enter
 * and f3d_borrow_exit, not the production copied boundary.
 * It performs no slice, no copy, and creates no TransportCopyLedger record.
 *
 * In production, the safe host transport boundary requires synchronous owned
 * copies to protect against memory growth detachment and reentrancy hazards
 * before effectful host/user calls. This borrowed view helper is provided
 * strictly for matched zero-copy measurement comparisons within Rust-scoped
 * borrow lifecycles.
 */

import { validateMemoryView } from "./memory_transport.js";

/**
 * Borrows a direct Uint8Array view into WebAssembly.Memory without copying.
 *
 * Valid only between f3d_borrow_enter and f3d_borrow_exit, not the production
 * copied boundary. Performs no slice, no copy, and no TransportCopyLedger record.
 *
 * @param {WebAssembly.Memory} wasmMemory
 * @param {number} ptr - Non-negative safe-integer byte offset in Wasm memory
 * @param {number} len - Non-negative safe-integer byte length
 * @returns {Uint8Array} Direct view into Wasm linear memory
 */
export function borrowPacketView(wasmMemory, ptr, len) {
  if (!wasmMemory || !(wasmMemory instanceof WebAssembly.Memory)) {
    throw new TypeError("borrowPacketView: requires a valid WebAssembly.Memory instance");
  }
  if (!Number.isSafeInteger(ptr) || ptr < 0) {
    throw new RangeError(`borrowPacketView: ptr must be a non-negative safe integer, got ${ptr}`);
  }
  if (!Number.isSafeInteger(len) || len < 0) {
    throw new RangeError(`borrowPacketView: len must be a non-negative safe integer, got ${len}`);
  }
  if (ptr + len > wasmMemory.buffer.byteLength) {
    throw new RangeError(
      `borrowPacketView: range out of bounds (ptr ${ptr} + len ${len} > buffer ${wasmMemory.buffer.byteLength})`
    );
  }

  const view = new Uint8Array(wasmMemory.buffer, ptr, len);
  validateMemoryView(view, wasmMemory);
  return view;
}

/**
 * Returns an up-to-date direct Uint8Array view into WebAssembly.Memory.
 *
 * Returns the cached view object only when view.buffer === wasmMemory.buffer,
 * view.byteOffset === ptr, and view.byteLength === len; otherwise rebuilds
 * through borrowPacketView. Reuse requires matching buffer, offset, and length,
 * because Rust can rebuild the slot at a new ptr after borrow exit with no memory growth.
 *
 * Valid only between f3d_borrow_enter and f3d_borrow_exit, not the production
 * copied boundary. Performs no slice, no copy, and no TransportCopyLedger record.
 *
 * @param {Uint8Array|null} view
 * @param {WebAssembly.Memory} wasmMemory
 * @param {number} ptr
 * @param {number} len
 * @returns {Uint8Array}
 */
export function currentPacketView(view, wasmMemory, ptr, len) {
  if (
    view &&
    wasmMemory &&
    view.buffer === wasmMemory.buffer &&
    view.byteOffset === ptr &&
    view.byteLength === len
  ) {
    return view;
  }
  return borrowPacketView(wasmMemory, ptr, len);
}
