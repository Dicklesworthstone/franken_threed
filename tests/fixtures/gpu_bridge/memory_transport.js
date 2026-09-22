/**
 * memory_transport.js - Copied Host Transport for Wasm Memory (§6.6, §13.1, §13.4, vqa.4)
 *
 * Enforces safe host transport without false zero-copy lifetime claims:
 * - Scoped callback borrows cannot enforce memory lifetimes across engine boundaries
 * - The safe production boundary is an owned copy obtained synchronously from actual
 *   application memory before effectful host/user calls
 * - Detects and rejects detached ArrayBuffers while allowing live zero-byte ranges
 * - Transparent copy accounting for actual bytes copied at this boundary
 */

/**
 * Returns true if the provided ArrayBuffer or TypedArray view is detached.
 * @param {ArrayBuffer|ArrayBufferView} bufferOrView
 * @returns {boolean}
 */
export function isDetached(bufferOrView) {
  if (!bufferOrView) return true;
  const buf = bufferOrView.buffer || bufferOrView;
  if (buf.detached === true) return true;
  if (buf.byteLength === 0) {
    // Older hosts lack ArrayBuffer.detached. View construction distinguishes a
    // detached buffer from a live empty buffer without allocating backing bytes.
    try {
      new Uint8Array(buf, 0, 0);
    } catch {
      return true;
    }
  }
  return false;
}

/**
 * Validates a view against an active WebAssembly.Memory instance.
 * Rejects detached views, stale buffers, and out-of-bounds ranges.
 * @param {Uint8Array|ArrayBufferView} view
 * @param {WebAssembly.Memory} [wasmMemory]
 */
export function validateMemoryView(view, wasmMemory = null) {
  if (!view || !(view instanceof Uint8Array || ArrayBuffer.isView(view))) {
    throw new Error("validateMemoryView: invalid view argument (expected TypedArray or DataView)");
  }
  if (isDetached(view)) {
    throw new Error("validateMemoryView: view is detached or references a detached ArrayBuffer");
  }
  if (wasmMemory && wasmMemory instanceof WebAssembly.Memory) {
    if (view.buffer !== wasmMemory.buffer) {
      throw new Error(
        "validateMemoryView: view references a stale ArrayBuffer after memory growth",
      );
    }
    if (view.byteOffset + view.byteLength > wasmMemory.buffer.byteLength) {
      throw new Error(
        `validateMemoryView: view out of bounds (offset ${view.byteOffset} + length ${view.byteLength} > memory ${wasmMemory.buffer.byteLength})`,
      );
    }
  }
}

/**
 * TransportCopyLedger - Simple counter tracking actual transport copies.
 */
export class TransportCopyLedger {
  constructor() {
    this.copiedBytes = 0;
    this.copyCount = 0;
  }

  recordCopy(byteLength) {
    this.copiedBytes += byteLength;
    this.copyCount++;
  }

  getMetrics() {
    return {
      copiedBytes: this.copiedBytes,
      copyCount: this.copyCount,
    };
  }
}

export const globalTransportLedger = new TransportCopyLedger();

/**
 * Synchronously copies a byte range from WebAssembly.Memory into an owned Uint8Array.
 * The resulting owned copy is completely decoupled from Wasm memory and safe from
 * subsequent memory.grow() detachment and re-entrancy hazards.
 *
 * Validates that offset and length are finite, non-negative safe integers within buffer bounds.
 *
 * @param {WebAssembly.Memory} wasmMemory
 * @param {number} offset
 * @param {number} length
 * @param {TransportCopyLedger} [ledger]
 * @returns {Uint8Array} Owned typed array copy
 */
export function copyFromWasmMemory(wasmMemory, offset, length, ledger = globalTransportLedger) {
  if (!wasmMemory || !(wasmMemory instanceof WebAssembly.Memory)) {
    throw new Error("copyFromWasmMemory: requires a valid WebAssembly.Memory instance");
  }
  if (isDetached(wasmMemory.buffer)) {
    throw new Error("copyFromWasmMemory: WebAssembly.Memory buffer is detached");
  }
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new Error(
      `copyFromWasmMemory: offset must be a non-negative safe integer, got ${offset}`,
    );
  }
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new Error(
      `copyFromWasmMemory: length must be a non-negative safe integer, got ${length}`,
    );
  }
  if (offset + length > wasmMemory.buffer.byteLength) {
    throw new Error(
      `copyFromWasmMemory: range out of bounds (offset ${offset} + length ${length} > memory ${wasmMemory.buffer.byteLength})`,
    );
  }

  // Create an owned copy via slice to decouple from linear memory
  const owned = new Uint8Array(wasmMemory.buffer.slice(offset, offset + length));
  if (ledger) {
    ledger.recordCopy(length);
  }
  return owned;
}

/**
 * Validates that packet bytes are a valid, non-detached Uint8Array before command execution.
 *
 * NOTE: This validator does NOT produce an owned copy. It validates the provided view
 * against detachment and invalid types. The production WebGPU bridge relies on wasm-bindgen's
 * native Vec<u8> return ABI (getArrayU8FromWasm0(ptr, len).slice()) to ensure that packet bytes
 * are already owned copies decoupled from Wasm linear memory before reaching this point.
 *
 * @param {Uint8Array} packetBytes
 * @returns {Uint8Array} The validated packetBytes view unchanged
 */
export function ensureSafePacketBytes(packetBytes) {
  if (!packetBytes || !(packetBytes instanceof Uint8Array)) {
    throw new Error("ensureSafePacketBytes: packetBytes must be a valid Uint8Array");
  }
  if (packetBytes.byteLength === 0 || isDetached(packetBytes)) {
    throw new Error("ensureSafePacketBytes: received a detached ArrayBuffer or zero-length view");
  }
  return packetBytes;
}
