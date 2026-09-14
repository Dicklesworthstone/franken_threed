/**
 * @file borrowed_view.test.mjs
 * Unit tests for borrowed Wasm memory packet views (§6.6, §13.1, §13.4, §16.7, tmt.5 STEP 2a).
 *
 * Verifies:
 * - Direct Uint8Array view into WebAssembly.Memory without copying.
 * - In-place memory mutation visibility (view[5] === 200 without copy).
 * - Detachment/stale buffer invalidation upon memory.grow(1).
 * - validateMemoryView rejection of old view after growth.
 * - currentPacketView rebuilds view upon growth and preserves identical instance when unchanged.
 * - Out-of-bounds, negative, non-integer, NaN, and non-Memory argument rejections.
 * - Zero copy ledger recording (globalTransportLedger invariant).
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  validateMemoryView, globalTransportLedger, isDetached, copyFromWasmMemory,
  ensureSafePacketBytes, TransportCopyLedger,
} from "./memory_transport.js";
import { borrowPacketView, currentPacketView } from "./borrowed_view.js";

test("live zero-page memory supports zero-byte transport but not empty command packets", () => {
  const memory = new WebAssembly.Memory({ initial: 0, maximum: 1 });
  const ledger = new TransportCopyLedger();
  assert.equal(isDetached(memory.buffer), false);
  const copy = copyFromWasmMemory(memory, 0, 0, ledger);
  assert.equal(copy.length, 0);
  assert.notEqual(copy.buffer, memory.buffer);
  assert.deepEqual(ledger.getMetrics(), { copiedBytes: 0, copyCount: 1 });
  const oldView = borrowPacketView(memory, 0, 0);
  assert.equal(currentPacketView(oldView, memory, 0, 0), oldView);
  memory.grow(1);
  assert.equal(isDetached(oldView), true);
  assert.throws(() => validateMemoryView(oldView, memory), /detached|stale/);
  assert.equal(currentPacketView(oldView, memory, 0, 0).buffer, memory.buffer);

  assert.throws(() => ensureSafePacketBytes(copy), /zero-length/);
  assert.throws(() => ensureSafePacketBytes(new Uint8Array(memory.buffer, 0, 0)), /zero-length/);
});

test("transport still rejects actually detached buffers and preserves nonempty packets", () => {
  const buffer = new ArrayBuffer(8);
  const packet = new Uint8Array(buffer);
  packet[0] = 42;
  assert.equal(ensureSafePacketBytes(packet), packet);
  structuredClone(buffer, { transfer: [buffer] });
  assert.equal(isDetached(buffer), true);
  assert.equal(isDetached(packet), true);
  assert.throws(() => validateMemoryView(packet), /detached/);
  assert.throws(() => ensureSafePacketBytes(packet), /detached|zero-length/);
});

test("borrowPacketView and currentPacketView lifecycle and growth invariants", () => {
  const initialMetrics = globalTransportLedger.getMetrics();

  // 1. memory = new WebAssembly.Memory({initial:1, maximum:4})
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 4 });

  // 2. Write (i & 0xff) for i 0..47 at offset 1024
  const memoryU8 = new Uint8Array(memory.buffer);
  for (let i = 0; i < 48; i++) {
    memoryU8[1024 + i] = i & 0xff;
  }

  // 3. view = borrowPacketView(memory, 1024, 48): byteOffset 1024, length 48, bytes equal, view.buffer === memory.buffer
  const view = borrowPacketView(memory, 1024, 48);
  assert.equal(view.byteOffset, 1024);
  assert.equal(view.length, 48);
  for (let i = 0; i < 48; i++) {
    assert.equal(view[i], i & 0xff);
  }
  assert.equal(view.buffer, memory.buffer);

  // 4. Write 200 into memory at 1029 and assert view[5] === 200, proving no copy
  memoryU8[1029] = 200;
  assert.equal(view[5], 200);

  // 5. memory.grow(1) returns 1 and buffer.byteLength becomes 131072
  const oldView = view;
  const prevPages = memory.grow(1);
  assert.equal(prevPages, 1);
  assert.equal(memory.buffer.byteLength, 131072);

  // 6. validateMemoryView(oldView, memory) throws (oldView.buffer is detached)
  assert.equal(oldView.buffer.byteLength, 0);
  assert.throws(() => {
    validateMemoryView(oldView, memory);
  });

  // 7. currentPacketView(oldView, memory, 1024, 48) returns a different object whose buffer === memory.buffer,
  //    with bytes 0..47 identical and index 5 equal to 200
  const refreshedView = currentPacketView(oldView, memory, 1024, 48);
  assert.notEqual(refreshedView, oldView);
  assert.equal(refreshedView.buffer, memory.buffer);
  assert.equal(refreshedView.byteOffset, 1024);
  assert.equal(refreshedView.length, 48);
  for (let i = 0; i < 48; i++) {
    if (i === 5) {
      assert.equal(refreshedView[i], 200);
    } else {
      assert.equal(refreshedView[i], i & 0xff);
    }
  }
  assert.equal(refreshedView[5], 200);

  // 8. currentPacketView on a current view returns the identical object (===)
  const currentView = currentPacketView(refreshedView, memory, 1024, 48);
  assert.equal(currentView === refreshedView, true);

  // Offset change must return a new object (!== currentView) with byteOffset 2048, length 32 and buffer === memory.buffer
  const offsetView = currentPacketView(currentView, memory, 2048, 32);
  assert.notEqual(offsetView, currentView);
  assert.equal(offsetView.byteOffset, 2048);
  assert.equal(offsetView.length, 32);
  assert.equal(offsetView.buffer, memory.buffer);

  // Length change must return a new object with byteOffset 1024, length 16 and bytes 0..15 equal to (i & 0xff) except index 5 which is 200
  const lengthView = currentPacketView(currentView, memory, 1024, 16);
  assert.notEqual(lengthView, currentView);
  assert.equal(lengthView.byteOffset, 1024);
  assert.equal(lengthView.length, 16);
  assert.equal(lengthView.buffer, memory.buffer);
  for (let i = 0; i < 16; i++) {
    if (i === 5) {
      assert.equal(lengthView[i], 200);
    } else {
      assert.equal(lengthView[i], i & 0xff);
    }
  }
  assert.equal(lengthView[5], 200);

  // 9. After growth, borrowPacketView rejects ptr 131025 with len 48 (out of bounds),
  //    ptr -1, ptr 1.5, len NaN, and a non-Memory first argument
  assert.throws(() => {
    borrowPacketView(memory, 131025, 48);
  });
  assert.throws(() => {
    borrowPacketView(memory, -1, 48);
  });
  assert.throws(() => {
    borrowPacketView(memory, 1.5, 48);
  });
  assert.throws(() => {
    borrowPacketView(memory, 1024, NaN);
  });
  assert.throws(() => {
    borrowPacketView({}, 1024, 48);
  });
  assert.throws(() => {
    borrowPacketView(null, 1024, 48);
  });

  // 10. globalTransportLedger.getMetrics() is identical before and after all view calls
  const finalMetrics = globalTransportLedger.getMetrics();
  assert.deepEqual(finalMetrics, initialMetrics);
});
