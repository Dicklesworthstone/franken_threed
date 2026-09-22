/**
 * Exact Number -> i32 bit patterns for closed numeric Wasm expressions.
 * Inputs are already proven Numbers, not arbitrary JS values or BigInts.
 * ToInt32 and ToUint32 have identical low bits; the consumer chooses signed
 * or unsigned interpretation when promoting its result back to f64.
 *
 * https://tc39.es/ecma262/multipage/abstract-operations.html#sec-touint32
 */
function u32(value) {
  const bytes = [];
  do { const byte = value & 0x7f; value >>>= 7; bytes.push(byte | (value ? 0x80 : 0)); } while (value);
  return bytes;
}
const get = local => [0x20, ...u32(local)];
const set = local => [0x21, ...u32(local)];
function number(value) {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, true);
  return [0x44, ...bytes];
}

/** Emit one i32, evaluating operand once and allocating one private f64 local. */
export function emitToUint32(operand, allocateLocal) {
  const x = allocateLocal();
  // Truncate BEFORE taking the remainder: reducing a negative fraction first
  // would round it toward -Infinity after adding 2^32. Power-of-two division
  // and multiplication are exact here. The subtraction leaves an exact integer
  // in (-2^32, 2^32); adding 2^32 to a negative remainder is exact as well.
  // Every binary64 value with |x| >= 2^84 is already a multiple of 2^32.
  // The ordered comparison also excludes NaN and infinities. Thus the only
  // trapping conversion below receives a proven integer in [0, 2^32).
  return [...operand, 0x9d, ...set(x),
    ...get(x), 0x99, ...number(2 ** 84), 0x63,
    0x04, 0x7f,
      ...get(x), ...get(x), ...number(2 ** 32), 0xa3, 0x9d,
      ...number(2 ** 32), 0xa2, 0xa1, ...set(x),
      ...get(x), ...number(0), 0x63,
      0x04, 0x7c, ...get(x), ...number(2 ** 32), 0xa0,
      0x05, ...get(x), 0x0b, 0xab,
    0x05, 0x41, 0, 0x0b];
}

/** Integer element layouts. Narrow stores consume the low bits of ToUint32. */
export const INTEGER_ARRAY_LAYOUTS = Object.freeze(Object.fromEntries([
  ['i8[]', 0, 0x2c, 0x3a, true],
  ['u8[]', 0, 0x2d, 0x3a, false],
  ['u8c[]', 0, 0x2d, 0x3a, false],
  ['i16[]', 1, 0x2e, 0x3b, true],
  ['u16[]', 1, 0x2f, 0x3b, false],
  ['i32[]', 2, 0x28, 0x36, true],
  ['u32[]', 2, 0x28, 0x36, false],
].map(([type, alignment, load, store, signed]) =>
  [type, Object.freeze({ alignment, load, store, signed })])));

/** ToUint8Clamp: clamp first, then nearest with ties to EVEN (not Math.round). */
export function emitToUint8Clamp(operand, allocateLocal) {
  const x = allocateLocal();
  // !(x > 0) handles NaN, negative values and both zeros without a trap. The
  // final conversion only sees a finite integer in [0,255], including for
  // infinite inputs. Each source store clamps before a later load can see it.
  return [...operand, ...set(x), ...get(x), ...number(0), 0x64,
    0x04, 0x7f,
      ...get(x), ...number(255), 0x66,
      0x04, 0x7f, 0x41, 0xff, 0x01,
      0x05, ...get(x), 0x9e, 0xab, 0x0b,
    0x05, 0x41, 0, 0x0b];
}

export const BITWISE_OPS = Object.freeze({
  '&': 0x71, '|': 0x72, '^': 0x73, '<<': 0x74, '>>': 0x75, '>>>': 0x76,
});

/** Emit a Number result, or null when this is not a bitwise operator. */
export function emitBitwiseBinary(operator, left, right, allocateLocal) {
  if (!Object.hasOwn(BITWISE_OPS, operator)) return null;
  // i32 shifts already mask the converted shift count to its low five bits.
  // Only >>> promotes unsigned: every other Number bitwise result is signed.
  return [...emitToUint32(left, allocateLocal), ...emitToUint32(right, allocateLocal),
    BITWISE_OPS[operator], operator === '>>>' ? 0xb8 : 0xb7];
}

export function emitBitwiseNot(operand, allocateLocal) {
  return [...emitToUint32(operand, allocateLocal), 0x41, 0x7f, 0x73, 0xb7];
}
