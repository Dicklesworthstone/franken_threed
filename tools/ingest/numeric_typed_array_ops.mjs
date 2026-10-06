/** Closed, discarded-result typed-array bulk operations over staged views.
 * The host guards native methods before entry; this emitter never calls JS.
 * Receiver and argument values are snapshotted in source evaluation order.
 * Same-layout copies use memory.copy, preserving overlap and NaN payload bits.
 * Cross-layout copies use existing Number conversions; mixed-layout overlapping
 * source views remain a host ownership/alias refusal, never an unsafe copy.
 */
import {arrayPointer, arrayLength} from './numeric_array_references.mjs';
import {INTEGER_ARRAY_LAYOUTS, emitToUint32, emitToUint8Clamp} from './numeric_integer.mjs';

const names = new Set(['set', 'fill', 'copyWithin']);
const u32 = value => {
  const bytes = [];
  do { const byte = value & 127; value >>>= 7; bytes.push(byte | (value ? 128 : 0)); } while (value);
  return bytes;
};
const get = index => [0x20, ...u32(index)], set = index => [0x21, ...u32(index)];
const number = value => {
  const bytes = new Uint8Array(8); new DataView(bytes.buffer).setFloat64(0, value, true);
  return [0x44, ...bytes];
};
const trapUnless = test => [...test, 0x45, 0x04, 0x40, 0x00, 0x0b];
const layout = type => INTEGER_ARRAY_LAYOUTS[type] ?? {
  alignment:type === 'f64[]' ? 3 : 2, load:type === 'f64[]' ? 0x2b : 0x2a,
  store:type === 'f64[]' ? 0x39 : 0x38,
};

/** Null is not an admitted bulk call. Only discarded direct member calls are
 * accepted: result escape/chaining, dynamic properties and spreads stay in JS.
 * control is shared with every root/helper loop, including argument helpers.
 */
export function compileTypedArrayOperation(node, {
  resolveArray, expression, allocateLocal, control, fail, loopDepth = 0, owner = null,
}) {
  if (node?.type !== 'CallExpression' || node.callee.type !== 'MemberExpression' ||
      node.callee.computed || !names.has(node.callee.property.name)) return null;
  const method = node.callee.property.name, receiver = resolveArray(node.callee.object);
  if (!receiver || node.optional || node.callee.optional ||
      node.arguments.some(arg => arg.type === 'SpreadElement'))
    fail('Bulk operations require a closed typed-array receiver and positional arguments', node);
  if (!control) fail('Typed-array bulk operations require generalControl', node);
  if (node.arguments.length > (method === 'set' ? 2 : 3) ||
      (method === 'set' && !node.arguments.length))
    fail('Unsupported typed-array bulk argument count', node);
  const source = method === 'set' ? resolveArray(node.arguments[0]) : receiver;
  if (!source) fail('TypedArray.set requires a closed typed-array source', node);
  control.methods.add(method);
  control.recordLoop(node, loopDepth + 1, owner?.name ?? null);
  if (owner) owner.loopDepth = Math.max(owner.loopDepth, loopDepth + 1);
  receiver.mark({read:true, write:true});
  if (method === 'set') source.mark({read:true});
  const code = [];
  const save = bytes => { const local = allocateLocal(); code.push(...bytes, ...set(local)); return local; };
  const ptr = save([...arrayPointer(receiver), 0xb8]);
  const length = save([...arrayLength(receiver), 0xb8]);
  const arg = (index, fallback) => save(node.arguments[index] ? expression(node.arguments[index]) : fallback);
  // ToIntegerOrInfinity for already-proven Numbers. Truncation retains signed
  // zero/infinities, then NaN alone becomes zero. No coercion or host Math call.
  const integer = local => [...get(local), 0x9d, ...set(local),
    ...get(local), ...get(local), 0x62, 0x04, 0x40, ...number(0), ...set(local), 0x0b];
  const clamp = local => [...integer(local), ...get(local), ...number(0), 0x63,
    0x04, 0x7c, ...get(length), ...get(local), 0xa0, ...number(0), 0xa5,
    0x05, ...get(local), ...get(length), 0xa4, 0x0b, ...set(local)];
  const address = (base, index, type) => [...get(base), 0xab,
    ...get(index), 0xab, 0x41, layout(type).alignment, 0x74, 0x6a];
  const load = (base, index, type) => [...address(base, index, type),
    layout(type).load, layout(type).alignment, 0,
    ...(type === 'f64[]' ? [] : type === 'f32[]' ? [0xbb]
      : [layout(type).signed ? 0xb7 : 0xb8])];
  const store = (base, index, type, value) => [...address(base, index, type),
    ...(INTEGER_ARRAY_LAYOUTS[type] ? (type === 'u8c[]' ? emitToUint8Clamp : emitToUint32)(value, allocateLocal)
      : [...value, ...(type === 'f32[]' ? [0xb6] : [])]),
    layout(type).store, layout(type).alignment, 0];
  // Finite view lengths, clamped starts and checked set extents keep every byte
  // address/count inside the host's bounded allocation before i32 conversion.
  const copy = (fromPtr, from, to, count, type) => [...address(ptr, to, type),
    ...address(fromPtr, from, type), ...get(count), 0xab, 0x41, layout(type).alignment, 0x74,
    0xfc, 0x0a, 0, 0];
  const each = (index, end, body) => [0x02, 0x40, 0x03, 0x40,
    ...get(index), ...get(end), 0x66, 0x0d, 1, ...body,
    ...get(index), ...number(1), 0xa0, ...set(index), 0x0c, 0, 0x0b, 0x0b];

  if (method === 'set') {
    const fromPtr = save([...arrayPointer(source), 0xb8]), count = save([...arrayLength(source), 0xb8]);
    const offset = arg(1, number(0));
    code.push(...integer(offset), ...trapUnless([...get(offset), ...number(0), 0x66,
      ...get(offset), ...get(count), 0xa0, ...get(length), 0x65, 0x71]),
      ...control.charge(get(count)));
    const index = save(number(0));
    if (source.type === receiver.type) code.push(...copy(fromPtr, index, offset, count, receiver.type));
    else {
      const destination = allocateLocal();
      code.push(...each(index, count, [...get(offset), ...get(index), 0xa0, ...set(destination),
        ...store(ptr, destination, receiver.type, load(fromPtr, index, source.type))]));
    }
  } else if (method === 'fill') {
    const value = arg(0, number(NaN)), start = arg(1, number(0)), end = arg(2, get(length));
    code.push(...clamp(start), ...clamp(end));
    const count = save([...get(end), ...get(start), 0xa1, ...number(0), 0xa5]);
    code.push(...control.charge(get(count)), ...each(start, end, store(ptr, start, receiver.type, get(value))));
  } else {
    const target = arg(0, number(0)), start = arg(1, number(0)), end = arg(2, get(length));
    code.push(...clamp(target), ...clamp(start), ...clamp(end));
    const count = save([...get(end), ...get(start), 0xa1,
      ...get(length), ...get(target), 0xa1, 0xa4, ...number(0), 0xa5]);
    code.push(...control.charge(get(count)), ...copy(ptr, start, target, count, receiver.type));
  }
  return code;
}
