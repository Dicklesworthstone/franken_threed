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
const reductions = new Set(['reduce', 'reduceRight']);
const searches = new Set(['some', 'every', 'find', 'findIndex', 'findLast', 'findLastIndex']);
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

export function isNumericReductionCall(node) {
  return node?.type === 'CallExpression' && !node.optional &&
    node.callee.type === 'MemberExpression' && !node.callee.optional &&
    !node.callee.computed && reductions.has(node.callee.property.name);
}

export function isNumericForEachCall(node) {
  return node?.type === 'CallExpression' && !node.optional &&
    node.callee.type === 'MemberExpression' && !node.callee.optional &&
    !node.callee.computed && node.callee.property.name === 'forEach';
}

export function isNumericSearchCall(node) {
  return node?.type === 'CallExpression' && !node.optional &&
    node.callee.type === 'MemberExpression' && !node.callee.optional &&
    !node.callee.computed && searches.has(node.callee.property.name);
}

export function isNumericBooleanSearchCall(node) {
  return isNumericSearchCall(node) && ['some', 'every'].includes(node.callee.property.name);
}

// This is a result-type proof, not a coercion. In particular JS &&/|| may
// return a Number, so only Boolean operands establish a Boolean result.
export function isNumericBooleanExpression(node, booleanHelper = () => false, depth = 0) {
  if (!node || depth > 128) return false;
  if (node.type === 'Literal') return typeof node.value === 'boolean';
  if (node.type === 'UnaryExpression') return node.operator === '!';
  if (node.type === 'BinaryExpression') return ['===', '!==', '<', '>', '<=', '>='].includes(node.operator);
  if (node.type === 'LogicalExpression' && ['&&', '||'].includes(node.operator))
    return isNumericBooleanExpression(node.left, booleanHelper, depth + 1) && isNumericBooleanExpression(node.right, booleanHelper, depth + 1);
  if (node.type === 'ConditionalExpression')
    return isNumericBooleanExpression(node.consequent, booleanHelper, depth + 1) && isNumericBooleanExpression(node.alternate, booleanHelper, depth + 1);
  return isNumericBooleanSearchCall(node) ||
    (node.type === 'CallExpression' && node.callee.type === 'Identifier' && booleanHelper(node.callee.name));
}

function inlineCallback(node, parameters) {
  return node && ['ArrowFunctionExpression', 'FunctionExpression'].includes(node.type) &&
    !node.id && !node.async && !node.generator && node.params.length <= parameters &&
    node.params.every(param => param.type === 'Identifier') &&
    new Set(node.params.map(param => param.name)).size === node.params.length;
}

// Only a literal callback consumed by the guarded native method is inlined.
// Its identity cannot escape on that path. A changed method receives the real
// original callback through fallback, never a synthesized callable substitute.
export function numericReductionCallbackBody(node) {
  if (!inlineCallback(node, 4)) return null;
  if (node.type === 'ArrowFunctionExpression' && node.body.type !== 'BlockStatement') return node.body;
  const statements = node.body.body;
  return statements.length === 1 && statements[0].type === 'ReturnStatement' ? statements[0].argument : null;
}

export function numericForEachCallbackBody(node) {
  return inlineCallback(node, 3) ? node.body : null;
}

export const numericSearchCallbackBody = numericForEachCallbackBody;

/** Short-circuit predicates over private staged storage. The original loaded
 * value, iteration index and receiver are independent of mutable callback
 * parameters. Only visited elements spend fuel. Boolean results use f64 0/1
 * internally and carry an explicit host result type; a missing find value
 * aborts the transaction so original JS supplies undefined and its effects.
 */
export function compileTypedArraySearch(node, {
  resolveArray, compileCallback, allocateLocal, control, fail, loopDepth = 0, owner = null,
}) {
  if (!isNumericSearchCall(node)) return null;
  const receiver = resolveArray(node.callee.object), callback = node.arguments[0];
  const body = numericSearchCallbackBody(callback);
  if (!control || !receiver || !body || node.arguments.length !== 1)
    fail('Searches require generalControl, a closed typed-array receiver and one inline predicate callback', node);
  const method = node.callee.property.name, reverse = method.startsWith('findLast');
  const boolean = method === 'some' || method === 'every';
  const valueResult = method === 'find' || method === 'findLast';
  control.methods.add(method);
  control.recordLoop(node, loopDepth + 1, owner?.name ?? null);
  if (owner) owner.loopDepth = Math.max(owner.loopDepth, loopDepth + 1);
  receiver.mark({read:true});
  const view = {type:receiver.type, reference:true, mutable:false,
    index:allocateLocal(), lengthIndex:allocateLocal(), sources:new Set([receiver]),
    mark:effect => receiver.mark(effect)};
  const counter = allocateLocal(), original = allocateLocal(), value = allocateLocal(), index = allocateLocal();
  const predicate = allocateLocal(), result = allocateLocal(), found = valueResult ? allocateLocal() : null;
  const storage = layout(receiver.type);
  const callbackCode = compileCallback(callback, body, [value, index, view], predicate);
  return [...arrayPointer(receiver), 0xb8, ...set(view.index),
    ...arrayLength(receiver), 0xb8, ...set(view.lengthIndex),
    ...(reverse ? [...get(view.lengthIndex), ...number(1), 0xa1] : number(0)), ...set(counter),
    ...number(method === 'every' ? 1 : boolean ? 0 : -1), ...set(result),
    ...(valueResult ? [...number(0), ...set(found)] : []),
    0x02, 0x40, 0x03, 0x40,
    ...get(counter), ...(reverse ? [...number(0), 0x63] : [...get(view.lengthIndex), 0x66]), 0x0d, 1,
    ...control.charge(number(1)),
    ...arrayPointer(view), ...get(counter), 0xab, 0x41, storage.alignment, 0x74, 0x6a,
    storage.load, storage.alignment, 0,
    ...(receiver.type === 'f64[]' ? [] : receiver.type === 'f32[]' ? [0xbb] : [storage.signed ? 0xb7 : 0xb8]),
    ...set(original), ...get(original), ...set(value), ...get(counter), ...set(index),
    ...number(0), ...set(predicate), 0x02, 0x40, ...callbackCode, 0x0b,
    ...get(predicate), ...number(0), method === 'every' ? 0x61 : 0x62, 0x04, 0x40,
    ...(boolean ? number(method === 'some' ? 1 : 0) : get(valueResult ? original : counter)), ...set(result),
    ...(valueResult ? [...number(1), ...set(found)] : []), 0x0c, 2, 0x0b,
    ...get(counter), ...number(1), reverse ? 0xa1 : 0xa0, ...set(counter), 0x0c, 0, 0x0b, 0x0b,
    ...(valueResult ? trapUnless([...get(found), ...number(0), 0x62]) : []), ...get(result)];
}

/** Ordered numeric updates with a discarded result. The iteration counter and
 * receiver never alias the callback's mutable value/index parameters. Its body
 * gets one private block target for returns; the caller owns lexical scope and
 * effect tracking, just as for an explicit loop in the same transaction.
 */
export function compileTypedArrayForEach(node, {
  resolveArray, compileCallback, allocateLocal, control, fail, loopDepth = 0, owner = null,
}) {
  if (!isNumericForEachCall(node)) return null;
  const receiver = resolveArray(node.callee.object), callback = node.arguments[0];
  const body = numericForEachCallbackBody(callback);
  if (!control || !receiver || !body || node.arguments.length !== 1)
    fail('forEach requires generalControl, a closed typed-array receiver and one inline numeric callback', node);
  control.methods.add('forEach');
  control.recordLoop(node, loopDepth + 1, owner?.name ?? null);
  if (owner) owner.loopDepth = Math.max(owner.loopDepth, loopDepth + 1);
  receiver.mark({read:true});
  const view = {type:receiver.type, reference:true, mutable:false,
    index:allocateLocal(), lengthIndex:allocateLocal(), sources:new Set([receiver]),
    mark:effect => receiver.mark(effect)};
  const counter = allocateLocal(), value = allocateLocal(), index = allocateLocal();
  const storage = layout(receiver.type);
  const callbackCode = compileCallback(callback, body, [value, index, view]);
  return [...arrayPointer(receiver), 0xb8, ...set(view.index),
    ...arrayLength(receiver), 0xb8, ...set(view.lengthIndex), ...number(0), ...set(counter),
    ...control.charge(get(view.lengthIndex)), 0x02, 0x40, 0x03, 0x40,
    ...get(counter), ...get(view.lengthIndex), 0x66, 0x0d, 1,
    ...arrayPointer(view), ...get(counter), 0xab, 0x41, storage.alignment, 0x74, 0x6a,
    storage.load, storage.alignment, 0,
    ...(receiver.type === 'f64[]' ? [] : receiver.type === 'f32[]' ? [0xbb] : [storage.signed ? 0xb7 : 0xb8]),
    ...set(value), ...get(counter), ...set(index),
    0x02, 0x40, ...callbackCode, 0x0b,
    ...get(counter), ...number(1), 0xa0, ...set(counter), 0x0c, 0, 0x0b, 0x0b];
}

/** Ordered scalar reduction over the caller's staged storage. Callback bindings
 * belong to a private lexical scope supplied by the root/helper compiler. No
 * JavaScript callback, reference allocation or intermediate publication occurs.
 */
export function compileTypedArrayReduction(node, {
  resolveArray, expression, compileCallback, allocateLocal, control, fail, loopDepth = 0, owner = null,
}) {
  if (!isNumericReductionCall(node)) return null;
  const receiver = resolveArray(node.callee.object), callback = node.arguments[0];
  const body = numericReductionCallbackBody(callback);
  if (!control || !receiver || !body || node.arguments.length > 2 ||
      node.arguments.some(arg => arg.type === 'SpreadElement'))
    fail('Reductions require generalControl, a closed typed-array receiver and an inline numeric callback', node);
  const method = node.callee.property.name, reverse = method === 'reduceRight';
  control.methods.add(method);
  control.recordLoop(node, loopDepth + 1, owner?.name ?? null);
  if (owner) owner.loopDepth = Math.max(owner.loopDepth, loopDepth + 1);
  receiver.mark({read:true});
  const view = {type:receiver.type, reference:true, mutable:false,
    index:allocateLocal(), lengthIndex:allocateLocal(), sources:new Set([receiver]),
    mark:effect => receiver.mark(effect)};
  const accumulator = allocateLocal(), value = allocateLocal(), index = allocateLocal();
  const storage = layout(receiver.type);
  const load = () => [...arrayPointer(view), ...get(index), 0xab, 0x41, storage.alignment, 0x74, 0x6a,
    storage.load, storage.alignment, 0,
    ...(receiver.type === 'f64[]' ? [] : receiver.type === 'f32[]' ? [0xbb] : [storage.signed ? 0xb7 : 0xb8])];
  const advance = [...get(index), ...number(1), reverse ? 0xa1 : 0xa0, ...set(index)];
  // The receiver and its intrinsic length are captured BEFORE the initial-value
  // expression. The callback's array argument refers to this same snapshot even
  // when a later local reference is rebound by the enclosing source function.
  const code = [...arrayPointer(receiver), 0xb8, ...set(view.index),
    ...arrayLength(receiver), 0xb8, ...set(view.lengthIndex),
    ...(reverse ? [...get(view.lengthIndex), ...number(1), 0xa1] : number(0)), ...set(index)];
  const initial = node.arguments.length === 2;
  if (initial) code.push(...expression(node.arguments[1]), ...set(accumulator));
  else code.push(...trapUnless([...get(view.lengthIndex), ...number(0), 0x64]),
    ...load(), ...set(accumulator), ...advance);
  const callbackCode = compileCallback(callback, body, [accumulator, value, index, view]);
  const count = [...get(view.lengthIndex), ...(!initial ? [...number(1), 0xa1] : [])];
  const more = reverse ? [...get(index), ...number(0), 0x66]
    : [...get(index), ...get(view.lengthIndex), 0x63];
  // A one-element unseeded reduction invokes no callback and keeps the loaded
  // Number untouched. Empty unseeded reductions trap before work/publication,
  // letting the original method produce its source TypeError in place.
  code.push(...control.charge(count), 0x02, 0x40, 0x03, 0x40,
    ...more, 0x45, 0x0d, 1, ...load(), ...set(value),
    ...callbackCode, ...set(accumulator), ...advance, 0x0c, 0, 0x0b, 0x0b, ...get(accumulator));
  return code;
}

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
  // Reference setup may evaluate range/selector helpers. Snapshot the receiver
  // before evaluating set's source, and the source before its scalar offset.
  const code = [...(receiver.setup ?? [])];
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
    code.push(...(source.setup ?? []));
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
