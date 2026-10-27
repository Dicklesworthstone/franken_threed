import {INTEGER_ARRAY_LAYOUTS} from './numeric_integer.mjs';

/**
 * Borrowed local typed-array references, not new arrays or escaping JS objects.
 * Each local snapshots a pointer AND that view's own length at source assignment.
 * Locals use exact f64 representations of u32 values to share the existing local
 * allocator; only array operations convert them back to Wasm pointers/lengths.
 * No data is copied or published when a reference is initialized or rebound.
 */
const u32 = value => {
  const bytes = [];
  do { const byte = value & 127; value >>>= 7; bytes.push(byte | (value ? 128 : 0)); } while (value);
  return bytes;
};
const get = index => [0x20, ...u32(index)];
const set = index => [0x21, ...u32(index)];
const number = value => {
  const bytes = new Uint8Array(8); new DataView(bytes.buffer).setFloat64(0, value, true);
  return [0x44, ...bytes];
};
export const arrayPointer = array => [...get(array.index), ...(array.reference ? [0xab] : [])];
export const arrayLength = array => [...get(array.lengthIndex), ...(array.reference ? [0xab] : [])];

/** Value iteration over a private borrowed reference captured at loop entry.
 * Only the index advances: reassigning the source variable cannot retarget the
 * iterator, while writes to its staged storage remain visible on later steps.
 * The host guards the native iterator protocol before any speculative work.
 */
export function createArrayValueIteration(array, allocateLocal) {
  const index = allocateLocal();
  const integer = INTEGER_ARRAY_LAYOUTS[array.type];
  const alignment = integer?.alignment ?? (array.type === 'f64[]' ? 3 : 2);
  array.mark({read:true});
  return {
    initial: [...number(0), ...set(index)],
    predicate: [...get(index), ...arrayLength(array), 0xb8, 0x63],
    value: [...arrayPointer(array), ...get(index), 0xab, 0x41, alignment, 0x74, 0x6a,
      ...(integer ? [integer.load, alignment, 0, integer.signed ? 0xb7 : 0xb8]
        : array.type === 'f32[]' ? [0x2a, 2, 0, 0xbb] : [0x2b, 3, 0])],
    update: [...get(index), ...number(1), 0xa0, ...set(index)],
  };
}

/** Compiler-private, per-function reference/effect graph. The caller resolves
 * real lexical bindings (including TDZ), supplies scalar predicates, and refuses
 * all reference escape/coercion. A fixed storage layout is required per local;
 * incompatible conditional/reassignment layouts retain the original JavaScript.
 */
export function createArrayReferenceCompiler({resolveArray, allocateLocal, condition, expression, control, fail, enabled}) {
  const aliases = [];
  function choice(node, depth = 0) {
    if (!node || depth > 128) fail('Array reference expression exceeds the nesting limit', node);
    const array = resolveArray(node);
    if (array) return {type:array.type, sources:[array], emit(target) {
      return [...arrayPointer(array), 0xb8, ...set(target.index),
        ...arrayLength(array), 0xb8, ...set(target.lengthIndex)];
    }};
    if (node.type === 'CallExpression' && node.callee.type === 'MemberExpression' &&
        !node.callee.computed && node.callee.property.name === 'subarray') {
      if (!control || !enabled) fail('Borrowed subarray views require generalControl', node);
      if (node.optional || node.callee.optional || node.arguments.length > 2 ||
          node.arguments.some(arg => arg.type === 'SpreadElement'))
        fail('Borrowed subarray views require direct calls with positional numeric ranges', node);
      const receiver = choice(node.callee.object, depth + 1);
      if (!receiver) fail('Subarray requires a closed typed-array receiver', node);
      // Species, constructor and method descriptors are guarded before entry.
      // A nonescaping native view is just a borrowed pointer/length pair. Capture
      // the receiver before evaluating either argument, and both arguments before
      // clamping. Range helpers can mutate staged arrays in source order.
      control.methods.add('subarray');
      const view = {index:allocateLocal(), lengthIndex:allocateLocal()};
      const begin = allocateLocal(), end = allocateLocal();
      const beginCode = node.arguments[0] ? expression(node.arguments[0], depth + 1) : number(0);
      const endCode = node.arguments[1] ? expression(node.arguments[1], depth + 1) : get(view.lengthIndex);
      const clamp = local => [...get(local), 0x9d, ...set(local),
        ...get(local), ...get(local), 0x62, 0x04, 0x40, ...number(0), ...set(local), 0x0b,
        ...get(local), ...number(0), 0x63, 0x04, 0x7c,
        ...get(view.lengthIndex), ...get(local), 0xa0, ...number(0), 0xa5,
        0x05, ...get(local), ...get(view.lengthIndex), 0xa4, 0x0b, ...set(local)];
      const elementBytes = 2 ** (INTEGER_ARRAY_LAYOUTS[receiver.type]?.alignment ?? (receiver.type === 'f64[]' ? 3 : 2));
      return {type:receiver.type, sources:receiver.sources, emit(target) {
        return [...receiver.emit(view), ...beginCode, ...set(begin), ...endCode, ...set(end),
          ...clamp(begin), ...clamp(end),
          ...get(view.index), ...get(begin), ...number(elementBytes), 0xa2, 0xa0, ...set(target.index),
          ...get(end), ...get(begin), 0xa1, ...number(0), 0xa5, ...set(target.lengthIndex)];
      }};
    }
    if (node.type !== 'ConditionalExpression') return null;
    const left = choice(node.consequent, depth + 1), right = choice(node.alternate, depth + 1);
    if (!left && !right) return null;
    if (!left || !right || left.type !== right.type)
      fail('Conditional array references must select the same typed-array storage layout', node);
    const test = condition(node.test, depth + 1);
    return {type:left.type, sources:[...left.sources, ...right.sources], emit(target) {
      return [...test, 0x04, 0x40, ...left.emit(target), 0x05, ...right.emit(target), 0x0b];
    }};
  }
  function assign(target, value, node) {
    if (!value || value.type !== target.type)
      fail('Array references cannot change storage layout or become scalar values', node);
    for (const source of value.sources) target.sources.add(source);
    return value.emit(target);
  }
  return {
    declare(node, mutable) {
      const value = choice(node);
      if (!value) return null;
      if (!enabled) fail('Local array references require checkedIndexing', node);
      if (aliases.length >= 1024) fail('Local array reference graph exceeds the binding limit', node);
      const target = {type:value.type, reference:true, mutable,
        index:allocateLocal(), lengthIndex:allocateLocal(), sources:new Set(),
        read:false, write:false};
      // Access descriptors are copied when preparing a subscript. Keep the
      // effect owner in this closure, never in the caller's temporary `this`.
      target.mark = effect => {
        target.read ||= !!(effect.read || effect.write);
        target.write ||= !!effect.write;
      };
      aliases.push(target);
      return {binding:target, bytes:assign(target, value, node)};
    },
    rebind(target, node) {
      if (!target.reference || !target.mutable || node.operator !== '=')
        fail('Only mutable array-reference bindings may be rebound by simple assignment', node);
      return assign(target, choice(node.right), node);
    },
    finish() {
      // Do this AFTER compiling every assignment and helper call. A read before
      // a syntactically later swap can access that later source next iteration.
      // Trace the final graph, including cycles, rather than guessing one owner
      // at the point of the first use. Conservative extra copies are permitted;
      // omitting a potentially accessed ABI view is not. No alias escapes here.
      for (const alias of aliases) {
        if (!alias.read && !alias.write) continue;
        const pending = [...alias.sources], seen = new Set([alias]);
        while (pending.length) {
          const source = pending.pop();
          if (seen.has(source)) continue;
          seen.add(source);
          if (source.reference) pending.push(...source.sources);
          else source.mark({read:alias.read, write:alias.write});
        }
      }
    },
  };
}

/** Find potentially reassigned array parameters without entering another
 * function's execution scope. Shadowing may overapproximate this set; it only
 * allocates private parameter snapshots. Actual references are scope-resolved
 * by the caller and every assignment still passes the closed compiler.
 */
export function reboundArrayParameters(fn, parameterTypes) {
  const arrays = new Set(fn.params.filter((_param, index) => parameterTypes[index] !== 'f64').map(param => param.name));
  const rebound = new Set(), pending = [fn.body];
  while (pending.length && arrays.size) {
    const node = pending.pop();
    if (!node || ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
      'ClassDeclaration', 'ClassExpression'].includes(node.type)) continue;
    if (node.type === 'AssignmentExpression' && node.operator === '=' &&
        node.left.type === 'Identifier' && arrays.has(node.left.name)) rebound.add(node.left.name);
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) {
        for (const child of value) if (child && typeof child.type === 'string') pending.push(child);
      } else if (value && typeof value.type === 'string') pending.push(value);
    }
  }
  return rebound;
}
