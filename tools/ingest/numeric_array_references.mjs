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
export const arrayPointer = array => [...get(array.index), ...(array.reference ? [0xab] : [])];
export const arrayLength = array => [...get(array.lengthIndex), ...(array.reference ? [0xab] : [])];

/** Compiler-private, per-function reference/effect graph. The caller resolves
 * real lexical bindings (including TDZ), supplies scalar predicates, and refuses
 * all reference escape/coercion. A fixed storage layout is required per local;
 * incompatible conditional/reassignment layouts retain the original JavaScript.
 */
export function createArrayReferenceCompiler({resolveArray, allocateLocal, condition, fail, enabled}) {
  const aliases = [];
  function choice(node, depth = 0) {
    if (!node || depth > 128) fail('Array reference expression exceeds the nesting limit', node);
    const array = resolveArray(node);
    if (array) return {type:array.type, sources:[array], emit(target) {
      return [...arrayPointer(array), 0xb8, ...set(target.index),
        ...arrayLength(array), 0xb8, ...set(target.lengthIndex)];
    }};
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
        fail('Only mutable local array references may be rebound by simple assignment', node);
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
