/**
 * Import-free Wasm lowering for a deliberately bounded set of Math operations.
 * The caller proves lexical resolution; the host guards the actual Math object
 * and method identities before each execution. Never use this as a name-only
 * rewrite of arbitrary property calls. All operands must already be Numbers.
 *
 * Reference semantics: ECMA-262, Math object; Wasm core, floating-point numerics.
 * https://tc39.es/ecma262/multipage/numbers-and-dates.html#sec-math-object
 * https://webassembly.github.io/spec/core/exec/numerics.html
 */
const UNARY = Object.freeze({
  abs: [0x99], ceil: [0x9b], floor: [0x9c], trunc: [0x9d], sqrt: [0x9f],
  fround: [0xb6, 0xbb], // round to f32 here, then promote back to a JS Number
});
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

export function createMathIntrinsicCompiler(enabled, fail) {
  const used = new Set();
  return {
    call(node, emitArgument, shadowed, allocateLocal) {
      const callee = node.callee;
      if (callee.type !== 'MemberExpression' || callee.object.type !== 'Identifier' ||
          callee.object.name !== 'Math') return null;
      if (!enabled || shadowed || node.optional || callee.optional || callee.computed ||
          callee.property.type !== 'Identifier' || node.arguments.some(arg => arg.type === 'SpreadElement')) {
        fail('Math calls require an explicitly admitted, unshadowed binding and positional numbers', node);
      }
      const name = callee.property.name;
      const variadic = name === 'min' || name === 'max';
      if (!variadic && !Object.hasOwn(UNARY, name) && name !== 'round' && name !== 'sign') {
        fail(`Math.${name} has no closed numeric lowering`, node);
      }
      if (variadic ? node.arguments.length > 64 : node.arguments.length !== 1) {
        fail('Unary Math calls require one argument; min/max admit at most 64', node);
      }
      used.add(name);
      if (variadic) {
        if (!node.arguments.length) return number(name === 'min' ? Infinity : -Infinity);
        // Every argument is pure, numeric, evaluated once in source order. The
        // non-trapping fold preserves NaN propagation and both signs of zero.
        const bytes = emitArgument(node.arguments[0]);
        for (const arg of node.arguments.slice(1)) bytes.push(...emitArgument(arg), name === 'min' ? 0xa4 : 0xa5);
        return bytes;
      }
      const value = emitArgument(node.arguments[0]);
      if (Object.hasOwn(UNARY, name)) return [...value, ...UNARY[name]];
      const x = allocateLocal();
      if (name === 'sign') {
        return [...value, ...set(x), ...get(x), ...number(0), 0x61,
          ...get(x), ...get(x), 0x62, 0x72, // x == 0 || x != x
          0x04, 0x7c, ...get(x), 0x05, ...number(1), ...get(x), 0xa6, 0x0b];
      }
      const floor = allocateLocal();
      // JS rounds half ties toward +infinity, not Wasm nearest's ties-to-even.
      // Do not use floor(x + .5): that incorrectly rounds values just below .5.
      // copysign preserves -0 for [-.5, 0), including negative subnormals.
      return [...value, ...set(x), ...get(x), 0x9c, ...set(floor),
        ...get(x), ...get(floor), 0xa1, ...number(0.5), 0x66,
        0x04, 0x7c, ...get(floor), ...number(1), 0xa0, 0x05, ...get(floor), 0x0b,
        ...get(x), 0xa6];
    },
    requirements() { return Object.freeze([...used].sort()); },
  };
}
