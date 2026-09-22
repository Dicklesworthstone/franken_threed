/**
 * Bounded allocation-flow hints for AOT numeric storage alternatives.
 * This is NOT type inference used to eliminate guards: names may be shadowed,
 * constructors replaced, views use species, and bindings be reassigned. Every
 * emitted alternative is separately compiled and native-slot guarded on EVERY
 * call. Wrong or incomplete hints only retain JavaScript or spend an AOT slot.
 * No source execution, runtime profiling, constructor calls or JIT generation.
 */
import * as walk from 'acorn-walk';

const TYPES = Object.freeze(['i8[]', 'u8[]', 'u8c[]', 'i16[]', 'u16[]', 'i32[]', 'u32[]', 'f32[]', 'f64[]']);
const CONSTRUCTORS = new Map(['Int8Array', 'Uint8Array', 'Uint8ClampedArray', 'Int16Array',
  'Uint16Array', 'Int32Array', 'Uint32Array', 'Float32Array', 'Float64Array']
  .map((name, index) => [name, 1 << index]));
const INTEGER_MASK = (1 << 7) - 1;
const MAX_BINDINGS = 4096, MAX_EDGES = 32768, MAX_EXPRESSIONS = 32768;
const EMPTY = Object.freeze({ mask: 0, names: [] });

/** Return (callSites, parameterDescriptors) => at most 16 concrete layouts. */
export function discoverNumericStorageHints(ast) {
  const bindings = new Map(), cache = new WeakMap(), pending = [], queued = new Set();
  let exhausted = false, edges = 0, expressions = 0;
  const enqueue = name => {
    if (!queued.has(name)) { queued.add(name); pending.push(name); }
  };
  function binding(name) {
    if (!bindings.has(name)) {
      if (bindings.size >= MAX_BINDINGS) { exhausted = true; return null; }
      bindings.set(name, { mask: 0, users: new Set() });
    }
    return bindings.get(name);
  }
  function constructorMask(node) {
    const name = node?.type === 'Identifier' ? node.name
      : node?.type === 'MemberExpression' && !node.computed &&
        node.object.type === 'Identifier' && node.object.name === 'globalThis'
        ? node.property.name : null;
    return CONSTRUCTORS.get(name) ?? 0;
  }
  function expression(node, depth = 0) {
    if (!node || exhausted) return EMPTY;
    if (cache.has(node)) return cache.get(node);
    if (depth > 128 || ++expressions > MAX_EXPRESSIONS) { exhausted = true; return EMPTY; }
    let info = EMPTY;
    if (node.type === 'Identifier') info = { mask: 0, names: [node.name] };
    else if (node.type === 'NewExpression') info = { mask: constructorMask(node.callee), names: [] };
    else if (node.type === 'AssignmentExpression' && node.operator === '=') info = expression(node.right, depth + 1);
    else if (node.type === 'SequenceExpression') info = expression(node.expressions.at(-1), depth + 1);
    else if (node.type === 'ConditionalExpression' || node.type === 'LogicalExpression') {
      const left = expression(node.consequent ?? node.left, depth + 1);
      const right = expression(node.alternate ?? node.right, depth + 1);
      info = { mask: left.mask | right.mask, names: [...new Set([...left.names, ...right.names])] };
    } else if (node.type === 'CallExpression' && node.callee.type === 'MemberExpression' &&
        !node.callee.computed && !node.optional && !node.callee.optional) {
      const { object, property } = node.callee;
      if (['from', 'of'].includes(property.name)) info = { mask: constructorMask(object), names: [] };
      else if (['subarray', 'slice'].includes(property.name)) info = expression(object, depth + 1);
    }
    cache.set(node, info);
    return info;
  }
  function connect(name, value) {
    if (exhausted) return;
    const info = expression(value), target = binding(name);
    if (!target || exhausted) return;
    if (info.mask & ~target.mask) { target.mask |= info.mask; enqueue(name); }
    for (const source of info.names) {
      const input = binding(source);
      if (!input || exhausted) return;
      if (!input.users.has(name)) {
        if (++edges > MAX_EDGES) { exhausted = true; return; }
        input.users.add(name);
      }
    }
  }
  // Name unions across scopes are intentionally just hints, not a lexical
  // substitution. Following direct wrapper arguments lets linked application
  // allocations reach frame/update kernels without exposing a new user API.
  const functions = new Map();
  for (const statement of ast.body) {
    const fn = statement.declaration ?? statement;
    if (fn.type === 'FunctionDeclaration' && fn.id) functions.set(fn.id.name, fn);
  }
  walk.simple(ast, {
    VariableDeclarator(node) {
      if (node.id.type === 'Identifier') connect(node.id.name, node.init);
    },
    AssignmentExpression(node) {
      if (node.operator === '=' && node.left.type === 'Identifier') connect(node.left.name, node.right);
    },
    CallExpression(node) {
      const fn = node.callee.type === 'Identifier' ? functions.get(node.callee.name) : null;
      if (exhausted || !fn || node.optional || fn.params.length > 64 ||
          fn.params.length !== node.arguments.length || node.arguments.some(arg => arg.type === 'SpreadElement')) return;
      fn.params.forEach((param, i) => {
        if (param.type === 'Identifier') connect(param.name, node.arguments[i]);
      });
    },
  });
  // A monotone nine-bit worklist, not repeated whole-module scans. Each binding
  // can gain at most nine bits; cyclic alias/wrapper graphs therefore terminate.
  for (let at = 0; !exhausted && at < pending.length; at++) {
    const name = pending[at], source = bindings.get(name);
    queued.delete(name);
    for (const user of source.users) {
      const target = bindings.get(user);
      if (source.mask & ~target.mask) { target.mask |= source.mask; enqueue(user); }
    }
  }
  return (sites, parameters) => {
    if (exhausted) return [];
    const layouts = [], seen = new Set();
    const mask = node => {
      const info = expression(node);
      return info.names.reduce((bits, name) => bits | (bindings.get(name)?.mask ?? 0), info.mask);
    };
    for (const site of sites.slice(0, 256)) {
      if (site.arguments.length !== parameters.length || site.arguments.some(arg => arg.type === 'SpreadElement')) continue;
      const masks = parameters.map((param, i) => param.type === 'f64' ? 0 : mask(site.arguments[i]));
      if (exhausted) return [];
      // Leave float-only applications' existing bytecode/layout order alone.
      if (!masks.some(bits => bits & INTEGER_MASK)) continue;
      for (const [read, write] of [['f64[]', 'f64[]'], ['f32[]', 'f32[]'], ['f64[]', 'f32[]'], ['f32[]', 'f64[]']]) {
        let tuples = [[]];
        for (let i = 0; i < parameters.length; i++) {
          const param = parameters[i];
          const choices = param.type === 'f64' ? ['f64'] : masks[i]
            ? TYPES.filter((type, bit) => masks[i] & (1 << bit)) : [param.write ? write : read];
          const next = [];
          for (const tuple of tuples) for (const type of choices) {
            if (next.length < 16) next.push([...tuple, type]);
          }
          tuples = next;
        }
        for (const types of tuples) {
          const key = types.join(',');
          if (!seen.has(key) && types.some(type => TYPES.indexOf(type) >= 0 && TYPES.indexOf(type) < 7)) {
            seen.add(key); layouts.push(types);
            if (layouts.length === 16) return layouts;
          }
        }
      }
    }
    return layouts;
  };
}
