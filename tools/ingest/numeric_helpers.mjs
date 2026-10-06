/**
 * Close a bounded, acyclic graph of numeric JavaScript helpers into private Wasm
 * functions. No host imports, source evaluation, captures or implicit conversions
 * are admitted. Checked callers may pass typed-array views by private pointer and
 * intrinsic length; each storage signature gets a bounded native specialization.
 * Each call evaluates arguments once, left to right;
 * a helper owns its scalar parameters and lexical locals, just as in JavaScript.
 * Number bitwise operations share the kernel's exact modulo-2^32 lowering.
 * With the caller's general-control contract, helper for/while/do-while bodies
 * spend the SAME private invocation budget as the entry point. Helpers neither
 * reset that budget nor call the host. Abrupt completion and lexical scopes
 * remain source ordered. Numeric calls return a Number on every path; void
 * helpers are admitted only where the caller discards the result.
 */
import * as acorn from 'acorn';
import { createArrayReferenceCompiler, arrayPointer, arrayLength } from './numeric_array_references.mjs';
import { BITWISE_OPS, INTEGER_ARRAY_LAYOUTS, emitBitwiseBinary, emitBitwiseNot,
  emitToUint32, emitToUint8Clamp } from './numeric_integer.mjs';

const F64 = 0x7c;
const I32 = 0x7f;
const OPS = Object.freeze({ '+': 0xa0, '-': 0xa1, '*': 0xa2, '/': 0xa3 });
const COMPARE = Object.freeze({ '===': 0x61, '!==': 0x62, '<': 0x63, '>': 0x64, '<=': 0x65, '>=': 0x66 });
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

/**
 * helperSources maps immutable lexical function names to their declarations.
 * The caller is responsible for proving that these are the actual bindings at
 * the call site. The module specializer does this after linking, rejecting eval
 * and every statically mutable binding. Only reachable helpers are inspected.
 * Function/type index zero belongs to the caller's array-loop entry point.
 */
export function createScalarHelperCompiler(helperSources, fail, intrinsics = null, control = null, checkedArrays = false) {
  if (!(helperSources instanceof Map)) fail('helperSources must be a Map of immutable function declarations', null, 'INVALID_KERNEL_SOURCE');
  const entries = [];
  const compiled = new Map();
  const active = new Set();
  let sourceBytes = 0;
  let statementCount = 0;

  function requireHelper(name, callNode, parameterTypes) {
    const key = JSON.stringify([name, parameterTypes]);
    if (active.has(name)) fail(`Recursive scalar helper ${name} is not closed`, callNode);
    if (compiled.has(key)) return compiled.get(key);
    if (!helperSources.has(name)) fail(`Unresolved scalar helper ${name}`, callNode);
    if (entries.length >= 64 || active.size >= 32) fail('Scalar helper graph exceeds the function/depth limit', callNode);
    const source = helperSources.get(name);
    if (typeof source !== 'string' || source.length > 65536 || (sourceBytes += source.length) > 262144) {
      fail('Reachable scalar helper sources exceed the source limit', callNode);
    }
    let ast;
    try { ast = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module', locations: true }); }
    catch (error) { fail(`Invalid scalar helper ${name}: ${error.message}`, callNode); }
    const declaration = ast.body[0];
    const fn = declaration?.type === 'ExportNamedDeclaration' ? declaration.declaration : declaration;
    if (ast.body.length !== 1 || fn?.type !== 'FunctionDeclaration' || fn.id?.name !== name ||
        fn.async || fn.generator || fn.params.length > 64 || fn.params.some(param => param.type !== 'Identifier')) {
      fail(`Helper ${name} must be one named synchronous scalar function without defaults/destructuring`, callNode);
    }
    if (parameterTypes.length !== fn.params.length) fail(`Scalar helper ${name} argument count differs from its declaration`, callNode);
    // Only Number results may enter arithmetic. Void helpers are admitted solely
    // at discarded-value call sites; numeric/void mixed returns still refuse.
    const pending = [fn.body];
    let numericResult = false;
    while (pending.length) {
      const node = pending.pop();
      if (['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
        'ClassDeclaration', 'ClassExpression'].includes(node.type)) continue;
      if (node.type === 'ReturnStatement' && node.argument) numericResult = true;
      for (const value of Object.values(node)) {
        if (Array.isArray(value)) {
          for (const child of value) if (child && typeof child.type === 'string') pending.push(child);
        } else if (value && typeof value.type === 'string') pending.push(value);
      }
    }
    const wasmTypes = parameterTypes.flatMap(type => type === 'f64' ? [F64] : [I32, I32]);
    const entry = { name, arity: fn.params.length, index: entries.length + 1, body: null,
      parameterTypes, wasmTypes, resultType: numericResult ? 'f64' : 'void',
      effects: parameterTypes.map(() => ({ read: false, write: false })) };
    entries.push(entry);
    compiled.set(key, entry);
    active.add(name);
    try { entry.body = compileFunction(fn, entry); }
    finally { active.delete(name); }
    return entry;
  }

  function call(node, emitArgument, shadowed = false, owner = null, loopDepth = 0, resolveArray = null, resultUsed = true) {
    if (node.optional || node.callee.type !== 'Identifier' || shadowed ||
        node.arguments.some(arg => arg.type === 'SpreadElement')) {
      fail('Scalar calls require an unshadowed direct immutable helper binding and positional numbers', node);
    }
    const arrays = node.arguments.map(arg => resolveArray?.(arg) ?? null);
    if (!checkedArrays && arrays.some(Boolean)) fail('Typed-array helpers require checkedIndexing', node);
    const parameterTypes = arrays.map(array => array?.type ?? 'f64');
    const entry = requireHelper(node.callee.name, node, parameterTypes);
    if (resultUsed && entry.resultType === 'void') fail('A void helper call cannot be used as a Number', node);
    // Count lexical loop nesting THROUGH calls, not just within each function.
    // Acyclic helpers compile once, but every call site's nesting is checked.
    if (control) control.checkDepth(loopDepth + entry.loopDepth, node);
    if (owner) {
      owner.loopDepth = Math.max(owner.loopDepth, loopDepth + entry.loopDepth);
      owner.work += entry.work;
      owner.depth = Math.max(owner.depth, entry.depth + 1);
      if (owner.work > 4096 || owner.depth > 32) fail('Scalar helper expanded call graph exceeds the work/depth limit', node);
    }
    if (node.arguments.length !== entry.arity) fail(`Scalar helper ${entry.name} argument count differs from its declaration`, node);
    // Pointer/length pairs refer to the caller's SAME transaction storage.
    // Never repack a view or publish from a helper. Propagate effects through
    // every call site, including repeated use of a cached specialization.
    const argumentsCode = node.arguments.flatMap((arg, index) => {
      const array = arrays[index];
      if (!array) return emitArgument(arg);
      array.mark(entry.effects[index]);
      return [...arrayPointer(array), ...arrayLength(array)];
    });
    return [...argumentsCode, 0x10, ...u32(entry.index),
      ...(!resultUsed && entry.resultType === 'f64' ? [0x1a] : [])];
  }

  function compileFunction(fn, owner) {
    owner.work = 1;
    owner.depth = 1;
    owner.loopDepth = 0;
    let environment = new Map();
    let nextParameter = 0;
    fn.params.forEach((param, position) => {
      if (environment.has(param.name)) fail('Scalar helper parameters must be distinct', param);
      const type = owner.parameterTypes[position], index = nextParameter++;
      if (type === 'f64') environment.set(param.name, { index, mutable: true });
      else {
        if (!['f32[]', 'f64[]', ...Object.keys(INTEGER_ARRAY_LAYOUTS)].includes(type))
          fail('Unsupported helper array storage', param);
        const effect = owner.effects[position];
        environment.set(param.name, { type, index, lengthIndex: nextParameter++, mutable: false,
          mark({ read, write }) { effect.read ||= read || write; effect.write ||= write; } });
      }
    });
    let localCount = 0, loopDepth = 0, controlDepth = 0;
    const loopControls = [];
    function lookup(node, writing = false) {
      if (node?.type !== 'Identifier' || !environment.has(node.name)) fail('Scalar helpers cannot access captured or non-scalar bindings', node);
      const binding = environment.get(node.name);
      if (binding === null) fail(`Helper binding ${node.name} is used before initialization`, node);
      if (binding.type) fail('Typed-array references cannot be reassigned, escaped or coerced to Numbers', node);
      if (writing && !binding.mutable) fail(`Cannot assign to constant helper binding ${node.name}`, node);
      return binding.index;
    }
    const allocateLocal = () => owner.wasmTypes.length + localCount++;
    function resolveArray(node) {
      const binding = node?.type === 'Identifier' ? environment.get(node.name) : null;
      return binding?.type ? binding : null;
    }
    const arrayReferences = createArrayReferenceCompiler({resolveArray, allocateLocal, condition, fail, enabled:checkedArrays});
    function target(node, depth = 0) {
      const array = node?.type === 'MemberExpression' && !node.optional && node.computed
        ? resolveArray(node.object) : null;
      if (!array) fail('Helper array access requires a typed-array parameter', node);
      const value = expression(node.property, depth + 1), local = allocateLocal();
      // The check precedes RHS effects, uses THIS view's length, and does not
      // truncate a fractional/NaN/infinite subscript into a different property.
      const setup = [...value, ...set(local), ...get(local), ...number(0), 0x66,
        ...get(local), ...arrayLength(array), 0xb8, 0x63, 0x71,
        ...get(local), ...get(local), 0x9d, 0x61, 0x71,
        0x45, 0x04, 0x40, 0x00, 0x0b];
      const integer = INTEGER_ARRAY_LAYOUTS[array.type];
      const alignment = integer?.alignment ?? (array.type === 'f64[]' ? 3 : 2);
      return { array, integer, alignment, setup, address: [...arrayPointer(array),
        ...get(local), 0xab, 0x41, alignment, 0x74, 0x6a] };
    }
    function load(access, prepared = false) {
      const { array, integer, alignment } = access;
      array.mark({ read: true });
      return [...(prepared ? [] : access.setup), ...access.address,
        ...(integer ? [integer.load, alignment, 0, integer.signed ? 0xb7 : 0xb8]
          : array.type === 'f32[]' ? [0x2a, 2, 0, 0xbb] : [0x2b, 3, 0])];
    }
    function store(access, value) {
      const { array, integer, alignment } = access;
      array.mark({ read: true, write: true });
      const payload = integer ? (array.type === 'u8c[]' ? emitToUint8Clamp : emitToUint32)(value, allocateLocal) : value;
      return [...access.setup, ...access.address, ...payload,
        ...(integer ? [integer.store, alignment, 0]
          : array.type === 'f32[]' ? [0xb6, 0x38, 2, 0] : [0x39, 3, 0])];
    }
    function binary(operator, left, right) {
      return emitBitwiseBinary(operator, left, right, () => owner.wasmTypes.length + localCount++)
        ?? [...left, ...right, OPS[operator]];
    }
    function expression(node, depth = 0) {
      if (!node || depth > 128) fail('Scalar helper expression exceeds the nesting limit', node);
      if (node.type === 'Literal' && typeof node.value === 'number') return number(node.value);
      if (node.type === 'Identifier') return get(lookup(node));
      if (node.type === 'MemberExpression') {
        const array = resolveArray(node.object);
        if (array && !node.optional && !node.computed && node.property.name === 'length')
          return [...arrayLength(array), 0xb8];
        return load(target(node, depth));
      }
      if (node.type === 'UnaryExpression' && ['+', '-', '~'].includes(node.operator)) {
        const operand = expression(node.argument, depth + 1);
        if (node.operator === '~') return emitBitwiseNot(operand, () => owner.wasmTypes.length + localCount++);
        return [...operand, ...(node.operator === '-' ? [0x9a] : [])];
      }
      if (node.type === 'BinaryExpression' &&
          (Object.hasOwn(OPS, node.operator) || Object.hasOwn(BITWISE_OPS, node.operator))) {
        return binary(node.operator, expression(node.left, depth + 1), expression(node.right, depth + 1));
      }
      if (node.type === 'ConditionalExpression') {
        return [...condition(node.test, depth + 1), 0x04, F64,
          ...expression(node.consequent, depth + 1), 0x05, ...expression(node.alternate, depth + 1), 0x0b];
      }
      if (node.type === 'CallExpression') {
        const intrinsic = intrinsics?.call(node, arg => expression(arg, depth + 1),
          environment.has('Math') || helperSources.has('Math'), () => owner.wasmTypes.length + localCount++);
        if (intrinsic) return intrinsic;
        return call(node, arg => expression(arg, depth + 1), environment.has(node.callee.name), owner, loopDepth, resolveArray);
      }
      fail(`Scalar helper expression ${node.type} is not closed`, node);
    }
    function condition(node, depth = 0) {
      if (!node || depth > 128) fail('Scalar helper predicate exceeds the nesting limit', node);
      if (node.type === 'Literal' && typeof node.value === 'boolean') return [0x41, Number(node.value)];
      if (node.type === 'BinaryExpression' && Object.hasOwn(COMPARE, node.operator)) {
        return [...expression(node.left, depth + 1), ...expression(node.right, depth + 1), COMPARE[node.operator]];
      }
      if (node.type === 'UnaryExpression' && node.operator === '!') return [...condition(node.argument, depth + 1), 0x45];
      if (node.type === 'LogicalExpression' && node.operator === '&&') {
        return [...condition(node.left, depth + 1), 0x04, I32, ...condition(node.right, depth + 1), 0x05, 0x41, 0, 0x0b];
      }
      if (node.type === 'LogicalExpression' && node.operator === '||') {
        return [...condition(node.left, depth + 1), 0x04, I32, 0x41, 1, 0x05, ...condition(node.right, depth + 1), 0x0b];
      }
      if (node.type === 'ConditionalExpression') {
        return [...condition(node.test, depth + 1), 0x04, I32,
          ...condition(node.consequent, depth + 1), 0x05, ...condition(node.alternate, depth + 1), 0x0b];
      }
      return [...number(0), ...expression(node, depth + 1), 0x99, 0x63];
    }
    function effects(node, depth) {
      if (!node) return [];
      if (depth > 128) fail('Scalar helper effects exceed the nesting limit', node);
      if (node.type === 'SequenceExpression') {
        return node.expressions.flatMap(item => effects(item, depth + 1));
      }
      return block([{ type: 'ExpressionStatement', expression: node }], depth + 1).bytes;
    }
    function compileLoop(node, depth) {
      if (!control) fail('Scalar helper loops require generalControl', node);
      const parent = environment;
      environment = new Map(parent);
      loopDepth++;
      owner.loopDepth = Math.max(owner.loopDepth, loopDepth);
      try {
        control.recordLoop(node, loopDepth, fn.id.name);
        // The initializer's TDZ covers ALL its declarations. Its bindings live
        // through test/body/update, but a body's shadow cannot leak into update.
        const initial = node.init?.type === 'VariableDeclaration'
          ? block([node.init], depth + 1, true).bytes : effects(node.init, depth + 1);
        const predicate = node.test ? condition(node.test) : [0x41, 1];
        const parentDepth = controlDepth;
        loopControls.push({ breakDepth: parentDepth, continueDepth: parentDepth + 2 });
        controlDepth += 3;
        let body;
        try { body = block(node.body.type === 'BlockStatement' ? node.body.body : [node.body], depth + 1); }
        finally { controlDepth = parentDepth; loopControls.pop(); }
        const update = effects(node.update, depth + 1);
        const postTest = node.type === 'DoWhileStatement';
        const test = [...predicate, 0x45, 0x0d, 1];
        return {
          bytes: [...initial, 0x02, 0x40, 0x03, 0x40,
            ...(postTest ? [] : test), ...control.enterLoop(),
            0x02, 0x40, ...body.bytes, 0x0b, ...update,
            ...(postTest ? test : []), 0x0c, 0, 0x0b, 0x0b],
          // Conservatively allow a loop to exit. A numeric return after the
          // loop is required unless the surrounding branches already return.
          // Break/continue are consumed HERE, never treated as function returns.
          completions: new Set(['normal', ...(body.completions.has('return') ? ['return'] : [])]),
        };
      } finally { loopDepth--; environment = parent; }
    }
    function block(statements, depth = 0, retainScope = false) {
      if (depth > 128) fail('Scalar helper statements exceed the nesting limit', fn);
      const parent = environment;
      environment = new Map(parent);
      const declared = new Set();
      const bytes = [];
      // Track abrupt completion through branches. A return syntactically AFTER
      // an unconditional break/continue does not establish a numeric result.
      const completions = new Set(['normal']);
      const advance = next => {
        if (completions.delete('normal')) for (const kind of next) completions.add(kind);
      };
      try {
        // Install the whole block's TDZ before compiling any initializer or call.
        for (const statement of statements) {
          if (statement.type !== 'VariableDeclaration') continue;
          if (!['const', 'let'].includes(statement.kind)) fail('Scalar helpers require lexical locals', statement);
          for (const variable of statement.declarations) {
            if (variable.id.type !== 'Identifier' || !variable.init || declared.has(variable.id.name)) {
              fail('Scalar helper locals must be distinct initialized identifiers', variable);
            }
            declared.add(variable.id.name);
            environment.set(variable.id.name, null);
          }
        }
        const child = node => block(node.type === 'BlockStatement' ? node.body : [node], depth + 1);
        for (const statement of statements) {
          if (++statementCount > 4096) fail('Reachable scalar helpers exceed the statement limit', statement);
          if (statement.type === 'VariableDeclaration') {
            for (const variable of statement.declarations) {
              const reference = arrayReferences.declare(variable.init, statement.kind === 'let');
              if (reference) {
                environment.set(variable.id.name, reference.binding);
                bytes.push(...reference.bytes);
                continue;
              }
              const value = expression(variable.init);
              const index = owner.wasmTypes.length + localCount++;
              environment.set(variable.id.name, { index, mutable: statement.kind === 'let' });
              bytes.push(...value, ...set(index));
            }
          } else if (statement.type === 'ReturnStatement') {
            if (owner.resultType === 'f64' && !statement.argument) fail('Numeric helper returns cannot mix with void returns', statement);
            bytes.push(...(statement.argument ? expression(statement.argument) : []), 0x0f);
            advance(['return']);
          } else if (statement.type === 'BlockStatement') {
            const nested = child(statement);
            bytes.push(...nested.bytes);
            advance(nested.completions);
          } else if (statement.type === 'IfStatement') {
            const predicate = condition(statement.test);
            controlDepth++;
            let consequent, alternate;
            try {
              consequent = child(statement.consequent);
              alternate = statement.alternate ? child(statement.alternate) : null;
            } finally { controlDepth--; }
            bytes.push(...predicate, 0x04, 0x40, ...consequent.bytes);
            if (alternate) bytes.push(0x05, ...alternate.bytes);
            bytes.push(0x0b);
            advance([...consequent.completions, ...(alternate?.completions ?? ['normal'])]);
          } else if (control && ['ForStatement', 'WhileStatement', 'DoWhileStatement'].includes(statement.type)) {
            const nested = compileLoop(statement, depth);
            bytes.push(...nested.bytes);
            advance(nested.completions);
          } else if (control && ['BreakStatement', 'ContinueStatement'].includes(statement.type)) {
            const target = loopControls.at(-1);
            if (statement.label || !target) fail('Helper loop control requires an enclosing loop and no label', statement);
            const label = statement.type === 'BreakStatement' ? target.breakDepth : target.continueDepth;
            bytes.push(0x0c, ...u32(controlDepth - 1 - label));
            advance([statement.type === 'BreakStatement' ? 'break' : 'continue']);
          } else if (control && statement.type === 'EmptyStatement') {
            // Empty bodies/for clauses still spend credit at the body boundary.
          } else {
            const update = statement.type === 'ExpressionStatement' ? statement.expression : null;
            if (update?.type === 'CallExpression' && update.callee.type === 'Identifier') {
              bytes.push(...call(update, arg => expression(arg), environment.has(update.callee.name), owner, loopDepth, resolveArray, false));
            } else if (control && update?.type === 'SequenceExpression') {
              bytes.push(...effects(update, depth + 1));
            } else if (update?.type === 'UpdateExpression' && ['++', '--'].includes(update.operator)) {
              if (update.argument.type === 'MemberExpression') {
                const access = target(update.argument);
                bytes.push(...store(access, binary(update.operator[0], load(access, true), number(1))));
                continue;
              }
              const local = lookup(update.argument, true);
              bytes.push(...get(local), ...number(1), OPS[update.operator[0]], ...set(local));
            } else if (update?.type === 'AssignmentExpression' &&
                ['=', '+=', '-=', '*=', '/=', '&=', '|=', '^=', '<<=', '>>=', '>>>='].includes(update.operator)) {
              if (update.left.type === 'MemberExpression') {
                const access = target(update.left), value = expression(update.right);
                bytes.push(...store(access, update.operator === '=' ? value
                  : binary(update.operator.slice(0, -1), load(access, true), value)));
                continue;
              }
              const reference = resolveArray(update.left);
              if (reference) {
                bytes.push(...arrayReferences.rebind(reference, update));
                continue;
              }
              const local = lookup(update.left, true);
              const value = expression(update.right);
              bytes.push(...(update.operator === '=' ? value
                : binary(update.operator.slice(0, -1), get(local), value)), ...set(local));
            } else {
              fail(`Scalar helper statement ${statement.type} is not closed`, statement);
            }
          }
        }
        return { bytes, completions, returns: completions.size === 1 && completions.has('return') };
      } finally { if (!retainScope) environment = parent; }
    }
    const body = block(fn.body.body);
    arrayReferences.finish();
    if (owner.resultType === 'f64' && !body.returns) fail(`Scalar helper ${fn.id.name} must return a number on every path`, fn);
    // Every reachable path returns. unreachable makes the result type explicit
    // to the Wasm validator even when all returns occur in nested if branches.
    return [...(localCount ? [1, ...u32(localCount), F64] : [0]), ...body.bytes, ...(owner.resultType === 'f64' ? [0x00] : []), 0x0b];
  }

  return {
    call,
    finish() {
      return {
        types: entries.map(entry => [0x60, ...u32(entry.wasmTypes.length), ...entry.wasmTypes,
          ...(entry.resultType === 'f64' ? [1, F64] : [0])]),
        functions: entries.map(entry => u32(entry.index)),
        bodies: entries.map(entry => [...u32(entry.body.length), ...entry.body]),
        // Build-time provenance only: internal calls do not change the host ABI.
        helpers: Object.freeze(entries.map(({ name, arity, parameterTypes, resultType }) => Object.freeze({
          name, arity, ...(parameterTypes.some(type => type !== 'f64') || resultType === 'void'
            ? { parameterTypes: Object.freeze([...parameterTypes]), resultType } : {}),
        }))),
      };
    },
  };
}
