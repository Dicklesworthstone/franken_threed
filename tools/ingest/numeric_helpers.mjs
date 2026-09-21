/**
 * Close a bounded, acyclic graph of scalar JavaScript helpers into private Wasm
 * functions. No host imports, source evaluation, captures, arrays or implicit
 * conversions are admitted. Each call evaluates arguments once, left to right;
 * a helper owns its scalar parameters and lexical locals, just as in JavaScript.
 * Number bitwise operations share the kernel's exact modulo-2^32 lowering.
 */
import * as acorn from 'acorn';
import { BITWISE_OPS, emitBitwiseBinary, emitBitwiseNot } from './numeric_integer.mjs';

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
export function createScalarHelperCompiler(helperSources, fail, intrinsics = null) {
  if (!(helperSources instanceof Map)) fail('helperSources must be a Map of immutable function declarations', null, 'INVALID_KERNEL_SOURCE');
  const entries = [];
  const compiled = new Map();
  const active = new Set();
  let sourceBytes = 0;
  let statementCount = 0;

  function requireHelper(name, callNode) {
    if (active.has(name)) fail(`Recursive scalar helper ${name} is not closed`, callNode);
    if (compiled.has(name)) return compiled.get(name);
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
    const entry = { name, arity: fn.params.length, index: entries.length + 1, body: null };
    entries.push(entry);
    compiled.set(name, entry);
    active.add(name);
    try { entry.body = compileFunction(fn, entry); }
    finally { active.delete(name); }
    return entry;
  }

  function call(node, emitArgument, shadowed = false, owner = null) {
    if (node.optional || node.callee.type !== 'Identifier' || shadowed ||
        node.arguments.some(arg => arg.type === 'SpreadElement')) {
      fail('Scalar calls require an unshadowed direct immutable helper binding and positional numbers', node);
    }
    const entry = requireHelper(node.callee.name, node);
    if (owner) {
      owner.work += entry.work;
      owner.depth = Math.max(owner.depth, entry.depth + 1);
      if (owner.work > 4096 || owner.depth > 32) fail('Scalar helper expanded call graph exceeds the work/depth limit', node);
    }
    if (node.arguments.length !== entry.arity) fail(`Scalar helper ${entry.name} argument count differs from its declaration`, node);
    return [...node.arguments.flatMap(emitArgument), 0x10, ...u32(entry.index)];
  }

  function compileFunction(fn, owner) {
    owner.work = 1;
    owner.depth = 1;
    let environment = new Map();
    fn.params.forEach((param, index) => {
      if (environment.has(param.name)) fail('Scalar helper parameters must be distinct', param);
      environment.set(param.name, { index, mutable: true });
    });
    let localCount = 0;
    function lookup(node, writing = false) {
      if (node?.type !== 'Identifier' || !environment.has(node.name)) fail('Scalar helpers cannot access captured or non-scalar bindings', node);
      const binding = environment.get(node.name);
      if (binding === null) fail(`Helper binding ${node.name} is used before initialization`, node);
      if (writing && !binding.mutable) fail(`Cannot assign to constant helper binding ${node.name}`, node);
      return binding.index;
    }
    function binary(operator, left, right) {
      return emitBitwiseBinary(operator, left, right, () => fn.params.length + localCount++)
        ?? [...left, ...right, OPS[operator]];
    }
    function expression(node, depth = 0) {
      if (!node || depth > 128) fail('Scalar helper expression exceeds the nesting limit', node);
      if (node.type === 'Literal' && typeof node.value === 'number') return number(node.value);
      if (node.type === 'Identifier') return get(lookup(node));
      if (node.type === 'UnaryExpression' && ['+', '-', '~'].includes(node.operator)) {
        const operand = expression(node.argument, depth + 1);
        if (node.operator === '~') return emitBitwiseNot(operand, () => fn.params.length + localCount++);
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
          environment.has('Math') || helperSources.has('Math'), () => fn.params.length + localCount++);
        if (intrinsic) return intrinsic;
        return call(node, arg => expression(arg, depth + 1), environment.has(node.callee.name), owner);
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
    function block(statements, depth = 0) {
      if (depth > 128) fail('Scalar helper statements exceed the nesting limit', fn);
      const parent = environment;
      environment = new Map(parent);
      const declared = new Set();
      const bytes = [];
      let returns = false;
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
              const value = expression(variable.init);
              const index = fn.params.length + localCount++;
              environment.set(variable.id.name, { index, mutable: statement.kind === 'let' });
              bytes.push(...value, ...set(index));
            }
          } else if (statement.type === 'ReturnStatement') {
            bytes.push(...expression(statement.argument), 0x0f);
            returns = true;
          } else if (statement.type === 'BlockStatement') {
            const nested = child(statement);
            bytes.push(...nested.bytes);
            returns ||= nested.returns;
          } else if (statement.type === 'IfStatement') {
            const predicate = condition(statement.test);
            const consequent = child(statement.consequent);
            const alternate = statement.alternate ? child(statement.alternate) : null;
            bytes.push(...predicate, 0x04, 0x40, ...consequent.bytes);
            if (alternate) bytes.push(0x05, ...alternate.bytes);
            bytes.push(0x0b);
            returns ||= consequent.returns && !!alternate?.returns;
          } else {
            const update = statement.type === 'ExpressionStatement' ? statement.expression : null;
            if (update?.type === 'UpdateExpression' && ['++', '--'].includes(update.operator)) {
              const local = lookup(update.argument, true);
              bytes.push(...get(local), ...number(1), OPS[update.operator[0]], ...set(local));
            } else if (update?.type === 'AssignmentExpression' &&
                ['=', '+=', '-=', '*=', '/=', '&=', '|=', '^=', '<<=', '>>=', '>>>='].includes(update.operator)) {
              const local = lookup(update.left, true);
              const value = expression(update.right);
              bytes.push(...(update.operator === '=' ? value
                : binary(update.operator.slice(0, -1), get(local), value)), ...set(local));
            } else {
              fail(`Scalar helper statement ${statement.type} is not closed`, statement);
            }
          }
        }
        return { bytes, returns };
      } finally { environment = parent; }
    }
    const body = block(fn.body.body);
    if (!body.returns) fail(`Scalar helper ${fn.id.name} must return a number on every path`, fn);
    // Every reachable path returns. unreachable makes the result type explicit
    // to the Wasm validator even when all returns occur in nested if branches.
    return [...(localCount ? [1, ...u32(localCount), F64] : [0]), ...body.bytes, 0x00, 0x0b];
  }

  return {
    call,
    finish() {
      return {
        types: entries.map(entry => [0x60, ...u32(entry.arity), ...Array(entry.arity).fill(F64), 1, F64]),
        functions: entries.map(entry => u32(entry.index)),
        bodies: entries.map(entry => [...u32(entry.body.length), ...entry.body]),
        // Build-time provenance only: internal calls do not change the host ABI.
        helpers: Object.freeze(entries.map(({ name, arity }) => Object.freeze({ name, arity }))),
      };
    },
  };
}
