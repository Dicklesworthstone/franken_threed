/**
 * Closed f64 update-island compiler. This emits executable Wasm, not a route label.
 *
 * Admitted shape: one synchronous function, one counted loop, scalar arithmetic
 * and same-index Float64Array reads/writes, including closed conditionals.
 * No calls, escapes, implicit numeric
 * conversions, cross-element dependencies, or arithmetic reassociation.
 * Runtime shape/ownership guards live in numeric_kernel_runtime.mjs. This narrow
 * opt-in ABI is not proof of whole-application closure or a speedup claim.
 */
import * as acorn from 'acorn';

export const NUMERIC_KERNEL_SECTION = 'f3d.numeric-kernel';
const F64 = 0x7c;
const I32 = 0x7f;
const OPS = Object.freeze({ '+': 0xa0, '-': 0xa1, '*': 0xa2, '/': 0xa3 });
const COMPARE = Object.freeze({ '===': 0x61, '!==': 0x62, '<': 0x63, '>': 0x64, '<=': 0x65, '>=': 0x66 });
const DEFAULT_MAX_PAGES = 1024;

export class NumericKernelCompileError extends Error {
  constructor(code, message, node = null) {
    super(`${code}: ${message}`);
    this.name = 'NumericKernelCompileError';
    this.code = code;
    this.span = node ? {
      start: node.start, end: node.end,
      line: node.loc?.start.line, column: node.loc?.start.column,
    } : null;
  }
}

function fail(message, node, code = 'KERNEL_NOT_CLOSED') {
  throw new NumericKernelCompileError(code, message, node);
}

function u32(value) {
  const bytes = [];
  do {
    const byte = value & 0x7f;
    value >>>= 7;
    bytes.push(byte | (value ? 0x80 : 0));
  } while (value);
  return bytes;
}

function vector(items) { return [...u32(items.length), ...items.flat()]; }
function text(value) {
  const bytes = new TextEncoder().encode(value);
  return [...u32(bytes.length), ...bytes];
}
function section(id, bytes) { return [id, ...u32(bytes.length), ...bytes]; }
function get(local) { return [0x20, ...u32(local)]; }
function set(local) { return [0x21, ...u32(local)]; }
function number(value) {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, true);
  return [0x44, ...bytes];
}
function member(node, object, property, computed) {
  return node?.type === 'MemberExpression' && !node.optional &&
    node.computed === computed && node.object.type === 'Identifier' &&
    node.object.name === object && node.property.type === 'Identifier' &&
    node.property.name === property;
}

/**
 * Compile a single source function into a standalone, import-free Wasm module.
 * parameterTypes is positional and contains only 'f64[]' or 'f64'. A matching
 * manifest is embedded in the module, so runtime ABI metadata cannot drift from
 * a separately loaded JSON sidecar. The returned bytes are deterministic.
 */
export function compileNumericKernel(source, {
  parameterTypes,
  sourceName = '<numeric-kernel>',
  maxMemoryPages = DEFAULT_MAX_PAGES,
} = {}) {
  if (typeof source !== 'string' || source.length > 65536) {
    fail('Source must be a string of at most 65536 characters', null, 'INVALID_KERNEL_SOURCE');
  }
  if (!Number.isInteger(maxMemoryPages) || maxMemoryPages < 1 || maxMemoryPages > 16384) {
    fail('maxMemoryPages must be between 1 and 16384', null, 'INVALID_KERNEL_ABI');
  }
  let ast;
  try {
    ast = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module', locations: true });
  } catch (error) {
    fail(error.message, null, 'INVALID_KERNEL_SOURCE');
  }
  if (ast.body.length !== 1) fail('Expected exactly one function declaration', ast);
  const declaration = ast.body[0];
  const fn = declaration.type === 'ExportNamedDeclaration' ? declaration.declaration : declaration;
  if (fn?.type !== 'FunctionDeclaration' || !fn.id || fn.async || fn.generator) {
    fail('Only named synchronous non-generator functions are admitted', declaration);
  }
  if (!Array.isArray(parameterTypes) || parameterTypes.length !== fn.params.length ||
      parameterTypes.length > 64 ||
      parameterTypes.some(type => type !== 'f64[]' && type !== 'f64')) {
    fail('Supply one f64[] or f64 type per parameter (at most 64)', fn, 'INVALID_KERNEL_ABI');
  }
  const params = new Map();
  fn.params.forEach((param, index) => {
    if (param.type !== 'Identifier' || params.has(param.name)) {
      fail('Parameters must be distinct identifiers without defaults or destructuring', param);
    }
    params.set(param.name, { name: param.name, type: parameterTypes[index], index });
  });
  if (fn.body.body.length !== 1 || fn.body.body[0].type !== 'ForStatement') {
    fail('Function body must contain exactly one counted for loop', fn.body);
  }
  const loop = fn.body.body[0];
  const init = loop.init;
  if (init?.type !== 'VariableDeclaration' || init.kind !== 'let' || init.declarations.length !== 1) {
    fail('Loop must initialize a fresh let index to zero', init || loop);
  }
  const binding = init.declarations[0];
  if (binding.id.type !== 'Identifier' || binding.init?.type !== 'Literal' || binding.init.value !== 0) {
    fail('Loop index must be an identifier initialized to zero', binding);
  }
  const indexName = binding.id.name;
  if (params.has(indexName)) fail('Loop index must not shadow a parameter', binding);
  const bound = loop.test?.right;
  const boundParam = params.get(bound?.object?.name);
  if (loop.test?.type !== 'BinaryExpression' || loop.test.operator !== '<' ||
      loop.test.left.type !== 'Identifier' || loop.test.left.name !== indexName ||
      boundParam?.type !== 'f64[]' || !member(bound, boundParam.name, 'length', false)) {
    fail('Loop condition must be index < arrayParameter.length', loop.test || loop);
  }
  if (loop.update?.type !== 'UpdateExpression' || loop.update.operator !== '++' ||
      loop.update.argument.type !== 'Identifier' || loop.update.argument.name !== indexName) {
    fail('Loop update must increment the index by one', loop.update || loop);
  }

  const countLocal = params.size;
  const indexLocal = countLocal + 1;
  let temporaries = new Map();
  let temporaryCount = 0;
  let statementCount = 0;
  const reads = new Set();
  const writes = new Set();
  const arrayParameter = node => {
    const param = params.get(node?.object?.name);
    if (param?.type !== 'f64[]' || !member(node, param.name, indexName, true)) {
      fail('Array access must use exactly the current loop index', node);
    }
    return param;
  };
  const address = param => [...get(param.index), ...get(indexLocal), 0x41, 3, 0x74, 0x6a];
  const load = param => {
    reads.add(param.name);
    return [...address(param), 0x2b, 3, 0]; // f64.load align=8, offset=0
  };
  function expression(node, depth = 0) {
    if (!node || depth > 128) fail('Expression nesting exceeds the admitted bound', node);
    if (node.type === 'Literal' && typeof node.value === 'number') return number(node.value);
    if (node.type === 'Identifier') {
      if (node.name === indexName) return [...get(indexLocal), 0xb8]; // f64.convert_i32_u
      const param = params.get(node.name);
      if (param?.type === 'f64') return get(param.index);
      if (temporaries.has(node.name)) {
        const local = temporaries.get(node.name);
        if (local === null) fail(`Binding ${node.name} is used before initialization`, node);
        return get(local);
      }
      fail(`Unresolved or non-scalar binding ${node.name}`, node);
    }
    if (node.type === 'MemberExpression') return load(arrayParameter(node));
    if (node.type === 'UnaryExpression' && (node.operator === '+' || node.operator === '-')) {
      const operand = expression(node.argument, depth + 1);
      return node.operator === '-' ? [...operand, 0x9a] : operand;
    }
    if (node.type === 'BinaryExpression' && Object.hasOwn(OPS, node.operator)) {
      return [...expression(node.left, depth + 1), ...expression(node.right, depth + 1), OPS[node.operator]];
    }
    if (node.type === 'ConditionalExpression') {
      return [...condition(node.test, depth + 1), 0x04, F64,
        ...expression(node.consequent, depth + 1), 0x05,
        ...expression(node.alternate, depth + 1), 0x0b];
    }
    fail(`Unsupported expression ${node.type}; calls and implicit conversions are not closed`, node);
  }

  // Predicates produce i32, while all numeric expressions stay f64. Keeping
  // these contexts separate avoids confusing JavaScript booleans with numbers
  // in strict equality, temporary bindings, and conditional result types.
  function condition(node, depth = 0) {
    if (!node || depth > 128) fail('Predicate nesting exceeds the admitted bound', node);
    if (node.type === 'Literal' && typeof node.value === 'boolean') return [0x41, Number(node.value)];
    if (node.type === 'BinaryExpression' && Object.hasOwn(COMPARE, node.operator)) {
      return [...expression(node.left, depth + 1), ...expression(node.right, depth + 1), COMPARE[node.operator]];
    }
    if (node.type === 'UnaryExpression' && node.operator === '!') {
      return [...condition(node.argument, depth + 1), 0x45]; // i32.eqz
    }
    if (node.type === 'LogicalExpression' && node.operator === '&&') {
      return [...condition(node.left, depth + 1), 0x04, I32,
        ...condition(node.right, depth + 1), 0x05, 0x41, 0, 0x0b];
    }
    if (node.type === 'LogicalExpression' && node.operator === '||') {
      return [...condition(node.left, depth + 1), 0x04, I32, 0x41, 1, 0x05,
        ...condition(node.right, depth + 1), 0x0b];
    }
    if (node.type === 'ConditionalExpression') {
      return [...condition(node.test, depth + 1), 0x04, I32,
        ...condition(node.consequent, depth + 1), 0x05,
        ...condition(node.alternate, depth + 1), 0x0b];
    }
    // Numeric truthiness: 0 < abs(value) is false for both zero signs and NaN,
    // true for every other Number, and evaluates the source expression once.
    return [...number(0), ...expression(node, depth + 1), 0x99, 0x63];
  }

  function compileBlock(statements, conditional = false, depth = 0) {
    if (depth > 128) fail('Statement nesting exceeds the admitted bound', loop.body);
    const parentScope = temporaries;
    temporaries = new Map(parentScope);
    const declared = new Set();
    try {
      // Install lexical TDZ markers before compiling any statement. A nested
      // block's later declaration shadows its outer binding even before init.
      for (const statement of statements) {
        if (statement.type !== 'VariableDeclaration') continue;
        if (!['const', 'let'].includes(statement.kind)) fail('Only lexical scalar temporaries are admitted', statement);
        for (const variable of statement.declarations) {
          const name = variable.id.name;
          if (variable.id.type !== 'Identifier' || !variable.init || name === indexName ||
              params.has(name) || declared.has(name)) {
            fail('Temporaries must be initialized scalar bindings without parameter/index shadowing', variable);
          }
          declared.add(name);
          temporaries.set(name, null);
        }
      }
      const bytes = [];
      const child = statement => compileBlock(
        statement.type === 'BlockStatement' ? statement.body : [statement], true, depth + 1);
      for (const statement of statements) {
        if (++statementCount > 2048) fail('Loop body exceeds the statement limit', statement);
        if (statement.type === 'VariableDeclaration') {
          for (const variable of statement.declarations) {
            const initializer = expression(variable.init);
            const local = indexLocal + 1 + temporaryCount++;
            temporaries.set(variable.id.name, local);
            bytes.push(...initializer, ...set(local));
          }
          continue;
        }
        if (statement.type === 'BlockStatement') {
          bytes.push(...compileBlock(statement.body, conditional, depth + 1));
          continue;
        }
        if (statement.type === 'IfStatement') {
          bytes.push(...condition(statement.test), 0x04, 0x40, ...child(statement.consequent));
          if (statement.alternate) bytes.push(0x05, ...child(statement.alternate));
          bytes.push(0x0b);
          continue;
        }
        const assignment = statement.type === 'ExpressionStatement' ? statement.expression : null;
        if (assignment?.type !== 'AssignmentExpression' ||
            !['=', '+=', '-=', '*=', '/='].includes(assignment.operator)) {
          fail('Loop statements must initialize scalars, branch, or assign array[index]', statement);
        }
        const target = arrayParameter(assignment.left);
        const value = expression(assignment.right);
        const compound = assignment.operator !== '=';
        bytes.push(...address(target),
          ...(compound ? load(target) : []), ...value,
          ...(compound ? [OPS[assignment.operator[0]]] : []), 0x39, 3, 0);
        writes.add(target.name);
        // A skipped store must preserve the original element, not stale private
        // Wasm memory from a previous invocation. Mark it as a packing input.
        if (conditional) reads.add(target.name);
      }
      return bytes;
    } finally {
      temporaries = parentScope;
    }
  }

  const statements = loop.body.type === 'BlockStatement' ? loop.body.body : [loop.body];
  const instructions = compileBlock(statements);
  if (writes.size === 0) fail('Kernel must produce at least one array output', loop.body);

  const manifest = {
    version: 1,
    kind: 'closed-f64-loop',
    functionName: fn.id.name,
    sourceName: String(sourceName),
    sourceSpan: { start: fn.start, end: fn.end },
    parameters: [...params.values()].map(param => ({
      name: param.name, type: param.type,
      read: reads.has(param.name), write: writes.has(param.name),
    })),
    boundParameter: boundParam.index,
    maxMemoryPages,
    numericSemantics: 'f64-operator-order',
    automaticRouteAdmission: false,
  };
  const types = [...parameterTypes.map(type => type === 'f64[]' ? I32 : F64), I32];
  const locals = [[...u32(1), I32]];
  if (temporaryCount) locals.push([...u32(temporaryCount), F64]);
  const body = [
    ...vector(locals),
    0x02, 0x40, 0x03, 0x40, // block; loop
    ...get(indexLocal), ...get(countLocal), 0x4f, 0x0d, 1, // break if i >= count
    ...instructions,
    ...get(indexLocal), 0x41, 1, 0x6a, ...set(indexLocal), 0x0c, 0,
    0x0b, 0x0b, 0x0b,
  ];
  const wasm = new Uint8Array([
    0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...section(1, vector([[0x60, ...vector(types.map(type => [type])), 0]])),
    ...section(3, vector([[0]])),
    ...section(5, vector([[1, ...u32(1), ...u32(maxMemoryPages)]])),
    ...section(7, vector([[...text('run'), 0, 0], [...text('memory'), 2, 0]])),
    ...section(10, vector([[...u32(body.length), ...body]])),
    ...section(0, [...text(NUMERIC_KERNEL_SECTION), ...new TextEncoder().encode(JSON.stringify(manifest))]),
  ]);
  for (const parameter of manifest.parameters) Object.freeze(parameter);
  Object.freeze(manifest.parameters);
  Object.freeze(manifest.sourceSpan);
  return Object.freeze({ wasm, manifest: Object.freeze(manifest) });
}
