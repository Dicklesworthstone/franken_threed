/**
 * Closed f64 update-island compiler. This emits executable Wasm, not a route label.
 *
 * Admitted shape: pure scalar setup, one counted loop, scalar arithmetic
 * and fixed-stride Float32Array/Float64Array updates, including conditionals.
 * Scalar accumulators and a final numeric return execute in source iteration order.
 * No calls, escapes, implicit numeric
 * conversions or arithmetic reassociation.
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
 * parameterTypes is positional: 'f32[]', 'f64[]' or scalar 'f64'. A matching
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
      parameterTypes.some(type => !['f32[]', 'f64[]', 'f64'].includes(type))) {
    fail('Supply one f32[], f64[] or f64 type per parameter (at most 64)', fn, 'INVALID_KERNEL_ABI');
  }
  const params = new Map();
  fn.params.forEach((param, index) => {
    if (param.type !== 'Identifier' || params.has(param.name)) {
      fail('Parameters must be distinct identifiers without defaults or destructuring', param);
    }
    params.set(param.name, { name: param.name, type: parameterTypes[index], index });
  });
  const resultNode = fn.body.body.at(-1)?.type === 'ReturnStatement' ? fn.body.body.at(-1) : null;
  const loopPosition = fn.body.body.length - (resultNode ? 2 : 1);
  const prelude = fn.body.body.slice(0, loopPosition);
  const loop = fn.body.body[loopPosition];
  if (loop?.type !== 'ForStatement' || prelude.some(node => node.type !== 'VariableDeclaration')) {
    fail('Function body must contain scalar declarations followed by one counted for loop', fn.body);
  }
  if (resultNode && !resultNode.argument) fail('Final return must be a numeric expression', resultNode);
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
      !['f32[]', 'f64[]'].includes(boundParam?.type) || !member(bound, boundParam.name, 'length', false)) {
    fail('Loop condition must be index < arrayParameter.length', loop.test || loop);
  }
  const update = loop.update;
  let loopStride;
  if (update?.type === 'UpdateExpression' && update.operator === '++' &&
      update.argument.type === 'Identifier' && update.argument.name === indexName) {
    loopStride = 1;
  } else if (update?.type === 'AssignmentExpression' && update.operator === '+=' &&
      update.left.type === 'Identifier' && update.left.name === indexName &&
      update.right.type === 'Literal' && Number.isInteger(update.right.value) &&
      update.right.value >= 1 && update.right.value <= 16) {
    loopStride = update.right.value;
  } else {
    fail('Loop must increment the index by a constant stride between 1 and 16', update || loop);
  }

  const countLocal = params.size;
  const indexLocal = countLocal + 1;
  let temporaries = new Map();
  let temporaryCount = 0;
  let statementCount = 0;
  const mutableLocals = new Set();
  const scalarWrites = new Set();
  const reads = new Set();
  const writes = new Set();
  const indexed = new Set();
  const minimumLengths = new Map();
  let inLoop = false;
  const arrayParameter = (node, writing = false) => {
    const param = params.get(node?.object?.name);
    if (!['f32[]', 'f64[]'].includes(param?.type) || node?.type !== 'MemberExpression' ||
        node.optional || !node.computed || node.object.type !== 'Identifier') {
      fail('Array access must use the current disjoint loop record', node);
    }
    if (node.property.type === 'Literal' && Number.isInteger(node.property.value) &&
        node.property.value >= 0 && node.property.value < 65536) {
      if (writing) fail('Fixed-index uniform arrays are read-only', node);
      const minimum = node.property.value + 1;
      minimumLengths.set(param.name, Math.max(minimumLengths.get(param.name) ?? 0, minimum));
      return { ...param, elementOffset: node.property.value, fixed: true };
    }
    if (!inLoop) fail('Loop-indexed access is unavailable before the loop', node);
    let elementOffset;
    if (node.property.type === 'Identifier' && node.property.name === indexName) {
      elementOffset = 0;
    } else if (node.property.type === 'BinaryExpression' && node.property.operator === '+' &&
        node.property.left.type === 'Identifier' && node.property.left.name === indexName &&
        node.property.right.type === 'Literal' && Number.isInteger(node.property.right.value) &&
        node.property.right.value >= 0 && node.property.right.value < loopStride) {
      elementOffset = node.property.right.value;
    } else {
      fail('Array offset must be a constant within the current loop stride', node);
    }
    indexed.add(param.name);
    return { ...param, elementOffset, fixed: false };
  };
  const alignment = param => param.type === 'f32[]' ? 2 : 3;
  const memoryOffset = param => u32(param.elementOffset * (param.type === 'f32[]' ? 4 : 8));
  const address = param => param.fixed ? get(param.index)
    : [...get(param.index), ...get(indexLocal), 0x41, alignment(param), 0x74, 0x6a];
  const load = param => {
    reads.add(param.name);
    // JavaScript reads float32 storage as a Number. Promote before arithmetic;
    // using f32 operators here would introduce extra rounding at every operator.
    return param.type === 'f32[]'
      ? [...address(param), 0x2a, 2, ...memoryOffset(param), 0xbb] // f32.load; f64.promote_f32
      : [...address(param), 0x2b, 3, ...memoryOffset(param)]; // f64.load
  };
  function expression(node, depth = 0) {
    if (!node || depth > 128) fail('Expression nesting exceeds the admitted bound', node);
    if (node.type === 'Literal' && typeof node.value === 'number') return number(node.value);
    if (node.type === 'Identifier') {
      if (node.name === indexName) {
        if (!inLoop) fail('Loop index is not in scope outside the loop', node);
        return [...get(indexLocal), 0xb8]; // f64.convert_i32_u
      }
      const param = params.get(node.name);
      if (param?.type === 'f64') return get(param.index);
      if (temporaries.has(node.name)) {
        const local = temporaries.get(node.name);
        if (local === null) fail(`Binding ${node.name} is used before initialization`, node);
        return get(local);
      }
      fail(`Unresolved or non-scalar binding ${node.name}`, node);
    }
    // The bound view's intrinsic length is already guarded and cannot change
    // during an import-free call over fixed, unshared buffers.
    if (member(node, boundParam.name, 'length', false)) return [...get(countLocal), 0xb8];
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

  function mutableScalar(node) {
    if (node?.type !== 'Identifier' || node.name === indexName) {
      fail('Scalar updates require a mutable local or scalar parameter, not the loop index', node);
    }
    if (temporaries.has(node.name)) {
      const local = temporaries.get(node.name);
      if (local === null) fail(`Binding ${node.name} is used before initialization`, node);
      if (!mutableLocals.has(local)) fail(`Cannot assign to constant binding ${node.name}`, node);
      return local;
    }
    const param = params.get(node.name);
    if (param?.type === 'f64') return param.index;
    fail(`Unresolved or non-scalar assignment ${node.name}`, node);
  }

  function compileBlock(statements, conditional = false, depth = 0, retainScope = false) {
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
            if (statement.kind === 'let') mutableLocals.add(local);
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
        if (assignment?.type === 'UpdateExpression' && ['++', '--'].includes(assignment.operator)) {
          const local = mutableScalar(assignment.argument);
          bytes.push(...get(local), ...number(1), OPS[assignment.operator[0]], ...set(local));
          scalarWrites.add(local);
          continue;
        }
        if (assignment?.type !== 'AssignmentExpression' ||
            !['=', '+=', '-=', '*=', '/='].includes(assignment.operator)) {
          fail('Loop statements must initialize scalars, branch, or assign scalars/array[index]', statement);
        }
        if (assignment.left.type === 'Identifier') {
          const local = mutableScalar(assignment.left);
          bytes.push(...(assignment.operator === '=' ? [] : get(local)), ...expression(assignment.right),
            ...(assignment.operator === '=' ? [] : [OPS[assignment.operator[0]]]), ...set(local));
          scalarWrites.add(local);
          continue;
        }
        const target = arrayParameter(assignment.left, true);
        const value = expression(assignment.right);
        const compound = assignment.operator !== '=';
        bytes.push(...address(target),
          ...(compound ? load(target) : []), ...value,
          ...(compound ? [OPS[assignment.operator[0]]] : []),
          // Round at EACH float32 store, including stores read again in this loop.
          ...(target.type === 'f32[]' ? [0xb6, 0x38, 2] : [0x39, 3]), ...memoryOffset(target));
        writes.add(target.name);
        // A skipped store must preserve the original element, not stale private
        // Wasm memory from a previous invocation. Mark it as a packing input.
        // Strided stores may leave other record channels untouched.
        if (conditional || loopStride > 1) reads.add(target.name);
      }
      return bytes;
    } finally {
      if (!retainScope) temporaries = parentScope;
    }
  }

  const statements = loop.body.type === 'BlockStatement' ? loop.body.body : [loop.body];
  const setup = compileBlock(prelude, false, 0, true);
  inLoop = true;
  const instructions = compileBlock(statements);
  inLoop = false;
  const result = resultNode ? expression(resultNode.argument) : [];
  if (writes.size === 0 && !resultNode) fail('Kernel must produce an array output or numeric return', loop.body);
  for (const name of minimumLengths.keys()) {
    if (writes.has(name)) fail('Uniform reads must not depend on an array written by the loop', loop.body);
  }
  // v4 separates uniform extents from streamed record extents. Older kernels
  // keep their original ABI and deterministic bytes when this is not needed.
  // v5 records ordered scalar state and optional results. Array access extents
  // do not authorize parallelizing or reassociating loop-carried accumulators.
  const orderedAbi = scalarWrites.size > 0 || resultNode !== null;
  const extentAbi = orderedAbi || prelude.length > 0 || minimumLengths.size > 0;

  const manifest = {
    version: orderedAbi ? 5 : extentAbi ? 4 : loopStride > 1 ? 3 : parameterTypes.includes('f32[]') ? 2 : 1,
    kind: extentAbi || loopStride > 1 || parameterTypes.includes('f32[]') ? 'closed-numeric-loop' : 'closed-f64-loop',
    ...(extentAbi || loopStride > 1 ? { loopStride } : {}),
    ...(orderedAbi ? { resultType: resultNode ? 'f64' : 'void', iterationSemantics: 'ordered' } : {}),
    functionName: fn.id.name,
    sourceName: String(sourceName),
    sourceSpan: { start: fn.start, end: fn.end },
    parameters: [...params.values()].map(param => ({
      name: param.name, type: param.type,
      read: reads.has(param.name), write: writes.has(param.name),
      ...(extentAbi && param.type !== 'f64' ? { access: {
        indexed: indexed.has(param.name), minimumLength: minimumLengths.get(param.name) ?? 0,
      } } : {}),
    })),
    boundParameter: boundParam.index,
    maxMemoryPages,
    numericSemantics: 'f64-operator-order',
    automaticRouteAdmission: false,
  };
  const types = [...parameterTypes.map(type => type === 'f64' ? F64 : I32), I32];
  const locals = [[...u32(1), I32]];
  if (temporaryCount) locals.push([...u32(temporaryCount), F64]);
  const body = [
    ...vector(locals),
    ...setup,
    0x02, 0x40, 0x03, 0x40, // block; loop
    ...get(indexLocal), ...get(countLocal), 0x4f, 0x0d, 1, // break if i >= count
    ...instructions,
    ...get(indexLocal), 0x41, loopStride, 0x6a, ...set(indexLocal), 0x0c, 0,
    0x0b, 0x0b, ...result, 0x0b,
  ];
  const wasm = new Uint8Array([
    0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...section(1, vector([[0x60, ...vector(types.map(type => [type])), ...(resultNode ? [1, F64] : [0])]])),
    ...section(3, vector([[0]])),
    ...section(5, vector([[1, ...u32(1), ...u32(maxMemoryPages)]])),
    ...section(7, vector([[...text('run'), 0, 0], [...text('memory'), 2, 0]])),
    ...section(10, vector([[...u32(body.length), ...body]])),
    ...section(0, [...text(NUMERIC_KERNEL_SECTION), ...new TextEncoder().encode(JSON.stringify(manifest))]),
  ]);
  for (const parameter of manifest.parameters) {
    if (parameter.access) Object.freeze(parameter.access);
    Object.freeze(parameter);
  }
  Object.freeze(manifest.parameters);
  Object.freeze(manifest.sourceSpan);
  return Object.freeze({ wasm, manifest: Object.freeze(manifest) });
}
