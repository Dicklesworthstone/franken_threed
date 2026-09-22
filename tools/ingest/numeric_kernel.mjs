/**
 * Closed f64 update-island compiler. This emits executable Wasm, not a route label.
 *
 * Admitted shape: scalar setup/state, one or more ordered counted loops, arithmetic
 * and fixed-stride Float32Array/Float64Array updates, including conditionals.
 * Scalar accumulators and a final numeric return execute in source iteration order.
 * Closed scalar helpers execute as private Wasm functions. No host calls,
 * escapes, implicit numeric conversions or arithmetic reassociation.
 * Number bitwise operators use explicit modulo-2^32 lowering, returning f64.
 * Runtime shape/ownership guards live in numeric_kernel_runtime.mjs. This narrow
 * opt-in ABI is not proof of whole-application closure or a speedup claim.
 */
import * as acorn from 'acorn';
import { createScalarHelperCompiler } from './numeric_helpers.mjs';
import { createMathIntrinsicCompiler } from './numeric_intrinsics.mjs';
import { BITWISE_OPS, emitBitwiseBinary, emitBitwiseNot } from './numeric_integer.mjs';

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
 * helperSources optionally maps proven immutable lexical bindings to scalar
 * function declarations. Only reachable, acyclic, capture-free helpers compile.
 * These sources must match the original function's actual lexical environment;
 * specializeNumericModule establishes that proof for linked application code.
 * allowMath additionally requires a runtime resolveMath closure for the actual
 * shared lexical Math binding. Missing/changed bindings retain JavaScript.
 * checkedIndexing opts into ABI v7: arbitrary numeric subscripts, read-only
 * u16[]/u32[] topology, and source-ordered float gathers/scatters. Each executed
 * access checks its own view before conversion. The host packs full accessed
 * views and publishes no writes if any check traps. No check-elision or SIMD
 * independence is inferred; colliding scatters must remain ordered.
 * structuredLoops additionally admits nested counted loops with Number indices,
 * runtime bounds and constant positive steps (ascending or descending). Their
 * combined body entries are capped by maxNestedIterations per invocation.
 * Exhaustion traps before publication, so the existing host retains original
 * JavaScript exactly once. This is a native loop, not source unrolling, and
 * neither the array ABI nor the synchronous execution boundary changes.
 * This mode also supports unlabeled break/continue and early returns. A
 * numeric-returning kernel needs a final numeric return; a void kernel may
 * return only without a value. Skipped stores retain their original contents.
 */
export function compileNumericKernel(source, {
  parameterTypes,
  helperSources = new Map(),
  allowMath = false,
  checkedIndexing = false,
  structuredLoops = false,
  maxNestedIterations = 1000000,
  sourceName = '<numeric-kernel>',
  maxMemoryPages = DEFAULT_MAX_PAGES,
} = {}) {
  if (typeof source !== 'string' || source.length > 65536) {
    fail('Source must be a string of at most 65536 characters', null, 'INVALID_KERNEL_SOURCE');
  }
  if (!Number.isInteger(maxMemoryPages) || maxMemoryPages < 1 || maxMemoryPages > 16384) {
    fail('maxMemoryPages must be between 1 and 16384', null, 'INVALID_KERNEL_ABI');
  }
  if (typeof allowMath !== 'boolean') fail('allowMath must be a boolean', null, 'INVALID_KERNEL_ABI');
  if (typeof checkedIndexing !== 'boolean') fail('checkedIndexing must be a boolean', null, 'INVALID_KERNEL_ABI');
  if (typeof structuredLoops !== 'boolean' || (structuredLoops && !checkedIndexing)) {
    fail('structuredLoops must be a boolean and requires checkedIndexing', null, 'INVALID_KERNEL_ABI');
  }
  if (!Number.isInteger(maxNestedIterations) || maxNestedIterations < 1 || maxNestedIterations > 1000000000) {
    fail('maxNestedIterations must be between 1 and 1000000000', null, 'INVALID_KERNEL_ABI');
  }
  const arrayTypes = checkedIndexing ? ['f32[]', 'f64[]', 'u16[]', 'u32[]'] : ['f32[]', 'f64[]'];
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
      parameterTypes.some(type => type !== 'f64' && !arrayTypes.includes(type))) {
    fail('Supply one numeric ABI type per parameter (at most 64); integer arrays require checkedIndexing', fn, 'INVALID_KERNEL_ABI');
  }
  const params = new Map();
  fn.params.forEach((param, index) => {
    if (param.type !== 'Identifier' || params.has(param.name)) {
      fail('Parameters must be distinct identifiers without defaults or destructuring', param);
    }
    params.set(param.name, { name: param.name, type: parameterTypes[index], index });
  });
  const resultNode = fn.body.body.at(-1)?.type === 'ReturnStatement' ? fn.body.body.at(-1) : null;
  const loops = fn.body.body.filter(node => node.type === 'ForStatement');
  if (loops.length > 16) fail('Numeric pipeline exceeds the 16-pass limit', fn.body);
  const pipeline = loops.length > 1;
  const loopPosition = fn.body.body.length - (resultNode ? 2 : 1);
  const prelude = fn.body.body.slice(0, loopPosition);
  const loop = pipeline ? loops[0] : fn.body.body[loopPosition];
  if (!pipeline && (loop?.type !== 'ForStatement' || prelude.some(node => node.type !== 'VariableDeclaration'))) {
    fail('Function body must contain scalar declarations followed by one counted for loop', fn.body);
  }
  if (resultNode && !resultNode.argument) fail('Final return must be a numeric expression', resultNode);
  const bounds = new Map();
  // v7 appends every array's intrinsic length in parameter order. Unlike the
  // prefix ABI, an indirect access is checked against its OWN view, not the
  // loop bound or the shared Wasm allocation. Legacy signatures are unchanged.
  if (checkedIndexing) for (const param of params.values()) {
    if (param.type !== 'f64') bounds.set(param.name, params.size + bounds.size);
  }
  const passes = new Map(loops.map(node => {
    const init = node.init;
    if (init?.type !== 'VariableDeclaration' || init.kind !== 'let' || init.declarations.length !== 1) {
      fail('Loop must initialize a fresh let index to zero', init || node);
    }
    const binding = init.declarations[0];
    if (binding.id.type !== 'Identifier' || binding.init?.type !== 'Literal' || binding.init.value !== 0) {
      fail('Loop index must be an identifier initialized to zero', binding);
    }
    const indexName = binding.id.name;
    if (params.has(indexName)) fail('Loop index must not shadow a parameter', binding);
    const bound = node.test?.right;
    const boundParam = params.get(bound?.object?.name);
    if (node.test?.type !== 'BinaryExpression' || node.test.operator !== '<' ||
        node.test.left.type !== 'Identifier' || node.test.left.name !== indexName ||
        !arrayTypes.includes(boundParam?.type) || !member(bound, boundParam.name, 'length', false)) {
      fail('Loop condition must be index < arrayParameter.length', node.test || node);
    }
    const update = node.update;
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
      fail('Loop must increment the index by a constant stride between 1 and 16', update || node);
    }
    if (!bounds.has(boundParam.name)) bounds.set(boundParam.name, params.size + bounds.size);
    return [node, { indexName, boundParam, loopStride, countLocal: bounds.get(boundParam.name) }];
  }));
  // Lengths are separate guarded arguments; each pass owns a fresh zero-initialized
  // index local even when the source reuses the same lexical index name.
  let nextIndex = params.size + bounds.size;
  for (const pass of passes.values()) pass.indexLocal = nextIndex++;
  const temporaryBase = nextIndex;
  let { indexName, boundParam, loopStride, countLocal, indexLocal } = passes.get(loop);
  if (pipeline) indexName = null;

  const intrinsics = createMathIntrinsicCompiler(allowMath, fail);
  const helperCompiler = createScalarHelperCompiler(helperSources, fail, intrinsics);
  let temporaries = new Map();
  let temporaryCount = 0;
  let statementCount = 0;
  let nestedCount = 0, nestedDepth = 0, maxNestedDepth = 0, nestedFuel = null;
  const nestedIndices = new Set();
  const loopControls = [];
  let controlDepth = 0;
  const mutableLocals = new Set();
  const scalarWrites = new Set();
  const reads = new Set();
  const writes = new Set();
  const indexed = new Set();
  const indexedBounds = new Map();
  const minimumLengths = new Map();
  let inLoop = false;
  const arrayParameter = (node, writing = false, depth = 0) => {
    const param = params.get(node?.object?.name);
    if (!arrayTypes.includes(param?.type) || node?.type !== 'MemberExpression' ||
        node.optional || !node.computed || node.object.type !== 'Identifier') {
      fail('Array access must use a numeric array parameter', node);
    }
    if (checkedIndexing) {
      if (writing && ['u16[]', 'u32[]'].includes(param.type)) {
        fail('Integer topology arrays are read-only', node);
      }
      const value = expression(node.property, depth + 1);
      const local = temporaryBase + temporaryCount++;
      // Numeric -0 becomes the property key "0" in JS. Non-integers, NaN,
      // infinities and out-of-view indices must take the original JS path, not
      // be truncated/wrapped or read another parameter's packed memory.
      const setup = [...value, ...set(local),
        ...get(local), ...number(0), 0x66,
        ...get(local), ...get(bounds.get(param.name)), 0xb8, 0x63, 0x71,
        ...get(local), ...get(local), 0x9d, 0x61, 0x71,
        0x45, 0x04, 0x40, 0x00, 0x0b]; // if !valid: unreachable (transaction abort)
      return { ...param, elementOffset: 0, checkedLocal: local, setup };
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
    if (!indexedBounds.has(param.name)) indexedBounds.set(param.name, new Set());
    indexedBounds.get(param.name).add(boundParam.index);
    return { ...param, elementOffset, fixed: false };
  };
  const alignment = param => param.type === 'u16[]' ? 1 : param.type === 'f64[]' ? 3 : 2;
  const memoryOffset = param => u32(param.elementOffset * (2 ** alignment(param)));
  const address = param => param.checkedLocal !== undefined
    ? [...get(param.index), ...get(param.checkedLocal), 0xab, 0x41, alignment(param), 0x74, 0x6a]
    : param.fixed ? get(param.index)
    : [...get(param.index), ...get(indexLocal), 0x41, alignment(param), 0x74, 0x6a];
  const load = (param, prepared = false) => {
    reads.add(param.name);
    // JavaScript reads float32 storage as a Number. Promote before arithmetic;
    // using f32 operators here would introduce extra rounding at every operator.
    const prefix = [...(prepared ? [] : param.setup ?? []), ...address(param)];
    if (param.type === 'u16[]' || param.type === 'u32[]') {
      return [...prefix, param.type === 'u16[]' ? 0x2f : 0x28, alignment(param),
        ...memoryOffset(param), 0xb8]; // i32.load16_u/load; f64.convert_i32_u
    }
    return param.type === 'f32[]'
      ? [...prefix, 0x2a, 2, ...memoryOffset(param), 0xbb] // f32.load; f64.promote_f32
      : [...prefix, 0x2b, 3, ...memoryOffset(param)]; // f64.load
  };
  function binary(operator, left, right) {
    return emitBitwiseBinary(operator, left, right, () => temporaryBase + temporaryCount++)
      ?? [...left, ...right, OPS[operator]];
  }
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
    if (pipeline || checkedIndexing) {
      const name = node.object?.name;
      if (bounds.has(name) && member(node, name, 'length', false)) return [...get(bounds.get(name)), 0xb8];
    } else if (member(node, boundParam.name, 'length', false)) return [...get(countLocal), 0xb8];
    if (node.type === 'MemberExpression') return load(arrayParameter(node, false, depth));
    if (node.type === 'UnaryExpression' && ['+', '-', '~'].includes(node.operator)) {
      const operand = expression(node.argument, depth + 1);
      if (node.operator === '~') return emitBitwiseNot(operand, () => temporaryBase + temporaryCount++);
      return node.operator === '-' ? [...operand, 0x9a] : operand;
    }
    if (node.type === 'BinaryExpression' &&
        (Object.hasOwn(OPS, node.operator) || Object.hasOwn(BITWISE_OPS, node.operator))) {
      return binary(node.operator, expression(node.left, depth + 1), expression(node.right, depth + 1));
    }
    if (node.type === 'ConditionalExpression') {
      return [...condition(node.test, depth + 1), 0x04, F64,
        ...expression(node.consequent, depth + 1), 0x05,
        ...expression(node.alternate, depth + 1), 0x0b];
    }
    if (node.type === 'CallExpression') {
      const intrinsic = intrinsics.call(node, arg => expression(arg, depth + 1),
        params.has('Math') || temporaries.has('Math') || indexName === 'Math' ||
        fn.id.name === 'Math' || helperSources.has('Math'), () => temporaryBase + temporaryCount++);
      if (intrinsic) return intrinsic;
      return helperCompiler.call(node, arg => expression(arg, depth + 1),
        params.has(node.callee.name) || temporaries.has(node.callee.name) ||
        node.callee.name === indexName || node.callee.name === fn.id.name);
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
    if (node?.type !== 'Identifier' || node.name === indexName || nestedIndices.has(node.name)) {
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

  function compileNestedLoop(node, depth) {
    if (!structuredLoops || !inLoop) fail('Nested loops require structuredLoops inside an array-bounded pass', node);
    if (++nestedCount > 64 || nestedDepth >= 8) fail('Nested loops exceed the 64-loop/8-level limit', node);
    const init = node.init;
    if (init?.type !== 'VariableDeclaration' || init.kind !== 'let' || init.declarations.length !== 1) {
      fail('Nested loops require one fresh let index', init || node);
    }
    const binding = init.declarations[0], name = binding.id.name;
    if (binding.id.type !== 'Identifier' || !binding.init || name === indexName ||
        params.has(name) || temporaries.has(name)) {
      fail('Nested loop index must not shadow an enclosing binding', binding);
    }
    if (node.test?.type !== 'BinaryExpression' || !['<', '<=', '>', '>='].includes(node.test.operator) ||
        node.test.left.type !== 'Identifier' || node.test.left.name !== name) {
      fail('Nested loop test must compare its index with a numeric bound', node.test || node);
    }
    const update = node.update;
    let step, operator;
    if (update?.type === 'UpdateExpression' && ['++', '--'].includes(update.operator) &&
        update.argument.type === 'Identifier' && update.argument.name === name) {
      step = 1; operator = update.operator[0];
    } else if (update?.type === 'AssignmentExpression' && ['+=', '-='].includes(update.operator) &&
        update.left.type === 'Identifier' && update.left.name === name &&
        update.right.type === 'Literal' && typeof update.right.value === 'number' &&
        Number.isFinite(update.right.value) && update.right.value > 0) {
      step = update.right.value; operator = update.operator[0];
    } else {
      fail('Nested loop update must add or subtract a positive numeric constant', update || node);
    }
    const parentScope = temporaries;
    temporaries = new Map(parentScope);
    temporaries.set(name, null); // The initializer is inside this binding's TDZ.
    nestedDepth++;
    maxNestedDepth = Math.max(maxNestedDepth, nestedDepth);
    nestedIndices.add(name);
    try {
      const initial = expression(binding.init), local = temporaryBase + temporaryCount++;
      temporaries.set(name, local);
      if (nestedFuel === null) nestedFuel = temporaryBase + temporaryCount++;
      const predicate = condition(node.test);
      const body = compileLoopBody(node.body.type === 'BlockStatement' ? node.body.body : [node.body], depth + 1);
      // All nested loops share one zero-initialized f64 counter per call. It
      // stays an exact integer through the configured bound. Check only after
      // a true source predicate: an empty loop spends no body-entry credit.
      // Reevaluate the source bound on every iteration; never hoist a mutable
      // local/array read. Number indices also preserve fractional starts and
      // f64 stagnation, which the budget catches instead of silently wrapping.
      return [...initial, ...set(local), 0x02, 0x40, 0x03, 0x40,
        ...predicate, 0x45, 0x0d, 1,
        ...get(nestedFuel), ...number(maxNestedIterations), 0x66, 0x04, 0x40, 0x00, 0x0b,
        ...get(nestedFuel), ...number(1), 0xa0, ...set(nestedFuel),
        ...body, ...get(local), ...number(step), OPS[operator], ...set(local),
        0x0c, 0, 0x0b, 0x0b];
    } finally {
      nestedDepth--; nestedIndices.delete(name); temporaries = parentScope;
    }
  }

  function compileLoopBody(statements, depth) {
    if (!structuredLoops) return compileBlock(statements, false, depth);
    // Both loop emitters put this body inside block/loop. A third block makes
    // continue jump to the update, not directly to the next condition. Source
    // lexical blocks do not add Wasm labels; source if-statements do.
    const parentDepth = controlDepth;
    loopControls.push({breakDepth: parentDepth, continueDepth: parentDepth + 2});
    controlDepth += 3;
    try { return [0x02, 0x40, ...compileBlock(statements, true, depth), 0x0b]; }
    finally { controlDepth = parentDepth; loopControls.pop(); }
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
          if (variable.id.type !== 'Identifier' || !variable.init || name === indexName || nestedIndices.has(name) ||
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
            const local = temporaryBase + temporaryCount++;
            temporaries.set(variable.id.name, local);
            if (statement.kind === 'let') mutableLocals.add(local);
            bytes.push(...initializer, ...set(local));
          }
          continue;
        }
        if (pipeline && statement.type === 'ForStatement' && depth === 0) {
          ({ indexName, boundParam, loopStride, countLocal, indexLocal } = passes.get(statement));
          if (temporaries.has(indexName)) fail('Pipeline indices must not shadow function-scope locals', statement.init);
          inLoop = true;
          const body = statement.body.type === 'BlockStatement' ? statement.body.body : [statement.body];
          bytes.push(...emitLoop(compileLoopBody(body, depth + 1)));
          inLoop = false;
          indexName = null;
          continue;
        }
        if (statement.type === 'ForStatement') {
          bytes.push(...compileNestedLoop(statement, depth));
          continue;
        }
        if (statement.type === 'BlockStatement') {
          bytes.push(...compileBlock(statement.body, conditional, depth + 1));
          continue;
        }
        if (statement.type === 'IfStatement') {
          bytes.push(...condition(statement.test), 0x04, 0x40);
          controlDepth++;
          try {
            bytes.push(...child(statement.consequent));
            if (statement.alternate) bytes.push(0x05, ...child(statement.alternate));
          } finally { controlDepth--; }
          bytes.push(0x0b);
          continue;
        }
        if (structuredLoops && ['BreakStatement', 'ContinueStatement'].includes(statement.type)) {
          const target = loopControls.at(-1);
          if (statement.label || !target) fail('Loop control requires an enclosing loop and no label', statement);
          const label = statement.type === 'BreakStatement' ? target.breakDepth : target.continueDepth;
          bytes.push(0x0c, ...u32(controlDepth - 1 - label));
          continue;
        }
        if (structuredLoops && statement.type === 'ReturnStatement') {
          if (!!statement.argument !== !!resultNode) fail('Early return must match the kernel result type', statement);
          bytes.push(...(statement.argument ? expression(statement.argument) : []), 0x0f);
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
            !['=', '+=', '-=', '*=', '/=', '&=', '|=', '^=', '<<=', '>>=', '>>>='].includes(assignment.operator)) {
          fail('Loop statements must initialize scalars, branch, or assign scalars/array[index]', statement);
        }
        if (assignment.left.type === 'Identifier') {
          const local = mutableScalar(assignment.left);
          const value = expression(assignment.right);
          bytes.push(...(assignment.operator === '=' ? value
            : binary(assignment.operator.slice(0, -1), get(local), value)), ...set(local));
          scalarWrites.add(local);
          continue;
        }
        const target = arrayParameter(assignment.left, true);
        const value = expression(assignment.right);
        const compound = assignment.operator !== '=';
        // Evaluate a scatter destination exactly once, before the RHS. Compound
        // stores load through that same checked address, preserving collisions
        // and the Float32 rounding of EACH source-ordered store.
        bytes.push(...(target.setup ?? []), ...address(target),
          ...(compound ? binary(assignment.operator.slice(0, -1), load(target, true), value) : value),
          // Round at EACH float32 store, including stores read again in this loop.
          ...(target.type === 'f32[]' ? [0xb6, 0x38, 2] : [0x39, 3]), ...memoryOffset(target));
        writes.add(target.name);
        // A skipped store must preserve the original element, not stale private
        // Wasm memory from a previous invocation. Mark it as a packing input.
        // Strided stores may leave other record channels untouched.
        // A pipeline's accessed prefix may exceed a given pass's write extent.
        // Preserve all untouched channels/tails without guessing full coverage.
        if (checkedIndexing || pipeline || conditional || loopStride > 1) reads.add(target.name);
      }
      return bytes;
    } finally {
      if (!retainScope) temporaries = parentScope;
    }
  }

  function emitLoop(instructions) {
    return [
      0x02, 0x40, 0x03, 0x40, // block; loop
      ...get(indexLocal), ...get(countLocal), 0x4f, 0x0d, 1, // break if i >= count
      ...instructions,
      ...get(indexLocal), 0x41, loopStride, 0x6a, ...set(indexLocal), 0x0c, 0,
      0x0b, 0x0b,
    ];
  }
  let execution;
  if (pipeline) {
    // Scan the entire function scope once, including declarations between passes:
    // a later lexical declaration shadows a helper even in earlier loop bodies.
    execution = compileBlock(resultNode ? fn.body.body.slice(0, -1) : fn.body.body, false, 0, true);
  } else {
    const setup = compileBlock(prelude, false, 0, true);
    inLoop = true;
    const statements = loop.body.type === 'BlockStatement' ? loop.body.body : [loop.body];
    const instructions = compileLoopBody(statements, 0);
    inLoop = false;
    execution = [...setup, ...emitLoop(instructions)];
  }
  const result = resultNode ? expression(resultNode.argument) : [];
  if (writes.size === 0 && !resultNode) fail('Kernel must produce an array output or numeric return', loop.body);
  for (const name of minimumLengths.keys()) {
    if (writes.has(name)) fail('Uniform reads must not depend on an array written by the loop', loop.body);
  }
  // v4 separates uniform extents from streamed record extents. Older kernels
  // keep their original ABI and deterministic bytes when this is not needed.
  // v5 records ordered scalar state and optional results. Array access extents
  // do not authorize parallelizing or reassociating loop-carried accumulators.
  // v6 is a single transaction over ordered passes, not loop fusion/reassociation.
  const orderedAbi = checkedIndexing || pipeline || scalarWrites.size > 0 || resultNode !== null;
  const extentAbi = orderedAbi || prelude.length > 0 || minimumLengths.size > 0;

  const mathIntrinsics = intrinsics.requirements();
  const manifest = {
    version: checkedIndexing ? 7 : pipeline ? 6 : orderedAbi ? 5 : extentAbi ? 4 : loopStride > 1 ? 3 : parameterTypes.includes('f32[]') ? 2 : 1,
    kind: checkedIndexing ? 'closed-indexed-numeric' : pipeline ? 'closed-numeric-pipeline' : extentAbi || loopStride > 1 || parameterTypes.includes('f32[]') ? 'closed-numeric-loop' : 'closed-f64-loop',
    ...(!checkedIndexing && !pipeline && (extentAbi || loopStride > 1) ? { loopStride } : {}),
    ...(orderedAbi ? { resultType: resultNode ? 'f64' : 'void', iterationSemantics: 'ordered' } : {}),
    functionName: fn.id.name,
    sourceName: String(sourceName),
    sourceSpan: { start: fn.start, end: fn.end },
    parameters: [...params.values()].map(param => ({
      name: param.name, type: param.type,
      read: reads.has(param.name), write: writes.has(param.name),
      ...(!checkedIndexing && extentAbi && param.type !== 'f64' ? { access: {
        indexed: indexed.has(param.name), minimumLength: minimumLengths.get(param.name) ?? 0,
        ...(pipeline ? { loopBounds: [...(indexedBounds.get(param.name) ?? [])] } : {}),
      } } : {}),
    })),
    ...(checkedIndexing ? {
      indexSemantics: 'checked-integer-full-view-v1',
      lengthParameters: [...bounds.keys()].map(name => params.get(name).index),
      loops: [...passes.values()].map(pass => ({ boundParameter: pass.boundParam.index, loopStride: pass.loopStride })),
    } : pipeline ? {
      boundParameters: [...bounds.keys()].map(name => params.get(name).index),
      loops: [...passes.values()].map(pass => ({ boundParameter: pass.boundParam.index, loopStride: pass.loopStride })),
    } : { boundParameter: boundParam.index }),
    maxMemoryPages,
    // Older hosts reject this semantic contract, rather than skipping guards.
    numericSemantics: mathIntrinsics.length ? 'f64-operator-order+guarded-math-v1' : 'f64-operator-order',
    ...(mathIntrinsics.length ? { mathIntrinsics } : {}),
    automaticRouteAdmission: false,
  };
  const types = [...parameterTypes.map(type => type === 'f64' ? F64 : I32), ...Array(bounds.size).fill(I32)];
  const locals = [[...u32(loops.length), I32]];
  if (temporaryCount) locals.push([...u32(temporaryCount), F64]);
  const body = [
    ...vector(locals),
    ...execution, ...result, 0x0b,
  ];
  const helperCode = helperCompiler.finish();
  const wasm = new Uint8Array([
    0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...section(1, vector([[0x60, ...vector(types.map(type => [type])), ...(resultNode ? [1, F64] : [0])], ...helperCode.types])),
    ...section(3, vector([[0], ...helperCode.functions])),
    ...section(5, vector([[1, ...u32(1), ...u32(maxMemoryPages)]])),
    ...section(7, vector([[...text('run'), 0, 0], [...text('memory'), 2, 0]])),
    ...section(10, vector([[...u32(body.length), ...body], ...helperCode.bodies])),
    ...section(0, [...text(NUMERIC_KERNEL_SECTION), ...new TextEncoder().encode(JSON.stringify(manifest))]),
  ]);
  for (const parameter of manifest.parameters) {
    if (parameter.access?.loopBounds) Object.freeze(parameter.access.loopBounds);
    if (parameter.access) Object.freeze(parameter.access);
    Object.freeze(parameter);
  }
  if (checkedIndexing) Object.freeze(manifest.lengthParameters);
  if (pipeline || checkedIndexing) {
    if (manifest.boundParameters) Object.freeze(manifest.boundParameters);
    manifest.loops.forEach(Object.freeze);
    Object.freeze(manifest.loops);
  }
  Object.freeze(manifest.parameters);
  Object.freeze(manifest.sourceSpan);
  return Object.freeze({ wasm, manifest: Object.freeze(manifest), helpers: helperCode.helpers,
    // Build-time description only: the executable enforces the cap itself,
    // including on existing v7 hosts. No extra host argument or guard is needed.
    ...(nestedCount ? { nestedLoops: Object.freeze({ count: nestedCount,
      maxDepth: maxNestedDepth, maxIterations: maxNestedIterations }) } : {}),
  });
}
