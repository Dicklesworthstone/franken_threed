/**
 * Expand small literal-trip loops before numeric-island compilation. This is
 * code generation, not execution of application source. The downstream numeric
 * compiler must still prove the ENTIRE function and every called helper closed.
 *
 * Each iteration gets a distinct lexical index binding and original body scope.
 * No arithmetic is reassociated, and no source array store is removed or fused.
 * Useful for fixed influence counts, matrix components and neighborhood stencils.
 */
import * as acorn from "acorn";
import * as walk from "acorn-walk";

function literalInteger(node) {
  let value;
  if (node?.type === "Literal" && typeof node.value === "number") value = node.value;
  else if (
    node?.type === "UnaryExpression" &&
    ["+", "-"].includes(node.operator) &&
    node.argument.type === "Literal" &&
    typeof node.argument.value === "number"
  ) {
    value = node.operator === "-" ? -node.argument.value : +node.argument.value;
  }
  return Number.isSafeInteger(value) ? value : undefined;
}

function writesName(pattern, name) {
  if (!pattern) return false;
  if (pattern.type === "Identifier") return pattern.name === name;
  if (pattern.type === "RestElement") return writesName(pattern.argument, name);
  if (pattern.type === "AssignmentPattern") return writesName(pattern.left, name);
  if (pattern.type === "ArrayPattern")
    return pattern.elements.some((item) => writesName(item, name));
  if (pattern.type === "ObjectPattern")
    return pattern.properties.some((item) =>
      writesName(item.type === "RestElement" ? item.argument : item.value, name),
    );
  return false;
}

function fixedLoop(node, maxIterations) {
  if (
    node.init?.type !== "VariableDeclaration" ||
    node.init.kind !== "let" ||
    node.init.declarations.length !== 1
  )
    return null;
  const binding = node.init.declarations[0];
  if (binding.id.type !== "Identifier") return null;
  const name = binding.id.name,
    start = literalInteger(binding.init),
    end = literalInteger(node.test?.right);
  if (
    start === undefined ||
    end === undefined ||
    node.test?.type !== "BinaryExpression" ||
    node.test.left.type !== "Identifier" ||
    node.test.left.name !== name ||
    !["<", "<=", ">", ">="].includes(node.test.operator)
  )
    return null;
  const update = node.update;
  let step;
  if (
    update?.type === "UpdateExpression" &&
    update.argument.type === "Identifier" &&
    update.argument.name === name
  ) {
    if (update.operator === "++") step = 1;
    if (update.operator === "--") step = -1;
  } else if (
    update?.type === "AssignmentExpression" &&
    update.left.type === "Identifier" &&
    update.left.name === name
  ) {
    const delta = literalInteger(update.right);
    if (delta !== undefined && ["+=", "-="].includes(update.operator))
      step = update.operator === "+=" ? delta : -delta;
  }
  if (!Number.isSafeInteger(step) || step === 0) return null;
  const ascending = node.test.operator.startsWith("<");
  if (ascending ? step < 0 : step > 0) return null;
  const test = (value) =>
    node.test.operator === "<"
      ? value < end
      : node.test.operator === "<="
        ? value <= end
        : node.test.operator === ">"
          ? value > end
          : value >= end;
  const values = [];
  for (let value = start; test(value); value += step) {
    if (values.length >= maxIterations || !Number.isSafeInteger(value + step)) return null;
    values.push(value);
  }
  // Do not redirect break/continue to a different loop, change function-scoped
  // declarations, or erase a possible mutation of the loop-control binding.
  // Name-only mutation refusal is conservative across nested shadowing scopes.
  let safe = true;
  walk.full(node.body, (child) => {
    if (
      ["BreakStatement", "ContinueStatement", "FunctionDeclaration", "ClassDeclaration"].includes(
        child.type,
      ) ||
      (child.type === "VariableDeclaration" && child.kind === "var") ||
      (child.type === "CallExpression" &&
        child.callee.type === "Identifier" &&
        child.callee.name === "eval") ||
      (child.type === "AssignmentExpression" && writesName(child.left, name)) ||
      (child.type === "UpdateExpression" && writesName(child.argument, name)) ||
      (["ForInStatement", "ForOfStatement"].includes(child.type) && writesName(child.left, name))
    )
      safe = false;
  });
  return safe ? { name, values } : null;
}

/**
 * Bounds are per source unit, including multiplicative expansion of nested
 * loops. Failure to fit the budget retains the WHOLE original source unit.
 * The caller may attempt another compilation route, never a truncated kernel.
 */
export function expandNumericFixedLoops(
  source,
  { maxIterations = 16, maxExpandedIterations = 256, maxSourceLength = 65536 } = {},
) {
  if (typeof source !== "string") throw new TypeError("Fixed-loop expansion requires source text");
  if (
    !Number.isInteger(maxIterations) ||
    maxIterations < 1 ||
    maxIterations > 64 ||
    !Number.isInteger(maxExpandedIterations) ||
    maxExpandedIterations < 1 ||
    maxExpandedIterations > 4096 ||
    !Number.isInteger(maxSourceLength) ||
    maxSourceLength < 1 ||
    maxSourceLength > 1048576
  ) {
    throw new RangeError("Invalid fixed-loop expansion budget");
  }
  const unchanged = (reason) => ({
    source,
    changed: false,
    loops: [],
    expandedIterations: 0,
    reason,
  });
  if (source.length > maxSourceLength) return unchanged("SOURCE_BUDGET");
  let ast;
  try {
    ast = acorn.parse(source, { ecmaVersion: "latest", sourceType: "module", locations: true });
  } catch {
    return unchanged("PARSE_UNSUPPORTED");
  }
  const loops = [];
  let work = 0;
  const budgetExceeded = {};
  function render(root, start, end, multiplicity = 1) {
    const replacements = [];
    walk.recursive(root, null, {
      ForStatement(node, state, visit) {
        const fixed = fixedLoop(node, maxIterations);
        if (!fixed) {
          walk.base.ForStatement(node, state, visit);
          return;
        }
        const copies = fixed.values.length * multiplicity;
        work += copies;
        if (work > maxExpandedIterations || loops.length >= 64) throw budgetExceeded;
        loops.push({
          indexName: fixed.name,
          iterations: fixed.values.length,
          sourceSpan: {
            start: node.start,
            end: node.end,
            line: node.loc.start.line,
            column: node.loc.start.column,
          },
        });
        const body = copies ? render(node.body, node.body.start, node.body.end, copies) : "";
        const iterations = fixed.values
          .map(
            (value) =>
              `\n{ const ${fixed.name} = ${Object.is(value, -0) ? "-0" : value};\n${body}\n}`,
          )
          .join("");
        if (iterations.length > maxSourceLength) throw budgetExceeded;
        replacements.push({ start: node.start, end: node.end, text: `{${iterations}\n}` });
      },
    });
    replacements.sort((a, b) => a.start - b.start);
    const pieces = [];
    let cursor = start,
      size = 0;
    for (const replacement of replacements) {
      const prefix = source.slice(cursor, replacement.start);
      size += prefix.length + replacement.text.length;
      if (size > maxSourceLength) throw budgetExceeded;
      pieces.push(prefix, replacement.text);
      cursor = replacement.end;
    }
    const suffix = source.slice(cursor, end);
    if (size + suffix.length > maxSourceLength) throw budgetExceeded;
    pieces.push(suffix);
    return pieces.join("");
  }
  try {
    const expanded = render(ast, 0, source.length);
    return loops.length
      ? { source: expanded, changed: true, loops, expandedIterations: work, reason: null }
      : unchanged("NO_FIXED_LOOPS");
  } catch (error) {
    if (error !== budgetExceeded) throw error;
    return unchanged("EXPANSION_BUDGET");
  }
}
