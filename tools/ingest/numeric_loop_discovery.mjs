/**
 * Candidate discovery only, not a closure or termination proof. Follow source
 * statement containers without entering another function's execution scope.
 * When immutableHelpers is supplied, follow direct calls into those declaration
 * bodies as well. This is still NOT a binding/closure proof: shadowed, mutable,
 * captured, recursive and non-scalar calls must pass the actual compiler.
 * Every discovered function still passes whole-function numeric compilation.
 */
import {isNumericReductionCall, numericReductionCallbackBody} from './numeric_typed_array_ops.mjs';

// A method spelling only proposes a candidate. Actual receiver types, method
// identities, lexical scope and effects must still pass compiler/host guards.
export function isNumericBulkCall(node) {
  return node?.type === 'CallExpression' && !node.optional &&
    node.callee.type === 'MemberExpression' && !node.callee.optional &&
    !node.callee.computed && ['set', 'fill', 'copyWithin', 'reduce', 'reduceRight'].includes(node.callee.property.name);
}

// includeBulk is explicit so legacy single-entry package selection is stable.
export function hasNumericLoop(body, immutableHelpers = null, includeBulk = false) {
  const pending = [body], visited = new Set();
  while (pending.length) {
    const node = pending.pop();
    if (!node || visited.has(node)) continue;
    visited.add(node);
    if (['ForStatement', 'ForOfStatement', 'WhileStatement', 'DoWhileStatement'].includes(node.type)) return true;
    if (includeBulk && isNumericBulkCall(node.type === 'ExpressionStatement' ? node.expression : node)) return true;
    if (immutableHelpers) {
      // Never treat a nested declaration/callback/class as source execution.
      // Follow only explicitly supplied immutable module helper declarations;
      // visited makes mutual/self recursion finite without admitting recursion.
      if (['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
        'ClassDeclaration', 'ClassExpression'].includes(node.type)) continue;
      if (node.type === 'CallExpression' && !node.optional && node.callee.type === 'Identifier') {
        const helper = immutableHelpers.get(node.callee.name);
        if (helper?.type === 'FunctionDeclaration') pending.push(helper.body);
      }
      for (const value of Object.values(node)) {
        if (Array.isArray(value)) {
          for (const child of value) if (child && typeof child.type === 'string') pending.push(child);
        } else if (value && typeof value.type === 'string') pending.push(value);
      }
    } else if (node.type === 'BlockStatement') {
      for (const statement of node.body) pending.push(statement);
    } else if (node.type === 'IfStatement') {
      pending.push(node.consequent, node.alternate);
    }
  }
  return false;
}

/** Infer speculative array slots through local references and helper arguments.
 * This is a name-based overapproximation, NOT a lexical/type/alias proof. The
 * compiler independently resolves every binding and rejects invalid reference
 * flow. Dependencies include every source of a mutable local, including swaps,
 * conditional selection and assignments syntactically after the first access.
 * The bounded helper graph and finite local worklists converge even on cycles;
 * recursive helper execution itself remains a compiler refusal.
 */
export function discoverNumericArrayParameters(fn, immutableHelpers = new Map()) {
  const graph = new Map(), pending = [fn];
  // Only branch VALUES propose references. A selector is numeric control, not
  // another possible array source. Unsupported expression shapes propose none.
  function references(node) {
    const names = [], pending = [node];
    while (pending.length) {
      const value = pending.pop();
      if (value?.type === 'Identifier') names.push(value.name);
      else if (value?.type === 'ConditionalExpression') pending.push(value.consequent, value.alternate);
      else if (value?.type === 'CallExpression' && !value.optional &&
          value.callee.type === 'MemberExpression' && !value.callee.optional &&
          !value.callee.computed && value.callee.property.name === 'subarray') pending.push(value.callee.object);
    }
    return names;
  }
  while (pending.length && graph.size < 65) {
    const current = pending.pop();
    if (graph.has(current)) continue;
    const positions = new Map(current.params.flatMap((param, index) =>
      param.type === 'Identifier' ? [[param.name, index]] : []));
    const record = {positions, arrays:new Set(), calls:[], dependencies:new Map(), required:new Set()};
    graph.set(current, record);
    function depend(binding, value, hidden) {
      if (binding?.type !== 'Identifier' || hidden.has(binding.name)) return;
      if (!record.dependencies.has(binding.name)) record.dependencies.set(binding.name, new Set());
      for (const name of references(value)) if (!hidden.has(name)) record.dependencies.get(binding.name).add(name);
    }
    const nodes = [{node:current.body, hidden:new Set()}];
    while (nodes.length) {
      const {node, hidden} = nodes.pop();
      if (!node || ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
        'ClassDeclaration', 'ClassExpression'].includes(node.type)) continue;
      const visibleReferences = value => references(value).filter(name => !hidden.has(name));
      if (node.type === 'VariableDeclarator') depend(node.id, node.init, hidden);
      if (node.type === 'AssignmentExpression' && node.operator === '=') depend(node.left, node.right, hidden);
      if (node.type === 'MemberExpression' && node.object.type === 'Identifier' && !hidden.has(node.object.name))
        record.required.add(node.object.name);
      if (node.type === 'ForOfStatement')
        for (const name of visibleReferences(node.right)) record.required.add(name);
      // set's source may never be subscripted elsewhere, including through
      // aliases and reordered helper parameters. Treat it as a required view.
      if (isNumericBulkCall(node) && node.callee.property.name === 'set')
        for (const name of visibleReferences(node.arguments[0])) record.required.add(name);
      if (node.type === 'CallExpression' && node.callee.type === 'MemberExpression' &&
          !node.callee.computed && node.callee.property.name === 'subarray')
        for (const name of visibleReferences(node.callee.object)) record.required.add(name);
      if (isNumericReductionCall(node)) {
        const callback = node.arguments[0], body = numericReductionCallbackBody(callback);
        // This literal callback is the only nested function whose body is
        // proposed here. Its parameters shadow outer names; free array inputs
        // still propagate through the same independent compiler/guard proof.
        if (body) nodes.push({node:body, hidden:new Set([...hidden, ...callback.params.map(param => param.name)])});
      }
      if (node.type === 'CallExpression' && !node.optional && node.callee.type === 'Identifier' && !hidden.has(node.callee.name)) {
        const callee = immutableHelpers.get(node.callee.name);
        if (callee?.type === 'FunctionDeclaration') {
          record.calls.push({callee, arguments:node.arguments.map(visibleReferences)});
          if (!graph.has(callee)) pending.push(callee);
        }
      }
      for (const value of Object.values(node)) {
        if (Array.isArray(value)) {
          for (const child of value) if (child && typeof child.type === 'string') nodes.push({node:child, hidden});
        } else if (value && typeof value.type === 'string') nodes.push({node:value, hidden});
      }
    }
  }
  function require(record, names) {
    const pending = [...names];
    let changed = false;
    while (pending.length) {
      const name = pending.pop();
      if (record.required.has(name)) continue;
      record.required.add(name);
      const position = record.positions.get(name);
      if (position !== undefined && !record.arrays.has(position)) {
        record.arrays.add(position); changed = true;
      }
      pending.push(...(record.dependencies.get(name) ?? []));
    }
    return changed;
  }
  for (const record of graph.values()) {
    const names = record.required; record.required = new Set();
    require(record, names);
  }
  let changed;
  do {
    changed = false;
    for (const record of graph.values()) for (const call of record.calls) {
      for (const index of graph.get(call.callee)?.arrays ?? [])
        changed = require(record, call.arguments[index] ?? []) || changed;
    }
  } while (changed);
  const {positions, arrays} = graph.get(fn);
  return new Set([...positions].filter(([, index]) => arrays.has(index)).map(([name]) => name));
}
