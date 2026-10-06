/**
 * Candidate discovery only, not a closure or termination proof. Follow source
 * statement containers without entering another function's execution scope.
 * When immutableHelpers is supplied, follow direct calls into those declaration
 * bodies as well. This is still NOT a binding/closure proof: shadowed, mutable,
 * captured, recursive and non-scalar calls must pass the actual compiler.
 * Every discovered function still passes whole-function numeric compilation.
 */
export function hasNumericLoop(body, immutableHelpers = null) {
  const pending = [body], visited = new Set();
  while (pending.length) {
    const node = pending.pop();
    if (!node || visited.has(node)) continue;
    visited.add(node);
    if (['ForStatement', 'WhileStatement', 'DoWhileStatement'].includes(node.type)) return true;
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
    function depend(binding, value) {
      if (binding?.type !== 'Identifier') return;
      if (!record.dependencies.has(binding.name)) record.dependencies.set(binding.name, new Set());
      for (const name of references(value)) record.dependencies.get(binding.name).add(name);
    }
    const nodes = [current.body];
    while (nodes.length) {
      const node = nodes.pop();
      if (!node || ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
        'ClassDeclaration', 'ClassExpression'].includes(node.type)) continue;
      if (node.type === 'VariableDeclarator') depend(node.id, node.init);
      if (node.type === 'AssignmentExpression' && node.operator === '=') depend(node.left, node.right);
      if (node.type === 'MemberExpression' && node.object.type === 'Identifier')
        record.required.add(node.object.name);
      if (node.type === 'CallExpression' && !node.optional && node.callee.type === 'Identifier') {
        const callee = immutableHelpers.get(node.callee.name);
        if (callee?.type === 'FunctionDeclaration') {
          record.calls.push({callee, arguments:node.arguments.map(references)});
          if (!graph.has(callee)) pending.push(callee);
        }
      }
      for (const value of Object.values(node)) {
        if (Array.isArray(value)) {
          for (const child of value) if (child && typeof child.type === 'string') nodes.push(child);
        } else if (value && typeof value.type === 'string') nodes.push(value);
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
