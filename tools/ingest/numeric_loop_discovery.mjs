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

/** Infer speculative array slots through immutable helper argument positions.
 * A wrapper need not subscript a parameter itself: its callee may do so. This
 * only proposes ABI variants; the numeric compiler still proves the entire
 * body and every call binding, and the runtime guards each actual argument.
 * The graph is bounded like native helper compilation. Cycles reach a finite
 * fixed point here but remain an explicit compiler refusal, never native calls.
 */
export function discoverNumericArrayParameters(fn, immutableHelpers = new Map()) {
  const graph = new Map(), pending = [fn];
  while (pending.length && graph.size < 65) {
    const current = pending.pop();
    if (graph.has(current)) continue;
    const positions = new Map(current.params.flatMap((param, index) =>
      param.type === 'Identifier' ? [[param.name, index]] : []));
    const record = { positions, arrays: new Set(), calls: [] };
    graph.set(current, record);
    const nodes = [current.body];
    while (nodes.length) {
      const node = nodes.pop();
      if (!node || ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
        'ClassDeclaration', 'ClassExpression'].includes(node.type)) continue;
      if (node.type === 'MemberExpression' && node.object.type === 'Identifier' &&
          positions.has(node.object.name)) record.arrays.add(positions.get(node.object.name));
      if (node.type === 'CallExpression' && !node.optional && node.callee.type === 'Identifier') {
        const callee = immutableHelpers.get(node.callee.name);
        if (callee?.type === 'FunctionDeclaration') {
          record.calls.push({ callee, slots: node.arguments.map(argument =>
            argument.type === 'Identifier' ? positions.get(argument.name) : undefined) });
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
  let changed;
  do {
    changed = false;
    for (const record of graph.values()) for (const { callee, slots } of record.calls) {
      for (const index of graph.get(callee)?.arrays ?? []) {
        const slot = slots[index];
        if (slot !== undefined && !record.arrays.has(slot)) {
          record.arrays.add(slot); changed = true;
        }
      }
    }
  } while (changed);
  const { positions, arrays } = graph.get(fn);
  return new Set([...positions].filter(([, index]) => arrays.has(index)).map(([name]) => name));
}
