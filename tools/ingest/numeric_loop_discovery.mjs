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
