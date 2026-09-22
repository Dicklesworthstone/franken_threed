/**
 * Candidate discovery only, not a closure or termination proof. Follow source
 * statement containers without entering another function's execution scope.
 * Every discovered function still passes whole-function numeric compilation.
 */
export function hasNumericLoop(body) {
  const pending = [body];
  while (pending.length) {
    const node = pending.pop();
    if (!node) continue;
    if (['ForStatement', 'WhileStatement', 'DoWhileStatement'].includes(node.type)) return true;
    if (node.type === 'BlockStatement') {
      for (const statement of node.body) pending.push(statement);
    } else if (node.type === 'IfStatement') {
      pending.push(node.consequent, node.alternate);
    }
  }
  return false;
}
