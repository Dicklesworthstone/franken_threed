/** Numeric switch dispatch shared by root kernels and closed helpers.
 * The discriminant is evaluated outside the case lexical scope. Case tests
 * run once in source order, stopping at the first strict Number match; default
 * is selected only after all tests miss, wherever it appears in the source.
 */
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

export function numericSwitchBindings(node, fail) {
  if (node.cases.length > 256) fail('Numeric switches admit at most 256 cases', node);
  const names = new Set();
  for (const branch of node.cases) for (const statement of branch.consequent) {
    if (statement.type !== 'VariableDeclaration') continue;
    if (!['let', 'const'].includes(statement.kind)) fail('Numeric switch locals must be lexical', statement);
    for (const binding of statement.declarations) {
      if (binding.id.type !== 'Identifier' || !binding.init || names.has(binding.id.name))
        fail('Numeric switch locals must be distinct initialized identifiers', binding);
      names.add(binding.id.name);
    }
  }
  return names;
}

export function compileNumericSwitch(node, {discriminant, expression, compileCase, allocateLocal}) {
  const value = allocateLocal(), selected = allocateLocal(), count = node.cases.length;
  const bytes = [...discriminant, ...set(value), ...number(count), ...set(selected), 0x02, 0x40];
  const unmatched = () => [...get(selected), ...number(count), 0x61];
  let defaultIndex = -1;
  for (let i = 0; i < count; i++) {
    const branch = node.cases[i];
    if (!branch.test) { defaultIndex = i; continue; }
    bytes.push(...unmatched(), 0x04, 0x40,
      ...get(value), ...expression(branch.test), 0x61, 0x04, 0x40,
      ...number(i), ...set(selected), 0x0b, 0x0b);
  }
  if (defaultIndex !== -1)
    bytes.push(...unmatched(), 0x04, 0x40, ...number(defaultIndex), ...set(selected), 0x0b);
  const branches = [];
  for (let i = 0; i < count; i++) {
    const branch = compileCase(node.cases[i]);
    branches.push(branch);
    // Once a case is selected, subsequent bodies execute by fallthrough. Each
    // body adds one if label inside the enclosing switch-exit block.
    bytes.push(...get(selected), ...number(i), 0x65, 0x04, 0x40, ...branch.bytes, 0x0b);
  }
  bytes.push(0x0b);

  // Each case can be an entry point. Resolve fallthrough backwards; break is
  // consumed by this switch, while return/continue keep their outer targets.
  // Root kernels need only bytes; helpers supply completions to prove returns.
  let following = new Set(['normal']);
  const completions = new Set(defaultIndex === -1 ? ['normal'] : []);
  for (let i = branches.length - 1; i >= 0; i--) {
    const current = new Set();
    for (const kind of branches[i].completions ?? ['normal']) {
      if (kind === 'normal') for (const next of following) current.add(next);
      else current.add(kind === 'break' ? 'normal' : kind);
    }
    for (const kind of current) completions.add(kind);
    following = current;
  }
  return {bytes, completions};
}
