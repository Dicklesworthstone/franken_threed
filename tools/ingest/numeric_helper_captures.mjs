/** Compile-only closure conversion for immutable module helper declarations.
 * Captures are live declarative bindings, never properties or baked values.
 * Original functions stay intact; the ordinary numeric compiler must close
 * every converted body, and runtime dispatch guards each hidden scalar/view.
 */
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {discoverNumericArrayParameters} from './numeric_loop_discovery.mjs';

const contains = (outer, inner) => outer.start <= inner.start && inner.end <= outer.end;
const varScope = scope => ['Program', 'FunctionDeclaration', 'FunctionExpression',
  'ArrowFunctionExpression', 'StaticBlock'].includes(scope.type);
function assigned(node, visit) {
  if (!node) return;
  if (node.type === 'Identifier') visit(node);
  else if (node.type === 'RestElement') assigned(node.argument, visit);
  else if (node.type === 'AssignmentPattern') assigned(node.left, visit);
  else if (node.type === 'ArrayPattern') node.elements.forEach(value => assigned(value, visit));
  else if (node.type === 'ObjectPattern') node.properties.forEach(value =>
    assigned(value.type === 'RestElement' ? value.argument : value.value, visit));
  else if (node.type === 'VariableDeclaration') node.declarations.forEach(value => assigned(value.id, visit));
}

/** bindings is the existing numericLoopScopeBindings lexical resolver. Keeping
 * the same scope rules avoids borrowing body vars from parameter initializers
 * or case bindings from switch discriminants. A numeric capture is read-only
 * throughout the selected region AND its complete reachable helper graph.
 */
export function createNumericHelperCapturePlanner({source, ast, tokens, helperSources,
  helperDeclarations, bindings, fresh}) {
  const analyzed = new Map();
  function inspect(node, outerScopes) {
    const references = [], calls = [], writes = new Set();
    const owner = (name, position, scopes) => [...outerScopes, ...scopes].reverse()
      .find(scope => {
        const kind = bindings(scope, position).get(name);
        // The shared inventory includes var declarations on their immediate
        // block/loop as well as their hoist owner. Only the latter owns the
        // binding; a module var loop header must not hide an in-region write.
        return kind && (kind !== 'var' || varScope(scope));
      });
    walk.fullAncestor(node, (value, _state, scopes) => {
      if (value.type === 'AssignmentExpression' || value.type === 'UpdateExpression' ||
          value.type === 'ForInStatement' || value.type === 'ForOfStatement' ||
          (value.type === 'VariableDeclarator' && value.init)) {
        const target = value.type === 'UpdateExpression' ? value.argument :
          value.type === 'VariableDeclarator' ? value.id : value.left;
        assigned(target, identifier => {
          if (owner(identifier.name, identifier, scopes) === ast) writes.add(identifier.name);
        });
      }
      if (value.type === 'CallExpression' && value.callee.type === 'Identifier') {
        const name = value.callee.name, helper = helperDeclarations.get(name);
        const scope = owner(name, value.callee, scopes);
        if (helper && (scope === ast || scope === helper)) calls.push(value);
      }
      if (value.type !== 'Identifier' || value.name === 'Math' || helperDeclarations.has(value.name) ||
          owner(value.name, value, scopes) !== ast) return;
      // Declaration identifiers are not reads. Assignment targets are retained
      // here too, then rejected by the whole-region write check below.
      const parent = scopes.at(-2);
      if (parent?.type === 'VariableDeclarator' && contains(parent.id, value)) return;
      if ((parent?.type === 'FunctionDeclaration' || parent?.type === 'ClassDeclaration') && parent.id === value) return;
      if (parent?.type === 'CallExpression' && parent.callee === value) return;
      references.push(value);
    });
    return {node, references, calls, writes};
  }
  function helper(name) {
    if (!analyzed.has(name)) analyzed.set(name, inspect(helperDeclarations.get(name), [ast]));
    return analyzed.get(name);
  }

  return (root, ancestors = [ast]) => {
    const entry = inspect(root, ancestors), reachable = new Map(), pending = entry.calls.map(call => call.callee.name);
    while (pending.length) {
      const name = pending.pop();
      if (reachable.has(name)) continue;
      if (reachable.size === 64) return {refusal:'HELPER_CAPTURE_BUDGET'};
      const record = helper(name); reachable.set(name, record);
      for (const call of record.calls) pending.push(call.callee.name);
    }
    const captures = [...new Set([...reachable.values()].flatMap(record => record.references.map(value => value.name)))].sort();
    if (!captures.length) return null;
    if (captures.length > 64) return {refusal:'HELPER_CAPTURE_BUDGET'};
    const records = [entry, ...reachable.values()];
    if (records.some(record => captures.some(name => record.writes.has(name))))
      return {refusal:'HELPER_CAPTURE_WRITE'};
    const parameters = captures.map(() => fresh('helper_capture'));
    const replacements = new Map(captures.map((name, index) => [name, parameters[index]]));
    function editsFor(record, declaration) {
      const edits = record.references.filter(value => replacements.has(value.name)).map(value =>
        ({start:value.start, end:value.end, text:replacements.get(value.name)}));
      for (const call of record.calls) {
        const start = call.arguments.length ? call.arguments.at(-1).end : call.end - 1;
        edits.push({start, end:start, text:(call.arguments.length ? ',' : '') + parameters.join(',')});
      }
      if (declaration) {
        const fn = record.node;
        // The numeric compiler still rejects defaults, destructuring, async and
        // generator declarations. Never normalize their semantics into a kernel.
        const start = fn.params.length ? fn.params.at(-1).end :
          tokens.find(token => token.type.label === '(' && token.start >= fn.id.end && token.end <= fn.body.start)?.end;
        if (start === undefined) throw new Error('Missing helper parameter delimiter');
        edits.push({start, end:start, text:(fn.params.length ? ',' : '') + parameters.join(',')});
      }
      return edits.sort((a,b) => a.start - b.start || a.end - b.end);
    }
    const entryEdits = editsFor(entry, root.type === 'FunctionDeclaration');
    const helperEdits = new Map([...reachable].map(([name, record]) => [name, editsFor(record, true)]));
    function rewrite(start, end, edits, overrides = []) {
      const selected = edits.filter(edit => edit.start >= start && edit.end <= end &&
        !overrides.some(override => contains(override, edit)));
      selected.push(...overrides);
      let text = source.slice(start, end);
      for (const edit of selected.sort((a,b) => b.start - a.start || b.end - a.end))
        text = text.slice(0, edit.start - start) + edit.text + text.slice(edit.end - start);
      return text;
    }
    const convertedHelpers = new Map(helperSources);
    for (const [name, record] of reachable)
      convertedHelpers.set(name, rewrite(record.node.start, record.node.end, helperEdits.get(name)));
    // Propose storage slots from the converted helper graph, including aliases,
    // forwarding helpers and inline numeric callbacks. This is only discovery:
    // the numeric compiler must close every body for each emitted layout, and
    // the existing native view/alias guards still decide every invocation.
    let declarations;
    try {
      declarations = new Map([...reachable.keys()].map(name => [name,
        acorn.parse(convertedHelpers.get(name), {ecmaVersion:'latest', sourceType:'module'}).body[0]]));
    } catch (error) {
      // Appending to an unsupported rest/default parameter form can make the
      // speculative helper invalid. Retain the valid original application.
      if (!(error instanceof SyntaxError)) throw error;
      return {refusal:'KERNEL_NOT_CLOSED'};
    }
    const arrayParameters = new Set();
    for (const declaration of declarations.values())
      for (const name of discoverNumericArrayParameters(declaration, declarations))
        if (parameters.includes(name)) arrayParameters.add(name);
    const captureKinds = parameters.map(name => arrayParameters.has(name) ? 'array' : 'number');
    function originalOffset(node, edits, generated) {
      let delta = 0;
      for (const edit of edits) {
        const start = edit.start - node.start + delta;
        if (generated < start) break;
        if (generated < start + edit.text.length)
          return edit.start + Math.min(generated - start, edit.end - edit.start);
        delta += edit.text.length - (edit.end - edit.start);
      }
      return node.start + generated - delta;
    }
    return {
      captures, parameters, captureKinds, arrayParameters, helperSources:convertedHelpers,
      rewrite:(start = root.start, end = root.end, overrides = []) => rewrite(start, end, entryEdits, overrides),
      originalSpan(name, span) {
        const record = name ? reachable.get(name) : entry;
        const edits = name ? helperEdits.get(name) : entryEdits;
        const start = originalOffset(record.node, edits, span.start), end = originalOffset(record.node, edits, span.end);
        const prefix = source.slice(0, start), line = prefix.split('\n').length;
        return {start, end, line, column:start - prefix.lastIndexOf('\n') - 1};
      },
    };
  };
}
