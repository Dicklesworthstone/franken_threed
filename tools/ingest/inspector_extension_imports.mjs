/** Known lazy imports in the pinned r186 Inspector Settings implementation.
 * The public extension records remain mutable, so these are packaging hints,
 * never a proof that the runtime import or renderer ownership is closed.
 * Exact source bytes establish the summary; filenames establish nothing.
 */
import {createHash} from 'node:crypto';
import * as walk from 'acorn-walk';

const SETTINGS = '6f5926044e50465c596159b007bc399887f83bc9b7d0d38016f0eea56341ca09';
const property = (node, name) => node?.type === 'MemberExpression' && !node.computed && node.property.name === name;

export function verifiedInspectorExtensionImports(code, ast) {
  const imports = new Map();
  if (createHash('sha256').update(code).digest('hex') !== SETTINGS) return imports;
  // Read the registry from the verified implementation rather than maintaining
  // another URL list. The same bytes can live at a different module URL.
  const registry = ast.body.flatMap(node => node.type === 'VariableDeclaration' ? node.declarations : [])
    .map(declaration => declaration.init).find(node => node?.type === 'ArrayExpression');
  const candidates = registry.elements.map(record => record.properties.find(entry => entry.key.name === 'url').value.value);
  const urls = new Map();
  walk.simple(ast, {
    VariableDeclarator(node) {
      if (node.id.type !== 'Identifier' || !property(node.init, 'href')) return;
      const url = node.init.object;
      if (url.type !== 'NewExpression' || url.callee.type !== 'Identifier' || url.callee.name !== 'URL' ||
          !property(url.arguments[0], 'url') || !property(url.arguments[1], 'url') ||
          url.arguments[1].object.type !== 'MetaProperty') return;
      urls.set(node.id.name, url.arguments[0]);
    },
  });
  walk.simple(ast, {
    ImportExpression(node) {
      const argument = node.source.type === 'Identifier' ? urls.get(node.source.name) : null;
      if (!argument) return;
      imports.set(node.start, {
        classification: 'finite_set', specifier: null,
        candidates, specifiers: candidates, finite_set: candidates, finiteSet: candidates,
        runtime_fallback: true,
        // Map the relative selector before its existing URL construction. That
        // evaluates the getter and URL constructor once, preserves import(), and
        // avoids comparing a source-relative URL against a relocated chunk URL.
        url_argument_span: {start: argument.start, end: argument.end},
      });
    },
  });
  return imports;
}
