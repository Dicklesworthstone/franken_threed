import test from 'node:test';
import assert from 'node:assert/strict';
import {completeAnimationRuntimeModules} from './animation_package_modules.mjs';
const code = {code: 'GLTF_ANIMATION_MODULE'};

test('static import and re-export closure is transitive, cyclic and byte-preserving', () => {
  const source = new Map([
    ['entry.mjs', "import './a.mjs'; export * from './b.mjs'; export {n} from './a.mjs';\n"],
    ['a.mjs', "export {n} from './c.mjs';\n"],
    ['b.mjs', "import './a.mjs'; export const b=1;\n"],
    ['c.mjs', "import './entry.mjs'; export const n=7;\n"],
  ]);
  const outputs = new Map([['entry.mjs', source.get('entry.mjs')], ['state.json', '{}\n']]), reads = [];
  assert.equal(completeAnimationRuntimeModules(outputs, name => {reads.push(name);return source.get(name);}), outputs);
  assert.deepEqual(reads, ['a.mjs', 'b.mjs', 'c.mjs']);
  for (const [name, bytes] of source) assert.equal(outputs.get(name), bytes);
  assert.equal(outputs.get('state.json'), '{}\n');
  completeAnimationRuntimeModules(outputs, () => assert.fail('already complete'));
});

test('comments and import-looking strings are not edges; optional lazy roots stay explicit', () => {
  const text = "// import './comment.mjs'\nexport const s=\"import './string.mjs'\"; export function optional(){return import('./optional.mjs');}\n";
  const outputs = new Map([['entry.mjs', text]]);
  completeAnimationRuntimeModules(outputs, () => assert.fail('not a static edge'));
  assert.deepEqual([...outputs], [['entry.mjs', text]]);
});

for (const specifier of ['node:fs', 'three', '../secret.mjs', '/root/module.mjs', './nested/module.mjs', 'https://example.invalid/a.mjs'])
  test(`nonlocal runtime import is rejected before output publication: ${specifier}`, () => {
    const outputs = new Map([['entry.mjs', `import ${JSON.stringify(specifier)};`]]), before = [...outputs];
    assert.throws(() => completeAnimationRuntimeModules(outputs, () => assert.fail('must not read')), code);
    assert.deepEqual([...outputs], before);
  });

test('missing and malformed transitive modules leave the supplied output map unchanged', () => {
  for (const problem of ['missing', 'malformed']) {
    const outputs = new Map([['entry.mjs', "import './a.mjs'; import './b.mjs';"]]), before = [...outputs];
    assert.throws(() => completeAnimationRuntimeModules(outputs, name => {
      if (name === 'a.mjs') return 'export const a=1;';
      if (problem === 'missing') throw Error('missing module');
      return 'export const = broken;';
    }), code);
    assert.deepEqual([...outputs], before);
  }
});
