/** Close the static module graph of a generated animation package before its
 * manifest and byte budget are finalized. This is build-time AST inspection,
 * not source evaluation. Optional dynamic-import roots are selected by the
 * caller; following static edges does not enable a runtime feature or backend.
 */
import fs from 'node:fs';
import {parse} from 'acorn';
import {AnimationPoseError} from './animation_runtime.mjs';
const fail = message => { throw new AnimationPoseError('GLTF_ANIMATION_MODULE', message); };
const readRuntime = name => fs.readFileSync(new URL('./' + name, import.meta.url), 'utf8');

export function completeAnimationRuntimeModules(outputs, readModule = readRuntime) {
  if (!(outputs instanceof Map) || typeof readModule !== 'function') fail('Expected runtime outputs and a module reader');
  // Work transactionally: neither the caller's output map nor the destination
  // directory can contain a partially resolved graph after a missing import.
  const resolved = new Map(outputs), queue = [...outputs.keys()].filter(name => name.endsWith('.mjs'));
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const name = queue[cursor], source = resolved.get(name);
    if (typeof source !== 'string') fail(`Expected UTF-8 runtime source: ${name}`);
    let ast;
    try { ast = parse(source, {ecmaVersion: 'latest', sourceType: 'module'}); }
    catch (error) { fail(`Cannot parse runtime ${name}: ${error.message}`); }
    for (const node of ast.body) {
      if (!['ImportDeclaration', 'ExportNamedDeclaration', 'ExportAllDeclaration'].includes(node.type) || !node.source) continue;
      const specifier = node.source.value;
      // Animation runtime packages are flat. Do not accidentally ship a Node
      // build-time import, reach outside the toolkit, or depend on a registry.
      if (typeof specifier !== 'string' || !/^\.\/[A-Za-z0-9_-][A-Za-z0-9_.-]*\.mjs$/.test(specifier))
        fail(`Nonlocal runtime import ${JSON.stringify(specifier)} in ${name}`);
      const dependency = specifier.slice(2);
      if (resolved.has(dependency)) continue;
      let contents;
      try { contents = readModule(dependency); }
      catch (error) { fail(`Cannot read runtime ${dependency} imported by ${name}: ${error.message}`); }
      if (typeof contents !== 'string') fail(`Expected UTF-8 runtime source: ${dependency}`);
      resolved.set(dependency, contents); queue.push(dependency);
    }
  }
  for (const [name, source] of resolved) if (!outputs.has(name)) outputs.set(name, source);
  return outputs;
}
