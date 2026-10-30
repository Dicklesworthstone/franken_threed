/** Resolve the unchanged pinned CSM addon and its original dependencies in Node.
 * Browser applications use their ordinary source import map. */
import {registerHooks} from 'node:module';
const upstream = new URL('../../../upstream/three.js/', import.meta.url);
const aliases = new Map([
  ['three', new URL('build/three.webgpu.js', upstream).href],
  ['three/webgpu', new URL('build/three.webgpu.js', upstream).href],
  ['three/tsl', new URL('build/three.tsl.js', upstream).href]
]);
const hook = registerHooks({resolve(specifier, context, next) {
  return next(context.parentURL?.startsWith(upstream.href) ? aliases.get(specifier) ?? specifier : specifier, context);
}});
let CSMShadowNode;
try { ({CSMShadowNode} = await import(new URL('examples/jsm/csm/CSMShadowNode.js', upstream))); }
finally { hook.deregister(); }
export {CSMShadowNode};
