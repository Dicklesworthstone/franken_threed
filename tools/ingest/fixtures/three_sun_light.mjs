/** Load the exact pinned addon modules under Node's test runner. The browser
 * application resolves the same bare imports through its source import map. */
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
let SunLight, SunLightNode;
try {
  [{SunLight}, {SunLightNode}] = await Promise.all([
    import(new URL('examples/jsm/lights/SunLight.js', upstream)),
    import(new URL('examples/jsm/lights/SunLightNode.js', upstream))
  ]);
} finally { hook.deregister(); }
export {SunLight, SunLightNode};
