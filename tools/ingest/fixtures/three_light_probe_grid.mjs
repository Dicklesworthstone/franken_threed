/** Import the immutable source addon with the same aliases used by browsers. */
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
let LightProbeGrid, LightProbeGridNode;
try {
  [{LightProbeGrid}, {LightProbeGridNode}] = await Promise.all([
    import(new URL('examples/jsm/lighting/LightProbeGrid.js', upstream)),
    import(new URL('examples/jsm/tsl/lighting/LightProbeGridNode.js', upstream))
  ]);
} finally { hook.deregister(); }
export {LightProbeGrid, LightProbeGridNode};
