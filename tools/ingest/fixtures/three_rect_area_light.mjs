/** Resolve the unchanged pinned LTC addon against the same source constructors
 * used by native renderer tests. Applications use their normal import map. */
import {registerHooks} from 'node:module';
const upstream = new URL('../../../upstream/three.js/', import.meta.url);
const hook = registerHooks({resolve(specifier, context, next) {
  return next(specifier === 'three' && context.parentURL?.startsWith(upstream.href)
    ? new URL('build/three.webgpu.js', upstream).href : specifier, context);
}});
let RectAreaLightTexturesLib;
try { ({RectAreaLightTexturesLib} = await import(new URL('examples/jsm/lights/RectAreaLightTexturesLib.js', upstream))); }
finally { hook.deregister(); }
export {RectAreaLightTexturesLib};
