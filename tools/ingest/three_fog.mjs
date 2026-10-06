/** Live source Fog/FogExp2 and ordered native color receivers.
 * No Three.js import, source mutation, camera update, GPU allocation or clock.
 * The renderer must be constructed with fog:true. Mixed material flags split
 * adjacent color spans, like source shadow/environment receivers; never reorder
 * transparent draws, synthesize depth, or fog a background/fullscreen image.
 */
import {packAnimationFog} from './animation_fog.mjs';
import {snapshotAnimationCameraFog} from './animation_fog_camera.mjs';

export class ThreeFogError extends Error {
  constructor(code, message) {
    super(`THREE_FOG_${code}: ${message}`);
    this.name = 'ThreeFogError'; this.code = `THREE_FOG_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeFogError(code, message); };
const dataFields = object => {
  for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(object)))
    if (!Object.hasOwn(descriptor, 'value')) fail('HOOK', 'Accessor-backed fog/color fields are not admitted');
};

/** Validate source fog before allocating or updating any scene resources.
 * Colors already belong to the caller's linear working space. Exact built-in
 * fog prototypes avoid silently interpreting custom fog subclasses as native.
 * Return an owned snapshot, not the mutable source Color or fog object.
 */
export function inspectThreeFog(fog, three) {
  if (three?.REVISION !== '186' || typeof three.Fog !== 'function' ||
      typeof three.FogExp2 !== 'function' || typeof three.Color !== 'function')
    fail('SOURCE', 'Supply the pinned r186 fog and color classes');
  if (fog === null) return null;
  if (!fog || typeof fog !== 'object') fail('SOURCE', 'Supply a built-in r186 Fog/FogExp2 or null');
  const prototype = Object.getPrototypeOf(fog);
  const linear = prototype === three.Fog.prototype;
  if (!linear && prototype !== three.FogExp2.prototype)
    fail('SOURCE', 'Custom fog profiles need their own renderer');
  dataFields(fog);
  if (linear ? fog.isFog !== true || fog.isFogExp2 : fog.isFogExp2 !== true || fog.isFog)
    fail('SOURCE', 'Expected consistent built-in source fog flags');
  const color = fog.color;
  if (!(color instanceof three.Color)) fail('SOURCE', 'Expected a source Color');
  dataFields(color);
  if (color.isColor !== true) fail('SOURCE', 'Expected a source Color');
  const result = Object.freeze({type: linear ? 'linear' : 'exp2',
    color: Object.freeze([color.r, color.g, color.b]),
    ...(linear ? {near: fog.near, far: fog.far} : {density: fog.density})});
  // The native contract includes f32 quantization, nonnegative color/density
  // and distinct linear edges. No fake camera or guessed projection is needed.
  packAnimationFog({...result, depthFromClip: [0, 0, 0, 1]});
  return result;
}

/** Capture source fog against the SAME projection convention as cameraFrame.
 * Call after the application's existing camera update, not from a new loop.
 * The established (scene, camera, three) call remains supported. A direct fog
 * object/null is also accepted for already-captured scene state. Both routes
 * keep source camera failures in the THREE_FOG_CAMERA error namespace.
 * Null fog is an explicit reset and does not inspect a camera.
 */
export function threeFogDescriptor(sceneOrFog, camera, three) {
  const isFog = sceneOrFog && typeof sceneOrFog === 'object' &&
    ((typeof three?.Fog === 'function' && sceneOrFog instanceof three.Fog) ||
     (typeof three?.FogExp2 === 'function' && sceneOrFog instanceof three.FogExp2));
  const fog = !isFog && sceneOrFog && typeof sceneOrFog === 'object' && 'fog' in sceneOrFog
    ? sceneOrFog.fog : sceneOrFog;
  const source = inspectThreeFog(fog, three);
  if (source === null) return null;
  if (typeof three.Camera !== 'function' || !(camera instanceof three.Camera) ||
      (!camera.isPerspectiveCamera && !camera.isOrthographicCamera) || camera.isArrayCamera || camera.reversedDepth ||
      ![three.WebGLCoordinateSystem, three.WebGPUCoordinateSystem].includes(camera.coordinateSystem))
    fail('CAMERA', 'Supply a non-reversed perspective/orthographic source camera with a known clip convention');
  const projection = camera.projectionMatrix?.elements;
  if ((!Array.isArray(projection) && !ArrayBuffer.isView(projection)) || projection.length !== 16)
    fail('CAMERA', 'Expected a finite source projection');
  // Preserve the source helper's f32 admission, with fixed indexed reads rather
  // than trusting a borrowed matrix's iterator or cached inverse.
  const captured = Array.from({length: 16}, (_, i) => {
    const value = projection[i];
    if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)))
      fail('CAMERA', 'Expected a finite source projection');
    return value;
  });
  try {
    return snapshotAnimationCameraFog(source, captured, {
      clipSpace: camera.coordinateSystem === three.WebGLCoordinateSystem ? 'webgl' : 'webgpu',
    });
  } catch (error) {
    if (error?.code === 'ANIMATION_FOG_CAMERA') fail('CAMERA', error.message);
    throw error;
  }
}

// Canonical immutable native descriptor. Capture once, before any span submits;
// a caller changing its source arrays cannot give later spans different fog.
function frameFog(fog) {
  const data = packAnimationFog(fog);
  if (data[11] === 0) return null;
  return Object.freeze({type: data[11] === 1 ? 'linear' : 'exp2',
    depthFromClip: Object.freeze(Array.from(data.subarray(0, 4))),
    color: Object.freeze(Array.from(data.subarray(4, 7))),
    ...(data[11] === 1 ? {near: data[8], far: data[9]} : {density: data[10]})});
}

/** Apply live material.fog without changing registration or the draw ABI.
 * Strip only receiveFog; leave receiver flags for nested shadow/IBL owners.
 * One span with no effect is still submitted to honor attachment clears.
 * Mixed flags may increase color passes, but each logical draw occurs once.
 * A failure after a successful prefix is terminal: a partial frame cannot be
 * rolled back or safely retried. Counts describe the last COMPLETE frame.
 */
export function withThreeFogReceivers(renderer, maxDraws = 1024) {
  if (!Number.isSafeInteger(maxDraws) || maxDraws < 1)
    fail('LIMIT', 'Expected a positive bounded draw capacity');
  let busy = false, terminal = null, drawCount = 0, drawCallCount = 0, colorPassCount = 0;
  function live() {
    if (terminal) throw terminal;
    if (renderer.disposed) fail('DISPOSED', 'Source color renderer is disposed');
  }
  const owner = Object.freeze({
    addMesh(gpu, options) { live(); return renderer.addMesh(gpu, options); },
    get disposed() { return renderer.disposed; }, get failed() { return terminal !== null || renderer.failed; },
    get allocatedBytes() { return renderer.allocatedBytes; }, get bundleDiagnostics() { return renderer.bundleDiagnostics; },
    get drawCount() { return drawCount; }, get drawCallCount() { return drawCallCount; }, get colorPassCount() { return colorPassCount; },
    render(frame) {
      live(); if (busy) fail('REENTRANT', 'Fog color submission cannot be reentered');
      busy = true; let submitted = 0, calls = 0, passes = 0;
      try {
        if (!frame || typeof frame !== 'object' || Array.isArray(frame)) fail('FRAME', 'Expected a color frame');
        const inputs = frame.draws;
        if (!Array.isArray(inputs) || inputs.length > maxDraws) fail('LIMIT', 'Source color draw list exceeds capacity');
        const count = inputs.length, fog = frameFog(frame.fog), captured = {...frame, fog}, spans = [];
        // Indexed iteration fixes the admission bound independently of an
        // array's iterator. Validate ALL flags before the first queue effect.
        for (let i = 0; i < count; i++) {
          const input = inputs[i];
          if (!input || typeof input !== 'object' || Array.isArray(input)) fail('FRAME', 'Expected an explicit fog receiver draw');
          const {receiveFog, ...draw} = input;
          if (typeof receiveFog !== 'boolean') fail('FRAME', 'Expected an explicit boolean fog receiver flag');
          const enabled = receiveFog && fog !== null;
          if (!spans.length || spans.at(-1).enabled !== enabled) spans.push({enabled, draws: []});
          spans.at(-1).draws.push(draw);
        }
        if (!spans.length) spans.push({enabled: false, draws: []});
        for (const span of spans) {
          renderer.render({...captured, draws: span.draws, fog: span.enabled ? fog : null,
            ...(submitted ? {loadOp: 'load', depthLoadOp: 'load',
              ...(frame.stencilLoadOp !== undefined ? {stencilLoadOp: 'load'} : {})} : {})});
          submitted++; calls += renderer.drawCallCount; passes += renderer.colorPassCount ?? 1;
        }
        drawCount = count; drawCallCount = calls; colorPassCount = passes; return owner;
      } catch (error) {
        if (submitted || renderer.failed) { terminal = error; renderer.dispose(); }
        throw error;
      } finally { busy = false; }
    },
    async whenIdle() { live(); await renderer.whenIdle(); live(); return owner; },
    dispose() { if (busy) fail('REENTRANT', 'Cannot dispose during fog submission'); renderer.dispose(); },
  });
  return owner;
}
