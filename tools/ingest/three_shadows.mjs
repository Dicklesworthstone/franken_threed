/** Explicit r186 projected shadows over the existing depth/color renderers.
 * Owns a private native map, never light.shadow.map or the borrowed geometry.
 * Source camera controls are retained; filtering is the core fixed 3x3 PCF
 * profile, not a claim of WebGLRenderer shadow-map/filter equivalence.
 */
import {createGpuAnimationShadowMap} from './animation_shadow.mjs';

export class ThreeShadowError extends Error {
  constructor(code, message) {
    super(`THREE_SHADOW_${code}: ${message}`);
    this.name = 'ThreeShadowError'; this.code = `THREE_SHADOW_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeShadowError(code, message); };
const finite = (v, label) => {
  if (typeof v !== 'number' || !Number.isFinite(Math.fround(v))) fail('VALUE', `${label} must fit finite f32`);
  return v;
};
const positive = (v, label) => {
  if (!Number.isSafeInteger(v) || v < 1) fail('LIMIT', `Invalid ${label}`);
  return v;
};
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
// The public r186 build exports lights, but not their shadow constructors.
// Derive the native class from a fresh public light, never from the inspected
// shadow (which could carry custom hooks). Cache per source light constructor
// so ordinary frame inspection creates no temporary source objects.
const nativeShadowClasses = new WeakMap();
function nativeShadowClass(C, exported) {
  if (exported !== undefined) return exported;
  if (!nativeShadowClasses.has(C)) {
    const prototype = Object.getPrototypeOf(new C().shadow);
    nativeShadowClasses.set(C, prototype.constructor);
  }
  return nativeShadowClasses.get(C);
}

/** Metadata admission before any native allocation. Only one projected light
 * is selected by the scene bridge; cubemaps, cascades and node hooks refuse.
 */
export function inspectThreeShadow(light, three) {
  if (three?.REVISION !== '186') fail('SOURCE', 'Supply the pinned r186 module');
  if (light?.isSunLight === true) return inspectSunShadow(light, three);
  const C = light?.isDirectionalLight ? three.DirectionalLight : light?.isSpotLight ? three.SpotLight : null;
  if (typeof C !== 'function' || !(light instanceof C))
    fail('LIGHT', 'Source shadows require a native directional or spot light');
  const S = nativeShadowClass(C, light.isDirectionalLight ? three.DirectionalLightShadow : three.SpotLightShadow);
  if (typeof S !== 'function' || !(light.shadow instanceof S))
    fail('LIGHT', 'Source shadows require a native directional or spot light');
  const shadow = light.shadow, camera = shadow.camera;
  if (!(camera instanceof three.Camera) || camera.isArrayCamera || camera.reversedDepth ||
      (light.isDirectionalLight ? !camera.isOrthographicCamera : !camera.isPerspectiveCamera) ||
      ![three.WebGLCoordinateSystem, three.WebGPUCoordinateSystem].includes(camera.coordinateSystem))
    fail('CAMERA', 'Expected one non-reversed source light camera');
  for (const key of ['updateMatrices', 'getFrustum', 'getViewportCount'])
    if (typeof shadow[key] !== 'function' || shadow[key] !== S.prototype[key]) fail('HOOK', `Custom shadow ${key} is not admitted`);
  if (shadow.getViewportCount() !== 1 || shadow.biasNode != null)
    fail('PROFILE', 'Cascades and shadow shader nodes require a different profile');
  finite(shadow.radius, 'shadow radius');
  if (shadow.radius < 0) fail('VALUE', 'Invalid source shadow radius');
  for (const key of ['autoUpdate', 'needsUpdate']) if (typeof shadow[key] !== 'boolean') fail('VALUE', `Expected boolean shadow.${key}`);
  if (Math.abs(finite(shadow.bias, 'shadow bias')) > 1 || finite(shadow.normalBias, 'normal bias') < 0 ||
      finite(shadow.intensity, 'shadow intensity') < 0 || shadow.intensity > 1) fail('VALUE', 'Invalid source shadow factors');
  if (light.isSpotLight && (finite(shadow.focus, 'spot focus') <= 0 || shadow.focus > 1 || finite(shadow.aspect, 'spot aspect') <= 0))
    fail('CAMERA', 'Invalid spot shadow focus/aspect');
  const width = positive(shadow.mapSize?.x, 'map width'), height = positive(shadow.mapSize?.y, 'map height');
  if (!Number.isSafeInteger(width * height * 4)) fail('LIMIT', 'Shadow extent overflows its byte budget');
  return Object.freeze({light, shadow, camera, width, height,
    signature: Object.freeze([light, shadow, camera, width, height])});
}

/** r186 SunLight (examples/jsm/lights): two orthographic cascades in a 2x1
 * atlas, updated from the viewing camera every frame (SunLightShadow). */
function inspectSunShadow(light, three) {
  const shadow = light.shadow;
  if (!(light instanceof three.Light) || shadow?.isSunLightShadow !== true || !(shadow instanceof three.LightShadow) ||
      typeof shadow.getCamera !== 'function' || typeof shadow.getMatrix !== 'function' || !Array.isArray(shadow._cascadeData))
    fail('LIGHT', 'Expected an r186 SunLight with its SunLightShadow');
  if (shadow.getViewportCount() !== 2 || shadow._cascadeData.length !== 2 || shadow.biasNode != null)
    fail('PROFILE', 'Expected the two-cascade r186 sun shadow');
  for (const key of ['autoUpdate', 'needsUpdate']) if (typeof shadow[key] !== 'boolean') fail('VALUE', `Expected boolean shadow.${key}`);
  if (Math.abs(finite(shadow.bias, 'shadow bias')) > 1 || finite(shadow.normalBias, 'normal bias') < 0 ||
      finite(shadow.intensity, 'shadow intensity') < 0 || shadow.intensity > 1 || finite(shadow.radius, 'shadow radius') < 0)
    fail('VALUE', 'Invalid source shadow factors');
  const extents = shadow.getFrameExtents();
  const tileWidth = positive(shadow.mapSize?.x, 'map width'), tileHeight = positive(shadow.mapSize?.y, 'map height');
  const width = positive(tileWidth * extents.x, 'atlas width'), height = positive(tileHeight * extents.y, 'atlas height');
  if (!Number.isSafeInteger(width * height * 4)) fail('LIMIT', 'Shadow extent overflows its byte budget');
  return Object.freeze({light, shadow, camera: shadow.camera, width, height, sun: {tileWidth, tileHeight},
    signature: Object.freeze([light, shadow, width, height])});
}

/** Borrow source camera/state; own only depth resources and caster bindings.
 * Static map/camera identity changes require a replacement through prepare().
 * autoUpdate:false deliberately retains the last published map until needsUpdate.
 */
export async function createGpuThreeShadow(device, light, {three, maxBytes = 64 * 1024 * 1024,
  maxDraws = 1024, maxMeshes = 1024, label = 'f3d-three-shadow', signal, clipping = false, maxClippingPlanes = 8, textureTransforms = false, alphaMaps = false} = {}) {
  if(typeof clipping!=='boolean'||!Number.isSafeInteger(maxClippingPlanes)||maxClippingPlanes<1||maxClippingPlanes>64)
    fail('OPTIONS','Invalid clipping profile');
  if(typeof textureTransforms!=='boolean') fail('OPTIONS','textureTransforms must be boolean');
  if(typeof alphaMaps!=='boolean') fail('OPTIONS','alphaMaps must be boolean');
  const shape = inspectThreeShadow(light, three);
  positive(maxBytes, 'shadow budget'); positive(maxDraws, 'caster draw capacity'); positive(maxMeshes, 'caster binding capacity');
  if (typeof label !== 'string') fail('OPTIONS', 'Expected a shadow label');
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' ||
      typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected AbortSignal');
  if (signal?.aborted) fail('ABORTED', 'Source shadow construction was aborted');
  let map = null, disposed = false, terminal = null, busy = false, snapshot = null, renderedFrame = null, rendered = 0, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const clip = new three.Matrix4(), frames = new WeakSet();
  function release() { signal?.removeEventListener('abort', onAbort); map?.dispose(); map = null; snapshot = null; }
  function stop(error) { if (!terminal && !disposed) { terminal = error; rejectStopped(error); if (!busy) release(); } }
  function onAbort() { stop(new ThreeShadowError('ABORTED', 'Source shadows were aborted')); }
  function live() {
    if (disposed) fail('DISPOSED', 'Source shadows are disposed');
    if (terminal) throw terminal;
    if (map?.failed || map?.disposed) { stop(new ThreeShadowError('DEVICE', 'Projected shadow resources failed')); throw terminal; }
  }
  function check() {
    live();
    if (!same(shape.signature, inspectThreeShadow(light, three).signature)) fail('PREPARE', 'Light camera/map extent changed; call prepare()');
  }
  signal?.addEventListener('abort', onAbort, {once: true});
  try {
    if (signal?.aborted) onAbort(); live();
    const constructing = createGpuAnimationShadowMap(device, {
      width: shape.width, height: shape.height, maxBytes, maxDraws, maxMeshes, label, filter: 'linear', ...(alphaMaps?{alphaMaps}:{}), ...(textureTransforms?{textureTransforms}:{}), ...(clipping?{clipping,maxClippingPlanes}:{}),
    }).then(value => {
      if (disposed || terminal) { value.dispose(); throw terminal ?? new ThreeShadowError('DISPOSED', 'Source shadows are disposed'); }
      map = value; return value;
    });
    await Promise.race([constructing, stopped]); check();
  } catch (error) { stop(error); throw error; }
  const owner = Object.freeze({source: light, signature: shape.signature,
    get allocatedBytes() { return map?.allocatedBytes ?? 0; },
    get disposed() { return disposed; }, get failed() { return terminal !== null || !!map?.failed; },
    get version() { return snapshot?.version ?? 0; }, get renderCount() { return rendered; },
    matches() { live(); return same(shape.signature, inspectThreeShadow(light, three).signature); }, check,
    async addMesh(gpu, material) {
      check();
      const constructing = map.addMesh(gpu, material).then(mesh => {
        if (disposed || terminal) { mesh.dispose(); throw terminal ?? new ThreeShadowError('DISPOSED', 'Source shadows are disposed'); }
        return mesh;
      });
      const mesh = await Promise.race([constructing, stopped]);
      try { check(); return mesh; } catch (error) { mesh.dispose(); throw error; }
    },
    capture(viewCamera) {
      check(); if (busy) fail('REENTRANT', 'Cannot capture during shadow submission');
      const {shadow, camera} = shape, update = shadow.autoUpdate || shadow.needsUpdate;
      if (!update && !snapshot) fail('UNRENDERED', 'A manual shadow needs needsUpdate=true before its first frame');
      let viewProjection = null, frustum = null, cascades;
      if (shape.sun) {
        if (!(viewCamera instanceof three.Camera)) fail('CAMERA', 'Sun shadows follow the viewing camera');
        // Cascades depend on the viewing camera: refit every frame, as upstream.
        shadow.updateMatrices(light, viewCamera); check();
        const {tileWidth, tileHeight} = shape.sun;
        cascades = Object.freeze([0, 1].map(i => {
          const c = shadow.getCamera(i), v = shadow.getViewport(i);
          clip.multiplyMatrices(c.projectionMatrix, c.matrixWorldInverse);
          if (c.coordinateSystem === three.WebGLCoordinateSystem)
            for (let k = 0; k < 4; k++) clip.elements[k * 4 + 2] = 0.5 * (clip.elements[k * 4 + 2] + clip.elements[k * 4 + 3]);
          // Top-left pixel rect of the tile, exactly as SunShadowNode.renderShadow.
          const rect = [tileWidth * v.x, shape.height - tileHeight * (v.y + v.w), tileWidth * v.z, tileHeight * v.w];
          const d = shadow._cascadeData[i];
          return Object.freeze({viewProjection: Object.freeze(Array.from(clip.elements, x => finite(x, 'cascade clip matrix'))),
            viewport: Object.freeze(rect),
            tile: Object.freeze([rect[0] / shape.width, rect[1] / shape.height, rect[2] / shape.width, rect[3] / shape.height]),
            cascade: Object.freeze([d.x, d.y, d.z, d.w])});
        }));
        viewProjection = cascades[0].viewProjection;
        const frame = Object.freeze({update: true, viewProjection, frustum: null, cascades, bias: -shadow.bias,
          normalBias: shadow.normalBias, strength: shadow.intensity, radius: shadow.radius});
        frames.add(frame); return frame;
      }
      if (update) {
        shadow.updateMatrices(light); check();
        clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        if (camera.coordinateSystem === three.WebGLCoordinateSystem)
          for (let c = 0; c < 4; c++) clip.elements[c * 4 + 2] = 0.5 * (clip.elements[c * 4 + 2] + clip.elements[c * 4 + 3]);
        viewProjection = Object.freeze(Array.from(clip.elements, v => finite(v, 'light clip matrix')));
        frustum = shadow.getFrustum();
        if (!(frustum instanceof three.Frustum)) fail('CAMERA', 'Expected the native light frustum');
      }
      // r186 adds source bias to normalized depth; the core subtracts its bias.
      const frame = Object.freeze({update, viewProjection, frustum, bias: -shadow.bias,
        normalBias: shadow.normalBias, strength: shadow.intensity, radius: shadow.radius});
      frames.add(frame); return frame;
    },
    render(frame, draws) {
      check(); if (busy) fail('REENTRANT', 'Source shadow submission cannot be reentered');
      if (!frames.has(frame)) fail('FRAME', 'Capture the source light before submitting its shadow frame');
      if (!Array.isArray(draws) || draws.length > maxDraws) fail('LIMIT', 'Shadow caster draw list exceeds capacity');
      busy = true;
      try {
        if (frame.update) {
          if (frame.cascades) map.render({cascades: frame.cascades.map(c => ({viewProjection: c.viewProjection, viewport: c.viewport})), draws});
          else map.render({viewProjection: frame.viewProjection, draws});
          const next = map.sample(device); live();
          snapshot = next; rendered++;
          shape.shadow.needsUpdate = false;
        } else if (draws.length) fail('FRAME', 'A frozen shadow frame does not accept new caster draws');
        frames.delete(frame); renderedFrame = frame; live(); return owner;
      } finally { busy = false; if (terminal || disposed) release(); }
    },
    // Frozen maps intentionally do not recapture core pose dependencies: source
    // autoUpdate:false asks to keep OLD depths even after geometry/poses change.
    // This facade only returns a snapshot previously issued by the actual map.
    sample(borrowedDevice) {
      live(); if (borrowedDevice !== device) fail('DEVICE', 'Shadow and receiver must share one device');
      if (!snapshot) fail('UNRENDERED', 'Render the source shadow before receiving it');
      return snapshot;
    },
    descriptor(frame, lightIndex) {
      live();
      if (frame !== renderedFrame || !snapshot) fail('FRAME', 'Submit this captured frame before receiving its map');
      if (!Number.isInteger(lightIndex) || lightIndex < 0 || lightIndex > 7) fail('LIGHT', 'Invalid shadow light index');
      return {map: owner, lightIndex, bias: frame.bias, normalBias: frame.normalBias, strength: frame.strength,
        filter: 'vogel5', radius: frame.radius,
        ...(frame.cascades ? {cascades: frame.cascades.map(c => ({viewProjection: c.viewProjection, tile: c.tile, cascade: c.cascade}))} : {})};
    },
    async whenIdle() {
      live();
      try { await Promise.race([map.whenIdle(), stopped]); live(); return owner; }
      catch (error) { if (map?.failed) stop(error); throw error; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during source shadow submission');
      if (!disposed) { disposed = true; rejectStopped(new ThreeShadowError('DISPOSED', 'Source shadows are disposed')); release(); }
    },
  });
  return owner;
}

/** Selective receivers over already-prepared optional-shadow core pipelines.
 * Adjacent equal-receiver draws stay together; global source order is never
 * changed. Additional spans load prior color/depth, including MSAA storage.
 * This explicit composition trades extra passes for reusing the core pipeline;
 * it is not a single-pass or measured acceleration claim.
 */
export function withThreeShadowReceivers(renderer, maxDraws = 1024) {
  positive(maxDraws, 'color draw capacity');
  let busy = false, terminal = null, drawCount = 0, drawCallCount = 0, colorPassCount = 0;
  function live() { if (terminal) throw terminal; if (renderer.disposed) fail('DISPOSED', 'Source color renderer is disposed'); }
  const wrapped = Object.freeze({
    addMesh(gpu, options) { live(); return renderer.addMesh(gpu, options); },
    get failed() { return terminal !== null || renderer.failed; }, get disposed() { return renderer.disposed; },
    get allocatedBytes() { return renderer.allocatedBytes; }, get bundleDiagnostics() { return renderer.bundleDiagnostics; },
    get drawCount() { return drawCount; }, get drawCallCount() { return drawCallCount; }, get colorPassCount() { return colorPassCount; },
    render(frame) {
      live(); if (busy) fail('REENTRANT', 'Source color submission cannot be reentered');
      if (!frame || !Array.isArray(frame.draws) || frame.draws.length > maxDraws) fail('LIMIT', 'Source color draw list exceeds capacity');
      const spans = [];
      for (const input of frame.draws) {
        if (!input || typeof input !== 'object' || typeof input.receiveShadow !== 'boolean') fail('FRAME', 'Expected an explicit receiver flag');
        const {receiveShadow, ...draw} = input;
        const shadowed = receiveShadow && frame.shadow != null;
        if (!spans.length || spans.at(-1).shadowed !== shadowed) spans.push({shadowed, draws: []});
        spans.at(-1).draws.push(draw);
      }
      if (!spans.length) spans.push({shadowed: false, draws: []});
      busy = true; let submitted = 0, calls = 0;
      try {
        for (const span of spans) {
          renderer.render({...frame, draws: span.draws, shadow: span.shadowed ? frame.shadow : null,
            ...(submitted ? {loadOp: 'load', depthLoadOp: 'load',
              ...(frame.stencilLoadOp !== undefined ? {stencilLoadOp: 'load'} : {})} : {})});
          submitted++; calls += renderer.drawCallCount;
        }
        drawCount = frame.draws.length; drawCallCount = calls; colorPassCount = submitted;
        return wrapped;
      } catch (error) {
        // A failure after a preceding span is not a rollbackable frame. Do not
        // keep a partially submitted color owner available for accidental retry.
        if (submitted || renderer.failed) { terminal = error; renderer.dispose(); }
        throw error;
      } finally { busy = false; }
    },
    async whenIdle() { live(); await renderer.whenIdle(); live(); return wrapped; },
    dispose() { if (busy) fail('REENTRANT', 'Cannot dispose during source color submission'); renderer.dispose(); },
  });
  return wrapped;
}
