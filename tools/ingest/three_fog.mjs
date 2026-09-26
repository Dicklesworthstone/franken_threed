/** Live r186 Fog/FogExp2 capture and order-preserving material.fog routing.
 * Source scene/camera/color objects remain borrowed. No depth texture, radial
 * approximation, background fogging, source scene clone or frame loop.
 */
import {packAnimationFog} from './animation_fog.mjs';
export class ThreeFogError extends Error {
  constructor(code, message) {
    super(`THREE_FOG_${code}: ${message}`);
    this.name = 'ThreeFogError'; this.code = `THREE_FOG_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeFogError(code, message); };

/** Camera-independent source admission, safe before native allocation. */
export function inspectThreeFog(fog, three) {
  if (three?.REVISION !== '186' || typeof three.Fog !== 'function' || typeof three.FogExp2 !== 'function')
    fail('SOURCE', 'Supply the pinned r186 fog classes');
  if (fog === null) return null;
  let value;
  const color = fog?.color;
  if (!color?.isColor) fail('SOURCE', 'Expected a source fog Color');
  if (fog instanceof three.Fog && fog.isFog === true && !fog.isFogExp2)
    value = {type:'linear', color:[color.r,color.g,color.b], near:fog.near, far:fog.far};
  else if (fog instanceof three.FogExp2 && fog.isFogExp2 === true && !fog.isFog)
    value = {type:'exp2', color:[color.r,color.g,color.b], density:fog.density};
  else fail('SOURCE', 'Expected source Fog or FogExp2, not a duck-typed replacement');
  packAnimationFog({...value, depthFromClip:[0,0,0,0]});
  return value;
}

/** Capture -mvPosition.z using a row of the inverse NATIVE projection. The
 * scene bridge converts a WebGL projection's clip Z into [0,1], so perform the
 * same conversion before inversion here. Dot this row with clip position in
 * the vertex stage and interpolate it perspective-correctly; do not divide it
 * by clip W. This works for perspective, orthographic and off-axis projections.
 * The source camera's cached inverse is not trusted, updated or overwritten.
 */
export function threeFogDescriptor(scene, camera, three) {
  const value = inspectThreeFog(scene?.fog, three);
  if (value === null) return null;
  if (typeof three.Camera !== 'function' || typeof three.Matrix4 !== 'function' ||
      !(camera instanceof three.Camera) || (!camera.isPerspectiveCamera && !camera.isOrthographicCamera) ||
      camera.isArrayCamera || camera.reversedDepth ||
      ![three.WebGLCoordinateSystem,three.WebGPUCoordinateSystem].includes(camera.coordinateSystem))
    fail('CAMERA', 'Supply one non-reversed perspective or orthographic source camera');
  const source = camera.projectionMatrix?.elements;
  if (source?.length !== 16 || Array.from(source).some(x => typeof x !== 'number' || !Number.isFinite(Math.fround(x))))
    fail('CAMERA', 'Expected a finite source projection');
  const projection = new three.Matrix4().copy(camera.projectionMatrix);
  if (camera.coordinateSystem === three.WebGLCoordinateSystem) {
    const e = projection.elements;
    for (let column=0;column<4;column++) e[column*4+2]=.5*(e[column*4+2]+e[column*4+3]);
  }
  const determinant = projection.determinant();
  if (determinant === 0 || !Number.isFinite(determinant)) fail('CAMERA', 'Fog requires an invertible projection');
  const inverse = projection.invert().elements;
  const descriptor = {...value, color:Object.freeze(value.color),
    depthFromClip:Object.freeze([-inverse[2],-inverse[6],-inverse[10],-inverse[14]])};
  packAnimationFog(descriptor);
  return Object.freeze(descriptor);
}

/** Split only adjacent fog-receiver spans. A material opting out neither loses
 * its textures/lights/shadows nor changes transparent ordering. No second scene
 * traversal, geometry registration or extra draw is introduced. The first
 * span retains caller clear/load policy; later spans load both attachments.
 * Composes with the existing shadow/environment wrappers and render bundles.
 */
export function withThreeFogReceivers(renderer, maxDraws = 1024) {
  if (!Number.isSafeInteger(maxDraws) || maxDraws < 1) fail('LIMIT', 'Invalid draw capacity');
  let busy=false, terminal=null, drawCount=0, drawCallCount=0, colorPassCount=0;
  function live() {
    if (terminal) throw terminal;
    if (renderer.disposed) fail('DISPOSED', 'Source fog color renderer is disposed');
  }
  const owner = Object.freeze({
    addMesh(gpu, options) { live(); return renderer.addMesh(gpu, options); },
    get disposed() { return renderer.disposed; }, get failed() { return terminal !== null || renderer.failed; },
    get allocatedBytes() { return renderer.allocatedBytes; }, get bundleDiagnostics() { return renderer.bundleDiagnostics; },
    get drawCount() { return drawCount; }, get drawCallCount() { return drawCallCount; }, get colorPassCount() { return colorPassCount; },
    render(frame) {
      live(); if (busy) fail('REENTRANT', 'Fog color submission cannot be reentered');
      if (!frame || !Array.isArray(frame.draws) || frame.draws.length > maxDraws) fail('LIMIT', 'Source fog draw list exceeds capacity');
      // Snapshot all descriptor values and receiver flags before first submit.
      const packet = packAnimationFog(frame.fog ?? null);
      const fog = packet[11] === 0 ? null : {type:packet[11] === 1 ? 'linear' : 'exp2',
        color:Array.from(packet.slice(4,7)), depthFromClip:Array.from(packet.slice(0,4)),
        ...(packet[11] === 1 ? {near:packet[8],far:packet[9]} : {density:packet[10]})};
      const spans=[];
      for (const input of frame.draws) {
        if (!input || typeof input !== 'object' || typeof input.receiveFog !== 'boolean')
          fail('FRAME', 'Expected an explicit material fog receiver flag');
        const {receiveFog, ...draw}=input, enabled=receiveFog && fog !== null;
        if (!spans.length || spans.at(-1).enabled !== enabled) spans.push({enabled,draws:[]});
        spans.at(-1).draws.push(draw);
      }
      if (!spans.length) spans.push({enabled:false,draws:[]});
      busy=true; let submitted=0, calls=0, passes=0;
      try {
        for (const span of spans) {
          renderer.render({...frame, draws:span.draws, fog:span.enabled?fog:null,
            ...(submitted?{loadOp:'load',depthLoadOp:'load'}:{})});
          submitted++;calls+=renderer.drawCallCount;passes+=renderer.colorPassCount??1;
        }
        drawCount=frame.draws.length;drawCallCount=calls;colorPassCount=passes;return owner;
      } catch(error) {
        if (submitted || renderer.failed) {terminal=error;renderer.dispose();}
        throw error;
      } finally {busy=false;}
    },
    async whenIdle() {live();await renderer.whenIdle();live();return owner;},
    dispose() {if(busy)fail('REENTRANT','Cannot dispose during fog submission');renderer.dispose();},
  });
  return owner;
}
