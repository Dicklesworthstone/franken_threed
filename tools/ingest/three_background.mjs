/** Source-owned, unfiltered HDR backgrounds for the explicit r186 scene bridge.
 * Borrow the ready HDR DataTexture and source camera; own one RGBA16F panorama
 * and the native fullscreen renderer. Never replace scene.background, decode a
 * URL, run PMREM, install a frame loop or modify the source's upload versions.
 * This profile supports perspective cameras and backgroundBlurriness===0.
 */
import {inspectThreeEnvironment, captureThreeEnvironmentPixels} from './three_environment.mjs';
import {createGpuAnimationBackground, packAnimationBackgroundFrame} from './animation_background.mjs';

export class ThreeBackgroundError extends Error {
  constructor(code, message) {
    super(`THREE_BACKGROUND_${code}: ${message}`);
    this.name = 'ThreeBackgroundError'; this.code = `THREE_BACKGROUND_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeBackgroundError(code, message); };
const same = (a,b) => a.length === b.length && a.every((v,i) => v === b[i]);
export const inspectThreeBackground = inspectThreeEnvironment;

/** Admission for live scene controls, also used before scene GPU allocation. */
export function inspectThreeBackgroundState(scene) {
  if (typeof scene?.backgroundIntensity !== 'number' || scene.backgroundIntensity < 0 ||
      !Number.isFinite(Math.fround(scene.backgroundIntensity))) fail('VALUE', 'Background intensity must fit nonnegative finite f32');
  if (scene.backgroundBlurriness !== 0) fail('PROFILE', 'Raw source panoramas require backgroundBlurriness=0; PMREM blur is not inferred');
  const r = scene.backgroundRotation;
  if (!r?.isEuler || !['XYZ','YXZ','ZXY','ZYX','YZX','XZY'].includes(r.order) ||
      [r.x,r.y,r.z].some(v => typeof v !== 'number' || !Number.isFinite(v))) fail('VALUE', 'Expected a finite source background Euler rotation');
}

/** Pure source-state capture. The caller updates camera world/projection state
 * first (the scene bridge already does so). Camera translation is excluded;
 * inverse authored background rotation applies AFTER camera world rotation.
 * Both WebGL [-1,1] and WebGPU [0,1] projection matrices admit clip z=0.5.
 * Source projection inverse caches, transforms and Euler objects are untouched.
 */
export function threeBackgroundFrame(scene, camera, three) {
  inspectThreeBackgroundState(scene);
  if (three?.REVISION !== '186' || typeof three.Matrix4 !== 'function' || typeof three.Camera !== 'function' ||
      !(camera instanceof three.Camera) || !camera.isPerspectiveCamera || camera.isOrthographicCamera || camera.isArrayCamera || camera.reversedDepth ||
      ![three.WebGLCoordinateSystem,three.WebGPUCoordinateSystem].includes(camera.coordinateSystem))
    fail('CAMERA', 'HDR backgrounds require one non-reversed source perspective camera');
  const projection = camera.projectionMatrix?.elements, world = camera.matrixWorld?.elements;
  if (projection?.length !== 16 || world?.length !== 16 ||
      Array.from(projection).some(v => typeof v !== 'number' || !Number.isFinite(Math.fround(v))) ||
      Array.from(world).some(v => typeof v !== 'number' || !Number.isFinite(v)) ||
      world[3] !== 0 || world[7] !== 0 || world[11] !== 0 || world[15] !== 1)
    fail('CAMERA', 'Expected finite projection and affine camera world matrices');
  const inverse = new three.Matrix4().copy(camera.projectionMatrix);
  if (!Number.isFinite(inverse.determinant()) || inverse.determinant() === 0) fail('CAMERA', 'Source projection is singular');
  inverse.invert();
  const cameraRotation = new three.Matrix4().copy(camera.matrixWorld);
  cameraRotation.elements[12] = cameraRotation.elements[13] = cameraRotation.elements[14] = 0;
  const rotation = new three.Matrix4().makeRotationFromEuler(scene.backgroundRotation).transpose();
  const direction = new three.Matrix4().multiplyMatrices(rotation, cameraRotation).multiply(inverse);
  const captured = Object.freeze({directionFromClip: Object.freeze(Array.from(direction.elements)), intensity: scene.backgroundIntensity});
  packAnimationBackgroundFrame(captured); // Validate packed rays before any frame GPU work.
  return captured;
}

/** maxBytes charges the original-resolution RGBA16F panorama and 128-byte draw
 * packet, not an undersized filtered cube. Scene replacement also charges the
 * previous owner until its last submitted consumers drain. Pixel conversion and
 * orientation are identical to source environment lighting; filtering is absent.
 */
export async function createGpuThreeBackground(device, texture, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected background options');
  for (const key of Object.keys(options)) if (!['three','maxBytes','maxPixels','format','sampleCount','label','signal'].includes(key))
    fail('OPTIONS', `Unknown source background option: ${key}`);
  const {three, maxBytes = 128 * 1024 * 1024, maxPixels = 16 * 1024 * 1024, format = 'rgba8unorm',
    sampleCount = 1, label = 'f3d-source-background', signal} = options;
  const shape = inspectThreeBackground(texture, three, {maxPixels});
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || shape.sourceBytes + 128 > maxBytes)
    fail('LIMIT', 'Raw panorama and draw uniform exceed the background byte budget');
  if (!['rgba8unorm','rgba8unorm-srgb','bgra8unorm','bgra8unorm-srgb','rgba16float'].includes(format) ||
      ![1,4].includes(sampleCount) || typeof label !== 'string') fail('OPTIONS', 'Unsupported output format, samples or label');
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' ||
      typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected AbortSignal');
  if (signal?.aborted) fail('ABORTED', 'Source background creation was aborted');
  if (typeof device?.createTexture !== 'function' || typeof device.queue?.writeTexture !== 'function' ||
      typeof device.lost?.then !== 'function' || typeof device.pushErrorScope !== 'function' || typeof device.popErrorScope !== 'function')
    fail('DEVICE', 'Expected a native WebGPU device');
  if (!(shape.width <= device.limits?.maxTextureDimension2D) || !(shape.height <= device.limits?.maxTextureDimension2D))
    fail('LIMIT', 'Source background exceeds the native texture extent limit');
  const pixels = captureThreeEnvironmentPixels(texture, three, {maxPixels}), lifetime = new AbortController();
  let input = null, renderer = null, disposed = false, terminal = null, busy = false, rejectStop;
  const captures = new WeakSet(), stopped = new Promise((_,reject) => {rejectStop=reject;}); stopped.catch(()=>{});
  function release() {
    texture.removeEventListener('dispose', onSourceDispose); signal?.removeEventListener('abort', onAbort);
    lifetime.abort(); renderer?.dispose(); renderer=null; input?.destroy(); input=null;
  }
  function stop(error) { if (!terminal && !disposed) {terminal=error;rejectStop(error);if(!busy)release();} }
  function onAbort() {stop(new ThreeBackgroundError('ABORTED','Source background lifetime was aborted'));}
  function onSourceDispose() {stop(new ThreeBackgroundError('SOURCE_DISPOSED','Source background texture was disposed'));}
  function live() {
    if(disposed)fail('DISPOSED','Source background is disposed'); if(terminal)throw terminal;
    if(renderer?.failed || renderer?.disposed){stop(new ThreeBackgroundError('DEVICE','Native background renderer failed'));throw terminal;}
  }
  function check() {
    live(); if(!same(shape.signature,inspectThreeBackground(texture,three,{maxPixels}).signature))
      fail('PREPARE','Background source changed; prepare its current version before rendering');
  }
  texture.addEventListener('dispose',onSourceDispose);signal?.addEventListener('abort',onAbort,{once:true});
  device.lost.then(info=>stop(new ThreeBackgroundError('DEVICE',info?.message??'GPU device lost')),stop);
  try {
    if(signal?.aborted)onAbort();check();
    device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');let error;
    try {
      input=device.createTexture({label,size:[shape.width,shape.height,1],dimension:'2d',format:'rgba16float',
        mipLevelCount:1,sampleCount:1,usage:2|4});
      device.queue.writeTexture({texture:input},pixels,{bytesPerRow:shape.width*8},[shape.width,shape.height,1]);
    }catch(cause){error=cause;}
    const validation=device.popErrorScope(),memory=device.popErrorScope();
    const errors=await Promise.race([Promise.all([validation,memory]),stopped]);
    if(error)throw error;if(errors.some(Boolean))fail('DEVICE',errors.find(Boolean).message??'Background upload failed');check();
    const constructing=createGpuAnimationBackground(device,input,{format,sampleCount,label,mapping:'panorama',signal:lifetime.signal}).then(value=>{
      if(disposed||terminal){value.dispose();throw terminal??new ThreeBackgroundError('DISPOSED','Source background is disposed');}
      renderer=value;return value;
    });
    await Promise.race([constructing,stopped]);check();
  }catch(error){stop(error);release();throw error;}
  const owner=Object.freeze({source:texture,signature:shape.signature,
    get disposed(){return disposed;},get failed(){return terminal!==null||!!renderer?.failed;},
    get allocatedBytes(){return (input?shape.sourceBytes:0)+(renderer?.allocatedBytes??0);},
    get drawCount(){return renderer?.drawCount??0;},
    matches(){live();return same(shape.signature,inspectThreeBackground(texture,three,{maxPixels}).signature);},check,
    capture(scene,camera){
      check();if(busy)fail('REENTRANT','Cannot capture during background submission');
      if(scene.background!==texture)fail('PREPARE','Source scene selected another background');
      const captured=threeBackgroundFrame(scene,camera,three);captures.add(captured);return captured;
    },
    render(captured,attachments){
      check();if(busy)fail('REENTRANT','Source background submission cannot be reentered');
      if(!captures.has(captured))fail('FRAME','Capture this source background before rendering it');
      busy=true;
      try{renderer.render({...attachments,...captured});live();captures.delete(captured);return owner;}
      catch(error){if(renderer?.failed)stop(error);throw error;}
      finally{busy=false;if(disposed||terminal)release();}
    },
    async whenIdle(){live();try{await Promise.race([renderer.whenIdle(),stopped]);live();return owner;}catch(error){if(renderer?.failed)stop(error);throw error;}},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during source background submission');
      if(!disposed){disposed=true;rejectStop(new ThreeBackgroundError('DISPOSED','Source background is disposed'));release();}},
  });
  return owner;
}
