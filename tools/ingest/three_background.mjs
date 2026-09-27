/** Source-owned 2D, cube and unfiltered HDR backgrounds for the r186 bridge.
 * Borrow ready source textures and cameras; own texture residency plus one
 * native fullscreen renderer. Never replace scene.background, decode a
 * URL, run PMREM, install a frame loop or modify the source's upload versions.
 * Screen backgrounds also work with orthographic cameras; directional cube/HDR
 * backgrounds require perspective. PMREM blur is not inferred.
 */
import {inspectThreeEnvironment, captureThreeEnvironmentPixels} from './three_environment.mjs';
import {inspectThreeTexture, createGpuThreeTextures} from './three_textures.mjs';
import {createGpuAnimationBackground, packAnimationBackgroundFrame} from './animation_background.mjs';

export class ThreeBackgroundError extends Error {
  constructor(code, message) {
    super(`THREE_BACKGROUND_${code}: ${message}`);
    this.name = 'ThreeBackgroundError'; this.code = `THREE_BACKGROUND_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeBackgroundError(code, message); };
const same = (a,b) => a.length === b.length && a.every((v,i) => v === b[i]);
/** Shared metadata-only admission. Retain the existing HDR contract, including
 * its error namespace. Byte screen/cube storage reuses source texture admission;
 * no panorama conversion, image decoding or independent upload implementation.
 */
export function inspectThreeBackground(texture, three, {maxPixels = 16 * 1024 * 1024} = {}) {
  if (texture?.type !== three?.UnsignedByteType) {
    return Object.freeze({...inspectThreeEnvironment(texture, three, {maxPixels}), mapping: 'panorama'});
  }
  const shape = inspectThreeTexture(texture, three, {maxPixels, maxTextureBytes:Number.MAX_SAFE_INTEGER});
  const mapping = shape.cube ? 'cube' : 'screen';
  if (shape.channels !== 4 || (shape.cube
      ? ![three.CubeReflectionMapping, three.CubeRefractionMapping].includes(texture.mapping)
      : texture.mapping !== three.UVMapping))
    fail('PROFILE', 'Byte backgrounds require RGBA UVMapping textures or reflection/refraction CubeTextures');
  if (texture.onUpdate !== null) fail('HOOK', 'Background upload callbacks require their explicit texture owner');
  if (mapping === 'screen' && (typeof texture.matrixAutoUpdate !== 'boolean' ||
      texture.updateMatrix !== three.Texture.prototype.updateMatrix))
    fail('HOOK', 'Use the built-in source texture matrix update');
  const signature = [texture, texture.source, texture.image, texture.version, texture.source.version, texture.mapping, shape.key];
  for (const {image, level, layer} of shape.uploads) {
    signature.push(image, level, layer);
    if (shape.data) signature.push(image.data, image.data.buffer);
  }
  return Object.freeze({...shape, texture, mapping, sourceBytes: shape.bytes, signature: Object.freeze(signature)});
}

/** Admission for live scene controls, also used before scene GPU allocation. */
export function inspectThreeBackgroundState(scene) {
  if (typeof scene?.backgroundIntensity !== 'number' || scene.backgroundIntensity < 0 ||
      !Number.isFinite(Math.fround(scene.backgroundIntensity))) fail('VALUE', 'Background intensity must fit nonnegative finite f32');
  if (scene.backgroundBlurriness !== 0) fail('PROFILE', 'Source backgrounds require backgroundBlurriness=0; PMREM blur is not inferred');
  const r = scene.backgroundRotation;
  if (!r?.isEuler || !['XYZ','YXZ','ZXY','ZYX','YZX','XZY'].includes(r.order) ||
      [r.x,r.y,r.z].some(v => typeof v !== 'number' || !Number.isFinite(v))) fail('VALUE', 'Expected a finite source background Euler rotation');
}

/** Source-state capture. Screen UVs request the built-in updateMatrix only
 * when matrixAutoUpdate is true, matching the source background path. Otherwise
 * source matrices are untouched. The caller updates camera world/projection state
 * first (the scene bridge already does so). Camera translation is excluded;
 * inverse authored background rotation applies AFTER camera world rotation.
 * Both WebGL [-1,1] and WebGPU [0,1] projection matrices admit clip z=0.5.
 * Source camera inverse caches, camera transforms and Euler objects are untouched.
 */
export function threeBackgroundFrame(scene, camera, three) {
  inspectThreeBackgroundState(scene);
  const texture = scene.background;
  if (three?.REVISION !== '186') fail('CAMERA', 'Supply the pinned r186 module');
  if (texture && texture.isCubeTexture !== true && texture.mapping === three.UVMapping) {
    if (typeof three.Texture !== 'function' || !(texture instanceof three.Texture) ||
        typeof texture.matrixAutoUpdate !== 'boolean' || texture.updateMatrix !== three.Texture.prototype.updateMatrix)
      fail('HOOK', 'Use a built-in source texture matrix update');
    if (texture.matrixAutoUpdate) texture.updateMatrix();
    const e = texture.matrix?.elements;
    if (e?.length !== 9) fail('VALUE', 'Expected the source UV matrix');
    const captured = Object.freeze({uvTransform: Object.freeze([e[0],e[1],e[3],e[4],e[6],e[7]]),
      intensity: scene.backgroundIntensity});
    packAnimationBackgroundFrame(captured, 'screen');
    return captured;
  }
  if (three?.REVISION !== '186' || typeof three.Matrix4 !== 'function' || typeof three.Camera !== 'function' ||
      !(camera instanceof three.Camera) || !camera.isPerspectiveCamera || camera.isOrthographicCamera || camera.isArrayCamera || camera.reversedDepth ||
      ![three.WebGLCoordinateSystem,three.WebGPUCoordinateSystem].includes(camera.coordinateSystem))
    fail('CAMERA', 'Directional backgrounds require one non-reversed source perspective camera');
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
  // r186 WebGLBackground mirrors the lookup X axis for uploaded CubeTextures,
  // AFTER inverse authored rotation. Do not reorder or flip the six faces here.
  if (texture?.isCubeTexture === true && texture.isRenderTargetTexture === false)
    for (let column = 0; column < 4; column++) direction.elements[column * 4] *= -1;
  const captured = Object.freeze({directionFromClip: Object.freeze(Array.from(direction.elements)), intensity: scene.backgroundIntensity});
  packAnimationBackgroundFrame(captured); // Validate packed rays before any frame GPU work.
  return captured;
}

/** maxBytes charges all source faces/mips and the 128-byte draw packet.
 * The scene retains/charges the previous owner until submitted consumers drain.
 * HDR conversion is unchanged. Byte images/cubes use the same versioned texture
 * pool as materials; source changes require prepare(), UV/intensity changes do not.
 */
export async function createGpuThreeBackground(device, texture, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected background options');
  for (const key of Object.keys(options)) if (!['three','maxBytes','maxPixels','format','sampleCount','label','signal'].includes(key))
    fail('OPTIONS', `Unknown source background option: ${key}`);
  const {three, maxBytes = 128 * 1024 * 1024, maxPixels = 16 * 1024 * 1024, format = 'rgba8unorm',
    sampleCount = 1, label = 'f3d-source-background', signal} = options;
  const shape = inspectThreeBackground(texture, three, {maxPixels});
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || shape.sourceBytes + 128 > maxBytes)
    fail('LIMIT', 'Source texture and draw uniform exceed the background byte budget');
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
  const panorama = shape.mapping === 'panorama';
  const pixels = panorama ? captureThreeEnvironmentPixels(texture, three, {maxPixels}) : null;
  const lifetime = new AbortController();
  let input = null, textureOwner = null, renderer = null, disposed = false, terminal = null, busy = false, rejectStop;
  const captures = new WeakSet(), stopped = new Promise((_,reject) => {rejectStop=reject;}); stopped.catch(()=>{});
  function release() {
    texture.removeEventListener('dispose', onSourceDispose); signal?.removeEventListener('abort', onAbort);
    lifetime.abort(); renderer?.dispose(); renderer=null;
    if (textureOwner) textureOwner.dispose(); else input?.destroy();
    textureOwner=null; input=null;
  }
  function stop(error) { if (!terminal && !disposed) {terminal=error;rejectStop(error);if(!busy)release();} }
  function onAbort() {stop(new ThreeBackgroundError('ABORTED','Source background lifetime was aborted'));}
  function onSourceDispose() {stop(new ThreeBackgroundError('SOURCE_DISPOSED','Source background texture was disposed'));}
  function live() {
    if(disposed)fail('DISPOSED','Source background is disposed'); if(terminal)throw terminal;
    if(renderer?.failed || renderer?.disposed || textureOwner?.failed || textureOwner?.disposed){stop(new ThreeBackgroundError('DEVICE','Native background renderer failed'));throw terminal;}
  }
  function check() {
    live(); if(!same(shape.signature,inspectThreeBackground(texture,three,{maxPixels}).signature))
      fail('PREPARE','Background source changed; prepare its current version before rendering');
  }
  texture.addEventListener('dispose',onSourceDispose);signal?.addEventListener('abort',onAbort,{once:true});
  device.lost.then(info=>stop(new ThreeBackgroundError('DEVICE',info?.message??'GPU device lost')),stop);
  try {
    if(signal?.aborted)onAbort();check();
    let sampling = {};
    if (panorama) {
      device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');let error;
      try {
        input=device.createTexture({label,size:[shape.width,shape.height,1],dimension:'2d',format:'rgba16float',
          mipLevelCount:1,sampleCount:1,usage:2|4});
        device.queue.writeTexture({texture:input},pixels,{bytesPerRow:shape.width*8},[shape.width,shape.height,1]);
      }catch(cause){error=cause;}
      const validation=device.popErrorScope(),memory=device.popErrorScope();
      const errors=await Promise.race([Promise.all([validation,memory]),stopped]);
      if(error)throw error;if(errors.some(Boolean))fail('DEVICE',errors.find(Boolean).message??'Background upload failed');check();
    } else {
      textureOwner = createGpuThreeTextures(device, {three, maxTextures:1,
        maxTextureBytes:maxBytes-128, maxPixels, label});
      textureOwner.prepare([texture]);
      await Promise.race([textureOwner.whenIdle(), stopped]); check();
      input = textureOwner.nativeTexture(texture);
      sampling = {sampler:textureOwner.binding(texture).sampler, viewFormat:shape.viewFormat, autoLod:true};
    }
    const constructing=createGpuAnimationBackground(device,input,{format,sampleCount,label,mapping:shape.mapping,...sampling,signal:lifetime.signal}).then(value=>{
      if(disposed||terminal){value.dispose();throw terminal??new ThreeBackgroundError('DISPOSED','Source background is disposed');}
      renderer=value;return value;
    });
    await Promise.race([constructing,stopped]);check();
  }catch(error){stop(error);release();throw error;}
  const owner=Object.freeze({source:texture,signature:shape.signature,
    get disposed(){return disposed;},get failed(){return terminal!==null||!!renderer?.failed||!!textureOwner?.failed;},
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
      catch(error){if(renderer?.failed||textureOwner?.failed)stop(error);throw error;}
      finally{busy=false;if(disposed||terminal)release();}
    },
    async whenIdle(){live();try{await Promise.race([Promise.all([renderer.whenIdle(),textureOwner?.whenIdle()]),stopped]);live();return owner;}catch(error){if(renderer?.failed||textureOwner?.failed)stop(error);throw error;}},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during source background submission');
      if(!disposed){disposed=true;rejectStop(new ThreeBackgroundError('DISPOSED','Source background is disposed'));release();}},
  });
  return owner;
}
