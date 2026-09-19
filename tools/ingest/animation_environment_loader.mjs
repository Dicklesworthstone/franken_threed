/** Load one Radiance HDR panorama into the existing animation IBL preparation
 * pipeline. The returned map is directly usable as frame.environment.map.
 * CPU bytes and the temporary upload texture are released before publication;
 * the prepared map owns only its diffuse/specular/DFG textures, never the device.
 * No image element, canvas, Three loader, frame loop or import-time I/O.
 */
import {decodeAnimationHdr, AnimationHdrError} from './animation_hdr.mjs';
import {createGpuAnimationEnvironment, planAnimationEnvironment, AnimationEnvironmentError} from './animation_environment.mjs';
const fail = (code, message) => { throw new AnimationHdrError('ANIMATION_HDR_' + code, message); };
const positive = (value, label) => {
  if (!Number.isSafeInteger(value) || value < 1) fail('LIMIT', `Invalid ${label}`);
  return value;
};
const ignore = () => {};
function cancelBody(response, reason) {
  try { Promise.resolve(response?.body?.cancel(reason)).catch(ignore); } catch { /* A borrowed implementation may already have closed it. */ }
}
function sourceUrl(source, baseURL) {
  let url;
  try { url = new URL(source, baseURL); } catch { fail('URL', 'Relative HDR URLs require baseURL'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) fail('URL', 'Expected HTTP(S) URL without embedded credentials');
  url.hash = ''; return url.href;
}

/** source is ArrayBuffer/Uint8Array or HTTP(S) URL; relative URLs need baseURL.
 * Fetch uses credentials:omit and redirect:error. An injected fetch must implement
 * streaming Response.body; this is not an SSRF sandbox for custom transports.
 * maxInputBytes limits the actual streamed body even without Content-Length.
 * Streaming assembly grows a single bounded buffer, not an unbounded chunk list;
 * old/new buffers may coexist briefly during growth (at most 2*maxInputBytes).
 * maxDecodedBytes includes RGBA16F output and scanline scratch (see decoder).
 * maxTextureBytes bounds PEAK logical GPU texels: upload panorama + filtered maps.
 * Filter uniforms remain separately checked against the device's maxBufferSize.
 * Cancellation/device loss races every asynchronous boundary, including custom
 * fetch/read/pipeline implementations that ignore AbortSignal. Submitted GPU work
 * cannot be rolled back. The signal controls construction, not the returned map.
 */
export async function loadGpuAnimationEnvironment(device, source, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected HDR loading options');
  const filterKeys = ['size', 'diffuseSize', 'lutSize', 'samples', 'maxSampleWork'];
  for (const key of Object.keys(options)) if (!['baseURL', 'fetch', 'signal', 'maxInputBytes', 'maxDecodedBytes', 'maxHeaderBytes', 'maxTextureBytes', 'overflow', ...filterKeys].includes(key)) fail('OPTIONS', `Unknown HDR loading option: ${key}`);
  const {baseURL, fetch: fetcher = globalThis.fetch, signal, maxInputBytes = 64 * 1024 * 1024,
    maxDecodedBytes = 256 * 1024 * 1024, maxHeaderBytes = 64 * 1024,
    maxTextureBytes = 128 * 1024 * 1024, overflow = 'reject'} = options;
  for (const [key, value] of Object.entries({maxInputBytes, maxDecodedBytes, maxHeaderBytes, maxTextureBytes})) positive(value, key);
  if (!['reject', 'clamp'].includes(overflow)) fail('OPTIONS', 'overflow must be reject or clamp');
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' || typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected AbortSignal');
  if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError');
  for (const key of ['createTexture', 'createBuffer', 'createSampler', 'createBindGroupLayout', 'createPipelineLayout', 'createShaderModule', 'createRenderPipelineAsync', 'createBindGroup', 'createCommandEncoder', 'pushErrorScope', 'popErrorScope'])
    if (typeof device?.[key] !== 'function') fail('DEVICE', `Missing WebGPU ${key}`);
  for (const key of ['writeTexture', 'writeBuffer', 'submit', 'onSubmittedWorkDone']) if (typeof device.queue?.[key] !== 'function') fail('DEVICE', `Missing GPU queue.${key}`);
  if (typeof device.lost?.then !== 'function') fail('DEVICE', 'Expected device loss notification');
  const maxDimension = positive(device.limits?.maxTextureDimension2D, 'device texture dimension');
  const alignment = positive(device.limits?.minUniformBufferOffsetAlignment, 'device uniform alignment');
  const filter = Object.fromEntries(filterKeys.filter(key => options[key] !== undefined).map(key => [key, options[key]]));
  const plan = planAnimationEnvironment({...filter, maxTextureBytes, maxDimension, alignment});
  if (plan.uniformBytes > positive(device.limits?.maxBufferSize, 'device buffer limit')) fail('LIMIT', 'Filter uniforms exceed device limit');
  const url = typeof source === 'string' || source instanceof URL ? sourceUrl(source, baseURL) : null;
  if (url !== null && typeof fetcher !== 'function') fail('OPTIONS', 'Expected Fetch API implementation');
  const controller = new AbortController();
  let active = true, ended = false, stopReason, reader = null, uploadTexture = null, prepared = null;
  let rejectStop;
  const stopped = new Promise((_, reject) => { rejectStop = reject; }); stopped.catch(ignore);
  function stop(reason) {
    if (active && !ended) { ended = true; stopReason = reason; controller.abort(reason); rejectStop(reason); }
  }
  const onAbort = () => stop(signal.reason ?? new DOMException('Aborted', 'AbortError'));
  signal?.addEventListener('abort', onAbort, {once: true});
  if (signal?.aborted) onAbort();
  device.lost.then(info => stop(new AnimationEnvironmentError('ANIMATION_ENVIRONMENT_LOST', info?.message ?? 'Device lost')), stop);
  function check() { if (ended) throw stopReason; }
  const wait = promise => Promise.race([promise, stopped]);
  function releaseReader(cancel) {
    const current = reader; reader = null;
    if (current) {
      if (cancel) { try { Promise.resolve(current.cancel(stopReason)).catch(ignore); } catch {} }
      try { current.releaseLock(); } catch {}
    }
  }
  async function readUrl() {
    check();
    const request = Promise.resolve(fetcher(url, {credentials: 'omit', redirect: 'error', signal: controller.signal})).then(response => {
      if (ended) cancelBody(response, stopReason);
      return response;
    });
    const response = await wait(request);
    try {
      check();
      if (!response?.ok) fail('HTTP', `HDR request failed with status ${response?.status ?? 'unknown'}`);
      if (response.redirected || (response.url && sourceUrl(response.url) !== url)) fail('HTTP', 'HDR redirects are not allowed');
      const length = response.headers?.get?.('content-length');
      if (length != null && (!/^\d+$/.test(length) || !Number.isSafeInteger(Number(length)) || Number(length) > maxInputBytes)) fail('LIMIT', 'HDR Content-Length exceeds byte budget or is invalid');
      if (typeof response.body?.getReader !== 'function') fail('HTTP', 'HDR loading requires a streaming response body');
      reader = response.body.getReader();
    } catch (error) { cancelBody(response, error); throw error; }
    let storage = new Uint8Array(0), total = 0;
    for (;;) {
      const part = await wait(reader.read()); check();
      if (part.done) break;
      const chunk = part.value;
      if (!(chunk instanceof Uint8Array) || !(chunk.buffer instanceof ArrayBuffer) || chunk.buffer.resizable) fail('HTTP', 'Expected fixed, unshared response bytes');
      const next = total + chunk.byteLength;
      if (!Number.isSafeInteger(next) || next > maxInputBytes) fail('LIMIT', 'Streamed HDR exceeds byte budget');
      if (next > storage.length) {
        const grown = new Uint8Array(Math.min(maxInputBytes, Math.max(next, 65536, storage.length * 2)));
        grown.set(storage.subarray(0, total)); storage = grown;
      }
      storage.set(chunk, total); total = next;
    }
    releaseReader(false);
    return storage.subarray(0, total);
  }
  try {
    // Byte inputs are decoded synchronously before the first await, so the caller
    // cannot mutate borrowed pixels while network/driver work is outstanding.
    let encoded = url === null ? source : await readUrl(); check();
    let image = decodeAnimationHdr(encoded, {maxInputBytes, maxDecodedBytes, maxHeaderBytes, maxDimension, overflow});
    const inputBytes = encoded.byteLength; encoded = null; source = null;
    if (image.width !== image.height * 2) fail('SOURCE', 'Environment HDR must be a 2:1 equirectangular panorama');
    const peakTextureBytes = image.byteLength + plan.textureBytes;
    if (!Number.isSafeInteger(peakTextureBytes) || peakTextureBytes > maxTextureBytes) fail('LIMIT', 'Panorama plus filtered environment exceeds GPU texture budget');
    const sourceInfo = Object.freeze({width: image.width, height: image.height, inputBytes, exposure: image.exposure,
      clampedComponents: image.clampedComponents, uploadTextureBytes: image.byteLength, peakTextureBytes});
    check();
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let uploadError;
    try {
      uploadTexture = device.createTexture({label: 'Animation HDR panorama', size: [image.width, image.height, 1],
        dimension: '2d', format: 'rgba16float', mipLevelCount: 1, sampleCount: 1, usage: 4 | 2});
      // Queue.writeTexture permits tightly packed rows; the 256-byte copy-buffer
      // stride restriction does not apply. No staging/padded image copy is needed.
      device.queue.writeTexture({texture: uploadTexture}, image.data,
        {offset: 0, bytesPerRow: image.width * 8, rowsPerImage: image.height}, [image.width, image.height, 1]);
    } catch (error) { uploadError = error; }
    image = null;
    const uploadValidation = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(errors => {
      if (uploadError) throw uploadError;
      const error = errors.find(Boolean); if (error) fail('GPU', error.message ?? 'HDR upload failed');
    });
    await wait(uploadValidation); check();
    const filtering = createGpuAnimationEnvironment(device, uploadTexture, {...filter,
      maxTextureBytes: maxTextureBytes - sourceInfo.uploadTextureBytes, signal: controller.signal}).then(environment => {
      if (ended) environment.dispose(); // A late custom driver resolution cannot leak a completed map.
      return environment;
    });
    prepared = await wait(filtering); check();
    // Factory completion includes this earlier upload. Source is no longer used.
    uploadTexture.destroy(); uploadTexture = null;
    const environment = prepared;
    const result = Object.freeze({...environment, sourceInfo,
      get disposed() { return environment.disposed; }, get failed() { return environment.failed; },
      get textureBytes() { return environment.textureBytes; },
      async whenIdle() { await environment.whenIdle(); return result; },
    });
    return result;
  } catch (error) {
    stop(error); prepared?.dispose(); throw error;
  } finally {
    active = false; signal?.removeEventListener('abort', onAbort);
    releaseReader(true); uploadTexture?.destroy(); uploadTexture = null; source = null;
  }
}
