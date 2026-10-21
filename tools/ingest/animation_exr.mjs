/** Bounded single-part scanline OpenEXR -> owned, top-down linear RGBA16F.
 * Supports RGB(A)/Y(A), HALF/FLOAT/UINT and NONE/RLE/ZIPS/ZIP/PIZ compression.
 * ZIP uses the host DecompressionStream, not a second runtime or codec package.
 * No GPU, file I/O, tone mapping, exposure change or import-time work.
 * Layout: https://openexr.com/en/latest/OpenEXRFileLayout.html
 * Predictor/interleave: OpenEXR ImfZip.cpp / ImfRleCompressor.cpp (format rules).
 * Tiled, deep, multipart, subsampled, layered and other compression profiles
 * require their retained loader; unsupported input is never partially published.
 */
import {decodeExrPiz, EXR_PIZ_TABLE_BYTES} from './animation_exr_piz.mjs';

export class AnimationExrError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationExrError'; this.code = code; }
}
const fail = (code, message) => { throw new AnimationExrError('ANIMATION_EXR_' + code, message); };
const abort = signal => { if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError'); };
const positive = (n, label) => { if (!Number.isSafeInteger(n) || n < 1) fail('LIMIT', `Invalid ${label}`); return n; };
const COMPRESSION = Object.freeze(['none', 'rle', 'zips', 'zip', 'piz']);
function storage(input) {
  if (input instanceof ArrayBuffer) {
    try { input = new Uint8Array(input); } catch { fail('STORAGE', 'Input is detached'); }
  }
  if (!(input instanceof Uint8Array) || !(input.buffer instanceof ArrayBuffer) || input.buffer.resizable)
    fail('STORAGE', 'Expected fixed, unshared ArrayBuffer or Uint8Array');
  try { return new Uint8Array(input.buffer, input.byteOffset, input.byteLength); }
  catch { fail('STORAGE', 'Input is detached'); }
}
function options(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) fail('OPTIONS', 'Expected decode options');
  for (const key of Object.keys(input)) if (!['maxInputBytes','maxDecodedBytes','maxHeaderBytes','maxDimension','overflow','signal'].includes(key)) fail('OPTIONS', `Unknown option: ${key}`);
  const result = {maxInputBytes: 64*1024*1024, maxDecodedBytes: 256*1024*1024,
    maxHeaderBytes: 64*1024, maxDimension: 16384, overflow: 'reject', ...input};
  for (const key of ['maxInputBytes','maxDecodedBytes','maxHeaderBytes','maxDimension']) positive(result[key], key);
  if (!['reject','clamp'].includes(result.overflow)) fail('OPTIONS', 'overflow must be reject or clamp');
  const signal = result.signal;
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' || typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected AbortSignal');
  abort(signal); return result;
}
function parse(data, limits) {
  if (data.length > limits.maxInputBytes) fail('LIMIT', 'Encoded image exceeds input budget');
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (data.length < 9 || view.getUint32(0, true) !== 20000630) fail('HEADER', 'Invalid OpenEXR signature');
  const version = view.getUint32(4, true), longNames = (version & 1024) !== 0;
  if (version !== 2 && version !== 1026) fail('PROFILE', 'Expected version 2 single-part scanline image');
  const decode = new TextDecoder('utf-8', {fatal: true}), attributes = new Map();
  let at = 8;
  function string(end) {
    const start = at, maximum = longNames ? 255 : 31;
    while (at < end && data[at] !== 0 && at-start <= maximum) at++;
    if (at >= end || at-start > maximum) fail('HEADER', 'Truncated or excessive name');
    let value; try { value = decode.decode(data.subarray(start, at)); } catch { fail('HEADER', 'Invalid UTF-8 name'); }
    at++; return value;
  }
  const headerEnd = Math.min(data.length, limits.maxHeaderBytes);
  for (;;) {
    const name = string(headerEnd); if (!name) break;
    const type = string(headerEnd);
    if (!type || attributes.has(name) || at+4 > headerEnd) fail('HEADER', 'Invalid or duplicate attribute');
    const length = view.getInt32(at, true); at += 4;
    if (length < 0 || length > headerEnd-at) fail('HEADER', 'Attribute exceeds header extent/budget');
    attributes.set(name, {type, at, length}); at += length;
  }
  const table = at;
  function attr(name, type, length, required = true) {
    const a = attributes.get(name);
    if (!a && !required) return null;
    if (!a || a.type !== type || (length !== null && a.length !== length)) fail('HEADER', `Invalid ${name} attribute`);
    return a;
  }
  function float(name, required = true) {
    const a = attr(name, 'float', 4, required); if (!a) return undefined;
    const value = view.getFloat32(a.at, true); if (!Number.isFinite(value)) fail('HEADER', `Nonfinite ${name}`); return value;
  }
  function box(name) {
    const a = attr(name, 'box2i', 16), v = [0,4,8,12].map(i => view.getInt32(a.at+i, true));
    if (v[2] < v[0] || v[3] < v[1]) fail('HEADER', `Empty ${name}`);
    return Object.freeze(v);
  }
  const dataWindow = box('dataWindow'), displayWindow = box('displayWindow');
  const width = dataWindow[2]-dataWindow[0]+1, height = dataWindow[3]-dataWindow[1]+1;
  if (width > limits.maxDimension || height > limits.maxDimension) fail('LIMIT', 'Data window exceeds dimension budget');
  const compression = data[attr('compression', 'compression', 1).at];
  if (compression > 4) fail('PROFILE', 'Only NONE/RLE/ZIPS/ZIP/PIZ compression is supported');
  const order = data[attr('lineOrder', 'lineOrder', 1).at];
  if (order > 2) fail('HEADER', 'Invalid scanline order');
  if (float('pixelAspectRatio') !== 1) fail('PROFILE', 'Non-square pixels require the source loader');
  const center = attr('screenWindowCenter', 'v2f', 8);
  if (![0,4].every(i => Number.isFinite(view.getFloat32(center.at+i, true))) || !(float('screenWindowWidth') > 0)) fail('HEADER', 'Invalid screen window');
  const gamma = float('gamma', false);
  if (gamma !== undefined && gamma !== 1) fail('PROFILE', 'Nonlinear gamma requires explicit conversion');
  const chroma = attr('chromaticities', 'chromaticities', 32, false);
  if (chroma) {
    const srgb = [0.64,0.33,0.30,0.60,0.15,0.06,0.3127,0.3290];
    if (srgb.some((v,i) => !Number.isFinite(view.getFloat32(chroma.at+4*i, true)) || Math.abs(view.getFloat32(chroma.at+4*i, true)-v) > 1e-5)) fail('PROFILE', 'Only linear-sRGB/Rec.709 chromaticities are supported');
  }
  // Do not guess how a color-adaptation/view transform should alter radiance.
  for (const name of ['adoptedNeutral','colorCorrection','multiView']) if (attributes.has(name)) fail('PROFILE', `Unsupported image interpretation: ${name}`);
  const envmap = attr('envmap', 'envmap', 1, false);
  if (envmap && data[envmap.at] > 1) fail('HEADER', 'Invalid environment map type');
  const exposure = float('exposure', false), whiteLuminance = float('whiteLuminance', false);
  const list = attr('channels', 'chlist', null), channels = [];
  at = list.at; const end = at + list.length;
  for (;;) {
    const name = string(end); if (!name) break;
    if (channels.length === 4 || at+16 > end) fail('PROFILE', 'Expected at most four full-resolution color channels');
    const type = view.getInt32(at, true), linear = data[at+4];
    if (![0,1,2].includes(type) || linear > 1 || data[at+5] || data[at+6] || data[at+7]) fail('HEADER', 'Invalid channel descriptor');
    if (view.getInt32(at+8, true) !== 1 || view.getInt32(at+12, true) !== 1) fail('PROFILE', 'Subsampled channels require the source loader');
    if (!['R','G','B','A','Y'].includes(name) || channels.some(c => c.name === name)) fail('PROFILE', 'Expected unique RGB(A) or Y(A) channels');
    channels.push({name, type, bytes: type === 1 ? 2 : 4, component: {R:0,G:1,B:2,A:3,Y:-1}[name]}); at += 16;
  }
  if (at !== end) fail('HEADER', 'Trailing channel-list data');
  const names = channels.map(c => c.name), grayscale = names.includes('Y');
  if (grayscale ? names.some(n => n !== 'Y' && n !== 'A') : !['R','G','B'].every(n => names.includes(n))) fail('PROFILE', 'Incomplete RGB or mixed luminance channels');
  // The file layout, not attribute-list insertion order, defines pixel ordering.
  channels.sort((a,b) => a.name < b.name ? -1 : 1);
  const rows = compression === 4 ? 32 : compression === 3 ? 16 : 1, count = Math.ceil(height/rows);
  const rowBytes = width * channels.reduce((n,c) => n+c.bytes, 0);
  const byteLength = width*height*8, scratchBytes = compression ? rowBytes*Math.min(rows,height)*2 : 0;
  const peakDecodedBytes = byteLength + scratchBytes + (compression === 4 ? EXR_PIZ_TABLE_BYTES : 0);
  if (!Number.isSafeInteger(peakDecodedBytes) || peakDecodedBytes > limits.maxDecodedBytes) fail('LIMIT', 'RGBA16F output and block scratch exceed decoded budget');
  const tableEnd = table+count*8;
  if (tableEnd > data.length) fail('HEADER', 'Truncated chunk offset table');
  const chunks = [];
  for (let i=0; i<count; i++) {
    const offset64 = view.getBigUint64(table+i*8, true);
    if (offset64 > BigInt(Number.MAX_SAFE_INTEGER)) fail('DATA', 'Chunk offset exceeds safe integer');
    const offset = Number(offset64);
    if (offset < tableEnd || offset > data.length-8) fail('DATA', 'Chunk offset outside pixel storage');
    const y = view.getInt32(offset, true), length = view.getInt32(offset+4, true);
    const rowCount = Math.min(rows, height-i*rows), rawBytes = rowCount*rowBytes;
    if (y !== dataWindow[1]+i*rows || length < 1 || length > rawBytes || length > data.length-offset-8 || (!compression && length !== rawBytes)) fail('DATA', 'Invalid chunk row or extent');
    chunks.push({offset: offset+8, length, row: i*rows, rowCount, rawBytes});
  }
  const ranges = chunks.slice().sort((a,b) => a.offset-b.offset);
  for (let i=1; i<ranges.length; i++) if (ranges[i].offset-8 < ranges[i-1].offset+ranges[i-1].length) fail('DATA', 'Overlapping chunks');
  const metadata = Object.freeze({width, height, byteLength, peakDecodedBytes, dataWindow, displayWindow,
    compression: COMPRESSION[compression], channels: Object.freeze(names),
    ...(envmap ? {environmentMap: data[envmap.at] === 0 ? 'latlong' : 'cube'} : {}),
    ...(exposure === undefined ? {} : {exposure}), ...(whiteLuminance === undefined ? {} : {whiteLuminance})});
  return {metadata, chunks, channels, compression, grayscale};
}
/** Validate header and every chunk range before decompression or output allocation.
 * Memory accounting includes owned output, two worst-case decoded blocks and
 * PIZ typed Huffman/range tables when selected. It excludes
 * the input snapshot, JS metadata, or opaque native decompressor working memory.
 */
export function inspectAnimationExr(input, settings = {}) {
  const limits = options(settings); return parse(storage(input), limits).metadata;
}
function undoPredictor(data) {
  for (let i=1; i<data.length; i++) data[i] = data[i-1]+data[i]-128;
  const result = new Uint8Array(data.length), half = Math.ceil(data.length/2);
  for (let i=0; i<half; i++) { result[2*i] = data[i]; if (2*i+1 < data.length) result[2*i+1] = data[half+i]; }
  return result;
}
function rle(input, expected) {
  const result = new Uint8Array(expected); let at=0, out=0;
  while (at<input.length) {
    const code = input[at++], signed = code > 127 ? code-256 : code, length = signed < 0 ? -signed : signed+1;
    if (length > expected-out || (signed < 0 ? length : 1) > input.length-at) fail('DATA', 'Invalid RLE packet');
    if (signed < 0) { result.set(input.subarray(at, at+length), out); at += length; }
    else result.fill(input[at++], out, out+length);
    out += length;
  }
  if (out !== expected) fail('DATA', 'Incomplete RLE block'); return result;
}
async function inflate(input, expected, signal) {
  if (typeof DecompressionStream !== 'function') fail('DECOMPRESS', 'ZIP EXR requires DecompressionStream');
  const stream = new Blob([input]).stream().pipeThrough(new DecompressionStream('deflate'));
  const reader = stream.getReader(), result = new Uint8Array(expected); let at=0, complete=false;
  let rejectStop; const stopped = new Promise((_,reject) => { rejectStop=reject; }); stopped.catch(() => {});
  const cancel = () => { try { Promise.resolve(reader.cancel(signal?.reason)).catch(() => {}); } catch {} };
  const onAbort = () => { rejectStop(signal.reason ?? new DOMException('Aborted','AbortError')); cancel(); };
  signal?.addEventListener('abort', onAbort, {once:true});
  try {
    abort(signal);
    for (;;) {
      const part = await Promise.race([reader.read(), stopped]); abort(signal);
      if (part.done) { complete=true; break; }
      if (part.value.byteLength > expected-at) fail('DATA', 'ZIP output exceeds declared block');
      result.set(part.value, at); at += part.value.byteLength;
    }
    if (at !== expected) fail('DATA', 'Incomplete ZIP block'); return result;
  } catch (error) {
    abort(signal);
    if (error instanceof AnimationExrError) throw error;
    fail('DECOMPRESS', 'Invalid ZIP stream: ' + (error?.message ?? String(error)));
  } finally {
    signal?.removeEventListener('abort', onAbort); if (!complete) cancel();
    try { reader.releaseLock(); } catch {}
  }
}
const roundEven = value => { const n=Math.floor(value), tail=value-n; return n + (tail>0.5 || (tail===0.5 && n%2) ? 1 : 0); };
function half(value) {
  const sign = value<0 || Object.is(value,-0) ? 0x8000 : 0; value=Math.abs(value);
  if (value < 2**-14) return sign | roundEven(value*2**24);
  const exponent=Math.floor(Math.log2(value)), mantissa=roundEven(value/2**(exponent-10));
  return sign | ((exponent+15)<<10) + mantissa-1024;
}
/** Decode real EXR blocks. The input is snapshotted before the first await;
 * cancellation discards the whole unpublished image, never a partial result.
 * HALF values are copied bit-exactly; FLOAT/UINT narrow once with ties-to-even.
 * Nonfinite channels always reject. overflow:'clamp' explicitly allows signed
 * saturation to 65504 and reports its count. Alpha and signed radiance are kept;
 * exposure metadata, channel pLinear and display windows never rescale pixels.
 */
export async function decodeAnimationExr(input, settings = {}) {
  const limits=options(settings), source=storage(input), plan=parse(source, limits);
  const data=source.slice(), output=new Uint16Array(plan.metadata.byteLength/2);
  let clampedComponents=0;
  for (let i=3; i<output.length; i+=4) output[i]=0x3c00;
  for (const chunk of plan.chunks) {
    abort(limits.signal);
    let raw=data.subarray(chunk.offset,chunk.offset+chunk.length);
    if (raw.length !== chunk.rawBytes) {
      raw = plan.compression === 4
        ? decodeExrPiz(raw, plan.metadata.width, chunk.rowCount, plan.channels, fail)
        : undoPredictor(plan.compression === 1 ? rle(raw,chunk.rawBytes) : await inflate(raw,chunk.rawBytes,limits.signal));
    }
    abort(limits.signal);
    const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength); let at=0;
    for (let y=0; y<chunk.rowCount; y++) for (const channel of plan.channels) for (let x=0; x<plan.metadata.width; x++) {
      let encoded;
      if (channel.type===1) {
        encoded=view.getUint16(at,true);
        if ((encoded & 0x7c00) === 0x7c00) fail('VALUE','Nonfinite HALF sample');
      } else {
        let value=channel.type===2 ? view.getFloat32(at,true) : view.getUint32(at,true);
        if (!Number.isFinite(value)) fail('VALUE','Nonfinite FLOAT sample');
        if (Math.abs(value)>65504) {
          if (limits.overflow==='reject') fail('VALUE','Radiance exceeds RGBA16F range');
          value=Math.sign(value)*65504; clampedComponents += channel.name==='Y' ? 3 : 1;
        }
        encoded=half(value);
      }
      const pixel=((chunk.row+y)*plan.metadata.width+x)*4;
      if (channel.component===-1) output.fill(encoded,pixel,pixel+3);
      else output[pixel+channel.component]=encoded;
      at+=channel.bytes;
    }
  }
  abort(limits.signal);
  return Object.freeze({...plan.metadata, data:output, clampedComponents,
    colorSpace:'linear-srgb', format:'rgba16float', inputBytes:data.length});
}
