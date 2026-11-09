/** Source RenderTarget residency shared by all scenes of one renderer.
 * Native attachments stay top-left for rendering/readback. A lazily allocated
 * GPU-only sampling image preserves the selected source surface's texture UVs:
 * native WebGPU keeps rows unchanged; WebGL flips vertically. It is refreshed
 * once per written version, never downloaded or re-uploaded.
 * generateMipmaps:true keeps a full native color pyramid and copies each level
 * with the selected orientation into the sampling pyramid. Both chains are
 * charged, including NPOT tails.
 * The WebGL surface samples ordinary DepthTexture attachments through a separate
 * UV-oriented r32float image: no color conversion or half-float quantization.
 * Comparison inputs use an independent depth32float image and preserve the
 * source fixed/float comparison domain plus its initialized sampler lifetime.
 * Native node/compute inputs borrow the depth attachment directly, with distinct
 * ordinary and comparison samplers captured for each accepted use.
 * Native MRT inputs retain each ordered source color attachment, format and
 * sampler. All colors share the depth, budget and generation lifetime; the
 * WebGL UV replica continues to cover the primary color only.
 * Registered native PassNode histories bind persistent current/previous color
 * identities to frozen per-use attachment selections. Resolved and MSAA colors
 * alternate independently while depth and non-history colors remain shared.
 * Queued selections survive recognized source resize; every detached image is
 * charged and retained until its readers and submitted work have finished.
 * Native point shadows use six square cube faces. Rendering selects one face,
 * while native depth sampling borrows all six layers without a UV replica.
 * Cube targets admit one sample, authored levels and source-ordered generation.
 * Their allocation mip count stays fixed until the source texture version
 * changes; CubeCamera's temporary generation flag cannot replace prior faces.
 * Explicit face-aware transfers, readbacks and six-layer sampling share this
 * owner. Cube allocation extents follow the source face descriptors even when
 * setSize() changes only the logical target and viewport dimensions.
 * Layered RenderTarget and WebGLArrayRenderTarget colors retain native 2D-array
 * storage; the selected face integer addresses one array layer. Native array
 * MRT colors keep independent mip counts and generation choices, with every
 * allocated level retaining all array layers. Shared ordinary
 * depth is the source default, while useArrayDepthTexture or authored array
 * depth preserves independent layers. Native sampling borrows whole arrays;
 * WebGL array sampling retains every layer/mip in an oriented GPU replica.
 * Explicit transfers/readbacks select one color or depth layer at a time.
 * RenderTarget3D and WebGL3DRenderTarget retain 3D color storage. Their selected
 * face is a mip-local depth slice; native sampling borrows the entire volume.
 * WebGL base-level colors sample through a 3D replica, preserving interpolation
 * between oriented slices without CPU transfers or 2D-array substitution.
 * Color mip extents shrink in XYZ while attached depth remains 2D or an
 * independently selected array. Every volume MRT color retains its own authored
 * mip chain and precision; rendering selects a mip shared by all attachments.
 * Volume mip levels are authored, not generated.
 * Private stencil buffers and UnsignedInt248 DepthStencilFormat attachments
 * retain both aspects for rendering; depth sampling selects only the depth aspect.
 * Source FloatType R/RG/RGBA attachments retain full precision through storage,
 * UV replicas and readback. Nearest sampling needs no optional device feature;
 * linear filtering requires float32-filterable. This owner does not extend the
 * separate draw renderer's admitted attachment/pipeline formats.
 * writePixels(entry, data, options) and copyColor(source, destination, options)
 * update a colorIndex-selected native level-zero attachment. Only primary color
 * writes invalidate its WebGL UV replica; native MRT reads borrow actual images.
 * copyDepth(source, destination, options) instead invalidates only the depth
 * replica, preserving color mipmaps and stencil. All copies share admission,
 * allocation budgets, generation checks and nonreentrant native submission.
 * These explicit owner operations do not enqueue the public renderer's FIFO.
 * This is general WebGPU execution, not Rust specialization or a speedup claim.
 */
import * as gpuTargets from './gpu_render_target.mjs';
import {inspectGpuTargetWrite, inspectGpuTargetCopy, gpuTargetCopyOverlaps} from './gpu_target_transfer.mjs';
import {inspectGpuDepthCopy} from './gpu_depth_transfer.mjs';
import {threeDepthTextureFormat} from './three_depth_textures.mjs';

export class ThreeRenderTargetError extends Error {
  constructor(code, message) {
    super(`THREE_TARGET_${code}: ${message}`);
    this.name = 'ThreeRenderTargetError'; this.code = `THREE_TARGET_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeRenderTargetError(code, message); };
const positive = n => Number.isSafeInteger(n) && n > 0;
const extent = n => Number.isSafeInteger(n) && n >= 0;

function inspectCubeFaces(texture, kind) {
  const faces = texture.image;
  if (!Array.isArray(faces) || faces.length !== 6)
    fail('SOURCE', `Expected six source cube ${kind} face descriptors`);
  const width = faces[0]?.width, height = faces[0]?.height;
  if (!extent(width) || !extent(height) || (width || 1) !== (height || 1) ||
      faces.some(face => face?.width !== width || face?.height !== height || (face?.depth ?? 1) !== 1))
    fail('SIZE', 'Cube target face descriptors must have consistent square extents');
  // RenderTarget.setSize writes width/height onto the image array itself, not
  // its six elements. Pinned Textures.getSize reads images[0], so a resized
  // target keeps that face's storage extent and mip count. Its viewport stays
  // source-owned and can be smaller or larger than the attachment.
  return {width: width || 1, height: height || 1};
}

function depthComparison(texture, three) {
  if (texture.compareFunction == null) return null;
  const compares = new Map([[three.NeverCompare, 'never'], [three.LessCompare, 'less'],
    [three.EqualCompare, 'equal'], [three.LessEqualCompare, 'less-equal'],
    [three.GreaterCompare, 'greater'], [three.GreaterEqualCompare, 'greater-equal'],
    [three.AlwaysCompare, 'always'], [three.NotEqualCompare, 'not-equal']]);
  const compare = compares.get(texture.compareFunction);
  if (!compare) fail('SAMPLER', 'Expected a source depth comparison function');
  return compare;
}

function captureWebGLTargetSampler(texture, volume = false) {
  return Object.freeze({minFilter: texture.minFilter, magFilter: texture.magFilter,
    wrapS: texture.wrapS, wrapT: texture.wrapT, anisotropy: texture.anisotropy,
    ...(volume ? {isData3DTexture: true, wrapR: texture.wrapR} : {})});
}

function inspectThreeRenderTargetColor(target, three, texture, cubeMipLevelCount = null, mipProfile = null) {
  if (!three?.RenderTarget || !(target instanceof three.RenderTarget)) fail('SOURCE', 'Expected a source RenderTarget');
  const webglCube = target.isWebGLCubeRenderTarget === true;
  const webglArray = target.isWebGLArrayRenderTarget === true && texture?.isDataArrayTexture === true;
  const webglVolume = target.isWebGL3DRenderTarget === true && texture?.isData3DTexture === true;
  const cube = target.isCubeRenderTarget === true || webglCube;
  const volume = target.isRenderTarget3D === true || target.isWebGL3DRenderTarget === true;
  const array = !cube && !volume && (target.depth > 1 || target.isWebGLArrayRenderTarget === true || texture?.isDataArrayTexture === true || texture?.isArrayTexture === true);
  if (!extent(target.width) || !extent(target.height))
    fail('SIZE', 'Target dimensions must be non-negative integers');
  // Source Textures.getSize allocates one texel for a zero image dimension.
  // Keep the logical target/viewport unchanged: Bloom's final small-image mip
  // can have a zero viewport, while its cleared backing texture is still sampled.
  let width = target.width || 1, height = target.height || 1;
  if (!positive(target.depth)) fail('SIZE', 'Target depth must be a positive integer');
  if (target.multiview || cube && volume)
    fail('UNSUPPORTED', 'Expected ordinary 2D, array, volume or cube rendering');
  if (volume ? typeof three.Data3DTexture !== 'function' || !(texture instanceof three.Data3DTexture) || texture.isData3DTexture !== true
      : texture?.isData3DTexture)
    fail('SOURCE', 'Volume targets require source Data3DTexture color attachments');
  if (volume && (![0, 1].includes(target.samples) || texture.generateMipmaps))
    fail('UNSUPPORTED', 'Source volume targets require one sample and authored mip levels');
  if (cube && target.depth !== 1) fail('UNSUPPORTED', 'Cube targets use six faces, not an array depth');
  if (cube && (typeof three[webglCube ? 'WebGLCubeRenderTarget' : 'CubeRenderTarget'] !== 'function' ||
      !(target instanceof three[webglCube ? 'WebGLCubeRenderTarget' : 'CubeRenderTarget'])))
    fail('SOURCE', 'Expected a source cube render target');
  if (!(texture instanceof three.Texture) || texture.isRenderTargetTexture !== true)
    fail('SOURCE', 'Expected the target-owned color texture');
  if (cube) {
    // The pinned CubeRenderTarget replaces RenderTarget's color texture without
    // assigning its renderTarget pointer. The target itself owns that exact
    // CubeTexture; preserve the source's null pointer rather than rewriting it.
    if (typeof three.CubeTexture !== 'function' || !(texture instanceof three.CubeTexture) || texture.isCubeTexture !== true ||
        texture.renderTarget != null && texture.renderTarget !== target)
      fail('SOURCE', 'Expected the source cube target color texture');
    if (![0, 1].includes(target.samples) || target.stencilBuffer || target.useArrayDepthTexture)
      fail('UNSUPPORTED', 'Cube targets require one sample and no stencil or array depth');
    // WebGLTextures.setupFrameBufferTexture uses the target's extent. Native
    // Textures.getSize instead reads image[0], even after RenderTarget.setSize.
    if (!webglCube) ({width, height} = inspectCubeFaces(texture, 'color'));
    else if (width !== height) fail('SIZE', 'Cube target faces must be square');
  } else {
    if (texture.isCubeTexture || texture.renderTarget !== target && !((array && target.isWebGLArrayRenderTarget === true ||
        volume && target.isWebGL3DRenderTarget === true) && texture.renderTarget == null))
      fail('SOURCE', 'Expected the target-owned color texture');
    if ((array || volume) && (texture.image?.depth !== target.depth || texture.image?.width !== target.width || texture.image?.height !== target.height))
      fail('SIZE', 'Layered color image and target extents must match');
  }
  if (target.stencilBuffer && !target.depthBuffer)
    fail('UNSUPPORTED', 'Source stencil targets require depthBuffer');
  if (![0, 1, 4].includes(target.samples)) fail('UNSUPPORTED', 'Target samples must be zero, one or four');
  if (target.samples === 4 && target.resolveColorBuffer === false) fail('UNSUPPORTED', 'MSAA targets must resolve their color');
  const float32 = three.FloatType !== undefined && texture.type === three.FloatType;
  const halfFloat = three.HalfFloatType !== undefined && texture.type === three.HalfFloatType;
  const floatChannels = texture.format === three.RGBAFormat ? 'rgba'
    : three.RedFormat !== undefined && texture.format === three.RedFormat ? 'r'
    : three.RGFormat !== undefined && texture.format === three.RGFormat ? 'rg' : null;
  const floatFormat = floatChannels && (float32 || halfFloat) ? `${floatChannels}${float32 ? 32 : 16}float` : null;
  const integerType = three.IntType !== undefined && texture.type === three.IntType ? 'sint'
    : three.UnsignedIntType !== undefined && texture.type === three.UnsignedIntType ? 'uint' : null;
  const integerChannels = floatChannels ?? (three.RedIntegerFormat !== undefined && texture.format === three.RedIntegerFormat ? 'r'
    : three.RGIntegerFormat !== undefined && texture.format === three.RGIntegerFormat ? 'rg'
    : three.RGBAIntegerFormat !== undefined && texture.format === three.RGBAIntegerFormat ? 'rgba' : null);
  const integerFormat = integerType && integerChannels ? `${integerChannels}32${integerType}` : null;
  const red8 = three.RedFormat !== undefined && texture.format === three.RedFormat && texture.type === three.UnsignedByteType;
  const packedHDR = three.UnsignedInt101111Type !== undefined && texture.type === three.UnsignedInt101111Type &&
    texture.format === three.RGBFormat;
  if (texture.internalFormat != null || (texture.format !== three.RGBAFormat && !floatFormat && !integerFormat && !red8 && !packedHDR))
    fail('UNSUPPORTED', 'Targets require supported R/RG/RGBA or packed RGB11F storage, with no internal format override');
  if (typeof texture.generateMipmaps !== 'boolean') fail('UNSUPPORTED', 'generateMipmaps must be boolean');
  const linear = texture.colorSpace === three.NoColorSpace || texture.colorSpace === three.LinearSRGBColorSpace;
  let format;
  if (red8 && linear) format = 'r8unorm';
  else if (packedHDR && linear) format = 'rg11b10ufloat';
  else if (texture.format === three.RGBAFormat && texture.type === three.UnsignedByteType &&
      (linear || texture.colorSpace === three.SRGBColorSpace))
    format = linear ? 'rgba8unorm' : 'rgba8unorm-srgb';
  else if (floatFormat && linear) format = floatFormat;
  else if (integerFormat && texture.colorSpace === three.NoColorSpace && texture.normalized !== true) format = integerFormat;
  else fail('UNSUPPORTED', 'Target color must be RGBA8, linear R8/RGB11F, linear R/RG/RGBA16F/32F or unnormalized R/RG/RGBA32 integer storage with NoColorSpace');
  if (float32 && target.samples === 4) fail('UNSUPPORTED', 'Float32 targets currently require zero or one source samples');
  if (integerType && (target.samples === 4 || texture.generateMipmaps))
    fail('UNSUPPORTED', 'Integer targets require one sample and authored mip levels');
  const sampleCount = target.samples === 4 ? 4 : 1, depthTexture = target.depthTexture ?? null;
  const layers = cube ? 6 : array || volume ? target.depth : 1;
  const depthArray = (array || volume) && layers > 1 && (target.useArrayDepthTexture === true || (depthTexture?.image?.depth ?? 1) > 1);
  const depthLayers = cube || depthArray ? layers : 1;
  if (depthArray && sampleCount !== 1 && depthTexture !== null)
    fail('UNSUPPORTED', 'Sampled array depth requires one sample');
  const depthAttachment = target.depthBuffer === true || target.stencilBuffer === true;
  let depthFormat = target.depthBuffer ? (target.stencilBuffer ? 'depth24plus-stencil8' : 'depth24plus') : null;
  if (depthTexture !== null) {
    if (typeof three.DepthTexture !== 'function' || !(depthTexture instanceof three.DepthTexture) ||
        depthTexture.isDepthTexture !== true || !(depthTexture.renderTarget instanceof three.RenderTarget) ||
        depthTexture.renderTarget.depthTexture !== depthTexture)
      fail('SOURCE', 'Expected a source DepthTexture with its original render target owner');
    if (depthTexture.renderTarget !== target && (cube || array || volume ||
        depthTexture.renderTarget.isCubeRenderTarget || depthTexture.renderTarget.isWebGLCubeRenderTarget ||
        depthTexture.renderTarget.depth !== 1 || depthTexture.renderTarget.texture?.isData3DTexture))
      fail('UNSUPPORTED', 'Shared depth attachments currently require ordinary 2D targets');
    // Source updateRenderTarget initializes owned depth dimensions lazily, so
    // setSize() may leave the previous image depth visible until preparation.
    // The selected target extent determines the new native depth allocation.
    // A one-layer target always selects ordinary depth, including after an
    // array shrinks while its public depth flag/image still describe old storage.
    if ((!cube && depthTexture.isCubeTexture) || (!cube && !positive(depthTexture.image?.depth ?? 1)))
      fail('UNSUPPORTED', 'Source depth sampling requires a depth attachment of the target dimension');
    if (cube) {
      if (typeof three.CubeDepthTexture !== 'function' || !(depthTexture instanceof three.CubeDepthTexture) ||
          depthTexture.isCubeDepthTexture !== true || depthTexture.isCubeTexture !== true || depthTexture.mapping !== three.CubeReflectionMapping)
        fail('SOURCE', 'Cube targets require an attached source CubeDepthTexture');
      const depthSize = inspectCubeFaces(depthTexture, 'depth');
      if (depthSize.width !== width || depthSize.height !== height)
        fail('SIZE', 'Cube depth and color face extents must match');
    }
    const combined = three.DepthStencilFormat !== undefined && depthTexture.format === three.DepthStencilFormat;
    if (cube && combined) fail('UNSUPPORTED', 'Cube depth/stencil sampling is not admitted');
    if ((!combined && depthTexture.format !== three.DepthFormat) || depthTexture.internalFormat != null ||
        depthTexture.colorSpace !== three.NoColorSpace || depthTexture.generateMipmaps !== false ||
        depthTexture.mipmaps?.length || depthTexture.flipY !== false)
      fail('UNSUPPORTED', 'Depth textures require no mipmaps, UV flipping or color conversion');
    depthComparison(depthTexture, three);
    // WebGLTextures.setupDepthTexture selects the attachment from the texture's
    // format, not stencilBuffer: a depth-only texture has no stencil aspect,
    // while DepthStencilFormat attaches both even with stencilBuffer:false.
    depthFormat = threeDepthTextureFormat(depthTexture, three);
    if (!depthFormat) fail('UNSUPPORTED', 'Unsupported source depth format/type combination');
    // Match the ordinary sampler2D depth-read profile. Do not pretend a missing
    // float32-filterable device feature can be replaced by lower precision.
    if (![three.NearestFilter, three.LinearFilter].includes(depthTexture.minFilter) ||
        ![three.NearestFilter, three.LinearFilter].includes(depthTexture.magFilter) || depthTexture.anisotropy !== 1)
      fail('SAMPLER', 'Depth textures require nearest/linear filtering and anisotropy one');
    samplerDescriptor(depthTexture, three, 1);
  }
  const generateMipmaps = mipProfile?.generateMipmaps ?? texture.generateMipmaps;
  if (integerType && generateMipmaps) fail('UNSUPPORTED', 'Integer targets require authored mip levels');
  const maximumMipLevels = 1 + Math.floor(Math.log2(Math.max(width, height, volume ? layers : 1)));
  if (texture.mipmaps !== undefined && !Array.isArray(texture.mipmaps)) fail('UNSUPPORTED', 'Expected source mipmap descriptors');
  const authoredMipLevelCount = mipProfile?.authoredMipLevelCount ?? texture.mipmaps?.length ?? 0;
  // WebGLTextures.setupRenderTarget counts the base image in an authored cube
  // chain. A partial chain with mip filtering is incomplete in GL, unlike a
  // WebGPU sampler clamped to the available levels. Admit complete chains only.
  if (webglCube && authoredMipLevelCount && authoredMipLevelCount !== maximumMipLevels)
    fail('UNSUPPORTED', 'Authored WebGL cube targets require a complete mip chain including the base level');
  // Source Textures.getMipLevels uses authored descriptors before automatic
  // generation. Uncompressed CubeTexture descriptors omit the base level:
  // Textures.updateTexture adds it separately. Render-target descriptors
  // allocate levels, not uploaded pixels.
  // GL can allocate the pyramid later with generateMipmap. Reserve it now in
  // immutable WebGPU storage, but filter only at the captured source render
  // boundary (CubeCamera restores generation immediately before face 5).
  const mipLevelCount = mipProfile?.mipLevelCount ?? (webglCube ? maximumMipLevels
    : cube && cubeMipLevelCount !== null ? cubeMipLevelCount
    : texture.mipmaps?.length ? texture.mipmaps.length + (cube ? 1 : 0)
      : generateMipmaps ? maximumMipLevels : 1);
  if (!positive(mipLevelCount) || mipLevelCount > maximumMipLevels)
    fail('SIZE', 'Invalid target mip count for its extent');
  const texelBytes = format === 'r8unorm' ? 1 : format === 'r16float' ? 2 : format.startsWith('rgba32') ? 16
    : format === 'rgba16float' || format.startsWith('rg32') ? 8 : 4;
  const baseColorBytes = width * height * texelBytes * layers;
  let colorBytes = baseColorBytes, w = width, h = height, d = layers;
  for (let level = 1; level < mipLevelCount; level++) {
    w = Math.max(1, Math.floor(w / 2)); h = Math.max(1, Math.floor(h / 2));
    if (volume) d = Math.max(1, Math.floor(d / 2));
    colorBytes += w * h * texelBytes * d;
  }
  const depthOwner = depthTexture?.renderTarget ?? null, sharedDepth = depthOwner !== null && depthOwner !== target;
  const bytes = colorBytes + (sampleCount === 4 ? baseColorBytes * 4 : 0) +
    (depthFormat && !sharedDepth ? width * height * (depthFormat.endsWith('-stencil8') ? 8 : depthFormat === 'depth16unorm' ? 2 : 4) * sampleCount * depthLayers : 0);
  if (!Number.isSafeInteger(bytes)) fail('BUDGET', 'Target byte size exceeds safe integer precision');
  return {width, height, format, depthFormat, depthAttachment, sampleCount, bytes, colorBytes, baseColorBytes, texture, generateMipmaps,
    mipLevelCount, authoredMipLevelCount, maximumMipLevels, float32, integerType,
    ...(webglArray || webglVolume ? {webglSampler: captureWebGLTargetSampler(texture, webglVolume)} : {}),
    cube, webglCube, webglArray, webglVolume, array, volume, depth: volume ? layers : 1, layers, depthArray, depthLayers, viewDimension: cube ? 'cube' : volume ? '3d' : array ? '2d-array' : '2d',
    depthViewDimension: cube ? 'cube' : depthArray ? '2d-array' : '2d',
    sourceVersion: texture.version,
    depthTexture, depthOwner, sharedDepth, depthSampleBytes: depthTexture ? width * height * 4 * depthLayers : 0,
    key: `${cube ? `cube/${webglCube}/${width}/${height}/` : volume ? `volume/${webglVolume}/${layers}/${depthArray}/` : array ? `array/${webglArray}/${layers}/${depthArray}/` : ''}${target.width}/${target.height}/${format}/${depthFormat}/${depthAttachment}/${sampleCount}/${cube ? texture.version : generateMipmaps}/${mipLevelCount}${webglCube ? `/authored/${authoredMipLevelCount}` : ''}`};
}

export function inspectThreeRenderTarget(target, three, {cubeMipLevelCount = null, colorMipProfiles = null} = {}) {
  if (!three?.RenderTarget || !(target instanceof three.RenderTarget)) fail('SOURCE', 'Expected a source RenderTarget');
  if (!Array.isArray(target.textures) || !target.textures.length || target.textures.length > 8 ||
      new Set(target.textures).size !== target.textures.length || target.texture !== target.textures[0])
    fail('SOURCE', 'Expected one to eight distinct target-owned color attachments');
  const colors = target.textures.map((texture, index) => inspectThreeRenderTargetColor(target, three, texture,
    cubeMipLevelCount, colorMipProfiles?.[index])), primary = colors[0];
  if (colors.length > 1 && primary.cube)
    fail('UNSUPPORTED', 'Cube targets require a single color attachment');
  const bytes = primary.bytes + colors.slice(1).reduce((sum, color) => sum + color.colorBytes +
    (color.sampleCount === 4 ? color.baseColorBytes * 4 : 0), 0);
  if (!Number.isSafeInteger(bytes)) fail('BUDGET', 'Target byte size exceeds safe integer precision');
  return {...primary, colors, textures: colors.map(color => color.texture), colorFormats: colors.map(color => color.format), bytes,
    colorMipLevelCounts: colors.map(color => color.mipLevelCount), colorGenerateMipmaps: colors.map(color => color.generateMipmaps),
    key: colors.map(color => color.key).join('|')};
}

/** Validate a destination before any GPU allocation or queued submission. */
export function inspectThreeRenderTargetMip(shape, mipLevel = 0, face = 0, {webgl = false, generateMipmaps = shape.generateMipmaps} = {}) {
  if (!Number.isSafeInteger(mipLevel) || mipLevel < 0 ||
      (shape.colorMipLevelCounts ?? [shape.mipLevelCount]).some(count => mipLevel >= count))
    fail('MIP', 'Render mip level is not allocated');
  if ((webgl || shape.webglCube) && mipLevel > 0) {
    // r186 WebGL uses a depthless scratch framebuffer for higher levels. The
    // current dispatcher retains one attachment profile per target, so admit
    // color-only targets. Source WebGL array storage needs both the target and
    // DataArrayTexture flags; native RenderTarget depth alone is insufficient.
    // Array layers remain unchanged at every mip. Keep
    // volumes excluded: source setupFrameBufferTexture does not shrink their
    // authored Z extents, making a multi-slice volume chain incomplete in GL.
    if (shape.volume || shape.array && !shape.webglArray || shape.cube && !shape.webglCube ||
        shape.authoredMipLevelCount !== shape.maximumMipLevels ||
        shape.depthAttachment || shape.depthTexture !== null || shape.sampleCount !== 1)
      fail('MIP', 'WebGL mip rendering requires a complete authored 2D, array or cube chain, single-sample color and no depth/stencil attachment');
    // The facade supplies this flag at capture time. Owner submission must not
    // reread a later source flag: CubeCamera and subsequent base draws can
    // change it while an already accepted higher-mip request is queued.
    if (webgl && generateMipmaps !== false)
      fail('MIP', 'WebGL mip rendering requires automatic mip generation disabled at capture');
  }
  const slices = shape.volume ? Math.max(1, Math.floor(shape.layers / 2 ** mipLevel)) : shape.layers ?? (shape.cube ? 6 : 1);
  if (!Number.isSafeInteger(face) || face < 0 || face >= slices)
    fail('FACE', shape.cube ? 'Cube face must be an integer from zero to five' : shape.volume ? 'Volume slice is out of range for this mip'
      : shape.array ? 'Array layer is out of range' : '2D targets require face zero');
  if (mipLevel > 0 && (shape.sampleCount !== 1 || shape.depthTexture !== null))
    fail('MIP', 'Nonzero target mips require single-sample color and private depth');
  return {width: Math.max(1, Math.floor(shape.width / 2 ** mipLevel)),
    height: Math.max(1, Math.floor(shape.height / 2 ** mipLevel))};
}

function samplerDescriptor(texture, three, mipLevelCount) {
  const filters = new Map([[three.NearestFilter, 'nearest'], [three.LinearFilter, 'linear']]);
  const minification = new Map([
    [three.NearestFilter, ['nearest', 'linear', false]],
    [three.LinearFilter, ['linear', 'linear', false]],
    [three.NearestMipmapNearestFilter, ['nearest', 'nearest', true]],
    [three.NearestMipmapLinearFilter, ['nearest', 'linear', true]],
    [three.LinearMipmapNearestFilter, ['linear', 'nearest', true]],
    [three.LinearMipmapLinearFilter, ['linear', 'linear', true]],
  ]);
  const modes = new Map([[three.ClampToEdgeWrapping, 'clamp-to-edge'], [three.RepeatWrapping, 'repeat'], [three.MirroredRepeatWrapping, 'mirror-repeat']]);
  const magFilter = filters.get(texture.magFilter), min = minification.get(texture.minFilter);
  const addressModeU = modes.get(texture.wrapS), addressModeV = modes.get(texture.wrapT);
  const volume = texture.isData3DTexture === true || texture.is3DTexture === true, addressModeW = volume ? modes.get(texture.wrapR) : undefined;
  if (!magFilter || !min || !addressModeU || !addressModeV || volume && !addressModeW || !Number.isFinite(texture.anisotropy) || texture.anisotropy < 1)
    fail('SAMPLER', 'Expected source target filters, wrap modes and positive anisotropy');
  const [minFilter, mipmapFilter, mips] = min;
  return {magFilter, minFilter, mipmapFilter, addressModeU, addressModeV, ...(volume ? {addressModeW} : {}),
    lodMaxClamp: mips ? mipLevelCount - 1 : 0,
    maxAnisotropy: magFilter === 'linear' && minFilter === 'linear' && mipmapFilter === 'linear'
      ? Math.min(16, Math.floor(texture.anisotropy)) : 1};
}

const FLIP_WGSL = flipY => `
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let size = textureDimensions(source);
  return textureLoad(source, vec2i(i32(p.x), ${flipY ? 'i32(size.y) - 1 - i32(p.y)' : 'i32(p.y)'}), 0);
}`;

const VOLUME_FLIP_WGSL = flipY => `
@group(0) @binding(0) var source: texture_3d<f32>;
struct SliceVertex {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) slice: u32,
};
@vertex fn vs(@builtin(vertex_index) i: u32, @builtin(instance_index) slice: u32) -> SliceVertex {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return SliceVertex(vec4f(p[i], 0, 1), slice);
}
@fragment fn fs(v: SliceVertex) -> @location(0) vec4f {
  let size = textureDimensions(source);
  return textureLoad(source, vec3i(i32(v.position.x), ${flipY ? 'i32(size.y) - 1 - i32(v.position.y)' : 'i32(v.position.y)'}, i32(v.slice)), 0);
}`;

// A scalar R32F output preserves textureLoad's f32 depth value. Sampling the
// replica uses the program renderer's existing sampler2D float binding route.
// Its red channel is depth; G/B are zero and A is one. Never use RGBA8 or half
// floats here: small depth errors become large reconstructed-distance errors.
const DEPTH_FLIP_WGSL = (flipY, comparison = false) => `
@group(0) @binding(0) var source: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> ${comparison ? '@builtin(frag_depth)' : '@location(0)'} f32 {
  let size = textureDimensions(source);
  return textureLoad(source, vec2i(i32(p.x), ${flipY ? 'i32(size.y) - 1 - i32(p.y)' : 'i32(p.y)'}), 0);
}`;

/** getDevice is lazy so selecting a target before renderer.init() performs no GPU work. */
export function createThreeRenderTargets(three, {getDevice, maxTargets = 256, maxBytes = 512 * 1024 * 1024, flipY = true} = {}) {
  if (typeof getDevice !== 'function' || !positive(maxTargets) || !positive(maxBytes) || typeof flipY !== 'boolean')
    fail('OPTIONS', 'Invalid target owner options');
  const entries = new Map(), pipelines = new Map(), preserved = new Set(), retiring = new Set(), nativeResizes = new WeakMap();
  const historyGroups = new Map(), historyEntries = new Set(), nativePasses = new WeakMap();
  // r186 cube and WebGL array target constructors replace their color texture
  // without assigning its renderTarget pointer. Register the exact identities
  // privately; native sampling must not patch that public ownership pointer or
  // mistake uninitialized attachment descriptors for CPU upload sources.
  const ownedColorTargets = new WeakMap();
  const allEntries = () => new Set([...entries.values(), ...historyEntries, ...preserved]);
  let disposed = false, allocatedBytes = 0, terminal = null;
  const live = () => { if (disposed) fail('DISPOSED', 'Render target owner is disposed'); if (terminal) throw terminal; };
  function release(entry) {
    if (entry.busy || entry.programReferences || entry.depthReferences || entry.historyGroup?.busy || entry.historyGroup?.frame || entry.released) return;
    entry.released = true;
    try {
      try { entry._dispatcher?.dispose(); }
      finally { const program = entry._programSupport; entry._programSupport = null; program?.dispose(); }
    }
    finally {
      const group = entry.historyGroup;
      if (group) {
        group.entries.delete(entry);
        if (!group.entries.size) {
          group.storage?.dispose(); allocatedBytes -= group.charged; group.charged = 0;
          for (const texture of group.watched) texture.removeEventListener('dispose', group.onDispose);
          if (historyGroups.get(group.target) === group) historyGroups.delete(group.target);
        }
      } else entry.storage?.dispose();
      entry.sample?.destroy(); entry.depthSample?.destroy(); entry.depthComparison?.sample?.destroy();
      for (const sample of entry.colorSamples.values()) sample.sample?.destroy();
      entry.colorSamples.clear();
      allocatedBytes -= entry.charged; entry.charged = 0;
      entry.storage = entry.sample = entry.view = entry.group = entry.flipSteps = null;
      entry.depthSample = entry.depthView = entry.depthGroup = entry.depthSampler = entry.depthComparison = null;
      entry._wanted.clear();
      entry.releaseDepthOwner?.(); entry.releaseDepthOwner = null;
    }
  }
  function retire(entry) {
    if (!entry.alive) return;
    entry.alive = false;
    entry.lifetime.abort(new ThreeRenderTargetError('STALE', 'Source render target disposed or replaced'));
    entry.target.removeEventListener('dispose', entry.onDispose);
    for (const texture of entry.shape.textures) texture.removeEventListener('dispose', entry.onDispose);
    entry.shape.depthTexture?.removeEventListener('dispose', entry.onDispose);
    entry.shape.depthOwner?.removeEventListener('dispose', entry.onDepthOwnerDispose);
    if (entries.get(entry.target) === entry) entries.delete(entry.target);
    preserved.delete(entry); historyEntries.delete(entry);
    // Color disposal and direct owner configuration changes also invalidate
    // its depth, even when no public RenderTarget dispose event was emitted.
    for (const borrower of allEntries()) if (borrower.depthEntry === entry) retire(borrower);
    release(entry);
  }
  function collect(entry) {
    if (!entry.alive || !preserved.has(entry) || entry.references || entry.busy || entry.historyGroup?.busy || entry.historyGroup?.frame || entry.retirement) return;
    if (!entry.storage && !entry._dispatcher) { retire(entry); return; }
    // A sampled generation can have validation/fence observers after its last
    // submitted reader releases its lease. Let those observers finish before
    // destroying storage or aborting the generation's scene dispatcher.
    const task = Promise.all([entry.validation, entry.storage?.whenIdle(), entry._dispatcher?.whenIdle()]).then(() => {
      if (!entry.references) retire(entry);
    }, error => {
      if (entry.alive) { terminal ??= error; entry.error = error; retire(entry); }
    }).finally(() => { retiring.delete(task); entry.retirement = null; });
    entry.retirement = task; retiring.add(task);
  }
  const sameColors = (a, b) => a.textures.length === b.textures.length && a.textures.every((texture, index) =>
    texture === b.textures[index] && a.colorFormats[index] === b.colorFormats[index]);
  const sameShape = (a, b) => a.key === b.key && sameColors(a, b) && a.depthTexture === b.depthTexture && a.depthOwner === b.depthOwner;
  function historyShape(group, shape, base = group.shape) {
    return shape.key === base.key && shape.depthTexture === base.depthTexture && shape.depthOwner === base.depthOwner &&
      shape.textures.length === group.pairs.length && shape.textures.every((texture, index) => group.pairs[index].includes(texture)) &&
      (!group.mipInitialized || [...group.mipProfiles].every(([texture, profile]) => texture.version === profile.sourceVersion));
  }
  function initializeHistory(group) {
    if (group.mipInitialized) return;
    for (const [texture, profile] of group.mipProfiles)
      group.mipProfiles.set(texture, {...profile, sourceVersion: texture.version});
    group.mipInitialized = true;
  }
  function retireHistory(group) { for (const entry of [...group.entries]) retire(entry); }
  function sourceDisposed(entry, event) {
    const resize = nativeResizes.get(entry.target);
    const matchesSize = resize && (resize.dynamic
      ? extent(entry.target.width) && extent(entry.target.height) &&
        (entry.target.width !== resize.width || entry.target.height !== resize.height)
      : entry.target.width === resize.width && entry.target.height === resize.height);
    if (resize && event?.target === entry.target && (!resize.event || resize.event === event) &&
        matchesSize && entry.target.depth === resize.depth) {
      const shape = resize.shape ??= sourceShape(entry.target, entry);
      if ((entry.historyGroup ? shape.textures.every((texture, index) => entry.historyGroup.pairs[index]?.includes(texture)) : sameColors(shape, entry.shape)) && shape.depthTexture === entry.shape.depthTexture &&
          shape.depthFormat === entry.shape.depthFormat &&
          shape.sampleCount === entry.shape.sampleCount && (entry.historyGroup || shape.generateMipmaps === entry.shape.generateMipmaps)) {
        resize.event = event;
        // Capture leases can end before submitted GPU validation/fence
        // observers. Keep their generation alive until collect() has awaited
        // both owners, even when no queued consumer still retains a lease.
        // Source texture identities stay unchanged across setSize(). Only
        // this recognized automatic resize can detach a usable generation.
        const group = entry.historyGroup;
        if (group) {
          group.sourceShape = shape;
          if (historyGroups.get(entry.target) === group) historyGroups.delete(entry.target);
          for (const member of group.entries) {
            member.sourceShape = shape; preserved.add(member); historyEntries.delete(member);
            if (entries.get(entry.target) === member) entries.delete(entry.target);
          }
          for (const member of group.entries) collect(member);
        } else {
          entry.sourceShape = shape; preserved.add(entry);
          if (entries.get(entry.target) === entry) entries.delete(entry.target);
          collect(entry);
        }
        return;
      }
    }
    if (entry.historyGroup) retireHistory(entry.historyGroup); else retire(entry);
  }
  function retainGeneration(entry) {
    assertCurrent(entry); entry.references++;
    if (entry.shape.cube) entry.cubeInitialized = true;
    if (!flipY || entry.shape.textures.length > 1) entry.mrtInitialized = true;
    if (entry.historyGroup) initializeHistory(entry.historyGroup);
    let held = true;
    return () => {
      if (!held) return;
      held = false; entry.references--; collect(entry);
    };
  }
  function sourceShape(target, entry = entries.get(target)) {
    // CubeCamera disables generation for faces 0-4, then restores it for face 5.
    // Pinned Textures.updateTexture preserves the allocation until a version
    // update. Selection/layout queries alone have not initialized that image.
    const frozen = entry?.alive && entry.shape.cube && !entry.shape.webglCube && (entry.cubeInitialized || entry.storage) &&
      entry.shape.texture === target.texture && entry.shape.sourceVersion === target.texture.version;
    const group = entry?.historyGroup ?? historyGroups.get(target);
    const capturedMips = !nativePasses.has(target) && (target.textures?.length > 1 || !flipY &&
      !target.isCubeRenderTarget && !target.isWebGLCubeRenderTarget && !target.isRenderTarget3D &&
      !target.isWebGL3DRenderTarget && target.depth === 1 && !target.texture?.isDataArrayTexture && !target.texture?.isArrayTexture);
    const frozenMrt = capturedMips && entry?.alive && (entry.mrtInitialized || entry.storage) &&
      entry.shape.width === (target.width || 1) && entry.shape.height === (target.height || 1) &&
      entry.shape.sampleCount === (target.samples === 4 ? 4 : 1) && entry.shape.textures.length === target.textures.length &&
      entry.shape.colors.every((color, index) => color.texture === target.textures[index] &&
        (flipY || color.sourceVersion === color.texture.version));
    const base = group?.sourceShape ?? group?.shape;
    const frozenHistory = group?.mipInitialized && base.width === (target.width || 1) && base.height === (target.height || 1) &&
      base.sampleCount === (target.samples === 4 ? 4 : 1) && base.textures.length === target.textures.length &&
      target.textures.every(texture => group.mipProfiles.get(texture)?.sourceVersion === texture.version);
    const shape = inspectThreeRenderTarget(target, three, {cubeMipLevelCount: frozen ? entry.shape.mipLevelCount : null,
      colorMipProfiles: frozenHistory ? base.colors : frozenMrt ? entry.shape.colors : null});
    if (flipY && shape.textures.length > 1) for (const color of shape.colors)
      color.webglSampler ??= captureWebGLTargetSampler(color.texture, color.webglVolume);
    // Native allocation follows the texture version; WebGL MRT allocation
    // follows framebuffer setup. The facade captures per-render generation
    // flags separately so editing a flag cannot replace rendered contents.
    if (capturedMips && !flipY && shape.colors.some(color => color.mipLevelCount > 1))
      shape.key += `/mipVersions/${shape.colors.map(color => color.sourceVersion).join(',')}`;
    return shape;
  }
  function capture(target, {initialize = false} = {}) {
    live();
    const shape = sourceShape(target);
    if (flipY && shape.colors.some(color => color.integerType !== null))
      fail('UNSUPPORTED', 'Integer targets currently require the native WebGPU renderer surface');
    if (flipY && shape.textures.length > 1) {
      // Pinned WebGL setup allocates MRT framebuffer attachments at level zero;
      // authored framebuffer arrays do not use that path. Generated pyramids
      // are per-color and run at the captured render/copy boundary instead.
      if (shape.colors.some(color => color.authoredMipLevelCount > 0))
        fail('UNSUPPORTED', 'WebGL MRT requires generated mip pyramids, not authored framebuffer arrays');
      if (shape.colors.some(color => color.texture.generateMipmaps && color.mipLevelCount < color.maximumMipLevels))
        fail('MIP', 'Enable WebGL MRT mip generation before initializing that attachment, or recreate the target');
    }
    if (shape.cube || shape.array || shape.volume) for (const texture of shape.textures) {
      const owner = ownedColorTargets.get(texture);
      if (owner && owner !== target && owner.textures.includes(texture))
        fail('SOURCE', 'A color texture cannot belong to two registered render targets');
      ownedColorTargets.set(texture, target);
    }
    const device = getDevice();
    if (device && (shape.textures.length > 1 || shape.colorFormats.includes('rg11b10ufloat')))
      gpuTargets.inspectGpuRenderTargetColors(device, shape.colorFormats);
    // Source registration outlives an allocation group: PassNode.setSize()
    // disposes its old attachments before swapping and rendering the next pair.
    if (!historyGroups.has(target) && nativePasses.has(target)) registerNativePass(nativePasses.get(target));
    let group = historyGroups.get(target);
    if (group && !historyShape(group, shape)) {
      // PassNode.setup inherits renderer samples during shader construction,
      // after layout queries may have created allocation-free metadata. Such
      // observations have no queued owner or image to preserve. Accepted uses
      // and allocated histories still invalidate on unrecognized shape changes.
      const cold = !group.storage && !group.busy && !group.frame && [...group.entries].every(entry =>
        !entry.references && !entry._dispatcher && !entry.busy);
      const node = group.node;
      retireHistory(group);
      if (!cold) fail('STALE', 'Pass history attachments changed outside their registered source lifecycle');
      registerNativePass(node); group = historyGroups.get(target);
    }
    if (group) {
      const entry = historyEntry(group, shape); entries.set(target, entry);
      if (initialize) initializeHistory(group);
      return entry;
    }
    let entry = entries.get(target);
    if (entry && (!sameShape(entry.shape, shape) || entry.depthEntry && !entry.depthEntry.alive)) { retire(entry); entry = null; }
    if (entry) {
      if (initialize && shape.cube) entry.cubeInitialized = true;
      if (initialize && (!flipY || shape.textures.length > 1)) entry.mrtInitialized = true;
      return entry;
    }
    entry = createEntry(target, shape);
    entry.cubeInitialized = initialize && shape.cube;
    entry.mrtInitialized = initialize && (!flipY || shape.textures.length > 1);
    entries.set(target, entry);
    return entry;
  }
  function createEntry(target, shape) {
    const depthEntry = shape.sharedDepth ? capture(shape.depthOwner) : null;
    if (allEntries().size >= maxTargets) fail('BUDGET', 'Render target count exceeds maxTargets');
    const entry = {target, shape, alive: true, released: false, busy: false, charged: 0, storage: null, sample: null,
      references: 0, sourceShape: null, retirement: null, cubeInitialized: false,
      cubeMipReady: !shape.webglCube || shape.mipLevelCount === 1 || shape.generateMipmaps || shape.authoredMipLevelCount === shape.mipLevelCount,
      dirty: true, copies: 0, colorSamples: new Map(), depthDirty: true, depthCopies: 0, depthSample: null, error: null, lifetime: new AbortController(), validation: Promise.resolve(), _wanted: new Map(), _dispatcher: null};
    entry.onDispose = event => sourceDisposed(entry, event);
    target.addEventListener('dispose', entry.onDispose);
    for (const texture of shape.textures) texture.addEventListener('dispose', entry.onDispose);
    shape.depthTexture?.addEventListener('dispose', entry.onDispose);
    if (shape.sharedDepth) {
      entry.depthEntry = depthEntry;
      const finish = retainGeneration(depthEntry);
      depthEntry.depthReferences = (depthEntry.depthReferences ?? 0) + 1;
      entry.releaseDepthOwner = () => {
        depthEntry.depthReferences--; finish();
        if (!depthEntry.alive) release(depthEntry);
      };
      entry.onDepthOwnerDispose = () => {
        // Automatic effect/pass resizes preserve accepted older generations.
        // The borrowing color keeps the corresponding older depth generation.
        if (preserved.has(entry.depthEntry)) {
          entry.sourceShape = sourceShape(target, entry); preserved.add(entry);
          if (entries.get(target) === entry) entries.delete(target);
          collect(entry);
        } else retire(entry);
      };
      shape.depthOwner.addEventListener('dispose', entry.onDepthOwnerDispose);
    }
    return entry;
  }
  function historyEntry(group, shape) {
    const slots = Object.freeze(shape.textures.map((texture, index) => group.pairs[index].indexOf(texture)));
    const key = slots.join(',');
    let entry = group.variants.get(key);
    if (entry?.alive) return entry;
    entry = createEntry(group.target, shape);
    entry.historyGroup = group; entry.colorSlots = slots;
    group.variants.set(key, entry); group.entries.add(entry); historyEntries.add(entry);
    return entry;
  }
  function registerNativePass(node) {
    live();
    const target = node.renderTarget, existing = entries.get(target);
    // An already rendered image keeps its initialized mip allocation when a
    // previous-frame image is added later. History versions are tracked by
    // physical source texture, independently of the current attachment order.
    const inspected = sourceShape(target, existing);
    const shape = {...inspected, key: inspected.colors.map(color => color.key).join('|')};
    const names = Object.keys(node._previousTextures ?? {});
    if (!names.length) return;
    if (shape.cube || shape.volume)
      fail('UNSUPPORTED', 'Native pass histories require ordinary 2D or 2D-array color attachments');
    nativePasses.set(target, node);
    let group = historyGroups.get(target);
    if (group && (!historyShape(group, shape) || group.node !== node)) {
      retireHistory(group); group = null;
    }
    if (!group) {
      if (existing && !(sameColors(existing.shape, shape) && existing.shape.depthTexture === shape.depthTexture &&
          existing.shape.depthOwner === shape.depthOwner && existing.shape.colors.every((color, index) => color.key === shape.colors[index].key))) retire(existing);
      group = {node, target, shape, pairs: shape.textures.map(texture => [texture]), variants: new Map(),
        entries: new Set(), watched: new Set(), storage: null, charged: 0, mipDepthBytes: 0, sourceShape: null, busy: false, frame: null,
        mipProfiles: new Map(shape.colors.map(color => [color.texture, color])),
        mipInitialized: existing?.alive === true && !!(existing.storage || existing.mrtInitialized)};
      group.onDispose = () => retireHistory(group);
      historyGroups.set(target, group);
      if (existing?.alive) {
        if (existing.sample || existing.colorSamples.size || existing.depthSample || existing.depthComparison) fail('UNSUPPORTED', 'Pass histories require native attachment sampling');
        existing.historyGroup = group; existing.colorSlots = Object.freeze(shape.textures.map(() => 0));
        group.entries.add(existing); historyEntries.add(existing); group.variants.set(existing.colorSlots.join(','), existing);
        group.storage = existing.storage; group.charged = existing.charged; existing.charged = 0; existing.storage = null;
        group.mipDepthBytes = Math.max(0, (group.storage?.allocatedBytes ?? 0) - existing.shape.bytes);
      }
    }
    for (const name of names) {
      const current = node._textures[name], previous = node._previousTextures[name], index = shape.textures.indexOf(current);
      if (index < 0 || previous === current || previous?.isDepthTexture)
        fail('SOURCE', 'Pass histories require distinct current and previous color textures');
      const profile = group.mipInitialized && group.mipProfiles.get(previous)?.sourceVersion === previous.version
        ? shape.colors[index] : null;
      const color = inspectThreeRenderTargetColor(target, three, previous, null, profile);
      if (color.key !== shape.colors[index].key)
        fail('SOURCE', 'Current and previous pass color formats must match');
      const pair = group.pairs[index];
      if (pair.length === 1) pair.push(previous);
      else if (!pair.includes(previous) || !pair.includes(current))
        fail('STALE', 'Pass history texture identity changed outside its source swap');
      if (!group.mipInitialized || !group.mipProfiles.has(previous)) group.mipProfiles.set(previous, color);
      for (const texture of pair) if (!group.watched.has(texture)) {
        group.watched.add(texture); texture.addEventListener('dispose', group.onDispose);
      }
    }
    capture(target);
  }
  function assertCurrent(entry) {
    live();
    if (!entry?.alive || entries.get(entry.target) !== entry && !preserved.has(entry) && !historyEntries.has(entry))
      fail('STALE', 'Target was disposed or replaced before its queued use');
    const shape = sourceShape(entry.target, entry);
    if (entry.historyGroup ? !historyShape(entry.historyGroup, shape, entry.historyGroup.sourceShape ?? entry.historyGroup.shape)
      : !sameShape(shape, preserved.has(entry) ? entry.sourceShape : entry.shape))
      fail('STALE', 'Target configuration changed before its queued use');
    if (entry.error) throw entry.error;
    if (entry.storage?.failed) fail('GPU', 'Native target storage failed');
    if (entry.depthEntry) assertCurrent(entry.depthEntry);
    return entry;
  }
  function budget(bytes) {
    if (bytes > maxBytes - allocatedBytes) fail('BUDGET', 'Render targets exceed their shared byte budget');
  }
  function allocationBytes(entry) {
    const group = entry.historyGroup;
    return group ? entry.shape.bytes + group.pairs.reduce((sum, pair, index) => sum +
      (pair.length > 1 ? entry.shape.colors[index].colorBytes +
        (entry.shape.sampleCount === 4 ? entry.shape.colors[index].baseColorBytes * 4 : 0) : 0), 0) + group.mipDepthBytes - group.charged
      : entry.storage ? 0 : entry.shape.bytes;
  }
  function prepareDepthTexture(entry) {
    const {depthTexture, cube, width, height, depthLayers, depthViewDimension} = entry.shape;
    // The pinned Textures owner initializes an attached depth image lazily.
    // Public resize preserves this source identity, but its old layer count
    // must not select array depth again after a shrink to ordinary 2D. Earlier
    // retained effect/shadow generations cannot rewrite the current image.
    if (!depthTexture || cube || entry.shape.sharedDepth || preserved.has(entry)) return;
    depthTexture.isArrayTexture = depthViewDimension === '2d-array';
    Object.assign(depthTexture.image, {width, height, depth: depthLayers});
  }
  function captureDepthSampler(texture) {
    const parameters = Object.freeze({...captureWebGLTargetSampler(texture), compareFunction: texture.compareFunction});
    // Exact WebGLTextures.getTextureCacheKey fields. Even unpack controls with
    // no depth texel payload can select a new GL image after needsUpdate.
    const key = JSON.stringify([texture.wrapS, texture.wrapT, texture.wrapR || 0,
      texture.magFilter, texture.minFilter, texture.anisotropy, texture.internalFormat,
      texture.format, texture.type, texture.generateMipmaps, texture.premultiplyAlpha,
      texture.flipY, texture.unpackAlignment, texture.colorSpace]);
    return {parameters, key, source: texture.source,
      version: texture.version, sourceVersion: texture.source.version};
  }
  function depthSamplerState(entry, texture) {
    const previous = entry.depthSetupSampler;
    if (!previous) return captureDepthSampler(texture).parameters;
    // r186 setTexture2D uploads depth parameters only after needsUpdate. A
    // compare-only update reuses the GL image; clearing compareFunction never
    // issues TEXTURE_COMPARE_MODE=NONE, so the previously installed mode stays.
    if (texture.version === previous.version || texture.version === 0) return previous.parameters;
    const next = captureDepthSampler(texture);
    if (next.key !== previous.key || next.source !== previous.source)
      fail('SAMPLER', 'Changing initialized depth storage parameters requires recreating the source target');
    next.parameters = next.sourceVersion !== previous.sourceVersion
      ? Object.freeze({...previous.parameters, compareFunction: next.parameters.compareFunction ?? previous.parameters.compareFunction})
      : previous.parameters;
    entry.depthSetupSampler = next;
    return next.parameters;
  }
  function ensure(entry) {
    assertCurrent(entry);
    const depthEntry = entry.depthEntry;
    if (depthEntry) {
      // The source does not initialize an owning target through its borrower.
      // Keep this boundary explicit instead of silently changing its image
      // dimensions or allocating a private replacement for the shared depth.
      if (!depthEntry.storage) fail('UNINITIALIZED', 'Initialize the owning target before rendering with its shared depth');
      if (entry.shape.depthAttachment && ['width', 'height', 'sampleCount', 'depthFormat', 'depthLayers'].some(key =>
        entry.shape[key] !== depthEntry.shape[key]))
        fail('DEPTH', 'Shared depth attachment extent, format and samples must match its color target');
    }
    const group = entry.historyGroup;
    const additional = allocationBytes(entry), needed = group ? group.charged + additional : entry.shape.bytes;
    if (entry.storage && (!group || group.charged === needed)) return entry;
    budget(additional);
    const device = getDevice();
    if (!device) fail('DEVICE', 'Initialize the renderer before acquiring target storage');
    prepareDepthTexture(entry);
    if (flipY && entry.shape.depthTexture && !depthEntry)
      entry.depthSetupSampler ??= captureDepthSampler(entry.shape.depthTexture);
    const {width, height, format, colorFormats, colorMipLevelCounts, depthFormat, sampleCount, generateMipmaps, mipLevelCount, cube, array, volume, layers, depthArray, depthLayers} = entry.shape;
    const capturedMips = colorFormats.length > 1 || !flipY && !cube && !array && !volume || !!group;
    const mipOptions = capturedMips ? {colorMipLevelCounts, colorGenerateMipmaps: colorFormats.map(() => false)} : {};
    const depthOptions = depthEntry ? entry.shape.depthAttachment
      ? {depthSource: depthEntry.storage} : {depthFormat: null, sampleDepth: false} : {};
    if (group) {
      group.storage ??= gpuTargets.createGpuRenderTarget(device, {width, height, format, colorFormats, depthFormat, sampleCount, maxBytes, generateMipmaps: false, mipLevelCount, ...mipOptions,
        sampleDepth: entry.shape.depthTexture !== null, copyDestination: true, cube, ...depthOptions, ...(array ? {arrayLayers: layers, depthArray} : volume ? {volumeDepth: layers, depthArray} : {})});
      if (group.charged !== needed)
        group.storage.reserveColorHistory(group.pairs.flatMap((pair, index) => pair.length > 1 ? [index] : []));
      allocatedBytes += needed - group.charged; group.charged = needed;
      entry.storage = group.storage.selectColorSlots(entry.colorSlots);
      initializeHistory(group);
    } else {
      entry.storage = gpuTargets.createGpuRenderTarget(device, {width, height, format, colorFormats, depthFormat, sampleCount, maxBytes, generateMipmaps: cube || capturedMips ? false : generateMipmaps, mipLevelCount, ...mipOptions,
        sampleDepth: entry.shape.depthTexture !== null, copyDestination: true, cube, ...depthOptions, ...(array ? {arrayLayers: layers, depthArray} : volume ? {volumeDepth: layers, depthArray} : {})});
      entry.charged += entry.shape.bytes; allocatedBytes += entry.shape.bytes;
    }
    // The readback helper consumes this session-shaped object. It copies the
    // original attachment, not the UV-oriented sampling replica.
    entry.session = {device, get width() { return width; }, get height() { return height; },
      get disposed() { return !entry.alive; }, get failed() { return !!entry.error || !!entry.storage?.failed; },
      get diagnostics() { return {target: {format, colorFormats, mipLevelCount, colorMipLevelCounts, cube, array, volume, depth: volume ? layers : 1, layers, depthArray, depthLayers}}; },
      get validation() { assertCurrent(entry); return Promise.all([entry.validation, entry.storage.validation]); },
      withTexture(callback, options) {
        assertCurrent(entry);
        if (cube) fail('UNSUPPORTED', 'Cube target readbacks require an explicit face-aware transfer API');
        return entry.storage.withTexture(callback, options);
      }};
    return entry;
  }
  function checked(entry, operation) {
    const device = getDevice();
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let result, error;
    try { result = operation(device); } catch (cause) { error = cause; }
    const invalid = device.popErrorScope(), oom = device.popErrorScope();
    entry.validation = Promise.all([entry.validation, invalid, oom]).then(([, a, b]) => {
      if (error) throw error;
      if (a || b) throw new ThreeRenderTargetError('GPU', (a || b).message ?? 'Target sampling failed');
    });
    entry.validation.catch(cause => { terminal ??= cause; entry.error = cause; retire(entry); });
    if (error) { terminal ??= error; entry.error = error; retire(entry); throw error; }
    return result;
  }
  function colorSampler(entry, texture, samplerState = texture, {native = true} = {}) {
    const device = getDevice();
    const shape = entry.shape.colors[entry.shape.textures.indexOf(texture)];
    if (!shape) fail('STALE', 'Texture no longer belongs to its source target');
    const filterable = !shape.float32 || device.features?.has?.('float32-filterable') === true;
    const descriptor = samplerDescriptor(samplerState, three, shape.mipLevelCount);
    if (shape.integerType !== null) {
      // Pinned integer TextureNodes use textureLoad and have no sampler binding.
      // Preserve authored filter properties; the unused descriptor stays valid
      // for a non-filtering binding without inventing float sampling semantics.
      Object.assign(descriptor, {minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest',
        maxAnisotropy: 1, lodMaxClamp: shape.mipLevelCount - 1});
      return {descriptor, sampleType: shape.integerType};
    }
    // Pinned WebGL setup only applies anisotropy with mip-linear minification.
    // Layered targets retain that initialized sampler until recreation, even
    // when their source texture later changes its filtering or needsUpdate.
    if (!native && shape.webglSampler &&
        ![three.NearestMipmapLinearFilter, three.LinearMipmapLinearFilter].includes(samplerState.minFilter))
      descriptor.maxAnisotropy = 1;
    if (shape.cube && !shape.webglCube || native && (shape.volume || entry.historyGroup) ||
        !flipY && (entry.shape.textures.length > 1 || !shape.cube && !shape.array && !shape.volume)) {
      // Native source samplers retain explicit LOD access even for ordinary
      // LinearFilter/NearestFilter; only their mip interpolation is nearest.
      // WebGL cube replicas retain the source's base-only minification clamp.
      descriptor.lodMaxClamp = shape.mipLevelCount - 1;
      if ([three.NearestFilter, three.LinearFilter].includes(samplerState.minFilter)) {
        descriptor.mipmapFilter = 'nearest'; descriptor.maxAnisotropy = 1;
      }
    }
    if (!filterable) {
      if (samplerState.magFilter !== three.NearestFilter ||
          (samplerState.minFilter !== three.NearestFilter && samplerState.minFilter !== three.NearestMipmapNearestFilter))
        fail('SAMPLER', 'Linear float32 color sampling requires the device float32-filterable feature');
      // A non-filtering sampler must use nearest for all three filters, even
      // when the mip filter is inactive because lodMaxClamp is zero.
      descriptor.mipmapFilter = 'nearest'; descriptor.maxAnisotropy = 1;
    }
    return {descriptor, sampleType: filterable ? 'float' : 'unfilterable-float'};
  }
  function sampling(entry, texture) {
    assertCurrent(entry);
    const colorIndex = entry.shape.textures.indexOf(texture), shape = entry.shape.colors[colorIndex];
    if (!shape) fail('STALE', 'Texture no longer belongs to its source target');
    if (shape.integerType !== null)
      fail('UNSUPPORTED', 'Integer target sampling requires native typed texel-load bindings');
    if (shape.cube && !shape.webglCube || shape.array && !shape.webglArray || shape.volume && !shape.webglVolume)
      fail('UNSUPPORTED', 'Native layered target color sampling requires native texture bindings');
    if (shape.webglVolume && shape.mipLevelCount !== 1)
      fail('UNSUPPORTED', 'WebGL volume sampling requires a single allocated base level');
    if (entry.busy) fail('REENTRANT', 'Cannot sample color during a target submission');
    const device = getDevice();
    if (!device) fail('DEVICE', 'Initialize the renderer before sampling target color');
    // Source setTexture2D/2DArray/3D skip uploads for render-target textures, even
    // after needsUpdate. Their setup sampler survives until target recreation.
    const samplerState = shape.webglSampler ?? texture;
    const {descriptor, sampleType} = colorSampler(entry, texture, samplerState, {native: false});
    if ((shape.webglArray || shape.webglVolume || flipY && entry.shape.textures.length > 1) && shape.mipLevelCount !== shape.maximumMipLevels &&
        ![three.NearestFilter, three.LinearFilter].includes(samplerState.minFilter))
      fail('UNSUPPORTED', 'Mip-filtered WebGL layered or MRT targets require a complete source mip chain');
    if (shape.webglCube && descriptor.lodMaxClamp > 0 && !entry.cubeMipReady)
      fail('UNSUPPORTED', 'Mip-filtered WebGL cubes require a completed source mip generation');
    const samplerKey = JSON.stringify(descriptor);
    ensure(entry);
    // Each attachment owns its orientation replica and dirty bit. A selected
    // transfer refreshes only that image; untouched G-buffer channels stay warm.
    let sample = colorIndex === 0 ? entry : entry.colorSamples.get(colorIndex);
    if (!sample) { sample = {sample: null, dirty: true, copies: 0}; entry.colorSamples.set(colorIndex, sample); }
    entry.busy = true;
    try {
      if (!sample.sample) {
        budget(shape.colorBytes);
        checked(entry, device => {
          const {width, height, format, mipLevelCount, layers, cube, array, volume} = shape;
          const dimension = volume ? '3d' : '2d';
          sample.sample = device.createTexture({label: 'Source target UV sampling image',
            size: {width, height, depthOrArrayLayers: layers}, format, dimension, sampleCount: 1, mipLevelCount, usage: 4 | 16});
          entry.charged += shape.colorBytes; allocatedBytes += shape.colorBytes;
          sample.view = sample.sample.createView(cube ? {dimension: 'cube'} : array ? {dimension: '2d-array'} : volume ? {dimension} : {});
          const pipelineKey = `${format}/${dimension}`;
          let pipeline = pipelines.get(pipelineKey);
          if (!pipeline) {
            const module = device.createShaderModule({label: 'Source target UV orientation',
              code: (volume ? VOLUME_FLIP_WGSL : FLIP_WGSL)(flipY)});
            // textureLoad does not need filtering, but an automatic layout would
            // still infer sampleType:float and reject a feature-free float32 view.
            const layout = shape.float32 ? device.createPipelineLayout({bindGroupLayouts: [device.createBindGroupLayout({entries: [
              {binding: 0, visibility: 2, texture: {sampleType: 'unfilterable-float', viewDimension: dimension, multisampled: false}},
            ]})]}) : 'auto';
            pipeline = device.createRenderPipeline({label: 'Source target UV orientation', layout,
              vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format}]},
              primitive: {topology: 'triangle-list'}});
            pipelines.set(pipelineKey, pipeline);
          }
          sample.pipeline = pipeline;
          entry.storage.withTexture(source => {
            sample.flipSteps = [];
            for (let level = 0; level < mipLevelCount; level++) for (let face = 0; face < layers; face++) {
              const settings = {baseMipLevel: level, mipLevelCount: 1,
                ...(cube || array ? {dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1} : volume ? {dimension} : {})};
              const view = sample.sample.createView(settings);
              const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
                entries: [{binding: 0, resource: source.createView(settings)}]});
              sample.flipSteps.push({view, group, ...(volume ? {depthSlice: face} : {})});
            }
            sample.group = sample.flipSteps[0].group;
          }, {colorIndex});
        });
      }
      if (sample.samplerKey !== samplerKey) {
        sample.sampler = checked(entry, device => device.createSampler(descriptor));
        sample.samplerKey = samplerKey;
      }
      if (sample.dirty) {
        checked(entry, device => entry.storage.withTexture(() => {
          assertCurrent(entry);
          const encoder = device.createCommandEncoder({label: 'Source target UV orientation'});
          for (const {view, group, depthSlice} of sample.flipSteps) {
            const pass = encoder.beginRenderPass({colorAttachments: [{view, ...(depthSlice !== undefined ? {depthSlice} : {}),
              loadOp: 'clear', storeOp: 'store', clearValue: {r: 0, g: 0, b: 0, a: 0}}]});
            pass.setPipeline(sample.pipeline); pass.setBindGroup(0, group);
            if (depthSlice === undefined) pass.draw(3); else pass.draw(3, 1, 0, depthSlice);
            pass.end();
          }
          assertCurrent(entry);
          device.queue.submit([encoder.finish()]);
        }, {colorIndex}));
        sample.dirty = false; sample.copies++;
      }
      assertCurrent(entry);
      return {view: sample.view, sampler: sample.sampler, sampleType, viewDimension: shape.viewDimension,
        version: texture.version, sourceVersion: texture.source.version};
    } finally { entry.busy = false; if (!entry.alive) release(entry); }
  }
  function depthSampling(entry, texture) {
    assertCurrent(entry);
    if (entry.shape.cube || entry.shape.depthArray) fail('UNSUPPORTED', 'Cube and array depth sampling requires native texture bindings');
    if (entry.shape.sampleCount !== 1)
      fail('UNSUPPORTED', 'Multisampled depth requires a native per-sample binding, not the WebGL float replica');
    const samplerState = depthSamplerState(entry, texture);
    if (samplerState.compareFunction != null) return comparisonDepthSampling(entry, texture, samplerState);
    if (entry.busy) fail('REENTRANT', 'Cannot sample depth during a target submission');
    const device = getDevice();
    if (!device) fail('DEVICE', 'Initialize the renderer before sampling target depth');
    const filterable = device.features?.has?.('float32-filterable') === true;
    if (!filterable && (samplerState.minFilter !== three.NearestFilter || samplerState.magFilter !== three.NearestFilter))
      fail('SAMPLER', 'Linear depth sampling requires the device float32-filterable feature');
    const descriptor = {...samplerDescriptor(samplerState, three, 1), mipmapFilter: 'nearest', maxAnisotropy: 1};
    const samplerKey = JSON.stringify(descriptor);
    ensure(entry);
    // Retirement may be requested by a throwing/reentrant host boundary. Do not
    // release storage until the native withDepthTexture borrow has unwound.
    entry.busy = true;
    try {
      if (!entry.depthSample) {
        budget(entry.shape.depthSampleBytes);
        checked(entry, device => {
          const {width, height} = entry.shape;
          entry.depthSample = device.createTexture({label: 'Source target depth UV sampling image',
            size: {width, height, depthOrArrayLayers: 1}, format: 'r32float', dimension: '2d',
            sampleCount: 1, mipLevelCount: 1, usage: 4 | 16});
          entry.charged += entry.shape.depthSampleBytes; allocatedBytes += entry.shape.depthSampleBytes;
          entry.depthView = entry.depthSample.createView();
          let pipeline = pipelines.get('depth-to-r32float');
          if (!pipeline) {
            const module = device.createShaderModule({label: 'Source target depth UV orientation', code: DEPTH_FLIP_WGSL(flipY)});
            pipeline = device.createRenderPipeline({label: 'Source target depth UV orientation', layout: 'auto',
              vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format: 'r32float'}]},
              primitive: {topology: 'triangle-list'}});
            pipelines.set('depth-to-r32float', pipeline);
          }
          entry.depthPipeline = pipeline;
          entry.storage.withDepthTexture(source => {
            entry.depthGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
              entries: [{binding: 0, resource: source.createView({aspect: 'depth-only'})}]});
          });
        });
      }
      if (entry.depthSamplerKey !== samplerKey) {
        entry.depthSampler = checked(entry, device => device.createSampler(descriptor));
        entry.depthSamplerKey = samplerKey;
      }
      if (entry.depthDirty) {
        checked(entry, device => entry.storage.withDepthTexture(() => {
          assertCurrent(entry);
          const encoder = device.createCommandEncoder({label: 'Source target depth UV orientation'});
          const pass = encoder.beginRenderPass({colorAttachments: [{view: entry.depthView,
            loadOp: 'clear', storeOp: 'store', clearValue: {r: 0, g: 0, b: 0, a: 0}}]});
          pass.setPipeline(entry.depthPipeline); pass.setBindGroup(0, entry.depthGroup); pass.draw(3); pass.end();
          assertCurrent(entry);
          device.queue.submit([encoder.finish()]);
        }));
        entry.depthDirty = false; entry.depthCopies++;
      }
      assertCurrent(entry);
      return {view: entry.depthView, sampler: entry.depthSampler,
        sampleType: filterable ? 'float' : 'unfilterable-float', viewDimension: '2d',
        version: texture.version, sourceVersion: texture.source.version};
    } finally { entry.busy = false; if (!entry.alive) release(entry); }
  }
  // Public target attachments keep top-down rows. Comparison shaders need the
  // same source UV orientation as ordinary sampler2D reads, but an R32F color
  // replica cannot be bound as texture_depth_2d. Preserve the loaded depth in
  // depth32float instead; comparison filtering needs no float32-filterable
  // feature. Each replica has independent dirtiness and survives sampler edits.
  function comparisonDepthSampling(entry, texture, samplerState) {
    if (entry.busy) fail('REENTRANT', 'Cannot sample depth during a target submission');
    const descriptor = {...samplerDescriptor(samplerState, three, 1), mipmapFilter: 'nearest', maxAnisotropy: 1,
      compare: depthComparison(samplerState, three)};
    const samplerKey = JSON.stringify(descriptor);
    ensure(entry);
    entry.busy = true;
    try {
      let sample = entry.depthComparison;
      if (!sample) {
        budget(entry.shape.depthSampleBytes);
        sample = entry.depthComparison = {sample: null, dirty: true, copies: 0};
        checked(entry, device => {
          sample.sample = device.createTexture({label: 'Source target comparison depth UV sampling image',
            size: {width: entry.shape.width, height: entry.shape.height, depthOrArrayLayers: 1},
            format: 'depth32float', dimension: '2d', sampleCount: 1, mipLevelCount: 1, usage: 4 | 16});
          entry.charged += entry.shape.depthSampleBytes; allocatedBytes += entry.shape.depthSampleBytes;
          sample.view = sample.sample.createView({aspect: 'depth-only'});
          let pipeline = pipelines.get('depth-to-comparison');
          if (!pipeline) {
            const module = device.createShaderModule({label: 'Source target comparison depth UV orientation',
              code: DEPTH_FLIP_WGSL(flipY, true)});
            pipeline = device.createRenderPipeline({label: 'Source target comparison depth UV orientation', layout: 'auto',
              vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: []},
              primitive: {topology: 'triangle-list'},
              depthStencil: {format: 'depth32float', depthCompare: 'always', depthWriteEnabled: true}});
            pipelines.set('depth-to-comparison', pipeline);
          }
          sample.pipeline = pipeline;
          entry.storage.withDepthTexture(source => {
            sample.group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
              entries: [{binding: 0, resource: source.createView({aspect: 'depth-only'})}]});
          });
        });
      }
      if (sample.samplerKey !== samplerKey) {
        sample.sampler = checked(entry, device => device.createSampler(descriptor));
        sample.samplerKey = samplerKey;
      }
      if (sample.dirty) {
        checked(entry, device => entry.storage.withDepthTexture(() => {
          assertCurrent(entry);
          const encoder = device.createCommandEncoder({label: 'Source target comparison depth UV orientation'});
          const pass = encoder.beginRenderPass({colorAttachments: [], depthStencilAttachment: {
            view: sample.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store'}});
          pass.setPipeline(sample.pipeline); pass.setBindGroup(0, sample.group); pass.draw(3); pass.end();
          assertCurrent(entry); device.queue.submit([encoder.finish()]);
        }));
        sample.dirty = false; sample.copies++;
      }
      assertCurrent(entry);
      return {view: sample.view, sampler: sample.sampler, samplerDescriptor: Object.freeze(descriptor),
        depthReferenceClamp: !entry.shape.depthFormat.startsWith('depth32float'),
        sampleType: 'depth', viewDimension: '2d', version: texture.version, sourceVersion: texture.source.version};
    } finally { entry.busy = false; if (!entry.alive) release(entry); }
  }
  // r186 assigns renderTarget on DepthTexture without setting the color-only
  // isRenderTargetTexture flag. Never mutate source texture flags to route it.
  const textureTarget = texture => (texture?.isRenderTargetTexture === true || texture?.isDepthTexture === true)
    ? texture.renderTarget ?? ownedColorTargets.get(texture) ?? null : null;
  const targetTexture = texture => textureTarget(texture) !== null;
  // The existing scene bridge accepts a Map of externally owned bindings. Its
  // Ordinary has/get calls resolve current generations. Captured fullscreen
  // passes may instead select their retained generation across a trusted resize;
  // changed physical views use the bridge's existing preparation boundary.
  function bindingsFor(destination = null, retained = new Map(), capturedEntry = null) {
    if (!(retained instanceof Map)) fail('OPTIONS', 'Expected a Map of borrowed texture bindings');
    if (capturedEntry !== null && typeof capturedEntry !== 'function') fail('OPTIONS', 'Expected a captured target selector');
    return new class extends Map {
      has(texture) { return targetTexture(texture) || retained.has(texture); }
      get(texture) {
        if (!targetTexture(texture)) return retained.get(texture);
        const target = textureTarget(texture);
        if (destination && (destination.textures.includes(texture) ||
            (destination.depthBuffer || destination.stencilBuffer) && destination.depthTexture === texture))
          fail('FEEDBACK', 'A render pass cannot sample an attached color or depth image');
        const entry = capturedEntry?.(texture) ?? capture(target);
        if (entry.target !== target) fail('SOURCE', 'Captured texture generation belongs to another target');
        const depth = entry.shape.depthTexture === texture;
        if (!depth && !entry.shape.textures.includes(texture)) fail('STALE', 'Texture no longer belongs to its source target');
        const binding = depth ? depthSampling(entry, texture) : sampling(entry, texture);
        // Asynchronous source program preparation (PMREM) may hold this view
        // across pipeline compilation. The ordinary draw path stays synchronous;
        // the generator explicitly acquires/releases this existing generation.
        return {...binding, retain: () => {
          const finish = retainGeneration(entry);
          entry.programReferences = (entry.programReferences ?? 0) + 1;
          let held = true;
          return () => {
            if (!held) return;
            held = false; entry.programReferences--; finish();
            if (!entry.alive) release(entry);
          };
        }, validate: () => assertCurrent(entry)};
      }
    }();
  }
  function dirtyColor(entry, colorIndex) {
    if (colorIndex === 0) entry.dirty = true;
    else { const sample = entry.colorSamples.get(colorIndex); if (sample) sample.dirty = true; }
  }
  function dirtyDepth(entry) {
    entry.depthDirty = true;
    if (entry.depthComparison) entry.depthComparison.dirty = true;
  }
  function copyTarget(source, destination, options, depth = false, generateMipmaps = null) {
    assertCurrent(source); assertCurrent(destination);
    if (generateMipmaps !== null && typeof generateMipmaps !== 'boolean')
      fail('MIP', 'Color copy generation must be a captured boolean');
    if (depth) { source = source.depthEntry ?? source; destination = destination.depthEntry ?? destination; }
    const group = source.historyGroup && source.historyGroup === destination.historyGroup ? source.historyGroup : null;
    const sameOwner = source === destination || group !== null;
    if (depth && sameOwner) fail('FEEDBACK', 'In-place depth copies are not admitted');
    if (source.busy || destination.busy || group?.busy || group?.frame)
      fail('REENTRANT', 'Cannot copy during a target submission');
    // Source-shaped copies capture the live generation flag independently of
    // the initialized allocation. Raw transfer calls keep its authored policy.
    const destinationShape = generateMipmaps === null ? destination.shape : {...destination.shape,
      colorGenerateMipmaps: destination.shape.colorGenerateMipmaps.map((value, index) =>
        index === (options.colorIndex ?? 0) ? generateMipmaps : value)};
    const plan = depth
      ? inspectGpuDepthCopy({...source.shape, sampleDepth: source.shape.depthTexture !== null}, destinationShape, options)
      : inspectGpuTargetCopy(source.shape, destinationShape, options);
    if (!depth && sameOwner && source.shape.textures[plan.sourceColorIndex] === destination.shape.textures[plan.colorIndex] &&
        gpuTargetCopyOverlaps(source.shape, plan))
      fail('FEEDBACK', 'Copies cannot read and write overlapping color subresources');
    if (plan.empty) return false;
    budget(allocationBytes(source) + (sameOwner ? 0 : allocationBytes(destination)));
    ensure(source); ensure(destination);
    source.busy = destination.busy = true;
    if (group) group.busy = true;
    if (depth) { dirtyDepth(destination); if (destination.depthEntry) dirtyDepth(destination.depthEntry); }
    else dirtyColor(destination, plan.colorIndex);
    try {
      const result = destination.storage[depth ? 'copyDepthFrom' : 'copyFrom'](source.storage, options);
      if (!depth && generateMipmaps && plan.destinationMipLevel === 0) destination.storage.generateMipmaps([plan.colorIndex]);
      assertCurrent(source); assertCurrent(destination); return result;
    } finally {
      source.busy = destination.busy = false;
      if (group) group.busy = false;
      for (const entry of group?.entries ?? new Set([source, destination])) {
        if (!entry.alive) release(entry); else collect(entry);
      }
    }
  }
  function withNativeResize(target, width, height, run) {
    live();
    if (typeof run !== 'function' || !positive(width) || !positive(height)) fail('OPTIONS', 'Expected a source target resize and synchronous callback');
    if (!target || target.width === width && target.height === height) return run();
    if (nativeResizes.has(target)) fail('REENTRANT', 'Cannot reenter a source target resize');
    inspectThreeRenderTarget(target, three);
    nativeResizes.set(target, {width, height, depth: target.depth, event: null, shape: null});
    try { return run(); } finally { nativeResizes.delete(target); }
  }
  function withNativeEffectTargets(targets, run) {
    live();
    if (!Array.isArray(targets) || !targets.length || targets.length > 64 || typeof run !== 'function')
      fail('OPTIONS', 'Expected a finite native effect target set and synchronous callback');
    // This scope observes source resizing; it is not an allocation or a use of
    // every declared target. SSR, for example, constructs a 1x1 placeholder with
    // five authored mips, then sets the actual extent in updateBefore(). Cold
    // targets must reach the normal capture/admission boundary at their first
    // real use. Existing generations still require complete admission before
    // an actual size change can preserve their attachments and formats.
    const scopes = [...new Set(targets)].map(target => {
      if (!(target instanceof three.RenderTarget)) fail('SOURCE', 'Expected a source RenderTarget');
      if (target.isCubeRenderTarget || target.isWebGLCubeRenderTarget || target.isRenderTarget3D ||
          target.isWebGL3DRenderTarget || target.texture?.isData3DTexture)
        fail('UNSUPPORTED', 'Native effect resize scopes require ordinary 2D or 2D-array targets');
      if (entries.has(target)) inspectThreeRenderTarget(target, three);
      if (nativeResizes.has(target)) fail('REENTRANT', 'Cannot reenter a source target resize');
      const scope = {width: target.width, height: target.height, depth: target.depth,
        dynamic: true, event: null, shape: null};
      // A cold target can resize before its first owner entry exists. Observe
      // that event too, so a later explicit disposal cannot impersonate it.
      scope.onDispose = event => { scope.event ??= event; };
      return [target, scope];
    });
    const installed = [];
    try {
      for (const [target, scope] of scopes) {
        nativeResizes.set(target, scope); installed.push([target, scope]);
        target.addEventListener('dispose', scope.onDispose);
      }
      const result = run();
      if (result && typeof result.then === 'function') fail('ASYNC_USE', 'Native effect target scopes require a synchronous update');
      return result;
    } finally {
      for (const [target, scope] of installed) {
        target.removeEventListener('dispose', scope.onDispose); nativeResizes.delete(target);
      }
    }
  }
  return {
    capture, ensure, assertCurrent, bindingsFor, retainGeneration, registerNativePass,
    // Raw transfers need ownership without admitting a sampled/filtering use.
    // This also resolves constructors that leave the public owner pointer null.
    getTextureTarget(texture) { live(); return textureTarget(texture); },
    // Retained shadow/pass updates and declared private effect targets enter
    // these scopes.
    // Public setSize()/dispose() outside the synchronous operation still
    // invalidate pending requests and readers, including older generations.
    withNativeShadowResize: withNativeResize,
    withNativePassResize: withNativeResize,
    withNativeEffectTargets,
    // Building a shader requires a binding layout, not a usable allocation.
    // Source effects can describe an authored mip chain on a 1x1 placeholder
    // and select its actual size only in updateBefore(). For cold ordinary 2D
    // colors, inspect the base layout without capturing a generation or freezing
    // that placeholder's mip count. Actual capture below validates every level.
    // Keep established depth/layer/history initialization on its existing path.
    describeNativeTexture(texture, {comparison = false} = {}) {
      live();
      const target = textureTarget(texture);
      if (!target || flipY || entries.has(target) || nativePasses.has(target) || texture.isDepthTexture ||
          target.isCubeRenderTarget || target.isWebGLCubeRenderTarget || target.isRenderTarget3D ||
          target.isWebGL3DRenderTarget || target.isWebGLArrayRenderTarget || target.depth !== 1 ||
          texture.isDataArrayTexture || texture.isArrayTexture || texture.isData3DTexture)
        return this.captureNativeTexture(texture, {comparison});
      const shape = inspectThreeRenderTarget(target, three, {colorMipProfiles:
        Array.isArray(target.textures) ? target.textures.map(() => ({mipLevelCount: 1})) : null});
      const device = getDevice();
      if (!device) fail('DEVICE', 'Initialize the renderer before describing native target inputs');
      if (shape.textures.length > 1 || shape.colorFormats.includes('rg11b10ufloat'))
        gpuTargets.inspectGpuRenderTargetColors(device, shape.colorFormats);
      if (comparison) fail('SAMPLER', 'Comparison sampling requires a source depth texture with compareFunction');
      const {descriptor, sampleType} = colorSampler({shape}, texture), samples = shape.sampleCount;
      return Object.freeze({device, sampler: Object.freeze(descriptor), sampleType, viewDimension: shape.viewDimension,
        sampleData: Object.freeze({samples, primarySamples: 1, isMSAA: samples > 1})});
    },
    // Native WGSL uses the attachment's top-left coordinates, not the WebGL
    // bridge's UV-flipped sampling replica. Capture only generation and sampler
    // metadata; GPU contents are observed at the consumer's FIFO submission.
    captureNativeTexture(texture, {comparison = false, sampled = true} = {}) {
      live();
      if (!targetTexture(texture)) {
        if (texture?.isCubeTexture === true && texture.isRenderTargetTexture === true)
          fail('SOURCE', 'Select or render the CubeRenderTarget before sampling its color texture');
        fail('SOURCE', 'Expected a target-owned color or depth texture for native sampling');
      }
      let entry = capture(textureTarget(texture));
      const group = entry.historyGroup, device = getDevice();
      if (group && !texture.isDepthTexture && !entry.shape.textures.includes(texture)) {
        const index = group.pairs.findIndex(pair => pair.includes(texture));
        if (index < 0) fail('STALE', 'Texture is not a registered previous pass color');
        const colors = entry.shape.colors.slice(), textures = entry.shape.textures.slice();
        colors[index] = inspectThreeRenderTargetColor(entry.target, three, texture, null,
          group.mipInitialized ? entry.shape.colors[index] : null); textures[index] = texture;
        entry = historyEntry(group, {...entry.shape, ...colors[0], colors, textures,
          colorFormats: colors.map(color => color.format), bytes: entry.shape.bytes, key: entry.shape.key});
      }
      const depth = texture.isDepthTexture === true, colorIndex = depth ? -1 : entry.shape.textures.indexOf(texture);
      if (depth ? entry.shape.depthTexture !== texture : colorIndex < 0)
        fail('STALE', 'Texture no longer belongs to its source target');
      if (!device) fail('DEVICE', 'Initialize the renderer before capturing native target inputs');
      if (comparison && (!depth || texture.compareFunction == null))
        fail('SAMPLER', 'Comparison sampling requires a source depth texture with compareFunction');
      if (comparison && entry.shape.sampleCount !== 1)
        fail('SAMPLER', 'Multisampled depth supports native texel loads, not comparison sampling');
      // Native depth formats are depth bindings, not the WebGL bridge's R32F
      // replica. r186 forces ordinary depth samplers to nearest even when the
      // texture requests linear filtering. Comparison uses the source filters
      // and compare function; one depth view may have both kinds of consumers.
      // Raw copies use the same history-slot resolution without imposing a
      // sampler or optional float32-filterable requirement on copied pixels.
      const {descriptor, sampleType} = !sampled ? {descriptor: null, sampleType: null} : depth
        ? {descriptor: {...samplerDescriptor(texture, three, 1), mipmapFilter: 'nearest', maxAnisotropy: 1,
          ...(comparison ? {compare: depthComparison(texture, three)} : {magFilter: 'nearest', minFilter: 'nearest'})}, sampleType: 'depth'}
        : colorSampler(entry, texture);
      const samples = entry.shape.sampleCount;
      if (depth) prepareDepthTexture(entry);
      return Object.freeze({entry, texture, device, depth, colorIndex, sampler: descriptor === null ? null : Object.freeze(descriptor), sampleType, viewDimension: depth ? entry.shape.depthViewDimension : entry.shape.viewDimension,
        // r186 resolves color into its primary image. Depth retains every
        // sample and the source builder selects its own textureLoad index.
        sampleData: Object.freeze({samples, primarySamples: depth ? samples : 1, isMSAA: !depth && samples > 1})});
    },
    validateNativeTextures(sources) {
      live();
      let bytes = 0; const seen = new Set();
      for (const source of sources) {
        const {entry, texture, device} = source;
        assertCurrent(entry);
        if ((source.depth ? entry.shape.depthTexture : entry.shape.textures[source.colorIndex]) !== texture)
          fail('STALE', 'Native texture no longer belongs to its captured target');
        if (device !== getDevice() || entry.session && entry.session.device !== device)
          fail('DEVICE', 'Captured native target belongs to another GPUDevice');
        if ((entry.shape.cube || entry.shape.array || entry.shape.depthArray && entry.shape.depthFormat !== null) &&
            (!Number.isSafeInteger(device.limits?.maxTextureArrayLayers) || device.limits.maxTextureArrayLayers < entry.shape.layers))
          fail('DEVICE', 'Target layers exceed the device texture array limit');
        if (entry.shape.volume && (!Number.isSafeInteger(device.limits?.maxTextureDimension3D) ||
            Math.max(entry.shape.width, entry.shape.height, entry.shape.layers) > device.limits.maxTextureDimension3D))
          fail('DEVICE', 'Volume dimensions exceed the device 3D texture limit');
        if (entry.shape.volume && entry.shape.depthFormat !== null &&
            (!Number.isSafeInteger(device.limits?.maxTextureDimension2D) ||
              Math.max(entry.shape.width, entry.shape.height) > device.limits.maxTextureDimension2D))
          fail('DEVICE', 'Volume depth attachments exceed the device 2D texture limit');
        const group = entry.historyGroup;
        if (source.depth && [...allEntries()].some(active => (active.busy || active.historyGroup?.frame) &&
            active.shape.depthAttachment && active.shape.depthTexture === texture))
          fail('FEEDBACK', 'A pass cannot sample a shared depth image attached for writing');
        if (group?.frame) {
          if (source.depth && group.frame.depthAttachment || group.frame.writes.has(texture)) fail('FEEDBACK', 'A pass cannot sample its active color or depth attachments');
        } else if (entry.busy || group?.busy) fail('REENTRANT', 'Cannot sample a target during its submission');
        const owner = group ?? entry;
        if (!seen.has(owner)) bytes += allocationBytes(entry);
        seen.add(owner);
      }
      budget(bytes);
    },
    withFramebufferTexture(entry, callback, {depth = false} = {}) {
      assertCurrent(entry);
      if (depth) {
        if (typeof callback !== 'function') fail('CALLBACK', 'Expected a synchronous depth-copy callback');
        if (!entry.shape.depthAttachment || entry.shape.cube || entry.shape.array || entry.shape.volume)
          fail('DEPTH', 'Framebuffer depth copies require an ordinary 2D depth attachment');
        ensure(entry);
        const group = entry.historyGroup;
        if (entry.busy || group?.busy || group?.frame) fail('REENTRANT', 'Cannot reenter a framebuffer depth borrow');
        entry.busy = true; if (group) group.busy = true;
        try {
          return entry.storage.withFramebufferDepthTexture(native => {
            const result = callback(native); assertCurrent(entry); return result;
          });
        } finally {
          entry.busy = false; if (group) group.busy = false;
          for (const member of group?.entries ?? [entry]) { if (!member.alive) release(member); else collect(member); }
        }
      }
      const texture = entry.shape.textures[0];
      const source = {entry, texture, device: getDevice(), depth: false, colorIndex: 0};
      return this.withNativeTextures([source], textures => callback(textures.get(texture)));
    },
    withNativeTexture(source, callback) {
      if (typeof callback !== 'function') fail('CALLBACK', 'Expected a synchronous submitted-use callback');
      return this.withNativeTextures([source], textures => callback(textures.get(source.texture)));
    },
    withNativeTextures(sources, callback) {
      if (!Array.isArray(sources) || !sources.length || typeof callback !== 'function')
        fail('CALLBACK', 'Expected captured native inputs and a synchronous submitted-use callback');
      const {entry, device} = sources[0];
      this.validateNativeTextures(sources);
      const group = entry.historyGroup;
      if (sources.some(source => group ? source.entry.historyGroup !== group : source.entry !== entry))
        fail('SOURCE', 'A native borrow group must belong to one target allocation');
      const color = sources.some(source => !source.depth), depth = sources.some(source => source.depth);
      ensure(entry);
      if (entry.session.device !== device) fail('DEVICE', 'Native target storage belongs to another GPUDevice');
      const invoke = (nativeColors, nativeDepth) => {
        const textures = new Map(sources.map(source => [source.texture, source.depth ? nativeDepth : nativeColors[source.colorIndex]]));
        const result = callback(textures); assertCurrent(entry); return result;
      };
      if (group) {
        const historyInvoke = handles => {
          const textures = new Map(sources.map(source => [source.texture, source.depth ? handles.depth
            : source.entry.colorSlots[source.colorIndex] ? handles.historyColors[source.colorIndex] : handles.colors[source.colorIndex]]));
          const result = callback(textures);
          for (const source of sources) assertCurrent(source.entry);
          if (result?.then) { Promise.resolve(result).catch(() => {}); fail('ASYNC_USE', 'History texture borrows cannot cross an await'); }
          return result;
        };
        if (group.frame) return historyInvoke(group.frame.handles);
        group.busy = true;
        try { return group.storage.withAttachments((colors, depth, historyColors, withFrame) => {
          group.readBorrow = {withFrame, reads: new Set(sources.map(source => source.texture))};
          return historyInvoke({colors, depth, historyColors});
        }); }
        finally {
          group.readBorrow = null; group.busy = false;
          for (const member of group.entries) { if (!member.alive) release(member); else collect(member); }
        }
      }
      entry.busy = true; entry.nativeReads = new Set(sources.map(source => source.texture));
      try {
        if (entry.shape.textures.length > 1) return entry.storage.withAttachments(invoke);
        if (color && depth) return entry.storage.withSampledTextures((native, nativeDepth) => invoke([native], nativeDepth));
        if (depth) return entry.storage.withDepthTexture(native => invoke(null, native));
        return entry.storage.withTexture(native => invoke([native], null));
      } finally { entry.nativeReads = null; entry.busy = false; if (!entry.alive) release(entry); else collect(entry); }
    },
    // A raw transfer destination must not sample old mips or allocate a UV
    // replica. Keep identity, lifetime, allocation and validation with this owner.
    // Cube colors are the same six-layer native allocation: the caller's checked
    // copy origin/range selects faces, never the renderer's active face or a view.
    // As with face-aware copyColor, cube mip generation stays source-ordered.
    withNativeColorDestination(source, callback, {validation, generateMipmaps = false} = {}) {
      if (typeof callback !== 'function') fail('CALLBACK', 'Expected a synchronous submitted-use callback');
      if (typeof generateMipmaps !== 'boolean') fail('MIP', 'Native copy generation must be a captured boolean');
      this.validateNativeTextures([source]);
      const {entry, colorIndex} = source;
      if (source.depth || entry.shape.sampleCount !== 1)
        fail('UNSUPPORTED', 'Native color destinations require single-sample color attachments');
      ensure(entry);
      entry.busy = true;
      dirtyColor(entry, colorIndex);
      try {
        const result = entry.storage.withColorDestination(native => {
          const result = callback(native); assertCurrent(entry); return result;
        }, {colorIndex, validation});
        // Pinned copyTextureToTexture filters a generated level-zero destination
        // after the raw copy. Only that selected color becomes a new pyramid.
        if (generateMipmaps) entry.storage.generateMipmaps([colorIndex]);
        return result;
      } finally { entry.busy = false; if (!entry.alive) release(entry); else collect(entry); }
    },
    writePixels(entry, data, options = {}) {
      assertCurrent(entry);
      if (entry.busy) fail('REENTRANT', 'Cannot upload during a target submission');
      const plan = inspectGpuTargetWrite(entry.shape, data, options);
      if (plan.empty) return false;
      ensure(entry);
      entry.busy = true;
      dirtyColor(entry, plan.colorIndex);
      try { const result = entry.storage.writePixels(data, options); assertCurrent(entry); return result; }
      finally { entry.busy = false; if (!entry.alive) release(entry); }
    },
    copyColor(source, destination, options = {}, {generateMipmaps = null} = {}) {
      return copyTarget(source, destination, options, false, generateMipmaps);
    },
    copyDepth(source, destination, options = {}) { return copyTarget(source, destination, options, true); },
    get allocatedBytes() { return allocatedBytes; }, get size() { return allEntries().size; },
    render(entry, callback, {mipLevel = 0, face = 0, generateMipmaps = false, generateColorMipmaps = null, resolveColor = true} = {}) {
      assertCurrent(entry); inspectThreeRenderTargetMip(entry.shape, mipLevel, face);
      if (typeof resolveColor !== 'boolean') fail('OPTIONS', 'Color resolution requires a captured boolean');
      if (generateColorMipmaps !== null && (!Array.isArray(generateColorMipmaps) ||
          generateColorMipmaps.some(index => !Number.isSafeInteger(index) || index < 0 || index >= entry.shape.textures.length) ||
          new Set(generateColorMipmaps).size !== generateColorMipmaps.length))
        fail('MIP', 'MRT generation requires distinct captured color attachment indices');
      ensure(entry);
      const group = entry.historyGroup;
      const borrowed = group?.readBorrow;
      if (entry.busy || group?.busy && !borrowed || group?.frame) fail('REENTRANT', 'Cannot reenter a target submission');
      if (borrowed && [...entry.shape.textures, ...(entry.shape.depthAttachment ? [entry.shape.depthTexture] : [])].some(texture => borrowed.reads.has(texture)))
        fail('FEEDBACK', 'A pass cannot write an attachment held by its sampled inputs');
      const depthReads = entry.depthEntry?.nativeReads ?? entry.depthEntry?.historyGroup?.readBorrow?.reads;
      if (entry.shape.depthAttachment && depthReads?.has(entry.shape.depthTexture))
        fail('FEEDBACK', 'A pass cannot write shared depth held by its sampled inputs');
      budget(entry.storage.frameAllocationBytes(mipLevel, face));
      const allocatedBefore = entry.storage.allocatedBytes;
      entry.busy = true;
      if (resolveColor || entry.shape.sampleCount === 1) {
        entry.dirty = true;
        for (const sample of entry.colorSamples.values()) sample.dirty = true;
      }
      // Higher private depth levels never modify the source's level-zero depth.
      if (mipLevel === 0 && entry.shape.depthAttachment) {
        dirtyDepth(entry);
        if (entry.depthEntry) dirtyDepth(entry.depthEntry);
      }
      const withFrame = borrowed ? (run, options) => borrowed.withFrame(run, {...options, colorSlots: entry.colorSlots,
        generateColorMipmaps}) : entry.storage.withFrame;
      try {
        const result = withFrame((frame, texture, handles) => {
          if (group) group.frame = {handles, writes: new Set(entry.shape.textures), depthAttachment: entry.shape.depthAttachment};
          try {
            // WebGL clear() changes its multisample renderbuffers, but only a
            // render() resolves them. Leave the last sampled colors intact.
            if (!resolveColor) {
              const {resolveTarget, resolveTargets, ...unresolved} = frame;
              frame = unresolved;
            }
            // Source targets can own a depth texture solely for history copies.
            // Allocate and sample it normally, but a color render must neither
            // attach nor clear/write it when depthBuffer is disabled.
            const {depthView, ...colorFrame} = frame;
            const result = callback(entry.shape.depthAttachment ? frame : colorFrame, texture,
              entry.shape.depthAttachment ? handles.depth : null);
            assertCurrent(entry); return result;
          }
          finally { if (group) group.frame = null; }
        }, {mipLevel, face});
        // Finish the source render before filtering. Sampling and manual clear
        // do not infer generation from a later mutable public texture flag.
        if (entry.shape.cube && generateMipmaps) {
          entry.storage.generateMipmaps(); entry.cubeMipReady = true;
        }
        if (!borrowed && generateColorMipmaps !== null) entry.storage.generateMipmaps(generateColorMipmaps);
        return result;
      }
      finally {
        const delta = entry.storage.allocatedBytes - allocatedBefore;
        if (group) { group.charged += delta; group.mipDepthBytes += delta; } else entry.charged += delta;
        allocatedBytes += delta;
        entry.busy = false; if (!entry.alive) release(entry); else collect(entry);
        if (group) for (const member of group.entries) { if (!member.alive) release(member); else collect(member); }
      }
    },
    async whenIdle() {
      live();
      try { await Promise.all([...allEntries()].flatMap(entry =>
        entry.retirement ? [entry.retirement] : [entry.validation, entry.storage?.whenIdle(), entry._dispatcher?.whenIdle()]).concat([...retiring])); }
      catch (error) { throw terminal ?? error; }
      live();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of allEntries()) retire(entry);
      pipelines.clear();
    },
  };
}
