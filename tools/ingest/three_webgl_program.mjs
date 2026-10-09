/** Retained construction components ported from three.js r186 WebGLRenderer
 * (src/renderers/webgl/WebGLPrograms.js getParameters/getUniforms, WebGLProgram.js
 * source assembly, WebGLLights.js setup/setupView, WebGLMaterials.js uniform
 * refresh), MIT License, Copyright 2010-2025 three.js authors.
 *
 * Ownership route: RETAINED JAVASCRIPT (program construction and uniform values
 * only). These functions build the exact GLSL and uniform values WebGLRenderer
 * would use; essl_wgsl.mjs compiles the GLSL and the new WebGPU backend draws.
 * They take the application's own THREE module, so runtime edits to
 * ShaderChunk/ShaderLib/UniformsLib are honored as upstream does.
 *
 * Differences from upstream are explicit inputs, never silent: the caller
 * supplies the resolved environment map, light state, shadow/clipping counts and
 * render-target/tone-mapping context. Features the new backend cannot yet feed
 * (shadow maps, rect-area LTC textures, transmission targets, batching, morph
 * textures, light-probe grids) are rejected by inspectWebGLProgram().
 */
export class ThreeWebGLProgramError extends Error {
  constructor(code, message) { super(`THREE_WEBGL_PROGRAM_${code}: ${message}`); this.name = 'ThreeWebGLProgramError'; this.code = 'THREE_WEBGL_PROGRAM_' + code; }
}
const fail = (code, message) => { throw new ThreeWebGLProgramError(code, message); };

export const SHADER_IDS = Object.freeze({
  MeshDepthMaterial: 'depth', MeshDistanceMaterial: 'distance', MeshNormalMaterial: 'normal', MeshBasicMaterial: 'basic',
  MeshLambertMaterial: 'lambert', MeshPhongMaterial: 'phong', MeshToonMaterial: 'toon', MeshStandardMaterial: 'physical',
  MeshPhysicalMaterial: 'physical', MeshMatcapMaterial: 'matcap', LineBasicMaterial: 'basic', LineDashedMaterial: 'dashed',
  PointsMaterial: 'points', ShadowMaterial: 'shadow', SpriteMaterial: 'sprite',
});
export const materialNeedsLights = m => m.isMeshLambertMaterial || m.isMeshToonMaterial || m.isMeshPhongMaterial ||
  m.isMeshStandardMaterial || m.isShadowMaterial || (m.isShaderMaterial && m.lights === true);

// ---- WebGLPrograms.getParameters ------------------------------------------
/** ctx: {fog, envMap (resolved, or null), lights (webglLights().state), shadowMapEnabled,
 *  shadowMapType, toneMapping, outputColorSpace, precision, clipping: {numPlanes, numIntersection},
 *  logarithmicDepthBuffer, reversedDepthBuffer, side?} */
export function webglParameters(T, material, object, ctx) {
  const geometry = object.geometry, fog = ctx.fog ?? null, envMap = ctx.envMap ?? null, lights = ctx.lights;
  const envMapCubeUVHeight = !!envMap && envMap.mapping === T.CubeUVReflectionMapping ? envMap.image.height : null;
  const shaderID = SHADER_IDS[material.type];
  const precision = material.precision ?? ctx.precision ?? 'highp';
  const morphAttribute = geometry.morphAttributes.position || geometry.morphAttributes.normal || geometry.morphAttributes.color;
  const morphTargetsCount = morphAttribute !== undefined ? morphAttribute.length : 0;
  let morphTextureStride = 0;
  if (geometry.morphAttributes.position !== undefined) morphTextureStride = 1;
  if (geometry.morphAttributes.normal !== undefined) morphTextureStride = 2;
  if (geometry.morphAttributes.color !== undefined) morphTextureStride = 3;
  let vertexShader, fragmentShader;
  if (shaderID) { const shader = T.ShaderLib[shaderID]; vertexShader = shader.vertexShader; fragmentShader = shader.fragmentShader; }
  else { vertexShader = material.vertexShader; fragmentShader = material.fragmentShader; }
  const activeChannels = new Set();
  const getChannel = value => { activeChannels.add(value); return value === 0 ? 'uv' : `uv${value}`; };
  const side = ctx.side ?? material.side;
  const HAS_MAP = !!material.map, HAS_MATCAP = !!material.matcap, HAS_ENVMAP = !!envMap, HAS_AOMAP = !!material.aoMap;
  const HAS_LIGHTMAP = !!material.lightMap, HAS_BUMPMAP = !!material.bumpMap && material.wireframe === false;
  const HAS_NORMALMAP = !!material.normalMap, HAS_DISPLACEMENTMAP = !!material.displacementMap, HAS_EMISSIVEMAP = !!material.emissiveMap;
  const HAS_METALNESSMAP = !!material.metalnessMap, HAS_ROUGHNESSMAP = !!material.roughnessMap;
  const HAS_ANISOTROPY = material.anisotropy > 0, HAS_CLEARCOAT = material.clearcoat > 0, HAS_DISPERSION = material.dispersion > 0;
  const HAS_RETROREFLECTION = material.retroreflectivity > 0, HAS_IRIDESCENCE = material.iridescence > 0, HAS_SHEEN = material.sheen > 0;
  const HAS_TRANSMISSION = material.transmission > 0;
  const HAS_ANISOTROPYMAP = HAS_ANISOTROPY && !!material.anisotropyMap, HAS_CLEARCOATMAP = HAS_CLEARCOAT && !!material.clearcoatMap;
  const HAS_CLEARCOAT_NORMALMAP = HAS_CLEARCOAT && !!material.clearcoatNormalMap, HAS_CLEARCOAT_ROUGHNESSMAP = HAS_CLEARCOAT && !!material.clearcoatRoughnessMap;
  const HAS_IRIDESCENCEMAP = HAS_IRIDESCENCE && !!material.iridescenceMap, HAS_IRIDESCENCE_THICKNESSMAP = HAS_IRIDESCENCE && !!material.iridescenceThicknessMap;
  const HAS_SHEEN_COLORMAP = HAS_SHEEN && !!material.sheenColorMap, HAS_SHEEN_ROUGHNESSMAP = HAS_SHEEN && !!material.sheenRoughnessMap;
  const HAS_SPECULARMAP = !!material.specularMap, HAS_SPECULAR_COLORMAP = !!material.specularColorMap, HAS_SPECULAR_INTENSITYMAP = !!material.specularIntensityMap;
  const HAS_TRANSMISSIONMAP = HAS_TRANSMISSION && !!material.transmissionMap, HAS_THICKNESSMAP = HAS_TRANSMISSION && !!material.thicknessMap;
  const HAS_GRADIENTMAP = !!material.gradientMap, HAS_ALPHAMAP = !!material.alphaMap, HAS_ALPHATEST = material.alphaTest > 0;
  const HAS_ALPHAHASH = !!material.alphaHash, HAS_EXTENSIONS = !!material.extensions;
  const toneMapping = material.toneMapped ? ctx.toneMapping ?? T.NoToneMapping : T.NoToneMapping;
  const isPackedRG = format => format === T.RGFormat || format === T.RG11_EAC_Format || format === T.RED_GREEN_RGTC2_Format;
  const p = {
    shaderID, shaderType: material.type, shaderName: material.name, vertexShader, fragmentShader, defines: material.defines,
    isRawShaderMaterial: material.isRawShaderMaterial === true, glslVersion: material.glslVersion, precision,
    batching: object.isBatchedMesh === true, batchingColor: object.isBatchedMesh === true && object._colorsTexture !== null,
    instancing: object.isInstancedMesh === true, instancingColor: object.isInstancedMesh === true && object.instanceColor !== null,
    instancingMorph: object.isInstancedMesh === true && object.morphTexture !== null,
    outputColorSpace: ctx.outputColorSpace, alphaToCoverage: !!material.alphaToCoverage,
    map: HAS_MAP, matcap: HAS_MATCAP, envMap: HAS_ENVMAP, envMapMode: HAS_ENVMAP && envMap.mapping, envMapCubeUVHeight,
    aoMap: HAS_AOMAP, lightMap: HAS_LIGHTMAP, bumpMap: HAS_BUMPMAP, normalMap: HAS_NORMALMAP, displacementMap: HAS_DISPLACEMENTMAP, emissiveMap: HAS_EMISSIVEMAP,
    normalMapObjectSpace: HAS_NORMALMAP && material.normalMapType === T.ObjectSpaceNormalMap,
    normalMapTangentSpace: HAS_NORMALMAP && material.normalMapType === T.TangentSpaceNormalMap,
    packedNormalMap: HAS_NORMALMAP && material.normalMapType === T.TangentSpaceNormalMap && isPackedRG(material.normalMap.format),
    metalnessMap: HAS_METALNESSMAP, roughnessMap: HAS_ROUGHNESSMAP, anisotropy: HAS_ANISOTROPY, anisotropyMap: HAS_ANISOTROPYMAP,
    clearcoat: HAS_CLEARCOAT, clearcoatMap: HAS_CLEARCOATMAP, clearcoatNormalMap: HAS_CLEARCOAT_NORMALMAP, clearcoatRoughnessMap: HAS_CLEARCOAT_ROUGHNESSMAP,
    dispersion: HAS_DISPERSION, retroreflection: HAS_RETROREFLECTION, iridescence: HAS_IRIDESCENCE, iridescenceMap: HAS_IRIDESCENCEMAP,
    iridescenceThicknessMap: HAS_IRIDESCENCE_THICKNESSMAP, sheen: HAS_SHEEN, sheenColorMap: HAS_SHEEN_COLORMAP, sheenRoughnessMap: HAS_SHEEN_ROUGHNESSMAP,
    specularMap: HAS_SPECULARMAP, specularColorMap: HAS_SPECULAR_COLORMAP, specularIntensityMap: HAS_SPECULAR_INTENSITYMAP,
    transmission: HAS_TRANSMISSION, transmissionMap: HAS_TRANSMISSIONMAP, thicknessMap: HAS_THICKNESSMAP, gradientMap: HAS_GRADIENTMAP,
    opaque: material.transparent === false && material.blending === T.NormalBlending && material.alphaToCoverage === false,
    alphaMap: HAS_ALPHAMAP, alphaTest: HAS_ALPHATEST, alphaHash: HAS_ALPHAHASH, combine: material.combine,
    mapUv: HAS_MAP && getChannel(material.map.channel), aoMapUv: HAS_AOMAP && getChannel(material.aoMap.channel),
    lightMapUv: HAS_LIGHTMAP && getChannel(material.lightMap.channel), bumpMapUv: HAS_BUMPMAP && getChannel(material.bumpMap.channel),
    normalMapUv: HAS_NORMALMAP && getChannel(material.normalMap.channel), displacementMapUv: HAS_DISPLACEMENTMAP && getChannel(material.displacementMap.channel),
    emissiveMapUv: HAS_EMISSIVEMAP && getChannel(material.emissiveMap.channel), metalnessMapUv: HAS_METALNESSMAP && getChannel(material.metalnessMap.channel),
    roughnessMapUv: HAS_ROUGHNESSMAP && getChannel(material.roughnessMap.channel), anisotropyMapUv: HAS_ANISOTROPYMAP && getChannel(material.anisotropyMap.channel),
    clearcoatMapUv: HAS_CLEARCOATMAP && getChannel(material.clearcoatMap.channel), clearcoatNormalMapUv: HAS_CLEARCOAT_NORMALMAP && getChannel(material.clearcoatNormalMap.channel),
    clearcoatRoughnessMapUv: HAS_CLEARCOAT_ROUGHNESSMAP && getChannel(material.clearcoatRoughnessMap.channel),
    iridescenceMapUv: HAS_IRIDESCENCEMAP && getChannel(material.iridescenceMap.channel),
    iridescenceThicknessMapUv: HAS_IRIDESCENCE_THICKNESSMAP && getChannel(material.iridescenceThicknessMap.channel),
    sheenColorMapUv: HAS_SHEEN_COLORMAP && getChannel(material.sheenColorMap.channel), sheenRoughnessMapUv: HAS_SHEEN_ROUGHNESSMAP && getChannel(material.sheenRoughnessMap.channel),
    specularMapUv: HAS_SPECULARMAP && getChannel(material.specularMap.channel), specularColorMapUv: HAS_SPECULAR_COLORMAP && getChannel(material.specularColorMap.channel),
    specularIntensityMapUv: HAS_SPECULAR_INTENSITYMAP && getChannel(material.specularIntensityMap.channel),
    transmissionMapUv: HAS_TRANSMISSIONMAP && getChannel(material.transmissionMap.channel), thicknessMapUv: HAS_THICKNESSMAP && getChannel(material.thicknessMap.channel),
    alphaMapUv: HAS_ALPHAMAP && getChannel(material.alphaMap.channel),
    vertexTangents: !!geometry.attributes.tangent && (HAS_NORMALMAP || HAS_ANISOTROPY), vertexNormals: !!geometry.attributes.normal,
    vertexColors: material.vertexColors, vertexAlphas: material.vertexColors === true && !!geometry.attributes.color && geometry.attributes.color.itemSize === 4,
    pointsUvs: object.isPoints === true && !!geometry.attributes.uv && (HAS_MAP || HAS_ALPHAMAP),
    fog: !!fog, useFog: material.fog === true, fogExp2: !!fog && fog.isFogExp2 === true,
    flatShading: material.wireframe === false && (material.flatShading === true || (geometry.attributes.normal === undefined && HAS_NORMALMAP === false &&
      (material.isMeshLambertMaterial || material.isMeshPhongMaterial || material.isMeshStandardMaterial || material.isMeshPhysicalMaterial))),
    sizeAttenuation: material.sizeAttenuation === true, logarithmicDepthBuffer: !!ctx.logarithmicDepthBuffer, reversedDepthBuffer: !!ctx.reversedDepthBuffer,
    skinning: object.isSkinnedMesh === true, hasPositionAttribute: geometry.attributes.position !== undefined,
    morphTargets: geometry.morphAttributes.position !== undefined, morphNormals: geometry.morphAttributes.normal !== undefined,
    morphColors: geometry.morphAttributes.color !== undefined, morphTargetsCount, morphTextureStride,
    numSunLights: lights.sun.length, numDirLights: lights.directional.length, numPointLights: lights.point.length, numSpotLights: lights.spot.length,
    numSpotLightMaps: lights.spotLightMap.length, numRectAreaLights: lights.rectArea.length, numHemiLights: lights.hemi.length,
    numSunLightShadows: lights.sunShadowMap.length, numDirLightShadows: lights.directionalShadowMap.length, numPointLightShadows: lights.pointShadowMap.length,
    numSpotLightShadows: lights.spotShadowMap.length, numSpotLightShadowsWithMaps: lights.numSpotLightShadowsWithMaps, numLightProbes: lights.numLightProbes,
    numLightProbeGrids: 0, numClippingPlanes: ctx.clipping?.numPlanes ?? 0, numClipIntersection: ctx.clipping?.numIntersection ?? 0,
    dithering: material.dithering, shadowMapEnabled: !!ctx.shadowMapEnabled, shadowMapType: ctx.shadowMapType ?? T.PCFShadowMap, toneMapping,
    decodeVideoTexture: HAS_MAP && material.map.isVideoTexture === true && T.ColorManagement.getTransfer(material.map.colorSpace) === T.SRGBTransfer,
    decodeVideoTextureEmissive: HAS_EMISSIVEMAP && material.emissiveMap.isVideoTexture === true && T.ColorManagement.getTransfer(material.emissiveMap.colorSpace) === T.SRGBTransfer,
    premultipliedAlpha: material.premultipliedAlpha, doubleSided: side === T.DoubleSide, flipSided: side === T.BackSide,
    useDepthPacking: material.depthPacking >= 0, depthPacking: material.depthPacking || 0,
    extensionClipCullDistance: false, extensionMultiDraw: false,
  };
  // The usage of getChannel() determines the active texture channels.
  p.vertexUv1s = activeChannels.has(1); p.vertexUv2s = activeChannels.has(2); p.vertexUv3s = activeChannels.has(3);
  return p;
}

// ---- WebGLProgram source assembly -------------------------------------------
const PRECISION_TYPES = ['float', 'int', 'sampler2D', 'samplerCube', 'sampler3D', 'sampler2DArray', 'sampler2DShadow', 'samplerCubeShadow',
  'sampler2DArrayShadow', 'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray', 'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray'];
function generatePrecision(p) {
  let s = PRECISION_TYPES.map(t => `precision ${p.precision} ${t};`).join('\n\t') + '\n\t';
  s = PRECISION_TYPES.map((t, i) => (i === 0 ? '' : '\t') + `precision ${p.precision} ${t};`).join('\n') + '\n\t';
  if (p.precision === 'highp') s += '\n#define HIGH_PRECISION';
  else if (p.precision === 'mediump') s += '\n#define MEDIUM_PRECISION';
  else if (p.precision === 'lowp') s += '\n#define LOW_PRECISION';
  return s;
}
function generateDefines(defines) {
  const chunks = [];
  for (const name in defines) { const value = defines[name]; if (value !== false) chunks.push('#define ' + name + ' ' + value); }
  return chunks.join('\n');
}
const filterEmptyLine = s => s !== '';
const includePattern = /^[ \t]*#include +<([\w\d./]+)>/gm;
function resolveIncludes(T, string) {
  return string.replace(includePattern, (match, include) => {
    const s = T.ShaderChunk[include];
    if (s === undefined) fail('SOURCE', `Can not resolve #include <${include}>`);
    return resolveIncludes(T, s);
  });
}
const unrollLoopPattern = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;
const unrollLoops = s => s.replace(unrollLoopPattern, (m, start, end, snippet) => {
  let out = '';
  for (let i = parseInt(start); i < parseInt(end); i++) out += snippet.replace(/\[\s*i\s*\]/g, '[ ' + i + ' ]').replace(/UNROLLED_LOOP_INDEX/g, i);
  return out;
});
function replaceLightNums(s, p) {
  const numSpotLightCoords = p.numSpotLightShadows + p.numSpotLightMaps - p.numSpotLightShadowsWithMaps;
  return s.replace(/NUM_SUN_LIGHTS/g, p.numSunLights).replace(/NUM_DIR_LIGHTS/g, p.numDirLights).replace(/NUM_SPOT_LIGHTS/g, p.numSpotLights)
    .replace(/NUM_SPOT_LIGHT_MAPS/g, p.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g, numSpotLightCoords)
    .replace(/NUM_RECT_AREA_LIGHTS/g, p.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g, p.numPointLights).replace(/NUM_HEMI_LIGHTS/g, p.numHemiLights)
    .replace(/NUM_SUN_LIGHT_SHADOWS/g, p.numSunLightShadows).replace(/NUM_DIR_LIGHT_SHADOWS/g, p.numDirLightShadows)
    .replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g, p.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g, p.numSpotLightShadows)
    .replace(/NUM_POINT_LIGHT_SHADOWS/g, p.numPointLightShadows);
}
const replaceClippingPlaneNums = (s, p) => s.replace(/NUM_CLIPPING_PLANES/g, p.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g, p.numClippingPlanes - p.numClipIntersection);
function encodingFunction(T, name, colorSpace) {
  const m = new T.Matrix3();
  T.ColorManagement._getMatrix(m, T.ColorManagement.workingColorSpace, colorSpace);
  const matrix = `mat3( ${m.elements.map(v => v.toFixed(4))} )`;
  const transfer = T.ColorManagement.getTransfer(colorSpace);
  const oetf = transfer === T.SRGBTransfer ? 'sRGBTransferOETF' : 'LinearTransferOETF';
  return [`vec4 ${name}( vec4 value ) {`, `\treturn ${oetf}( vec4( value.rgb * ${matrix}, value.a ) );`, '}'].join('\n');
}
function toneMappingFunction(T, toneMapping) {
  const names = {[T.LinearToneMapping]: 'Linear', [T.ReinhardToneMapping]: 'Reinhard', [T.CineonToneMapping]: 'Cineon',
    [T.ACESFilmicToneMapping]: 'ACESFilmic', [T.AgXToneMapping]: 'AgX', [T.NeutralToneMapping]: 'Neutral', [T.CustomToneMapping]: 'Custom'};
  const n = names[toneMapping];
  return n === undefined ? 'vec3 toneMapping( vec3 color ) { return LinearToneMapping( color ); }' : `vec3 toneMapping( vec3 color ) { return ${n}ToneMapping( color ); }`;
}
function luminanceFunction(T) {
  const v = new T.Vector3(); T.ColorManagement.getLuminanceCoefficients(v);
  return ['float luminance( const in vec3 rgb ) {', `\tconst vec3 weights = vec3( ${v.x.toFixed(4)}, ${v.y.toFixed(4)}, ${v.z.toFixed(4)} );`, '\treturn dot( weights, rgb );', '}'].join('\n');
}
function cubeUVSize(p) {
  const imageHeight = p.envMapCubeUVHeight;
  if (imageHeight === null) return null;
  const maxMip = Math.log2(imageHeight) - 2;
  return {texelWidth: 1.0 / (3 * Math.max(Math.pow(2, maxMip), 7 * 16)), texelHeight: 1.0 / imageHeight, maxMip};
}

/** The exact (vertex, fragment) GLSL WebGLProgram builds for these parameters. */
export function webglProgramSources(T, p) {
  const shadowMapTypeDefine = {[T.PCFShadowMap]: 'SHADOWMAP_TYPE_PCF', [T.VSMShadowMap]: 'SHADOWMAP_TYPE_VSM'}[p.shadowMapType] || 'SHADOWMAP_TYPE_BASIC';
  const envMapTypeDefine = p.envMap === false ? 'ENVMAP_TYPE_CUBE' : ({[T.CubeReflectionMapping]: 'ENVMAP_TYPE_CUBE', [T.CubeRefractionMapping]: 'ENVMAP_TYPE_CUBE',
    [T.CubeUVReflectionMapping]: 'ENVMAP_TYPE_CUBE_UV'}[p.envMapMode] || 'ENVMAP_TYPE_CUBE');
  const envMapModeDefine = p.envMap === false ? 'ENVMAP_MODE_REFLECTION' : ({[T.CubeRefractionMapping]: 'ENVMAP_MODE_REFRACTION'}[p.envMapMode] || 'ENVMAP_MODE_REFLECTION');
  const envMapBlendingDefine = p.envMap === false ? 'ENVMAP_BLENDING_NONE' : ({[T.MultiplyOperation]: 'ENVMAP_BLENDING_MULTIPLY',
    [T.MixOperation]: 'ENVMAP_BLENDING_MIX', [T.AddOperation]: 'ENVMAP_BLENDING_ADD'}[p.combine] || 'ENVMAP_BLENDING_NONE');
  const envMapCubeUVSize = cubeUVSize(p);
  const customDefines = generateDefines(p.defines);
  let vertexShader = p.vertexShader, fragmentShader = p.fragmentShader;
  let prefixVertex, prefixFragment;
  let versionString = p.glslVersion ? '#version ' + p.glslVersion + '\n' : '';
  if (p.isRawShaderMaterial) {
    prefixVertex = ['#define SHADER_TYPE ' + p.shaderType, '#define SHADER_NAME ' + p.shaderName, customDefines].filter(filterEmptyLine).join('\n');
    if (prefixVertex.length > 0) prefixVertex += '\n';
    prefixFragment = ['#define SHADER_TYPE ' + p.shaderType, '#define SHADER_NAME ' + p.shaderName, customDefines].filter(filterEmptyLine).join('\n');
    if (prefixFragment.length > 0) prefixFragment += '\n';
  } else {
    const d = (flag, name) => flag ? '#define ' + name : '';
    const uv = (value, name) => value ? `#define ${name} ${value}` : '';
    prefixVertex = [generatePrecision(p), '#define SHADER_TYPE ' + p.shaderType, '#define SHADER_NAME ' + p.shaderName, customDefines,
      d(p.extensionClipCullDistance, 'USE_CLIP_DISTANCE'), d(p.batching, 'USE_BATCHING'), d(p.batchingColor, 'USE_BATCHING_COLOR'),
      d(p.instancing, 'USE_INSTANCING'), d(p.instancingColor, 'USE_INSTANCING_COLOR'), d(p.instancingMorph, 'USE_INSTANCING_MORPH'),
      d(p.useFog && p.fog, 'USE_FOG'), d(p.useFog && p.fogExp2, 'FOG_EXP2'),
      d(p.map, 'USE_MAP'), d(p.envMap, 'USE_ENVMAP'), p.envMap ? '#define ' + envMapModeDefine : '', d(p.lightMap, 'USE_LIGHTMAP'), d(p.aoMap, 'USE_AOMAP'),
      d(p.bumpMap, 'USE_BUMPMAP'), d(p.normalMap, 'USE_NORMALMAP'), d(p.normalMapObjectSpace, 'USE_NORMALMAP_OBJECTSPACE'),
      d(p.normalMapTangentSpace, 'USE_NORMALMAP_TANGENTSPACE'), d(p.displacementMap, 'USE_DISPLACEMENTMAP'), d(p.emissiveMap, 'USE_EMISSIVEMAP'),
      d(p.anisotropy, 'USE_ANISOTROPY'), d(p.anisotropyMap, 'USE_ANISOTROPYMAP'), d(p.clearcoatMap, 'USE_CLEARCOATMAP'),
      d(p.clearcoatRoughnessMap, 'USE_CLEARCOAT_ROUGHNESSMAP'), d(p.clearcoatNormalMap, 'USE_CLEARCOAT_NORMALMAP'),
      d(p.iridescenceMap, 'USE_IRIDESCENCEMAP'), d(p.iridescenceThicknessMap, 'USE_IRIDESCENCE_THICKNESSMAP'),
      d(p.specularMap, 'USE_SPECULARMAP'), d(p.specularColorMap, 'USE_SPECULAR_COLORMAP'), d(p.specularIntensityMap, 'USE_SPECULAR_INTENSITYMAP'),
      d(p.roughnessMap, 'USE_ROUGHNESSMAP'), d(p.metalnessMap, 'USE_METALNESSMAP'), d(p.alphaMap, 'USE_ALPHAMAP'), d(p.alphaHash, 'USE_ALPHAHASH'),
      d(p.transmission, 'USE_TRANSMISSION'), d(p.transmissionMap, 'USE_TRANSMISSIONMAP'), d(p.thicknessMap, 'USE_THICKNESSMAP'),
      d(p.sheenColorMap, 'USE_SHEEN_COLORMAP'), d(p.sheenRoughnessMap, 'USE_SHEEN_ROUGHNESSMAP'),
      uv(p.mapUv, 'MAP_UV'), uv(p.alphaMapUv, 'ALPHAMAP_UV'), uv(p.lightMapUv, 'LIGHTMAP_UV'), uv(p.aoMapUv, 'AOMAP_UV'),
      uv(p.emissiveMapUv, 'EMISSIVEMAP_UV'), uv(p.bumpMapUv, 'BUMPMAP_UV'), uv(p.normalMapUv, 'NORMALMAP_UV'), uv(p.displacementMapUv, 'DISPLACEMENTMAP_UV'),
      uv(p.metalnessMapUv, 'METALNESSMAP_UV'), uv(p.roughnessMapUv, 'ROUGHNESSMAP_UV'), uv(p.anisotropyMapUv, 'ANISOTROPYMAP_UV'),
      uv(p.clearcoatMapUv, 'CLEARCOATMAP_UV'), uv(p.clearcoatNormalMapUv, 'CLEARCOAT_NORMALMAP_UV'), uv(p.clearcoatRoughnessMapUv, 'CLEARCOAT_ROUGHNESSMAP_UV'),
      uv(p.iridescenceMapUv, 'IRIDESCENCEMAP_UV'), uv(p.iridescenceThicknessMapUv, 'IRIDESCENCE_THICKNESSMAP_UV'),
      uv(p.sheenColorMapUv, 'SHEEN_COLORMAP_UV'), uv(p.sheenRoughnessMapUv, 'SHEEN_ROUGHNESSMAP_UV'), uv(p.specularMapUv, 'SPECULARMAP_UV'),
      uv(p.specularColorMapUv, 'SPECULAR_COLORMAP_UV'), uv(p.specularIntensityMapUv, 'SPECULAR_INTENSITYMAP_UV'),
      uv(p.transmissionMapUv, 'TRANSMISSIONMAP_UV'), uv(p.thicknessMapUv, 'THICKNESSMAP_UV'),
      d(p.vertexTangents && p.flatShading === false, 'USE_TANGENT'), d(p.vertexNormals, 'HAS_NORMAL'), d(p.vertexColors, 'USE_COLOR'),
      d(p.vertexAlphas, 'USE_COLOR_ALPHA'), d(p.vertexUv1s, 'USE_UV1'), d(p.vertexUv2s, 'USE_UV2'), d(p.vertexUv3s, 'USE_UV3'),
      d(p.pointsUvs, 'USE_POINTS_UV'), d(p.flatShading, 'FLAT_SHADED'), d(p.skinning, 'USE_SKINNING'), d(p.morphTargets, 'USE_MORPHTARGETS'),
      d(p.morphNormals && p.flatShading === false, 'USE_MORPHNORMALS'), d(p.morphColors, 'USE_MORPHCOLORS'),
      p.morphTargetsCount > 0 ? '#define MORPHTARGETS_TEXTURE_STRIDE ' + p.morphTextureStride : '',
      p.morphTargetsCount > 0 ? '#define MORPHTARGETS_COUNT ' + p.morphTargetsCount : '',
      d(p.doubleSided, 'DOUBLE_SIDED'), d(p.flipSided, 'FLIP_SIDED'), d(p.shadowMapEnabled, 'USE_SHADOWMAP'), p.shadowMapEnabled ? '#define ' + shadowMapTypeDefine : '',
      d(p.sizeAttenuation, 'USE_SIZEATTENUATION'), d(p.numLightProbes > 0, 'USE_LIGHT_PROBES'),
      d(p.logarithmicDepthBuffer, 'USE_LOGARITHMIC_DEPTH_BUFFER'), d(p.reversedDepthBuffer, 'USE_REVERSED_DEPTH_BUFFER'),
      'uniform mat4 modelMatrix;', 'uniform mat4 modelViewMatrix;', 'uniform mat4 projectionMatrix;', 'uniform mat4 viewMatrix;',
      'uniform mat3 normalMatrix;', 'uniform vec3 cameraPosition;', 'uniform bool isOrthographic;',
      '#ifdef USE_INSTANCING', '\tattribute mat4 instanceMatrix;', '#endif', '#ifdef USE_INSTANCING_COLOR', '\tattribute vec3 instanceColor;', '#endif',
      '#ifdef USE_INSTANCING_MORPH', '\tuniform sampler2D morphTexture;', '#endif',
      'attribute vec3 position;', 'attribute vec3 normal;', 'attribute vec2 uv;',
      '#ifdef USE_UV1', '\tattribute vec2 uv1;', '#endif', '#ifdef USE_UV2', '\tattribute vec2 uv2;', '#endif', '#ifdef USE_UV3', '\tattribute vec2 uv3;', '#endif',
      '#ifdef USE_TANGENT', '\tattribute vec4 tangent;', '#endif',
      '#if defined( USE_COLOR_ALPHA )', '\tattribute vec4 color;', '#elif defined( USE_COLOR )', '\tattribute vec3 color;', '#endif',
      '#ifdef USE_SKINNING', '\tattribute vec4 skinIndex;', '\tattribute vec4 skinWeight;', '#endif', '\n'].filter(filterEmptyLine).join('\n');
    prefixFragment = [generatePrecision(p), '#define SHADER_TYPE ' + p.shaderType, '#define SHADER_NAME ' + p.shaderName, customDefines,
      d(p.useFog && p.fog, 'USE_FOG'), d(p.useFog && p.fogExp2, 'FOG_EXP2'), d(p.alphaToCoverage, 'ALPHA_TO_COVERAGE'),
      d(p.map, 'USE_MAP'), d(p.matcap, 'USE_MATCAP'), d(p.envMap, 'USE_ENVMAP'), p.envMap ? '#define ' + envMapTypeDefine : '',
      p.envMap ? '#define ' + envMapModeDefine : '', p.envMap ? '#define ' + envMapBlendingDefine : '',
      envMapCubeUVSize ? '#define CUBEUV_TEXEL_WIDTH ' + envMapCubeUVSize.texelWidth : '',
      envMapCubeUVSize ? '#define CUBEUV_TEXEL_HEIGHT ' + envMapCubeUVSize.texelHeight : '',
      envMapCubeUVSize ? '#define CUBEUV_MAX_MIP ' + envMapCubeUVSize.maxMip + '.0' : '',
      d(p.lightMap, 'USE_LIGHTMAP'), d(p.aoMap, 'USE_AOMAP'), d(p.bumpMap, 'USE_BUMPMAP'), d(p.normalMap, 'USE_NORMALMAP'),
      d(p.normalMapObjectSpace, 'USE_NORMALMAP_OBJECTSPACE'), d(p.normalMapTangentSpace, 'USE_NORMALMAP_TANGENTSPACE'),
      d(p.packedNormalMap, 'USE_PACKED_NORMALMAP'), d(p.emissiveMap, 'USE_EMISSIVEMAP'), d(p.anisotropy, 'USE_ANISOTROPY'), d(p.anisotropyMap, 'USE_ANISOTROPYMAP'),
      d(p.clearcoat, 'USE_CLEARCOAT'), d(p.clearcoatMap, 'USE_CLEARCOATMAP'), d(p.clearcoatRoughnessMap, 'USE_CLEARCOAT_ROUGHNESSMAP'),
      d(p.clearcoatNormalMap, 'USE_CLEARCOAT_NORMALMAP'), d(p.dispersion, 'USE_DISPERSION'), d(p.retroreflection, 'USE_RETROREFLECTION'),
      d(p.iridescence, 'USE_IRIDESCENCE'), d(p.iridescenceMap, 'USE_IRIDESCENCEMAP'), d(p.iridescenceThicknessMap, 'USE_IRIDESCENCE_THICKNESSMAP'),
      d(p.specularMap, 'USE_SPECULARMAP'), d(p.specularColorMap, 'USE_SPECULAR_COLORMAP'), d(p.specularIntensityMap, 'USE_SPECULAR_INTENSITYMAP'),
      d(p.roughnessMap, 'USE_ROUGHNESSMAP'), d(p.metalnessMap, 'USE_METALNESSMAP'), d(p.alphaMap, 'USE_ALPHAMAP'), d(p.alphaTest, 'USE_ALPHATEST'),
      d(p.alphaHash, 'USE_ALPHAHASH'), d(p.sheen, 'USE_SHEEN'), d(p.sheenColorMap, 'USE_SHEEN_COLORMAP'), d(p.sheenRoughnessMap, 'USE_SHEEN_ROUGHNESSMAP'),
      d(p.transmission, 'USE_TRANSMISSION'), d(p.transmissionMap, 'USE_TRANSMISSIONMAP'), d(p.thicknessMap, 'USE_THICKNESSMAP'),
      d(p.vertexTangents && p.flatShading === false, 'USE_TANGENT'), d(p.vertexColors || p.instancingColor, 'USE_COLOR'),
      d(p.vertexAlphas || p.batchingColor, 'USE_COLOR_ALPHA'), d(p.vertexUv1s, 'USE_UV1'), d(p.vertexUv2s, 'USE_UV2'), d(p.vertexUv3s, 'USE_UV3'),
      d(p.pointsUvs, 'USE_POINTS_UV'), d(p.gradientMap, 'USE_GRADIENTMAP'), d(p.flatShading, 'FLAT_SHADED'),
      d(p.doubleSided, 'DOUBLE_SIDED'), d(p.flipSided, 'FLIP_SIDED'), d(p.shadowMapEnabled, 'USE_SHADOWMAP'), p.shadowMapEnabled ? '#define ' + shadowMapTypeDefine : '',
      d(p.premultipliedAlpha, 'PREMULTIPLIED_ALPHA'), d(p.numLightProbes > 0, 'USE_LIGHT_PROBES'), d(p.numLightProbeGrids > 0, 'USE_LIGHT_PROBES_GRID'),
      d(p.decodeVideoTexture, 'DECODE_VIDEO_TEXTURE'), d(p.decodeVideoTextureEmissive, 'DECODE_VIDEO_TEXTURE_EMISSIVE'),
      d(p.logarithmicDepthBuffer, 'USE_LOGARITHMIC_DEPTH_BUFFER'), d(p.reversedDepthBuffer, 'USE_REVERSED_DEPTH_BUFFER'),
      'uniform mat4 viewMatrix;', 'uniform vec3 cameraPosition;', 'uniform bool isOrthographic;',
      p.toneMapping !== T.NoToneMapping ? '#define TONE_MAPPING' : '', p.toneMapping !== T.NoToneMapping ? T.ShaderChunk['tonemapping_pars_fragment'] : '',
      p.toneMapping !== T.NoToneMapping ? toneMappingFunction(T, p.toneMapping) : '',
      d(p.dithering, 'DITHERING'), d(p.opaque, 'OPAQUE'), T.ShaderChunk['colorspace_pars_fragment'],
      encodingFunction(T, 'linearToOutputTexel', p.outputColorSpace), luminanceFunction(T),
      p.useDepthPacking ? '#define DEPTH_PACKING ' + p.depthPacking : '', '\n'].filter(filterEmptyLine).join('\n');
  }
  vertexShader = replaceClippingPlaneNums(replaceLightNums(resolveIncludes(T, vertexShader), p), p);
  fragmentShader = replaceClippingPlaneNums(replaceLightNums(resolveIncludes(T, fragmentShader), p), p);
  vertexShader = unrollLoops(vertexShader); fragmentShader = unrollLoops(fragmentShader);
  if (p.isRawShaderMaterial !== true) {
    versionString = '#version 300 es\n';
    prefixVertex = ['', '#define attribute in', '#define varying out', '#define texture2D texture'].join('\n') + '\n' + prefixVertex;
    prefixFragment = ['#define varying in', p.glslVersion === T.GLSL3 ? '' : 'layout(location = 0) out highp vec4 pc_fragColor;',
      p.glslVersion === T.GLSL3 ? '' : '#define gl_FragColor pc_fragColor', '#define gl_FragDepthEXT gl_FragDepth', '#define texture2D texture',
      '#define textureCube texture', '#define texture2DProj textureProj', '#define texture2DLodEXT textureLod', '#define texture2DProjLodEXT textureProjLod',
      '#define textureCubeLodEXT textureLod', '#define texture2DGradEXT textureGrad', '#define texture2DProjGradEXT textureProjGrad',
      '#define textureCubeGradEXT textureGrad'].join('\n') + '\n' + prefixFragment;
  }
  const vertex = versionString + prefixVertex + vertexShader, fragment = versionString + prefixFragment + fragmentShader;
  return {vertex, fragment, key: vertex + '\u0000' + fragment};
}

// ---- WebGLLights -----------------------------------------------------------
/** options.floatLinear(): WebGL OES_texture_float_linear (WebGPU float32-filterable)
 * selects the float LTC tables for rect-area lights, as WebGLLights does. */
export function webglLights(T, {floatLinear = () => false} = {}) {
  const cache = new Map();
  const uniformsFor = light => {
    let u = cache.get(light.id);
    if (u) return u;
    switch (light.type) {
      case 'SunLight': case 'DirectionalLight': u = {direction: new T.Vector3(), color: new T.Color()}; break;
      case 'SpotLight': u = {position: new T.Vector3(), direction: new T.Vector3(), color: new T.Color(), distance: 0, coneCos: 0, penumbraCos: 0, decay: 0}; break;
      case 'PointLight': u = {position: new T.Vector3(), color: new T.Color(), distance: 0, decay: 0}; break;
      case 'HemisphereLight': u = {direction: new T.Vector3(), skyColor: new T.Color(), groundColor: new T.Color()}; break;
      case 'RectAreaLight': u = {color: new T.Color(), position: new T.Vector3(), halfWidth: new T.Vector3(), halfHeight: new T.Vector3()}; break;
    }
    cache.set(light.id, u);
    return u;
  };
  const state = {version: 0, ambient: [0, 0, 0], probe: [], sun: [], sunShadow: [], sunShadowMap: [], sunShadowMatrix: [], sunShadowCascade: [],
    directional: [], directionalShadow: [], directionalShadowMap: [], directionalShadowMatrix: [], spot: [], spotLightMap: [], spotShadow: [],
    spotShadowMap: [], spotLightMatrix: [], rectArea: [], rectAreaLTC1: null, rectAreaLTC2: null, point: [], pointShadow: [], pointShadowMap: [],
    pointShadowMatrix: [], hemi: [], numSpotLightShadowsWithMaps: 0, numLightProbes: 0};
  for (let i = 0; i < 9; i++) state.probe.push(new T.Vector3());
  const vector3 = new T.Vector3(), matrix4 = new T.Matrix4(), matrix42 = new T.Matrix4();
  let signature = '';
  const shadowCache = new Map();
  const shadowUniformsFor = light => {
    let u = shadowCache.get(light.id);
    if (u) return u;
    u = {shadowIntensity: 1, shadowBias: 0, shadowNormalBias: 0, shadowRadius: 1, shadowMapSize: new T.Vector2()};
    if (light.type === 'PointLight') Object.assign(u, {shadowCameraNear: 1, shadowCameraFar: 1000});
    shadowCache.set(light.id, u);
    return u;
  };
  const shadowFields = (u, shadow) => { u.shadowIntensity = shadow.intensity; u.shadowBias = shadow.bias; u.shadowNormalBias = shadow.normalBias; u.shadowRadius = shadow.radius; };
  /** WebGLLights.setup: lights in source traversal order (sorted here, as r186 does). */
  function setup(lights) {
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < 9; i++) state.probe[i].set(0, 0, 0);
    let sunLength = 0, numSunShadows = 0, numSunShadowCascades = 0, directionalLength = 0, pointLength = 0, spotLength = 0, rectAreaLength = 0, hemiLength = 0;
    let numDirectionalShadows = 0, numPointShadows = 0, numSpotShadows = 0, numSpotMaps = 0, numSpotShadowsWithMaps = 0, numLightProbes = 0;
    // r186 sorts shadow-casting and textured lights first (stable sort).
    lights.sort((a, bb) => (bb.castShadow ? 2 : 0) - (a.castShadow ? 2 : 0) + (bb.map ? 1 : 0) - (a.map ? 1 : 0));
    for (const light of lights) {
      const color = light.color, intensity = light.intensity, distance = light.distance;
      let shadowMap = null;
      if (light.shadow && light.shadow.map) {
        // VSM: blurred RG color texture; other types: the depth texture.
        shadowMap = light.shadow.map.texture.format === T.RGFormat ? light.shadow.map.texture : light.shadow.map.depthTexture || light.shadow.map.texture;
      }
      if (light.isAmbientLight) { r += color.r * intensity; g += color.g * intensity; b += color.b * intensity; }
      else if (light.isLightProbe) { for (let j = 0; j < 9; j++) state.probe[j].addScaledVector(light.sh.coefficients[j], intensity); numLightProbes++; }
      else if (light.isSunLight) {
        const u = uniformsFor(light); u.color.copy(light.color).multiplyScalar(light.intensity);
        if (light.castShadow) {
          const shadow = light.shadow, su = shadowUniformsFor(light);
          shadowFields(su, shadow); su.shadowMapSize.copy(shadow.mapSize).multiply(shadow.getFrameExtents());
          state.sunShadow[numSunShadows] = su; state.sunShadowMap[numSunShadows] = shadowMap;
          const cascadeCount = shadow.getViewportCount();
          for (let j = 0; j < cascadeCount; j++) { state.sunShadowMatrix[numSunShadowCascades + j] = shadow.getMatrix(j); state.sunShadowCascade[numSunShadowCascades + j] = shadow._cascadeData[j]; }
          numSunShadowCascades += cascadeCount; numSunShadows++;
        }
        state.sun[sunLength++] = u;
      } else if (light.isDirectionalLight) {
        const u = uniformsFor(light); u.color.copy(light.color).multiplyScalar(light.intensity);
        if (light.castShadow) {
          const shadow = light.shadow, su = shadowUniformsFor(light);
          shadowFields(su, shadow); su.shadowMapSize = shadow.mapSize;
          state.directionalShadow[directionalLength] = su; state.directionalShadowMap[directionalLength] = shadowMap;
          state.directionalShadowMatrix[directionalLength] = light.shadow.matrix;
          numDirectionalShadows++;
        }
        state.directional[directionalLength++] = u;
      } else if (light.isSpotLight) {
        const u = uniformsFor(light);
        u.position.setFromMatrixPosition(light.matrixWorld); u.color.copy(color).multiplyScalar(intensity); u.distance = distance;
        u.coneCos = Math.cos(light.angle); u.penumbraCos = Math.cos(light.angle * (1 - light.penumbra)); u.decay = light.decay;
        state.spot[spotLength] = u;
        const shadow = light.shadow;
        if (light.map) {
          state.spotLightMap[numSpotMaps++] = light.map;
          shadow.updateMatrices(light);
          if (light.castShadow) numSpotShadowsWithMaps++;
        }
        state.spotLightMatrix[spotLength] = shadow.matrix;
        if (light.castShadow) {
          const su = shadowUniformsFor(light);
          shadowFields(su, shadow); su.shadowMapSize = shadow.mapSize;
          state.spotShadow[spotLength] = su; state.spotShadowMap[spotLength] = shadowMap;
          numSpotShadows++;
        }
        spotLength++;
      } else if (light.isRectAreaLight) {
        const u = uniformsFor(light);
        u.color.copy(color).multiplyScalar(intensity); u.halfWidth.set(light.width * 0.5, 0, 0); u.halfHeight.set(0, light.height * 0.5, 0);
        state.rectArea[rectAreaLength++] = u;
      } else if (light.isPointLight) {
        const u = uniformsFor(light); u.color.copy(light.color).multiplyScalar(light.intensity); u.distance = light.distance; u.decay = light.decay;
        if (light.castShadow) {
          const shadow = light.shadow, su = shadowUniformsFor(light);
          shadowFields(su, shadow); su.shadowMapSize = shadow.mapSize; su.shadowCameraNear = shadow.camera.near; su.shadowCameraFar = shadow.camera.far;
          state.pointShadow[pointLength] = su; state.pointShadowMap[pointLength] = shadowMap; state.pointShadowMatrix[pointLength] = light.shadow.matrix;
          numPointShadows++;
        }
        state.point[pointLength++] = u;
      } else if (light.isHemisphereLight) {
        const u = uniformsFor(light); u.skyColor.copy(light.color).multiplyScalar(intensity); u.groundColor.copy(light.groundColor).multiplyScalar(intensity);
        state.hemi[hemiLength++] = u;
      }
    }
    if (rectAreaLength > 0) {
      if (T.UniformsLib.LTC_FLOAT_1 === undefined) throw new Error('RectAreaLight needs RectAreaLightUniformsLib.init(), as in WebGLRenderer');
      const float = floatLinear() === true;
      state.rectAreaLTC1 = float ? T.UniformsLib.LTC_FLOAT_1 : T.UniformsLib.LTC_HALF_1;
      state.rectAreaLTC2 = float ? T.UniformsLib.LTC_FLOAT_2 : T.UniformsLib.LTC_HALF_2;
    }
    state.ambient[0] = r; state.ambient[1] = g; state.ambient[2] = b;
    state.sun.length = sunLength; state.directional.length = directionalLength; state.spot.length = spotLength; state.rectArea.length = rectAreaLength;
    state.point.length = pointLength; state.hemi.length = hemiLength;
    state.sunShadow.length = numSunShadows; state.sunShadowMap.length = numSunShadows;
    state.sunShadowMatrix.length = numSunShadowCascades; state.sunShadowCascade.length = numSunShadowCascades;
    state.directionalShadow.length = numDirectionalShadows; state.directionalShadowMap.length = numDirectionalShadows; state.directionalShadowMatrix.length = numDirectionalShadows;
    state.pointShadow.length = numPointShadows; state.pointShadowMap.length = numPointShadows; state.pointShadowMatrix.length = numPointShadows;
    state.spotShadow.length = numSpotShadows; state.spotShadowMap.length = numSpotShadows;
    state.spotLightMatrix.length = numSpotShadows + numSpotMaps - numSpotShadowsWithMaps; state.spotLightMap.length = numSpotMaps;
    state.numSpotLightShadowsWithMaps = numSpotShadowsWithMaps; state.numLightProbes = numLightProbes;
    const next = [sunLength, directionalLength, pointLength, spotLength, rectAreaLength, hemiLength, numSunShadows, numDirectionalShadows,
      numPointShadows, numSpotShadows, numSpotMaps, numLightProbes].join(',');
    if (next !== signature) { signature = next; state.version++; }
  }
  function setupView(lights, camera) {
    let sunLength = 0, directionalLength = 0, pointLength = 0, spotLength = 0, rectAreaLength = 0, hemiLength = 0;
    const viewMatrix = camera.matrixWorldInverse;
    for (const light of lights) {
      if (light.isSunLight) { const u = state.sun[sunLength++]; u.direction.setFromMatrixPosition(light.matrixWorld); u.direction.transformDirection(viewMatrix); }
      else if (light.isDirectionalLight) {
        const u = state.directional[directionalLength++];
        u.direction.setFromMatrixPosition(light.matrixWorld); vector3.setFromMatrixPosition(light.target.matrixWorld);
        u.direction.sub(vector3); u.direction.transformDirection(viewMatrix);
      } else if (light.isSpotLight) {
        const u = state.spot[spotLength++];
        u.position.setFromMatrixPosition(light.matrixWorld); u.position.applyMatrix4(viewMatrix);
        u.direction.setFromMatrixPosition(light.matrixWorld); vector3.setFromMatrixPosition(light.target.matrixWorld);
        u.direction.sub(vector3); u.direction.transformDirection(viewMatrix);
      } else if (light.isRectAreaLight) {
        const u = state.rectArea[rectAreaLength++];
        u.position.setFromMatrixPosition(light.matrixWorld); u.position.applyMatrix4(viewMatrix);
        matrix42.identity(); matrix4.copy(light.matrixWorld); matrix4.premultiply(viewMatrix); matrix42.extractRotation(matrix4);
        u.halfWidth.set(light.width * 0.5, 0, 0); u.halfHeight.set(0, light.height * 0.5, 0);
        u.halfWidth.applyMatrix4(matrix42); u.halfHeight.applyMatrix4(matrix42);
      } else if (light.isPointLight) { const u = state.point[pointLength++]; u.position.setFromMatrixPosition(light.matrixWorld); u.position.applyMatrix4(viewMatrix); }
      else if (light.isHemisphereLight) { const u = state.hemi[hemiLength++]; u.direction.setFromMatrixPosition(light.matrixWorld); u.direction.transformDirection(viewMatrix); }
    }
  }
  return {setup, setupView, state};
}

// ---- WebGLPrograms.getUniforms + WebGLMaterials ------------------------------
export function webglMaterialUniforms(T, material) {
  const id = SHADER_IDS[material.type];
  return id ? T.UniformsUtils.clone(T.ShaderLib[id].uniforms) : material.uniforms;
}
const _m1Cache = new WeakMap();
/** WebGLRenderer.getProgram light wiring + refreshFogUniforms + refreshMaterialUniforms.
 * ctx: {fog, lights (state), envMap, envMapRotation, pixelRatio, height, unlitColorSpace, environmentIntensity?} */
export function refreshWebGLMaterialUniforms(T, uniforms, material, ctx) {
  let scratch = _m1Cache.get(T);
  if (!scratch) {
    const flip = new T.Matrix3(); flip.set(-1, 0, 0, 0, 1, 0, 0, 0, 1);
    scratch = {m1: new T.Matrix4(), flip};
    _m1Cache.set(T, scratch);
  }
  const refreshTransform = (map, uniform) => { if (map.matrixAutoUpdate === true) map.updateMatrix(); uniform.value.copy(map.matrix); };
  if (materialNeedsLights(material) && uniforms.ambientLightColor) {
    const s = ctx.lights;
    uniforms.ambientLightColor.value = s.ambient; uniforms.lightProbe.value = s.probe; uniforms.sunLights.value = s.sun;
    uniforms.sunLightShadows.value = s.sunShadow; uniforms.directionalLights.value = s.directional; uniforms.directionalLightShadows.value = s.directionalShadow;
    uniforms.spotLights.value = s.spot; uniforms.spotLightShadows.value = s.spotShadow; uniforms.rectAreaLights.value = s.rectArea;
    uniforms.ltc_1.value = s.rectAreaLTC1; uniforms.ltc_2.value = s.rectAreaLTC2; uniforms.pointLights.value = s.point;
    uniforms.pointLightShadows.value = s.pointShadow; uniforms.hemisphereLights.value = s.hemi;
    uniforms.sunShadowMatrix.value = s.sunShadowMatrix; uniforms.sunShadowCascade.value = s.sunShadowCascade;
    uniforms.directionalShadowMatrix.value = s.directionalShadowMatrix; uniforms.spotLightMatrix.value = s.spotLightMatrix;
    uniforms.spotLightMap.value = s.spotLightMap; uniforms.pointShadowMatrix.value = s.pointShadowMatrix;
  }
  if (ctx.environmentIntensity !== undefined && uniforms.envMapIntensity) uniforms.envMapIntensity.value = ctx.environmentIntensity;
  if (ctx.fog && material.fog === true) {
    ctx.fog.color.getRGB(uniforms.fogColor.value, ctx.unlitColorSpace ?? T.ColorManagement.workingColorSpace);
    if (ctx.fog.isFog) { uniforms.fogNear.value = ctx.fog.near; uniforms.fogFar.value = ctx.fog.far; }
    else if (ctx.fog.isFogExp2) uniforms.fogDensity.value = ctx.fog.density;
  }
  const common = () => {
    const m = material;
    uniforms.opacity.value = m.opacity;
    if (m.color) uniforms.diffuse.value.copy(m.color);
    if (m.emissive) uniforms.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity);
    if (m.map) { uniforms.map.value = m.map; refreshTransform(m.map, uniforms.mapTransform); }
    if (m.alphaMap) { uniforms.alphaMap.value = m.alphaMap; refreshTransform(m.alphaMap, uniforms.alphaMapTransform); }
    if (m.bumpMap) { uniforms.bumpMap.value = m.bumpMap; refreshTransform(m.bumpMap, uniforms.bumpMapTransform); uniforms.bumpScale.value = m.bumpScale; if (m.side === T.BackSide) uniforms.bumpScale.value *= -1; }
    if (m.normalMap) { uniforms.normalMap.value = m.normalMap; refreshTransform(m.normalMap, uniforms.normalMapTransform); uniforms.normalScale.value.copy(m.normalScale); if (m.side === T.BackSide) uniforms.normalScale.value.negate(); }
    if (m.displacementMap) { uniforms.displacementMap.value = m.displacementMap; refreshTransform(m.displacementMap, uniforms.displacementMapTransform); uniforms.displacementScale.value = m.displacementScale; uniforms.displacementBias.value = m.displacementBias; }
    if (m.emissiveMap) { uniforms.emissiveMap.value = m.emissiveMap; refreshTransform(m.emissiveMap, uniforms.emissiveMapTransform); }
    if (m.specularMap) { uniforms.specularMap.value = m.specularMap; refreshTransform(m.specularMap, uniforms.specularMapTransform); }
    if (m.alphaTest > 0) uniforms.alphaTest.value = m.alphaTest;
    const envMap = ctx.envMap ?? null;
    if (envMap) {
      uniforms.envMap.value = envMap;
      // Orthonormal rotation: transpose in lieu of invert (upstream comment).
      uniforms.envMapRotation.value.setFromMatrix4(scratch.m1.makeRotationFromEuler(ctx.envMapRotation ?? m.envMapRotation)).transpose();
      if (envMap.isCubeTexture && envMap.isRenderTargetTexture === false) uniforms.envMapRotation.value.premultiply(scratch.flip);
      uniforms.reflectivity.value = m.reflectivity; uniforms.ior.value = m.ior; uniforms.refractionRatio.value = m.refractionRatio;
    }
    if (m.lightMap) { uniforms.lightMap.value = m.lightMap; uniforms.lightMapIntensity.value = m.lightMapIntensity; refreshTransform(m.lightMap, uniforms.lightMapTransform); }
    if (m.aoMap) { uniforms.aoMap.value = m.aoMap; uniforms.aoMapIntensity.value = m.aoMapIntensity; refreshTransform(m.aoMap, uniforms.aoMapTransform); }
  };
  const m = material;
  if (m.isMeshBasicMaterial) common();
  else if (m.isMeshLambertMaterial) { common(); if (m.envMap) uniforms.envMapIntensity.value = m.envMapIntensity; }
  else if (m.isMeshToonMaterial) { common(); if (m.gradientMap) uniforms.gradientMap.value = m.gradientMap; }
  else if (m.isMeshPhongMaterial) {
    common(); uniforms.specular.value.copy(m.specular); uniforms.shininess.value = Math.max(m.shininess, 1e-4);
    if (m.envMap) uniforms.envMapIntensity.value = m.envMapIntensity;
  } else if (m.isMeshStandardMaterial) {
    common();
    uniforms.metalness.value = m.metalness;
    if (m.metalnessMap) { uniforms.metalnessMap.value = m.metalnessMap; refreshTransform(m.metalnessMap, uniforms.metalnessMapTransform); }
    uniforms.roughness.value = m.roughness;
    if (m.roughnessMap) { uniforms.roughnessMap.value = m.roughnessMap; refreshTransform(m.roughnessMap, uniforms.roughnessMapTransform); }
    if (m.envMap) uniforms.envMapIntensity.value = m.envMapIntensity;
    if (m.isMeshPhysicalMaterial) refreshPhysical(T, uniforms, m, refreshTransform, ctx);
  } else if (m.isMeshMatcapMaterial) { common(); if (m.matcap) uniforms.matcap.value = m.matcap; }
  else if (m.isMeshDepthMaterial) common();
  else if (m.isMeshDistanceMaterial) {
    // refreshUniformsDistance: the point light this shadow pass renders (ctx.distanceLight).
    common();
    const light = ctx.distanceLight;
    uniforms.referencePosition.value.setFromMatrixPosition(light.matrixWorld);
    uniforms.nearDistance.value = light.shadow.camera.near; uniforms.farDistance.value = light.shadow.camera.far;
  }
  else if (m.isMeshNormalMaterial) common();
  else if (m.isLineBasicMaterial) {
    uniforms.diffuse.value.copy(m.color); uniforms.opacity.value = m.opacity;
    if (m.map) { uniforms.map.value = m.map; refreshTransform(m.map, uniforms.mapTransform); }
    if (m.isLineDashedMaterial) { uniforms.dashSize.value = m.dashSize; uniforms.totalSize.value = m.dashSize + m.gapSize; uniforms.scale.value = m.scale; }
  } else if (m.isPointsMaterial) {
    uniforms.diffuse.value.copy(m.color); uniforms.opacity.value = m.opacity;
    uniforms.size.value = m.size * ctx.pixelRatio; uniforms.scale.value = ctx.height * 0.5;
    if (m.map) { uniforms.map.value = m.map; refreshTransform(m.map, uniforms.uvTransform); }
    if (m.alphaMap) { uniforms.alphaMap.value = m.alphaMap; refreshTransform(m.alphaMap, uniforms.alphaMapTransform); }
    if (m.alphaTest > 0) uniforms.alphaTest.value = m.alphaTest;
  } else if (m.isShadowMaterial) { uniforms.color.value.copy(m.color); uniforms.opacity.value = m.opacity; }
  else if (m.isSpriteMaterial) {
    uniforms.diffuse.value.copy(m.color); uniforms.opacity.value = m.opacity; uniforms.rotation.value = m.rotation;
    if (m.map) { uniforms.map.value = m.map; refreshTransform(m.map, uniforms.mapTransform); }
    if (m.alphaMap) { uniforms.alphaMap.value = m.alphaMap; refreshTransform(m.alphaMap, uniforms.alphaMapTransform); }
    if (m.alphaTest > 0) uniforms.alphaTest.value = m.alphaTest;
  }
}
function refreshPhysical(T, u, m, refreshTransform, ctx = {}) {
  u.ior.value = m.ior;
  if (m.sheen > 0) {
    u.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen); u.sheenRoughness.value = m.sheenRoughness;
    if (m.sheenColorMap) { u.sheenColorMap.value = m.sheenColorMap; refreshTransform(m.sheenColorMap, u.sheenColorMapTransform); }
    if (m.sheenRoughnessMap) { u.sheenRoughnessMap.value = m.sheenRoughnessMap; refreshTransform(m.sheenRoughnessMap, u.sheenRoughnessMapTransform); }
  }
  if (m.clearcoat > 0) {
    u.clearcoat.value = m.clearcoat; u.clearcoatRoughness.value = m.clearcoatRoughness;
    if (m.clearcoatMap) { u.clearcoatMap.value = m.clearcoatMap; refreshTransform(m.clearcoatMap, u.clearcoatMapTransform); }
    if (m.clearcoatRoughnessMap) { u.clearcoatRoughnessMap.value = m.clearcoatRoughnessMap; refreshTransform(m.clearcoatRoughnessMap, u.clearcoatRoughnessMapTransform); }
    if (m.clearcoatNormalMap) {
      u.clearcoatNormalMap.value = m.clearcoatNormalMap; refreshTransform(m.clearcoatNormalMap, u.clearcoatNormalMapTransform);
      u.clearcoatNormalScale.value.copy(m.clearcoatNormalScale); if (m.side === T.BackSide) u.clearcoatNormalScale.value.negate();
    }
  }
  if (m.dispersion > 0) u.dispersion.value = m.dispersion;
  if (m.retroreflectivity > 0) u.retroreflectivity.value = m.retroreflectivity;
  if (m.iridescence > 0) {
    u.iridescence.value = m.iridescence; u.iridescenceIOR.value = m.iridescenceIOR;
    u.iridescenceThicknessMinimum.value = m.iridescenceThicknessRange[0]; u.iridescenceThicknessMaximum.value = m.iridescenceThicknessRange[1];
    if (m.iridescenceMap) { u.iridescenceMap.value = m.iridescenceMap; refreshTransform(m.iridescenceMap, u.iridescenceMapTransform); }
    if (m.iridescenceThicknessMap) { u.iridescenceThicknessMap.value = m.iridescenceThicknessMap; refreshTransform(m.iridescenceThicknessMap, u.iridescenceThicknessMapTransform); }
  }
  if (m.anisotropy > 0) {
    u.anisotropyVector.value.set(m.anisotropy * Math.cos(m.anisotropyRotation), m.anisotropy * Math.sin(m.anisotropyRotation));
    if (m.anisotropyMap) { u.anisotropyMap.value = m.anisotropyMap; refreshTransform(m.anisotropyMap, u.anisotropyMapTransform); }
  }
  if (m.transmission > 0) {
    // refreshUniformsPhysical: the camera's transmission render target (ctx.transmissionRenderTarget).
    u.transmission.value = m.transmission;
    const target = ctx.transmissionRenderTarget;
    if (target) { u.transmissionSamplerMap.value = target.texture; u.transmissionSamplerSize.value.set(target.width, target.height); }
    if (m.transmissionMap) { u.transmissionMap.value = m.transmissionMap; refreshTransform(m.transmissionMap, u.transmissionMapTransform); }
    u.thickness.value = m.thickness;
    if (m.thicknessMap) { u.thicknessMap.value = m.thicknessMap; refreshTransform(m.thicknessMap, u.thicknessMapTransform); }
    u.attenuationDistance.value = m.attenuationDistance; u.attenuationColor.value.copy(m.attenuationColor);
  }
  u.specularIntensity.value = m.specularIntensity; u.specularColor.value.copy(m.specularColor);
  if (m.specularColorMap) { u.specularColorMap.value = m.specularColorMap; refreshTransform(m.specularColorMap, u.specularColorMapTransform); }
  if (m.specularIntensityMap) { u.specularIntensityMap.value = m.specularIntensityMap; refreshTransform(m.specularIntensityMap, u.specularIntensityMapTransform); }
}
