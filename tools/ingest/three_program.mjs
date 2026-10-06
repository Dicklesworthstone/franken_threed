/** Source ShaderMaterial / RawShaderMaterial programs on the new WebGPU backend.
 *
 * threeProgramSources() reproduces r186 WebGLProgram's source assembly for the
 * parameters a ShaderMaterial can reach (WebGLPrograms.getParameters): precision
 * block, SHADER_TYPE/NAME, material.defines, instancing/fog/vertex-attribute
 * defines, built-in uniform/attribute declarations, ESSL 3.00 conversion macros,
 * #include resolution against the live ShaderChunk, light/clipping count
 * substitution and `#pragma unroll_loop` unrolling, tone-mapping and output
 * encoding functions. The resulting GLSL is what WebGLRenderer would hand to the
 * driver; essl_wgsl.mjs compiles it.
 *
 * threeProgramRaster() maps material state with WebGLState semantics (blend
 * table, BACK culling with frontFace flips for BackSide / negative determinant,
 * depth/color masks), because ShaderMaterial is a WebGLRenderer feature.
 *
 * packThreeProgramUniforms() writes current uniform values (built-ins from the
 * camera/object, material.uniforms otherwise; undeclared values stay zero, as GL
 * leaves them) into the reflected uniform-buffer layout every frame.
 *
 * Not admitted yet (explicit errors): material.lights, skinning, morph targets,
 * batching, clipping planes, uniformsGroups, logarithmic/reversed depth, custom
 * extensions, onBeforeCompile hooks. No performance claim.
 */
import {compileEsslProgram} from './essl_wgsl.mjs';
import {createGpuProgramGeometry, programGeometrySnapshot} from './gpu_buffer_geometry.mjs';

export class ThreeProgramError extends Error {
  constructor(code, message) { super(`THREE_PROGRAM_${code}: ${message}`); this.name = 'ThreeProgramError'; this.code = 'THREE_PROGRAM_' + code; }
}
const fail = (code, message) => { throw new ThreeProgramError(code, message); };

const PRECISION_TYPES = ['float', 'int', 'sampler2D', 'samplerCube', 'sampler3D', 'sampler2DArray', 'sampler2DShadow', 'samplerCubeShadow',
  'sampler2DArrayShadow', 'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray', 'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray'];
function precisionBlock(p) {
  return PRECISION_TYPES.map(t => `precision ${p} ${t};`).join('\n') + `\n#define ${{highp: 'HIGH', mediump: 'MEDIUM', lowp: 'LOW'}[p]}_PRECISION`;
}
function defineLines(defines) {
  const out = [];
  for (const name in defines) { const v = defines[name]; if (v !== false) out.push(`#define ${name} ${v}`); }
  return out.join('\n');
}
const nonEmpty = s => s !== '';
const includePattern = /^[ \t]*#include +<([\w\d./]+)>/gm;
function resolveIncludes(T, s) {
  return s.replace(includePattern, (m, name) => {
    const chunk = T.ShaderChunk[name];
    if (chunk === undefined) fail('SOURCE', `Can not resolve #include <${name}>`);
    return resolveIncludes(T, chunk);
  });
}
const unrollPattern = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;
const unrollLoops = s => s.replace(unrollPattern, (m, start, end, snippet) => {
  let out = '';
  for (let i = parseInt(start); i < parseInt(end); i++) out += snippet.replace(/\[\s*i\s*\]/g, '[ ' + i + ' ]').replace(/UNROLLED_LOOP_INDEX/g, i);
  return out;
});
// No admitted ShaderMaterial uses source lights or clipping yet: the counts are zero.
const replaceCounts = s => s.replace(/NUM_SUN_LIGHTS|NUM_DIR_LIGHTS|NUM_SPOT_LIGHTS|NUM_SPOT_LIGHT_MAPS|NUM_SPOT_LIGHT_COORDS|NUM_RECT_AREA_LIGHTS|NUM_POINT_LIGHTS|NUM_HEMI_LIGHTS|NUM_SUN_LIGHT_SHADOWS|NUM_DIR_LIGHT_SHADOWS|NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS|NUM_SPOT_LIGHT_SHADOWS|NUM_POINT_LIGHT_SHADOWS|NUM_CLIPPING_PLANES|UNION_CLIPPING_PLANES/g, '0');

function encodingFunction(T, name, colorSpace) {
  const m = new T.Matrix3();
  T.ColorManagement._getMatrix(m, T.ColorManagement.workingColorSpace, colorSpace);
  const matrix = `mat3( ${m.elements.map(v => v.toFixed(4))} )`;
  const transfer = T.ColorManagement.getTransfer(colorSpace) === T.SRGBTransfer ? 'sRGBTransferOETF' : 'LinearTransferOETF';
  return [`vec4 ${name}( vec4 value ) {`, `\treturn ${transfer}( vec4( value.rgb * ${matrix}, value.a ) );`, '}'].join('\n');
}
function toneMappingFunction(T, toneMapping) {
  const names = {[T.LinearToneMapping]: 'Linear', [T.ReinhardToneMapping]: 'Reinhard', [T.CineonToneMapping]: 'Cineon', [T.ACESFilmicToneMapping]: 'ACESFilmic',
    [T.AgXToneMapping]: 'AgX', [T.NeutralToneMapping]: 'Neutral', [T.CustomToneMapping]: 'Custom'};
  return `vec3 toneMapping( vec3 color ) { return ${names[toneMapping] ?? 'Linear'}ToneMapping( color ); }`;
}
function luminanceFunction(T) {
  const v = new T.Vector3(); T.ColorManagement.getLuminanceCoefficients(v);
  return ['float luminance( const in vec3 rgb ) {', `\tconst vec3 weights = vec3( ${v.x.toFixed(4)}, ${v.y.toFixed(4)}, ${v.z.toFixed(4)} );`, '\treturn dot( weights, rgb );', '}'].join('\n');
}

/** Admission of a source program material/object pair on this backend. */
export function inspectThreeProgram(T, material, object) {
  if (!material?.isShaderMaterial) fail('SOURCE', 'Expected a ShaderMaterial or RawShaderMaterial');
  if (typeof material.vertexShader !== 'string' || typeof material.fragmentShader !== 'string') fail('SOURCE', 'Program sources must be strings');
  if (material.lights) fail('LIGHTS', 'ShaderMaterial with lights: true is not admitted yet');
  if (object.isSkinnedMesh || Object.values(object.geometry?.morphAttributes ?? {}).some(a => a?.length) || object.isBatchedMesh)
    fail('OBJECT', 'Skinned, morphed and batched ShaderMaterial objects are not admitted yet');
  if (object.isInstancedMesh && object.morphTexture != null) fail('OBJECT', 'Instanced morph textures are not admitted');
  if (material.clipping && material.clippingPlanes?.length) fail('CLIPPING', 'ShaderMaterial clipping planes are not admitted yet');
  if (material.uniformsGroups?.length) fail('UNIFORMS', 'Uniform buffer groups are not admitted yet');
  if (material.onBeforeCompile !== T.Material.prototype.onBeforeCompile || material.onBeforeRender !== T.Material.prototype.onBeforeRender)
    fail('HOOK', 'Program hooks require their original component');
  if (material.extensions?.clipCullDistance || material.extensions?.multiDraw) fail('EXTENSION', 'Program extensions are not admitted');
  if (material.wireframe) fail('MATERIAL', 'Wireframe ShaderMaterial is not admitted yet');
  if (material.alphaToCoverage) fail('MATERIAL', 'alphaToCoverage is not admitted');
  if (material.stencilWrite) fail('MATERIAL', 'Stencil ShaderMaterial is not admitted yet');
}

/** r186 WebGLProgram source assembly for ShaderMaterial/RawShaderMaterial. */
export function threeProgramSources(T, material, object, {fog = null, toneMapping, outputColorSpace, precision = 'highp', shadowMapEnabled = false, shadowMapType = T.PCFShadowMap, side = material.side} = {}) {
  inspectThreeProgram(T, material, object);
  const geometry = object.geometry, attributes = geometry.attributes;
  const p = {
    shaderType: material.type, shaderName: material.name, defines: material.defines, glslVersion: material.glslVersion,
    precision: material.precision ?? precision,
    instancing: object.isInstancedMesh === true, instancingColor: object.isInstancedMesh === true && object.instanceColor !== null,
    useFog: material.fog === true, fog: !!fog, fogExp2: !!fog && fog.isFogExp2 === true,
    vertexNormals: !!attributes.normal, vertexColors: material.vertexColors,
    vertexAlphas: material.vertexColors === true && !!attributes.color && attributes.color.itemSize === 4,
    vertexUv1s: !!attributes.uv1, vertexUv2s: !!attributes.uv2, vertexUv3s: !!attributes.uv3,
    flatShading: material.wireframe === false && material.flatShading === true,
    doubleSided: side === T.DoubleSide, flipSided: side === T.BackSide,
    shadowMapEnabled, shadowMapType,
    alphaTest: material.alphaTest > 0, dithering: material.dithering, premultipliedAlpha: material.premultipliedAlpha,
    opaque: material.transparent === false && material.blending === T.NormalBlending && material.alphaToCoverage === false,
    toneMapping: material.toneMapped ? toneMapping : T.NoToneMapping,
    outputColorSpace, useDepthPacking: material.depthPacking >= 0, depthPacking: material.depthPacking || 0,
  };
  const customDefines = defineLines(p.defines);
  let prefixVertex, prefixFragment;
  let versionString = p.glslVersion ? '#version ' + p.glslVersion + '\n' : '';
  if (material.isRawShaderMaterial) {
    prefixVertex = [`#define SHADER_TYPE ${p.shaderType}`, `#define SHADER_NAME ${p.shaderName}`, customDefines].filter(nonEmpty).join('\n');
    if (prefixVertex.length) prefixVertex += '\n';
    prefixFragment = prefixVertex;
  } else {
    const shadowDefine = {[T.PCFShadowMap]: 'SHADOWMAP_TYPE_PCF', [T.VSMShadowMap]: 'SHADOWMAP_TYPE_VSM'}[p.shadowMapType] || 'SHADOWMAP_TYPE_BASIC';
    prefixVertex = [precisionBlock(p.precision), `#define SHADER_TYPE ${p.shaderType}`, `#define SHADER_NAME ${p.shaderName}`, customDefines,
      p.instancing ? '#define USE_INSTANCING' : '', p.instancingColor ? '#define USE_INSTANCING_COLOR' : '',
      p.useFog && p.fog ? '#define USE_FOG' : '', p.useFog && p.fogExp2 ? '#define FOG_EXP2' : '',
      p.vertexNormals ? '#define HAS_NORMAL' : '', p.vertexColors ? '#define USE_COLOR' : '', p.vertexAlphas ? '#define USE_COLOR_ALPHA' : '',
      p.vertexUv1s ? '#define USE_UV1' : '', p.vertexUv2s ? '#define USE_UV2' : '', p.vertexUv3s ? '#define USE_UV3' : '',
      p.flatShading ? '#define FLAT_SHADED' : '', p.doubleSided ? '#define DOUBLE_SIDED' : '', p.flipSided ? '#define FLIP_SIDED' : '',
      p.shadowMapEnabled ? '#define USE_SHADOWMAP' : '', p.shadowMapEnabled ? '#define ' + shadowDefine : '',
      'uniform mat4 modelMatrix;', 'uniform mat4 modelViewMatrix;', 'uniform mat4 projectionMatrix;', 'uniform mat4 viewMatrix;',
      'uniform mat3 normalMatrix;', 'uniform vec3 cameraPosition;', 'uniform bool isOrthographic;',
      '#ifdef USE_INSTANCING', '\tattribute mat4 instanceMatrix;', '#endif', '#ifdef USE_INSTANCING_COLOR', '\tattribute vec3 instanceColor;', '#endif',
      '#ifdef USE_INSTANCING_MORPH', '\tuniform sampler2D morphTexture;', '#endif',
      'attribute vec3 position;', 'attribute vec3 normal;', 'attribute vec2 uv;',
      '#ifdef USE_UV1', '\tattribute vec2 uv1;', '#endif', '#ifdef USE_UV2', '\tattribute vec2 uv2;', '#endif', '#ifdef USE_UV3', '\tattribute vec2 uv3;', '#endif',
      '#ifdef USE_TANGENT', '\tattribute vec4 tangent;', '#endif',
      '#if defined( USE_COLOR_ALPHA )', '\tattribute vec4 color;', '#elif defined( USE_COLOR )', '\tattribute vec3 color;', '#endif',
      '#ifdef USE_SKINNING', '\tattribute vec4 skinIndex;', '\tattribute vec4 skinWeight;', '#endif', '\n'].filter(nonEmpty).join('\n');
    prefixFragment = [precisionBlock(p.precision), `#define SHADER_TYPE ${p.shaderType}`, `#define SHADER_NAME ${p.shaderName}`, customDefines,
      p.useFog && p.fog ? '#define USE_FOG' : '', p.useFog && p.fogExp2 ? '#define FOG_EXP2' : '',
      p.alphaTest ? '#define USE_ALPHATEST' : '',
      p.vertexColors || p.instancingColor ? '#define USE_COLOR' : '', p.vertexAlphas ? '#define USE_COLOR_ALPHA' : '',
      p.vertexUv1s ? '#define USE_UV1' : '', p.vertexUv2s ? '#define USE_UV2' : '', p.vertexUv3s ? '#define USE_UV3' : '',
      p.flatShading ? '#define FLAT_SHADED' : '', p.doubleSided ? '#define DOUBLE_SIDED' : '', p.flipSided ? '#define FLIP_SIDED' : '',
      p.shadowMapEnabled ? '#define USE_SHADOWMAP' : '', p.shadowMapEnabled ? '#define ' + shadowDefine : '',
      p.premultipliedAlpha ? '#define PREMULTIPLIED_ALPHA' : '',
      'uniform mat4 viewMatrix;', 'uniform vec3 cameraPosition;', 'uniform bool isOrthographic;',
      p.toneMapping !== T.NoToneMapping ? '#define TONE_MAPPING' : '',
      p.toneMapping !== T.NoToneMapping ? T.ShaderChunk.tonemapping_pars_fragment : '',
      p.toneMapping !== T.NoToneMapping ? toneMappingFunction(T, p.toneMapping) : '',
      p.dithering ? '#define DITHERING' : '', p.opaque ? '#define OPAQUE' : '',
      T.ShaderChunk.colorspace_pars_fragment, encodingFunction(T, 'linearToOutputTexel', p.outputColorSpace), luminanceFunction(T),
      p.useDepthPacking ? '#define DEPTH_PACKING ' + p.depthPacking : '', '\n'].filter(nonEmpty).join('\n');
  }
  let vertexShader = unrollLoops(replaceCounts(resolveIncludes(T, material.vertexShader)));
  let fragmentShader = unrollLoops(replaceCounts(resolveIncludes(T, material.fragmentShader)));
  if (material.isRawShaderMaterial !== true) {
    versionString = '#version 300 es\n';
    prefixVertex = ['#define attribute in', '#define varying out', '#define texture2D texture'].join('\n') + '\n' + prefixVertex;
    prefixFragment = ['#define varying in',
      p.glslVersion === T.GLSL3 ? '' : 'layout(location = 0) out highp vec4 pc_fragColor;',
      p.glslVersion === T.GLSL3 ? '' : '#define gl_FragColor pc_fragColor',
      '#define gl_FragDepthEXT gl_FragDepth', '#define texture2D texture', '#define textureCube texture', '#define texture2DProj textureProj',
      '#define texture2DLodEXT textureLod', '#define texture2DProjLodEXT textureProjLod', '#define textureCubeLodEXT textureLod',
      '#define texture2DGradEXT textureGrad', '#define texture2DProjGradEXT textureProjGrad', '#define textureCubeGradEXT textureGrad'].join('\n') + '\n' + prefixFragment;
  }
  const vertex = versionString + prefixVertex + vertexShader, fragment = versionString + prefixFragment + fragmentShader;
  return {vertex, fragment, key: vertex + '\u0000' + fragment, parameters: p};
}

/** WebGLState.setMaterial raster state as WebGPU pipeline fragments. */
export function threeProgramRaster(T, m, {frontFaceCW = false, topology = 'triangles', side = m.side} = {}) {
  const factor = new Map([[T.ZeroFactor, 'zero'], [T.OneFactor, 'one'], [T.SrcColorFactor, 'src'], [T.OneMinusSrcColorFactor, 'one-minus-src'],
    [T.SrcAlphaFactor, 'src-alpha'], [T.OneMinusSrcAlphaFactor, 'one-minus-src-alpha'], [T.DstAlphaFactor, 'dst-alpha'], [T.OneMinusDstAlphaFactor, 'one-minus-dst-alpha'],
    [T.DstColorFactor, 'dst'], [T.OneMinusDstColorFactor, 'one-minus-dst'], [T.SrcAlphaSaturateFactor, 'src-alpha-saturated'],
    [T.ConstantColorFactor, 'constant'], [T.OneMinusConstantColorFactor, 'one-minus-constant'], [T.ConstantAlphaFactor, 'constant'], [T.OneMinusConstantAlphaFactor, 'one-minus-constant']]);
  const equation = new Map([[T.AddEquation, 'add'], [T.SubtractEquation, 'subtract'], [T.ReverseSubtractEquation, 'reverse-subtract'], [T.MinEquation, 'min'], [T.MaxEquation, 'max']]);
  const set = (src, dst, srcA, dstA) => ({color: {operation: 'add', srcFactor: src, dstFactor: dst}, alpha: {operation: 'add', srcFactor: srcA, dstFactor: dstA}});
  let blend = null, blendConstant = null;
  if (!(m.blending === T.NormalBlending && m.transparent === false) && m.blending !== T.NoBlending) {
    const pre = m.premultipliedAlpha;
    if (m.blending === T.NormalBlending) blend = pre ? set('one', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha') : set('src-alpha', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha');
    else if (m.blending === T.AdditiveBlending) blend = pre ? set('one', 'one', 'one', 'one') : set('src-alpha', 'one', 'one', 'one');
    else if (m.blending === T.SubtractiveBlending && pre) blend = set('zero', 'one-minus-src', 'zero', 'one');
    else if (m.blending === T.MultiplyBlending && pre) blend = set('dst', 'one-minus-src-alpha', 'zero', 'one');
    else if (m.blending === T.CustomBlending) {
      const f = x => { const v = factor.get(x); if (!v) fail('MATERIAL', 'Unsupported blend factor'); return v; };
      const e = x => { const v = equation.get(x); if (!v) fail('MATERIAL', 'Unsupported blend equation'); return v; };
      const eqA = m.blendEquationAlpha || m.blendEquation, srcA = m.blendSrcAlpha || m.blendSrc, dstA = m.blendDstAlpha || m.blendDst;
      const comp = (op, s, d) => (op === 'min' || op === 'max') ? {operation: op, srcFactor: 'one', dstFactor: 'one'} : {operation: op, srcFactor: f(s), dstFactor: f(d)};
      blend = {color: comp(e(m.blendEquation), m.blendSrc, m.blendDst), alpha: comp(e(eqA), srcA, dstA)};
      // WebGL's single blend color: constant alpha factors read its alpha.
      const alphaOnly = [m.blendSrc, m.blendDst, srcA, dstA].some(x => x === T.ConstantAlphaFactor || x === T.OneMinusConstantAlphaFactor);
      const colorOnly = [m.blendSrc, m.blendDst, srcA, dstA].some(x => x === T.ConstantColorFactor || x === T.OneMinusConstantColorFactor);
      if (alphaOnly && colorOnly) fail('MATERIAL', 'Mixed constant-color and constant-alpha factors are not expressible in WebGPU');
      if (alphaOnly) blendConstant = [m.blendAlpha, m.blendAlpha, m.blendAlpha, m.blendAlpha];
      else if (colorOnly) blendConstant = [m.blendColor.r, m.blendColor.g, m.blendColor.b, m.blendAlpha];
    } else fail('MATERIAL', `Blending ${m.blending} with premultipliedAlpha=${pre} is a WebGLState error, not admitted`);
  }
  const DEPTH = {[T.NeverDepth]: 'never', [T.AlwaysDepth]: 'always', [T.LessDepth]: 'less', [T.LessEqualDepth]: 'less-equal',
    [T.EqualDepth]: 'equal', [T.GreaterEqualDepth]: 'greater-equal', [T.GreaterDepth]: 'greater', [T.NotEqualDepth]: 'not-equal'};
  if (!DEPTH[m.depthFunc]) fail('MATERIAL', 'Unsupported depth function');
  let flip = side === T.BackSide;
  if (frontFaceCW) flip = !flip;
  const triangles = topology === 'triangles';
  return {
    blend, blendConstant, writeMask: m.colorWrite ? 0xf : 0,
    depthCompare: m.depthTest ? DEPTH[m.depthFunc] : 'always', depthWriteEnabled: m.depthTest ? m.depthWrite : m.depthWrite,
    cullMode: side === T.DoubleSide || !triangles ? 'none' : 'back', frontFace: flip ? 'cw' : 'ccw',
    depthBias: m.polygonOffset && triangles ? m.polygonOffsetUnits : 0, depthBiasSlopeScale: m.polygonOffset && triangles ? m.polygonOffsetFactor : 0,
  };
}

// ---- uniform values ---------------------------------------------------------
const BUILTIN = new Set(['modelMatrix', 'modelViewMatrix', 'projectionMatrix', 'viewMatrix', 'normalMatrix', 'cameraPosition', 'isOrthographic', 'toneMappingExposure']);

/** Write one frame's uniform values for one draw. Returns textures to bind. */
export function packThreeProgramUniforms(T, reflection, material, object, camera, bytes, {toneMappingExposure = 1, fog = null, targetSize = null} = {}) {
  bytes.fill(0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (material.fog === true && fog) refreshFog(T, material.uniforms, fog);
  const cameraPosition = new T.Vector3().setFromMatrixPosition(camera.matrixWorld);
  const builtin = {
    modelMatrix: object.matrixWorld, modelViewMatrix: object.modelViewMatrix, projectionMatrix: camera.projectionMatrix,
    viewMatrix: camera.matrixWorldInverse, normalMatrix: object.normalMatrix, cameraPosition, isOrthographic: camera.isOrthographicCamera === true,
    toneMappingExposure,
  };
  for (const u of reflection.uniforms) {
    const value = BUILTIN.has(u.name) ? builtin[u.name] : material.uniforms?.[u.name]?.value;
    if (value === undefined || value === null) continue;
    write(view, u.node, value, u.name);
  }
  if (reflection.targetOffset !== null) {
    if (!targetSize || targetSize.length !== 4) fail('FRAME', 'gl_FragCoord and point sprites need the framebuffer and viewport size');
    for (let i = 0; i < 4; i++) view.setFloat32(reflection.targetOffset + 4 * i, targetSize[i], true);
  }
  return reflection.textures.map(t => {
    const v = material.uniforms?.[t.name]?.value;
    return t.element === null ? v ?? null : v?.[t.element] ?? null;
  });
}
function refreshFog(T, uniforms, fog) {
  if (!uniforms) return;
  uniforms.fogColor?.value?.copy?.(fog.color);
  if (fog.isFog) { if (uniforms.fogNear) uniforms.fogNear.value = fog.near; if (uniforms.fogFar) uniforms.fogFar.value = fog.far; }
  else if (fog.isFogExp2 && uniforms.fogDensity) uniforms.fogDensity.value = fog.density;
}
function components(value, n, name) {
  if (typeof value === 'number' || typeof value === 'boolean') return [Number(value)];
  if (value.isColor) return n === 4 ? [value.r, value.g, value.b, 1] : [value.r, value.g, value.b];
  if (value.isVector2) return [value.x, value.y];
  if (value.isVector3) return [value.x, value.y, value.z];
  if (value.isVector4 || value.isQuaternion) return [value.x, value.y, value.z, value.w];
  if (value.isMatrix3 || value.isMatrix4 || value.isMatrix2) return value.elements;
  if (Array.isArray(value) || ArrayBuffer.isView(value)) return value;
  fail('VALUE', `Unsupported value for uniform ${name}`);
}
function writeScalar(view, offset, s, v) {
  if (s === 'float') view.setFloat32(offset, v, true);
  else if (s === 'int') view.setInt32(offset, v, true);
  else view.setUint32(offset, s === 'bool' ? (v ? 1 : 0) : v >>> 0, true);
}
function write(view, node, value, name, base = 0) {
  switch (node.k) {
    case 'num': {
      const list = components(value, node.n, name);
      for (let i = 0; i < node.n; i++) writeScalar(view, base + node.offset + i * 4, node.s, list[i] ?? 0);
      return;
    }
    case 'mat': {
      const e = components(value, node.c * node.r, name);
      for (let c = 0; c < node.c; c++) for (let r = 0; r < node.r; r++) view.setFloat32(base + node.offset + c * node.colStride + r * 4, e[c * node.r + r] ?? 0, true);
      return;
    }
    case 'array': {
      // Arrays: one value per element, or a flat numeric array (WebGLUniforms flatten).
      const elementValues = Array.isArray(value) && value.length && typeof value[0] === 'object' ? value : null;
      if (elementValues) { for (let i = 0; i < Math.min(node.n, elementValues.length); i++) write(view, node.elem, elementValues[i], name, base + node.offset + i * node.stride); return; }
      const flat = components(value, 0, name), per = node.elem.k === 'num' ? node.elem.n : node.elem.k === 'mat' ? node.elem.c * node.elem.r : null;
      if (per === null) fail('VALUE', `Uniform ${name} needs one object per struct element`);
      for (let i = 0; i < node.n; i++) {
        const slice = Array.prototype.slice.call(flat, i * per, i * per + per);
        if (!slice.length) break;
        write(view, node.elem, per === 1 ? slice[0] : slice, name, base + node.offset + i * node.stride);
      }
      return;
    }
    case 'struct':
      for (const f of node.fields) if (value[f.name] !== undefined && value[f.name] !== null) write(view, f.node, value[f.name], `${name}.${f.name}`, base);
      return;
  }
}

/** Injectable program support for the source-scene bridge (three_scene.mjs).
 * state() reports the renderer-level inputs of WebGLPrograms.getParameters:
 * {toneMapping, toneMappingExposure, outputColorSpace}. Compiled programs are
 * cached by their exact assembled source text (bounded). */
export function createThreeProgramSupport({three: T, state, maxPrograms = 256, maxPointSize = 1024}) {
  const compiled = new Map();
  function compile(material, object, {fog = null, side = material.side} = {}) {
    const s = state();
    const sources = threeProgramSources(T, material, object, {fog, side, toneMapping: s.toneMapping, outputColorSpace: s.outputColorSpace});
    // GL rasterizes Points as gl_PointSize squares: compile the point-sprite form.
    const points = object.isPoints === true, key = sources.key + (points ? '\u0000points' : '');
    let entry = compiled.get(key);
    if (!entry) {
      const program = compileEsslProgram(sources.vertex, sources.fragment, {points, maxPointSize});
      entry = {key, program, attributesKey: JSON.stringify(program.reflection.attributes)};
      if (compiled.size >= maxPrograms) compiled.delete(compiled.keys().next().value);
      compiled.set(key, entry);
    }
    return entry;
  }
  return Object.freeze({
    compile,
    raster: (material, options) => threeProgramRaster(T, material, options),
    pack: (reflection, material, object, camera, bytes, options) =>
      packThreeProgramUniforms(T, reflection, material, object, camera, bytes, {toneMappingExposure: state().toneMappingExposure ?? 1, ...options}),
    createGeometry: (device, source, attributes, options) => createGpuProgramGeometry(device, source, attributes, options),
    geometrySnapshot: (gpu, device) => programGeometrySnapshot(gpu, device),
  });
}
