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
import {SHADER_IDS, webglParameters, webglProgramSources, webglLights, webglMaterialUniforms, refreshWebGLMaterialUniforms, materialNeedsLights} from './three_webgl_program.mjs';
import {createGpuProgramGeometry, programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {animationDfgHalves} from './animation_dfg.mjs';

export class ThreeProgramError extends Error {
  constructor(code, message) { super(`THREE_PROGRAM_${code}: ${message}`); this.name = 'ThreeProgramError'; this.code = 'THREE_PROGRAM_' + code; }
}
const fail = (code, message) => { throw new ThreeProgramError(code, message); };

/** Admission of a source program material/object pair on this backend. */
export function inspectThreeProgram(T, material, object) {
  const builtin = SHADER_IDS[material?.type] !== undefined;
  if (!material?.isShaderMaterial && !builtin) fail('SOURCE', 'Expected a ShaderMaterial, RawShaderMaterial or ShaderLib material');
  if (material.isShaderMaterial && (typeof material.vertexShader !== 'string' || typeof material.fragmentShader !== 'string')) fail('SOURCE', 'Program sources must be strings');
  if (object.isBatchedMesh) fail('OBJECT', 'Batched program objects are not admitted yet');
  if (object.isInstancedMesh && object.morphTexture != null) fail('OBJECT', 'Instanced morph textures are not admitted yet');
  if (object.isSkinnedMesh && !(object.skeleton instanceof T.Skeleton)) fail('OBJECT', 'Skinned program objects need their skeleton');
  if (material.uniformsGroups?.length) fail('UNIFORMS', 'Uniform buffer groups are not admitted yet');
  if (material.onBeforeCompile !== T.Material.prototype.onBeforeCompile || material.onBeforeRender !== T.Material.prototype.onBeforeRender)
    fail('HOOK', 'Program hooks require their original component');
  if (material.extensions?.clipCullDistance || material.extensions?.multiDraw) fail('EXTENSION', 'Program extensions are not admitted');
  if (material.wireframe) fail('MATERIAL', 'Wireframe programs are not admitted yet');
  if (material.alphaHash) fail('MATERIAL', 'alphaHash programs are not admitted yet');
  if (material.stencilWrite) fail('MATERIAL', 'Stencil programs are not admitted yet');
  if (material.transmission > 0) fail('MATERIAL', 'Transmission needs the transmission render target, not admitted yet');
  if (material.isSpriteMaterial) fail('MATERIAL', `${material.type} programs are not admitted yet`);
}

/** The exact GLSL r186 WebGLProgram builds for (material, object) in this context.
 * ctx: {fog, envMap, lights (webglLights().state), toneMapping, outputColorSpace, side}. */
export function threeProgramSources(T, material, object, ctx = {}) {
  inspectThreeProgram(T, material, object);
  const lights = ctx.lights ?? EMPTY_LIGHTS;
  const parameters = webglParameters(T, material, object, {...ctx, lights, shadowMapEnabled: ctx.shadowMapEnabled === true,
    clipping: ctx.clipping ?? {numPlanes: 0, numIntersection: 0}});
  return {...webglProgramSources(T, parameters), parameters};
}
const EMPTY_LIGHTS = Object.freeze({ambient: [0, 0, 0], probe: [], sun: [], sunShadowMap: [], directional: [], directionalShadowMap: [], point: [],
  pointShadowMap: [], spot: [], spotShadowMap: [], spotLightMap: [], rectArea: [], hemi: [], numSpotLightShadowsWithMaps: 0, numLightProbes: 0});

/** r186 WebGLClipping.setState for one draw: {numPlanes, numIntersection, planes}.
 * controls: {planes: renderer.clippingPlanes, localClippingEnabled} or null when
 * clipping is off. Planes are view space for `camera` (global first, then local);
 * shadow passes (WebGLClipping.beginShadows) drop global planes and apply local
 * ones only with material.clipShadows. `planes` is null when there are none. */
export function threeClippingState(T, controls, material, camera, {shadows = false} = {}) {
  const global = shadows || !controls ? [] : controls.planes ?? [];
  const local = material.clippingPlanes;
  const useLocal = !!controls?.localClippingEnabled && local !== null && local !== undefined && local.length !== 0 && !(shadows && !material.clipShadows);
  const list = useLocal ? [...global, ...local] : global;
  if (!list.length) return {numPlanes: 0, numIntersection: 0, planes: null};
  const viewMatrix = camera.matrixWorldInverse, normalMatrix = new T.Matrix3().getNormalMatrix(viewMatrix), plane = new T.Plane();
  const planes = new Float32Array(list.length * 4);
  list.forEach((p, i) => { plane.copy(p).applyMatrix4(viewMatrix, normalMatrix); plane.normal.toArray(planes, i * 4); planes[i * 4 + 3] = plane.constant; });
  return {numPlanes: list.length, numIntersection: useLocal && material.clipIntersection ? local.length : 0, planes};
}
/** Whether WebGLRenderer binds the clipping uniform for this material. */
export const bindsClippingPlanes = m => (!m.isShaderMaterial && !m.isRawShaderMaterial) || m.clipping === true;

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
    // SAMPLE_ALPHA_TO_COVERAGE: GL ignores it on single-sampled framebuffers;
    // the pipeline enables it only when its target is multisampled.
    alphaToCoverage: m.alphaToCoverage === true,
  };
}

// ---- uniform values ---------------------------------------------------------
const BUILTIN = new Set(['modelMatrix', 'modelViewMatrix', 'projectionMatrix', 'viewMatrix', 'normalMatrix', 'cameraPosition', 'isOrthographic', 'toneMappingExposure', 'receiveShadow',
  'bindMatrix', 'bindMatrixInverse', 'morphTargetBaseInfluence', 'morphTargetInfluences', 'morphTargetsTextureSize']);

// ---- WebGLMorphtargets (retained-JS port) -----------------------------------
const morphCaches = new WeakMap();
const morphAttributeOf = g => g.morphAttributes.position || g.morphAttributes.normal || g.morphAttributes.color;
/** The geometry's morph DataArrayTexture (Float32 RGBA texels: position, normal,
 * color per vertex), built once per geometry and target count, as r186 does. */
export function threeMorphTargets(T, geometry, maxTextureSize = 8192) {
  let cache = morphCaches.get(T);
  if (!cache) morphCaches.set(T, cache = new WeakMap());
  const morphAttribute = morphAttributeOf(geometry), count = morphAttribute !== undefined ? morphAttribute.length : 0;
  let entry = cache.get(geometry);
  if (entry === undefined || entry.count !== count) {
    if (entry !== undefined) entry.texture.dispose();
    const hasPosition = geometry.morphAttributes.position !== undefined, hasNormal = geometry.morphAttributes.normal !== undefined, hasColor = geometry.morphAttributes.color !== undefined;
    const targets = geometry.morphAttributes.position || [], normals = geometry.morphAttributes.normal || [], colors = geometry.morphAttributes.color || [];
    const vertexDataCount = hasColor ? 3 : hasNormal ? 2 : hasPosition ? 1 : 0;
    let width = geometry.attributes.position.count * vertexDataCount, height = 1;
    if (width > maxTextureSize) { height = Math.ceil(width / maxTextureSize); width = maxTextureSize; }
    const buffer = new Float32Array(width * height * 4 * count), texture = new T.DataArrayTexture(buffer, width, height, count);
    texture.type = T.FloatType; texture.needsUpdate = true; texture.isF3DMorphTexture = true;
    const stride = vertexDataCount * 4, v = new T.Vector4();
    for (let i = 0; i < count; i++) {
      const offset = width * height * 4 * i, target = targets[i], normal = normals[i], color = colors[i];
      for (let j = 0; j < (target ?? normal ?? color).count; j++) {
        const at = offset + j * stride;
        if (hasPosition) { v.fromBufferAttribute(target, j); buffer[at] = v.x; buffer[at + 1] = v.y; buffer[at + 2] = v.z; buffer[at + 3] = 0; }
        if (hasNormal) { v.fromBufferAttribute(normal, j); buffer[at + 4] = v.x; buffer[at + 5] = v.y; buffer[at + 6] = v.z; buffer[at + 7] = 0; }
        if (hasColor) { v.fromBufferAttribute(color, j); buffer[at + 8] = v.x; buffer[at + 9] = v.y; buffer[at + 10] = v.z; buffer[at + 11] = color.itemSize === 4 ? v.w : 1; }
      }
    }
    entry = {count, texture, size: new T.Vector2(width, height)};
    cache.set(geometry, entry);
    const disposeTexture = () => { texture.dispose(); cache.delete(geometry); geometry.removeEventListener('dispose', disposeTexture); };
    geometry.addEventListener('dispose', disposeTexture);
  }
  return entry;
}

/** Write one frame's uniform values for one draw from a WebGLRenderer-style
 * uniforms object (material.uniforms, or a refreshed ShaderLib clone). Values
 * the program declares but nothing sets stay zero, as GL leaves them.
 * Returns the texture (or null) for each reflected sampler binding. */
export function packThreeProgramUniforms(T, reflection, uniforms, object, camera, bytes, {toneMappingExposure = 1, targetSize = null, samplers = null, values = null} = {}) {
  bytes.fill(0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const cameraPosition = new T.Vector3().setFromMatrixPosition(camera.matrixWorld);
  const builtin = {
    modelMatrix: object.matrixWorld, modelViewMatrix: object.modelViewMatrix, projectionMatrix: camera.projectionMatrix,
    viewMatrix: camera.matrixWorldInverse, normalMatrix: object.normalMatrix, cameraPosition, isOrthographic: camera.isOrthographicCamera === true,
    toneMappingExposure, receiveShadow: object.receiveShadow === true,
    // setProgram: setOptional(object, 'bindMatrix' / 'bindMatrixInverse') for skinned meshes.
    bindMatrix: object.isSkinnedMesh ? object.bindMatrix : null, bindMatrixInverse: object.isSkinnedMesh ? object.bindMatrixInverse : null,
    morphTargetBaseInfluence: null, morphTargetInfluences: null, morphTargetsTextureSize: null,
  };
  const geometry = object.geometry;
  if (geometry && morphAttributeOf(geometry) !== undefined) {
    // WebGLMorphtargets.update: base influence and influences (non-instanced).
    const influences = object.morphTargetInfluences, sum = influences.reduce((a, b) => a + b, 0);
    builtin.morphTargetBaseInfluence = geometry.morphTargetsRelative ? 1 : 1 - sum;
    builtin.morphTargetInfluences = influences;
    builtin.morphTargetsTextureSize = threeMorphTargets(T, geometry).size;
  }
  for (const u of reflection.uniforms) {
    // receiveShadow: set from the object each draw, then overwritten by a
    // material uniform of that name when one exists (WebGLRenderer.setProgram order).
    const value = BUILTIN.has(u.name) && !(u.name === 'receiveShadow' && uniforms?.receiveShadow) ? builtin[u.name]
      : values && Object.hasOwn(values, u.name) ? values[u.name] : uniforms?.[u.name]?.value;
    if (value === undefined || value === null) continue;
    write(view, u.node, value, u.name);
  }
  if (reflection.targetOffset !== null) {
    if (!targetSize || targetSize.length !== 4) fail('FRAME', 'gl_FragCoord and point sprites need the framebuffer and viewport size');
    for (let i = 0; i < 4; i++) view.setFloat32(reflection.targetOffset + 4 * i, targetSize[i], true);
  }
  return reflection.textures.map(t => {
    const v = samplers?.[t.name] ?? uniforms?.[t.name]?.value;
    return t.element === null ? v ?? null : v?.[t.element] ?? null;
  });
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
 * state() reports renderer-level inputs: {toneMapping, toneMappingExposure,
 * outputColorSpace, pixelRatio, height (drawing-buffer height)}. One instance
 * owns one WebGLLights state (setLights/setLightsView per frame, as
 * WebGLRenderer.render does) and per-material ShaderLib uniform clones.
 * Compiled programs are cached by their exact assembled source text (bounded). */
export function createThreeProgramSupport({three: T, state, maxPrograms = 256, maxPointSize = 1024, pmrem = null, shadows = null}) {
  const compiled = new Map(), lights = webglLights(T), clones = new WeakMap();
  let dfgLUT = null;
  /** r186 getDFGLUT(): the 16x16 RG half-float DFG table, linear, clamped. */
  function getDFGLUT() {
    if (dfgLUT === null) {
      dfgLUT = new T.DataTexture(animationDfgHalves(), 16, 16, T.RGFormat, T.HalfFloatType);
      dfgLUT.name = 'DFG_LUT'; dfgLUT.minFilter = T.LinearFilter; dfgLUT.magFilter = T.LinearFilter;
      dfgLUT.wrapS = T.ClampToEdgeWrapping; dfgLUT.wrapT = T.ClampToEdgeWrapping; dfgLUT.generateMipmaps = false; dfgLUT.needsUpdate = true;
    }
    return dfgLUT;
  }
  let lightList = [];
  /** shadows: renderer.shadowMap is enabled and this frame has shadow-casting
   * lights (WebGLPrograms shadowMapEnabled). renderTarget: an offscreen pass
   * (shadow depth): no tone mapping, linear output, GL row order. */
  function compile(material, object, {fog = null, side = material.side, envMap = null, shadows = false, renderTarget = false, clipping = null} = {}) {
    const s = state();
    const sources = threeProgramSources(T, material, object, {fog, side, envMap, lights: lights.state, clipping,
      toneMapping: renderTarget ? T.NoToneMapping : s.toneMapping, outputColorSpace: renderTarget ? T.LinearSRGBColorSpace : s.outputColorSpace,
      shadowMapEnabled: shadows, shadowMapType: s.shadowMapType ?? T.PCFShadowMap});
    // GL rasterizes Points as gl_PointSize squares: compile the point-sprite form.
    const points = object.isPoints === true, key = sources.key + (points ? '\u0000points' : '') + (renderTarget ? '\u0000gl-rows' : '');
    let entry = compiled.get(key);
    if (!entry) {
      const program = compileEsslProgram(sources.vertex, sources.fragment, {points, maxPointSize, rows: renderTarget ? 'gl' : 'webgpu'});
      entry = {key, program, attributesKey: JSON.stringify(program.reflection.attributes)};
      if (compiled.size >= maxPrograms) compiled.delete(compiled.keys().next().value);
      compiled.set(key, entry);
    }
    return entry;
  }
  /** WebGLRenderer.getUniforms: material.uniforms, or this material's ShaderLib clone. */
  function uniformsFor(material) {
    if (material.isShaderMaterial) return material.uniforms;
    let u = clones.get(material);
    if (!u) clones.set(material, u = webglMaterialUniforms(T, material));
    return u;
  }
  /** WebGLRenderer.setProgram binds shadow maps from the light state by uniform
   * name for every material that needs lights (not from material uniforms). */
  function shadowSamplers(material) {
    if (!materialNeedsLights(material)) return null;
    const st = lights.state;
    return {sunShadowMap: st.sunShadowMap, directionalShadowMap: st.directionalShadowMap, spotShadowMap: st.spotShadowMap, pointShadowMap: st.pointShadowMap};
  }
  /** Samplers WebGLRenderer.setProgram binds by name outside material uniforms:
   * light-state shadow maps, and a skinned mesh's bone texture (computed on
   * first use, as setProgram does). */
  function objectSamplers(material, object) {
    let out = material ? shadowSamplers(material) : null;
    if (object?.isSkinnedMesh) {
      const skeleton = object.skeleton;
      if (skeleton.boneTexture === null) skeleton.computeBoneTexture();
      out = {...out, boneTexture: skeleton.boneTexture};
    }
    if (object?.geometry && morphAttributeOf(object.geometry) !== undefined) out = {...out, morphTargetsTexture: threeMorphTargets(T, object.geometry).texture};
    return out;
  }
  /** setProgram's refreshMaterial work: lights, fog and material values. */
  function refresh(material, {fog = null, envMap = null, envMapRotation, distanceLight = null} = {}) {
    const uniforms = uniformsFor(material), s = state();
    refreshWebGLMaterialUniforms(T, uniforms, material, {fog, lights: lights.state, envMap, envMapRotation, distanceLight,
      pixelRatio: s.pixelRatio ?? 1, height: s.height ?? 1, unlitColorSpace: s.outputColorSpace});
    if (uniforms.dfgLUT !== undefined) uniforms.dfgLUT.value = getDFGLUT();
    return uniforms;
  }
  const support = Object.freeze({
    compile, uniformsFor, refresh, needsLights: materialNeedsLights,
    /** r186 PMREM owner (three_program_pmrem.mjs) when injected. */
    createPMREM: pmrem ? (device, bindingOf) => pmrem({three: T, device, bindingOf}) : null,
    /** r186 WebGLShadowMap port (three_program_shadows.mjs) when injected. */
    createShadows: shadows ? (device, options) => shadows({three: T, device, support, ...options}) : null,
    get lightList() { return lightList; }, shaderLibMaterial: m => SHADER_IDS[m?.type] !== undefined,
    /** WebGLLights.setup for this frame's light list (source traversal order). */
    setLights(list) { lightList = [...list]; lights.setup(lightList); return lights.state.version; },
    setLightsView(camera) { lights.setupView(lightList, camera); },
    get lightsVersion() { return lights.state.version; },
    raster: (material, options) => threeProgramRaster(T, material, options),
    /** options.material: lit materials read shadow maps from the light state. */
    pack: (reflection, uniforms, object, camera, bytes, {material = null, ...options} = {}) =>
      packThreeProgramUniforms(T, reflection, uniforms, object, camera, bytes, {toneMappingExposure: state().toneMappingExposure ?? 1,
        samplers: objectSamplers(material, object), ...options}),
    shadowSamplers, objectSamplers,
    state,
    clippingState: (controls, material, camera, options) => threeClippingState(T, controls, material, camera, options),
    bindsClippingPlanes,
    createGeometry: (device, source, attributes, options) => createGpuProgramGeometry(device, source, attributes, options),
    geometrySnapshot: (gpu, device) => programGeometrySnapshot(gpu, device),
  });
  return support;
}
