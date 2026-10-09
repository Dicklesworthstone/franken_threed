/** r186 WebGLShadowMap on the WebGL-surface program route.
 *
 * Route: a retained-JavaScript port (MIT, three.js authors) of r186
 * `WebGLShadowMap.render` / `getDepthMaterial` / `renderObject`, operating on the
 * application's own THREE classes: it allocates `light.shadow.map` as r186 does
 * (WebGLRenderTarget + DepthTexture, or WebGLCubeRenderTarget + CubeDepthTexture
 * for point lights; LessEqual compare and linear filtering for PCF, nearest for
 * Basic), updates shadow cameras/matrices/frustums, culls casters per face,
 * selects MeshDepthMaterial / MeshDistanceMaterial (or the object's custom
 * depth/distance material, or a per-material clone when alpha test, displacement
 * or clipping state must be reflected), mirrors side/shadowSide, and calls
 * onBeforeShadow / onAfterShadow around each caster draw, in source order.
 *
 * Each recorded caster draw is that depth material's ShaderLib program (r186
 * WebGLProgram source, three_program.mjs) compiled with `rows: 'gl'`, drawn into
 * a depth32float map (cube maps: one 2D layer per face) with the shadow
 * camera's uniforms packed at the draw's source call. Receivers bind the same
 * map: PCF as `sampler2DShadow` / `samplerCubeShadow` (comparison sampler,
 * less-equal, linear), Basic as an unfilterable `sampler2D` / `samplerCube`.
 *
 * Differences stated, not hidden: maps are depth32float where r186 WebGL
 * allocates 24-bit DEPTH_COMPONENT24 (UnsignedIntType); comparisons are at least
 * as precise. Line casters draw gl.LINES / gl.LINE_STRIP (a LineLoop's closing
 * strip) with the depth/distance program.
 *
 * VSM (directional/spot lights, as r186): casters write the depth32float
 * `shadow.map.depthTexture` (r186: FloatType DEPTH_COMPONENT32F), then r186's own
 * `vsm` ShaderMaterial runs as two full-screen passes, vertical (depth ->
 * `shadow.mapPass`, RG16F) and horizontal (mapPass -> `shadow.map.texture`,
 * RG16F, with the depth texture still attached), compiled by the same program
 * route. Receivers sample `shadow.map.texture` (linear filtering) as r186's
 * WebGLLights selects. Caster passes skip the RG color writes r186 makes with
 * the depth material: the horizontal pass clears and covers every texel, so the
 * final map contents are the same. VSM point lights warn and are skipped as in
 * r186. Explicit errors: point casters, wireframe casters, reversed depth,
 * a point light whose existing map changes to VSM. A new light, a map-size change or a shadow-type change is a
 * preparation boundary (the map's GPU texture and the receivers' programs change).
 * No performance claim.
 */
import {createProgramMeshes} from './animation_program_mesh.mjs';

// r186 src/renderers/shaders/ShaderLib/vsm.glsl.js (MIT, three.js authors), verbatim;
// it is not reachable from the public THREE namespace.
const VSM_VERTEX = /* glsl */`
void main() {

	gl_Position = vec4( position, 1.0 );

}
`;
const VSM_FRAGMENT = /* glsl */`
uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;

void main() {

	const float samples = float( VSM_SAMPLES );

	float mean = 0.0;
	float squared_mean = 0.0;

	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {

		float uvOffset = uvStart + i * uvStride;

		#ifdef HORIZONTAL_PASS

			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;

		#else

			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;

		#endif

	}

	mean = mean / samples;
	squared_mean = squared_mean / samples;

	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );

	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );

}
`;

export class ThreeProgramShadowError extends Error {
  constructor(code, message) { super(`THREE_PROGRAM_SHADOW_${code}: ${message}`); this.name = 'ThreeProgramShadowError'; this.code = 'THREE_PROGRAM_SHADOW_' + code; }
}
const fail = (code, message) => { throw new ThreeProgramShadowError(code, message); };
const DEPTH_FORMAT = 'depth32float';

function scoped(device, operation) {
  device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory');
  let value, error;
  try { value = operation(); } catch (caught) { error = caught; }
  const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(values => {
    if (error) throw error;
    const reported = values.find(Boolean);
    if (reported) fail('DEVICE', reported.message || 'WebGPU operation failed');
  });
  return {value, error, errors};
}

/** support: createThreeProgramSupport(); bindingOf(texture) -> {view, sampler, sampleType};
 * sourceOf(object) -> the attribute source a program reads (geometry or InstancedMesh view). */
export function createThreeProgramShadows({three: T, device, support, bindingOf, sourceOf = o => o.geometry, clipping = () => null, updateObject = () => {}, label = 'f3d-program-shadows',
  maxDraws = 4096, maxGeometryBytes = 256 * 1024 * 1024}) {
  const maxTextureSize = device.limits.maxTextureDimension2D ?? 8192;
  const depthMaterialBase = new T.MeshDepthMaterial(), distanceMaterialBase = new T.MeshDistanceMaterial();
  const materialCache = {}, distanceLights = new WeakMap();
  const shadowSide = {[T.FrontSide]: T.BackSide, [T.BackSide]: T.FrontSide, [T.DoubleSide]: T.DoubleSide};
  const cubeDirections = [new T.Vector3(1, 0, 0), new T.Vector3(-1, 0, 0), new T.Vector3(0, 1, 0), new T.Vector3(0, -1, 0), new T.Vector3(0, 0, 1), new T.Vector3(0, 0, -1)];
  const cubeUps = [new T.Vector3(0, -1, 0), new T.Vector3(0, -1, 0), new T.Vector3(0, 0, 1), new T.Vector3(0, 0, -1), new T.Vector3(0, -1, 0), new T.Vector3(0, -1, 0)];
  const projScreenMatrix = new T.Matrix4(), lightPositionWorld = new T.Vector3(), lookTarget = new T.Vector3();
  const shadowMapSize = new T.Vector2(), viewportSize = new T.Vector2(), viewport = new T.Vector4();
  let previousType = T.PCFShadowMap, disposed = false;
  const maps = new Map(), geometries = new Map(), records = new Map(), lost = new Promise(() => {}), identityCamera = new T.Camera();
  const meshFail = (code, message) => { throw new ThreeProgramShadowError(code.replace(/^ANIMATION_RENDER_/, ''), message); };
  const meshes = createProgramMeshes({device, format: null, depthFormat: DEPTH_FORMAT, sampleCount: 1, maxDraws, label, fail: meshFail, scoped, lost});
  // ---- VSM blur (WebGLShadowMap's shadowMaterialVertical/Horizontal + VSMPass) --
  const VSM_FORMAT = 'rg16float';
  const blurMeshes = createProgramMeshes({device, format: VSM_FORMAT, depthFormat: DEPTH_FORMAT, sampleCount: 1, maxDraws: 64, label: `${label}/vsm`, fail: meshFail, scoped, lost});
  const shadowMaterialVertical = new T.ShaderMaterial({defines: {VSM_SAMPLES: 8},
    uniforms: {shadow_pass: {value: null}, resolution: {value: new T.Vector2()}, radius: {value: 4.0}}, vertexShader: VSM_VERTEX, fragmentShader: VSM_FRAGMENT});
  const shadowMaterialHorizontal = shadowMaterialVertical.clone();
  shadowMaterialHorizontal.defines.HORIZONTAL_PASS = 1;
  const fullScreenTri = new T.BufferGeometry();
  fullScreenTri.setAttribute('position', new T.BufferAttribute(new Float32Array([-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5]), 3));
  const fullScreenMesh = new T.Mesh(fullScreenTri, shadowMaterialVertical);
  const blurGeometries = new Map(), colorBindings = new Map();
  const isVSMTarget = target => target.isWebGLCubeRenderTarget !== true && target.texture.format === T.RGFormat;

  const ids = new WeakMap();
  let nextId = 0;
  const idOf = o => { let id = ids.get(o); if (id === undefined) ids.set(o, id = ++nextId); return id; };
  // A record owns one geometry residency: the layout signature alone is shared by
  // unrelated geometries, so the residency's identity is part of the key.
  const recordKey = (compiled, gpu, geometry, glRaster, textures) =>
    [compiled.key, idOf(gpu), geometry.signature, JSON.stringify(glRaster), ...textures.map(t => idOf(t.view) + ':' + idOf(t.sampler))].join('\u0001');
  const live = () => { if (disposed) fail('DISPOSED', 'Program shadow owner is disposed'); };
  const normalizedType = settings => settings.type === T.PCFSoftShadowMap ? T.PCFShadowMap : settings.type;
  const runs = settings => settings.enabled === true && (settings.autoUpdate !== false || settings.needsUpdate === true);
  const updates = light => !(light.shadow.autoUpdate === false && light.shadow.needsUpdate === false);

  // ---- WebGLShadowMap.getDepthMaterial (port) ------------------------------
  function onMaterialDispose(event) {
    const material = event.target;
    material.removeEventListener('dispose', onMaterialDispose);
    for (const id in materialCache) {
      const cache = materialCache[id];
      if (material.uuid in cache) { cache[material.uuid].dispose(); delete cache[material.uuid]; }
    }
  }
  function getDepthMaterial(object, material, light, type, renderer) {
    let result = null;
    const customMaterial = light.isPointLight === true ? object.customDistanceMaterial : object.customDepthMaterial;
    if (customMaterial !== undefined) result = customMaterial;
    else {
      result = light.isPointLight === true ? distanceMaterialBase : depthMaterialBase;
      if ((clipping()?.localClippingEnabled && material.clipShadows === true && Array.isArray(material.clippingPlanes) && material.clippingPlanes.length !== 0) ||
        (material.displacementMap && material.displacementScale !== 0) || (material.alphaMap && material.alphaTest > 0) ||
        (material.map && material.alphaTest > 0) || material.alphaToCoverage === true) {
        const keyA = result.uuid, keyB = material.uuid;
        let variants = materialCache[keyA];
        if (variants === undefined) materialCache[keyA] = variants = {};
        let cached = variants[keyB];
        if (cached === undefined) { cached = result.clone(); variants[keyB] = cached; material.addEventListener('dispose', onMaterialDispose); }
        result = cached;
      }
    }
    result.visible = material.visible; result.wireframe = material.wireframe;
    result.side = type === T.VSMShadowMap ? (material.shadowSide !== null ? material.shadowSide : material.side)
      : (material.shadowSide !== null ? material.shadowSide : shadowSide[material.side]);
    result.alphaMap = material.alphaMap;
    result.alphaTest = material.alphaToCoverage === true ? 0.5 : material.alphaTest;
    result.map = material.map;
    result.clipShadows = material.clipShadows; result.clippingPlanes = material.clippingPlanes; result.clipIntersection = material.clipIntersection;
    result.displacementMap = material.displacementMap; result.displacementScale = material.displacementScale; result.displacementBias = material.displacementBias;
    result.wireframeLinewidth = material.wireframeLinewidth; result.linewidth = material.linewidth;
    if (light.isPointLight === true && result.isMeshDistanceMaterial === true) distanceLights.set(result, light);
    return result;
  }

  // ---- map allocation (the first part of WebGLShadowMap.render's light loop) --
  /** Returns true when the map's GPU texture had to change. */
  function allocate(light, type, typeChanged) {
    const shadow = light.shadow;
    shadowMapSize.copy(shadow.mapSize);
    const frameExtents = shadow.getFrameExtents();
    shadowMapSize.multiply(frameExtents);
    viewportSize.copy(shadow.mapSize);
    if (shadowMapSize.x > maxTextureSize || shadowMapSize.y > maxTextureSize) {
      if (shadowMapSize.x > maxTextureSize) { viewportSize.x = Math.floor(maxTextureSize / frameExtents.x); shadowMapSize.x = viewportSize.x * frameExtents.x; shadow.mapSize.x = viewportSize.x; }
      if (shadowMapSize.y > maxTextureSize) { viewportSize.y = Math.floor(maxTextureSize / frameExtents.y); shadowMapSize.y = viewportSize.y * frameExtents.y; shadow.mapSize.y = viewportSize.y; }
    }
    shadow.camera._reversedDepth = false;
    if (shadow.map === null || typeChanged === true) {
      if (shadow.map !== null) {
        if (shadow.map.depthTexture !== null) { shadow.map.depthTexture.dispose(); shadow.map.depthTexture = null; }
        shadow.map.dispose();
      }
      if (type === T.VSMShadowMap) {
        if (light.isPointLight) {
          if (shadow.map !== null) fail('TYPE', 'A point light whose shadow map changes to VSM keeps a disposed map in r186; not admitted');
          return false;
        }
        shadow.map = new T.WebGLRenderTarget(shadowMapSize.x, shadowMapSize.y, {format: T.RGFormat, type: T.HalfFloatType,
          minFilter: T.LinearFilter, magFilter: T.LinearFilter, generateMipmaps: false});
        shadow.map.texture.name = light.name + '.shadowMap';
        // Native depth texture for VSM - depth is captured here, then blurred into the color texture.
        shadow.map.depthTexture = new T.DepthTexture(shadowMapSize.x, shadowMapSize.y, T.FloatType);
        shadow.map.depthTexture.name = light.name + '.shadowMapDepth';
        shadow.map.depthTexture.format = T.DepthFormat;
        shadow.map.depthTexture.compareFunction = null;
        shadow.map.depthTexture.minFilter = T.NearestFilter; shadow.map.depthTexture.magFilter = T.NearestFilter;
        shadow.camera.updateProjectionMatrix();
      } else if (light.isPointLight) {
        shadow.map = new T.WebGLCubeRenderTarget(shadowMapSize.x);
        shadow.map.depthTexture = new T.CubeDepthTexture(shadowMapSize.x, T.UnsignedIntType);
      } else {
        shadow.map = new T.WebGLRenderTarget(shadowMapSize.x, shadowMapSize.y);
        shadow.map.depthTexture = new T.DepthTexture(shadowMapSize.x, shadowMapSize.y, T.UnsignedIntType);
      }
      if (type !== T.VSMShadowMap) {
      shadow.map.depthTexture.name = light.name + '.shadowMap';
      shadow.map.depthTexture.format = T.DepthFormat;
      if (type === T.PCFShadowMap) {
        shadow.map.depthTexture.compareFunction = T.LessEqualCompare;
        shadow.map.depthTexture.minFilter = T.LinearFilter; shadow.map.depthTexture.magFilter = T.LinearFilter;
      } else {
        shadow.map.depthTexture.compareFunction = null;
        shadow.map.depthTexture.minFilter = T.NearestFilter; shadow.map.depthTexture.magFilter = T.NearestFilter;
      }
      shadow.camera.updateProjectionMatrix();
      }
    }
    if (light.isPointLight && type === T.VSMShadowMap && shadow.map === null) return false;
    if (shadow.map.isWebGLCubeRenderTarget !== true && (shadow.map.width !== shadowMapSize.x || shadow.map.height !== shadowMapSize.y))
      shadow.map.setSize(shadowMapSize.x, shadowMapSize.y);
    return true;
  }
  function destroyMap(m) {
    m.texture.destroy();
    if (m.vsm) { m.vsm.map.destroy(); m.vsm.pass.destroy(); m.vsm.passDepth.destroy(); colorBindings.delete(m.vsm.texture); if (m.vsm.passTexture) colorBindings.delete(m.vsm.passTexture); }
  }
  /** shadow.mapPass, created and sized as VSMPass does; it binds the owned RG16F pass texture. */
  function mapPassOf(shadow, m) {
    if (shadow.mapPass === null) shadow.mapPass = new T.WebGLRenderTarget(shadowMapSize.x, shadowMapSize.y, {format: T.RGFormat, type: T.HalfFloatType});
    else if (shadow.mapPass.width !== shadow.map.width || shadow.mapPass.height !== shadow.map.height) shadow.mapPass.setSize(shadow.map.width, shadow.map.height);
    if (m.vsm.passTexture !== shadow.mapPass.texture) {
      if (m.vsm.passTexture) colorBindings.delete(m.vsm.passTexture);
      m.vsm.passTexture = shadow.mapPass.texture;
      colorBindings.set(m.vsm.passTexture, m.vsm.passBinding);
    }
    return shadow.mapPass;
  }
  function blurMaterials(shadow) {
    if (shadowMaterialVertical.defines.VSM_SAMPLES !== shadow.blurSamples) {
      shadowMaterialVertical.defines.VSM_SAMPLES = shadow.blurSamples;
      shadowMaterialHorizontal.defines.VSM_SAMPLES = shadow.blurSamples;
      shadowMaterialVertical.needsUpdate = true;
      shadowMaterialHorizontal.needsUpdate = true;
    }
  }
  function setBlurUniforms(material, shadow, source) {
    material.uniforms.shadow_pass.value = source;
    material.uniforms.resolution.value.set(shadow.map.width, shadow.map.height);
    material.uniforms.radius.value = shadow.radius;
  }
  /** One record per (blur program, source binding); built at preparation, looked up by frames. */
  function blurLookup(material, create) {
    fullScreenMesh.material = material;
    const compiled = support.compile(material, fullScreenMesh, {renderTarget: true});
    let gpu = blurGeometries.get(compiled.attributesKey);
    if (!gpu) {
      if (!create) fail('PREPARE', 'VSM blur geometry needs prepare()');
      gpu = support.createGeometry(device, fullScreenTri, compiled.program.reflection.attributes, {maxBytes: 4096, label: `${label}/vsm-geometry`});
      blurGeometries.set(compiled.attributesKey, gpu);
      gpu.update({maxAdditionalBytes: 4096});
    }
    const geometry = support.geometrySnapshot(gpu, device);
    const uniforms = support.refresh(material);
    const textures = compiled.program.reflection.textures.map(t => {
      const texture = uniforms[t.name]?.value, b = texture && (maps.get(texture)?.depthBinding ?? colorBindings.get(texture));
      if (!b) fail('PREPARE', 'VSM blur source texture is not prepared');
      return {view: b.view, sampler: b.sampler, sampleType: b.sampleType, texture};
    });
    const raster = support.raster(material, {side: material.side});
    const glRaster = {...raster, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw', blend: null};
    const key = 'vsm\u0001' + recordKey(compiled, gpu, geometry, glRaster, textures);
    return {key, compiled, gpu, textures, glRaster, entry: records.get(key)};
  }
  async function blurRecord(material) {
    const {key, compiled, gpu, textures, glRaster, entry} = blurLookup(material, true);
    if (entry) return entry;
    const created = {record: await blurMeshes.add(gpu, {program: compiled.program, textures, raster: glRaster, topology: 'triangles'}), textures, gpu};
    records.set(key, created);
    return created;
  }
  function blurRecordSync(material) {
    return blurLookup(material, false).entry ?? fail('PREPARE', 'A VSM blur program or binding changed; call prepare()');
  }
  function blurCommand(entry, material, camera, width, height) {
    const reflection = entry.record.reflection, bytes = new Uint8Array(reflection.uniformBufferSize);
    const current = support.pack(reflection, support.refresh(material), fullScreenMesh, camera, bytes, {targetSize: [width, height, width, height]});
    if (current.some((t, k) => (t ?? null) !== entry.textures[k].texture)) fail('PREPARE', 'VSM blur textures changed; call prepare()');
    const command = {};
    blurMeshes.stage(entry.record, {programUniforms: bytes, first: 0, count: 3, frontFaceCW: false, instanceCount: 1}, command);
    return command;
  }
  function mapState(target) {
    const depth = target.depthTexture, cube = target.isWebGLCubeRenderTarget === true;
    let m = maps.get(depth);
    if (m && m.width === target.width && m.height === target.height && m.cube === cube) return m;
    if (m) destroyMap(m);
    const texture = device.createTexture({label: `${label}/${depth.name}`, size: [target.width, target.height, cube ? 6 : 1], format: DEPTH_FORMAT, usage: 16 | 4});
    const compare = depth.compareFunction != null;
    m = {texture, width: target.width, height: target.height, cube, depth,
      faces: Array.from({length: cube ? 6 : 1}, (_, face) => texture.createView({dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1})),
      binding: {view: texture.createView({dimension: cube ? 'cube' : '2d'}),
        sampler: device.createSampler({label, ...(compare ? {compare: 'less-equal', magFilter: 'linear', minFilter: 'linear'} : {}),
          addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge'}),
        sampleType: compare ? 'depth' : 'unfilterable-float'}};
    if (compare && depth.compareFunction !== T.LessEqualCompare) fail('MAP', 'Only LessEqual shadow comparisons are produced by r186 WebGLShadowMap');
    if (isVSMTarget(target)) {
      // map.texture / mapPass.texture: RG16F with linear filtering (r186 options);
      // mapPass also owns a depth buffer that VSMPass clears and tests against.
      const color = name => device.createTexture({label: `${label}/${name}`, size: [target.width, target.height], format: VSM_FORMAT, usage: 16 | 4});
      const linear = device.createSampler({label, magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
      m.vsm = {map: color(target.texture.name), pass: color(`${target.texture.name}.pass`),
        passDepth: device.createTexture({label: `${label}/vsm-pass-depth`, size: [target.width, target.height], format: DEPTH_FORMAT, usage: 16})};
      m.vsm.mapBinding = {view: m.vsm.map.createView(), sampler: linear, sampleType: 'float'};
      m.vsm.passBinding = {view: m.vsm.pass.createView(), sampler: linear, sampleType: 'float'};
      m.vsm.texture = target.texture;
      colorBindings.set(target.texture, m.vsm.mapBinding);
    }
    // The VSM vertical pass samples the depth texture as a regular (non-comparison) texture.
    m.depthBinding = compare ? null : m.binding;
    maps.set(depth, m);
    return m;
  }

  // ---- caster traversal (renderObject port) --------------------------------
  function casters(object, camera, light, type, frustum, visit) {
    if (object.visible === false) return;
    const visible = object.layers.test(camera.layers);
    if (visible && (object.isMesh || object.isLine || object.isPoints)) {
      if ((object.castShadow || (object.receiveShadow && type === T.VSMShadowMap)) && (frustum === null || !object.frustumCulled || object.intersectsFrustum(frustum))) {
        if (object.isPoints) fail('CASTER', 'Point shadow casters are not admitted yet');
        const material = object.material;
        if (Array.isArray(material)) {
          for (const group of object.geometry.groups) {
            const groupMaterial = material[group.materialIndex];
            if (groupMaterial && groupMaterial.visible) visit(object, groupMaterial, group);
          }
        } else if (material.visible) visit(object, material, null);
      }
    }
    for (const child of object.children) casters(child, camera, light, type, frustum, visit);
  }
  // renderBufferDirect's mode: gl.LINES / gl.LINE_STRIP (LineLoop: the source view's closing strip) / gl.TRIANGLES.
  const topologyOf = object => object.isLineSegments ? 'lines' : object.isLine ? 'line-strip' : 'triangles';
  const stripFormat = (object, topology) => topology === 'line-strip' && sourceOf(object).index ? (sourceOf(object).index.array instanceof Uint32Array ? 'uint32' : 'uint16') : undefined;
  function compile(depthMaterial, object) {
    if (depthMaterial.wireframe) fail('CASTER', 'Wireframe shadow casters are not admitted yet');
    // WebGLClipping during shadows: local planes only, and only with clipShadows.
    const clip = support.clippingState(clipping(), depthMaterial, identityCamera, {shadows: true});
    return support.compile(depthMaterial, object, {renderTarget: true, clipping: {numPlanes: clip.numPlanes, numIntersection: clip.numIntersection}});
  }
  function geometryFor(object, compiled) {
    const source = sourceOf(object);
    let byKey = geometries.get(source);
    if (!byKey) geometries.set(source, byKey = new Map());
    let gpu = byKey.get(compiled.attributesKey);
    if (!gpu) {
      gpu = support.createGeometry(device, source, compiled.program.reflection.attributes, {maxBytes: maxGeometryBytes, label: `${label}/geometry`});
      byKey.set(compiled.attributesKey, gpu);
    }
    return gpu;
  }
  function textureBindings(reflection, uniforms, object) {
    const samplers = support.objectSamplers?.(null, object) ?? null;
    return reflection.textures.map(t => {
      const value = samplers?.[t.name] ?? uniforms?.[t.name]?.value, texture = t.element === null ? value : value?.[t.element];
      if (!texture) fail('TEXTURE', `Depth program sampler ${t.name} has no texture`);
      const b = bindingOf(texture);
      return {view: b.view, sampler: b.sampler, sampleType: b.sampleType ?? 'float', texture};
    });
  }
  function refresh(depthMaterial) {
    return support.refresh(depthMaterial, {distanceLight: distanceLights.get(depthMaterial)});
  }
  /** One record per (program, geometry layout, textures, side); built at preparation. */
  async function recordFor(depthMaterial, object) {
    const compiled = compile(depthMaterial, object);
    const gpu = geometryFor(object, compiled);
    gpu.update({maxAdditionalBytes: maxGeometryBytes});
    const geometry = support.geometrySnapshot(gpu, device);
    const uniforms = refresh(depthMaterial), textures = textureBindings(compiled.program.reflection, uniforms, object);
    const topology = topologyOf(object), raster = support.raster(depthMaterial, {side: depthMaterial.side, topology});
    // rows:'gl' mirrors clip Y: GL's counter-clockwise front faces are clockwise here.
    const glRaster = {...raster, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw', blend: null, writeMask: 0};
    const key = recordKey(compiled, gpu, geometry, glRaster, textures) + '\u0001' + topology;
    let entry = records.get(key);
    if (entry) return entry;
    const strip = stripFormat(object, topology);
    const record = await meshes.add(gpu, {program: compiled.program, textures, raster: glRaster, topology, ...(strip ? {stripIndexFormat: strip} : {})});
    entry = {record, textures, gpu};
    records.set(key, entry);
    return entry;
  }

  /** Preparation: allocate maps as the next render would, and build every
   * caster program/pipeline it can reach (no callbacks, no frustum culling). */
  async function prepare(lights, scene, camera, settings, renderer) {
    live();
    const type = normalizedType(settings);
    if (!runs(settings) || lights.length === 0) return;
    const typeChanged = previousType !== type;
    for (const light of lights) {
      if (light.shadow === undefined || !updates(light)) continue;
      if (allocate(light, type, typeChanged) === false) continue;
      mapState(light.shadow.map);
    }
    if (typeChanged) {
      scene.traverse(object => {
        if (object.material) for (const m of Array.isArray(object.material) ? object.material : [object.material]) m.needsUpdate = true;
      });
      previousType = type;
    }
    for (const light of lights) {
      if (light.shadow === undefined || !light.shadow.map) continue;
      const pending = [];
      casters(scene, camera, light, type, null, (object, material) => pending.push([object, material]));
      // The shared depth/distance material is re-stated per caster (as each
      // render does) right before its record captures program and bindings.
      for (const [object, material] of pending) await recordFor(getDepthMaterial(object, material, light, type, renderer), object);
      if (light.isPointLight !== true && type === T.VSMShadowMap && isVSMTarget(light.shadow.map)) {
        const shadow = light.shadow, m = mapState(shadow.map);
        blurMaterials(shadow);
        setBlurUniforms(shadowMaterialVertical, shadow, shadow.map.depthTexture);
        await blurRecord(shadowMaterialVertical);
        setBlurUniforms(shadowMaterialHorizontal, shadow, mapPassOf(shadow, m).texture);
        await blurRecord(shadowMaterialHorizontal);
      }
    }
  }

  /** WebGLShadowMap.render for this frame: source-visible effects (shadow cameras,
   * matrices, modelViewMatrix, callbacks, needsUpdate flags) happen now; the
   * returned function submits the recorded depth passes (after texture uploads). */
  function render(lights, scene, camera, settings, renderer) {
    live();
    if (settings.enabled === false) return null;
    if (settings.autoUpdate === false && settings.needsUpdate === false) return null;
    if (lights.length === 0) return null;
    if (settings.type === T.PCFSoftShadowMap) {
      (T.warn ?? console.warn)('WebGLShadowMap: PCFSoftShadowMap has been removed. Using PCFShadowMap instead.');
      settings.type = T.PCFShadowMap;
    }
    const type = settings.type;
    if (previousType !== type) fail('PREPARE', 'Shadow map type changed; call prepare()');
    meshes.begin(); blurMeshes.begin();
    const passes = [];
    for (const light of lights) {
      const shadow = light.shadow;
      if (shadow === undefined) { (T.warn ?? console.warn)('WebGLShadowMap:', light, 'has no shadow.'); continue; }
      if (!updates(light)) continue;
      if (type === T.VSMShadowMap && light.isPointLight && shadow.map === null) {
        (T.warn ?? console.warn)('WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.');
        continue;
      }
      if (shadow.map === null) fail('PREPARE', 'A new shadow light needs prepare()');
      allocate(light, type, false);
      const map = maps.get(shadow.map.depthTexture);
      if (!map || map.width !== shadow.map.width || map.height !== shadow.map.height) fail('PREPARE', 'Shadow map size changed; call prepare()');
      const cube = shadow.map.isWebGLCubeRenderTarget === true;
      const faceCount = cube ? 6 : shadow.getViewportCount();
      if (light.isPointLight !== true) shadow.updateMatrices(light, camera);
      for (let face = 0; face < faceCount; face++) {
        const shadowCamera = shadow.getCamera(face);
        if (light.isPointLight) {
          const c = shadow.camera, shadowMatrix = shadow.matrix, far = light.distance || c.far;
          if (far !== c.far) { c.far = far; c.updateProjectionMatrix(); }
          lightPositionWorld.setFromMatrixPosition(light.matrixWorld);
          c.position.copy(lightPositionWorld);
          lookTarget.copy(c.position); lookTarget.add(cubeDirections[face]);
          c.up.copy(cubeUps[face]); c.lookAt(lookTarget); c.updateMatrixWorld();
          shadowMatrix.makeTranslation(-lightPositionWorld.x, -lightPositionWorld.y, -lightPositionWorld.z);
          projScreenMatrix.multiplyMatrices(c.projectionMatrix, c.matrixWorldInverse);
          shadow._frustum.setFromProjectionMatrix(projScreenMatrix, c.coordinateSystem, c.reversedDepth);
        }
        let rect;
        if (cube) rect = [0, 0, map.width, map.height];
        else {
          const v = shadow.getViewport(face);
          viewport.set(viewportSize.x * v.x, viewportSize.y * v.y, viewportSize.x * v.z, viewportSize.y * v.w);
          rect = [viewport.x, viewport.y, viewport.z, viewport.w];
        }
        // r186 clears each cube face, and a 2D map once before its first viewport.
        const pass = {view: map.faces[cube ? face : 0], clear: cube || face === 0, rect, commands: []};
        const frustum = shadow.getFrustum(face);
        casters(scene, camera, light, type, frustum, (object, material, group) => {
          object.modelViewMatrix.multiplyMatrices(shadowCamera.matrixWorldInverse, object.matrixWorld);
          const depthMaterial = getDepthMaterial(object, material, light, type, renderer);
          const geometry = object.geometry;
          object.onBeforeShadow(renderer, object, camera, shadowCamera, geometry, depthMaterial, group);
          // renderBufferDirect(shadowCamera, null, geometry, depthMaterial, object, group)
          pass.commands.push(draw(object, depthMaterial, group, shadowCamera));
          object.onAfterShadow(renderer, object, camera, shadowCamera, geometry, depthMaterial, group);
        });
        passes.push(pass);
      }
      // do blur pass for VSM
      if (shadow.isPointLightShadow !== true && type === T.VSMShadowMap) {
        const m = map.vsm ?? fail('PREPARE', 'VSM shadow map needs prepare()');
        blurMaterials(shadow);
        const mapPass = mapPassOf(shadow, map);
        if (mapPass.width !== map.width || mapPass.height !== map.height) fail('PREPARE', 'VSM pass size changed; call prepare()');
        // vertical pass - read from native depth texture
        setBlurUniforms(shadowMaterialVertical, shadow, shadow.map.depthTexture);
        fullScreenMesh.material = shadowMaterialVertical;
        passes.push({color: m.pass.createView(), view: m.passDepth.createView(), clear: true, rect: [0, 0, map.width, map.height],
          commands: [blurCommand(blurRecordSync(shadowMaterialVertical), shadowMaterialVertical, camera, map.width, map.height)]});
        // horizontal pass
        setBlurUniforms(shadowMaterialHorizontal, shadow, mapPass.texture);
        fullScreenMesh.material = shadowMaterialHorizontal;
        passes.push({color: m.map.createView(), view: map.faces[0], clear: true, rect: [0, 0, map.width, map.height],
          commands: [blurCommand(blurRecordSync(shadowMaterialHorizontal), shadowMaterialHorizontal, camera, map.width, map.height)]});
      }
      shadow.needsUpdate = false;
    }
    previousType = type;
    settings.needsUpdate = false;
    return () => submit(passes);
  }
  const updated = new Set();
  function draw(object, depthMaterial, group, shadowCamera) {
    const entry = recordForSync(depthMaterial, object);
    const reflection = entry.record.reflection, bytes = new Uint8Array(reflection.uniformBufferSize);
    const uniforms = refresh(depthMaterial);
    updateObject(object);
    const clip = support.clippingState(clipping(), depthMaterial, shadowCamera, {shadows: true});
    const current = support.pack(reflection, uniforms, object, shadowCamera, bytes, {values: support.bindsClippingPlanes(depthMaterial) ? {clippingPlanes: clip.planes} : null});
    if (current.some((t, k) => (t ?? null) !== entry.textures[k].texture)) fail('PREPARE', 'Shadow caster textures changed; call prepare()');
    const g = object.geometry;
    const start = group ? group.start : 0, count = group ? group.count : Number.MAX_SAFE_INTEGER;
    const instanceCount = object.isInstancedMesh ? object.count : g.isInstancedBufferGeometry ? g.instanceCount : 1;
    const command = {};
    meshes.stage(entry.record, {programUniforms: bytes, first: start, count, frontFaceCW: object.isMesh === true && object.matrixWorld.determinant() < 0, instanceCount}, command);
    return command;
  }
  function recordForSync(depthMaterial, object) {
    // Same key as recordFor(), but frames never create pipelines.
    const compiled = compile(depthMaterial, object);
    const source = sourceOf(object), gpu = geometries.get(source)?.get(compiled.attributesKey);
    if (!gpu) fail('PREPARE', 'A shadow caster geometry needs prepare()');
    // Same state as preparation: current residency before its layout signature.
    if (!updated.has(gpu)) { gpu.update({maxAdditionalBytes: maxGeometryBytes}); updated.add(gpu); }
    const geometry = support.geometrySnapshot(gpu, device);
    const uniforms = refresh(depthMaterial), textures = textureBindings(compiled.program.reflection, uniforms, object);
    const topology = topologyOf(object), raster = support.raster(depthMaterial, {side: depthMaterial.side, topology});
    const glRaster = {...raster, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw', blend: null, writeMask: 0};
    const entry = records.get(recordKey(compiled, gpu, geometry, glRaster, textures) + '\u0001' + topology);
    if (!entry) fail('PREPARE', 'A shadow caster program or binding changed; call prepare()');
    return entry;
  }
  function submit(passes) {
    updated.clear();
    if (!passes.length) return;
    meshes.write(); blurMeshes.write();
    const encoder = device.createCommandEncoder({label});
    for (const pass of passes) {
      // VSM blur passes: renderer.clear() on an RG16F target with a depth buffer.
      const colorAttachments = pass.color ? [{view: pass.color, clearValue: {r: 0, g: 0, b: 0, a: 0}, loadOp: 'clear', storeOp: 'store'}] : [];
      const p = encoder.beginRenderPass({label, colorAttachments,
        depthStencilAttachment: {view: pass.view, depthClearValue: 1, depthLoadOp: pass.clear ? 'clear' : 'load', depthStoreOp: 'store'}});
      // rows:'gl': WebGPU row index equals GL's bottom-up window y.
      const [x, y, w, h] = pass.rect;
      p.setViewport(x, y, w, h, 0, 1);
      for (const c of pass.commands) {
        if (!c.count || !c.instanceCount) continue;
        p.setPipeline(c.pipeline);
        p.setBindGroup(0, c.program.group, [c.program.offset]);
        if (c.record.textureGroup) p.setBindGroup(1, c.record.textureGroup);
        c.vertexBuffers.forEach((b, slot) => p.setVertexBuffer(slot, b));
        if (c.indexBuffer) { p.setIndexBuffer(c.indexBuffer, c.indexFormat); p.drawIndexed(c.count, c.instanceCount, c.first, 0, 0); }
        else p.draw(c.count, c.instanceCount, c.first, c.firstInstance ?? 0);
      }
      p.end();
    }
    device.queue.submit([encoder.finish()]);
  }
  return Object.freeze({
    prepare, render,
    /** Receiver binding of a shadow map's depth texture, or undefined. */
    binding: texture => maps.get(texture)?.binding ?? colorBindings.get(texture),
    dispose() {
      disposed = true;
      for (const m of maps.values()) destroyMap(m);
      maps.clear();
      for (const gpu of blurGeometries.values()) gpu.dispose();
      blurGeometries.clear(); blurMeshes.dispose(); fullScreenTri.dispose();
      shadowMaterialVertical.dispose(); shadowMaterialHorizontal.dispose();
      for (const byKey of geometries.values()) for (const gpu of byKey.values()) gpu.dispose();
      geometries.clear(); records.clear(); meshes.dispose();
      depthMaterialBase.dispose(); distanceMaterialBase.dispose();
    },
  });
}
