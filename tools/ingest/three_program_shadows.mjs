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
 * as precise. Explicit errors: VSM maps, line/point casters, wireframe casters,
 * reversed depth. A new light, a map-size change or a shadow-type change is a
 * preparation boundary (the map's GPU texture and the receivers' programs change).
 * No performance claim.
 */
import {createProgramMeshes} from './animation_program_mesh.mjs';

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
export function createThreeProgramShadows({three: T, device, support, bindingOf, sourceOf = o => o.geometry, clipping = () => null, label = 'f3d-program-shadows',
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
  const meshes = createProgramMeshes({device, format: null, depthFormat: DEPTH_FORMAT, sampleCount: 1, maxDraws, label,
    fail: (code, message) => { throw new ThreeProgramShadowError(code.replace(/^ANIMATION_RENDER_/, ''), message); }, scoped, lost});

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
      if (type === T.VSMShadowMap) fail('TYPE', 'VSM shadow maps are not admitted yet');
      if (light.isPointLight) {
        shadow.map = new T.WebGLCubeRenderTarget(shadowMapSize.x);
        shadow.map.depthTexture = new T.CubeDepthTexture(shadowMapSize.x, T.UnsignedIntType);
      } else {
        shadow.map = new T.WebGLRenderTarget(shadowMapSize.x, shadowMapSize.y);
        shadow.map.depthTexture = new T.DepthTexture(shadowMapSize.x, shadowMapSize.y, T.UnsignedIntType);
      }
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
    if (shadow.map.isWebGLCubeRenderTarget !== true && (shadow.map.width !== shadowMapSize.x || shadow.map.height !== shadowMapSize.y))
      shadow.map.setSize(shadowMapSize.x, shadowMapSize.y);
  }
  function mapState(target) {
    const depth = target.depthTexture, cube = target.isWebGLCubeRenderTarget === true;
    let m = maps.get(depth);
    if (m && m.width === target.width && m.height === target.height && m.cube === cube) return m;
    if (m) m.texture.destroy();
    const texture = device.createTexture({label: `${label}/${depth.name}`, size: [target.width, target.height, cube ? 6 : 1], format: DEPTH_FORMAT, usage: 16 | 4});
    const compare = depth.compareFunction != null;
    m = {texture, width: target.width, height: target.height, cube, depth,
      faces: Array.from({length: cube ? 6 : 1}, (_, face) => texture.createView({dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1})),
      binding: {view: texture.createView({dimension: cube ? 'cube' : '2d'}),
        sampler: device.createSampler({label, ...(compare ? {compare: 'less-equal', magFilter: 'linear', minFilter: 'linear'} : {}),
          addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge'}),
        sampleType: compare ? 'depth' : 'unfilterable-float'}};
    if (compare && depth.compareFunction !== T.LessEqualCompare) fail('MAP', 'Only LessEqual shadow comparisons are produced by r186 WebGLShadowMap');
    maps.set(depth, m);
    return m;
  }

  // ---- caster traversal (renderObject port) --------------------------------
  function casters(object, camera, light, type, frustum, visit) {
    if (object.visible === false) return;
    const visible = object.layers.test(camera.layers);
    if (visible && (object.isMesh || object.isLine || object.isPoints)) {
      if ((object.castShadow || (object.receiveShadow && type === T.VSMShadowMap)) && (frustum === null || !object.frustumCulled || object.intersectsFrustum(frustum))) {
        if (!object.isMesh) fail('CASTER', 'Line and point shadow casters are not admitted yet');
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
  function textureBindings(reflection, uniforms) {
    return reflection.textures.map(t => {
      const value = uniforms?.[t.name]?.value, texture = t.element === null ? value : value?.[t.element];
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
    const uniforms = refresh(depthMaterial), textures = textureBindings(compiled.program.reflection, uniforms);
    const raster = support.raster(depthMaterial, {side: depthMaterial.side});
    // rows:'gl' mirrors clip Y: GL's counter-clockwise front faces are clockwise here.
    const glRaster = {...raster, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw', blend: null, writeMask: 0};
    const key = recordKey(compiled, gpu, geometry, glRaster, textures);
    let entry = records.get(key);
    if (entry) return entry;
    const record = await meshes.add(gpu, {program: compiled.program, textures, raster: glRaster, topology: 'triangles'});
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
      allocate(light, type, typeChanged);
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
    meshes.begin();
    const passes = [];
    for (const light of lights) {
      const shadow = light.shadow;
      if (shadow === undefined) { (T.warn ?? console.warn)('WebGLShadowMap:', light, 'has no shadow.'); continue; }
      if (!updates(light)) continue;
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
    const clip = support.clippingState(clipping(), depthMaterial, shadowCamera, {shadows: true});
    const current = support.pack(reflection, uniforms, object, shadowCamera, bytes, {values: support.bindsClippingPlanes(depthMaterial) ? {clippingPlanes: clip.planes} : null});
    if (current.some((t, k) => (t ?? null) !== entry.textures[k].texture)) fail('PREPARE', 'Shadow caster textures changed; call prepare()');
    const g = object.geometry;
    const start = group ? group.start : 0, count = group ? group.count : Number.MAX_SAFE_INTEGER;
    const instanceCount = object.isInstancedMesh ? object.count : g.isInstancedBufferGeometry ? g.instanceCount : 1;
    const command = {};
    meshes.stage(entry.record, {programUniforms: bytes, first: start, count, frontFaceCW: object.matrixWorld.determinant() < 0, instanceCount}, command);
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
    const uniforms = refresh(depthMaterial), textures = textureBindings(compiled.program.reflection, uniforms);
    const raster = support.raster(depthMaterial, {side: depthMaterial.side});
    const glRaster = {...raster, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw', blend: null, writeMask: 0};
    const entry = records.get(recordKey(compiled, gpu, geometry, glRaster, textures));
    if (!entry) fail('PREPARE', 'A shadow caster program or binding changed; call prepare()');
    return entry;
  }
  function submit(passes) {
    updated.clear();
    if (!passes.length) return;
    meshes.write();
    const encoder = device.createCommandEncoder({label});
    for (const pass of passes) {
      const p = encoder.beginRenderPass({label, colorAttachments: [],
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
    binding: texture => maps.get(texture)?.binding,
    dispose() {
      disposed = true;
      for (const m of maps.values()) m.texture.destroy();
      maps.clear();
      for (const byKey of geometries.values()) for (const gpu of byKey.values()) gpu.dispose();
      geometries.clear(); records.clear(); meshes.dispose();
      depthMaterialBase.dispose(); distanceMaterialBase.dispose();
    },
  });
}
