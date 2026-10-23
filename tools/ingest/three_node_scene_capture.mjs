/** Synchronous selection and uniform capture for the native TSL graphics route.
 * Source object/material identities reach the retained builder and NodeFrame.
 * Pipeline preparation consumes these packets later and never repeats updates.
 * Geometry bytes still follow the bridge's existing upload/version contract.
 */
import {hasThreeNodeProgram, ThreeNodeProgramError} from './three_node_program.mjs';
import {getThreeNativeShadowPass, withThreeShadowMaterial, captureThreeShadowMaterialState,
  withThreeNativeCapture} from './three_node_shadows.mjs';

const fail = (code, message) => { throw new ThreeNodeProgramError(code, message); };
const integer = value => Number.isSafeInteger(value) && value >= 0;
const transparent = material => material.transparent === true || material.transmission > 0 ||
  material.transmissionNode?.isNode === true || material.backdropNode?.isNode === true;

export function captureThreeNodeScene(three, root, camera, {getPrograms, renderer,
  overrideMaterial = null, maxNodes = 16384, maxDraws = 16384, uniforms = true,
  maxUniformBytes = 64 * 1024 * 1024, raster, getTextures, getStorage, destination = null, backgroundMesh = null} = {}) {
  if (!integer(maxNodes) || maxNodes < 1 || !integer(maxDraws) || maxDraws < 1 || !integer(maxUniformBytes))
    fail('LIMIT', 'Invalid native scene capture capacity');
  if (backgroundMesh !== null && (!(backgroundMesh instanceof three.Mesh) || backgroundMesh === root ||
      backgroundMesh.parent !== null || backgroundMesh.children.length || Array.isArray(backgroundMesh.material) ||
      !hasThreeNodeProgram(backgroundMesh.material)))
    fail('BACKGROUND', 'Native backgrounds require a separate childless mesh with one authored node material');
  const shadowPass = getThreeNativeShadowPass(renderer, overrideMaterial);
  const hierarchy = new Map(), objects = new Set(), bindings = [], pairs = new Map(), stack = [root];
  // The source inserts a private background object into the render list. Admit
  // and prepare it without adding it to, or updating through, the user's root.
  if (backgroundMesh) stack.push(backgroundMesh);
  // Validate the complete hierarchy without inspecting inactive renderables.
  // The source projects only objects whose complete ancestor chain is visible.
  while (stack.length) {
    const object = stack.pop();
    if (!(object instanceof three.Object3D) || hierarchy.has(object) || !Array.isArray(object.children))
      fail('GRAPH', 'Expected an acyclic source Object3D hierarchy');
    if (hierarchy.size + stack.length + object.children.length >= maxNodes)
      fail('LIMIT', 'Native scene traversal exceeds its node budget');
    const children = object.children.slice(); hierarchy.set(object, children);
    for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
  }
  const visibility = new Map(), excludedObjects = new Set();
  let matricesUpdated = false, lodCaptured = false;
  const updateMatrices = () => {
    if (matricesUpdated) return;
    if (root.matrixWorldAutoUpdate === true) root.updateMatrixWorld();
    if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
    matricesUpdated = true;
  };
  const pendingVisibility = [{object: root, parentVisible: true}];
  if (backgroundMesh) pendingVisibility.push({object: backgroundMesh, parentVisible: true});
  while (pendingVisibility.length) {
    const {object, parentVisible} = pendingVisibility.pop();
    const active = object === backgroundMesh || parentVisible && object.visible !== false;
    visibility.set(object, active);
    if (!active) { excludedObjects.add(object); objects.add(object); }
    // LOD mutates child visibility during source projection. Run it once before
    // admitting those children, then retain that selection through preparation.
    if (active && object.isLOD && object.autoUpdate === true && object.layers.test(camera.layers)) {
      updateMatrices(); object.update(camera); lodCaptured = true;
    }
    const children = hierarchy.get(object);
    for (let i = children.length - 1; i >= 0; i--)
      pendingVisibility.push({object: children[i], parentVisible: active});
  }
  // A native shadow receiver must share one shadow owner with every caster and
  // ordinary material in its scene. Keep genuinely fixed-only scenes on their
  // established route; explicit node materials and ShadowMaterial opt the
  // complete shadow scene into native compilation. Unsupported mixed material
  // classes then fail at the compiler instead of creating a second shadow map.
  const nativeShadowScene = !shadowPass && renderer?.shadowMap?.enabled === true &&
    [...hierarchy.keys()].some(object => visibility.get(object) && object !== backgroundMesh &&
      (Array.isArray(object.material) ? object.material : [object.material]).some(material =>
        hasThreeNodeProgram(material) || material instanceof three.NodeMaterial || material?.isShadowMaterial === true));
  const nativeMaterial = material => nativeShadowScene ? material != null : hasThreeNodeProgram(material);
  for (const object of hierarchy.keys()) {
    if (!visibility.get(object)) continue;
    if (!object.material) continue;
    if (shadowPass) {
      // The retained source selector, rather than the ordinary fixed route,
      // owns every renderable in a shadow pass, including excluded noncasters.
      objects.add(object);
      if (object.castShadow !== true) continue;
    }
    const originals = Array.isArray(object.material) ? object.material.slice() : [object.material];
    if (originals.length > maxNodes) fail('LIMIT', 'Native scene material array exceeds its budget');
    if (shadowPass && originals.some(original => original && original.allowOverride !== true))
      fail('SHADOW', 'Native shadow casters currently require the source shadow override material');
    const effective = originals.map(original => object !== backgroundMesh && original?.allowOverride === true && overrideMaterial !== null ? overrideMaterial : original);
    for (let i = 0; i < originals.length; i++) {
      const original = originals[i], material = effective[i];
      if (shadowPass || original === material || (!nativeMaterial(original) && !nativeMaterial(material))) continue;
      // r186 renderObject inherits these values before applying an override.
      // The first profile admits the already-equivalent state without mutating
      // a shared override material or changing its callback identity.
      if ((original.positionNode?.isNode && original.positionNode !== material.positionNode) ||
          ['alphaTest', 'alphaMap', 'displacementMap', 'displacementScale', 'displacementBias'].some(key => original[key] !== material[key]) ||
          transparent(original) !== material.transparent)
        fail('EFFECT', 'Native node material overrides require matching inherited position, alpha, displacement and transparency state');
    }
    if (!effective.some(nativeMaterial)) continue;
    if (effective.some(material => material && !nativeMaterial(material)))
      fail('MATERIAL', 'A native node mesh cannot mix authored node and fixed materials');
    if (!(object instanceof three.Mesh) || object.intersectsFrustum !== three.Mesh.prototype.intersectsFrustum)
      fail('OBJECT', 'Native node scenes require ordinary meshes with the source frustum implementation');
    if (object.onBeforeRender !== three.Object3D.prototype.onBeforeRender || object.onAfterRender !== three.Object3D.prototype.onAfterRender)
      fail('HOOK', 'Native node object render hooks are not admitted');
    const geometry = object.geometry;
    if (!(geometry instanceof three.BufferGeometry) || !Array.isArray(geometry.groups) || geometry.groups.length > maxDraws)
      fail('GEOMETRY', 'Native node meshes require bounded source BufferGeometry groups');
    const owners = new Set(Object.values(geometry.attributes).map(attribute => attribute.isInterleavedBufferAttribute ? attribute.data : attribute));
    if (geometry.index) owners.add(geometry.index);
    if (object.isInstancedMesh) {
      if (object.instanceMatrix) owners.add(object.instanceMatrix);
      if (object.instanceColor) owners.add(object.instanceColor);
    }
    for (const owner of owners) {
      const prototype = owner.isInterleavedBuffer ? three.InterleavedBuffer.prototype : three.BufferAttribute.prototype;
      if (owner.onUploadCallback !== prototype.onUploadCallback)
        fail('HOOK', 'Native node attributes require the source upload callback');
    }
    const byMaterial = new Map();
    if (shadowPass) {
      for (const original of originals) if (original && !byMaterial.has(original)) {
        const variants = [];
        withThreeShadowMaterial(three, renderer, overrideMaterial, original, material => {
          const sides = material.transparent && material.side === three.DoubleSide && !material.forceSinglePass
            ? [three.BackSide, three.FrontSide] : [material.side];
          for (const side of sides) {
            if (bindings.length >= maxDraws) fail('LIMIT', 'Native scene program binding capacity exceeded');
            const binding = {object, geometry, material, original, shadowSide: side, nodeProgram: null, raster: null};
            variants.push(binding); bindings.push(binding);
          }
        });
        byMaterial.set(original, variants);
      }
    } else for (const material of effective) if (material && !byMaterial.has(material)) {
        if (bindings.length >= maxDraws) fail('LIMIT', 'Native scene program binding capacity exceeded');
        const binding = {object, geometry, material, nodeProgram: null, raster: null};
        byMaterial.set(material, binding); bindings.push(binding);
      }
    pairs.set(object, {originals, effective, byMaterial, array: Array.isArray(object.material), geometry});
    objects.add(object);
  }
  if (!bindings.length) {
    if (!objects.size && !shadowPass && !lodCaptured) return null;
    // The facade marks every captured pass as having updated source matrices,
    // including an empty pass. Hidden skins still participate in CPU animation.
    updateMatrices();
    if (camera.coordinateSystem !== three.WebGPUCoordinateSystem) {
      if (typeof camera.updateProjectionMatrix !== 'function') fail('CAMERA', 'The source camera cannot update its WebGPU projection');
      camera.coordinateSystem = three.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
    }
    // An exclusion-only capture still owns its source visibility. In particular,
    // a later visible=true cannot send an accepted hidden mesh to the fixed path.
    return Object.freeze({objects, excludedObjects, visibility, bindings: Object.freeze([]), draws: Object.freeze([]),
      bytes: 0, background: false, prepareAttributes() {}, installAttributes() {},
      textureOwner: null, textureUses: Object.freeze([]), storageOwner: null, storageUses: Object.freeze([])});
  }
  if (!(camera instanceof three.Camera) || camera.isArrayCamera || camera.isMultiViewCamera || camera.reversedDepth || renderer?.reversedDepthBuffer)
    fail('CAMERA', 'Native node scenes require an ordinary non-reversed camera');
  if (typeof getPrograms !== 'function' || typeof raster !== 'function' || !renderer?._nodes?.nodeFrame)
    fail('SOURCE', 'Native scene capture needs its source compiler, renderer NodeFrame and raster adapter');
  if (renderer.xr?.isPresenting) fail('CAMERA', 'Native node scenes do not yet capture XR camera views');
  // Source construction can render nested preparation passes (PMREM does so).
  // Preserve this logical render's clock before any builder advances info.calls.
  const frame = renderer._nodes.nodeFrame;
  const frameStamp = {frameId: frame.frameId, time: frame.time, deltaTime: frame.deltaTime,
    renderId: renderer.info.calls};
  const programs = getPrograms(), context = {scene: root, camera, lights: Object.freeze([]),
    lightingEnabled: renderer.lighting?.enabled !== false};
  const withBinding = (binding, run) => shadowPass
    ? withThreeShadowMaterial(three, renderer, binding.material, binding.original, material => {
      material.side = binding.shadowSide; return run();
    }) : run();
  // Every active binding is admitted before graph construction or application
  // draw updates. Previously hidden graphs are re-admitted on their first live
  // call, so a refusal cannot follow partial visible draw callbacks.
  for (const binding of bindings) withBinding(binding, () => programs.inspect(binding.material, binding.object, context));
  updateMatrices();
  if (backgroundMesh?.matrixWorldAutoUpdate === true) backgroundMesh.updateMatrixWorld();
  if (camera.coordinateSystem !== three.WebGPUCoordinateSystem) {
    if (typeof camera.updateProjectionMatrix !== 'function') fail('CAMERA', 'The source camera cannot update its WebGPU projection');
    camera.coordinateSystem = three.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  }
  const opaque = [], translucent = [], backgroundDraws = [], draws = [], lights = [], preparationBindings = new Set();
  let fixedDrawCandidates = false;
  {
    const projection = new three.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const frustum = new three.Frustum().setFromProjectionMatrix(projection, camera.coordinateSystem, camera.reversedDepth);
    const center = new three.Vector4(), sortObjects = renderer.sortObjects === true;
    const visible = [{object: root, groupOrder: 0}];
    const append = (object, state, original, material, group, groupOrder, z, background = false) => {
      if (!original?.visible || !material) return;
      const variants = shadowPass ? state.byMaterial.get(original) : null;
      const binding = variants?.[0] ?? state.byMaterial.get(material);
      for (const selected of variants ?? [binding]) preparationBindings.add(selected);
      if (!uniforms) return;
      if (opaque.length + translucent.length + backgroundDraws.length >= maxDraws) fail('LIMIT', 'Native source draw list exceeds its capacity');
      const {geometry, byMaterial} = state, range = geometry.drawRange;
      const extent = geometry.index ? geometry.index.count : geometry.attributes.position?.count;
      if (!integer(extent) || !integer(range?.start) || !(range.count === Infinity || integer(range.count)) ||
          (group && (!integer(group.start) || !(group.count === Infinity || integer(group.count)))))
        fail('GEOMETRY', 'Native node draw ranges require nonnegative integer starts and counts');
      const first = Math.min(extent, Math.max(range.start, group?.start ?? 0));
      const end = Math.min(extent, range.start + range.count, group ? group.start + group.count : Infinity);
      const listMaterial = Object.freeze({id: original.id, transparent: transparent(original), transmission: original.transmission ?? 0});
      const packet = {binding, object, geometry, material, listMaterial,
        group: group ? Object.freeze({...group}) : null, groupOrder, z, renderOrder: object.renderOrder,
        first, count: Math.max(0, end - first), frontFaceCW: object.matrixWorld.determinant() < 0,
        nodeRangeCaptured: true, nodeUniforms: null, background,
        ...(shadowPass ? {sourceMaterial: original, sourceGroup: group, shadowBindings: variants} : {})};
      (background ? backgroundDraws : listMaterial.transparent ? translucent : opaque).push(packet);
    };
    if (backgroundMesh) {
      const state = pairs.get(backgroundMesh);
      append(backgroundMesh, state, state.originals[0], state.effective[0], null, 0, 0, true);
    }
    while (visible.length) {
      const item = visible.pop(), object = item.object; let {groupOrder} = item;
      if (!visibility.get(object)) continue;
      if (object.layers.test(camera.layers)) {
        if (object.isGroup) groupOrder = object.renderOrder;
        else if (object.isLOD) { /* Selection was captured before binding admission. */ }
        else if (object.isLight) lights.push(object);
        // Fixed draws are selected and merged by the core after this native
        // capture. Their order can precede draws[0], including when sorting is
        // disabled, so that native prefix is not a whole-scene boundary.
        else if (uniforms && !objects.has(object) && object.material &&
            (Array.isArray(object.material) ? object.material : [object.material]).some(material => material?.visible))
          fixedDrawCandidates = true;
        else if (pairs.has(object) && (!object.frustumCulled || object.intersectsFrustum(frustum))) {
          const state = pairs.get(object), {geometry} = state;
          let z = 0;
          if (sortObjects) {
            if (geometry.boundingSphere === null) geometry.computeBoundingSphere();
            z = center.copy(geometry.boundingSphere.center).applyMatrix4(object.matrixWorld).applyMatrix4(projection).z;
          }
          if (state.array) for (const group of geometry.groups)
            append(object, state, state.originals[group.materialIndex], state.effective[group.materialIndex], group, groupOrder, z);
          else append(object, state, state.originals[0], state.effective[0], null, groupOrder, z);
        }
      }
      const children = hierarchy.get(object);
      for (let i = children.length - 1; i >= 0; i--) visible.push({object: children[i], groupOrder});
    }
    if (sortObjects) {
      const order = (a, b) => a.groupOrder - b.groupOrder || a.renderOrder - b.renderOrder;
      // Source WebGPU opaque sorting deliberately has no material-id key.
      opaque.sort((a, b) => order(a, b) || a.z - b.z || a.object.id - b.object.id);
      translucent.sort((a, b) => order(a, b) || b.z - a.z || a.object.id - b.object.id);
    }
    draws.push(...backgroundDraws, ...opaque, ...translucent);
  }
  // The source projects lights for compileAsync as well as rendering. Keep the
  // selected source objects (including LOD visibility), not packed radiance;
  // its shared LightsNode/NodeFrame own light uniforms and update frequencies.
  context.lights = Object.freeze(lights);
  // Preflight every material against the complete light list before the first
  // graph is constructed or application node update runs. The compiler admits
  // only lights used by a lit graph, so unused unsupported lights stay inert.
  for (const binding of bindings) withBinding(binding, () => programs.inspect(binding.material, binding.object, context));
  let textureOwner = null, storageOwner = null;
  for (const binding of bindings) withBinding(binding, () => {
    binding.nodeProgram = programs.compile(binding.material, binding.object, context);
    // Retain the selected private generated streams now. A later graph can
    // retire its compiler entry while this accepted render is still queued.
    binding.programGeometry = programs.geometry?.(binding.nodeProgram) ?? binding.geometry;
    binding.raster = raster(binding.material);
    if (shadowPass) binding.shadowState = Object.freeze(captureThreeShadowMaterialState(binding.material));
    Object.freeze(binding);
    if (programs.textures(binding.nodeProgram).length) {
      if (typeof getTextures !== 'function') fail('SOURCE', 'Native sampled graphs require the renderer texture owner');
      textureOwner ??= getTextures();
    }
    if (programs.storage(binding.nodeProgram).length) {
      if (typeof getStorage !== 'function') fail('SOURCE', 'Native storage graphs require the renderer storage owner');
      storageOwner ??= getStorage();
    }
  });
  let bytes = 0;
  for (const draw of draws) {
    for (const binding of draw.shadowBindings ?? [draw.binding])
      bytes += (programs.attributeBytes?.(binding.nodeProgram) ?? 0) +
        binding.nodeProgram.groups.reduce((sum, group) => sum + group.byteLength, 0);
    if (!Number.isSafeInteger(bytes) || bytes > maxUniformBytes)
      fail('LIMIT', 'Native scene capture exceeds its pending uniform and attribute byte budget');
  }
  const frameContext = {...context, ...frameStamp};
  const textureUses = [], storageUses = [];
  const capturedDraws = [];
  let recordTexture = () => {};
  const captureDraw = draw => {
      draw.object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, draw.object.matrixWorld);
      draw.object.normalMatrix.getNormalMatrix(draw.object.modelViewMatrix);
      const updateContext = {...frameContext, object: draw.object, material: draw.material};
      // ShadowNode recursively renders here. Keep the complete source before
      // sequence outside this draw's resource transaction, without hoisting it
      // ahead of the preceding draw or replaying false-returning callbacks.
      const beforeDone = typeof programs.before === 'function';
      if (beforeDone) programs.before(draw.binding.nodeProgram, updateContext);
      const capture = () => {
      const captured = programs.captureDraw(draw.binding.nodeProgram, {...updateContext, beforeDone});
      draw.nodeUniforms = captured.uniforms;
      draw.nodeAttributes = captured.attributes ?? null;
      // Lease each post-update texture before after() or the next logical draw
      // can edit it. The shared owner snapshots CPU upload/sampler histories;
      // target leases hold generation metadata and observe contents in the FIFO.
      draw.nodeTextures = Object.freeze(captured.textures.map(binding => {
        if (destination && binding.texture.renderTarget === destination)
          fail('FEEDBACK', 'A native draw cannot sample its own render target');
        const lease = textureOwner.captureBinding(binding);
        textureUses.push(lease);
        recordTexture(binding.texture);
        return Object.freeze({group: binding.group, binding: binding.binding, lease});
      }));
      draw.nodeStorage = Object.freeze(captured.storage.map(binding => {
        const lease = storageOwner.capture(binding.attribute);
        storageUses.push(lease);
        return Object.freeze({group: binding.group, binding: binding.binding, lease});
      }));
      // r186 reads Mesh.count after node/binding updates and before after-updates.
      // Capture it here even for zero instances: callbacks still ran, while the
      // backend will omit the empty draw. Later queued calls cannot replace it.
      const requestedInstances = draw.object.count;
      draw.instanceCount = requestedInstances === undefined ? 1 : Math.max(0, requestedInstances);
      if (!integer(draw.instanceCount) || draw.instanceCount > 0xffffffff)
        fail('OBJECT', 'Native node instance counts must fit a nonnegative 32-bit integer');
      if (draw.binding.nodeProgram.instanceCapacity !== undefined &&
          draw.instanceCount > draw.binding.nodeProgram.instanceCapacity)
        fail('OBJECT', 'Native node instance count exceeds its captured matrix or color capacity');
      programs.after(draw.binding.nodeProgram, updateContext);
      Object.freeze(draw.nodeUniforms); Object.freeze(draw);
      capturedDraws.push(draw);
      };
      const captureStorage = () => storageOwner ? storageOwner.captureDispatch(capture) : capture();
      if (textureOwner) textureOwner.captureDispatch(captureStorage);
      else captureStorage();
  };
  const capture = () => {
    for (const draw of draws) {
      if (!shadowPass) { captureDraw(draw); continue; }
      // Execute the pinned caster selector and Renderer.renderObject lifecycle
      // once per source list item. Its temporary shared override remains live
      // through native uniform/resource capture and is restored by the source.
      const previous = renderer._handleObjectFunction;
      renderer._handleObjectFunction = (object, material, scene, selectedCamera, lightsNode, group) => {
        if (object !== draw.object || material !== draw.material || scene !== root || selectedCamera !== camera || group !== draw.sourceGroup)
          fail('SHADOW', 'A shadow callback changed the source draw selection');
        const binding = draw.shadowBindings.find(candidate => candidate.shadowSide === material.side);
        if (!binding || Object.entries(binding.shadowState).some(([key, value]) => material[key] !== value))
          fail('CHANGED', 'A shadow callback changed its admitted override structure; prepare the scene again');
        captureDraw({...draw, binding});
      };
      try {
        shadowPass.renderObject(draw.object, root, camera, draw.geometry, draw.sourceMaterial,
          draw.sourceGroup, null, null, null);
      } finally { renderer._handleObjectFunction = previous; }
    }
  };
  try {
    if (draws.length) {
      const first = draws[0];
      first.object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, first.object.matrixWorld);
      first.object.normalMatrix.getNormalMatrix(first.object.modelViewMatrix);
      programs.preparePMREM(draws.map(draw => draw.binding.nodeProgram),
        {...frameContext, object: first.object, material: first.material}, first.background || !fixedDrawCandidates);
    }
    withThreeNativeCapture(renderer, record => {
      recordTexture = record;
      if (draws.length) capture();
      else {
        // Compile-only captures keep their source-owner transaction boundary
        // without executing before/current/after callbacks or publishing data.
        const captureStorage = () => storageOwner ? storageOwner.captureDispatch(capture) : capture();
        if (textureOwner) textureOwner.captureDispatch(captureStorage);
        else captureStorage();
      }
    });
    storageOwner?.validate(storageUses);
  } catch (error) {
    for (const lease of textureUses) textureOwner.release(lease);
    for (const lease of storageUses) storageOwner.release(lease);
    throw error;
  }
  draws.splice(0, draws.length, ...capturedDraws);
  let preparedAttributes = null;
  return Object.freeze({objects, excludedObjects, visibility, bindings: Object.freeze(bindings), draws: Object.freeze(draws), bytes, background: backgroundMesh !== null,
    prepareAttributes() {
      if (uniforms) return;
      // compileAsync uploads visible geometry even though this preparation
      // profile runs no application updates. Establish that initial resident
      // history at preparation, so later unmarked edits remain stale. Inactive
      // objects have no binding and do not acquire a source upload history.
      preparedAttributes ??= [...preparationBindings].map(binding => withBinding(binding, () => programs.captureAttributes?.(
        binding.nodeProgram, {...frameContext, object: binding.object, material: binding.material}))).filter(Boolean);
      for (const attributes of preparedAttributes) attributes.install();
    },
    installAttributes() { for (const draw of draws) draw.nodeAttributes?.install(); },
    textureOwner, textureUses: Object.freeze(textureUses), storageOwner, storageUses: Object.freeze(storageUses)});
}
