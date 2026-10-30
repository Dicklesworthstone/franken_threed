/** Synchronous selection and uniform capture for the native TSL graphics route.
 * Source object/material identities reach the retained builder and NodeFrame.
 * Pipeline preparation consumes these packets later and never repeats updates.
 * Geometry bytes still follow the bridge's existing upload/version contract.
 */
import {hasThreeNodeProgram, hasThreeNativeMedia, ThreeNodeProgramError} from './three_node_program.mjs';
import {getThreeNativeShadowPass, withThreeShadowMaterial, captureThreeShadowMaterialState,
  withThreeNativeCapture, markThreeNativeDrawCaptured, withThreeNativeObjectHooks} from './three_node_shadows.mjs';
import {inspectThreeLightingContext, inspectThreeNativeCamera} from './three_node_passes.mjs';
import {captureThreeNodeClipping} from './three_node_clipping.mjs';
import {inspectThreeNativeBatch, captureThreeNativeBatch} from './three_node_batch.mjs';
import {updateThreeCameraProjection} from './three_pass_snapshot.mjs';

const fail = (code, message) => { throw new ThreeNodeProgramError(code, message); };
const integer = value => Number.isSafeInteger(value) && value >= 0;
const transmissive = material => material.transmission > 0 || material.transmissionNode?.isNode === true;
const transparent = material => material.transparent === true || transmissive(material) || material.backdropNode?.isNode === true;

// Additional source material features use the retained native shader builder.
// Users must not add a dummy TSL slot to reach an existing implementation.
// Keep admission separate: every selected material is checked before updates.
const nativeFeatures = material => hasThreeNodeProgram(material) || hasThreeNativeMedia(material) || material?.isMaterial === true && (
  material.alphaHash === true ||
  material.isVolumeNodeMaterial === true ||
  material.isPointsNodeMaterial === true || material.isLineBasicNodeMaterial === true ||
  material.isLineDashedNodeMaterial === true || material.isLineDashedMaterial === true || material.isLine2NodeMaterial === true ||
  material.isPointsMaterial === true && (material.map != null || material.alphaMap != null) ||
  material.wireframe === true &&
    ['MeshBasic', 'MeshToon', 'MeshLambert', 'MeshPhong', 'MeshStandard', 'MeshPhysical'].some(name =>
      material[`is${name}Material`] === true || material[`is${name}NodeMaterial`] === true) ||
  material.bumpMap != null || material.displacementMap != null || material.lightMap != null ||
  ((material.isMeshStandardMaterial === true || material.isMeshStandardNodeMaterial === true ||
    material.isMeshPhysicalMaterial === true || material.isMeshPhysicalNodeMaterial === true) && material.envMap != null) ||
  (material.isMeshPhysicalMaterial === true || material.isMeshPhysicalNodeMaterial === true) && (
    ['sheen', 'iridescence', 'anisotropy', 'retroreflectivity', 'transmission', 'dispersion'].some(key => material[key] > 0) ||
    ['sheenColorMap', 'sheenRoughnessMap', 'iridescenceMap', 'iridescenceThicknessMap',
      'anisotropyMap', 'retroreflectivityMap', 'transmissionMap', 'thicknessMap'].some(key => material[key] != null)));
const litMaterial = material => material?.isMaterial === true && material.lights !== false &&
  (material.isVolumeNodeMaterial === true || ['MeshToon', 'MeshLambert', 'MeshPhong', 'MeshStandard', 'MeshPhysical'].some(name =>
    material[`is${name}Material`] === true || material[`is${name}NodeMaterial`] === true));

export function captureThreeNodeScene(three, root, camera, {getPrograms, renderer,
  overrideMaterial = null, maxNodes = 16384, maxDraws = 16384, uniforms = true,
  maxUniformBytes = 64 * 1024 * 1024, raster, getTextures, getStorage, destination = null, backgroundMesh = null,
  withFramebufferCopies = null, renderLists = null, renderId = renderer?.info?.calls,
  bundles = null, bundleDepth = 0} = {}) {
  if (!integer(maxNodes) || maxNodes < 1 || !integer(maxDraws) || maxDraws < 1 || !integer(maxUniformBytes))
    fail('LIMIT', 'Invalid native scene capture capacity');
  const sortObjects = renderer?.sortObjects === true;
  // The facade supplies its source controls. Standalone capture callers retain
  // the complete-list default; each control is observed at its source phase.
  const listEnabled = key => renderLists === null || renderLists[key] === true;
  const phases = {opaque: false, transparent: false, output: false};
  // Native PBR owns node/cube environments. Fixed companions and inactive
  // lighting must not acquire the legacy panorama owner for these inputs.
  const sceneEnvironmentNode = root?.environmentNode?.isNode === true || root?.environment?.isCubeTexture === true;
  // Keep the accepted source selection even when queued preparation observes a
  // later graph replacement. The fixed adapter must not apply scene.fog again.
  const sceneFogNode = !!root?.fogNode;
  const lightingContext = inspectThreeLightingContext(three, renderer?.contextNode);
  if (backgroundMesh !== null && (!(backgroundMesh instanceof three.Mesh) || backgroundMesh === root ||
      backgroundMesh.parent !== null || backgroundMesh.children.length || Array.isArray(backgroundMesh.material) ||
      !hasThreeNodeProgram(backgroundMesh.material)))
    fail('BACKGROUND', 'Native backgrounds require a separate childless mesh with one authored node material');
  const shadowPass = getThreeNativeShadowPass(renderer, overrideMaterial);
  // Match source RenderContexts identity for the ordinary-material fog
  // observer. Target objects with the same attachment state share a context.
  const sourceRenderContext = !shadowPass && bundles ? `${destination === null ? 'default' :
    [destination.textures.length, destination.texture.format, destination.texture.type, destination.samples,
      destination.depthBuffer, destination.stencilBuffer].join(':')}-${renderer.getMRT?.()?.id ?? 'default'}-${bundleDepth}` : undefined;
  const hierarchy = new Map(), objects = new Set(), bindings = [], pairs = new Map(), stack = [root], hookedObjects = new Set();
  let nativeOverrides = false;
  const phaseBindings = {opaque: new Set(), transparent: new Set()}, compiledBindings = new Set();
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
  const bundleMembers = new Map(), bundleObjects = new Set(), bundleSelections = [];
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
    const {object, parentVisible, parentBundle = null, suppressed = false} = pendingVisibility.pop();
    let bundle = parentBundle, blocked = suppressed;
    if (bundles && object.isBundleGroup === true) { bundleObjects.add(object); objects.add(object); }
    const active = !blocked && (object === backgroundMesh || parentVisible && object.visible !== false);
    if (bundles && active && object.isBundleGroup === true) {
      if (bundle) blocked = true; // The pinned backend does not execute nested bundles.
      else {
        bundle = bundles.select(object, camera, renderer.getRenderTarget?.() ?? null, renderer.getMRT?.() ?? null, bundleDepth);
        bundleSelections.push(bundle);
      }
    }
    if (bundle) { bundleMembers.set(object, bundle); bundleObjects.add(object); objects.add(object); }
    visibility.set(object, active);
    if (!active || blocked) { excludedObjects.add(object); objects.add(object); }
    // LOD mutates child visibility during source projection. Run it once before
    // admitting those children, then retain that selection through preparation.
    if (active && !blocked && object.isLOD && object.autoUpdate === true && object.layers.test(camera.layers)) {
      updateMatrices(); object.update(camera); lodCaptured = true;
    }
    const children = hierarchy.get(object);
    for (let i = children.length - 1; i >= 0; i--)
      pendingVisibility.push({object: children[i], parentVisible: active, parentBundle: bundle,
        suppressed: blocked || !!bundle && (bundle.warm || !uniforms)});
  }
  // Native owners capture probe/area, authored and many-light selection. The fixed owner must not
  // reread or reject these lights, including inactive lights and lights ignored
  // by a material's source lighting model. Their children remain traversable.
  const authoredLight = object => object.isLight === true && (object.colorNode || object.isSpotLight === true && object.map);
  const projectedLights = new Set([...hierarchy.keys()].filter(object => object.isLight === true &&
    visibility.get(object) && !bundleMembers.has(object) && object.layers.test(camera.layers)));
  // Eight is the fixed shader's storage capacity, not a source-light cutoff.
  // Select one native owner for the complete material list beyond that size,
  // including unlit companions and lighting-disabled passes. The source graph
  // decides each material's contributions; no fixed frame can drop a tail or
  // reinterpret a shadow's index in a partially filtered light list.
  const nativeManyLightScene = !shadowPass && projectedLights.size > 8;
  const nativeLights = new Set([...hierarchy.keys()].filter(object => object.isLightProbe === true || object.isRectAreaLight === true ||
    authoredLight(object) || nativeManyLightScene && object.isLight === true));
  const selectedLight = object => !shadowPass && renderer?.lighting?.enabled !== false &&
    projectedLights.has(object);
  const nativeProbeScene = [...nativeLights].some(object => object.isLightProbe === true && selectedLight(object));
  const nativeAreaScene = [...nativeLights].some(object => object.isRectAreaLight === true && selectedLight(object));
  const nativeAuthoredScene = [...nativeLights].some(object => authoredLight(object) && selectedLight(object));
  // Non-PBR lighting models have no rectangular-light contribution, but their
  // source light node still constructs its LTC inputs and runs its lifecycle.
  const nativeLightMaterial = material => (nativeProbeScene || nativeAreaScene || nativeAuthoredScene) && litMaterial(material);
  // Basic uses a source lighting model for AO and baked light as well. Even an
  // otherwise fixed material must reach that model when the pass supplies AO.
  const nativeContextMaterial = material => lightingContext !== null && material?.lights !== false &&
    (litMaterial(material) || material?.isMeshBasicMaterial === true || material?.isMeshBasicNodeMaterial === true);
  // Authored scene nodes and cube environments use the source PBR graph.
  // Material environments take precedence, while unused scene inputs stay inert.
  const nativeEnvironmentMaterial = material => (root.environmentNode?.isNode === true || root.environment?.isCubeTexture === true) &&
    renderer?.lighting?.enabled !== false && material?.lights !== false && material?.fragmentNode == null &&
    !material?.envNode && !material?.envMap && ['MeshStandard', 'MeshPhysical'].some(name =>
      material?.[`is${name}Material`] === true || material?.[`is${name}NodeMaterial`] === true);
  const nativeFogMaterial = material => sceneFogNode && material?.fog === true;
  // A native shadow receiver must share one shadow owner with every caster and
  // ordinary material in its scene. Keep genuinely fixed-only scenes on their
  // established route; explicit node materials, ShadowMaterial, transmitted
  // shadows and point-light cubes opt the complete scene into native compilation. Mixed material
  // classes then fail at the compiler instead of creating a second shadow map.
  const nativeShadowScene = !shadowPass && renderer?.shadowMap?.enabled === true &&
    [...hierarchy.keys()].some(object => visibility.get(object) && object !== backgroundMesh &&
      ((object.isPointLight === true || object.isLight === true &&
        (renderer.shadowMap.type === three.VSMShadowMap || renderer.shadowMap.transmitted === true)) &&
        object.castShadow === true && object.layers.test(camera.layers) ||
        object.geometry?.isInstancedBufferGeometry === true || object.isBatchedMesh === true ||
        (Array.isArray(object.material) ? object.material : [object.material]).some(material =>
          nativeFeatures(material) || nativeLightMaterial(material) || nativeContextMaterial(material) || nativeEnvironmentMaterial(material) || nativeFogMaterial(material) ||
          material instanceof three.NodeMaterial || material?.isShadowMaterial === true)));
  const mrt = destination ? renderer?.getMRT?.() ?? null : null;
  const nativeMRTScene = mrt !== null || (destination?.textures?.length ?? 0) > 1;
  const nativeClippingScene = [...hierarchy.keys()].some(object => visibility.get(object) &&
    object.isClippingGroup && object.enabled && object.layers.test(camera.layers) && (!shadowPass || object.clipShadows));
  // The source builder owns fragment depth and camera conventions. A fixed
  // color shader must never silently render a logarithmic or reversed scene.
  const nativeDepthScene = renderer?.logarithmicDepthBuffer === true || renderer?.reversedDepthBuffer === true;
  // The pinned wide-line addon updates raycast resolution in onBeforeRender.
  // Its ordinary companions need the same ordered source capture, including
  // when transparent ribbons copy their already-rendered opaque background.
  const nativeWideLineScene = [...hierarchy.keys()].some(object => visibility.get(object) &&
    object.layers.test(camera.layers) && (Array.isArray(object.material) ? object.material : [object.material])
      .some(material => material?.visible !== false && material && Object.getPrototypeOf(material) === three.Line2NodeMaterial?.prototype));
  // Source geometry constructors can schedule compute before their draw. Keep
  // ordinary companions in the same capture so their prior resource uses are
  // known when admitting those prerequisites, including native backgrounds.
  const nativeGeometryScene = [...hierarchy.keys()].some(object => visibility.get(object) &&
    object.layers.test(camera.layers) && (Array.isArray(object.material) ? object.material : [object.material])
      .some(material => material?.visible !== false && material?.geometryNode != null));
  // Occlusion results are read during source callbacks. Capture the complete
  // ordered scene so tested and ordinary companions share that boundary.
  let nativeOcclusionScene = false;
  for (const object of hierarchy.keys()) if (visibility.get(object) && object.layers.test(camera.layers)) {
    if (object.occlusionTest === true) {
      renderer?._occlusionObjects?.add(object); nativeOcclusionScene = true;
    } else if (renderer?._occlusionObjects?.has(object)) nativeOcclusionScene = true;
  }
  // Once queried, an object's callbacks may keep observing its last completed
  // result after the application disables queries. Retain that capture route.
  const nativeMaterial = material => nativeManyLightScene || nativeShadowScene || nativeMRTScene || nativeClippingScene || nativeDepthScene || nativeWideLineScene || nativeGeometryScene || nativeOcclusionScene
    ? material != null : nativeFeatures(material) || nativeLightMaterial(material) || nativeContextMaterial(material) || nativeEnvironmentMaterial(material) || nativeFogMaterial(material);
  for (const object of hierarchy.keys()) {
    if (!visibility.get(object)) continue;
    if (!object.material) continue;
    // Preserve the source renderer's report-and-skip branch in the fixed
    // projection owner; a native material must not turn a loop into a strip.
    if (object.isLineLoop === true) continue;
    if (shadowPass) {
      // The retained source selector, rather than the ordinary fixed route,
      // owns every renderable in a shadow pass, including excluded noncasters.
      objects.add(object);
      if (object.castShadow !== true && !(object.receiveShadow && renderer.shadowMap.type === three.VSMShadowMap)) continue;
    }
    if (excludedObjects.has(object)) continue;
    const bundle = bundleMembers.get(object);
    const instancedGeometry = object.geometry?.isInstancedBufferGeometry === true;
    const batch = object.isBatchedMesh === true;
    const originals = Array.isArray(object.material) ? object.material.slice() : [object.material];
    if (originals.length > maxNodes) fail('LIMIT', 'Native scene material array exceeds its budget');
    if (shadowPass && originals.some(original => original && original.allowOverride !== true))
      fail('SHADOW', 'Native shadow casters currently require the source shadow override material');
    const effective = originals.map(original => object !== backgroundMesh && original?.allowOverride === true && overrideMaterial !== null ? overrideMaterial : original);
    // A Mesh can use the source billboard shader too (for particle instances).
    // Ordinary Sprite objects without authored nodes retain the fixed path.
    const spriteMesh = (object.isMesh === true || object.isSprite === true) && effective.some(material =>
      object.isMesh === true && (material?.isSpriteMaterial === true || material?.isSpriteNodeMaterial === true) ||
      material?.isPointsMaterial === true || material?.isPointsNodeMaterial === true);
    if (!bundle && !instancedGeometry && !batch && !spriteMesh && object.geometry?.indirect == null && !effective.some(nativeMaterial)) continue;
    if (!shadowPass && effective.some((material, index) => material !== originals[index])) nativeOverrides = true;
    // A mesh's material groups share one geometry owner. Admit all its groups
    // through the source library conversion when any group needs native shading;
    // unsupported classes still fail the complete preflight below.
    const sprite = object instanceof three.Sprite && object.isSprite === true;
    const points = object instanceof three.Points && object.isPoints === true;
    const line = object instanceof three.Line && object.isLine === true;
    if (!(object instanceof three.Mesh || sprite || points || line) || object.intersectsFrustum !==
        (sprite ? three.Sprite.prototype : points ? three.Points.prototype : line ? three.Line.prototype :
          object.isSkinnedMesh ? three.SkinnedMesh.prototype : three.Mesh.prototype).intersectsFrustum)
      fail('OBJECT', 'Native node scenes require source meshes, sprites, points or lines with their source frustum implementation');
    if (sprite && Array.isArray(object.material))
      fail('MATERIAL', 'Source sprites require one material');
    if (batch) inspectThreeNativeBatch(three, object, fail);
    if (typeof object.onBeforeRender !== 'function' || typeof object.onAfterRender !== 'function')
      fail('HOOK', 'Native object render callbacks must be callable');
    if (object.onBeforeRender !== (batch ? three.BatchedMesh.prototype : three.Object3D.prototype).onBeforeRender ||
        object.onAfterRender !== three.Object3D.prototype.onAfterRender) hookedObjects.add(object);
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
            const binding = {object, geometry, material, original, shadowSide: side,
              passId: sides.length > 1 && side === three.BackSide ? 'backSide' : null, nodeProgram: null, raster: null};
            variants.push(binding); bindings.push(binding);
          }
        });
        byMaterial.set(original, variants);
      }
    } else for (const material of effective) if (material && !byMaterial.has(material)) {
        const doubleSide = (material.transparent === true || transmissive(material)) &&
          material.side === three.DoubleSide && material.forceSinglePass === false;
        const transmissionDoublePass = doubleSide && transmissive(material);
        const sides = doubleSide ? [three.BackSide, three.FrontSide] : [material.side], variants = [];
        for (const side of sides) {
          if (bindings.length >= maxDraws) fail('LIMIT', 'Native scene program binding capacity exceeded');
          const binding = {object, geometry, material, nodeProgram: null, raster: null,
            ...(doubleSide ? {colorSide: side} : {}), transmissionDoublePass,
            passId: doubleSide && side === three.BackSide ? 'backSide' : null};
          variants.push(binding); bindings.push(binding);
        }
        byMaterial.set(material, variants);
      }
    if (bundle) for (const variants of byMaterial.values()) for (const binding of variants) {
      binding.bundle = bundle.group;
      binding.bundleToken = bundles.token(bundle.entry, object, binding.material, binding.passId);
    }
    for (let i = 0; i < originals.length; i++) {
      const original = originals[i], material = effective[i];
      if (!original || !material) continue;
      const phase = object !== backgroundMesh && transparent(original) ? 'transparent' : 'opaque';
      for (const binding of byMaterial.get(shadowPass ? original : material)) phaseBindings[phase].add(binding);
    }
    pairs.set(object, {originals, effective, byMaterial, array: Array.isArray(object.material), geometry});
    objects.add(object);
  }
  for (const selection of bundleSelections) if (selection.warm && uniforms)
    for (const draw of selection.entry.draws) {
      bindings.push(draw.binding); compiledBindings.add(draw.binding);
      objects.add(draw.object); bundleObjects.add(draw.object);
    }
  const nativeHooks = hookedObjects.size > 0, nativeLifecycle = nativeHooks || nativeOverrides;
  if (nativeLifecycle && (root.isScene !== true || !bundles))
    fail('HOOK', 'Native object callbacks and overrides require a source Scene and its immutable geometry owner');
  if (!bindings.length) {
    if (!objects.size && !nativeLights.size && !shadowPass && !lodCaptured && !sceneEnvironmentNode && !sceneFogNode && renderLists === null) return null;
    phases.opaque = listEnabled('opaque'); phases.transparent = listEnabled('transparent'); phases.output = phases.opaque;
    // The facade marks every captured pass as having updated source matrices,
    // including an empty pass. Hidden skins still participate in CPU animation.
    updateMatrices();
    updateThreeCameraProjection(three, renderer, camera);
    // An exclusion-only capture still owns its source visibility. In particular,
    // a later visible=true cannot send an accepted hidden mesh to the fixed path.
    if (uniforms) for (const selection of bundleSelections) if (!selection.warm) bundles.commit(selection, []);
    return Object.freeze({objects, excludedObjects, visibility, nativeLights, bundleObjects, sortObjects, sceneEnvironmentNode, sceneFogNode, phases: Object.freeze(phases), bindings: Object.freeze([]), draws: Object.freeze([]),
      bytes: 0, background: false, prepareAttributes() {}, installAttributes() {},
      textureOwner: null, textureUses: Object.freeze([]), storageOwner: null, storageUses: Object.freeze([])});
  }
  inspectThreeNativeCamera(three, camera);
  if (typeof getPrograms !== 'function' || typeof raster !== 'function' || !renderer?._nodes?.nodeFrame)
    fail('SOURCE', 'Native scene capture needs its source compiler, renderer NodeFrame and raster adapter');
  if (renderer.xr?.isPresenting) fail('CAMERA', 'Native node scenes do not yet capture XR camera views');
  // Source construction can render nested preparation passes (PMREM does so).
  // Preserve this logical render's clock before any builder advances info.calls.
  const frame = renderer._nodes.nodeFrame;
  const frameStamp = {frameId: frame.frameId, time: frame.time, deltaTime: frame.deltaTime,
    renderId};
  let programs = null;
  const context = {scene: root, camera, lights: Object.freeze([]),
    lightingEnabled: renderer.lighting?.enabled !== false};
  const withBinding = (binding, run) => {
    const previous = context.clippingContext, previousPass = context.passId, previousSide = binding.material.side;
    context.clippingContext = binding.clippingContext ?? null;
    context.passId = binding.passId;
    try {
      if (binding.colorSide !== undefined) binding.material.side = binding.colorSide;
      return shadowPass ? withThreeShadowMaterial(three, renderer, binding.material, binding.original, material => {
        material.side = binding.shadowSide; return run();
      }) : run();
    } finally {
      if (binding.colorSide !== undefined) binding.material.side = previousSide;
      context.clippingContext = previous; context.passId = previousPass;
    }
  };
  updateMatrices();
  if (backgroundMesh?.matrixWorldAutoUpdate === true) backgroundMesh.updateMatrixWorld();
  updateThreeCameraProjection(three, renderer, camera);
  const clippingContexts = nativeClippingScene ? captureThreeNodeClipping(three, {root, camera, hierarchy, visibility,
    shadowPass: !!shadowPass, maxPlanes: maxNodes, backgroundMesh, fail}) : null;
  for (const binding of bindings) if (binding.nodeProgram === null)
    binding.clippingContext = clippingContexts?.get(binding.object) ?? null;
  const opaque = [], translucent = [], backgroundDraws = [], draws = [], lights = [], preparationBindings = new Set();
  let fixedDrawCandidates = false, expandedDraws = 0;
  {
    const projection = new three.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const frustum = camera.isArrayCamera ? new three.FrustumArray().setFromArrayCamera(camera) :
      new three.Frustum().setFromProjectionMatrix(projection, camera.coordinateSystem, camera.reversedDepth);
    const center = new three.Vector4();
    const visible = [{object: root, groupOrder: 0}];
    const append = (object, state, original, material, group, groupOrder, z, background = false) => {
      if (!original?.visible || !material) return;
      const variants = state.byMaterial.get(shadowPass ? original : material), binding = variants[0];
      expandedDraws += variants.length;
      if (expandedDraws > maxDraws) fail('LIMIT', 'Native source draw list exceeds its capacity');
      const {geometry, byMaterial} = state, range = object.isBatchedMesh ? {start: 0, count: Infinity} : geometry.drawRange;
      const extent = geometry.index ? geometry.index.count : geometry.attributes.position?.count;
      if (!integer(extent) || !integer(range?.start) || !(range.count === Infinity || integer(range.count)) ||
          (group && (!integer(group.start) || !(group.count === Infinity || integer(group.count)))))
        fail('GEOMETRY', 'Native node draw ranges require nonnegative integer starts and counts');
      const first = Math.min(extent, Math.max(range.start, object.isBatchedMesh ? 0 : group?.start ?? 0));
      const end = Math.min(extent, range.start + range.count, group && !object.isBatchedMesh ? group.start + group.count : Infinity);
      const listMaterial = Object.freeze({id: original.id, transparent: transparent(original), transmission: original.transmission ?? 0});
      const packet = {binding, object, geometry, material, listMaterial,
        group: group ? Object.freeze({...group}) : null, groupOrder, z, renderOrder: object.renderOrder,
        first, count: Math.max(0, end - first), frontFaceCW: object.isMesh === true && object.matrixWorld.determinant() < 0,
        nodeRangeCaptured: true, nodeUniforms: null, background, sourceMaterial: original, sourceGroup: group,
        transmissionDoublePass: transmissive(original) && original.side === three.DoubleSide && original.forceSinglePass === false,
        ...(shadowPass ? {shadowBindings: variants} : {colorBindings: variants})};
      const selection = bundleMembers.get(object);
      if (selection) {
        packet.nodeBundle = true; packet.bundleSelection = selection; packet.bundleScene = root;
        (listMaterial.transparent ? selection.transparent : selection.opaque).push(packet);
      } else (background ? backgroundDraws : listMaterial.transparent ? translucent : opaque).push(packet);
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
        else if (object.isLight && !bundleMembers.has(object)) lights.push(object);
        // Fixed draws are selected and merged by the core after this native
        // capture. Their order can precede draws[0], including when sorting is
        // disabled, so that native prefix is not a whole-scene boundary.
        else if ((uniforms || nativeLifecycle) && !objects.has(object) && object.material &&
            (Array.isArray(object.material) ? object.material : [object.material]).some(material => material?.visible))
          fixedDrawCandidates = true;
        else if (pairs.has(object) && (!object.frustumCulled || object.intersectsFrustum(frustum))) {
          const state = pairs.get(object), {geometry} = state;
          let z = 0;
          if (sortObjects) {
            if (object.isSprite) z = center.setFromMatrixPosition(object.matrixWorld).applyMatrix4(projection).z;
            else {
              const bounds = object.boundingSphere !== undefined ? object : geometry;
              if (bounds.boundingSphere === null) bounds.computeBoundingSphere();
              z = center.copy(bounds.boundingSphere.center).applyMatrix4(object.matrixWorld).applyMatrix4(projection).z;
            }
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
  if (nativeLifecycle && fixedDrawCandidates)
    fail('HOOK', 'Object callbacks in mixed fixed/native lists require one shared source draw capture');
  // RenderList counts requests during projection, before object/node updates
  // and independently of the renderer's opaque/transparent phase switches.
  const occlusionRequested = draws.some(draw => draw.object.occlusionTest === true);
  // The source projects lights for compileAsync as well as rendering. Keep the
  // selected source objects (including LOD visibility), not packed radiance;
  // its shared LightsNode/NodeFrame own light uniforms and update frequencies.
  context.lights = Object.freeze(lights);
  let textureOwner = null, storageOwner = null, bytes = 0;
  const textureInputs = () => bundles ? bundles.resources(getTextures()) : getTextures();
  const storageInputs = () => bundles ? bundles.resources(getStorage()) : getStorage();
  const compileBindings = selectedBindings => {
    for (const binding of selectedBindings) {
      compiledBindings.add(binding);
      if (binding.nodeProgram !== null) continue;
      withBinding(binding, () => {
        binding.nodeProgram = programs.compile(binding.material, binding.object, context);
        // Retain generated streams while this accepted render waits for submit.
        binding.programGeometry = programs.geometry?.(binding.nodeProgram) ?? binding.geometry;
        binding.raster = raster(binding.material, {mrt, textures: destination?.textures ?? null,
          topology: binding.nodeProgram.topology ?? (binding.nodeProgram.wireframe ? 'lines' : 'triangles')});
        if (shadowPass) binding.shadowState = Object.freeze(captureThreeShadowMaterialState(binding.material));
        Object.freeze(binding);
        if (programs.textures(binding.nodeProgram).length) {
          if (typeof getTextures !== 'function') fail('SOURCE', 'Native sampled graphs require the renderer texture owner');
          textureOwner ??= textureInputs();
        }
        if (programs.storage(binding.nodeProgram).length || programs.vertexStorage?.(binding.nodeProgram).length ||
            binding.geometry.indirect != null && !binding.object.isBatchedMesh) {
          if (typeof getStorage !== 'function') fail('SOURCE', 'Native storage graphs require the renderer storage owner');
          storageOwner ??= storageInputs();
        }
      });
    }
  };
  const chargeBindings = selectedBindings => {
    for (const binding of selectedBindings)
      bytes += (programs.attributeBytes?.(binding.nodeProgram) ?? 0) +
        binding.nodeProgram.groups.reduce((sum, group) => sum + group.byteLength, 0);
    if (!Number.isSafeInteger(bytes) || bytes > maxUniformBytes)
      fail('LIMIT', 'Native scene capture exceeds its pending uniform and attribute byte budget');
  };
  const prepareEnvironment = (selectedBindings, first) => {
    if (!camera.isArrayCamera || !camera.cameras.length) {
      first.object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, first.object.matrixWorld);
      first.object.normalMatrix.getNormalMatrix(first.object.modelViewMatrix);
    }
    programs.preparePMREM(selectedBindings.map(binding => binding.nodeProgram),
      {...frameContext, object: first.object, material: first.material},
      capturedDraws.length === 0 && (first.background || !fixedDrawCandidates));
  };
  // Select a phase before constructing its programs. A disabled list must not
  // compile authored graphs, acquire resources or run node lifecycle updates.
  // Transparent selection follows opaque updates, as in Renderer._renderScene.
  const preparePhase = (selected, phase, bundled = false) => {
    const selectedBindings = bundled ? [...new Set(selected.flatMap(draw => draw.shadowBindings ?? draw.colorBindings))]
      : [...phaseBindings[phase]].filter(binding => !binding.bundle);
    if (!selectedBindings.length) return;
    programs ??= getPrograms();
    const deferred = new Set();
    for (const draw of selected)
      for (const binding of draw.shadowBindings ?? draw.colorBindings) {
        if (!nativeLifecycle || draw.background) preparationBindings.add(binding);
        if (nativeLifecycle && !draw.background || uniforms && draw.object.isBatchedMesh) deferred.add(binding);
      }
    for (const binding of selectedBindings)
      if (!nativeLifecycle || !deferred.has(binding)) withBinding(binding, () => programs.inspect(binding.material, binding.object, context));
    const readyBindings = selectedBindings.filter(binding => !deferred.has(binding));
    compileBindings(readyBindings);
    if (!uniforms) return;
    const readyDraws = selected.filter(draw => (!nativeLifecycle || draw.background) && !draw.object.isBatchedMesh);
    for (const draw of readyDraws) chargeBindings(draw.shadowBindings ?? draw.colorBindings);
    if (readyDraws.length) prepareEnvironment(readyBindings, readyDraws[0]);
  };
  // BatchedMesh.customSort runs inside the source object callback. It may
  // allocate the first colors texture or change material sides before source
  // render-object construction. Prepare fresh bindings at that exact point;
  // an earlier queued draw must keep its own immutable program and resources.
  const prepareBatchBindings = (draw, selectedBindings) => {
    if (draw.object.geometry !== draw.geometry)
      fail('CHANGED', 'An object callback replaced its selected geometry; prepare the scene again');
    const prepared = selectedBindings.map(binding => ({...binding, nodeProgram: null, raster: null}));
    for (const binding of prepared)
      withBinding(binding, () => programs.inspect(binding.material, binding.object, context));
    compileBindings(prepared);
    chargeBindings(prepared);
    prepareEnvironment(prepared, draw);
    return prepared;
  };
  const frameContext = {...context, ...frameStamp};
  const textureUses = [], storageUses = [], vertexStorageUses = [], bundleGeometries = [], chargedGeometry = new Set(), deferredHookBindings = [];
  const captureVertexAttribute = (attribute, view) => {
    recordBuffer(attribute);
    const lease = storageOwner.capture(attribute);
    // Retain first so any subsequent proof/capture failure releases the upload.
    storageUses.push(lease); vertexStorageUses.push(lease);
    return storageOwner.vertexAttribute(lease, view);
  };
  const capturedDraws = [], capturedBundleDraws = [], sharedFogResources = [], indirectUploads = new Map();
  const preparedIndirectUses = [];
  let indirectCalls = 0;
  const captureIndirectUpload = (object, geometry) => {
    const attribute = geometry.indirect;
    if (attribute == null || object.isBatchedMesh) return null;
    if (typeof getStorage !== 'function') fail('SOURCE', 'Indirect draws require the renderer storage owner');
    storageOwner ??= storageInputs();
    // Geometries updates one attribute once per source render, before ordinary
    // node callbacks. Later CPU writes wait for the next admitted upload.
    let record = indirectUploads.get(attribute);
    if (!record) {
      const lease = storageOwner.capture(attribute, {indirect: true}); storageUses.push(lease);
      record = Object.freeze({attribute, lease, owner: storageOwner, byteLength: attribute.array.byteLength});
      indirectUploads.set(attribute, record);
    }
    return record;
  };
  let recordTexture = () => {}, recordBuffer = () => {};
  const captureDraw = draw => {
    // RenderObject receives the geometry's current range before this physical
    // pass's node lifecycle. A preceding pass may replace the range object.
    if (!draw.bundleWarm) draw.sourceRange = draw.geometry.drawRange;
    const beforeCopies = [], afterCopies = [];
    let copies = beforeCopies;
    const recordCopy = marker => {
      if (!textureOwner) {
        if (typeof getTextures !== 'function') fail('SOURCE', 'Framebuffer copies require the renderer texture owner');
        textureOwner = textureInputs();
      }
      textureUses.push(marker.lease); copies.push(marker);
    };
    const run = () => {
      if (!draw.bundleWarm && (!camera.isArrayCamera || !camera.cameras.length)) {
        draw.object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, draw.object.matrixWorld);
        draw.object.normalMatrix.getNormalMatrix(draw.object.modelViewMatrix);
      }
      const updateContext = {...frameContext, scene: draw.bundleScene ?? root, object: draw.object, material: draw.material,
        captureVertexAttribute, clippingContext: draw.binding.clippingContext,
        captureIndirect: () => captureIndirectUpload(draw.object, draw.geometry),
        ...(sourceRenderContext !== undefined && (sceneFogNode && draw.material.fog === true || draw.geometry.indirect != null)
          ? {fogRenderContext: sourceRenderContext,
          renderContext: {width: destination?.width ?? renderer.domElement?.width ?? 1,
            height: destination?.height ?? renderer.domElement?.height ?? 1}} : {}),
        ...(draw.nodeBundle ? {bundle: draw.binding.bundle, bundleToken: draw.binding.bundleToken,
          renderContext: {width: destination?.width ?? renderer.domElement?.width ?? 1,
            height: destination?.height ?? renderer.domElement?.height ?? 1}} : {}),
        ...(bundles ? {captureGeometry: (view, attributes) => bundles.captureGeometry(view, attributes, renderId)} : {})};
      if (draw.nodeBundle || updateContext.fogRenderContext !== undefined)
        updateContext.refreshType = programs.refresh(draw.binding.nodeProgram, updateContext);
      // ShadowNode recursively renders here. Keep the complete source before
      // sequence outside this draw's resource transaction, without hoisting it
      // ahead of the preceding draw or replaying false-returning callbacks.
      const beforeDone = typeof programs.before === 'function';
      if (beforeDone) programs.before(draw.binding.nodeProgram, updateContext);
      const capture = () => {
      markThreeNativeDrawCaptured(renderer, true);
      const vertexStart = vertexStorageUses.length;
      const captured = programs.captureDraw(draw.binding.nodeProgram, {...updateContext, beforeDone});
      for (const attribute of [...Object.values(draw.geometry.attributes), draw.geometry.index, draw.geometry.indirect])
        if (attribute) recordBuffer(attribute.isInterleavedBufferAttribute ? attribute.data : attribute);
      for (const binding of captured.storage) recordBuffer(binding.attribute);
      for (const binding of captured.textures) recordTexture(binding.texture);
      draw.nodeUniforms = captured.uniforms;
      draw.nodeAttributes = captured.attributes ?? null;
      if (captured.bundleGeometry) {
        // Bundle and ordinary fog observer records own their packet's lease.
        // Each accepted render needs an independent lifetime for that view.
        draw.bundleGeometry = captured.refreshType !== undefined ? captured.bundleGeometry.retain() : captured.bundleGeometry;
        bundleGeometries.push(draw.bundleGeometry);
        if (!chargedGeometry.has(draw.bundleGeometry.geometry)) {
          chargedGeometry.add(draw.bundleGeometry.geometry); bytes += draw.bundleGeometry.bytes;
          if (bytes > maxUniformBytes) fail('LIMIT', 'Native geometry exceeds the pending capture byte budget');
        }
      }
      const capturedGeometry = draw.bundleGeometry?.geometry ?? captured.programGeometry;
      if (capturedGeometry) {
        // Distinct edge versions used by the same program in one render need
        // distinct GPU index residencies, not consecutive writes before submit.
        const previous = draw.binding;
        draw.binding = Object.freeze({...previous, programGeometry: capturedGeometry});
        compiledBindings.delete(previous); compiledBindings.add(draw.binding);
      }
      if (draw.binding.nodeProgram.wireframe && !draw.object.isBatchedMesh && !draw.bundleWarm) {
        // The source doubles range/group bounds, then clips against the actual
        // derived index. Node updates may edit the selected range object.
        const range = draw.sourceRange, group = draw.sourceGroup, extent = captured.wireframeIndexCount;
        if (!integer(extent) || !integer(range?.start) || !(range.count === Infinity || integer(range.count)) ||
            group && (!integer(group.start) || !(group.count === Infinity || integer(group.count))))
          fail('GEOMETRY', 'Native wireframe ranges require bounded source draw and group ranges');
        draw.first = Math.min(extent, Math.max(range.start, group?.start ?? 0) * 2);
        const end = Math.min(extent, (range.start + range.count) * 2,
          group ? (group.start + group.count) * 2 : Infinity);
        draw.count = Math.max(0, end - draw.first);
      }
      if (draw.hookCapture && !draw.binding.nodeProgram.wireframe && !draw.object.isBatchedMesh) {
        const range = draw.sourceRange, group = draw.sourceGroup;
        const geometry = capturedGeometry ?? draw.geometry;
        const extent = geometry.index?.count ?? geometry.attributes.position?.count;
        if (!integer(extent) || !integer(range?.start) || !(range.count === Infinity || integer(range.count)) ||
            group && (!integer(group.start) || !(group.count === Infinity || integer(group.count))))
          fail('GEOMETRY', 'Native callback draw ranges must fit nonnegative source ranges');
        draw.first = Math.min(extent, Math.max(range.start, group?.start ?? 0));
        const end = Math.min(extent, range.start + range.count, group ? group.start + group.count : Infinity);
        draw.count = Math.max(0, end - draw.first);
        draw.frontFaceCW = draw.object.isMesh === true && draw.object.matrixWorld.determinant() < 0;
      }
      // Lease each post-update texture before after() or the next logical draw
      // can edit it. The shared owner snapshots CPU upload/sampler histories;
      // target leases hold generation metadata and observe contents in the FIFO.
      if (destination && captured.textures.some(binding =>
        destination.textures.includes(binding.texture) ||
        (destination.depthBuffer || destination.stencilBuffer) && destination.depthTexture === binding.texture))
        fail('FEEDBACK', 'A native draw cannot sample its own render target');
      const reuseResources = draw.bundleWarm && captured.refreshType !== three.RenderObjectRefreshType.FULL;
      if (captured.fogResources) {
        const resources = captured.fogResources;
        sharedFogResources.push({resources, descriptor: draw.binding.nodeProgram, context: updateContext});
        textureOwner ??= resources.textureOwner; storageOwner ??= resources.storageOwner;
        draw.nodeTextures = resources.textures; draw.nodeStorage = resources.storage;
        draw.nodeVertexStorage = resources.vertexStorage;
        for (const binding of draw.nodeTextures) textureUses.push(resources.textureOwner.retain(binding.lease));
      } else if (reuseResources) {
        textureOwner ??= draw.bundleTextureOwner; storageOwner ??= draw.bundleStorageOwner;
        for (const binding of draw.nodeTextures) textureUses.push(textureOwner.retain(binding.lease));
      } else {
      draw.nodeVertexStorage = Object.freeze(vertexStorageUses.slice(vertexStart));
      draw.nodeTextures = Object.freeze(captured.textures.map(binding => {
        const lease = textureOwner.captureBinding(binding);
        textureUses.push(lease);
        return Object.freeze({group: binding.group, binding: binding.binding, lease});
      }));
      draw.nodeStorage = Object.freeze(captured.storage.map(binding => {
        const lease = storageOwner.capture(binding.attribute, {writable: binding.writable === true});
        storageUses.push(lease);
        return Object.freeze({group: binding.group, binding: binding.binding, lease});
      }));
      }
      if (draw.nodeBundle) { draw.bundleTextureOwner = textureOwner; draw.bundleStorageOwner = storageOwner; }
      // r186 reads geometry.instanceCount before Mesh.count, after node/binding
      // updates and before after-updates. It does not infer a count from the
      // first instance attribute or clamp it to an attribute's capacity.
      // Capture it here even for zero instances: callbacks still ran, while the
      // backend will omit the empty draw. Later queued calls cannot replace it.
      if (!draw.bundleWarm) {
        const requestedInstances = draw.geometry.isInstancedBufferGeometry === true
          ? draw.geometry.instanceCount : draw.object.count;
        draw.instanceCount = draw.geometry.isInstancedBufferGeometry === true
          ? requestedInstances : requestedInstances === undefined ? 1 : Math.max(0, requestedInstances);
      }
      const hasIndirect = draw.bundleWarm ? draw.nodeIndirect != null : draw.geometry.indirect != null && !draw.object.isBatchedMesh;
      if ((!integer(draw.instanceCount) || draw.instanceCount > 0xffffffff) && !(hasIndirect && draw.instanceCount === Infinity))
        fail('OBJECT', 'Native node instance counts must fit a nonnegative 32-bit integer');
      if (!hasIndirect && draw.binding.nodeProgram.instanceCapacity !== undefined &&
          draw.instanceCount > draw.binding.nodeProgram.instanceCapacity)
        fail('OBJECT', 'Native node instance count exceeds its captured matrix or color capacity');
      const batchDraws = draw.bundleWarm ? draw.bundleRanges ?? null : captureThreeNativeBatch(draw.object, maxDraws - capturedDraws.length, fail, draw.instanceCount,
        {wireframe: draw.binding.nodeProgram.wireframe === true, indexCount: captured.wireframeIndexCount});
      if (draw.nodeBundle) draw.bundleRanges = batchDraws;
      // Indirect arguments are read by the GPU at execution, while their
      // source identity and byte offsets are selected before after-updates.
      // BatchedMesh owns its own direct subdraws and takes source precedence.
      const previousIndirect = draw.nodeIndirect;
      if (previousIndirect && draw.bundleWarm && (reuseResources || captured.indirect?.attribute !== previousIndirect.attribute)) {
        storageOwner ??= draw.bundleStorageOwner;
        const lease = storageOwner.refreshIndirect(previousIndirect.lease); storageUses.push(lease);
        draw.nodeIndirect = Object.freeze({...previousIndirect, lease});
      } else {
        const record = captured.fogResources?.indirect ?? captured.indirect;
        const attribute = draw.bundleWarm ? previousIndirect?.attribute : draw.geometry.indirect;
        draw.nodeIndirect = null;
        if (attribute != null && !draw.object.isBatchedMesh) {
          if (record?.attribute !== attribute)
            fail('INDIRECT', 'A node update replaced the indirect attribute after its geometry upload boundary');
          const lease = indirectUploads.get(attribute) === record ? record.lease : storageOwner.refreshIndirect(record.lease);
          if (lease !== record.lease) storageUses.push(lease);
          const selected = draw.bundleWarm ? previousIndirect.offsets : draw.geometry.indirectOffset;
          const offsets = Array.isArray(selected) ? selected.slice() : [selected];
          const geometry = capturedGeometry ?? draw.binding.programGeometry;
          const index = geometry.index, words = index ? 5 : 4, byteLength = record.byteLength;
          if (offsets.length > maxDraws || offsets.some(offset => !integer(offset) || offset % 4 ||
              !Number.isSafeInteger(offset + words * 4) || offset + words * 4 > byteLength))
            fail('INDIRECT', 'Indirect draw offsets require aligned, in-bounds 16-byte or indexed 20-byte records');
          const factor = draw.binding.nodeProgram.wireframe ? 2 : 1;
          const range = draw.sourceRange, group = draw.sourceGroup;
          const extent = index?.count ?? geometry.attributes.position?.count;
          const start = Math.max(range.start, group?.start ?? 0) * factor;
          const end = Math.min(extent, (range.start + range.count) * factor,
            group ? (group.start + group.count) * factor : Infinity);
          draw.nodeIndirect = Object.freeze({...record, lease, offsets: Object.freeze(offsets),
            skip: draw.bundleWarm ? previousIndirect.skip : end < start});
        }
      }
      if (draw.nodeIndirect && (indirectCalls += draw.nodeIndirect.offsets.length) > maxDraws)
        fail('LIMIT', 'Native indirect calls exceed the draw budget');
      if (captured.fogResources || reuseResources) {
        const retainStorage = lease => {
          const alias = draw.nodeIndirect ? storageOwner.captureIndirectAlias(lease, draw.nodeIndirect.lease) : null;
          const retained = alias ?? storageOwner.refreshStorage(lease) ?? storageOwner.retain(lease);
          storageUses.push(retained); return retained;
        };
        draw.nodeStorage = Object.freeze(draw.nodeStorage.map(binding =>
          Object.freeze({...binding, lease: retainStorage(binding.lease)})));
        draw.nodeVertexStorage = Object.freeze((draw.nodeVertexStorage ?? []).map(lease => {
          const retained = retainStorage(lease); vertexStorageUses.push(retained); return retained;
        }));
        if (draw.nodeVertexStorage.length && draw.bundleGeometry) {
          const refreshed = draw.bundleGeometry.refreshStorage(attribute => {
            for (const lease of draw.nodeVertexStorage) {
              const view = storageOwner.refreshVertexAttribute(lease, attribute);
              if (view !== null) return view;
            }
            return null;
          });
          if (refreshed) {
            bundleGeometries.push(refreshed); draw.bundleGeometry = refreshed;
            const previous = draw.binding;
            draw.binding = Object.freeze({...previous, programGeometry: refreshed.geometry});
            compiledBindings.delete(previous); compiledBindings.add(draw.binding);
          }
        }
      }
      if (draw.nodeBundle) draw.bundleStorageOwner = storageOwner;
      if (updateContext.fogRenderContext !== undefined && !captured.fogResources)
        programs.retainFogResources(draw.binding.nodeProgram, updateContext, Object.freeze({
          textures: draw.nodeTextures, storage: draw.nodeStorage, vertexStorage: draw.nodeVertexStorage,
          indirect: draw.nodeIndirect, textureOwner, storageOwner}));
      if (batchDraws === null && capturedDraws.length >= maxDraws)
        fail('LIMIT', 'Native source draw list exceeds its capacity');
      // WebGPUBackend.draw selects views after node/binding updates, before
      // updateAfter and the object's after hook. Nodes still see the parent and
      // run once per source material side, even when no child layer matches.
      const recordedViews = draw.bundleWarm ? draw.nodeViews : null;
      const viewPacket = (index, viewport) => {
        const uniforms = draw.nodeUniforms.slice(), location = draw.binding.nodeProgram.cameraIndex;
        if (location) {
          const packet = uniforms[location.group].slice();
          new DataView(packet.buffer, packet.byteOffset, packet.byteLength).setUint32(location.offset, index, true);
          uniforms[location.group] = packet; bytes += packet.byteLength;
          if (bytes > maxUniformBytes) fail('LIMIT', 'Native camera-index captures exceed the pending byte budget');
        }
        return Object.freeze({index, viewport: viewport ? Object.freeze(viewport) : null, uniforms: Object.freeze(uniforms)});
      };
      draw.nodeViews = null;
      if (draw.bundleWarm) {
        // The source replays its recorded camera-index bindings and layer
        // selection. Only the shared camera/object uniform data can refresh;
        // adding a view, changing layers or defining viewports cannot rerecord
        // a BundleGroup without its explicit version invalidation.
        if (recordedViews !== null) draw.nodeViews = Object.freeze(recordedViews.map(view => viewPacket(view.index, null)));
      } else if (camera.isArrayCamera && camera.cameras.length) {
        inspectThreeNativeCamera(three, camera);
        draw.nodeViews = Object.freeze(camera.cameras.flatMap((child, index) => {
          if (!draw.object.layers.test(child.layers)) return [];
          // r186's authored BundleGroup uses one GPURenderBundleEncoder for
          // ordinary targets. That encoder has no setViewport; viewport-less
          // child draws inherit the pass viewport when the bundle executes.
          if (draw.nodeBundle && child.viewport)
            fail('CAMERA', 'Pinned ArrayCamera BundleGroup recording does not support child viewports');
          const viewport = child.viewport ? ['x', 'y', 'z', 'w'].map(key => child.viewport[key]) : null;
          if (viewport?.some(value => !Number.isFinite(value) || value < 0))
            fail('CAMERA', 'Native ArrayCamera viewports require nonnegative finite components');
          return [viewPacket(index, viewport)];
        }));
      }
      copies = afterCopies;
      draw.occlusionTest = draw.object.occlusionTest === true;
      if (draw.nodeBundle && !draw.bundleWarm && draw.occlusionTest && !(camera.isArrayCamera && camera.cameras.length))
        three.warnOnce('WebGPUBackend: Occlusion queries can not be recorded into render bundles.');
      programs.after(draw.binding.nodeProgram, updateContext);
      draw.nodeFramebufferCopies = Object.freeze(beforeCopies);
      draw.nodeFramebufferAfterCopies = Object.freeze(afterCopies);
      Object.freeze(draw.nodeUniforms); Object.freeze(draw);
      // Source renderObjects contains one entry per material-side invocation,
      // not one per generated view or BatchedMesh subdraw. Warm refresh must
      // keep that callback/resource multiplicity while retaining all commands.
      if (draw.nodeBundle) capturedBundleDraws.push(draw);
      // One source material pass owns one NodeFrame/resource capture. The
      // backend loops its immutable subdraws without repeating user updates.
      // Copy effects remain at that pass's boundaries, not at every subdraw.
      if (batchDraws === null) capturedDraws.push(draw);
      else {
        const parts = batchDraws.length && draw.nodeViews?.length !== 0
          ? batchDraws : [{first: 0, count: 0, instanceCount: 0, firstInstance: 0}];
        // WebGPUBackend.draw selects a camera, then _draw executes the whole
        // batch. Expanding ranges before views would interleave overlapping
        // translucent views. Source callbacks and resource capture still run
        // once per material side; every subdraw shares that immutable history.
        // An empty view selection retains one inert packet for copy boundaries
        // and preparation, without issuing any geometry commands.
        const views = draw.nodeViews?.length ? draw.nodeViews.map(view => Object.freeze([view])) : [draw.nodeViews];
        const expanded = parts.length * views.length;
        if (expanded > maxDraws - capturedDraws.length)
          fail('LIMIT', 'Native batch views exceed the captured scene draw budget');
        let index = 0;
        for (const nodeViews of views) for (const range of parts) {
          capturedDraws.push(Object.freeze({...draw, ...range, nodeViews,
            nodeFramebufferCopies: index === 0 ? draw.nodeFramebufferCopies : Object.freeze([]),
            nodeFramebufferAfterCopies: index === expanded - 1 ? draw.nodeFramebufferAfterCopies : Object.freeze([])}));
          index++;
        }
      }
      };
      const captureStorage = () => storageOwner ? storageOwner.captureDispatch(capture) : capture();
      if (textureOwner) textureOwner.captureDispatch(captureStorage);
      else captureStorage();
    };
    if (withFramebufferCopies) withFramebufferCopies(recordCopy, run);
    else run();
  };
  const capturePhase = (selected, phase, bundled = false) => {
    // Source compileAsync selects object callbacks first; geometry-node updates
    // run later at its pipeline preparation boundary. Render capture runs them
    // in draw order below, before a DoubleSide split.
    if (!uniforms && !nativeLifecycle) for (const draw of selected) if (draw.object.isBatchedMesh)
      draw.object.onBeforeRender(renderer, root, camera, draw.geometry, draw.sourceMaterial, draw.sourceGroup);
    preparePhase(selected, phase, bundled);
    if (!uniforms && !nativeLifecycle) return;
    const captureSource = (draw, transmissionBack = false) => {
      const previous = renderer._handleObjectFunction;
      renderer._handleObjectFunction = (object, material, scene, selectedCamera, _lights, group, clippingContext, passId = null) =>
        withThreeNativeObjectHooks(renderer, () => {
          if (object !== draw.object || scene !== root || selectedCamera !== camera || group !== draw.sourceGroup)
            fail('HOOK', 'An object callback changed its source draw selection');
          if (object.geometry !== draw.geometry)
            fail('CHANGED', 'An object callback replaced its selected geometry; prepare the scene again');
          const binding = {...draw.binding, material, nodeProgram: null, raster: null, passId,
            clippingContext: clippingContext ?? draw.binding.clippingContext,
            ...(shadowPass ? {original: draw.sourceMaterial, shadowSide: material.side} : {colorSide: material.side}),
            ...(draw.nodeBundle ? {bundleToken: bundles.token(draw.bundleSelection.entry, object, material, passId)} : {})};
          if (!uniforms) {
            // Source compileAsync first queues every render-object selection,
            // including before/after callbacks, then constructs the pipelines.
            deferredHookBindings.push(binding); preparationBindings.add(binding); return;
          }
          const current = {...draw, material};
          const prepared = prepareBatchBindings(current, [binding])[0];
          captureDraw({...current, binding: prepared, hookCapture: true, transmissionBack});
        }, false);
      try {
        withThreeNativeObjectHooks(renderer, () => {
          if (shadowPass) shadowPass.renderObject(draw.object, root, camera, draw.geometry,
            draw.sourceMaterial, draw.sourceGroup, null, draw.binding.clippingContext, transmissionBack ? 'backSide' : null);
          else three.Renderer.prototype.renderObject.call(renderer, draw.object, root, camera, draw.geometry,
            draw.sourceMaterial, draw.sourceGroup, null, draw.binding.clippingContext, transmissionBack ? 'backSide' : null);
        });
      } finally { renderer._handleObjectFunction = previous; }
    };
    if (!shadowPass) {
      const captureColor = (draw, side = null, transmissionBack = false) => {
        if (nativeLifecycle && !draw.background) return captureSource(draw, transmissionBack);
        if (!uniforms) return;
        let selectedBindings;
        if (draw.object.isBatchedMesh) {
          draw.object.onBeforeRender(renderer, root, camera, draw.geometry, draw.sourceMaterial, draw.sourceGroup);
          const doubleSide = draw.material.transparent === true && draw.material.side === three.DoubleSide &&
            draw.material.forceSinglePass === false;
          const sides = doubleSide ? [three.BackSide, three.FrontSide] : [draw.material.side];
          selectedBindings = prepareBatchBindings(draw, sides.map(colorSide => ({...draw.binding, colorSide,
            passId: doubleSide && colorSide === three.BackSide || transmissionBack ? 'backSide' : null})));
        } else selectedBindings = draw.colorBindings.filter(binding => side === null || binding.colorSide === side);
        // Renderer.renderObject executes both sides consecutively for ordinary
        // alpha transparency. Each physical draw receives source NodeFrame
        // updates; its FRAME/RENDER markers still deduplicate shared nodes.
        for (const binding of selectedBindings)
          withBinding(binding, () => captureDraw({...draw, binding, transmissionBack}));
      };
      const transparentDraws = selected.filter(draw => !draw.background && draw.listMaterial.transparent);
      const doublePassDraws = transparentDraws.filter(draw => draw.transmissionDoublePass);
      for (const draw of selected) if (draw.background || !draw.listMaterial.transparent) captureColor(draw);
      // r186 _renderTransparents renders all transmissive back faces before
      // the complete transparent list. Every participating material already
      // has the selected side when any of that phase's node callbacks run.
      const sides = new Map(doublePassDraws.map(draw => [draw.sourceMaterial, draw.sourceMaterial.side]));
      try {
        for (const material of sides.keys()) material.side = three.BackSide;
        for (const draw of doublePassDraws) captureColor(draw, three.BackSide, true);
        for (const material of sides.keys()) material.side = three.FrontSide;
        for (const draw of transparentDraws)
          captureColor(draw, draw.transmissionDoublePass ? three.FrontSide : null);
      } finally {
        for (const [material, side] of sides) material.side = side;
      }
      return;
    }
    for (const draw of selected) {
      if (nativeLifecycle) { captureSource(draw); continue; }
      // Execute the pinned caster selector and Renderer.renderObject lifecycle
      // once per source list item. Its temporary shared override remains live
      // through native uniform/resource capture and is restored by the source.
      const previous = renderer._handleObjectFunction;
      renderer._handleObjectFunction = (object, material, scene, selectedCamera, lightsNode, group, clippingContext, passId = null) => {
        if (object !== draw.object || material !== draw.material || scene !== root || selectedCamera !== camera || group !== draw.sourceGroup)
          fail('SHADOW', 'A shadow callback changed the source draw selection');
        const binding = object.isBatchedMesh
          ? prepareBatchBindings(draw, [{...draw.shadowBindings[0], shadowSide: material.side, passId}])[0]
          : draw.shadowBindings.find(candidate => candidate.shadowSide === material.side);
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
  const capturedBundles = [];
  const capture = () => {
    // Source bundles update before the ordinary lists, while the backend
    // executes their recorded GPU commands after the ordinary scene draws.
    // Warm records retain physical draw order and ignore current list flags.
    if (uniforms) for (const selection of bundleSelections) {
      const first = capturedBundleDraws.length;
      if (selection.warm) {
        programs ??= getPrograms();
        for (const recorded of selection.entry.draws) {
          textureOwner ??= recorded.bundleTextureOwner;
          storageOwner ??= recorded.bundleStorageOwner;
          chargeBindings([recorded.binding]);
          captureDraw({...recorded, bundleSelection: selection, bundleWarm: true});
        }
      } else {
        if (listEnabled('opaque')) capturePhase(selection.opaque, 'opaque', true);
        if (listEnabled('transparent')) capturePhase(selection.transparent, 'transparent', true);
      }
      // A later ordinary draw may invalidate this group. Record the source
      // version at the end of this bundle, not at the end of the scene.
      capturedBundles.push({selection, draws: capturedBundleDraws.slice(first),
        version: selection.warm ? selection.entry.version : selection.group.version});
    }
    phases.opaque = listEnabled('opaque');
    if (phases.opaque) capturePhase(draws.filter(draw => draw.background || !draw.listMaterial.transparent), 'opaque');
    phases.transparent = listEnabled('transparent');
    if (phases.transparent) capturePhase(draws.filter(draw => !draw.background && draw.listMaterial.transparent), 'transparent');
    // The source output transform renders an opaque quad after scene updates.
    // Capture this later decision separately from the accepted scene list.
    phases.output = listEnabled('opaque');
    if (sharedFogResources.length) {
      const refresh = () => {
        const replacements = new Map();
        for (const {resources, descriptor, context} of sharedFogResources) {
          let textures = replacements.get(resources.textures);
          if (!textures) {
            textures = Object.freeze(resources.textures.map(binding => {
              const lease = resources.textureOwner.captureShared?.(binding.lease);
              if (!lease) return binding;
              textureUses.push(lease);
              return Object.freeze({...binding, lease});
            }));
            replacements.set(resources.textures, textures);
          }
          programs.retainFogResources(descriptor, context, Object.freeze({...resources, textures}));
        }
        // Source writes from later FULL draws also reach an earlier SHARED
        // binding before submission. Freeze that final accepted image here;
        // a subsequent queued frame must not change these selected pixels.
        for (let i = 0; i < capturedDraws.length; i++) {
          const textures = replacements.get(capturedDraws[i].nodeTextures);
          if (textures) capturedDraws[i] = Object.freeze({...capturedDraws[i], nodeTextures: textures});
        }
      };
      if (textureOwner) textureOwner.captureDispatch(refresh);
      else refresh();
    }
  };
  // A deferred request starts with its accepted selector, then source callbacks
  // may change it for this and subsequent draws. Never reinstall it per object.
  const previousOverride = root.overrideMaterial;
  const installOverride = nativeLifecycle && previousOverride !== overrideMaterial;
  if (installOverride) root.overrideMaterial = overrideMaterial;
  try {
    withThreeNativeCapture(renderer, (record, buffer) => {
      recordTexture = record; recordBuffer = buffer;
      // Fixed companions may sort before these native draws. Their reads are
      // not captured here, so this mixed pass cannot prove a compute prefix.
      if (fixedDrawCandidates) markThreeNativeDrawCaptured(renderer);
      capture();
      if (!uniforms) {
        compileBindings(deferredHookBindings);
        const captureStorage = () => storageOwner ? storageOwner.captureDispatch(() => {}) : undefined;
        if (textureOwner) textureOwner.captureDispatch(captureStorage);
        else captureStorage();
      }
    }, destination);
    storageOwner?.validate(storageUses);
    for (const {selection, draws: packets, version} of capturedBundles)
      bundles.commit(selection, packets, version);
  } catch (error) {
    for (const lease of textureUses) textureOwner.release(lease);
    for (const lease of storageUses) storageOwner.release(lease);
    for (const lease of bundleGeometries) lease.release();
    throw error;
  } finally {
    if (installOverride && root.overrideMaterial === overrideMaterial) root.overrideMaterial = previousOverride;
  }
  draws.splice(0, draws.length, ...capturedDraws);
  let preparedAttributes = null, preparedBindings = null;
  const acceptedBindings = Object.freeze([...compiledBindings]);
  const precompile = run => {
    const previous = renderer._isPreCompiling; renderer._isPreCompiling = true;
    try { return run(); } finally { renderer._isPreCompiling = previous; }
  };
  const publishAttributes = () => {
    for (const attributes of preparedAttributes ?? []) attributes.install();
    // Publish at preparation's FIFO boundary. Unchanged upload versions retain
    // GPU-authored bytes, including the buffers shared by geometry compute.
    if (vertexStorageUses.length) storageOwner.withReads(vertexStorageUses, () => {});
    if (preparedIndirectUses.length) storageOwner.withReads(preparedIndirectUses, () => {});
  };
  function* preparationSteps(updates) {
    let dirty = true;
    if (!uniforms && preparedAttributes === null) {
      preparedAttributes = []; const ready = [];
      for (const binding of preparationBindings) {
        const compute = updates && programs.hasGeometryCompute?.(binding.nodeProgram);
        const updateContext = {...frameContext, ...(compute ? {renderId: frame.renderId, frameId: frame.frameId, time: frame.time, deltaTime: frame.deltaTime} : {}),
          object: binding.object, material: binding.material, clippingContext: binding.clippingContext, captureVertexAttribute,
          captureIndirect: () => captureIndirectUpload(binding.object, binding.geometry),
          ...(bundles ? {captureGeometry: (view, attributes) => bundles.captureGeometry(view, attributes, renderId)} : {})};
        withBinding(binding, () => {
          // Public source compilation executes before -> geometry -> current;
          // raw graph/pipeline preparation remains inert. Keep the immutable
          // geometry upload before current callbacks can mutate its CPU arrays.
          const captured = compute ? precompile(() => programs.captureDraw(binding.nodeProgram, updateContext)) : null;
          const attributes = compute ? captured.attributes : programs.captureAttributes?.(binding.nodeProgram, updateContext);
          if (attributes) { preparedAttributes.push(attributes); attributes.install(); }
          const indirect = compute ? captured.indirect : captureIndirectUpload(binding.object, binding.geometry);
          if (indirect) preparedIndirectUses.push(indirect.lease);
          if (bundles) {
            const lease = captured?.bundleGeometry ?? bundles.captureGeometry(attributes?.geometry ?? binding.programGeometry, null, renderId);
            bundleGeometries.push(lease);
            ready.push(Object.freeze({...binding, programGeometry: lease.geometry}));
          } else ready.push(binding);
        });
        preparedBindings = Object.freeze(ready.slice()); dirty = true;
        if (compute) {
          publishAttributes(); yield; dirty = false;
          // Pinned compileAsync completes this object's pipeline and after
          // callbacks before advancing to the next render object. No await may
          // leave the renderer's precompilation flag installed.
          withBinding(binding, () => precompile(() => programs.after(binding.nodeProgram, updateContext)));
        }
      }
    }
    if (dirty) { publishAttributes(); yield; }
  }
  const geometryCompilation = !uniforms && [...preparationBindings].some(binding => programs.hasGeometryCompute?.(binding.nodeProgram));
  return Object.freeze({objects, excludedObjects, visibility, nativeLights, bundleObjects,
    get bundleGeometries() { return Object.freeze(bundleGeometries.slice()); },
    sortObjects, sceneEnvironmentNode, sceneFogNode, occlusionRequested, phases: Object.freeze(phases),
    get bindings() { return preparedBindings ?? acceptedBindings; }, draws: Object.freeze(draws), bytes, background: backgroundMesh !== null,
    prepareAttributes() { for (const step of preparationSteps(false)) { /* publish without node updates */ } },
    ...(geometryCompilation ? {async compileAttributes(prepare) { for (const step of preparationSteps(true)) await prepare(); }} : {}),
    installAttributes() { for (const draw of draws) draw.nodeAttributes?.install(); },
    textureOwner, textureUses: Object.freeze(textureUses), storageOwner,
    get storageUses() { return Object.freeze(storageUses.slice()); }});
}
