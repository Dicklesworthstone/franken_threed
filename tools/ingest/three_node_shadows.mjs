/** Retained r186 shadow lifecycle adapters. Source ShadowNode owns construction,
 * camera fitting and update frequency; the facade captures every resulting pass
 * for its own submission. No upstream renderer or backend is constructed here.
 */
const renderFunctions = new WeakMap(), captures = new WeakMap(), computePrerequisites = new WeakMap(), objectHooks = new WeakSet();
const reject = message => {
  const error = new Error(`THREE_NODE_PROGRAM_SHADOW: ${message}`);
  error.name = 'ThreeNodeProgramError'; error.code = 'THREE_NODE_PROGRAM_SHADOW'; throw error;
};

/** ShadowNode selects a truthy callable filter during source graph construction.
 * Do not invoke it during preflight: generated resources and writes pass through
 * the ordinary native node admission after the retained builder expands it.
 */
export function inspectThreeNativeShadowFilter(shadow) {
  const filter = shadow.filterNode || null;
  if (filter !== null && typeof filter !== 'function')
    reject('Native shadow filters require a callable source TSL filter');
  return filter;
}

/** Authored shadow compositions retain their own graph construction and node
 * updates. Only the pinned base owns caster selection and shadow materials;
 * arbitrary renderer/backend hooks do not acquire that privilege here. */
export function inspectThreeNativeShadowGraph(three, node, light) {
  if (node == null) return null;
  const base = three.ShadowBaseNode.prototype;
  if (!(node instanceof three.ShadowBaseNode) || node.light !== light ||
      ['getShadowMaterial', 'disposeShadowMaterial', 'getShadowRenderObjectFunction', 'setupShadowPosition']
        .some(key => node[key] !== base[key]))
    reject('Authored shadow compositions require a source ShadowBaseNode coupled to their selected light and the pinned caster/material helpers');
  return node;
}

/** CSM's private casters are Object3D instances, not scene Light instances.
 * Record actual source builder ancestry before those ShadowNodes are expanded.
 * A type/name flag alone never authorizes a virtual caster: its ancestor must
 * subsequently be the shadow graph selected by an admitted analytic light.
 * Cached source analytic nodes may retain an earlier authored graph, so use
 * their actual selected identity rather than the later public shadow slot. */
export function captureThreeNativeShadowGraphs(three, builder) {
  const ancestors = new Map(), addChain = builder.addChain;
  builder.addChain = function(node) {
    if (node instanceof three.ShadowNode && !(node.light instanceof three.Light)) {
      let roots = ancestors.get(node);
      if (!roots) ancestors.set(node, roots = new Set());
      for (const parent of this.chaining) if (parent instanceof three.ShadowBaseNode && !(parent instanceof three.ShadowNode))
        roots.add(parent);
    }
    return addChain.call(this, node);
  };
  return analyticNodes => {
    const roots = new Map(), children = new Map();
    for (const analytic of analyticNodes) {
      const root = analytic.shadowNode;
      if (root instanceof three.ShadowBaseNode) {
        inspectThreeNativeShadowGraph(three, root, analytic.light);
        roots.set(root, analytic.light);
      }
    }
    for (const [node, candidates] of ancestors) {
      const root = [...candidates].find(candidate => roots.has(candidate));
      if (root) children.set(node, Object.freeze({root, parentLight: roots.get(root), light: node.light, shadow: node.shadow}));
    }
    return {roots, children};
  };
}

/** The source cube constructor replaces its color texture without an owner
 * pointer. Register that private target before WGSL asks for its sample layout;
 * capture() records metadata only and observes the final depth attachment later.
 */
export function installThreeNativeShadowTargets(renderer, builder) {
  if (typeof renderer._targets?.capture !== 'function') return;
  const create = builder.createCubeRenderTarget;
  builder.createCubeRenderTarget = function(...args) {
    const target = create.apply(this, args);
    renderer._targets.capture(target);
    return target;
  };
}

/** Register the exact source caster selector before updateShadow installs it.
 * Direct ShadowNode subclasses retain the source update lifecycle, as the
 * registered SunLight addon does; their constructed graphs are still admitted
 * by the ordinary native compiler and their draws by the native scene capture.
 */
export function registerThreeNativeShadow(three, renderer, node, provenance = null) {
  if (!(node instanceof three.ShadowNode)) return false;
  const source = three.ShadowNode.prototype, base = three.ShadowBaseNode.prototype;
  const prototype = Object.getPrototypeOf(node), ordinary = prototype === source;
  const point = prototype === three.PointShadowNode?.prototype && node.light?.constructor === three.PointLight &&
    node.shadow?.isPointLightShadow === true;
  const sun = !ordinary && Object.getPrototypeOf(prototype) === source &&
    node.light?.isSunLight === true && node.shadow?.isSunLightShadow === true;
  const virtual = provenance !== null && ordinary;
  if (virtual) {
    inspectThreeNativeShadowGraph(three, provenance.root, provenance.parentLight);
    if (node.light !== provenance.light || node.shadow !== provenance.shadow ||
        !(node.light instanceof three.Object3D) || node.light instanceof three.Light ||
        !(node.light.target instanceof three.Object3D) || provenance.parentLight.constructor !== three.DirectionalLight ||
        node.shadow?.isDirectionalLightShadow !== true ||
        Object.getPrototypeOf(Object.getPrototypeOf(node.shadow)) !== three.LightShadow.prototype ||
        ['updateMatrices', 'getViewportCount', 'getViewport', 'getFrameExtents']
          .some(key => node.shadow[key] !== three.LightShadow.prototype[key]))
      reject('Virtual cascade casters require their captured source Object3D, directional shadow and pinned shadow camera lifecycle');
  }
  if ((!ordinary && !sun && !point) || !virtual && !(node.light instanceof three.Light) || node.shadow !== node.light.shadow ||
      !(node.shadow instanceof three.LightShadow) || !point && node.shadow.isPointLightShadow ||
      ordinary && !virtual && !(node.light.constructor === three.DirectionalLight || node.light.constructor === three.SpotLight))
    reject('Native shadow maps require directional, spot, point or registered SunLight source shadows');
  if (![three.BasicShadowMap, three.PCFShadowMap, three.VSMShadowMap].includes(renderer.shadowMap?.type))
    reject('Native shadow maps require Basic, PCF or VSM filtering');
  inspectThreeNativeShadowFilter(node.shadow);
  if (node.shadow.shadowNode != null && node.shadow.shadowNode !== node)
    reject('Shadow replacement nodes require an explicit native adapter');
  const passSource = point ? three.PointShadowNode.prototype : source;
  if (['setup', 'updateBefore', 'updateShadow', 'dispose', '_reset', 'getShadowFilterFn', 'setupShadowCoord', 'setupShadowFilter']
      .some(key => node[key] !== passSource[key]) ||
      ['getShadowMaterial', 'disposeShadowMaterial', 'getShadowRenderObjectFunction', 'setupShadowPosition']
        .some(key => node[key] !== base[key]) ||
      ['getUpdateBeforeType', 'getUpdateType', 'getUpdateAfterType', 'updateReference']
        .some(key => node[key] !== three.Node.prototype[key]) ||
      node.updateBeforeType !== 'render' || node.updateType !== 'none' || node.updateAfterType !== 'none')
    reject('Native shadow nodes must retain the pinned source lifecycle');
  if ((ordinary || point) && ['setupRenderTarget', 'setupShadow', 'renderShadow', 'vsmPass'].some(key => node[key] !== passSource[key]))
    reject('Native directional, spot and point shadows require the pinned source pass construction');
  if (sun && (node.shadow.getViewportCount() !== 2 || node.shadow.getFrameExtents().x !== 2 ||
      node.shadow.getFrameExtents().y !== 1))
    reject('Native SunLight shadows require the source two-cascade horizontal atlas');
  const renderObject = node.getShadowRenderObjectFunction(renderer, node.shadow);
  let functions = renderFunctions.get(renderer);
  if (!functions) renderFunctions.set(renderer, functions = new WeakMap());
  functions.set(renderObject, {node, renderObject, material: node.getShadowMaterial()});
  return true;
}

export function getThreeNativeShadowPass(renderer, overrideMaterial, renderObject = renderer.getRenderObjectFunction?.()) {
  if (renderObject == null) return null;
  const pass = renderFunctions.get(renderer)?.get(renderObject);
  // Ordinary application dispatch belongs to the normal native lifecycle.
  // Only an exact registered source selector acquires shadow-pass privileges.
  if (!pass) return null;
  if (pass.material !== overrideMaterial)
    reject('Registered shadow render-object functions require their source override material');
  return pass;
}

/** NodeFrame keeps the source RENDER update markers, including false returns.
 * Source updateShadow restores its public state on success, but has no finally
 * for a rejected recursive render. Restore that state when native admission
 * fails, while leaving successful source updates completely source-controlled.
 */
export function updateThreeNativeShadow(three, renderer, node, frame) {
  // The pinned NodeFrame and ShadowNode can both skip a before-update. Dormant
  // mapSize edits must not request a resize that the source would never perform.
  // Leave update-marker changes to NodeFrame even on these skipped paths.
  const update = () => frame.updateBeforeNode(node);
  if (renderer._isPreCompiling === true || !(node.shadow.needsUpdate || node.shadow.autoUpdate) ||
      node._cameraFrameId.get(frame.camera) === frame.frameId ||
      (frame.updateBeforeMap.get(node)?.renderId ?? 0) === frame.renderId)
    return update();
  const scene = frame.scene, camera = node.shadow.camera, layers = camera.layers.mask;
  const name = scene.name, state = three.RendererUtils.saveRendererAndSceneState(renderer, scene);
  try {
    const extents = node.shadow.getFrameExtents(), point = node instanceof three.PointShadowNode;
    const updateMap = () => renderer._targets?.withNativeShadowResize
      ? renderer._targets.withNativeShadowResize(node.shadowMap,
        node.shadow.mapSize.width * (point ? 1 : extents.x),
        point ? node.shadow.mapSize.width : node.shadow.mapSize.height * extents.y, update)
      : update();
    // Source VSM resizes both private moment targets before its two blur draws.
    // Keep earlier queued generations alive through those source-owned resizes;
    // an explicit application disposal still invalidates its accepted readers.
    const moments = [node.vsmShadowMapVertical, node.vsmShadowMapHorizontal].filter(Boolean);
    return moments.length && renderer._targets?.withNativeEffectTargets
      ? renderer._targets.withNativeEffectTargets(moments, updateMap) : updateMap();
  }
  catch (error) {
    camera.layers.mask = layers; scene.name = name;
    three.RendererUtils.restoreRendererAndSceneState(renderer, scene, state);
    throw error;
  }
}

const inherited = ['colorNode', 'depthNode', 'positionNode', 'side', 'alphaTest', 'alphaMap',
  'displacementMap', 'displacementScale', 'displacementBias', 'transparent'];

export const captureThreeShadowMaterialState = material =>
  Object.fromEntries(inherited.map(key => [key, material[key]]));

/** Callback-free admission sees the same material identity and fields that
 * Renderer.renderObject installs for the actual source shadow draw. Restore all
 * fields here: preflight is not itself an observable source draw.
 */
export function withThreeShadowMaterial(three, renderer, material, original, run) {
  const previous = captureThreeShadowMaterialState(material);
  const previousSource = renderer._currentSourceMaterial;
  try {
    renderer._currentSourceMaterial = original;
    if (original.positionNode?.isNode) material.positionNode = original.positionNode;
    for (const key of ['alphaTest', 'alphaMap', 'displacementMap', 'displacementScale', 'displacementBias'])
      material[key] = original[key];
    material.transparent = original.transparent || original.transmission > 0 ||
      (original.transmissionNode && original.transmissionNode.isNode) ||
      (original.backdropNode && original.backdropNode.isNode);
    const {colorNode, depthNode, positionNode} = renderer._getShadowNodes(original);
    material.side = original.shadowSide !== null ? original.shadowSide : renderer.shadowMap.type === three.VSMShadowMap
      ? original.side : original.side === three.FrontSide ? three.BackSide : original.side === three.BackSide ? three.FrontSide : three.DoubleSide;
    if (colorNode !== null) material.colorNode = colorNode;
    if (depthNode !== null) material.depthNode = depthNode;
    if (positionNode !== null) material.positionNode = positionNode;
    return run(material);
  } finally {
    Object.assign(material, previous); renderer._currentSourceMaterial = previousSource;
  }
}

/** Parent packets submit after their recursively captured child passes. Reject
 * a child rewrite of a texture already read by an earlier parent draw: keeping
 * callbacks in order alone cannot preserve that earlier draw's GPU contents.
 */
export function withThreeNativeCapture(renderer, run, target = null) {
  let stack = captures.get(renderer);
  if (!stack) captures.set(renderer, stack = []);
  const scope = {textures: new Set(), buffers: new Set(), draws: 0, unknown: false,
    attachments: new Set([...(target?.textures ?? []), target?.depthTexture].filter(Boolean))}; stack.push(scope);
  try { return run(texture => scope.textures.add(texture), buffer => scope.buffers.add(buffer)); }
  finally { stack.pop(); if (!stack.length) captures.delete(renderer); }
}

/** A source effect may compute its inputs before this pass's first draw. Its
 * nested passes already join the renderer FIFO before their pending parent.
 * Mark only the current pass: completed child captures do not become earlier
 * parent draws, but every still-active ancestor must remain free of reads.
 */
export function markThreeNativeDrawCaptured(renderer, known = false) {
  const scope = captures.get(renderer)?.at(-1);
  if (scope) { scope.draws++; scope.unknown ||= !known; }
}

/** Object callbacks execute synchronously around their accepted draw. Nested
 * GPU writes need a split parent pass; do not silently enqueue them before an
 * already captured parent draw. The retained renderObject lifecycle calls its
 * draw handler between callbacks, where registered node effects remain legal.
 */
export function withThreeNativeObjectHooks(renderer, run, active = true) {
  const previous = objectHooks.has(renderer);
  if (active) objectHooks.add(renderer); else objectHooks.delete(renderer);
  try { return run(); }
  finally { if (previous) objectHooks.add(renderer); else objectHooks.delete(renderer); }
}

const assertOutsideObjectHook = renderer => {
  if (objectHooks.has(renderer)) {
    const error = new Error('THREE_NODE_PROGRAM_REENTRY: Native object render callbacks cannot issue nested GPU writes before a separate pass boundary');
    error.name = 'ThreeNodeProgramError'; error.code = 'THREE_NODE_PROGRAM_REENTRY'; throw error;
  }
};

export function withThreeNativeComputePrerequisites(renderer, run, disjoint = false) {
  const previous = computePrerequisites.get(renderer);
  computePrerequisites.set(renderer, previous === true || disjoint);
  try { return run(); }
  finally { if (previous === undefined) computePrerequisites.delete(renderer); else computePrerequisites.set(renderer, previous); }
}

export function assertThreeNativeTargetWrite(renderer, target) {
  assertOutsideObjectHook(renderer);
  if (target == null) return;
  const attachments = [...(target.textures ?? [target.texture]), target.depthTexture].filter(Boolean);
  for (const texture of attachments) assertThreeNativeTextureWrite(renderer, texture);
}

export function assertThreeNativeTextureWrite(renderer, texture) {
  assertOutsideObjectHook(renderer);
  for (const {textures} of captures.get(renderer) ?? [])
    if (textures.has(texture))
      reject('A nested write cannot overwrite a texture sampled by an earlier draw in its pending parent pass');
}

export function assertThreeNativeComputeOutsideCapture(renderer, resources = null) {
  assertOutsideObjectHook(renderer);
  const stack = captures.get(renderer);
  const disjoint = computePrerequisites.get(renderer);
  // The pinned backend submits a separate compute encoder before its pending
  // render encoder. Geometry prerequisites retain that order. Restrict their
  // extension beyond the first draw to known, disjoint resources; overlapping
  // GPU/host upload histories still need their own source-equivalence proof.
  if (stack && (disjoint === undefined || stack.some(scope => scope.unknown ||
      !disjoint && (scope.draws || scope.textures.size) ||
      resources?.some(resource => scope.buffers.has(resource) || scope.textures.has(resource) || scope.attachments.has(resource))))) {
    const error = new Error('THREE_COMPUTE_REENTRY: Compute inside native draw capture requires a separate submission boundary; compute before rendering the scene');
    error.name = 'ThreeComputeError'; error.code = 'THREE_COMPUTE_REENTRY'; throw error;
  }
}
