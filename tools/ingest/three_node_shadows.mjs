/** Retained r186 shadow lifecycle adapters. Source ShadowNode owns construction,
 * camera fitting and update frequency; the facade captures every resulting pass
 * for its own submission. No upstream renderer or backend is constructed here.
 */
const renderFunctions = new WeakMap(), captures = new WeakMap();
const reject = message => {
  const error = new Error(`THREE_NODE_PROGRAM_SHADOW: ${message}`);
  error.name = 'ThreeNodeProgramError'; error.code = 'THREE_NODE_PROGRAM_SHADOW'; throw error;
};

/** Register the exact source caster selector before updateShadow installs it.
 * Direct ShadowNode subclasses retain the source update lifecycle, as the
 * registered SunLight addon does; their constructed graphs are still admitted
 * by the ordinary native compiler and their draws by the native scene capture.
 */
export function registerThreeNativeShadow(three, renderer, node) {
  if (!(node instanceof three.ShadowNode)) return false;
  const source = three.ShadowNode.prototype, base = three.ShadowBaseNode.prototype;
  const prototype = Object.getPrototypeOf(node), ordinary = prototype === source;
  const sun = !ordinary && Object.getPrototypeOf(prototype) === source &&
    node.light?.isSunLight === true && node.shadow?.isSunLightShadow === true;
  if ((!ordinary && !sun) || !(node.light instanceof three.Light) || node.shadow !== node.light.shadow ||
      !(node.shadow instanceof three.LightShadow) || node.shadow.isPointLightShadow ||
      ordinary && !(node.light.constructor === three.DirectionalLight || node.light.constructor === three.SpotLight))
    reject('Native shadow maps require directional, spot or registered SunLight source shadows');
  if (![three.BasicShadowMap, three.PCFShadowMap].includes(renderer.shadowMap?.type) || renderer.shadowMap.transmitted)
    reject('Native shadow maps currently require Basic or PCF filtering without transmitted shadows');
  if (node.shadow.filterNode != null || node.shadow.shadowNode != null && node.shadow.shadowNode !== node)
    reject('Custom shadow filter and shadow replacement nodes require an explicit native adapter');
  if (['setup', 'updateBefore', 'updateShadow', 'dispose', '_reset', 'getShadowFilterFn', 'setupShadowCoord', 'setupShadowFilter']
      .some(key => node[key] !== source[key]) ||
      ['getShadowMaterial', 'disposeShadowMaterial', 'getShadowRenderObjectFunction', 'setupShadowPosition']
        .some(key => node[key] !== base[key]) ||
      ['getUpdateBeforeType', 'getUpdateType', 'getUpdateAfterType', 'updateReference']
        .some(key => node[key] !== three.Node.prototype[key]) ||
      node.updateBeforeType !== 'render' || node.updateType !== 'none' || node.updateAfterType !== 'none')
    reject('Native shadow nodes must retain the pinned source lifecycle');
  if (ordinary && ['setupRenderTarget', 'setupShadow', 'renderShadow', 'vsmPass'].some(key => node[key] !== source[key]))
    reject('Native directional and spot shadows require the pinned source pass construction');
  if (sun && (node.shadow.getViewportCount() !== 2 || node.shadow.getFrameExtents().x !== 2 ||
      node.shadow.getFrameExtents().y !== 1))
    reject('Native SunLight shadows require the source two-cascade horizontal atlas');
  const renderObject = node.getShadowRenderObjectFunction(renderer, node.shadow);
  let functions = renderFunctions.get(renderer);
  if (!functions) renderFunctions.set(renderer, functions = new WeakMap());
  functions.set(renderObject, {node, renderObject, material: node.getShadowMaterial()});
  return true;
}

export function getThreeNativeShadowPass(renderer, overrideMaterial) {
  const renderObject = renderer.getRenderObjectFunction?.();
  if (renderObject == null) return null;
  const pass = renderFunctions.get(renderer)?.get(renderObject);
  if (!pass || pass.material !== overrideMaterial)
    reject('Native render-object functions require a registered source shadow pass');
  return pass;
}

/** NodeFrame keeps the source RENDER update markers, including false returns.
 * Source updateShadow restores its public state on success, but has no finally
 * for a rejected recursive render. Restore that state when native admission
 * fails, while leaving successful source updates completely source-controlled.
 */
export function updateThreeNativeShadow(three, renderer, node, frame) {
  const scene = frame.scene, camera = node.shadow.camera, layers = camera.layers.mask;
  const name = scene.name, state = three.RendererUtils.saveRendererAndSceneState(renderer, scene);
  try {
    const extents = node.shadow.getFrameExtents();
    const update = () => frame.updateBeforeNode(node);
    return renderer._targets?.withNativeShadowResize
      ? renderer._targets.withNativeShadowResize(node.shadowMap,
        node.shadow.mapSize.width * extents.x, node.shadow.mapSize.height * extents.y, update)
      : update();
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
    material.side = original.shadowSide !== null ? original.shadowSide :
      original.side === three.FrontSide ? three.BackSide : original.side === three.BackSide ? three.FrontSide : three.DoubleSide;
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
export function withThreeNativeCapture(renderer, run) {
  let stack = captures.get(renderer);
  if (!stack) captures.set(renderer, stack = []);
  const textures = new Set(); stack.push(textures);
  try { return run(texture => textures.add(texture)); }
  finally { stack.pop(); if (!stack.length) captures.delete(renderer); }
}

export function assertThreeNativeTargetWrite(renderer, target) {
  if (target == null) return;
  const attachments = [...(target.textures ?? [target.texture]), target.depthTexture].filter(Boolean);
  for (const texture of attachments) assertThreeNativeTextureWrite(renderer, texture);
}

export function assertThreeNativeTextureWrite(renderer, texture) {
  for (const textures of captures.get(renderer) ?? [])
    if (textures.has(texture))
      reject('A nested write cannot overwrite a texture sampled by an earlier draw in its pending parent pass');
}

export function assertThreeNativeComputeOutsideCapture(renderer) {
  if (captures.has(renderer)) {
    const error = new Error('THREE_COMPUTE_REENTRY: Compute inside native draw capture requires a separate submission boundary; compute before rendering the scene');
    error.name = 'ThreeComputeError'; error.code = 'THREE_COMPUTE_REENTRY'; throw error;
  }
}
