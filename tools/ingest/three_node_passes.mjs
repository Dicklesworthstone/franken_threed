/** Pinned native scene passes, MRT outputs, color histories and the RenderPipeline graph context.
 * Source nodes retain their construction and update frequencies; recursive
 * scene draws and sampled attachments remain owned by the native renderer.
 */
const fail = (message, code = 'PASS') => {
  const error = new Error(`THREE_NODE_PROGRAM_${code}: ${message}`);
  error.name = 'ThreeNodeProgramError'; error.code = `THREE_NODE_PROGRAM_${code}`; throw error;
};
const nativePasses = new WeakMap();

/** Capture the source's named output mapping. Uniform values and clear colors
 * stay live; changing graph, attachment order/type or blend layout is structural.
 * A renderer MRT has no effect on the canvas, matching NodeMaterial.setup().
 */
export function inspectThreeMRT(three, renderer, material = null) {
  const target = renderer.getRenderTarget?.() ?? null;
  if (target === null) return null;
  const mrt = renderer.getMRT?.() ?? null, materialMRT = material?.mrtNode ?? null;
  const outputStruct = material?.fragmentNode?.isOutputStructNode === true;
  if (mrt === null && materialMRT === null && !outputStruct && target.textures?.length === 1) return null;
  const textures = target.textures;
  if (!Array.isArray(textures) || textures.length < 1 ||
      textures.length > (renderer._device?.limits?.maxColorAttachments ?? 8))
    fail('MRT outputs require a bounded source color attachment array', 'MRT');
  const nodes = [mrt, materialMRT].filter(Boolean), source = three.MRTNode?.prototype;
  const signatures = nodes.map(node => {
    if (!source || Object.getPrototypeOf(node) !== source ||
        ['setup', 'merge', 'has', 'get', 'getBlendMode', 'getClearColor'].some(key => node[key] !== source[key]))
      fail('Native MRT contexts require the pinned source MRTNode', 'MRT');
    if (!node.outputNodes || Object.values(node.outputNodes).some(value => value?.isNode !== true))
      fail('Native MRT output values must be source nodes', 'MRT');
    const blends = textures.map(texture => {
      const mode = node.getBlendMode(texture.name);
      if (!(mode instanceof three.BlendMode) || Object.getPrototypeOf(mode) !== three.BlendMode.prototype)
        fail('Native MRT blending requires source BlendMode values', 'MRT');
      return ['blending', 'blendSrc', 'blendDst', 'blendEquation', 'blendSrcAlpha', 'blendDstAlpha',
        'blendEquationAlpha', 'premultipliedAlpha'].map(key => mode[key]);
    });
    return [node.uuid, node.getCacheKey(), blends];
  });
  const pass = nativePasses.get(target);
  const attachmentIdentity = texture => {
    const name = pass && Object.keys(pass._textures).find(name => pass._textures[name] === texture);
    const previous = name !== undefined ? pass?._previousTextures[name] : null;
    // Source attachment bindings change on a history swap, but the shader's
    // named output layout does not. Rebuilding would repeat authored Fn setup.
    return previous ? [texture.uuid, previous.uuid].sort() : texture.uuid;
  };
  return {target, mrt, nodes, key: JSON.stringify([target.uuid,
    textures.map(texture => [attachmentIdentity(texture), texture.name, texture.format, texture.type]), signatures])};
}

/** A RenderPipeline supplies a material context for output transforms and
 * effect callbacks. Admit that exact source relationship, not arbitrary graph
 * contexts which can change the meaning of all implicit shader inputs.
 * The returned key is structural; ordinary uniform values remain live.
 */
export function inspectThreePipelineContext(three, renderer, material, object) {
  const context = material.contextNode;
  if (context == null) return null;
  const pipeline = context.value?.renderPipeline, source = three.RenderPipeline?.prototype;
  const prototype = pipeline && Object.getPrototypeOf(pipeline);
  if (!source || !(pipeline instanceof three.RenderPipeline) ||
      prototype !== source && prototype !== three.PostProcessing?.prototype ||
      pipeline.renderer !== renderer || pipeline._quadMesh !== object || object.material !== material ||
      !(context instanceof three.ContextNode) || Object.getPrototypeOf(context) !== three.ContextNode.prototype ||
      context.node !== null || context.value !== pipeline._contextData ||
      context.getFlowContextData !== three.ContextNode.prototype.getFlowContextData ||
      ['render', '_update', '_updateContext'].some(key => pipeline[key] !== source[key]))
    fail('Native material contexts require the pinned RenderPipeline and its own output quad');
  const data = context.value, keys = new Set(['renderPipeline', 'renderPipelineState',
    'onBeforePipelineCallbacks', 'onAfterPipelineCallbacks', 'toneMapping', 'outputColorSpace']);
  if (Object.keys(data).some(key => !keys.has(key)) ||
      !Array.isArray(data.onBeforePipelineCallbacks) || !Array.isArray(data.onAfterPipelineCallbacks) ||
      data.renderPipelineState?.viewOffsetOwner !== null)
    fail('Native pipeline contexts do not admit custom context fields or view-offset history');
  if (pipeline.outputColorTransform === true) {
    if (data.toneMapping !== undefined || data.outputColorSpace !== undefined)
      fail('The automatic source output transform must retain its source context');
  } else if (pipeline.outputColorTransform !== false || data.toneMapping !== pipeline._toneMapping ||
      data.outputColorSpace !== pipeline._outputColorSpace)
    fail('Explicit source output transforms must retain the pipeline tone mapping and color space');
  return JSON.stringify([context.uuid, pipeline.outputColorTransform, pipeline._toneMapping, pipeline._outputColorSpace]);
}

export function inspectThreeNativePass(three, renderer, node) {
  const source = three.PassNode?.prototype;
  if (!source || Object.getPrototypeOf(node) !== source ||
      ['setup', 'updateBefore', 'setSize', 'getTexture', 'getTextureNode', 'getPreviousTexture', 'getPreviousTextureNode', 'toggleTexture',
        'getLinearDepthNode', 'getViewZNode'].some(key => node[key] !== source[key]) ||
      ['getUpdateBeforeType', 'getUpdateType', 'getUpdateAfterType', 'updateReference']
        .some(key => node[key] !== three.Node.prototype[key]) ||
      node.updateBeforeType !== 'frame' || node.updateType !== 'none' || node.updateAfterType !== 'none')
    fail('Native scene passes require the pinned PassNode lifecycle');
  if (!(node.scene instanceof three.Scene) || !(node.camera instanceof three.Camera) ||
      node.camera.isArrayCamera || node.camera.isMultiViewCamera || node.camera.reversedDepth)
    fail('Native scene passes require an ordinary source scene and non-reversed camera');
  if (node.scope !== three.PassNode.COLOR && node.scope !== three.PassNode.DEPTH)
    fail('Native scene passes require color or depth output');
  if (!node.renderTarget?.textures?.length || node.renderTarget.texture !== node._textures.output ||
      node.renderTarget.textures.some(texture => !Object.values(node._textures).includes(texture)))
    fail('Native scene passes require their own named color outputs');
  const histories = Object.entries(node._previousTextures), textures = new Set(node.renderTarget.textures);
  for (const [name, previous] of histories) {
    const current = node._textures[name], currentNode = node._textureNodes[name], previousNode = node._previousTextureNodes[name];
    // The source swaps color attachments and both PassMultipleTextureNode
    // references together. A cloned depth image is not a previous depth
    // attachment: PassNode.toggleTexture does not swap renderTarget.depthTexture.
    if (!node.renderTarget.textures.includes(current) || !(previous instanceof three.Texture) ||
        previous.isDepthTexture || previous.isRenderTargetTexture !== true || previous.renderTarget !== node.renderTarget ||
        textures.has(previous) || currentNode?.isPassMultipleTextureNode !== true ||
        previousNode?.isPassMultipleTextureNode !== true || currentNode.passNode !== node || previousNode.passNode !== node ||
        currentNode.textureName !== name || previousNode.textureName !== name ||
        currentNode.previousTexture !== false || previousNode.previousTexture !== true ||
        currentNode.value !== current || previousNode.value !== previous)
      fail('Native color histories require the source current and previous texture-node pair');
    textures.add(previous);
  }
  if (Object.keys(node._previousTextureNodes).some(name => node._previousTextures[name] === undefined))
    fail('Native color histories require their source previous textures');
  inspectThreeMRT(three, {getRenderTarget: () => node.renderTarget, getMRT: () => node._mrt, _device: renderer._device});
  if (node.contextNode !== null)
    fail('Native scene passes currently require ordinary context');
  if (renderer.getOutputRenderTarget() !== null || renderer.xr?.isPresenting)
    fail('Native scene passes currently target the ordinary canvas output');
  if (histories.length) {
    if (typeof renderer._targets?.registerNativePass !== 'function')
      fail('Native color histories require the renderer target residency owner');
    nativePasses.set(node.renderTarget, node);
    renderer._targets.registerNativePass(node);
  }
  return true;
}

/** Source PassNode automatically follows the drawing buffer. Preserve earlier
 * accepted target generations across precisely that resize, and restore all
 * source temporary state when a recursive admission/callback throws.
 */
export function updateThreeNativePass(three, renderer, node, frame) {
  inspectThreeNativePass(three, renderer, node);
  const update = () => frame.updateBeforeNode(node);
  if ((frame.updateBeforeMap.get(node)?.frameId ?? 0) === frame.frameId) return update();
  const size = renderer.getDrawingBufferSize(new three.Vector2());
  const width = Math.floor(size.width * node._resolutionScale), height = Math.floor(size.height * node._resolutionScale);
  const scene = node.scene, camera = node.camera;
  const state = three.RendererUtils.saveRendererAndSceneState(renderer, scene), name = scene.name, mask = camera.layers.mask;
  const fields = Object.fromEntries(['autoClearColor', 'autoClearDepth', 'autoClearStencil',
    'transparent', 'opaque', 'lighting', 'contextNode'].map(key => [key, renderer[key]]));
  try { return renderer._targets.withNativePassResize(node.renderTarget, width, height, update); }
  catch (error) {
    camera.layers.mask = mask; scene.name = name;
    Object.assign(renderer, fields);
    three.RendererUtils.restoreRendererAndSceneState(renderer, scene, state);
    throw error;
  }
}
