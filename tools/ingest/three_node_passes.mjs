/** Pinned single-output scene passes and the RenderPipeline graph context.
 * Source nodes retain their construction and update frequencies; recursive
 * scene draws and sampled attachments remain owned by the native renderer.
 */
const fail = message => {
  const error = new Error(`THREE_NODE_PROGRAM_PASS: ${message}`);
  error.name = 'ThreeNodeProgramError'; error.code = 'THREE_NODE_PROGRAM_PASS'; throw error;
};

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
      ['setup', 'updateBefore', 'setSize', 'getTexture', 'getTextureNode', 'toggleTexture',
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
  if (node._mrt !== null || node.renderTarget?.textures?.length !== 1 ||
      node.renderTarget.texture !== node._textures.output || Object.keys(node._previousTextures).length ||
      Object.keys(node._previousTextureNodes).length)
    fail('Native scene passes require one color output without MRT or history swaps');
  if (node.contextNode !== null || node.transparent !== true || node.opaque !== true)
    fail('Native scene passes currently require ordinary context and both opaque and transparent draws');
  if (renderer.getOutputRenderTarget() !== null || renderer.xr?.isPresenting)
    fail('Native scene passes currently target the ordinary canvas output');
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
