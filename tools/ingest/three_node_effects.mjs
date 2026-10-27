/** Native target-producing node lifecycles. The source effect owns its graph,
 * materials and passes; the renderer owns execution and target generations.
 * This protocol does not select effects by class name or replace their shaders.
 */
export function createThreeNodeEffects(three, renderer, {fail, maxTargets = 64} = {}) {
  if (typeof fail !== 'function' || !Number.isSafeInteger(maxTargets) || maxTargets < 1)
    throw new TypeError('Expected native effect limits and an admission error factory');
  let producers = new WeakMap(), sharedContexts = new WeakMap(), contextSerial = 0;
  // PassTextureNode is intentionally not a public namespace export. Its
  // retained TSL factory supplies the exact constructor/prototype identity.
  let passTexturePrototype;

  function targetsFor(node) {
    const targets = new Set();
    const add = value => {
      if (!(value instanceof three.RenderTarget)) return;
      if (targets.size >= maxTargets && !targets.has(value)) fail('LIMIT', 'Native effect target capacity exceeded');
      targets.add(value);
    };
    // Source effects keep their private targets in own fields or finite arrays.
    // Inspect data properties without evaluating application accessors.
    for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(node))) {
      const value = descriptor.value;
      if (Array.isArray(value)) {
        if (value.length > 8192) fail('LIMIT', 'Native effect resource arrays exceed their inspection capacity');
        for (const item of Object.values(Object.getOwnPropertyDescriptors(value))) add(item.value);
      } else add(value);
    }
    return [...targets];
  }

  function inspectProducer(node) {
    if (!(node instanceof three.Node) || node.isPassNode === true ||
        node.updateBeforeType !== 'frame' || node.updateType !== 'none' || node.updateAfterType !== 'none' ||
        typeof node.updateBefore !== 'function' || node.updateBefore === three.Node.prototype.updateBefore ||
        ['getUpdateBeforeType', 'getUpdateType', 'getUpdateAfterType', 'updateReference']
          .some(key => node[key] !== three.Node.prototype[key]))
      fail('PASS', 'Native texture-producing effects require an ordinary source NodeFrame lifecycle');
    const targets = targetsFor(node);
    if (!targets.length) fail('PASS', 'Native texture-producing effects must own their render targets');
    let record = producers.get(node);
    if (!record) { record = {outputs: new Set(), targets}; producers.set(node, record); }
    record.targets = targets;
    for (const texture of record.outputs) {
      const target = texture.renderTarget;
      if (!targets.includes(target) || !(target.textures.includes(texture) || target.depthTexture === texture))
        fail('PASS', 'An effect output must remain an attachment of its producer-owned target');
    }
    return record;
  }

  function inspectTexture(node) {
    passTexturePrototype ??= Object.getPrototypeOf(three.TSL.passTexture(null, null));
    if (Object.getPrototypeOf(node) !== passTexturePrototype ||
        node.setup !== passTexturePrototype.setup || node.clone !== passTexturePrototype.clone ||
        !(node.value instanceof three.Texture) || node.value.isRenderTargetTexture !== true)
      fail('PASS', 'Native effect outputs require the source PassTextureNode and a target-owned texture');
    const record = inspectProducer(node.passNode), target = node.value.renderTarget;
    if (!record.targets.includes(target) || !(target.textures.includes(node.value) || target.depthTexture === node.value))
      fail('PASS', 'A pass texture must belong to a target owned by its producer');
    record.outputs.add(node.value);
  }

  function installBuilder(builder) {
    const sourceSharedContext = builder.getSharedContext;
    builder.getSharedContext = function (...args) {
      const value = sourceSharedContext.apply(this, args);
      // Keep the exact object returned by the retained builder. Contexts
      // invented by an application never acquire this construction provenance.
      sharedContexts.set(value, {id: ++contextSerial, entries: Object.entries(value)});
      return value;
    };
  }

  function contextKey(context) {
    const record = context && sharedContexts.get(context.value);
    if (!record) return null;
    if (!(context instanceof three.ContextNode) || Object.getPrototypeOf(context) !== three.ContextNode.prototype ||
        context.node !== null || context.getFlowContextData !== three.ContextNode.prototype.getFlowContextData ||
        Object.keys(context.value).length !== record.entries.length ||
        record.entries.some(([key, value]) => context.value[key] !== value))
      fail('PASS', 'A generated shared context must retain its source construction values');
    if (context.value.renderPipelineState && context.value.renderPipelineState.viewOffsetOwner !== null)
      fail('PASS', 'Native effect contexts do not yet admit view-offset histories');
    return JSON.stringify(['source-shared-context', context.uuid, record.id]);
  }

  function update(node, frame) {
    const record = inspectProducer(node);
    const run = () => frame.updateBeforeNode(node);
    if (frame.updateBeforeMap.get(node)?.frameId === frame.frameId) return run();
    if (typeof renderer._targets?.withNativeEffectTargets !== 'function')
      fail('PASS', 'Native effects require the renderer target-generation owner');
    const state = three.RendererUtils.saveRendererState(renderer);
    const fields = Object.fromEntries(['autoClearColor', 'autoClearDepth', 'autoClearStencil',
      'transparent', 'opaque', 'lighting', 'contextNode'].map(key => [key, renderer[key]]));
    try { return renderer._targets.withNativeEffectTargets(record.targets, run); }
    catch (error) {
      Object.assign(renderer, fields);
      three.RendererUtils.restoreRendererState(renderer, state);
      throw error;
    }
  }

  return {
    inspectTexture, installBuilder, contextKey,
    owns: node => producers.has(node), update,
    dispose() { producers = new WeakMap(); sharedContexts = new WeakMap(); },
  };
}
