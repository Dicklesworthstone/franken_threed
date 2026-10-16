/** Native render-target inputs for the r186 TSL compute route.
 * Capture target generation and sampler state now; observe GPU contents at the
 * renderer FIFO slot. Never snapshot GPU pixels, flip UVs, allocate source IDs,
 * or own/destroy the renderer's target storage. All native views are borrowed
 * through submission, including every source in a multi-target dispatch.
 */
export class ThreeComputeTargetError extends Error {
  constructor(code, message) {
    super(`THREE_COMPUTE_TARGET_${code}: ${message}`);
    this.name = 'ThreeComputeTargetError'; this.code = `THREE_COMPUTE_TARGET_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeComputeTargetError(code, message); };
const positive = n => Number.isSafeInteger(n) && n > 0;

export function createThreeComputeTargets({targets, device, signal = null,
  maxBindings = 4096, maxSamplers = 256, maxPending = 1024} = {}) {
  if (!targets || !['captureNativeTexture', 'validateNativeTextures', 'withNativeTexture', 'assertCurrent'].every(key => typeof targets[key] === 'function') ||
      !device?.queue || ![maxBindings, maxSamplers, maxPending].every(positive))
    fail('OPTIONS', 'Expected the renderer target owner, GPUDevice and positive resource limits');
  const records = new Map(), leases = new WeakSet(), samplers = new Map(), views = new WeakMap(), pending = new Set();
  let terminal = null, disposed = false, active = null, rejectStop;
  const stopped = new Promise((_, reject) => { rejectStop = reject; }); stopped.catch(() => {});
  function stop(error) {
    if (terminal) return;
    // Preserve driver/producer errors from other realms, including DOMException.
    terminal = error && (typeof error === 'object' || typeof error === 'function')
      ? error : new ThreeComputeTargetError('ABORTED', String(error ?? 'Renderer lifetime ended'));
    records.clear(); samplers.clear(); rejectStop(terminal);
    signal?.removeEventListener('abort', onAbort);
  }
  const onAbort = () => stop(signal.reason);
  if (signal?.aborted) onAbort(); else signal?.addEventListener('abort', onAbort, {once: true});
  device.lost?.then(info => stop(new ThreeComputeTargetError('LOST', info?.message ?? 'WebGPU device lost')), stop);
  const live = () => {
    if (disposed) fail('DISPOSED', 'Compute target inputs are disposed');
    if (terminal) throw terminal;
    if (signal?.aborted) throw signal.reason ?? new ThreeComputeTargetError('ABORTED', 'Renderer lifetime ended');
  };
  function watch(promise) {
    const task = Promise.resolve(promise).catch(error => { stop(error); throw error; });
    pending.add(task);
    task.finally(() => pending.delete(task)).catch(() => {});
  }
  function describe(binding) {
    live();
    if (!binding?.isSampler && !binding?.isSampledTexture) fail('BINDING', 'Expected a texture or sampler binding');
    if (binding.store) fail('ACCESS', 'Render-target inputs are read-only; use StorageTexture for compute writes');
    const source = targets.captureNativeTexture(binding.texture);
    if (source.device !== device) fail('DEVICE', 'Compute and render targets must use the same GPUDevice');
    const sampler = binding.isSampler && !binding.isSampledTexture;
    const layout = sampler ? {sampler: {type: source.sampleType === 'unfilterable-float' ? 'non-filtering' : 'filtering'}}
      : {texture: {sampleType: source.sampleType, viewDimension: '2d', multisampled: false}};
    return {source, sampler, layout};
  }
  function validate(snapshot) {
    live();
    const record = records.get(snapshot);
    if (!record) fail('STALE', 'Compute target binding was released or belongs to another owner');
    targets.assertCurrent(record.source.entry);
    if (record.source.device !== device || record.source.entry.session && record.source.entry.session.device !== device)
      fail('DEVICE', 'The captured target belongs to another GPUDevice');
    return record;
  }
  function native(run) {
    try { return run(); } catch (error) { stop(error); throw error; }
  }
  function checked(run) {
    native(() => { device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation'); });
    try { return run(); }
    finally {
      const scopes = native(() => [device.popErrorScope(), device.popErrorScope()]);
      watch(Promise.all(scopes).then(errors => {
        const error = errors.find(Boolean);
        if (error) throw new ThreeComputeTargetError('GPU', error.message ?? 'Native compute target binding failed');
      }));
    }
  }
  return {
    owns(snapshot) { return leases.has(snapshot); },
    layoutOf(binding) { return describe(binding).layout; },
    sampleData(texture) { return describe({texture, isSampledTexture: true}).source.sampleData; },
    captureBinding(binding) {
      if (active) fail('REENTRY', 'Cannot capture target inputs during a borrowed dispatch');
      const record = describe(binding);
      if (records.size >= maxBindings) fail('BUDGET', 'Queued render-target bindings exceed maxBindings');
      const snapshot = Object.freeze({}); records.set(snapshot, record); leases.add(snapshot); return snapshot;
    },
    layout(snapshot) { return validate(snapshot).layout; },
    validateDispatch(snapshots) { for (const snapshot of snapshots) validate(snapshot); },
    withDispatch(snapshots, run) {
      live();
      if (typeof run !== 'function') fail('CALLBACK', 'Expected a synchronous submitted-use callback');
      if (active) fail('REENTRY', 'Cannot reenter a borrowed compute dispatch');
      if (!snapshots.length) return run();
      const sources = new Map();
      for (const snapshot of snapshots) { const {source} = validate(snapshot); sources.set(source.entry, source); }
      if (pending.size + sources.size + 1 > maxPending) fail('BUDGET', 'Too many target submissions are awaiting completion');
      // Hold each native owner once, even when the kernel uses several texture
      // and sampler bindings for the same source. No borrow crosses an await.
      const list = [...sources.values()], borrowed = new Map();
      targets.validateNativeTextures(list);
      active = {tokens: new Set(snapshots), textures: new Map()};
      const visit = i => {
        if (i === list.length) {
          for (const snapshot of snapshots) validate(snapshot);
          const result = run();
          if (result?.then) {
            Promise.resolve(result).catch(() => {});
            fail('ASYNC_USE', 'Submit compute work before returning from the native target borrow');
          }
          return;
        }
        const source = list[i];
        targets.withNativeTexture(source, texture => {
          borrowed.set(source.entry, source.entry.storage);
          active.textures.set(source.entry, texture);
          visit(i + 1);
        });
      };
      try { checked(() => visit(0)); }
      finally {
        active = null;
        // Producer validation is part of compute completion. Start these fences
        // only after every native borrow unwinds; do not call whenIdle while busy.
        for (const [entry, storage] of borrowed) {
          try { watch(Promise.all([entry.validation, storage.whenIdle()])); }
          catch (error) { watch(Promise.reject(error)); }
        }
      }
    },
    resolve(snapshot) {
      const {source, sampler} = validate(snapshot);
      if (!active?.tokens.has(snapshot)) fail('BORROW', 'Resolve target inputs only inside their submitted dispatch');
      if (sampler) {
        const key = JSON.stringify(source.sampler);
        let value = samplers.get(key);
        if (!value) {
          value = native(() => device.createSampler(source.sampler));
          if (samplers.size >= maxSamplers) samplers.delete(samplers.keys().next().value);
        } else samplers.delete(key);
        samplers.set(key, value); return value;
      }
      const texture = active.textures.get(source.entry);
      if (!texture) fail('BORROW', 'The native source has not been borrowed');
      let view = views.get(texture);
      if (!view) {
        view = native(() => texture.createView({dimension: '2d', baseMipLevel: 0, mipLevelCount: source.entry.shape.mipLevelCount}));
        views.set(texture, view);
      }
      return view;
    },
    release(snapshot) { records.delete(snapshot); },
    async whenIdle() { live(); await Promise.race([Promise.all([...pending]), stopped]); live(); },
    get diagnostics() { return {bindings: records.size, samplers: samplers.size, pending: pending.size}; },
    dispose() { if (!disposed) { disposed = true; stop(new ThreeComputeTargetError('DISPOSED', 'Compute target inputs are disposed')); } },
  };
}
