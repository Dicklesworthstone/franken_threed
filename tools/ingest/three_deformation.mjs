/** Source-owned skin/morph capture feeding the existing fused GPU deformer.
 * The returned deformer is the actual registered core GPU owner, not a facsimile
 * of its public vertex-buffer fields. Submit consumers before the next update.
 */
import {createGpuAnimationDeformer, updateGpuAnimationDeformers} from './animation_webgpu.mjs';
import {createThreeDeformationBinding, ThreeDeformationError} from './three_deformation_binding.mjs';
export {hasThreeDeformation, inspectThreeDeformation, createThreeDeformationBinding, ThreeDeformationError} from './three_deformation_binding.mjs';
const states = new WeakMap();
const fail = (code, message) => { throw new ThreeDeformationError(code, message); };

export async function createGpuThreeDeformation(device, source, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('OPTIONS', 'Expected source deformation options');
  const {signal, maxBytes = 128 * 1024 * 1024, label = 'f3d-three-deformation', ...bindingOptions} = options;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || typeof label !== 'string') fail('LIMIT', 'Invalid GPU deformation budget or label');
  if (signal !== undefined && (!signal || typeof signal.aborted !== 'boolean' ||
      typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function')) fail('OPTIONS', 'Expected AbortSignal');
  if (signal?.aborted) fail('ABORTED', 'Source deformation initialization was aborted');
  const binding = createThreeDeformationBinding(source, bindingOptions);
  let gpu = null, disposed = false, terminal = null, busy = false, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const closed = () => disposed || terminal !== null;
  function release() {
    signal?.removeEventListener('abort', onAbort);
    const owned = gpu; gpu = null;
    try { owned?.dispose(); } finally { binding.dispose(); }
  }
  function stop(error) {
    if (closed()) return;
    terminal = error; rejectStopped(error);
    if (!busy) release();
  }
  function live() {
    if (disposed) fail('DISPOSED', 'Source GPU deformation is disposed');
    if (terminal) throw terminal;
    if (gpu?.disposed || gpu?.failed) {
      stop(new ThreeDeformationError('GPU', 'The core GPU deformer is unavailable'));
      throw terminal;
    }
  }
  const onAbort = () => stop(new ThreeDeformationError('ABORTED', 'Source GPU deformation was aborted'));
  signal?.addEventListener('abort', onAbort, {once: true});
  try {
    if (signal?.aborted) onAbort(); live();
    const construction = createGpuAnimationDeformer(device, binding.pose, binding.geometry, {
      maxBytes, label, ...(bindingOptions.maxComponents === undefined ? {} : {maxComponents: bindingOptions.maxComponents}),
    }).then(value => {
      if (closed()) { value.dispose(); throw terminal ?? new ThreeDeformationError('DISPOSED', 'Source owner is closed'); }
      gpu = value; return value;
    });
    await Promise.race([construction, stopped]); live();
    binding.check();
  } catch (error) { stop(error); throw error; }
  const result = Object.freeze({source, surface: binding.surface, signature: binding.signature,
    vertexCount: binding.vertexCount, indexCount: binding.indexCount,
    get deformer() { live(); return gpu; },
    get bufferBytes() { return gpu?.bufferBytes ?? 0; },
    get disposed() { return disposed; }, get failed() { return terminal !== null || !!gpu?.failed; },
    matches() { live(); return binding.matches(); },
    check() { live(); binding.check(); },
    update() { updateGpuThreeDeformations([result]); return result; },
    async whenIdle() {
      live();
      try { await Promise.race([gpu.whenIdle(), stopped]); live(); return result; }
      catch (error) { if (gpu?.failed) stop(error); throw error; }
    },
    dispose() {
      if (busy) fail('REENTRANT', 'Cannot dispose during source GPU submission');
      if (!disposed) {
        disposed = true; rejectStopped(new ThreeDeformationError('DISPOSED', 'Source GPU deformation is disposed')); release();
      }
    },
  });
  states.set(result, {device, live, capture: () => binding.capture(), native: () => gpu,
    enter() { live(); if (busy) fail('REENTRANT', 'Source GPU update cannot be reentered'); busy = true; },
    leave() { busy = false; if (closed()) release(); },
    failed(error) { if (gpu?.failed) stop(error); },
  });
  return result;
}

/** Capture every source first, then validate/upload/dispatch the whole core batch.
 * No CPU vertex loop occurs here. Different meshes sharing geometry retain
 * distinct palettes/morph weights; material groups share their mesh's owner.
 */
export function updateGpuThreeDeformations(owners) {
  if (!Array.isArray(owners) || owners.length > 65536) fail('OPTIONS', 'Expected a bounded source deformation batch');
  const seen = new Set(), batch = [];
  for (const owner of owners) {
    const state = states.get(owner);
    if (!state || seen.has(owner)) fail('OWNER', 'Expected unique live source deformation owners');
    if (batch.length && state.device !== batch[0].device) fail('DEVICE', 'A source deformation batch must use one device');
    seen.add(owner); batch.push(state);
  }
  const entered = [];
  try {
    for (const state of batch) { state.enter(); entered.push(state); }
    for (const state of batch) state.capture();
    for (const state of batch) state.live();
    if (batch.length) updateGpuAnimationDeformers(batch.map(state => state.native()));
  } catch (error) {
    for (const state of batch) state.failed(error);
    throw error;
  } finally { for (const state of entered) state.leave(); }
}
