/** One synchronous submission slot for native copies that retain target
 * generations. GPU completion belongs to the existing owners, not this FIFO.
 */
import {ThreeStorageCopyError} from './three_texture_copy_plan.mjs';

/** Capture target generations transactionally, including admission failures. */
export function captureTextureCopy(renderer, capture) {
  const entries = new Set(), releases = [];
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    // Owner releases do not issue application callbacks. Still unwind every
    // lease when one owner is already terminal rather than leaking the rest.
    let failure;
    for (const done of releases.splice(0)) {
      try { done(); } catch (error) { failure ??= error; }
    }
    if (failure) throw failure;
  };
  try {
    const request = capture(entry => {
      if (entries.has(entry)) return;
      releases.push(renderer._targets.retainGeneration(entry)); entries.add(entry);
    });
    return Object.freeze({...request, release,
      issue() {
        if (released) throw new ThreeStorageCopyError('STALE', 'Texture copy was already submitted or released');
        try { return request.issue(); } finally { release(); }
      },
      signals: Object.freeze([...entries].map(entry => entry.lifetime?.signal).filter(Boolean))});
  } catch (error) { release(); throw error; }
}

/** Keep source void behavior and compute-turn ordering. Aborted and rejected
 * requests release immediately, even when an earlier preparation never settles.
 * No private queue, GPU fence, texture allocation or Promise pool is introduced.
 */
export function submitTextureCopy(renderer, request, onError) {
  const signals = [...new Set([renderer._targetLifetime?.signal, ...request.signals ?? []].filter(Boolean))];
  let finished = false, reported = false;
  const release = () => {
    if (finished) return;
    finished = true;
    for (const signal of signals) signal.removeEventListener('abort', abort);
    request.release?.();
  };
  const rejected = error => {
    release();
    if (!reported) { reported = true; onError(error); }
  };
  const abort = () => rejected(signals.find(signal => signal.aborted)?.reason ??
    new ThreeStorageCopyError('DISPOSED', 'Texture copy owner was disposed'));
  const issue = () => {
    if (finished) return;
    try { return request.issue(); } finally { release(); }
  };
  try {
    const stopped = signals.find(signal => signal.aborted);
    if (stopped) throw stopped.reason ?? new ThreeStorageCopyError('DISPOSED', 'Texture copy owner was disposed');
    if (request.empty) { release(); return; }
    for (const signal of signals) signal.addEventListener('abort', abort, {once: true});
    if (renderer._computeTurn) renderer._computeTurn.submit(issue)?.catch?.(rejected);
    else if (!renderer._drain && !renderer._preparing && !renderer._pending?.length) issue();
    else renderer._enqueue({transfer: true, issue, resolve() {}, reject: rejected});
  } catch (error) { release(); throw error; }
}
