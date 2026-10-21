/** Private per-asset I/O queue. A permit covers the complete streamed resource,
 * not just the response headers. No global scheduler or retained idle listeners.
 * Cancellation revokes publication even when injected I/O ignores AbortSignal;
 * callers must still dispose any late resource using waitForGltfResource.
 */
const reasonFor = signal => signal.reason ?? new DOMException("Aborted", "AbortError");

export function createGltfResourceQueue(concurrency, signal) {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 32)
    throw new RangeError("Expected resource concurrency in [1, 32]");
  let pending = [], cursor = 0, closed = false, reason, listening = false;
  const active = new Set();
  const onAbort = () => cancel(reasonFor(signal));
  function detach() {
    if (listening) signal.removeEventListener("abort", onAbort);
    listening = false;
  }
  function cancel(error) {
    if (closed) return;
    closed = true;
    reason = error;
    detach();
    const jobs = [...active, ...pending.slice(cursor)];
    active.clear();
    pending = [];
    cursor = 0;
    // Close the queue before abort callbacks can reenter it. Preserve the first
    // error, including non-Error reasons, for all admitted and future requests.
    for (const job of jobs) job.reject(error);
    for (const job of jobs) job.controller?.abort(error);
  }
  function finish(job, failed, value) {
    if (failed && job.failFast) cancel(value);
    active.delete(job);
    if (!closed) (failed ? job.reject : job.resolve)(value);
    pump();
  }
  function pump() {
    while (!closed && active.size < concurrency && cursor < pending.length) {
      const job = pending[cursor++];
      job.controller = new AbortController();
      active.add(job);
      // Start in a microtask: the caller can cache the promise before supplied
      // fetch functions run. Observe both late resolutions and late failures.
      Promise.resolve().then(() => {
        if (closed) throw reason;
        return job.work(job.controller.signal);
      }).then(value => finish(job, false, value), error => finish(job, true, error));
    }
    if (cursor === pending.length) {
      pending = [];
      cursor = 0;
    }
    if (!active.size && !pending.length) detach();
  }
  return Object.freeze({
    run(work, { failFast = false } = {}) {
      if (signal?.aborted) cancel(reasonFor(signal));
      if (closed) return Promise.reject(reason);
      if (typeof work !== "function") return Promise.reject(new TypeError("Expected resource work"));
      if (signal && !listening) {
        signal.addEventListener("abort", onAbort, { once: true });
        listening = true;
      }
      const promise = new Promise((resolve, reject) => {
        pending.push({ work, failFast, resolve, reject });
      });
      pump();
      return promise;
    },
    cancel,
  });
}

/** Wait without allowing uncooperative fetch/read promises to delay abort.
 * A discarded Response must have its unread body cancelled by the caller.
 * Cleanup is best-effort and never replaces the original cancellation reason.
 */
export function waitForGltfResource(pending, signal, discard) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const detach = () => signal?.removeEventListener("abort", onAbort);
    const onAbort = () => {
      if (settled) return;
      settled = true;
      detach();
      reject(reasonFor(signal));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    Promise.resolve(pending).then(value => {
      if (signal?.aborted) onAbort();
      if (settled) {
        try { Promise.resolve(discard?.(value)).catch(() => {}); } catch {}
        return;
      }
      settled = true;
      detach();
      resolve(value);
    }, error => {
      if (settled) return;
      settled = true;
      detach();
      reject(error);
    });
    // Attach handlers even for a synchronous abort during fetch invocation.
    if (signal?.aborted) onAbort();
  });
}
