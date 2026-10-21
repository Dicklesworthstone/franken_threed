/** Bounded lookahead for the serial texture decoder/uploader. A window includes
 * the current image until the next take(), plus pending or completed lookahead.
 * Only requested image indices are admitted. Provider operations receive a
 * borrowed AbortSignal; providers that ignore it may finish, but cannot publish
 * into a cancelled window. Encoded-byte ownership/budgets remain the provider's.
 */
import { createGltfResourceQueue } from "./gltf_resource_queue.mjs";

export function createGltfImagePrefetch(indices, readImage, { windowSize = 4, signal } = {}) {
  if (!Number.isSafeInteger(windowSize) || windowSize < 1 || windowSize > 32)
    throw new RangeError("Expected image prefetch window in [1, 32]");
  const order = Array.from(indices), pending = new Map(), controller = new AbortController();
  const queue = createGltfResourceQueue(windowSize, controller.signal);
  let next = 0, taken = 0, current = -1, closed = false;
  function cancel(reason) {
    if (closed) return;
    closed = true;
    signal?.removeEventListener("abort", onAbort);
    pending.clear();
    controller.abort(reason);
  }
  const onAbort = () => cancel(signal.reason ?? new DOMException("Aborted", "AbortError"));
  signal?.addEventListener("abort", onAbort, { once: true });
  if (signal?.aborted) onAbort();
  function live() {
    if (closed) throw controller.signal.reason;
  }
  function fill() {
    while (!closed && next < order.length && pending.size < windowSize) {
      const at = next++;
      // The queue gives each provider its own cancellation signal and observes
      // late failures without retaining completed image values in an abort race.
      const result = queue.run(requestSignal => readImage(order[at], { signal: requestSignal }),
        { failFast: true });
      // Fail fast even when a later image fails while the current decoder or
      // upload is still waiting. Observe every result, including abandoned work.
      result.catch(cancel);
      pending.set(at, result);
    }
  }
  return Object.freeze({
    signal: controller.signal,
    async take(index) {
      live();
      if (taken >= order.length || index !== order[taken])
        throw new Error("Images must be consumed in their prepared order");
      pending.delete(current);
      fill();
      current = taken++;
      return pending.get(current);
    },
    cancel,
    dispose() { cancel(new DOMException("Image prefetch closed", "AbortError")); },
  });
}
