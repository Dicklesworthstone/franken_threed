/** One loss observer per native promise, with removable resource subscriptions.
 * GPUDevice.lost cannot be unsubscribed. Direct per-resource continuations keep
 * disposed source graphs alive until the device is lost, even after GPU cleanup.
 */
const observers = new WeakMap();

function deliver(subscription, rejected, value) {
  if (!subscription) return;
  const callback = rejected ? subscription.reject : subscription.fulfill;
  subscription.fulfill = subscription.reject = null;
  callback?.(value);
}

function observer(lost) {
  let record = observers.get(lost);
  if (record) return record;
  record = {subscriptions: new Set(), settled: false, rejected: false, value: undefined};
  observers.set(lost, record);
  const settle = (rejected, value) => {
    record.settled = true; record.rejected = rejected; record.value = value;
    const pending = [...record.subscriptions]; record.subscriptions.clear();
    for (const subscription of pending) {
      try { deliver(subscription, rejected, value); }
      catch (error) {
        // Independent promise reactions do not prevent other owners cleaning
        // up when one callback throws. Preserve that rejected-reaction behavior.
        Promise.reject(error);
      }
    }
  };
  lost.then(value => settle(false, value), error => settle(true, error));
  return record;
}

/** Return an idempotent unsubscribe; neither subscription nor disposal owns the
 * device. Rejected loss promises retain their original reason and error route.
 */
export function subscribeGpuDeviceLoss(lost, fulfill, reject) {
  let record = observer(lost), subscription = {fulfill, reject};
  if (record.settled) {
    const {rejected, value} = record;
    // Subscribing to an already lost device still notifies asynchronously.
    Promise.resolve().then(() => deliver(subscription, rejected, value));
  } else record.subscriptions.add(subscription);
  return () => {
    if (!subscription) return;
    record.subscriptions.delete(subscription);
    subscription.fulfill = subscription.reject = null;
    subscription = record = null;
  };
}
