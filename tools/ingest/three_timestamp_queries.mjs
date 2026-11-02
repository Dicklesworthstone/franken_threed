/** GPU timestamps for the retained r186 Inspector. One pair measures one
 * logical call across its actual native passes: the first pass writes start,
 * each subsequent pass replaces end. No clock estimate or timing-only pass.
 * Pending queries and published frames are bounded independently. */
export class ThreeTimestampError extends Error {
  constructor(code, message) {
    super(`THREE_TIMESTAMP_${code}: ${message}`);
    this.name = 'ThreeTimestampError'; this.code = `THREE_TIMESTAMP_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeTimestampError(code, message); };
const types = ['render', 'compute'];

export function createThreeTimestampQueries({getDevice, enabled, signal = null, maxQueries = 2048,
  maxFrames = 512, warn = () => {}} = {}) {
  if (typeof getDevice !== 'function' || typeof enabled !== 'function' ||
      !Number.isSafeInteger(maxQueries) || maxQueries < 2 || maxQueries % 2 ||
      !Number.isSafeInteger(maxFrames) || maxFrames < 1) fail('OPTIONS', 'Expected device/tracking accessors and positive query limits');
  const records = new Set(), contexts = new WeakMap();
  const states = new Map(types.map(type => [type, {pending: new Set(), skipped: new Set(), frames: [], values: new Map(), last: 0, seen: false, resolving: null}]));
  let terminal = null, watchedDevice = null, rejectStopped;
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  const live = () => { if (terminal) throw terminal; };
  const stateOf = type => { if (!states.has(type)) fail('TYPE', 'Timestamp type must be render or compute'); return states.get(type); };
  function destroy(record) {
    if (!records.delete(record)) return;
    record.query?.destroy(); record.resolve?.destroy(); record.read?.destroy();
  }
  function stop(error) {
    if (terminal) return;
    terminal = error; rejectStopped(error);
    for (const record of records) destroy(record);
    for (const state of states.values()) { state.pending.clear(); state.skipped.clear(); state.frames = []; state.values.clear(); }
    signal?.removeEventListener('abort', abort);
  }
  const abort = () => stop(signal.reason ?? new ThreeTimestampError('ABORTED', 'Renderer lifetime ended'));
  if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, {once: true});
  function device() {
    live(); const value = getDevice();
    if (!value?.features?.has('timestamp-query')) return null;
    if (watchedDevice && watchedDevice !== value) fail('DEVICE', 'Timestamp queries cannot migrate between devices');
    if (!watchedDevice) {
      watchedDevice = value;
      value.lost?.then(info => stop(new ThreeTimestampError('LOST', info?.message ?? 'WebGPU device lost')), stop).catch(() => {});
    }
    return value;
  }
  function checked(d, run) {
    d.pushErrorScope('out-of-memory'); d.pushErrorScope('validation');
    let value, error;
    try { value = run(); } catch (cause) { error = cause; }
    const validation = Promise.all([d.popErrorScope(), d.popErrorScope()]).then(errors => {
      if (error) throw error;
      const invalid = errors.find(Boolean);
      if (invalid) throw new ThreeTimestampError('GPU', invalid.message ?? 'GPU timestamp validation failed');
    });
    validation.catch(stop);
    if (error) throw error;
    return {value, validation};
  }
  function skip(state, frame) {
    state.skipped.add(frame);
    if (state.skipped.size > maxFrames) state.skipped.delete(state.skipped.values().next().value);
  }
  function begin(type, uid, context = null) {
    live(); const state = stateOf(type), match = typeof uid === 'string' && uid.match(/:f(\d+)$/);
    if (!match || !Number.isSafeInteger(Number(match[1]))) fail('UID', 'Timestamp UID must include its captured frame');
    if (context && typeof context === 'object') contexts.set(context, uid);
    if (!enabled()) return null;
    const frame = Number(match[1]); let record = null, cancelled = false;
    return Object.freeze({
      writes() {
        live(); if (cancelled) return undefined;
        const d = device(); if (!d) return undefined;
        let first = false;
        if (!record) {
          if (records.size * 2 >= maxQueries) {
            // Instrumentation must never block a synchronous draw or reuse a
            // query still owned by an outstanding mapping operation. A later
            // resolution marks this call unavailable to the Inspector.
            skip(state, frame); cancelled = true;
            warn('WebGPURenderer: Timestamp query capacity exceeded.'); return undefined;
          }
          record = {uid, frame, type, query: null, resolve: null, read: null, validation: null};
          records.add(record); state.pending.add(record); state.seen = true;
          try {
            record.validation = checked(d, () => {
              record.query = d.createQuerySet({label: `f3d-${type}-timestamp`, type: 'timestamp', count: 2});
            }).validation;
          } catch (error) { stop(error); throw error; }
          first = true;
        } else if (!state.pending.has(record)) fail('STALE', 'Cannot append passes after resolving their timestamp');
        return {querySet: record.query, ...(first ? {beginningOfPassWriteIndex: 0} : {}), endOfPassWriteIndex: 1};
      },
      validate(validation) {
        if (record && !cancelled) {
          record.validation = Promise.all([record.validation, validation]);
          record.validation.catch(stop);
        }
      },
      cancel() {
        cancelled = true;
        if (record) { state.pending.delete(record); destroy(record); skip(state, frame); }
      },
    });
  }
  function resolve(type = 'render') {
    live(); const state = stateOf(type);
    if (!enabled() || !device()) return Promise.resolve(undefined);
    if (state.resolving) return state.resolving;
    const batch = [...state.pending], skipped = [...state.skipped];
    if (!batch.length && !skipped.length) return Promise.resolve(state.seen ? state.last : undefined);
    state.pending.clear(); state.skipped.clear();
    const run = (async () => {
      try {
        const d = device();
        const submitted = checked(d, () => {
          if (!batch.length) return;
          const encoder = d.createCommandEncoder({label: `f3d-${type}-timestamp-readback`});
          for (const record of batch) {
            record.resolve = d.createBuffer({label: 'f3d-timestamp-resolve', size: 16, usage: 512 | 4});
            record.read = d.createBuffer({label: 'f3d-timestamp-readback', size: 16, usage: 8 | 1});
            encoder.resolveQuerySet(record.query, 0, 2, record.resolve, 0);
            encoder.copyBufferToBuffer(record.resolve, 0, record.read, 0, 16);
          }
          d.queue.submit([encoder.finish()]);
        });
        await Promise.race([Promise.all([submitted.validation, ...batch.map(record => record.validation),
          ...batch.map(record => record.read.mapAsync(1, 0, 16))]), stopped]);
        live();
        const values = new Map(), totals = new Map(), frames = new Set(skipped);
        for (const record of batch) {
          const times = new BigUint64Array(record.read.getMappedRange(0, 16));
          const duration = Number(times[1] - times[0]) / 1e6;
          if (!Number.isFinite(duration) || duration < 0) fail('VALUE', 'GPU returned invalid timestamp ordering');
          record.read.unmap(); frames.add(record.frame);
          values.set(record.uid, (values.get(record.uid) ?? 0) + duration);
          totals.set(record.frame, (totals.get(record.frame) ?? 0) + duration);
        }
        state.frames = [...frames].sort((a, b) => a - b).slice(-maxFrames);
        const kept = new Set(state.frames);
        state.values = new Map([...values].filter(([uid]) => kept.has(Number(uid.match(/:f(\d+)$/)[1]))));
        state.last = totals.get(state.frames.at(-1)) ?? 0;
        return state.last;
      } catch (error) { stop(error); throw error; }
      finally { for (const record of batch) destroy(record); }
    })();
    state.resolving = run;
    run.finally(() => { if (state.resolving === run) state.resolving = null; }).catch(() => {});
    return run;
  }
  return Object.freeze({begin, resolve,
    get supported() { return !!getDevice()?.features?.has('timestamp-query'); },
    getTimestampUID: context => contexts.get(context),
    getTimestampFrames: type => stateOf(type).frames,
    hasTimestampQuery: uid => [...states.values()].some(state => state.values.has(uid)),
    getTimestamp(uid) {
      for (const state of states.values()) if (state.values.has(uid)) return state.values.get(uid);
      warn(`TimestampQueryPool: No timestamp available for uid ${uid}.`); return 0;
    },
    get pendingCount() { return records.size; },
    dispose() { stop(new ThreeTimestampError('DISPOSED', 'Timestamp owner is disposed')); },
  });
}
