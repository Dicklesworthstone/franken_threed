/** Device-local WGSL modules shared by source-program warmup and real draws.
 * Pipelines, bindings and geometry keep their existing owners. Shader modules
 * have no destroy operation; retain only a bounded LRU of exact source strings.
 * Failed compilation never poisons a later retry, and devices never share GPU
 * handles. Every device error scope closes before any promise is awaited.
 */
const devices = new WeakMap(), MAX_MODULES = 256, MAX_SOURCE_BYTES = 16 * 1024 * 1024;

function failure(message) {
  const error = new Error(`GPU_PROGRAM_MODULE_DEVICE: ${message}`);
  error.code = 'GPU_PROGRAM_MODULE_DEVICE';
  return error;
}

export function gpuProgramModule(device, {code, label}) {
  if (typeof code !== 'string') throw new TypeError('Program shader source must be WGSL text');
  let cache = devices.get(device);
  if (!cache) {
    cache = {modules: new Map(), bytes: 0}; devices.set(device, cache);
    device.lost?.then(() => { cache.modules.clear(); cache.bytes = 0; }).catch(() => {});
  }
  const cached = cache.modules.get(code);
  if (cached) {
    cache.modules.delete(code); cache.modules.set(code, cached);
    return cached;
  }
  let module, caught, scopes = 0;
  const validation = [];
  try {
    device.pushErrorScope('validation'); scopes++;
    device.pushErrorScope('out-of-memory'); scopes++;
    module = device.createShaderModule({code, label});
  } catch (error) { caught = error; }
  finally {
    while (scopes-- > 0) {
      try { validation.push(Promise.resolve(device.popErrorScope())); }
      catch (error) { validation.push(Promise.reject(error)); }
    }
  }
  const compilation = Promise.resolve().then(() => module?.getCompilationInfo?.()).then(info => {
    const errors = info?.messages?.filter(message => message.type === 'error') ?? [];
    if (errors.length) throw failure(errors.map(error => error.message).join('\n'));
  });
  const ready = Promise.all([...validation, compilation]).then(reports => {
    if (caught) throw caught;
    const reported = reports.find(Boolean);
    if (reported) throw failure(reported.message || 'Shader module creation failed');
    return module;
  });
  const entry = Object.freeze({module, ready}), bytes = code.length * 2;
  ready.catch(() => {
    if (cache.modules.get(code) === entry) { cache.modules.delete(code); cache.bytes -= bytes; }
  });
  if (caught) throw caught;
  if (bytes <= MAX_SOURCE_BYTES) {
    while (cache.modules.size >= MAX_MODULES || cache.bytes + bytes > MAX_SOURCE_BYTES) {
      const oldest = cache.modules.keys().next().value;
      cache.modules.delete(oldest); cache.bytes -= oldest.length * 2;
    }
    cache.modules.set(code, entry); cache.bytes += bytes;
  }
  return entry;
}
