/** Device-aligned, paged uniforms for compiled program draws.
 * Registration prepares capacity from reflected block sizes and maxDraws, capped
 * by maxBytes (GPU storage and a separate equally bounded CPU shadow). It never
 * replaces published pages or bind groups. Growth cannot invalidate earlier
 * commands, including commands staged before an asynchronous registration.
 *
 * All native allocation/validation happens in registration, not stage()/write().
 * Submit the preceding frame before begin(); queue writes then precede this
 * frame's submission. Stable page/group/offset choices remain bundle-cacheable.
 * A tight byte budget may admit fewer than maxDraws; stage fails, never truncates.
 */
export function createProgramUniformArena(device, {maxDraws, label, fail, scoped, lost = device.lost, maxBytes = 64 * 1024 * 1024}) {
  const limit = message => fail('ANIMATION_RENDER_LIMIT', message);
  const alignment = device.limits.minUniformBufferOffsetAlignment ?? 256;
  const maximum = Math.floor(Math.min(device.limits.maxBufferSize ?? 268435456, maxBytes, 0xffffffff) / 4) * 4;
  const bindingLimit = Math.min(device.limits.maxUniformBufferBindingSize ?? 65536, maximum);
  if (!Number.isSafeInteger(maxDraws) || maxDraws < 1 || maxDraws > 65536 ||
      !Number.isSafeInteger(maxBytes) || maxBytes < 4 ||
      !Number.isSafeInteger(alignment) || alignment < 4 || !Number.isInteger(Math.log2(alignment)) ||
      alignment > 0x100000000 || !Number.isSafeInteger(maximum) || maximum < 4 ||
      !Number.isSafeInteger(bindingLimit) || bindingLimit < 4)
    limit('Invalid program uniform arena capacity or alignment');
  const pages = [], resources = new Set(), bindings = new Map();
  let allocatedBytes = 0, draws = 0, stoppedError = null, serial = Promise.resolve(), rejectStopped;
  let registering = 0, validation = Promise.resolve();
  const stopped = new Promise((_, reject) => { rejectStopped = reject; }); stopped.catch(() => {});
  function error(code, message) { try { fail(code, message); } catch (e) { return e; } }
  function live() { if (stoppedError) throw stoppedError; }
  function destroy(page) {
    if (!resources.delete(page)) return;
    allocatedBytes -= page.size;
    page.buffer.destroy();
  }
  function stop(reason) {
    if (stoppedError) return;
    stoppedError = reason;
    rejectStopped(reason);
    pages.length = 0; bindings.clear();
    for (const page of resources) destroy(page);
  }
  lost.then(info => stop(error('ANIMATION_RENDER_LOST', info?.message || 'WebGPU device lost')), stop).catch(() => {});
  function checkPacketSize(size) {
    live();
    if (!Number.isSafeInteger(size) || size < 4 || size % 4 || size > maximum)
      limit('Program uniform packet exceeds the buffer or byte limit');
    return size;
  }
  function checkSize(size) {
    checkPacketSize(size);
    if (size > bindingLimit) limit('Program uniform block exceeds the binding or byte limit');
    return size;
  }
  const keyOf = (size, packet) => packet ? `packet:${size}` : size;
  const strideOf = size => Math.ceil(size / alignment) * alignment;
  const slots = (bytes, size) => bytes < size ? 0 : 1 + Math.floor((bytes - size) / strideOf(size));
  function allocate(size, packet) {
    live();
    const key = keyOf(size, packet);
    // Serialize registrations, not draws. Failed growth is private and cannot
    // publish invalid groups or strand memory needed by a later registration.
    let missing = maxDraws - pages.reduce((n, page) => n + slots(page.size, size), 0);
    let remaining = maxBytes - allocatedBytes;
    const planned = [];
    while (missing > 0 && remaining >= size) {
      const bytes = Math.floor(Math.min(maximum, remaining, Math.max(65536, (missing - 1) * strideOf(size) + size)) / 4) * 4;
      planned.push(bytes); remaining -= bytes; missing -= slots(bytes, size);
    }
    if (!pages.some(page => page.size >= size) && !planned.length)
      limit('Program uniform budget cannot fit this block alongside existing pages');
    const added = [], groups = [], oldBindings = [...bindings.values()];
    let issued;
    try {
      issued = scoped(device, () => {
        // A source group can pack several individually bounded UBOs into one
        // reservation. Its caller owns those binding ranges and their layout;
        // never manufacture an oversized standalone binding for the packet.
        const layout = packet ? null : device.createBindGroupLayout({label: `${label}/program-uniforms`, entries: [{binding: 0, visibility: 3,
          buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: size}}]});
        const binding = Object.freeze({layout, size, packet, groups: new WeakMap()});
        for (const bytes of planned) {
          const staged = new Uint8Array(bytes);
          const buffer = device.createBuffer({label: `${label}/program-uniforms`, size: bytes, usage: 64 | 8});
          const page = {buffer, size: bytes, staged, used: 0};
          added.push(page); resources.add(page); allocatedBytes += bytes;
        }
        for (const b of [...oldBindings, binding]) for (const page of b === binding ? [...pages, ...added] : added) {
          if (b.packet || page.size < b.size) continue;
          const group = device.createBindGroup({label: `${label}/program-uniforms`, layout: b.layout,
            entries: [{binding: 0, resource: {buffer: page.buffer, size: b.size}}]});
          groups.push({binding: b, page, group});
        }
        return binding;
      });
      if (issued.error) throw issued.error;
      return {issued, added, groups, key};
    } catch (e) {
      issued?.errors.catch(() => {});
      for (const page of added) destroy(page);
      throw e;
    }
  }
  function publish({issued, added, groups, key}) {
    live();
    for (const {binding, page, group} of groups) binding.groups.set(page, group);
    pages.push(...added); bindings.set(key, issued.value); return issued.value;
  }
  async function prepare(size, packet) {
    live();
    const key = keyOf(size, packet);
    if (bindings.has(key)) return bindings.get(key);
    const allocation = allocate(size, packet);
    try { await Promise.race([allocation.issued.errors, stopped]); return publish(allocation); }
    catch (error) { for (const page of allocation.added) destroy(page); throw error; }
  }
  function registerImmediate(size) {
    checkSize(size);
    if (bindings.has(size)) return bindings.get(size);
    if (registering) fail('ANIMATION_RENDER_REENTRANT', 'Immediate program registration cannot overlap asynchronous preparation');
    const allocation = allocate(size, false);
    const binding = publish(allocation);
    // A source draw is synchronous. WebGPU scopes still close before returning
    // to callbacks, and delayed failures terminalize this arena and its owner.
    validation = Promise.all([validation, allocation.issued.errors]).then(() => live());
    validation.catch(stop);
    return binding;
  }
  function register(size, packet) {
    // Check before joining the queue; invalid input never reserves resources.
    if (packet) checkPacketSize(size); else checkSize(size);
    registering++;
    const ready = serial.then(() => prepare(size, packet)).finally(() => { registering--; });
    serial = ready.catch(() => {});
    return ready;
  }
  return Object.freeze({
    checkSize, checkPacketSize, live, stopped,
    registerImmediate,
    register(size) { return register(size, false); },
    registerPacket(size) { return register(size, true); },
    stage(binding, bytes) {
      live();
      if (!binding || bindings.get(keyOf(binding.size, binding.packet)) !== binding)
        fail('ANIMATION_RENDER_VALUE', 'Program uniform binding is not registered in this arena');
      if (!(bytes instanceof Uint8Array) || bytes.byteLength !== binding.size ||
          !(bytes.buffer instanceof ArrayBuffer) || bytes.buffer.resizable)
        fail('ANIMATION_RENDER_VALUE', 'Program uniforms need exact, fixed unshared byte storage');
      if (draws >= maxDraws) limit('Program uniform draw capacity exhausted for this frame');
      for (const page of pages) {
        const offset = Math.ceil(page.used / alignment) * alignment, group = binding.packet ? null : binding.groups.get(page);
        if ((!binding.packet && !group) || offset + binding.size > page.size) continue;
        page.staged.set(bytes, offset);
        page.used = offset + binding.size; draws++;
        // Borrowed buffer metadata lets native source layouts combine this
        // arena slice with texture/sampler bindings without owning another arena.
        return {group, offset, resource: {buffer: page.buffer, size: binding.size}};
      }
      limit('Program uniform byte budget exhausted for this frame');
    },
    begin() { live(); draws = 0; for (const page of pages) page.used = 0; },
    write() {
      live();
      for (const page of pages) if (page.used) device.queue.writeBuffer(page.buffer, 0, page.staged, 0, page.used);
    },
    dispose() { stop(error('ANIMATION_RENDER_DISPOSED', 'Program uniform arena is disposed')); },
    async whenIdle() { await Promise.race([validation, stopped]); live(); },
    get diagnostics() { return Object.freeze({allocatedBytes, shadowBytes: allocatedBytes, pages: pages.length,
      bindings: bindings.size, draws, maxDraws, maxBytes, alignment}); },
  });
}
