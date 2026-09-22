import assert from "node:assert/strict";
import test from "node:test";
import { planAnimationEnvironment } from "./animation_environment.mjs";
import { loadGpuAnimationEnvironment } from "./animation_environment_loader.mjs";
import { decodeAnimationHdr } from "./animation_hdr.mjs";

const tick = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((r, j) => {
    resolve = r;
    reject = j;
  });
  return { promise, resolve, reject };
};
const small = { size: 8, diffuseSize: 2, lutSize: 4, samples: 16 };
function hdr(width = 8, height = 4, e = 130) {
  const header = new TextEncoder().encode(
    `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y ${height} +X ${width}\n`,
  );
  return new Uint8Array([
    ...header,
    ...Array.from({ length: width * height }, (_, i) => [128 + (i % 128), 0, 255, e]).flat(),
  ]);
}
// Only the browser/driver boundary is recorded. The actual HDR decoder, planner,
// upload path, filtering owner, snapshots and cleanup all execute unchanged.
function gpu({ compilation, completion, uploadScope, failTexture = 0, uploadThrow = null } = {}) {
  const textures = [],
    buffers = [],
    events = [],
    scopes = [],
    lost = deferred();
  let depth = 0,
    pops = 0;
  const device = {
    limits: {
      maxTextureDimension2D: 4096,
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 2 ** 26,
    },
    lost: lost.promise,
    queue: {
      writeTexture(destination, data, layout, size) {
        events.push(["upload", destination, data.slice(), layout, size]);
        if (uploadThrow) throw uploadThrow;
      },
      writeBuffer(buffer, offset, data) {
        events.push(["uniforms", buffer, offset, new Uint8Array(data).slice()]);
      },
      submit(value) {
        events.push(["submit", value]);
      },
      onSubmittedWorkDone: () => completion?.promise ?? Promise.resolve(),
    },
    pushErrorScope(kind) {
      depth++;
      scopes.push(kind);
    },
    popErrorScope() {
      depth--;
      scopes.pop();
      pops++;
      return pops === 1 && uploadScope ? uploadScope.promise : Promise.resolve(null);
    },
    createTexture(d) {
      if (textures.length + 1 === failTexture) throw new Error("texture allocation");
      const texture = {
        ...d,
        width: d.size[0],
        height: d.size[1],
        depthOrArrayLayers: d.size[2],
        sampleCount: d.sampleCount ?? 1,
        destroyed: 0,
        createView(descriptor = {}) {
          return { texture: this, descriptor };
        },
        destroy() {
          this.destroyed++;
        },
      };
      textures.push(texture);
      return texture;
    },
    createBuffer(d) {
      const b = {
        ...d,
        destroyed: 0,
        destroy() {
          this.destroyed++;
        },
      };
      buffers.push(b);
      return b;
    },
    createSampler: (d) => d,
    createBindGroupLayout: (d) => d,
    createPipelineLayout: (d) => d,
    createShaderModule(d) {
      events.push(["shader", d]);
      return d;
    },
    createRenderPipelineAsync: (d) => compilation?.promise ?? Promise.resolve(d),
    createBindGroup(d) {
      events.push(["group", d]);
      return d;
    },
    createCommandEncoder: () => ({
      beginRenderPass(d) {
        events.push(["pass", d]);
        return {
          setPipeline() {},
          setBindGroup() {},
          draw(n) {
            events.push(["draw", n]);
          },
          end() {},
        };
      },
      finish: () => ({}),
    }),
  };
  return {
    device,
    textures,
    buffers,
    events,
    lost,
    get depth() {
      return depth;
    },
  };
}
const errorCode = (code) => (e) => e.code === code;
async function assertPrompt(promise, expected) {
  // No wall-clock benchmark: an ignored cancellation must settle by the next
  // event-loop turn instead of leaving a pending task with no handles.
  const result = promise.then(
    () => ({ resolved: true }),
    (error) => ({ error }),
  );
  const answer = await Promise.race([result, tick().then(() => ({ pending: true }))]);
  assert.equal(answer.pending, undefined, "operation did not reject promptly");
  assert.ok(Object.hasOwn(answer, "error"));
  if (expected) assert.ok(expected(answer.error), String(answer.error));
}
function streamed(chunks, { length, status = 200, redirected = false, url = "" } = {}) {
  let cancelled = 0,
    reads = 0,
    unlocked = 0;
  const reader = {
    async read() {
      reads++;
      return chunks.length ? { value: chunks.shift(), done: false } : { done: true };
    },
    cancel() {
      cancelled++;
      return Promise.resolve();
    },
    releaseLock() {
      unlocked++;
    },
  };
  const response = {
    ok: status >= 200 && status < 300,
    status,
    redirected,
    url,
    headers: { get: () => length ?? null },
    body: {
      getReader: () => reader,
      cancel() {
        cancelled++;
        return Promise.resolve();
      },
    },
  };
  return {
    response,
    reader,
    get cancelled() {
      return cancelled;
    },
    get reads() {
      return reads;
    },
    get unlocked() {
      return unlocked;
    },
  };
}

test("HDR bytes upload once then execute the actual IBL filter and release only temporary storage", async () => {
  const g = gpu(),
    bytes = hdr(),
    map = await loadGpuAnimationEnvironment(g.device, bytes, small);
  const upload = g.events.find((e) => e[0] === "upload");
  assert.deepEqual(upload[2], decodeAnimationHdr(bytes).data);
  assert.deepEqual(upload[3], { offset: 0, bytesPerRow: 64, rowsPerImage: 4 });
  assert.deepEqual(upload[4], [8, 4, 1]);
  assert.equal(g.events.filter((e) => e[0] === "upload").length, 1);
  assert.equal(g.events.filter((e) => e[0] === "submit").length, 1);
  assert.equal(
    g.events.filter((e) => e[0] === "pass").length,
    planAnimationEnvironment(small).passes.length,
  );
  assert.equal(g.textures.length, 4);
  assert.equal(g.textures[0].usage, 6);
  assert.equal(g.textures[0].destroyed, 1);
  assert.ok(g.textures.slice(1).every((t) => t.destroyed === 0));
  assert.ok(g.buffers.every((b) => b.destroyed === 1));
  assert.equal(
    g.events.find((e) => e[0] === "group")[1].entries[2].resource.texture,
    g.textures[0],
  );
  const snapshot = map.sample(g.device);
  assert.equal(snapshot.profile, "f3d-animation-environment-v1");
  assert.equal(snapshot.diffuseView, map.diffuseView);
  assert.equal(await map.whenIdle(), map);
  assert.equal(map.sourceInfo.inputBytes, bytes.length);
  assert.equal(map.sourceInfo.peakTextureBytes, map.textureBytes + 256);
  assert.ok(Object.isFrozen(map.sourceInfo));
  assert.equal(g.depth, 0);
  assert.throws(() => map.sample({}), errorCode("ANIMATION_ENVIRONMENT_DEVICE"));
  map.dispose();
  map.dispose();
  assert.equal(map.disposed, true);
  assert.equal(map.textureBytes, 0);
  assert.ok(g.textures.every((t) => t.destroyed === 1));
});
test("byte inputs snapshot decoded pixels before the first asynchronous driver boundary", async () => {
  const uploadScope = deferred(),
    g = gpu({ uploadScope }),
    bytes = hdr(),
    expected = decodeAnimationHdr(bytes).data;
  const pending = loadGpuAnimationEnvironment(g.device, bytes, small);
  bytes.fill(0);
  assert.deepEqual(g.events.find((e) => e[0] === "upload")[2], expected);
  assert.equal(g.depth, 0);
  uploadScope.resolve(null);
  const map = await pending;
  map.dispose();
});
test("HTTP streaming resolves relative URLs, preserves query, strips fragment and sends no credentials", async () => {
  const bytes = hdr(),
    s = streamed(Array.from(bytes, (b) => Uint8Array.of(b)));
  let request;
  const g = gpu(),
    map = await loadGpuAnimationEnvironment(g.device, "../studio.hdr?v=2#view", {
      ...small,
      baseURL: "https://assets.example/maps/scene/",
      fetch: async (...args) => {
        request = args;
        return s.response;
      },
    });
  assert.equal(request[0], "https://assets.example/maps/studio.hdr?v=2");
  assert.equal(request[1].credentials, "omit");
  assert.equal(request[1].redirect, "error");
  assert.equal(request[1].signal.aborted, false);
  assert.equal(s.cancelled, 0);
  assert.equal(s.unlocked, 1);
  assert.deepEqual(g.events.find((e) => e[0] === "upload")[2], decodeAnimationHdr(bytes).data);
  map.dispose();
});
test("stream assembly handles reusable offset chunks without retaining their mutable views", async () => {
  const bytes = hdr(),
    reusable = new Uint8Array(7);
  let at = 0;
  const response = {
    ok: true,
    body: {
      getReader: () => ({
        async read() {
          if (at === bytes.length) return { done: true };
          reusable.fill(0);
          const end = Math.min(bytes.length, at + 5);
          reusable.set(bytes.subarray(at, end), 1);
          const length = end - at;
          at = end;
          return { done: false, value: reusable.subarray(1, 1 + length) };
        },
        releaseLock() {},
        cancel() {},
      }),
    },
  };
  const g = gpu(),
    map = await loadGpuAnimationEnvironment(g.device, "https://assets.example/env.hdr", {
      ...small,
      fetch: async () => response,
    });
  assert.deepEqual(g.events.find((e) => e[0] === "upload")[2], decodeAnimationHdr(bytes).data);
  map.dispose();
});
for (const length of [undefined, "1", String(hdr().length + 1)])
  test("encoded budget rejects actual bytes/declared size: " + length, async () => {
    const bytes = hdr(),
      s = streamed([bytes], { length }),
      g = gpu();
    await assert.rejects(
      loadGpuAnimationEnvironment(g.device, "https://assets.example/env.hdr", {
        ...small,
        maxInputBytes: bytes.length - 1,
        fetch: async () => s.response,
      }),
      errorCode("ANIMATION_HDR_LIMIT"),
    );
    assert.equal(s.cancelled, 1);
    assert.equal(g.textures.length, 0);
  });
test("exact peak texture and decoded scratch budgets accept, one byte less fails before GPU effects", async () => {
  const bytes = hdr(),
    plan = planAnimationEnvironment(small),
    exact = plan.textureBytes + 256;
  const g = gpu(),
    map = await loadGpuAnimationEnvironment(g.device, bytes, {
      ...small,
      maxInputBytes: bytes.length,
      maxDecodedBytes: 288,
      maxTextureBytes: exact,
    });
  assert.equal(map.sourceInfo.peakTextureBytes, exact);
  map.dispose();
  for (const options of [{ maxTextureBytes: exact - 1 }, { maxDecodedBytes: 287 }]) {
    const h = gpu();
    await assert.rejects(
      loadGpuAnimationEnvironment(h.device, bytes, { ...small, ...options }),
      errorCode("ANIMATION_HDR_LIMIT"),
    );
    assert.equal(h.events.length, 0);
  }
});
test("non-panorama HDR and unsupported input/profile reject without GPU allocation", async () => {
  for (const source of [
    hdr(4, 4),
    new Uint8Array(10),
    {},
    new Uint8Array(new SharedArrayBuffer(10)),
  ]) {
    const g = gpu();
    await assert.rejects(loadGpuAnimationEnvironment(g.device, source, small));
    assert.equal(g.events.length, 0);
  }
});
test("clamping is explicit and its count survives in the prepared map metadata", async () => {
  const g = gpu();
  await assert.rejects(
    loadGpuAnimationEnvironment(g.device, hdr(2, 1, 150), small),
    errorCode("ANIMATION_HDR_RANGE"),
  );
  const map = await loadGpuAnimationEnvironment(g.device, hdr(2, 1, 150), {
    ...small,
    overflow: "clamp",
  });
  assert.equal(map.sourceInfo.clampedComponents, 4);
  map.dispose();
});
test("invalid options/device/URL fail before fetching", async () => {
  let fetched = 0;
  const fetcher = async () => {
    fetched++;
    throw new Error("unexpected fetch");
  };
  for (const source of [
    "file:///env.hdr",
    "data:image/vnd.radiance,bytes",
    "https://name:secret@example.com/env.hdr",
    "relative.hdr",
  ])
    await assert.rejects(
      loadGpuAnimationEnvironment(gpu().device, source, { ...small, fetch: fetcher }),
      errorCode("ANIMATION_HDR_URL"),
    );
  for (const options of [
    { samples: 0 },
    { size: 7 },
    { maxInputBytes: 0 },
    { maxSampleWork: 1 },
    { signal: {} },
    { unknown: true },
  ])
    await assert.rejects(
      loadGpuAnimationEnvironment(gpu().device, "https://example.com/env.hdr", {
        ...small,
        ...options,
        fetch: fetcher,
      }),
    );
  const g = gpu();
  g.device.queue.writeTexture = null;
  await assert.rejects(
    loadGpuAnimationEnvironment(g.device, "https://example.com/env.hdr", {
      ...small,
      fetch: fetcher,
    }),
    errorCode("ANIMATION_HDR_DEVICE"),
  );
  assert.equal(fetched, 0);
});
for (const responseOptions of [
  { status: 404 },
  { redirected: true },
  { url: "https://other.example/env.hdr" },
])
  test(
    "HTTP/redirect failure cancels body before reading " + JSON.stringify(responseOptions),
    async () => {
      const s = streamed([hdr()], responseOptions),
        g = gpu();
      await assert.rejects(
        loadGpuAnimationEnvironment(g.device, "https://assets.example/env.hdr", {
          ...small,
          fetch: async () => s.response,
        }),
        errorCode("ANIMATION_HDR_HTTP"),
      );
      assert.equal(s.cancelled, 1);
      assert.equal(s.reads, 0);
      assert.equal(g.textures.length, 0);
    },
  );
test("pre-abort avoids all work and late abort does not revoke a successfully prepared map", async () => {
  const g = gpu(),
    c = new AbortController();
  c.abort("stop");
  await assert.rejects(
    loadGpuAnimationEnvironment(g.device, hdr(), { ...small, signal: c.signal }),
    (e) => e === "stop",
  );
  assert.equal(g.events.length, 0);
  const d = new AbortController(),
    map = await loadGpuAnimationEnvironment(g.device, hdr(), { ...small, signal: d.signal });
  d.abort("too late");
  assert.equal(map.failed, false);
  assert.equal(await map.whenIdle(), map);
  map.dispose();
});
test("abort promptly rejects a fetch that ignores signal and cancels its late response", async () => {
  const g = gpu(),
    c = new AbortController(),
    request = deferred(),
    s = streamed([hdr()]);
  let transportSignal;
  const pending = loadGpuAnimationEnvironment(g.device, "https://example.com/env.hdr", {
    ...small,
    signal: c.signal,
    fetch: (url, options) => {
      transportSignal = options.signal;
      return request.promise;
    },
  });
  c.abort("cancel fetch");
  await assertPrompt(pending, (e) => e === "cancel fetch");
  assert.equal(transportSignal.aborted, true);
  request.resolve(s.response);
  await tick();
  assert.equal(s.cancelled, 1);
  assert.equal(g.textures.length, 0);
});
test("abort promptly cancels an unresolved reader, releases its lock and ignores late bytes", async () => {
  const g = gpu(),
    c = new AbortController(),
    reading = deferred(),
    s = streamed([]);
  s.reader.read = () => reading.promise;
  const pending = loadGpuAnimationEnvironment(g.device, "https://example.com/env.hdr", {
    ...small,
    signal: c.signal,
    fetch: async () => s.response,
  });
  await tick();
  c.abort("cancel read");
  await assertPrompt(pending, (e) => e === "cancel read");
  assert.equal(s.cancelled, 1);
  assert.equal(s.unlocked, 1);
  reading.resolve({ done: false, value: hdr() });
  await tick();
  assert.equal(g.textures.length, 0);
});
for (const boundary of ["uploadScope", "compilation", "completion"])
  test("abort at " + boundary + " releases every owned resource exactly once", async () => {
    const delay = deferred(),
      g = gpu({ [boundary]: delay }),
      c = new AbortController();
    const pending = loadGpuAnimationEnvironment(g.device, hdr(), { ...small, signal: c.signal });
    await tick();
    assert.ok(g.textures.length >= 1);
    assert.equal(g.textures[0].destroyed, 0);
    assert.equal(g.depth, 0);
    c.abort();
    await assertPrompt(pending, (e) => e.name === "AbortError");
    assert.ok(g.textures.every((t) => t.destroyed === 1));
    delay.resolve(boundary === "uploadScope" ? null : {});
    await tick();
    assert.ok(g.textures.every((t) => t.destroyed === 1));
  });
test("completion is awaited before retiring the panorama or publishing receiver views", async () => {
  const completion = deferred(),
    g = gpu({ completion });
  let published = false;
  const pending = loadGpuAnimationEnvironment(g.device, hdr(), small).then((map) => {
    published = true;
    return map;
  });
  await tick();
  assert.equal(published, false);
  assert.equal(g.textures.length, 4);
  assert.equal(g.textures[0].destroyed, 0);
  completion.resolve();
  const map = await pending;
  assert.equal(g.textures[0].destroyed, 1);
  map.dispose();
});
for (const options of [
  { failTexture: 1 },
  { failTexture: 2 },
  { failTexture: 3 },
  { uploadThrow: new Error("upload failed") },
])
  test(
    "partial upload/filter allocation failure cleans owned resources " + JSON.stringify(options),
    async () => {
      const g = gpu(options);
      await assert.rejects(loadGpuAnimationEnvironment(g.device, hdr(), small));
      assert.ok(g.textures.every((t) => t.destroyed === 1));
      assert.ok(g.buffers.every((b) => b.destroyed === 1));
      assert.equal(g.depth, 0);
    },
  );
test("GPU validation and completion errors reject rather than publishing a broken environment", async () => {
  const uploadScope = deferred(),
    g = gpu({ uploadScope }),
    pending = loadGpuAnimationEnvironment(g.device, hdr(), small);
  uploadScope.resolve({ message: "invalid upload" });
  await assert.rejects(pending, errorCode("ANIMATION_HDR_GPU"));
  assert.equal(g.textures.length, 1);
  assert.equal(g.textures[0].destroyed, 1);
  const completion = deferred(),
    h = gpu({ completion }),
    failed = loadGpuAnimationEnvironment(h.device, hdr(), small);
  await tick();
  completion.reject(new Error("queue failed"));
  await assert.rejects(failed, /queue failed/);
  assert.ok(h.textures.every((t) => t.destroyed === 1));
});
test("device loss cancels network work; after publication loss propagates through map getters", async () => {
  const g = gpu(),
    request = deferred();
  let signal;
  const pending = loadGpuAnimationEnvironment(g.device, "https://example.com/env.hdr", {
    ...small,
    fetch: (url, options) => {
      signal = options.signal;
      return request.promise;
    },
  });
  g.lost.resolve({ message: "lost during fetch" });
  await assertPrompt(pending, errorCode("ANIMATION_ENVIRONMENT_LOST"));
  assert.equal(signal.aborted, true);
  const s = streamed([hdr()]);
  request.resolve(s.response);
  await tick();
  assert.equal(s.cancelled, 1);
  const h = gpu(),
    map = await loadGpuAnimationEnvironment(h.device, hdr(), small);
  h.lost.resolve({ message: "lost after load" });
  await tick();
  assert.equal(map.failed, true);
  assert.equal(map.textureBytes, 0);
  await assert.rejects(map.whenIdle(), errorCode("ANIMATION_ENVIRONMENT_LOST"));
  assert.throws(() => map.sample(h.device), errorCode("ANIMATION_ENVIRONMENT_LOST"));
  map.dispose();
  assert.ok(h.textures.every((t) => t.destroyed === 1));
});

test("streaming grows past 64 KiB without losing packet bytes or weakening the exact input limit", async () => {
  const bytes = hdr(256, 128),
    chunks = [];
  for (let at = 0; at < bytes.length; at += 3001) chunks.push(bytes.subarray(at, at + 3001));
  const s = streamed(chunks),
    g = gpu();
  const map = await loadGpuAnimationEnvironment(g.device, "https://example.com/env.hdr", {
    ...small,
    maxInputBytes: bytes.length,
    fetch: async () => s.response,
  });
  assert.equal(map.sourceInfo.inputBytes, bytes.length);
  assert.deepEqual(g.events.find((e) => e[0] === "upload")[2], decodeAnimationHdr(bytes).data);
  map.dispose();
});
test("stored exposure is reported without modifying GPU radiance and construction listeners detach", async () => {
  const body = hdr(2, 1),
    text = new TextDecoder().decode(body.subarray(0, body.length - 8));
  const bytes = new Uint8Array([
    ...new TextEncoder().encode(text.replace("FORMAT=", "EXPOSURE=2\nEXPOSURE=3\nFORMAT=")),
    ...body.subarray(-8),
  ]);
  const c = new AbortController();
  let added = 0,
    removed = 0;
  const signal = {
    get aborted() {
      return c.signal.aborted;
    },
    get reason() {
      return c.signal.reason;
    },
    addEventListener(...args) {
      added++;
      c.signal.addEventListener(...args);
    },
    removeEventListener(...args) {
      removed++;
      c.signal.removeEventListener(...args);
    },
  };
  const g = gpu(),
    map = await loadGpuAnimationEnvironment(g.device, bytes, { ...small, signal });
  assert.equal(map.sourceInfo.exposure, 6);
  assert.deepEqual(g.events.find((e) => e[0] === "upload")[2], decodeAnimationHdr(body).data);
  assert.equal(added, 1);
  assert.equal(removed, 1);
  c.abort();
  assert.equal(map.failed, false);
  map.dispose();
});
