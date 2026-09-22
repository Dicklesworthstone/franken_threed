import assert from "node:assert/strict";
import test from "node:test";
import { imageBytes, jpegBase64, pngBase64 } from "./fixtures/animation/image_fixture.mjs";
import { createGltfTextureResources, gltfImageDimensions } from "./gltf_textures.mjs";

const png = imageBytes(pngBase64),
  jpeg = imageBytes(jpegBase64);
function gpu({ errorAt = 0, throwAt = 0 } = {}) {
  const textures = [],
    samplers = [],
    copies = [],
    passes = [],
    pipelines = [],
    scopes = [];
  let pops = 0,
    creates = 0,
    submits = 0,
    done = 0,
    lose;
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise((resolve) => {
      lose = resolve;
    }),
    pushErrorScope(type) {
      scopes.push(type);
    },
    popErrorScope() {
      assert.ok(scopes.pop());
      pops++;
      return Promise.resolve(pops === errorAt ? { message: "injected validation" } : null);
    },
    createTexture(descriptor) {
      if (++creates === throwAt) throw new Error("injected create failure");
      const texture = {
        descriptor,
        destroyed: 0,
        destroy() {
          this.destroyed++;
        },
        createView(settings = {}) {
          return { texture: this, settings };
        },
      };
      textures.push(texture);
      return texture;
    },
    createSampler(descriptor) {
      const sampler = { descriptor };
      samplers.push(sampler);
      return sampler;
    },
    createShaderModule(descriptor) {
      return descriptor;
    },
    async createRenderPipelineAsync(descriptor) {
      assert.equal(scopes.length, 2);
      const pipeline = {
        descriptor,
        getBindGroupLayout() {
          return {};
        },
      };
      pipelines.push(pipeline);
      return pipeline;
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(descriptor) {
          const pass = {
            descriptor,
            setPipeline(p) {
              this.pipeline = p;
            },
            setBindGroup(i, g) {
              this.group = g;
            },
            draw(n) {
              this.count = n;
            },
            end() {
              this.ended = true;
            },
          };
          passes.push(pass);
          return pass;
        },
        finish() {
          return {};
        },
      };
    },
    queue: {
      copyExternalImageToTexture(source, destination, size) {
        copies.push({ source, destination, size });
      },
      submit() {
        submits++;
      },
      async onSubmittedWorkDone() {
        done++;
      },
    },
  };
  return {
    device,
    textures,
    samplers,
    copies,
    passes,
    pipelines,
    scopes,
    lose,
    get submits() {
      return submits;
    },
    get done() {
      return done;
    },
  };
}
const request = (textureIndex = 0, imageIndex = 0, colorSpace = "srgb", sampler = {}) => ({
  textureIndex,
  imageIndex,
  colorSpace,
  sampler,
});
function decoder() {
  const bitmaps = [],
    calls = [];
  return {
    bitmaps,
    calls,
    async decode(blob, options) {
      calls.push({ blob, options });
      const d = gltfImageDimensions(new Uint8Array(await blob.arrayBuffer()), blob.type);
      const bitmap = {
        ...d,
        closed: 0,
        close() {
          this.closed++;
        },
      };
      bitmaps.push(bitmap);
      return bitmap;
    },
  };
}
const readImage = async () => ({ bytes: png, mimeType: "image/png" });
test("extracts dimensions from real PNG and JPEG fixtures", () => {
  assert.deepEqual(gltfImageDimensions(png, "image/png"), { width: 2, height: 2 });
  assert.deepEqual(gltfImageDimensions(jpeg, "image/jpeg"), { width: 2, height: 2 });
  for (const [data, type] of [
    [png, "image/jpeg"],
    [jpeg, "image/png"],
    [png.subarray(0, 20), "image/png"],
    [jpeg.subarray(0, 10), "image/jpeg"],
  ])
    assert.throws(() => gltfImageDimensions(data, type), { code: "GLTF_TEXTURE_IMAGE" });
});
test("uploads once per image/color space, realizes mip levels and preserves orientation", async () => {
  const g = gpu(),
    d = decoder();
  const r = await createGltfTextureResources(g.device, [request()], readImage, {
    createImageBitmap: d.decode,
  });
  assert.equal(r.textureCount, 1);
  assert.equal(r.textureBytes, 20);
  assert.equal(g.textures[0].descriptor.mipLevelCount, 2);
  assert.equal(g.textures[0].descriptor.format, "rgba8unorm");
  assert.deepEqual(g.textures[0].descriptor.viewFormats, ["rgba8unorm-srgb"]);
  assert.deepEqual(d.calls[0].options, {
    imageOrientation: "none",
    premultiplyAlpha: "none",
    colorSpaceConversion: "none",
  });
  assert.equal(g.copies[0].source.flipY, false);
  assert.equal(g.copies[0].destination.premultipliedAlpha, false);
  assert.equal(d.bitmaps[0].closed, 1);
  assert.equal(g.passes.length, 1);
  assert.equal(g.passes[0].count, 3);
  assert.ok(g.passes[0].ended);
  assert.equal(g.passes[0].descriptor.colorAttachments[0].view.settings.baseMipLevel, 1);
  assert.equal(g.passes[0].group.entries[0].resource.settings.baseMipLevel, 0);
  assert.equal(g.pipelines[0].descriptor.fragment.targets[0].format, "rgba8unorm-srgb");
  assert.equal(g.done, 1);
  assert.equal(r.resolveTexture(request()).view.settings.format, "rgba8unorm-srgb");
  assert.deepEqual(g.scopes, []);
  r.dispose();
  r.dispose();
  assert.equal(g.textures[0].destroyed, 1);
  assert.equal(r.textureBytes, 0);
  assert.throws(() => r.resolveTexture(request()), { code: "GLTF_TEXTURE_DISPOSED" });
});
test("same image in sRGB and linear maps gets separate mip domains, one image decode", async () => {
  const g = gpu(),
    d = decoder();
  let reads = 0;
  const r = await createGltfTextureResources(
    g.device,
    [request(), request(0, 0, "linear")],
    async () => {
      reads++;
      return readImage();
    },
    { createImageBitmap: d.decode },
  );
  assert.equal(reads, 1);
  assert.equal(d.calls.length, 1);
  assert.equal(r.textureCount, 2);
  assert.equal(r.textureBytes, 40);
  assert.notEqual(r.resolveTexture(request()).view, r.resolveTexture(request(0, 0, "linear")).view);
  assert.deepEqual(
    g.pipelines.map((p) => p.descriptor.fragment.targets[0].format),
    ["rgba8unorm-srgb", "rgba8unorm"],
  );
  r.dispose();
});
test("different samplers share image pixels but not sampler behavior; nonmip LOD stays zero", async () => {
  const g = gpu(),
    d = decoder(),
    a = request(0, 0, "linear", { minFilter: 9728, magFilter: 9728, wrapS: 33071, wrapT: 33648 }),
    b = request(1, 0, "linear", { minFilter: 9987 });
  const r = await createGltfTextureResources(g.device, [a, b, b], readImage, {
    createImageBitmap: d.decode,
  });
  const ar = r.resolveTexture(a),
    br = r.resolveTexture(b);
  assert.equal(ar.view, br.view);
  assert.notEqual(ar.sampler, br.sampler);
  assert.deepEqual(ar.sampler.descriptor, {
    addressModeU: "clamp-to-edge",
    addressModeV: "mirror-repeat",
    magFilter: "nearest",
    minFilter: "nearest",
    mipmapFilter: "nearest",
    lodMaxClamp: 0,
  });
  assert.equal(br.sampler.descriptor.lodMaxClamp, undefined);
  assert.equal(r.textureCount, 1);
  r.dispose();
});
for (const [minFilter, min, mip, mips] of [
  [9728, "nearest", "nearest", false],
  [9729, "linear", "nearest", false],
  [9984, "nearest", "nearest", true],
  [9985, "linear", "nearest", true],
  [9986, "nearest", "linear", true],
  [9987, "linear", "linear", true],
])
  test(`realizes glTF minification filter ${minFilter}`, async () => {
    const g = gpu(),
      d = decoder(),
      q = request(0, 0, "linear", { minFilter });
    const r = await createGltfTextureResources(g.device, [q], readImage, {
      createImageBitmap: d.decode,
    });
    const s = r.resolveTexture(q).sampler.descriptor;
    assert.equal(s.minFilter, min);
    assert.equal(s.mipmapFilter, mip);
    assert.equal(g.passes.length, mips ? 1 : 0);
    assert.equal(g.textures[0].descriptor.mipLevelCount, mips ? 2 : 1);
    r.dispose();
  });
test("default sampling policy can be changed without overriding explicit source filters", async () => {
  const g = gpu(),
    d = decoder(),
    a = request(),
    b = request(1, 0, "srgb", { minFilter: 9987, magFilter: 9729 });
  const r = await createGltfTextureResources(g.device, [a, b], readImage, {
    createImageBitmap: d.decode,
    defaultMinFilter: 9728,
    defaultMagFilter: 9728,
  });
  assert.equal(r.resolveTexture(a).sampler.descriptor.magFilter, "nearest");
  assert.equal(r.resolveTexture(b).sampler.descriptor.magFilter, "linear");
  r.dispose();
});
test("invalid and conflicting requests fail before provider, decoder or GPU effects", async () => {
  const cases = [
    [request(0, 0, "bad")],
    [request(0, 0, "linear", { wrapS: 99 })],
    [request(0, 0, "linear", { minFilter: 99 })],
    [request(-1)],
    [request(), request(0, 1)],
    [request(), request(0, 0, "srgb", { minFilter: 9728 })],
  ];
  for (const requests of cases)
    await assert.rejects(
      createGltfTextureResources(null, requests, () => assert.fail("No image effects")),
    );
});
test("dimensions and all color-space allocations are budgeted before decoding", async () => {
  for (const options of [{ maxImagePixels: 3 }, { maxTextureBytes: 39 }]) {
    const g = gpu();
    await assert.rejects(
      createGltfTextureResources(g.device, [request(), request(0, 0, "linear")], readImage, {
        ...options,
        createImageBitmap: () => assert.fail("No decode"),
      }),
      { code: "GLTF_TEXTURE_LIMIT" },
    );
    assert.equal(g.textures.length, 0);
  }
  const g = gpu();
  g.device.limits.maxTextureDimension2D = 1;
  await assert.rejects(
    createGltfTextureResources(g.device, [request()], readImage, {
      createImageBitmap: () => assert.fail("No decode"),
    }),
    { code: "GLTF_TEXTURE_LIMIT" },
  );
});
test("image decode rejection cleans previous allocations", async () => {
  const g = gpu(),
    d = decoder();
  let n = 0;
  await assert.rejects(
    createGltfTextureResources(g.device, [request(), request(1, 1)], readImage, {
      createImageBitmap: async (...args) => {
        if (++n === 2) throw new Error("decode failed");
        return d.decode(...args);
      },
    }),
    /decode failed/,
  );
  assert.equal(g.textures[0].destroyed, 1);
  assert.equal(d.bitmaps[0].closed, 1);
});
test("dimension mismatch closes the native image without allocating textures", async () => {
  const g = gpu();
  let closed = 0;
  await assert.rejects(
    createGltfTextureResources(g.device, [request()], readImage, {
      createImageBitmap: async () => ({
        width: 3,
        height: 2,
        close() {
          closed++;
        },
      }),
    }),
    { code: "GLTF_TEXTURE_IMAGE" },
  );
  assert.equal(closed, 1);
  assert.equal(g.textures.length, 0);
});
test("GPU validation and synchronous failures release every owned texture and bitmap", async () => {
  for (const settings of [{ errorAt: 1 }, { errorAt: 3 }, { throwAt: 2 }]) {
    const g = gpu(settings),
      d = decoder();
    await assert.rejects(
      createGltfTextureResources(g.device, [request(), request(1, 1)], readImage, {
        createImageBitmap: d.decode,
      }),
    );
    for (const t of g.textures) assert.equal(t.destroyed, 1);
    for (const b of d.bitmaps) assert.equal(b.closed, 1);
    assert.deepEqual(g.scopes, []);
  }
});
test("abort after native decoding starts closes its late result and creates no textures", async () => {
  const g = gpu(),
    controller = new AbortController();
  let finish,
    closed = 0;
  const pending = createGltfTextureResources(g.device, [request()], readImage, {
    signal: controller.signal,
    createImageBitmap: () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  });
  await new Promise((resolve) => setImmediate(resolve));
  controller.abort();
  finish({
    width: 2,
    height: 2,
    close() {
      closed++;
    },
  });
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(closed, 1);
  assert.equal(g.textures.length, 0);
});
test("device loss releases uploads and makes resolution fail explicitly", async () => {
  const g = gpu(),
    d = decoder(),
    r = await createGltfTextureResources(g.device, [request()], readImage, {
      createImageBitmap: d.decode,
    });
  g.lose({ message: "test device lost" });
  await Promise.resolve();
  assert.equal(r.failed, true);
  assert.equal(g.textures[0].destroyed, 1);
  assert.throws(() => r.resolveTexture(request()), { code: "GLTF_TEXTURE_DEVICE_LOST" });
  r.dispose();
  assert.equal(g.textures[0].destroyed, 1);
});
test("queue completion failures reject publication and release resources", async () => {
  const g = gpu(),
    d = decoder();
  g.device.queue.onSubmittedWorkDone = async () => {
    throw new Error("queue failed");
  };
  await assert.rejects(
    createGltfTextureResources(g.device, [request()], readImage, { createImageBitmap: d.decode }),
    /queue failed/,
  );
  assert.equal(g.textures[0].destroyed, 1);
  assert.equal(d.bitmaps[0].closed, 1);
});
test("empty request sets need no GPU or image decoder", async () => {
  const r = await createGltfTextureResources(null, [], () => assert.fail("No image read"), {
    createImageBitmap: null,
  });
  assert.equal(r.textureCount, 0);
  r.dispose();
});
test("only prepared texture identities resolve and image decoding runs outside GPU scopes", async () => {
  const g = gpu(),
    d = decoder();
  const r = await createGltfTextureResources(g.device, [request()], readImage, {
    createImageBitmap: async (...args) => {
      assert.deepEqual(g.scopes, []);
      return d.decode(...args);
    },
  });
  assert.throws(() => r.resolveTexture(request(99)), { code: "GLTF_TEXTURE_REQUEST" });
  assert.throws(() => r.resolveTexture(request(0, 99)), { code: "GLTF_TEXTURE_REQUEST" });
  r.dispose();
});
