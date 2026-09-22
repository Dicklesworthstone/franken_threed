/** Material coordination unit tests. Execute the production scene body and
 * renderer; replace only the controller/deformer imports with explicit stubs.
 * Existing animation_scene.test.mjs covers the real animation/deformer chain.
 * This test does not execute WGSL, emulate rasterization or assert pixels.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const encoded = (source) => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const stub = encoded(`
export function createAnimationController(pose) {
  return {time:0,disposed:false,update(dt){if(!Number.isFinite(dt)||dt<0)throw Error('invalid delta');this.time+=dt;pose.version++;},dispose(){this.disposed=true;}};
}
export async function createGpuAnimationDeformer(device,pose,geometry,{maxBytes}) {
  const size=geometry.positions.length/3*40;
  if(size>maxBytes)throw Object.assign(Error('deformation budget'),{code:'ANIMATION_GPU_LIMIT'});
  const vertexBuffer=device.createBuffer({size,usage:160}),attributes=[{shaderLocation:0,offset:0,format:'float32x3'}];
  if(geometry.normals)attributes.push({shaderLocation:1,offset:12,format:'float32x3'});
  if(geometry.tangents)attributes.push({shaderLocation:2,offset:24,format:'float32x4'});
  return {vertexBuffer,vertexCount:size/40,bufferBytes:size,worldMatrix:new Float64Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),
    vertexLayout:{arrayStride:40,stepMode:'vertex',attributes},poseVersion:pose.version,disposed:false,failed:false,
    update(){this.poseVersion=pose.version;},whenIdle:async()=>{},dispose(){this.disposed=true;vertexBuffer.destroy();}};
}`);
let source = readFileSync(new URL("./animation_scene.mjs", import.meta.url), "utf8");
for (const name of ["animation_controller.mjs", "animation_webgpu.mjs", "animation_render.mjs"]) {
  const quoted = "'./" + name + "'";
  assert.equal(source.split(quoted).length, 2);
  source = source.replace(
    quoted,
    JSON.stringify(
      name === "animation_render.mjs" ? new URL("./" + name, import.meta.url).href : stub,
    ),
  );
}
const { createGpuAnimationScene } = await import(encoded(source));
function deviceSpy() {
  const buffers = [],
    pipelines = [],
    passes = [],
    writes = [],
    scopes = [];
  const d = {
    buffers,
    pipelines,
    passes,
    writes,
    lost: new Promise(() => {}),
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 1 << 26,
      maxUniformBufferBindingSize: 65536,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxBindGroups: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxVertexBuffers: 8,
      maxVertexAttributes: 16,
      maxVertexBufferArrayStride: 2048,
      maxSamplersPerShaderStage: 16,
      maxSampledTexturesPerShaderStage: 16,
    },
    pushErrorScope(v) {
      scopes.push(v);
    },
    popErrorScope() {
      assert.ok(scopes.pop());
      return Promise.resolve(null);
    },
    createBuffer(options) {
      const data = new ArrayBuffer(options.size),
        buffer = {
          ...options,
          data,
          destroyed: false,
          getMappedRange: () => data,
          unmap() {},
          destroy() {
            this.destroyed = true;
          },
        };
      buffers.push(buffer);
      return buffer;
    },
    createShaderModule: (v) => v,
    createBindGroupLayout: (v) => v,
    createPipelineLayout: (v) => v,
    createBindGroup: (v) => v,
    createRenderPipelineAsync(v) {
      pipelines.push(v);
      return Promise.resolve(v);
    },
    createCommandEncoder() {
      const draws = [];
      return {
        beginRenderPass() {
          let pipeline;
          const groups = new Map(),
            vertices = new Map();
          const draw = () =>
            draws.push({ pipeline, groups: new Map(groups), vertices: new Map(vertices) });
          return {
            setPipeline(v) {
              pipeline = v;
            },
            setBindGroup(slot, v, offsets = []) {
              groups.set(slot, { value: v, offset: offsets[0] ?? 0 });
            },
            setVertexBuffer(slot, v) {
              vertices.set(slot, v);
            },
            setIndexBuffer() {},
            draw,
            drawIndexed: draw,
            end() {},
          };
        },
        finish() {
          return draws;
        },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data, start = 0, size = data.length - start) {
        const n = data.BYTES_PER_ELEMENT,
          bytes = new Uint8Array(data.buffer, data.byteOffset + start * n, size * n);
        new Uint8Array(buffer.data, offset, bytes.length).set(bytes);
        writes.push(buffer);
      },
      submit(commands) {
        for (const draws of commands)
          for (const draw of draws) {
            const { value, offset } = draw.groups.get(0);
            draw.uniforms = new Float32Array(
              value.entries[0].resource.buffer.data,
              offset,
              64,
            ).slice();
            passes.push(draw);
          }
      },
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  };
  return d;
}
function drawable() {
  return {
    geometry: {
      node: 0,
      positions: [-1, -1, 0, 1, -1, 0, 0, 1, 0],
      normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
    },
    indices: [0, 1, 2],
    shading: "metallic-roughness",
    metallicFactor: 0.25,
    roughnessFactor: 0.5,
    emissiveFactor: [0.1, 0, 0],
    texCoords: [0, 0, 1, 0, 0, 1],
    vertexColors: Array(9).fill(0.5),
    baseColorTexture: { view: {}, sampler: {} },
    uvTransform: [1, 0, 0, 1, 0.25, 0.5],
  };
}
const frame = () => ({
  colorView: {},
  depthView: {},
  viewProjection: identity(),
  lighting: { viewDirection: [0, 0, 1], lights: [{ type: "directional" }] },
});

test("scene forwards snapshotted textures, UVs, colors and lit factors into the actual renderer", async () => {
  const d = deviceSpy(),
    pose = { version: 0, disposed: false },
    a = drawable(),
    b = drawable(),
    view = a.baseColorTexture.view;
  b.metallicFactor = 0.75;
  const pending = createGpuAnimationScene(d, pose, [a, b], { maxBytes: 1460 });
  a.texCoords.fill(99);
  a.vertexColors.fill(99);
  a.emissiveFactor.fill(99);
  a.uvTransform.fill(99);
  a.baseColorTexture.view = { replaced: true };
  a.metallicFactor = 99;
  const scene = await pending;
  assert.equal(scene.bufferBytes, 1456);
  for (let i = 0; i < 60; i++) {
    scene.update(1 / 60);
    scene.render(frame());
  }
  await scene.whenIdle();
  const [first, second] = d.passes.slice(-2);
  assert.equal(first.uniforms[22], 2);
  assert.equal(first.uniforms[23], 0.25);
  assert.equal(second.uniforms[23], 0.75);
  assert.equal(first.uniforms[63], 0.5);
  assert.ok(Math.abs(first.uniforms[60] - 0.1) < 1e-6);
  assert.deepEqual([...first.uniforms.slice(24, 32)], [1, 0, 0.25, 0, 0, 1, 0.5, 0]);
  assert.equal(first.groups.get(1).value.entries[1].resource, view);
  assert.deepEqual(
    [...new Float32Array(first.vertices.get(1).data).slice(0, 6)],
    [0, 0, 0.5, 0.5, 0.5, 1],
  );
  assert.equal(
    d.buffers.filter((b) => b.size === 544).length,
    1,
    "shared light block reserved once",
  );
  assert.equal(scene.poseVersion, 60);
  assert.ok(Math.abs(scene.controller.time - 1) < 1e-12);
  scene.dispose();
  assert.equal(pose.disposed, false);
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.deepEqual(view, {});
});

test("aggregate scene budget reserves attributes and light storage before deformation allocation", async () => {
  const d = deviceSpy(),
    pose = { version: 0, disposed: false };
  await assert.rejects(createGpuAnimationScene(d, pose, [drawable()], { maxBytes: 999 }), {
    code: "ANIMATION_GPU_LIMIT",
  });
  assert.equal(d.buffers.length, 1);
  assert.ok(d.buffers.every((b) => b.destroyed));
  const next = deviceSpy(),
    scene = await createGpuAnimationScene(next, pose, [drawable()], { maxBytes: 1004 });
  assert.equal(scene.bufferBytes, 1000);
  scene.dispose();
});

test("bad later material unwinds the whole scene and retains borrowed texture ownership", async () => {
  const d = deviceSpy(),
    pose = { version: 0, disposed: false },
    a = drawable(),
    b = drawable();
  b.shading = "unsupported";
  await assert.rejects(createGpuAnimationScene(d, pose, [a, b]), {
    code: "ANIMATION_RENDER_OPTIONS",
  });
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.equal(pose.disposed, false);
  assert.deepEqual(a.baseColorTexture, { view: {}, sampler: {} });
});

test("unknown texture behavior is not stripped or silently accepted by scene forwarding", async () => {
  const d = deviceSpy(),
    item = drawable();
  item.baseColorTexture.flipY = true;
  await assert.rejects(createGpuAnimationScene(d, { version: 0, disposed: false }, [item]), {
    code: "ANIMATION_RENDER_OPTIONS",
  });
  assert.ok(d.buffers.every((b) => b.destroyed));
});

test("invalid lighting can be corrected without advancing scene playback a second time", async () => {
  const d = deviceSpy(),
    pose = { version: 0, disposed: false },
    scene = await createGpuAnimationScene(d, pose, [drawable()]);
  scene.update(0.5);
  const bad = frame();
  delete bad.lighting;
  assert.throws(() => scene.render(bad), { code: "ANIMATION_RENDER_LIGHT" });
  assert.equal(scene.controller.time, 0.5);
  assert.equal(d.passes.length, 0);
  scene.render({
    ...frame(),
    draws: [
      { mesh: scene.draws[0], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 1, 0] },
    ],
  });
  assert.equal(d.passes[0].uniforms[23], 0);
  assert.equal(d.passes[0].uniforms[61], 1);
  assert.equal(scene.controller.time, 0.5);
  scene.dispose();
});

test("oversized or shared auxiliary arrays fail before any device allocation", async () => {
  for (const modify of [
    (a) => {
      a.texCoords = new Float32Array(1000);
    },
    (a) => {
      a.vertexColors = new Float32Array(new SharedArrayBuffer(36));
    },
  ]) {
    const d = deviceSpy(),
      item = drawable();
    modify(item);
    await assert.rejects(
      createGpuAnimationScene(d, { version: 0, disposed: false }, [item], { maxBytes: 1004 }),
    );
    assert.equal(d.buffers.length, 0);
  }
});

const mapFields = [
  "baseColorTexture",
  "metallicRoughnessTexture",
  "normalTexture",
  "emissiveTexture",
];
function mappedDrawable() {
  const item = drawable();
  item.geometry.tangents = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1];
  item.normalScale = 0.25;
  for (const field of mapFields) item[field] = { view: { name: field }, sampler: { name: field } };
  return item;
}

test("all four maps survive scene snapshotting and share the same bounded surface storage", async () => {
  const d = deviceSpy(),
    pose = { version: 0, disposed: false },
    a = mappedDrawable(),
    b = mappedDrawable();
  const views = mapFields.map((key) => a[key].view),
    samplers = mapFields.map((key) => a[key].sampler);
  const pending = createGpuAnimationScene(d, pose, [a, b], { maxBytes: 1460 });
  for (const field of mapFields) {
    a[field].view = { replaced: true };
    a[field].sampler = { replaced: true };
  }
  a.normalScale = 99;
  a.texCoords.fill(99);
  const scene = await pending,
    bufferCount = d.buffers.length;
  assert.equal(scene.bufferBytes, 1456);
  for (let i = 0; i < 60; i++) {
    scene.update(1 / 60);
    scene.render(frame());
  }
  await scene.whenIdle();
  const draw = d.passes.at(-2),
    group = draw.groups.get(1).value;
  assert.deepEqual(
    group.entries.filter((e) => e.binding % 2 === 1).map((e) => e.resource),
    views,
  );
  assert.deepEqual(
    group.entries.filter((e) => e.binding % 2 === 0).map((e) => e.resource),
    samplers,
  );
  assert.equal(draw.uniforms[27], 0.25);
  assert.equal(draw.uniforms[31], 1);
  assert.ok(
    draw.pipeline.vertex.buffers[0].attributes.some(
      (a) => a.shaderLocation === 2 && a.offset === 24,
    ),
  );
  assert.equal(d.buffers.length, bufferCount);
  assert.equal(scene.poseVersion, 60);
  assert.equal(d.buffers.filter((b) => b.size === 544).length, 1);
  scene.dispose();
  assert.equal(pose.disposed, false);
  assert.equal(scene.bufferBytes, 0);
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.ok(views.every((v) => v.destroyed === undefined));
});

test("normal and emissive maps without a base-color map retain budget and draw override semantics", async () => {
  const d = deviceSpy(),
    pose = { version: 0, disposed: false },
    item = mappedDrawable();
  delete item.baseColorTexture;
  delete item.metallicRoughnessTexture;
  delete item.metallicFactor;
  delete item.roughnessFactor;
  item.shading = "lambert";
  const scene = await createGpuAnimationScene(d, pose, [item], { maxBytes: 1004 });
  assert.equal(scene.bufferBytes, 1000);
  scene.update(0.5);
  const submitted = d.passes.length;
  assert.throws(
    () => scene.render({ ...frame(), draws: [{ mesh: scene.draws[0], normalScale: NaN }] }),
    { code: "ANIMATION_RENDER_VALUE" },
  );
  assert.equal(d.passes.length, submitted);
  assert.equal(scene.controller.time, 0.5);
  assert.equal(scene.failed, false);
  scene.render({ ...frame(), draws: [{ mesh: scene.draws[0], normalScale: 0 }] });
  await scene.whenIdle();
  assert.equal(d.passes.at(-1).uniforms[27], 0);
  assert.deepEqual(
    d.passes
      .at(-1)
      .groups.get(1)
      .value.entries.map((e) => e.binding),
    [4, 5, 6, 7],
  );
  scene.dispose();
});

test("new map descriptors preserve unsupported fields instead of silently dropping requested behavior", async () => {
  for (const field of mapFields.slice(1)) {
    const d = deviceSpy(),
      item = mappedDrawable();
    item[field].texCoord = 1;
    await assert.rejects(createGpuAnimationScene(d, { version: 0, disposed: false }, [item]), {
      code: "ANIMATION_RENDER_OPTIONS",
    });
    assert.ok(d.buffers.every((b) => b.destroyed));
    assert.equal(item[field].view.destroyed, undefined);
  }
});

test("a later normal map without source tangents unwinds earlier mapped meshes", async () => {
  const d = deviceSpy(),
    pose = { version: 0, disposed: false },
    first = mappedDrawable(),
    second = mappedDrawable();
  delete second.geometry.tangents;
  await assert.rejects(createGpuAnimationScene(d, pose, [first, second]), {
    code: "ANIMATION_RENDER_NORMAL",
  });
  assert.ok(d.buffers.every((b) => b.destroyed));
  assert.equal(pose.disposed, false);
  assert.ok(mapFields.every((field) => first[field].view.destroyed === undefined));
});

test("map capacity and aggregate scene limits fail without leaking partially built children", async () => {
  for (const limited of ["maxSamplersPerShaderStage", "maxSampledTexturesPerShaderStage"]) {
    const d = deviceSpy();
    d.limits[limited] = 3;
    await assert.rejects(
      createGpuAnimationScene(d, { version: 0, disposed: false }, [mappedDrawable()]),
      { code: "ANIMATION_RENDER_LIMIT" },
    );
    assert.ok(d.buffers.every((b) => b.destroyed));
  }
  const d = deviceSpy();
  await assert.rejects(
    createGpuAnimationScene(d, { version: 0, disposed: false }, [mappedDrawable()], {
      maxBytes: 999,
    }),
    { code: "ANIMATION_GPU_LIMIT" },
  );
  assert.equal(d.buffers.length, 1);
  assert.ok(d.buffers.every((b) => b.destroyed));
});
