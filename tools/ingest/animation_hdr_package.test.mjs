import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

// Execute the production builder unchanged in an isolated toolkit. Only unrelated
// glTF pose/material/scene boundaries are substitutes. The actual HDR decoder,
// loader and complete environment filter are copied, emitted and executed.
async function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-hdr-package-")),
    toolkit = path.join(root, "toolkit");
  fs.mkdirSync(toolkit);
  for (const file of [
    "build_animation.mjs",
    "animation_hdr.mjs",
    "animation_environment_loader.mjs",
    "animation_environment.mjs",
  ])
    fs.copyFileSync(new URL("./" + file, import.meta.url), path.join(toolkit, file));
  fs.writeFileSync(
    path.join(toolkit, "animation_gltf.mjs"),
    "export function decodeGltfAnimation(model){return model;}",
  );
  fs.writeFileSync(
    path.join(toolkit, "animation_runtime.mjs"),
    `export class AnimationPoseError extends Error {constructor(code,message){super(message);this.code=code;}}
export function createAnimationPlayer(def){return {nodeCount:def.nodes.length,clips:[],instances:[],morphWeights:[],dispose(){}};}`,
  );
  const unused = {
    "animation_controller.mjs": ["createAnimationController"],
    "animation_deformer.mjs": ["createAnimationDeformer"],
    "animation_webgpu.mjs": ["createGpuAnimationDeformer"],
    "animation_render.mjs": ["createGpuAnimationRenderer"],
    "animation_scene.mjs": ["createGpuAnimationScene"],
    "animation_shadow.mjs": ["createGpuAnimationShadowMap"],
    "animation_shadow_view.mjs": ["fitAnimationShadowView", "animationShadowWorldBounds"],
    "animation_draw_order.mjs": [],
    "animation_bounds.mjs": [],
    "animation_shadow_receiver.mjs": [],
    "animation_scene_shadow.mjs": [],
    "animation_environment_receiver.mjs": [],
  };
  for (const [file, names] of Object.entries(unused))
    fs.writeFileSync(
      path.join(toolkit, file),
      names
        .map(
          (name) =>
            `export function ${name}(){throw new Error('Unrelated test boundary must not execute');}`,
        )
        .join("\n"),
    );
  const entry = path.join(root, "asset.gltf");
  fs.writeFileSync(entry, JSON.stringify({ asset: { version: "2.0" }, nodes: [{}] }));
  return {
    root,
    toolkit,
    entry,
    ...(await import(pathToFileURL(path.join(toolkit, "build_animation.mjs")))),
  };
}
function recordingGpu() {
  const textures = [],
    uploads = [];
  let passes = 0,
    submissions = 0,
    scopeDepth = 0;
  const device = {
    limits: {
      maxTextureDimension2D: 4096,
      minUniformBufferOffsetAlignment: 256,
      maxBufferSize: 2 ** 20,
    },
    lost: new Promise(() => {}),
    queue: {
      writeTexture(target, bytes) {
        uploads.push(bytes.slice());
      },
      writeBuffer() {},
      submit() {
        submissions++;
      },
      onSubmittedWorkDone: async () => {},
    },
    pushErrorScope() {
      scopeDepth++;
    },
    popErrorScope() {
      scopeDepth--;
      return Promise.resolve(null);
    },
    createTexture(d) {
      const t = {
        ...d,
        width: d.size[0],
        height: d.size[1],
        depthOrArrayLayers: d.size[2],
        sampleCount: 1,
        destroyed: 0,
        createView(descriptor) {
          return { texture: t, descriptor };
        },
        destroy() {
          t.destroyed++;
        },
      };
      textures.push(t);
      return t;
    },
    createBuffer: (d) => ({ ...d, destroy() {} }),
    createSampler: (d) => d,
    createBindGroupLayout: (d) => d,
    createPipelineLayout: (d) => d,
    createShaderModule: (d) => d,
    createRenderPipelineAsync: async (d) => d,
    createBindGroup: (d) => d,
    createCommandEncoder: () => ({
      beginRenderPass() {
        passes++;
        return { setPipeline() {}, setBindGroup() {}, draw() {}, end() {} };
      },
      finish: () => ({}),
    }),
  };
  return {
    device,
    textures,
    uploads,
    get passes() {
      return passes;
    },
    get submissions() {
      return submissions;
    },
    get scopeDepth() {
      return scopeDepth;
    },
  };
}
const options = { webgpu: true, environment: true, hdr: true };
const hdr = () =>
  new Uint8Array([
    ...new TextEncoder().encode("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n"),
    255,
    128,
    0,
    130,
    0,
    0,
    255,
    128,
  ]);

test("public GPU entry decodes, uploads and filters HDR after relocation without its source toolkit", async () => {
  const f = await fixture(),
    built = f.buildAnimation(f.entry, path.join(f.root, "package"), options);
  for (const file of ["animation_hdr.mjs", "animation_environment_loader.mjs"]) {
    const bytes = fs.readFileSync(path.join(built.outDir, file)),
      artifact = built.artifacts.find((a) => a.file === file);
    assert.deepEqual(bytes, fs.readFileSync(new URL("./" + file, import.meta.url)));
    assert.equal(artifact.bytes, bytes.length);
    assert.equal(artifact.sha256, createHash("sha256").update(bytes).digest("hex"));
  }
  const deployed = path.join(f.root, "deployed");
  fs.cpSync(built.outDir, deployed, { recursive: true });
  fs.renameSync(f.toolkit, f.toolkit + ".unavailable");
  const fetcher = globalThis.fetch;
  let api;
  try {
    globalThis.fetch = () => {
      throw Error("Package import must not fetch");
    };
    api = await import(pathToFileURL(path.join(deployed, built.gpuEntry)));
  } finally {
    globalThis.fetch = fetcher;
  }
  const bytes = hdr(),
    decoded = api.decodeAnimationHdr(bytes),
    g = recordingGpu();
  assert.equal(decoded.data[0], 0x4400);
  assert.equal(decoded.width, 2);
  assert.equal(decoded.height, 1);
  const map = await api.loadGpuAnimationEnvironment(g.device, bytes, {
    size: 4,
    diffuseSize: 2,
    lutSize: 4,
    samples: 8,
  });
  assert.deepEqual(g.uploads, [decoded.data]);
  assert.equal(g.submissions, 1);
  assert.equal(g.passes, 25);
  assert.equal(g.scopeDepth, 0);
  assert.equal(g.textures.length, 4);
  assert.equal(g.textures[0].destroyed, 1);
  assert.equal(map.sample(g.device).profile, "f3d-animation-environment-v1");
  assert.equal(await map.whenIdle(), map);
  assert.equal(map.sourceInfo.inputBytes, bytes.length);
  map.dispose();
  assert.ok(g.textures.every((t) => t.destroyed === 1));
});
test("HDR package modules participate in exact pre-write budgets and invalid routes never create output", async () => {
  const f = await fixture(),
    built = f.buildAnimation(f.entry, path.join(f.root, "size"), options),
    short = path.join(f.root, "short");
  assert.throws(
    () => f.buildAnimation(f.entry, short, { ...options, maxBytes: built.outputBytes - 1 }),
    { code: "GLTF_ANIMATION_LIMIT" },
  );
  assert.equal(fs.existsSync(short), false);
  assert.equal(
    f.buildAnimation(f.entry, path.join(f.root, "exact"), {
      ...options,
      maxBytes: built.outputBytes,
    }).outputBytes,
    built.outputBytes,
  );
  for (const invalid of [
    { hdr: true },
    { webgpu: true, hdr: true },
    { environment: true, hdr: true },
    { ...options, hdr: 1 },
  ]) {
    assert.throws(() => f.buildAnimation(f.entry, short, invalid), TypeError);
    assert.equal(fs.existsSync(short), false);
  }
});
test("omitting HDR preserves CPU, GPU and GPU-IBL output bytes and does not require its modules", async () => {
  const f = await fixture(),
    source = fs.readFileSync(path.join(f.toolkit, "build_animation.mjs"), "utf8");
  const prior = source
    .replace(",hdr=false", "")
    .replace(
      "  if(typeof hdr!=='boolean'||(hdr&&!environment))throw new TypeError('hdr must be boolean and requires environment:true');\n",
      "",
    )
    .replace(/ {2}if\(hdr\) \{[\s\S]*?\n {2}\}\n {2}const manifest/, "  const manifest");
  assert.notEqual(prior, source);
  fs.writeFileSync(path.join(f.toolkit, "prior.mjs"), prior);
  const { buildAnimation: before } = await import(pathToFileURL(path.join(f.toolkit, "prior.mjs")));
  for (const file of ["animation_hdr.mjs", "animation_environment_loader.mjs"])
    fs.renameSync(path.join(f.toolkit, file), path.join(f.toolkit, file + ".unused"));
  for (const [i, profile] of [
    {},
    { webgpu: true },
    { webgpu: true, environment: true },
  ].entries()) {
    const a = before(f.entry, path.join(f.root, "old-" + i), profile),
      b = f.buildAnimation(f.entry, path.join(f.root, "new-" + i), profile);
    assert.equal(a.outputBytes, b.outputBytes);
    assert.deepEqual(a.emittedFiles, b.emittedFiles);
    assert.ok(!b.emittedFiles.includes("animation_hdr.mjs"));
    assert.ok(!b.emittedFiles.includes("animation_environment_loader.mjs"));
    for (const file of a.emittedFiles)
      assert.deepEqual(
        fs.readFileSync(path.join(a.outDir, file)),
        fs.readFileSync(path.join(b.outDir, file)),
      );
  }
});
