/** GPU-resident, bounded HDR bloom for the explicit animation rendering route.
 * Linear scene -> bright extraction -> multiscale separable blur -> linear HDR.
 * No scene traversal, frame clock, CPU pixel readback, tone mapping or transfer.
 * Device/source/target remain borrowed. This is not UnrealBloomPass parity.
 */
export class AnimationBloomError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationBloomError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationBloomError(`ANIMATION_BLOOM_${code}`, message);
};
const object = (value, label) => {
  if (!value || typeof value !== "object" || Array.isArray(value))
    fail("OPTIONS", `Expected ${label}`);
  return value;
};
const integer = (value, min, max, label) => {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail("LIMIT", `Invalid ${label}`);
  return value;
};
const scalar = (value, max, label) => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > max)
    fail("VALUE", `Invalid ${label}`);
  return value;
};
const keys = (value, allowed) => {
  for (const key of Object.keys(value))
    if (!allowed.includes(key)) fail("OPTIONS", `Unknown option: ${key}`);
};
const SETTINGS = ["threshold", "softKnee", "strength"];
const DEFAULTS = Object.freeze({ threshold: 1, softKnee: 0.5, strength: 1 });
export function animationBloomSettings(input = {}, defaults = DEFAULTS) {
  object(input, "bloom settings");
  return Object.freeze({
    threshold: scalar(input.threshold ?? defaults.threshold, 65504, "threshold"),
    softKnee: scalar(input.softKnee ?? defaults.softKnee, 1, "softKnee"),
    strength: scalar(input.strength ?? defaults.strength, 64, "strength"),
  });
}

/** Exact owned texture payload accounting (two rgba16float images per level).
 * Ceil-halving includes odd border pixels; stop after reaching 1x1.
 * Does not include driver overhead, borrowed textures or JS/descriptor storage.
 */
export function animationBloomPlan(
  width,
  height,
  { levels = 5, maxBytes = 128 * 1024 * 1024 } = {},
) {
  integer(width, 1, 32768, "width");
  integer(height, 1, 32768, "height");
  integer(levels, 1, 6, "levels");
  integer(maxBytes, 16, Number.MAX_SAFE_INTEGER, "maxBytes");
  const sizes = [];
  let textureBytes = 0;
  for (let level = 0; level < levels; level++) {
    width = Math.ceil(width / 2);
    height = Math.ceil(height / 2);
    textureBytes += width * height * 16;
    if (textureBytes + 16 > maxBytes) fail("LIMIT", "Bloom texture payload exceeds maxBytes");
    sizes.push(Object.freeze({ width, height }));
    if (width === 1 && height === 1) break;
  }
  return Object.freeze({
    sizes: Object.freeze(sizes),
    textureBytes,
    allocatedBytes: textureBytes + 16,
  });
}

const VERTEX = /* wgsl */ `
@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
`;
const INFO = "struct BloomInfo { threshold: f32, knee: f32, strength: f32, levels: u32 }";
const LOAD = /* wgsl */ `
fn pixel(image: texture_2d<f32>, p: vec2<i32>) -> vec4<f32> {
  return textureLoad(image, clamp(p, vec2<i32>(0), vec2<i32>(textureDimensions(image)) - vec2<i32>(1)), 0);
}
`;
export const ANIMATION_BLOOM_FILTER_WGSL = /* wgsl */ `
${INFO}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> info: BloomInfo;
${VERTEX}
${LOAD}
fn bright(color: vec3<f32>) -> vec3<f32> {
  let c = max(color, vec3<f32>(0.0));
  let luminance = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  var soft = 0.0;
  if (info.knee > 0.0) {
    let x = clamp(luminance - info.threshold + info.knee, 0.0, 2.0 * info.knee);
    soft = x * x / (4.0 * info.knee);
  }
  return c * (max(luminance - info.threshold, soft) / max(luminance, 1e-5));
}
@fragment fn extract_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
  let q = vec2<i32>(p.xy) * 2;
  // Extract before downsampling so a bright subpixel is not thresholded away.
  let c = bright(pixel(source, q).rgb) + bright(pixel(source, q + vec2<i32>(1,0)).rgb)
        + bright(pixel(source, q + vec2<i32>(0,1)).rgb) + bright(pixel(source, q + vec2<i32>(1,1)).rgb);
  return vec4<f32>(c * 0.25, 1.0);
}
@fragment fn downsample_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
  let q = vec2<i32>(p.xy) * 2;
  let c = pixel(source, q).rgb + pixel(source, q + vec2<i32>(1,0)).rgb
        + pixel(source, q + vec2<i32>(0,1)).rgb + pixel(source, q + vec2<i32>(1,1)).rgb;
  return vec4<f32>(c * 0.25, 1.0);
}
fn blurred(p: vec2<i32>, axis: vec2<i32>) -> vec4<f32> {
  let c = pixel(source, p - 2 * axis).rgb + 4.0 * pixel(source, p - axis).rgb
        + 6.0 * pixel(source, p).rgb + 4.0 * pixel(source, p + axis).rgb + pixel(source, p + 2 * axis).rgb;
  return vec4<f32>(c * 0.0625, 1.0);
}
@fragment fn horizontal_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
  return blurred(vec2<i32>(p.xy), vec2<i32>(1,0));
}
@fragment fn vertical_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
  return blurred(vec2<i32>(p.xy), vec2<i32>(0,1));
}
`;
export function animationBloomCompositeWgsl(levels) {
  integer(levels, 1, 6, "levels");
  const bindings = Array.from(
    { length: levels },
    (_, i) => `@group(0) @binding(${i + 2}) var bloom${i}: texture_2d<f32>;`,
  ).join("\n");
  const samples = Array.from(
    { length: levels },
    (_, i) => `if (info.levels > ${i}u) { glow += bilinear(bloom${i}, uv); }`,
  ).join("\n  ");
  return /* wgsl */ `
${INFO}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> info: BloomInfo;
${bindings}
${VERTEX}
${LOAD}
fn bilinear(image: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
  let q = uv * vec2<f32>(textureDimensions(image)) - vec2<f32>(0.5);
  let lo = vec2<i32>(floor(q)); let f = fract(q);
  return mix(mix(pixel(image, lo).rgb, pixel(image, lo + vec2<i32>(1,0)).rgb, f.x),
             mix(pixel(image, lo + vec2<i32>(0,1)).rgb, pixel(image, lo + vec2<i32>(1,1)).rgb, f.x), f.y);
}
@fragment fn composite_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
  let scene = textureLoad(source, vec2<i32>(p.xy), 0);
  if (info.strength == 0.0) { return scene; }
  let uv = p.xy / vec2<f32>(textureDimensions(source));
  var glow = vec3<f32>(0.0);
  ${samples}
  let color = max(scene.rgb, vec3<f32>(0.0)) + glow * (info.strength / f32(info.levels));
  // Explicit half-float storage boundary, not a display tone operator.
  return vec4<f32>(min(color, vec3<f32>(65504.0)), scene.a);
}
`;
}

/** Reusable bloom stages with one queue submission per render().
 * Source: single-sample linear rgba16float/rgba8unorm/bgra8unorm with TEXTURE_BINDING.
 * Target: distinct, equally sized rgba16float with RENDER_ATTACHMENT.
 * Input radiance must be finite. Alpha is preserved, not spread with the glow;
 * opaque scenes are the supported presentation profile. No implicit sRGB decode.
 * maxBytes bounds texture payload + uniform, including old/new overlap on resize.
 * whenIdle() observes both scoped validation and completion of submitted work.
 */
export async function createGpuAnimationBloom(device, options = {}) {
  object(options, "bloom options");
  keys(options, [...SETTINGS, "levels", "maxBytes", "signal"]);
  const levels = integer(options.levels ?? 5, 1, 6, "levels");
  const maxBytes = integer(
    options.maxBytes ?? 128 * 1024 * 1024,
    16,
    Number.MAX_SAFE_INTEGER,
    "maxBytes",
  );
  const defaults = animationBloomSettings(options),
    signal = options.signal;
  if (
    signal !== undefined &&
    (!signal ||
      typeof signal.addEventListener !== "function" ||
      typeof signal.removeEventListener !== "function")
  )
    fail("OPTIONS", "Expected an AbortSignal");
  const aborted = () => {
    if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
  };
  aborted();
  for (const name of [
    "createTexture",
    "createBuffer",
    "createShaderModule",
    "createBindGroupLayout",
    "createPipelineLayout",
    "createRenderPipelineAsync",
    "createBindGroup",
    "createCommandEncoder",
    "pushErrorScope",
    "popErrorScope",
  ]) {
    if (typeof device?.[name] !== "function") fail("DEVICE", `Missing WebGPU ${name}`);
  }
  for (const name of ["writeBuffer", "submit", "onSubmittedWorkDone"])
    if (typeof device.queue?.[name] !== "function") fail("DEVICE", `Missing WebGPU queue.${name}`);
  if (device.limits?.maxSampledTexturesPerShaderStage < levels + 1)
    fail("DEVICE", "Insufficient sampled texture bindings");
  let uniform,
    pack,
    filterLayout,
    compositeLayout,
    pipelines,
    disposed = false,
    terminal = null,
    busy = false;
  let validation = Promise.resolve(),
    version = 0,
    lastRender = null;
  let rejectStop;
  const stopped = new Promise((_, reject) => {
    rejectStop = reject;
  });
  stopped.catch(() => {});
  const destroyPack = (value) => {
    for (const texture of value?.textures ?? []) texture.destroy();
  };
  function release() {
    destroyPack(pack);
    pack = undefined;
    uniform?.destroy();
    uniform = undefined;
    pipelines = undefined;
  }
  function stop(error) {
    if (!disposed && !terminal) {
      terminal = error;
      release();
      rejectStop(error);
    }
  }
  function live() {
    if (terminal) throw terminal;
    if (disposed) fail("DISPOSED", "Bloom is disposed");
  }
  function checked(operation) {
    live();
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    let value, error;
    try {
      value = operation();
    } catch (cause) {
      error = cause;
    }
    const invalid = device.popErrorScope(),
      oom = device.popErrorScope();
    validation = Promise.all([validation, invalid, oom]).then(([, a, b]) => {
      if (error) throw error;
      if (a || b) fail("GPU", (a || b).message ?? "GPU bloom failure");
    });
    validation.catch(stop);
    if (error) {
      stop(error);
      throw error;
    }
    return value;
  }
  if (device.lost && typeof device.lost.then === "function")
    device.lost.then(
      (info) =>
        stop(
          new AnimationBloomError("ANIMATION_BLOOM_DEVICE_LOST", info?.message ?? "Device lost"),
        ),
      stop,
    );
  const onAbort = () => stop(signal.reason ?? new DOMException("Aborted", "AbortError"));
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const compiled = checked(() => {
      const textureEntry = (binding) => ({
        binding,
        visibility: 2,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      });
      const base = [
        textureEntry(0),
        { binding: 1, visibility: 2, buffer: { type: "uniform", minBindingSize: 16 } },
      ];
      filterLayout = device.createBindGroupLayout({ entries: base });
      compositeLayout = device.createBindGroupLayout({
        entries: [...base, ...Array.from({ length: levels }, (_, i) => textureEntry(i + 2))],
      });
      const module = device.createShaderModule({
        label: "Animation bloom filters",
        code: ANIMATION_BLOOM_FILTER_WGSL,
      });
      const composite = device.createShaderModule({
        label: "Animation bloom composite",
        code: animationBloomCompositeWgsl(levels),
      });
      const make = (module, entryPoint, layout) =>
        device.createRenderPipelineAsync({
          label: `Animation bloom ${entryPoint}`,
          layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          vertex: { module, entryPoint: "vertex_main" },
          fragment: { module, entryPoint, targets: [{ format: "rgba16float" }] },
          primitive: { topology: "triangle-list" },
        });
      return Promise.all(
        ["extract_main", "downsample_main", "horizontal_main", "vertical_main"]
          .map((entry) => make(module, entry, filterLayout))
          .concat(make(composite, "composite_main", compositeLayout)),
      );
    });
    pipelines = await Promise.race([compiled, stopped]);
    await Promise.race([validation, stopped]);
    live();
    aborted();
    checked(() => {
      uniform = device.createBuffer({
        label: "Animation bloom parameters",
        size: 16,
        usage: 8 | 64,
      });
    });
    await Promise.race([validation, stopped]);
    live();
    aborted();
  } catch (error) {
    stop(error);
    release();
    throw error;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }

  const view = (texture) =>
    texture.createView({
      dimension: "2d",
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 1,
    });
  const group = (layout, views) =>
    device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: views[0] },
        { binding: 1, resource: { buffer: uniform, size: 16 } },
        ...views.slice(1).map((resource, i) => ({ binding: i + 2, resource })),
      ],
    });
  function allocate(width, height, plan) {
    const next = {
      width,
      height,
      bytes: plan.textureBytes,
      textures: [],
      stages: [],
      source: null,
      extract: null,
      composite: null,
    };
    try {
      for (const size of plan.sizes) {
        const pair = [];
        for (let i = 0; i < 2; i++) {
          const texture = device.createTexture({
            label: `Animation bloom ${size.width}x${size.height}`,
            size,
            format: "rgba16float",
            usage: 4 | 16,
          });
          next.textures.push(texture);
          pair.push(view(texture));
        }
        const previous = next.stages.at(-1);
        next.stages.push({
          a: pair[0],
          b: pair[1],
          horizontal: group(filterLayout, [pair[0]]),
          vertical: group(filterLayout, [pair[1]]),
          downsample: previous ? group(filterLayout, [previous.a]) : null,
        });
      }
      return next;
    } catch (error) {
      destroyPack(next);
      throw error;
    }
  }
  function texture(value, label, usage) {
    object(value, label);
    const limit = Math.min(device.limits?.maxTextureDimension2D ?? 8192, 32768);
    const width = integer(value.width, 1, limit, `${label} width`),
      height = integer(value.height, 1, limit, `${label} height`);
    if (
      value.dimension !== "2d" ||
      value.depthOrArrayLayers !== 1 ||
      value.sampleCount !== 1 ||
      !Number.isInteger(value.usage) ||
      (value.usage & usage) !== usage ||
      typeof value.createView !== "function"
    )
      fail("TEXTURE", `Invalid ${label} storage, usage or samples`);
    return { value, width, height, format: value.format };
  }
  function pass(encoder, target, pipeline, bindings) {
    const p = encoder.beginRenderPass({
      label: "Animation bloom",
      colorAttachments: [
        { view: target, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      ],
    });
    p.setPipeline(pipeline);
    p.setBindGroup(0, bindings);
    p.draw(3);
    p.end();
  }
  const result = Object.freeze({
    defaults,
    levels,
    get version() {
      return version;
    },
    get lastRender() {
      return lastRender;
    },
    get allocatedBytes() {
      return uniform ? 16 + (pack?.bytes ?? 0) : 0;
    },
    get disposed() {
      return disposed;
    },
    get failed() {
      return terminal !== null;
    },
    render(frame) {
      live();
      if (busy) fail("REENTRANT", "Bloom operation cannot be reentered");
      busy = true;
      try {
        object(frame, "bloom frame");
        keys(frame, ["source", "target", ...SETTINGS]);
        const source = texture(frame.source, "source", 4),
          target = texture(frame.target, "target", 16);
        const settings = animationBloomSettings(frame, defaults);
        if (
          !["rgba16float", "rgba8unorm", "bgra8unorm"].includes(source.format) ||
          target.format !== "rgba16float"
        )
          fail("FORMAT", "Bloom requires linear source and rgba16float target");
        if (
          source.value === target.value ||
          source.width !== target.width ||
          source.height !== target.height
        )
          fail("TEXTURE", "Source/target must be distinct and equally sized");
        const enabled = settings.strength > 0;
        const resize =
          enabled && (!pack || pack.width !== source.width || pack.height !== source.height);
        const plan = resize
          ? animationBloomPlan(source.width, source.height, { levels, maxBytes })
          : null;
        if (resize && plan.allocatedBytes + (pack?.bytes ?? 0) > maxBytes)
          fail("LIMIT", "Bloom resize old/new overlap exceeds maxBytes");
        checked(() => {
          if (resize) {
            const next = allocate(source.width, source.height, plan);
            destroyPack(pack);
            pack = next;
          }
          let composite;
          if (enabled) {
            if (pack.source !== source.value) {
              const sourceView = view(source.value);
              const extract = group(filterLayout, [sourceView]);
              composite = group(compositeLayout, [
                sourceView,
                ...Array.from(
                  { length: levels },
                  (_, i) => pack.stages[Math.min(i, pack.stages.length - 1)].a,
                ),
              ]);
              pack.source = source.value;
              pack.extract = extract;
              pack.composite = composite;
            } else composite = pack.composite;
          } else {
            const sourceView = view(source.value);
            composite = group(compositeLayout, Array(levels + 1).fill(sourceView));
          }
          const bytes = new ArrayBuffer(16),
            data = new DataView(bytes);
          data.setFloat32(0, settings.threshold, true);
          data.setFloat32(4, settings.threshold * settings.softKnee, true);
          data.setFloat32(8, settings.strength, true);
          data.setUint32(12, enabled ? pack.stages.length : 0, true);
          device.queue.writeBuffer(uniform, 0, bytes);
          const encoder = device.createCommandEncoder({ label: "Animation bloom frame" });
          if (enabled)
            for (let i = 0; i < pack.stages.length; i++) {
              const stage = pack.stages[i];
              pass(
                encoder,
                stage.a,
                pipelines[i === 0 ? 0 : 1],
                i === 0 ? pack.extract : stage.downsample,
              );
              pass(encoder, stage.b, pipelines[2], stage.horizontal);
              pass(encoder, stage.a, pipelines[3], stage.vertical);
            }
          pass(encoder, view(target.value), pipelines[4], composite);
          device.queue.submit([encoder.finish()]);
        });
        version++;
        lastRender = Object.freeze({
          version,
          width: source.width,
          height: source.height,
          levels: enabled ? pack.stages.length : 0,
          passes: enabled ? 3 * pack.stages.length + 1 : 1,
          settings,
        });
        return result;
      } finally {
        busy = false;
      }
    },
    async whenIdle() {
      live();
      if (busy) fail("REENTRANT", "Cannot await during a bloom operation");
      try {
        await Promise.race([
          Promise.all([validation, device.queue.onSubmittedWorkDone()]),
          stopped,
        ]);
        live();
        return result;
      } catch (error) {
        stop(error);
        throw error;
      }
    },
    dispose() {
      if (busy) fail("REENTRANT", "Cannot dispose during a bloom operation");
      if (!disposed) {
        disposed = true;
        release();
        rejectStop(new AnimationBloomError("ANIMATION_BLOOM_DISPOSED", "Bloom is disposed"));
      }
    },
  });
  return result;
}
