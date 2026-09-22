/** Compose the existing explicit scene/output routes with GPU-resident bloom.
 * Opaque, linear-sRGB scene profile; no Three.js EffectComposer emulation.
 * update(), cameras, device, final target and scene lifetime remain caller-owned.
 */
import {
  animationBloomPlan,
  animationBloomSettings,
  createGpuAnimationBloom,
} from "./animation_bloom.mjs";
import { animationOutputSettings, createGpuAnimationOutput } from "./animation_output.mjs";

export class AnimationPostprocessError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationPostprocessError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationPostprocessError(`ANIMATION_POSTPROCESS_${code}`, message);
};
const object = (value, label) => {
  if (!value || typeof value !== "object" || Array.isArray(value))
    fail("OPTIONS", `Expected ${label}`);
  return value;
};
const keys = (value, allowed) => {
  for (const key of Object.keys(value))
    if (!allowed.includes(key)) fail("OPTIONS", `Unknown option: ${key}`);
};
const BLOOM_KEYS = ["threshold", "softKnee", "strength"];
const OUTPUT_KEYS = ["toneMapping", "exposure"];

/**
 * render({source,target,bloom?,output?}) consumes an existing opaque linear image.
 * renderScene(scene,{target,frame?,camera?,bloom?,output?}) also owns the HDR/depth
 * attachments; configure the borrowed renderer as rgba16float, sampleCount:1,
 * depthFormat matching this object's depthFormat (default depth32float).
 * With camera, call the borrowed model's renderCamera; otherwise call render.
 * No pose/controller update is performed implicitly. Settings are per-call,
 * not persistent mutations. maxBytes covers this composition's owned payload,
 * including child uniforms/pyramids and conservative old/new resize overlap.
 */
export async function createGpuAnimationPostprocessor(device, options = {}) {
  object(options, "postprocessor options");
  keys(options, ["bloom", "output", "depthFormat", "maxBytes", "signal"]);
  const bloomOptions = { ...object(options.bloom ?? {}, "bloom options") };
  const outputOptions = { ...object(options.output ?? {}, "output options") };
  keys(bloomOptions, [...BLOOM_KEYS, "levels"]);
  keys(outputOptions, [...OUTPUT_KEYS, "format", "outputColorSpace"]);
  const signal = options.signal;
  const maxBytes = options.maxBytes ?? 256 * 1024 * 1024,
    depthFormat = options.depthFormat === undefined ? "depth32float" : options.depthFormat;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 32)
    fail("LIMIT", "maxBytes must include both 16-byte uniforms");
  if (depthFormat !== null && depthFormat !== "depth32float")
    fail("FORMAT", "Managed depth is depth32float or null");
  let bloom,
    output,
    targets,
    disposed = false,
    terminal = null,
    busy = false,
    version = 0,
    lastRender = null;
  let bloomWidth = 0,
    bloomHeight = 0,
    validation = Promise.resolve(),
    completion = Promise.resolve();
  let rejectStop;
  const stopped = new Promise((_, reject) => {
    rejectStop = reject;
  });
  stopped.catch(() => {});
  const destroyTargets = (value) => {
    value?.color?.destroy();
    value?.depth?.destroy();
    value?.intermediate?.destroy();
  };
  function release() {
    destroyTargets(targets);
    targets = undefined;
    bloom?.dispose();
    output?.dispose();
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
    if (disposed) fail("DISPOSED", "Postprocessor is disposed");
    if (bloom?.failed || output?.failed) {
      const error = new AnimationPostprocessError(
        "ANIMATION_POSTPROCESS_GPU",
        "A child pass failed",
      );
      stop(error);
      throw error;
    }
  }
  try {
    output = await createGpuAnimationOutput(device, {
      toneMapping: "aces-filmic",
      ...outputOptions,
      inputAlpha: "straight",
      outputAlpha: "opaque",
      signal,
    });
    bloom = await createGpuAnimationBloom(device, {
      ...bloomOptions,
      maxBytes: maxBytes - 16,
      signal,
    });
    live();
  } catch (error) {
    release();
    throw error;
  }
  if (device.lost && typeof device.lost.then === "function")
    device.lost.then(
      (info) =>
        stop(
          new AnimationPostprocessError(
            "ANIMATION_POSTPROCESS_DEVICE_LOST",
            info?.message ?? "Device lost",
          ),
        ),
      stop,
    );

  function checked(operation) {
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
      if (a || b) fail("GPU", (a || b).message ?? "Postprocessor allocation failed");
    });
    validation.catch(stop);
    if (error) {
      stop(error);
      throw error;
    }
    return value;
  }
  function texture(value, label, usage) {
    object(value, label);
    const width = value.width,
      height = value.height,
      format = value.format;
    const limit = Math.min(device.limits?.maxTextureDimension2D ?? 8192, 32768);
    if (
      ![width, height].every((v) => Number.isSafeInteger(v) && v > 0 && v <= limit) ||
      value.dimension !== "2d" ||
      value.depthOrArrayLayers !== 1 ||
      value.sampleCount !== 1 ||
      !Number.isInteger(value.usage) ||
      (value.usage & usage) !== usage ||
      typeof value.createView !== "function"
    )
      fail("TEXTURE", `Invalid ${label} storage or usage`);
    return { value, width, height, format };
  }
  function prepare(input, managed) {
    object(input, "postprocessing frame");
    keys(
      input,
      managed
        ? ["target", "frame", "camera", "bloom", "output"]
        : ["source", "target", "bloom", "output"],
    );
    const target = texture(input.target, "target", 16);
    if (
      target.format !== output.format &&
      !(output.format.endsWith("-srgb") && target.format === output.format.slice(0, -5))
    )
      fail("FORMAT", "Target format differs from output pipeline");
    const b = { ...object(input.bloom ?? {}, "frame bloom settings") },
      o = { ...object(input.output ?? {}, "frame output settings") };
    keys(b, BLOOM_KEYS);
    keys(o, OUTPUT_KEYS);
    const bloomSettings = animationBloomSettings(b, bloom.defaults),
      outputSettings = animationOutputSettings(o, output.defaults);
    let source;
    if (!managed) {
      source = texture(input.source, "source", 4);
      if (!["rgba16float", "rgba8unorm", "bgra8unorm"].includes(source.format))
        fail("FORMAT", "Source must be linear radiance");
      if (
        source.value === target.value ||
        source.width !== target.width ||
        source.height !== target.height
      )
        fail("TEXTURE", "Source and target must be distinct with matching extents");
    }
    const enabled = bloomSettings.strength > 0,
      width = target.width,
      height = target.height;
    const needsTargets = managed || enabled;
    const resizeTargets =
      needsTargets &&
      (!targets ||
        targets.width !== width ||
        targets.height !== height ||
        (managed && !targets.color) ||
        (enabled && !targets.intermediate));
    const resizeBloom = enabled && (bloomWidth !== width || bloomHeight !== height);
    const plan = resizeBloom
      ? animationBloomPlan(width, height, { levels: bloom.levels, maxBytes })
      : null;
    // Preserve already-owned attachments at the same extent. Replacing them is
    // atomic with respect to synchronous allocation failures, not GPU execution.
    const sameSize = targets?.width === width && targets?.height === height;
    const wantColor = managed || (sameSize && Boolean(targets.color)),
      wantIntermediate = enabled || (sameSize && Boolean(targets.intermediate));
    const newBytes = resizeTargets
      ? width * height * ((wantColor ? 8 + (depthFormat ? 4 : 0) : 0) + (wantIntermediate ? 8 : 0))
      : 0;
    const existing = (targets?.bytes ?? 0) + bloom.allocatedBytes + output.allocatedBytes;
    if (existing + newBytes + (plan?.textureBytes ?? 0) > maxBytes)
      fail("LIMIT", "Postprocessing payload/resize overlap exceeds maxBytes");
    if (version === Number.MAX_SAFE_INTEGER) fail("VERSION", "Submission counter exhausted");
    return {
      target,
      source,
      bloomSettings,
      outputSettings,
      enabled,
      width,
      height,
      resizeTargets,
      wantColor,
      wantIntermediate,
      newBytes,
    };
  }
  function allocate(prepared) {
    if (!prepared.resizeTargets) return;
    const { width, height, newBytes, wantColor, wantIntermediate } = prepared;
    checked(() => {
      const next = { width, height, bytes: newBytes };
      try {
        const make = (format, usage, label) =>
          device.createTexture({ label, size: { width, height }, format, usage });
        if (wantColor) {
          next.color = make("rgba16float", 4 | 16, "Postprocess linear scene");
          if (depthFormat) next.depth = make(depthFormat, 16, "Postprocess scene depth");
        }
        if (wantIntermediate)
          next.intermediate = make("rgba16float", 4 | 16, "Postprocess bloom output");
      } catch (error) {
        destroyTargets(next);
        throw error;
      }
      destroyTargets(targets);
      targets = next;
    });
  }
  function submit(prepared, source, scene) {
    if (prepared.enabled) {
      bloom.render({ source, target: targets.intermediate, ...prepared.bloomSettings });
      bloomWidth = prepared.width;
      bloomHeight = prepared.height;
      source = targets.intermediate;
    }
    output.render({ source, target: prepared.target.value, ...prepared.outputSettings });
    // Observe native failure even if the application does not explicitly await.
    // These are completion observers, not an additional frame loop or scheduler.
    completion = Promise.all([
      completion,
      validation,
      bloom.whenIdle(),
      output.whenIdle(),
      scene?.whenIdle(),
    ]).then(() => undefined);
    completion.catch(stop);
    version++;
    lastRender = Object.freeze({
      version,
      width: prepared.width,
      height: prepared.height,
      bloom: prepared.enabled ? bloom.lastRender : null,
      output: Object.freeze(prepared.outputSettings),
      managedScene: Boolean(scene),
      postprocessSubmissions: prepared.enabled ? 2 : 1,
    });
  }
  function exclusive(operation) {
    live();
    if (busy) fail("REENTRANT", "Postprocessing cannot be reentered");
    busy = true;
    try {
      operation();
      return result;
    } catch (error) {
      if (bloom.failed || output.failed) stop(error);
      throw error;
    } finally {
      busy = false;
    }
  }
  const result = Object.freeze({
    format: output.format,
    outputColorSpace: output.outputColorSpace,
    depthFormat,
    get version() {
      return version;
    },
    get lastRender() {
      return lastRender;
    },
    get allocatedBytes() {
      return (targets?.bytes ?? 0) + bloom.allocatedBytes + output.allocatedBytes;
    },
    get disposed() {
      return disposed;
    },
    get failed() {
      return terminal !== null || bloom.failed || output.failed;
    },
    render(input) {
      return exclusive(() => {
        const p = prepare(input, false);
        allocate(p);
        submit(p, p.source.value);
      });
    },
    renderScene(scene, input) {
      return exclusive(() => {
        const p = prepare(input, true);
        object(scene, "borrowed scene");
        const frame = { ...object(input.frame ?? {}, "scene frame") },
          camera = input.camera;
        for (const key of ["colorView", "depthView", "resolveTarget"])
          if (Object.hasOwn(frame, key))
            fail("ATTACHMENT", "renderScene owns the HDR/depth attachments");
        if (
          (frame.loadOp !== undefined && frame.loadOp !== "clear") ||
          (frame.depthLoadOp !== undefined && frame.depthLoadOp !== "clear")
        )
          fail("ATTACHMENT", "Managed frames clear attachments; accumulation is not supported");
        const clearColor = Array.from(frame.clearColor ?? [0, 0, 0, 1]);
        if (
          clearColor.length !== 4 ||
          clearColor.some((v) => typeof v !== "number" || !Number.isFinite(v)) ||
          clearColor[3] !== 1
        )
          fail("ALPHA", "Managed scene clear color must be finite and opaque");
        const render = camera === undefined ? scene.render : scene.renderCamera;
        if (typeof render !== "function" || typeof scene.whenIdle !== "function")
          fail("SCENE", "Expected synchronous scene render/renderCamera and whenIdle");
        allocate(p);
        const attachments = checked(() => ({
          colorView: targets.color.createView(),
          ...(targets.depth ? { depthView: targets.depth.createView() } : {}),
        }));
        render.call(
          scene,
          { ...frame, ...attachments, clearColor, loadOp: "clear", depthLoadOp: "clear" },
          camera,
        );
        submit(p, targets.color, scene);
      });
    },
    async whenIdle() {
      live();
      if (busy) fail("REENTRANT", "Cannot await inside a postprocess operation");
      try {
        await Promise.race([Promise.all([completion, validation]), stopped]);
        live();
        return result;
      } catch (error) {
        stop(error);
        throw error;
      }
    },
    dispose() {
      if (busy) fail("REENTRANT", "Cannot dispose during a postprocess operation");
      if (!disposed) {
        disposed = true;
        release();
        rejectStop(
          new AnimationPostprocessError(
            "ANIMATION_POSTPROCESS_DISPOSED",
            "Postprocessor is disposed",
          ),
        );
      }
    },
  });
  return result;
}
