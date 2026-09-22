/** Explicit final output pass for linear-sRGB scene textures. No frame loop,
 * scene traversal, texture allocation, or ownership of the device/input/output.
 * Tone operators port the pinned Three.js r186 shader equations (MIT):
 * https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js
 * Copyright three.js authors. This is an opt-in whole-image operation, NOT a
 * substitute for per-material toneMapped semantics or complete renderer parity.
 * Input is a single-sample linear rgba16float/rgba8unorm/bgra8unorm texture.
 * Source and destination extents must match; no implicit resize/filter/flip.
 * Negative radiance is clamped before mapping. Alpha is not tone mapped.
 */
export class AnimationOutputError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationOutputError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationOutputError("ANIMATION_OUTPUT_" + code, message);
};
export const ANIMATION_TONE_MAPPINGS = Object.freeze([
  "none",
  "linear",
  "reinhard",
  "cineon",
  "aces-filmic",
  "agx",
  "neutral",
]);
const formats = [
  "rgba8unorm",
  "bgra8unorm",
  "rgba8unorm-srgb",
  "bgra8unorm-srgb",
  "rgba16float",
  "rgba32float",
];
const sourceFormats = ["rgba8unorm", "bgra8unorm", "rgba16float"];
const alphaModes = ["straight", "premultiplied", "opaque"];
const object = (v, label) => {
  if (!v || typeof v !== "object" || Array.isArray(v)) fail("OPTIONS", `Expected ${label}`);
  return v;
};
const positive = (v, label) => {
  if (!Number.isSafeInteger(v) || v < 1) fail("TEXTURE", `Invalid ${label}`);
  return v;
};
export function animationOutputSettings(input, defaults) {
  const mode = input.toneMapping ?? defaults.toneMapping,
    exposure = input.exposure ?? defaults.exposure;
  const inputAlpha = input.inputAlpha ?? defaults.inputAlpha,
    outputAlpha = input.outputAlpha ?? defaults.outputAlpha;
  if (!ANIMATION_TONE_MAPPINGS.includes(mode))
    fail("TONE_MAPPING", "Unknown tone mapping operator");
  // Bounded f32 exposure keeps the admitted half-float radiance domain within
  // the operators' polynomial arithmetic range, including unpremultiplication.
  if (
    typeof exposure !== "number" ||
    !Number.isFinite(exposure) ||
    exposure < 0 ||
    exposure > 65504
  )
    fail("EXPOSURE", "Exposure must be in [0,65504]");
  if (!["straight", "premultiplied"].includes(inputAlpha) || !alphaModes.includes(outputAlpha))
    fail("ALPHA", "Invalid alpha convention");
  return { toneMapping: mode, exposure, inputAlpha, outputAlpha };
}
// Column-major matrices, like the pinned GLSL constructors. Do not transpose
// these a second time when moving between GLSL, WGSL and scalar test oracles.
export const ANIMATION_OUTPUT_WGSL = /* wgsl */ `
struct OutputInfo { exposure: f32, mode: u32, input_premultiplied: u32, output_alpha: u32 }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> info: OutputInfo;
fn linear_to_srgb(v: vec3<f32>) -> vec3<f32> {
  return select(12.92*v,1.055*pow(max(v,vec3<f32>(0.0)),vec3<f32>(1.0/2.4))-0.055,v>vec3<f32>(0.0031308));
}
fn srgb_to_linear(v: vec3<f32>) -> vec3<f32> {
  return select(v/12.92,pow((max(v,vec3<f32>(0.0))+0.055)/1.055,vec3<f32>(2.4)),v>vec3<f32>(0.04045));
}
fn tone_map(input_color: vec3<f32>) -> vec3<f32> {
  // NoToneMapping, unlike LinearToneMapping, ignores exposure and does not clip.
  if (info.mode == 0u) { return input_color; }
  var color = input_color * info.exposure;
  if (info.mode == 1u) { return clamp(color,vec3<f32>(0.0),vec3<f32>(1.0)); }
  if (info.mode == 2u) { return clamp(color/(vec3<f32>(1.0)+color),vec3<f32>(0.0),vec3<f32>(1.0)); }
  if (info.mode == 3u) {
    color=max(vec3<f32>(0.0),color-0.004);
    return pow((color*(6.2*color+0.5))/(color*(6.2*color+1.7)+0.06),vec3<f32>(2.2));
  }
  if (info.mode == 4u) {
    let input_matrix=mat3x3<f32>(vec3<f32>(0.59719,0.07600,0.02840),vec3<f32>(0.35458,0.90834,0.13383),vec3<f32>(0.04823,0.01566,0.83777));
    let output_matrix=mat3x3<f32>(vec3<f32>(1.60475,-0.10208,-0.00327),vec3<f32>(-0.53108,1.10813,-0.07276),vec3<f32>(-0.07367,-0.00605,1.07602));
    color=input_matrix*(color/0.6);
    color=(color*(color+0.0245786)-0.000090537)/(color*(0.983729*color+0.4329510)+0.238081);
    return clamp(output_matrix*color,vec3<f32>(0.0),vec3<f32>(1.0));
  }
  if (info.mode == 5u) {
    let to_rec2020=mat3x3<f32>(vec3<f32>(0.6274,0.0691,0.0164),vec3<f32>(0.3293,0.9195,0.0880),vec3<f32>(0.0433,0.0113,0.8956));
    let from_rec2020=mat3x3<f32>(vec3<f32>(1.6605,-0.1246,-0.0182),vec3<f32>(-0.5876,1.1329,-0.1006),vec3<f32>(-0.0728,-0.0083,1.1187));
    let inset=mat3x3<f32>(vec3<f32>(0.856627153315983,0.137318972929847,0.11189821299995),vec3<f32>(0.0951212405381588,0.761241990602591,0.0767994186031903),vec3<f32>(0.0482516061458583,0.101439036467562,0.811302368396859));
    let outset=mat3x3<f32>(vec3<f32>(1.1271005818144368,-0.1413297634984383,-0.14132976349843826),vec3<f32>(-0.11060664309660323,1.157823702216272,-0.11060664309660294),vec3<f32>(-0.016493938717834573,-0.016493938717834257,1.2519364065950405));
    color=inset*(to_rec2020*color);
    color=clamp((log2(max(color,vec3<f32>(1e-10)))+12.47393)/(4.026069+12.47393),vec3<f32>(0.0),vec3<f32>(1.0));
    let x2=color*color; let x4=x2*x2;
    color=15.5*x4*x2-40.14*x4*color+31.96*x4-6.868*x2*color+0.4298*x2+0.1191*color-0.00232;
    color=pow(max(outset*color,vec3<f32>(0.0)),vec3<f32>(2.2));
    return clamp(from_rec2020*color,vec3<f32>(0.0),vec3<f32>(1.0));
  }
  // Khronos PBR Neutral. Compression threshold is 0.8 - 0.04.
  let x=min(color.r,min(color.g,color.b));
  let offset=select(0.04,x-6.25*x*x,x<0.08);
  color-=offset;
  let peak=max(color.r,max(color.g,color.b));
  if (peak<0.76) { return color; }
  let d=0.24; let new_peak=1.0-d*d/(peak+d-0.76);
  color*=new_peak/peak;
  let g=1.0-1.0/(0.15*(peak-new_peak)+1.0);
  return mix(color,vec3<f32>(new_peak),g);
}
@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  let p=vec2<f32>(f32((index<<1u)&2u),f32(index&2u));
  return vec4<f32>(p*2.0-1.0,0.0,1.0);
}
@fragment fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let pixel=textureLoad(source,vec2<i32>(position.xy),0);
  let alpha=clamp(pixel.a,0.0,1.0);
  var color=max(pixel.rgb,vec3<f32>(0.0));
  if (info.input_premultiplied != 0u) {
    if (alpha>0.0) { color/=alpha; } else { color=vec3<f32>(0.0); }
  }
  color=tone_map(color);
  // OUTPUT_TRANSFER
  if (info.output_alpha == 1u) { color*=alpha; }
  // ATTACHMENT_TRANSFER: inverse only when hardware performs the final OETF.
  return vec4<f32>(color,select(alpha,1.0,info.output_alpha==2u));
}
`;
/** Create one reusable output pipeline. render() consumes two GPUTextures, not
 * opaque views, so dimensions, usage, format, samples and feedback are checked
 * before writes. A canvas current texture is a valid destination. An -srgb view
 * is created only for an explicitly configured -srgb format; the caller must
 * have enabled that viewFormat on a non-sRGB base texture/canvas configuration.
 * Output color space is 'srgb' or 'linear'. Premultiplication happens AFTER
 * the chosen transfer, matching canvas premultiplied display storage. For an
 * -srgb attachment the shader undoes that transfer before hardware reapplies it;
 * otherwise semi-transparent pixels would be double-encoded or darkened.
 * signal cancels initialization, not an already submitted GPU operation.
 */
export async function createGpuAnimationOutput(device, options = {}) {
  object(options, "output options");
  for (const key of Object.keys(options))
    if (
      ![
        "format",
        "outputColorSpace",
        "toneMapping",
        "exposure",
        "inputAlpha",
        "outputAlpha",
        "signal",
      ].includes(key)
    )
      fail("OPTIONS", `Unknown option: ${key}`);
  const format = options.format ?? "bgra8unorm",
    outputColorSpace = options.outputColorSpace ?? "srgb",
    signal = options.signal;
  if (
    !formats.includes(format) ||
    !["linear", "srgb"].includes(outputColorSpace) ||
    (format.endsWith("-srgb") && outputColorSpace !== "srgb")
  )
    fail("FORMAT", "Unsupported output format/color-space combination");
  const defaults = Object.freeze(
    animationOutputSettings(options, {
      toneMapping: "none",
      exposure: 1,
      inputAlpha: "premultiplied",
      outputAlpha: "premultiplied",
    }),
  );
  if (
    signal !== undefined &&
    (!signal ||
      typeof signal.addEventListener !== "function" ||
      typeof signal.removeEventListener !== "function")
  )
    fail("OPTIONS", "Expected an AbortSignal");
  const abort = () => {
    if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
  };
  abort();
  for (const name of [
    "createBuffer",
    "createShaderModule",
    "createBindGroupLayout",
    "createPipelineLayout",
    "createRenderPipelineAsync",
    "createBindGroup",
    "createCommandEncoder",
    "pushErrorScope",
    "popErrorScope",
  ])
    if (typeof device?.[name] !== "function") fail("DEVICE", `Missing WebGPU ${name}`);
  for (const name of ["writeBuffer", "submit", "onSubmittedWorkDone"])
    if (typeof device.queue?.[name] !== "function") fail("DEVICE", `Missing WebGPU queue.${name}`);
  let disposed = false,
    terminal = null,
    busy = false,
    uniform,
    pipeline,
    layout,
    boundSource,
    boundGroup,
    version = 0,
    validation = Promise.resolve();
  let rejectStop;
  const stopped = new Promise((_, reject) => {
    rejectStop = reject;
  });
  stopped.catch(() => {});
  function release() {
    uniform?.destroy();
    uniform = undefined;
    boundSource = undefined;
    boundGroup = undefined;
    pipeline = undefined;
    layout = undefined;
  }
  function stop(error) {
    if (!terminal && !disposed) {
      terminal = error;
      release();
      rejectStop(error);
    }
  }
  function live() {
    if (terminal) throw terminal;
    if (disposed) fail("DISPOSED", "Output pass is disposed");
  }
  if (device.lost && typeof device.lost.then === "function")
    device.lost.then(
      (info) =>
        stop(
          new AnimationOutputError("ANIMATION_OUTPUT_DEVICE_LOST", info?.message ?? "Device lost"),
        ),
      stop,
    );
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
    // Pop synchronously: do not capture unrelated work on this borrowed device.
    const v = device.popErrorScope(),
      m = device.popErrorScope();
    const checked = Promise.all([value, v, m]).then(([result, invalid, oom]) => {
      if (error) throw error;
      if (invalid || oom) fail("GPU", (invalid || oom).message ?? "GPU output failure");
      live();
      return result;
    });
    checked.catch(stop);
    return checked;
  }
  const onAbort = () => stop(signal.reason ?? new DOMException("Aborted", "AbortError"));
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    abort();
    const code = ANIMATION_OUTPUT_WGSL.replace(
      "// OUTPUT_TRANSFER",
      outputColorSpace === "srgb" ? "color=linear_to_srgb(color);" : "",
    ).replace(
      "// ATTACHMENT_TRANSFER:",
      format.endsWith("-srgb") ? "color=srgb_to_linear(color); //" : "//",
    );
    const compiled = checked(() => {
      const module = device.createShaderModule({ label: "Animation HDR output", code });
      layout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: 2,
            texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
          },
          { binding: 1, visibility: 2, buffer: { type: "uniform", minBindingSize: 16 } },
        ],
      });
      return device.createRenderPipelineAsync({
        label: "Animation HDR output " + format,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: "vertex_main" },
        fragment: { module, entryPoint: "fragment_main", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
    });
    pipeline = await Promise.race([compiled, stopped]);
    live();
    abort();
    await Promise.race([
      checked(() => {
        uniform = device.createBuffer({
          label: "Animation output parameters",
          size: 16,
          usage: 8 | 64,
        });
      }),
      stopped,
    ]);
    live();
    abort();
  } catch (error) {
    release();
    throw error;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
  function texture(input, label, usage) {
    object(input, label);
    const result = {
      texture: input,
      width: positive(input.width, label + " width"),
      height: positive(input.height, label + " height"),
      format: input.format,
    };
    if (
      input.dimension !== "2d" ||
      input.depthOrArrayLayers !== 1 ||
      input.sampleCount !== 1 ||
      !Number.isInteger(input.usage) ||
      (input.usage & usage) !== usage ||
      typeof input.createView !== "function"
    )
      fail("TEXTURE", `Invalid ${label} dimension, samples, usage or view`);
    return result;
  }
  const result = Object.freeze({
    format,
    outputColorSpace,
    defaults,
    get version() {
      return version;
    },
    get allocatedBytes() {
      return uniform ? 16 : 0;
    },
    get disposed() {
      return disposed;
    },
    get failed() {
      return terminal !== null;
    },
    render(frame) {
      live();
      if (busy) fail("REENTRANT", "Output operation cannot be reentered");
      busy = true;
      try {
        object(frame, "output frame");
        for (const key of Object.keys(frame))
          if (
            !["source", "target", "toneMapping", "exposure", "inputAlpha", "outputAlpha"].includes(
              key,
            )
          )
            fail("OPTIONS", `Unknown frame option: ${key}`);
        const source = texture(frame.source, "source", 4),
          target = texture(frame.target, "target", 16),
          selected = animationOutputSettings(frame, defaults);
        if (source.texture === target.texture)
          fail("FEEDBACK", "Source and destination must differ");
        if (!sourceFormats.includes(source.format))
          fail("FORMAT", "Source must contain linear rgba16float/rgba8unorm/bgra8unorm radiance");
        if (
          target.format !== format &&
          !(format.endsWith("-srgb") && target.format === format.slice(0, -5))
        )
          fail("FORMAT", "Target format differs from the pipeline");
        if (source.width !== target.width || source.height !== target.height)
          fail("EXTENT", "Source and target sizes must match");
        if (version === Number.MAX_SAFE_INTEGER) fail("VERSION", "Submission counter exhausted");
        const bytes = new ArrayBuffer(16),
          data = new DataView(bytes);
        data.setFloat32(0, selected.exposure, true);
        data.setUint32(4, ANIMATION_TONE_MAPPINGS.indexOf(selected.toneMapping), true);
        data.setUint32(8, selected.inputAlpha === "premultiplied" ? 1 : 0, true);
        data.setUint32(12, alphaModes.indexOf(selected.outputAlpha), true);
        let syncError;
        const next = checked(() => {
          try {
            if (boundSource !== source.texture) {
              const view = source.texture.createView({
                dimension: "2d",
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: 0,
                arrayLayerCount: 1,
              });
              boundGroup = device.createBindGroup({
                layout,
                entries: [
                  { binding: 0, resource: view },
                  { binding: 1, resource: { buffer: uniform, size: 16 } },
                ],
              });
              boundSource = source.texture;
            }
            const view = target.texture.createView({
              format,
              dimension: "2d",
              baseMipLevel: 0,
              mipLevelCount: 1,
              baseArrayLayer: 0,
              arrayLayerCount: 1,
            });
            device.queue.writeBuffer(uniform, 0, bytes);
            const encoder = device.createCommandEncoder({ label: "Animation output" });
            const pass = encoder.beginRenderPass({
              colorAttachments: [
                { view, loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] },
              ],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, boundGroup);
            pass.draw(3);
            pass.end();
            device.queue.submit([encoder.finish()]);
          } catch (error) {
            syncError = error;
            throw error;
          }
        });
        validation = Promise.all([validation, next]);
        validation.catch(stop);
        if (syncError) {
          stop(syncError);
          throw syncError;
        }
        version++;
        return result;
      } finally {
        busy = false;
      }
    },
    async whenIdle() {
      live();
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
      if (busy) fail("REENTRANT", "Cannot dispose during output submission");
      if (!disposed) {
        disposed = true;
        release();
        rejectStop(new AnimationOutputError("ANIMATION_OUTPUT_DISPOSED", "Output pass disposed"));
      }
    },
  });
  return result;
}
