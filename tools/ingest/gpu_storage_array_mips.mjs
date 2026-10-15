/** GPU-only mip reduction of every layer of a storage texture array.
 * Layers never filter into one another. Each dependent level has its own
 * compute pass with nonoverlapping mip views of the same allocation.
 * The caller owns validation scopes, resource lifetime and the pipeline cache. */
const FORMATS = new Set(['rgba8unorm', 'rgba16float', 'r32float', 'rg32float', 'rgba32float']);
export function storageArrayMipShader(format) {
  if (!FORMATS.has(format)) throw new Error('Unsupported storage mip format');
  return `
@group(0) @binding(0) var source: texture_2d_array<f32>;
@group(0) @binding(1) var destination: texture_storage_2d_array<${format}, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let target = textureDimensions(destination);
  if (any(id.xy >= target) || id.z >= textureNumLayers(destination)) { return; }
  let size = vec2i(textureDimensions(source));
  let uv = (vec2f(id.xy) + 0.5) / vec2f(target) * vec2f(size) - 0.5;
  let base = vec2i(floor(uv)); let f = fract(uv); let hi = size - 1;
  let layer = i32(id.z);
  let a = textureLoad(source, clamp(base, vec2i(0), hi), layer, 0);
  let b = textureLoad(source, clamp(base + vec2i(1,0), vec2i(0), hi), layer, 0);
  let c = textureLoad(source, clamp(base + vec2i(0,1), vec2i(0), hi), layer, 0);
  let d = textureLoad(source, clamp(base + vec2i(1,1), vec2i(0), hi), layer, 0);
  textureStore(destination, vec2i(id.xy), layer, mix(mix(a,b,f.x), mix(c,d,f.x), f.y));
}`;
}

export function submitStorageArrayMips(device, texture, shape, pipelines) {
  const {width, height, depth, format, mipLevelCount} = shape;
  // Validate the entire dispatch sequence before encoding/submitting its prefix.
  const limit = device.limits.maxComputeWorkgroupsPerDimension ?? 65535;
  const steps = Array.from({length: mipLevelCount - 1}, (_, i) => {
    const level = i + 1;
    const size = [Math.ceil(Math.max(1, Math.floor(width / 2 ** level)) / 8),
      Math.ceil(Math.max(1, Math.floor(height / 2 ** level)) / 8), depth];
    if (size.some(n => n > limit)) throw new Error('Storage mip dispatch exceeds device limits');
    return {level, size};
  });
  if (!steps.length) return;
  const key = `array/${format}`;
  let pipeline = pipelines.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({label: 'f3d-storage-array-mips', code: storageArrayMipShader(format)});
    const layout = device.createBindGroupLayout({entries: [
      {binding: 0, visibility: 4, texture: {sampleType: 'unfilterable-float', viewDimension: '2d-array'}},
      {binding: 1, visibility: 4, storageTexture: {access: 'write-only', format, viewDimension: '2d-array'}},
    ]});
    pipeline = device.createComputePipeline({layout: device.createPipelineLayout({bindGroupLayouts: [layout]}), compute: {module, entryPoint: 'main'}});
    pipelines.set(key, pipeline);
  }
  const encoder = device.createCommandEncoder({label: 'f3d-storage-array-mips'});
  for (const {level, size} of steps) {
    const view = mip => texture.createView({dimension: '2d-array', baseMipLevel: mip, mipLevelCount: 1});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: view(level - 1)}, {binding: 1, resource: view(level)},
    ]});
    const pass = encoder.beginComputePass({label: `f3d-storage-array-mip-${level}`});
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(...size); pass.end();
  }
  device.queue.submit([encoder.finish()]);
}
