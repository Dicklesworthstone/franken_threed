/** GPU-only trilinear reduction for native 3D storage textures.
 * All three spatial dimensions shrink. Eight clamped textureLoad taps retain
 * float32 precision and work without float32-filterable. Each dependent mip
 * runs in a separate pass; the caller owns scopes, lifetime and cache. */
const FORMATS = new Set(['rgba8unorm', 'rgba16float', 'r32float', 'rg32float', 'rgba32float']);
export function storageVolumeMipShader(format) {
  if (!FORMATS.has(format)) throw new Error('Unsupported storage mip format');
  return `
@group(0) @binding(0) var source: texture_3d<f32>;
@group(0) @binding(1) var destination: texture_storage_3d<${format}, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let target = textureDimensions(destination);
  if (any(id >= target)) { return; }
  let size = vec3i(textureDimensions(source));
  let uv = (vec3f(id) + 0.5) / vec3f(target) * vec3f(size) - 0.5;
  let base = vec3i(floor(uv)); let f = fract(uv); let hi = size - 1;
  let a = textureLoad(source, clamp(base, vec3i(0), hi), 0);
  let b = textureLoad(source, clamp(base + vec3i(1,0,0), vec3i(0), hi), 0);
  let c = textureLoad(source, clamp(base + vec3i(0,1,0), vec3i(0), hi), 0);
  let d = textureLoad(source, clamp(base + vec3i(1,1,0), vec3i(0), hi), 0);
  let e = textureLoad(source, clamp(base + vec3i(0,0,1), vec3i(0), hi), 0);
  let g = textureLoad(source, clamp(base + vec3i(1,0,1), vec3i(0), hi), 0);
  let h = textureLoad(source, clamp(base + vec3i(0,1,1), vec3i(0), hi), 0);
  let j = textureLoad(source, clamp(base + vec3i(1,1,1), vec3i(0), hi), 0);
  let front = mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
  let back = mix(mix(e,g,f.x), mix(h,j,f.x), f.y);
  textureStore(destination, vec3i(id), mix(front, back, f.z));
}`;
}

export function submitStorageVolumeMips(device, texture, shape, pipelines) {
  const {width, height, depth, format, mipLevelCount} = shape;
  const limit = device.limits.maxComputeWorkgroupsPerDimension ?? 65535;
  const steps = Array.from({length: mipLevelCount - 1}, (_, i) => {
    const level = i + 1;
    const size = [width, height, depth].map(n => Math.ceil(Math.max(1, Math.floor(n / 2 ** level)) / 4));
    if (size.some(n => n > limit)) throw new Error('Storage mip dispatch exceeds device limits');
    return {level, size};
  });
  if (!steps.length) return;
  const key = `volume/${format}`;
  let pipeline = pipelines.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({label: 'f3d-storage-volume-mips', code: storageVolumeMipShader(format)});
    const layout = device.createBindGroupLayout({entries: [
      {binding: 0, visibility: 4, texture: {sampleType: 'unfilterable-float', viewDimension: '3d'}},
      {binding: 1, visibility: 4, storageTexture: {access: 'write-only', format, viewDimension: '3d'}},
    ]});
    pipeline = device.createComputePipeline({layout: device.createPipelineLayout({bindGroupLayouts: [layout]}), compute: {module, entryPoint: 'main'}});
    pipelines.set(key, pipeline);
  }
  const encoder = device.createCommandEncoder({label: 'f3d-storage-volume-mips'});
  for (const {level, size} of steps) {
    const view = mip => texture.createView({dimension: '3d', baseMipLevel: mip, mipLevelCount: 1});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: view(level - 1)}, {binding: 1, resource: view(level)},
    ]});
    const pass = encoder.beginComputePass({label: `f3d-storage-volume-mip-${level}`});
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(...size); pass.end();
  }
  device.queue.submit([encoder.finish()]);
}
