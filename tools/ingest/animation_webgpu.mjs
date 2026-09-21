/**
 * Optional WebGPU morph/skin execution for packed animation poses. The caller
 * lends a device and pose; this module owns only its pipeline and mesh buffers.
 * It does not create an adapter, renderer, task scheduler, timer or frame loop.
 *
 * createGpuAnimationDeformer(device, pose, geometry) accepts the CPU deformer's
 * decoded geometry contract. Geometry is snapshotted and uploaded once. Each
 * update uploads only this instance's joint palette and morph weights, then
 * submits a compute pass. flatNormals:true adds a second, ordered compute pass
 * that rebuilds face normals from deformed triangles in the same vertex buffer.
 * There is NO CPU vertex deformation, GPU readback or extra mesh buffer on update.
 *
 * The stable vertexBuffer is interleaved: position XYZ, normal XYZ, tangent
 * XYZW (40 bytes/vertex). vertexLayout describes the attributes actually present
 * at shader locations 0/1/2. Values are already mesh-local morphed/skinned data;
 * the renderer must not apply those operations again. worldMatrix is the
 * matching CPU-side node-to-world snapshot; materials, indices, UVs and other
 * attributes remain with the caller. No CPU bounds are inferred from GPU data:
 * supply conservative animated bounds or disable culling on this opt-in path.
 *
 * Queue ordering is part of the API: update() submits its own compute command
 * buffer. Submit draws/copies consuming that version BEFORE the next update().
 * Merely recording a draw is not consumption. This preserves two poses used
 * within one frame instead of overwriting shared inputs before either dispatch.
 * version/poseVersion acknowledge SUBMISSION, not GPU completion. whenIdle()
 * acknowledges prior work and surfaces validation/OOM/device-loss errors.
 * An unchanged member of an explicitly input-sensitive batch acknowledges reuse
 * of its prior submission while publishing its current world/version snapshot.
 *
 * Arithmetic is WGSL f32, not the CPU sampler's f64 intermediate arithmetic;
 * this is an explicit numerical execution profile, never automatic replacement
 * of source-observable CPU calculations. No speedup claim is made. Disposal
 * destroys owned buffers, never the borrowed pose or device.
 */
import {AnimationPoseError} from './animation_runtime.mjs';
import {createAnimationDeformer} from './animation_deformer.mjs';
const fail = (code, message) => { throw new AnimationPoseError(code, message); };
const F32_BOUND = 3.4028234663852886e38 / 4;
// WebGPU's stable flag values; no browser globals are needed merely to import.
const COPY_SRC = 4, COPY_DST = 8, VERTEX = 32, UNIFORM = 64, STORAGE = 128;
const WORKGROUP = 64;
// Opaque membership prevents foreign buffers, duplicate writes and caller-made
// submission hooks from entering a batch. No device/pose ownership is transferred.
const batchStates = new WeakMap();
const copyWorld = Function.call.bind(Float64Array.prototype.set);

/** Update 0..4096 distinct deformers on one device in one queue submission.
 * All pose inputs are validated before encoding or queue writes. Each mesh keeps
 * its own palette/weights/output buffers and its existing compute pass boundaries,
 * including the ordered flat-normal pass. Inputs may belong to different poses.
 * Returns submission work counts, not a completion promise; drain each owner's
 * whenIdle() as usual. Preflight failures are recoverable and publish nothing;
 * encoding/queue failures make all participating deformers terminal.
 * skipUnchanged:true compares exact f32 input words (including signed zero) to
 * the last submitted palette/morph values, never to an unsubmitted preparation.
 * Unchanged meshes still validate and publish current world/version snapshots.
 * Only this module may write the returned vertexBuffer when using this option.
 * Input shadows cost inputCacheBytes CPU bytes per deformer, not GPU buffers.
 */
export function updateGpuAnimationDeformers(deformers, {skipUnchanged = false} = {}) {
  if (typeof skipUnchanged !== 'boolean') fail('ANIMATION_GPU_BATCH', 'skipUnchanged must be boolean');
  if (!Array.isArray(deformers) || deformers.length > 4096) fail('ANIMATION_GPU_BATCH', 'Expected at most 4096 GPU deformers');
  const count = deformers.length, states = [], seen = new Set();
  for (let i = 0; i < count; i++) {
    const state = batchStates.get(deformers[i]);
    if (!state || seen.has(state) || (i && state.device !== states[0].device)) {
      fail('ANIMATION_GPU_BATCH', 'Batch members must be distinct owned deformers on one device');
    }
    seen.add(state); states.push(state);
  }
  let locked = 0;
  try {
    for (const state of states) { state.lock(); locked++; }
    const versions = states.map(state => state.prepare());
    // A later pose getter must not invalidate an earlier member's snapshot.
    for (let i = 0; i < count; i++) states[i].check(versions[i]);
    const pending = skipUnchanged ? states.filter(state => state.changed()) : states;
    const stats = Object.freeze({meshes: count, dispatchedMeshes: pending.length, skippedMeshes: count - pending.length,
      submissions: pending.length ? 1 : 0,
      dispatches: pending.reduce((n, state) => n + state.dispatches, 0),
      bufferWrites: pending.reduce((n, state) => n + state.bufferWrites, 0),
      uploadedBytes: pending.reduce((n, state) => n + state.uploadedBytes, 0)});
    if (!pending.length) {
      // The shader inputs are bit-identical to the last submitted inputs. Only
      // world transforms and version acknowledgements change; retain outstanding
      // completion/error chains rather than inventing a GPU submission.
      for (let i = 0; i < count; i++) states[i].publish(versions[i], null);
      return stats;
    }
    const device = states[0].device;
    let submitted;
    try {
      submitted = scoped(device, () => {
        const encoder = device.createCommandEncoder({label: 'f3d-animation/batch'});
        for (const state of pending) state.encode(encoder);
        const command = encoder.finish();
        // Finish every pass before the first queue write. No member can overwrite
        // another's per-mesh inputs; duplicates were rejected before acquisition.
        for (let i = 0; i < count; i++) states[i].check(versions[i]);
        for (const state of pending) state.write();
        device.queue.submit([command]);
      });
      submitted.errors.catch(() => {});
      if (submitted.error) throw submitted.error;
      const completion = Promise.all([submitted.errors, device.queue.onSubmittedWorkDone()]).catch(error => {
        for (const state of states) state.reject(error);
        throw error;
      });
      completion.catch(() => {});
      for (let i = 0; i < count; i++) states[i].publish(versions[i], completion);
    } catch (error) {
      submitted?.errors.catch(() => {});
      for (const state of states) state.reject(error);
      throw error;
    }
    return stats;
  } finally {
    for (let i = 0; i < locked; i++) states[i].unlock();
  }
}

export const ANIMATION_DEFORM_WGSL = /* wgsl */`
struct Config { vertices: u32, targets: u32, influences: u32, groups_x: u32 }
struct Influence { joint: u32, weight: f32 }
@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> morphs: array<f32>;
@group(0) @binding(2) var<storage, read> influences: array<Influence>;
@group(0) @binding(3) var<storage, read> palette: array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> config: Config;
@compute @workgroup_size(64)
fn deform(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let vertex = (group.y * config.groups_x + group.x) * 64u + lane;
  if (vertex >= config.vertices) { return; }
  let b = vertex * 10u;
  var p = vec3<f32>(base[b], base[b+1u], base[b+2u]);
  var n = vec3<f32>(base[b+3u], base[b+4u], base[b+5u]);
  var t = vec3<f32>(base[b+6u], base[b+7u], base[b+8u]);
  for (var morph_index = 0u; morph_index < config.targets; morph_index++) {
    let w = weights[morph_index];
    if (w == 0.0) { continue; }
    let m = (morph_index * config.vertices + vertex) * 9u;
    p += w * vec3<f32>(morphs[m], morphs[m+1u], morphs[m+2u]);
    n += w * vec3<f32>(morphs[m+3u], morphs[m+4u], morphs[m+5u]);
    t += w * vec3<f32>(morphs[m+6u], morphs[m+7u], morphs[m+8u]);
  }
  if (config.influences != 0u) {
    var sp = vec3<f32>(0.0); var sn = vec3<f32>(0.0); var st = vec3<f32>(0.0);
    for (var k = 0u; k < config.influences; k++) {
      let influence = influences[vertex * config.influences + k];
      if (influence.weight == 0.0) { continue; }
      let bone = palette[influence.joint];
      sp += influence.weight * (bone * vec4<f32>(p, 1.0)).xyz;
      sn += influence.weight * (bone * vec4<f32>(n, 0.0)).xyz;
      st += influence.weight * (bone * vec4<f32>(t, 0.0)).xyz;
    }
    p = sp; n = sn; t = st;
  }
  output[b] = p.x; output[b+1u] = p.y; output[b+2u] = p.z;
  output[b+3u] = n.x; output[b+4u] = n.y; output[b+5u] = n.z;
  output[b+6u] = t.x; output[b+7u] = t.y; output[b+8u] = t.z;
  output[b+9u] = base[b+9u];
}
`;

// Each invocation owns all three normals of one deindexed triangle. A distinct
// compute pass follows deformation, so no workgroup reads another workgroup's
// unfinished vertex positions. Position/tangent words and the 40-byte ABI stay
// unchanged. Degenerate or f32-subnormal area receives a deterministic zero.
export const ANIMATION_FLAT_NORMALS_WGSL = /* wgsl */`
struct Config { vertices: u32, targets: u32, influences: u32, groups_x: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> config: Config;
@compute @workgroup_size(64)
fn flat_normals(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let triangles = config.vertices / 3u;
  let groups_x = min((triangles + 63u) / 64u, config.groups_x);
  let triangle = (group.y * groups_x + group.x) * 64u + lane;
  if (triangle >= triangles) { return; }
  let b = triangle * 30u;
  let p = vec3<f32>(output[b], output[b+1u], output[b+2u]);
  let a = vec3<f32>(output[b+10u], output[b+11u], output[b+12u]) - p;
  let c = vec3<f32>(output[b+20u], output[b+21u], output[b+22u]) - p;
  let extent = max(max(max(abs(a.x), abs(a.y)), abs(a.z)), max(max(abs(c.x), abs(c.y)), abs(c.z)));
  var normal = vec3<f32>(0.0);
  if (extent >= 1.17549435e-38) {
    let area = cross(a / extent, c / extent);
    let area_extent = max(max(abs(area.x), abs(area.y)), abs(area.z));
    if (area_extent >= 1.17549435e-38) {
      let scaled = area / area_extent;
      normal = scaled * inverseSqrt(dot(scaled, scaled));
    }
  }
  for (var vertex = 0u; vertex < 3u; vertex++) {
    let n = b + vertex * 10u + 3u;
    output[n] = normal.x; output[n+1u] = normal.y; output[n+2u] = normal.z;
  }
}
`;

function fixed(array, length) {
  if (!ArrayBuffer.isView(array) || array instanceof DataView ||
      !(array.buffer instanceof ArrayBuffer) || array.buffer.resizable || array.length !== length) {
    fail('ANIMATION_GPU_STORAGE', 'Expected fixed, unshared pose/geometry storage');
  }
  try { new Uint8Array(array.buffer, 0, 0); }
  catch { fail('ANIMATION_GPU_STORAGE', 'Detached animation storage'); }
}
function f32(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    fail('ANIMATION_GPU_VALUE', 'GPU inputs must be finite and representable as f32');
  }
  return Math.fround(value);
}
function snapshot(geometry, maxComponents) {
  if (!geometry || !Number.isSafeInteger(maxComponents) || maxComponents < 1) fail('ANIMATION_GPU_GEOMETRY', 'Invalid geometry or component budget');
  let count = 0;
  function copy(array) {
    if ((!Array.isArray(array) && !ArrayBuffer.isView(array)) || !Number.isSafeInteger(array.length)) fail('ANIMATION_GPU_GEOMETRY', 'Expected numeric geometry arrays');
    count += array.length;
    if (count > maxComponents) fail('ANIMATION_GPU_LIMIT', 'Geometry exceeds component budget');
    if (ArrayBuffer.isView(array)) fixed(array, array.length);
    // Preserve original values for the shared CPU contract validation. GPU
    // representability is checked before allocating any device resources.
    return Float64Array.from(array, value => { f32(value); return value; });
  }
  const result = {node: geometry.node};
  const flatNormals = geometry.flatNormals;
  if (flatNormals !== undefined) result.flatNormals = flatNormals;
  for (const field of ['positions', 'normals', 'tangents', 'joints', 'weights']) {
    if (geometry[field] !== undefined) result[field] = copy(geometry[field]);
  }
  if (geometry.influences !== undefined) result.influences = geometry.influences;
  const targets = geometry.morphTargets ?? [];
  if (!Array.isArray(targets) || targets.length > 4096) fail('ANIMATION_GPU_GEOMETRY', 'Invalid morph target list');
  result.morphTargets = targets.map(target => {
    if (!target || typeof target !== 'object' || Array.isArray(target)) fail('ANIMATION_GPU_GEOMETRY', 'Invalid morph target');
    return Object.fromEntries(Object.entries(target).map(([name, values]) => [name, copy(values)]));
  });
  return result;
}
// Pop synchronous scopes BEFORE any await: GPUDevice's scope stack is shared
// with other callers, not local to this asynchronous initialization function.
function scoped(device, operation) {
  device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory');
  let value, error;
  try { value = operation(); } catch (caught) { error = caught; }
  const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(results => {
    const reported = results.find(Boolean);
    if (error) throw error;
    if (reported) fail('ANIMATION_GPU_DEVICE', reported.message || 'WebGPU operation failed');
  });
  return {value, errors, error};
}

/** See the module contract for queue ordering and the opt-in f32 profile. */
export async function createGpuAnimationDeformer(device, pose, geometry, {
  maxComponents = 16777216, maxBytes = 128 * 1024 * 1024, label = 'f3d-animation',
} = {}) {
  if (!device?.queue || !device.limits || typeof device.createComputePipelineAsync !== 'function' ||
      typeof device.lost?.then !== 'function') fail('ANIMATION_GPU_DEVICE', 'Lend a live WebGPU device');
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || typeof label !== 'string') fail('ANIMATION_GPU_LIMIT', 'Invalid GPU allocation options');
  const source = snapshot(geometry, maxComponents);
  // Reuse the existing admission contract, including skin indices, normalized
  // weight sums, instance ranges, affine matrices and morph target shapes.
  // This initial CPU evaluation is NOT repeated by the GPU update path.
  const checked = createAnimationDeformer(pose, source, {maxComponents});
  const node = source.node, vertexCount = checked.vertexCount, flatNormals = source.flatNormals === true;
  checked.dispose();
  const skin = pose.instances.find(instance => instance.node === node);
  const targetCount = source.morphTargets.length, influenceCount = skin ? source.influences ?? 4 : 0;
  const paletteStart = skin?.offset ?? 0, paletteCount = skin ? skin.jointCount * 16 : 16;
  const morphStart = pose.morphOffsets[node];
  const posePalette = pose.jointMatrices, poseMorphs = pose.morphWeights, poseWorld = pose.worldMatrices;
  const paletteLength = posePalette.length, morphLength = poseMorphs.length, worldLength = poseWorld.length;
  const limits = device.limits;
  const limit = (name, needed) => {
    if (!Number.isSafeInteger(limits[name]) || limits[name] < needed) fail('ANIMATION_GPU_LIMIT', `Device ${name} is insufficient`);
  };
  limit('maxStorageBuffersPerShaderStage', 6); limit('maxUniformBuffersPerShaderStage', 1);
  limit('maxBindingsPerBindGroup', 7); limit('maxComputeInvocationsPerWorkgroup', WORKGROUP);
  limit('maxComputeWorkgroupSizeX', WORKGROUP); limit('maxComputeWorkgroupsPerDimension', 1);
  const groups = Math.ceil(vertexCount / WORKGROUP), groupsX = Math.min(groups, limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(groups / groupsX);
  limit('maxComputeWorkgroupsPerDimension', groupsY);
  const normalGroups = flatNormals ? Math.ceil(vertexCount / 3 / WORKGROUP) : 0;
  const normalGroupsX = flatNormals ? Math.min(normalGroups, groupsX) : 0;
  const normalGroupsY = flatNormals ? Math.ceil(normalGroups / normalGroupsX) : 0;
  const sizes = [vertexCount * 40, Math.max(4, vertexCount * targetCount * 36),
    Math.max(8, vertexCount * influenceCount * 8), paletteCount * 4, Math.max(4, targetCount * 4), vertexCount * 40, 16];
  if (sizes.reduce((a, b) => a + b, 0) > maxBytes) fail('ANIMATION_GPU_LIMIT', 'GPU buffers exceed byte budget');
  for (let i = 0; i < sizes.length; i++) {
    limit('maxBufferSize', sizes[i]);
    limit(i === 6 ? 'maxUniformBufferBindingSize' : 'maxStorageBufferBindingSize', sizes[i]);
  }
  const base = new Float32Array(vertexCount * 10), morphs = new Float32Array(sizes[1] / 4);
  const influenceData = new ArrayBuffer(sizes[2]), jointWords = new Uint32Array(influenceData), weightWords = new Float32Array(influenceData);
  const maxima = new Float64Array(3), morphMaxima = new Float64Array(targetCount * 3);
  const fields = ['positions', 'normals', 'tangents'];
  for (let v = 0; v < vertexCount; v++) {
    base[v * 10 + 9] = source.tangents ? source.tangents[v * 4 + 3] : 1;
    for (let field = 0; field < 3; field++) for (let axis = 0; axis < 3; axis++) {
      // Flat normals are derived after skinning, never transformed rest normals.
      const value = flatNormals && field === 1 ? 0 : source[fields[field]]?.[v * (field === 2 ? 4 : 3) + axis] ?? 0;
      base[v * 10 + field * 3 + axis] = value;
      maxima[field] = Math.max(maxima[field], Math.abs(Math.fround(value)));
      for (let target = 0; target < targetCount; target++) {
        const delta = Math.fround(source.morphTargets[target][fields[field]]?.[v * 3 + axis] ?? 0);
        morphs[(target * vertexCount + v) * 9 + field * 3 + axis] = delta;
        morphMaxima[target * 3 + field] = Math.max(morphMaxima[target * 3 + field], Math.abs(delta));
      }
    }
    for (let k = 0; k < influenceCount; k++) {
      const i = v * influenceCount + k;
      jointWords[i * 2] = source.joints[i]; weightWords[i * 2 + 1] = source.weights[i];
    }
  }
  const palette = new Float32Array(paletteCount), weights = new Float32Array(sizes[4] / 4);
  if (!skin) palette[0] = palette[5] = palette[10] = palette[15] = 1;
  // Separate staging and submitted shadows are essential: failed preparation
  // must not make a later retry appear already resident. Word equality preserves
  // signed zeros and the actual uploaded f32 rounding, without reading GPU data.
  const paletteWords = new Uint32Array(palette.buffer), morphWords = new Uint32Array(weights.buffer);
  let lastPalette = new Uint32Array(skin ? paletteCount : 0), lastMorphs = new Uint32Array(targetCount);
  const worldMatrix = new Float64Array(16), nextWorld = new Float64Array(16), bounds = new Float64Array(3);
  let disposed = false, terminal = null, busy = false, version = -1, poseVersion = -1, completion = Promise.resolve();
  const buffers = [];
  const release = () => { for (const buffer of buffers) buffer.destroy(); buffers.length = 0; lastPalette = null; lastMorphs = null; };
  const lost = device.lost.then(info => {
    terminal ??= new AnimationPoseError('ANIMATION_GPU_LOST', info?.message || 'WebGPU device lost');
    release(); throw terminal;
  });
  // Loss is observable through update/whenIdle, never an unhandled rejection.
  lost.catch(() => {});
  function live() {
    if (disposed) fail('ANIMATION_GPU_DISPOSED', 'GPU deformer has been disposed');
    if (terminal) throw terminal;
    if (pose.disposed) fail('ANIMATION_DISPOSED', 'Animation player has been disposed');
  }
  function prepare() {
    live(); fixed(posePalette, paletteLength); fixed(poseMorphs, morphLength); fixed(poseWorld, worldLength);
    fixed(worldMatrix, 16);
    const nextVersion = pose.version;
    if (!Number.isSafeInteger(nextVersion) || nextVersion < 0) fail('ANIMATION_GPU_VALUE', 'Invalid pose version');
    for (let i = 0; i < 16; i++) {
      const value = poseWorld[node * 16 + i];
      if (!Number.isFinite(value)) fail('ANIMATION_GPU_VALUE', 'Non-finite world matrix');
      nextWorld[i] = value;
    }
    if (nextWorld[3] || nextWorld[7] || nextWorld[11] || nextWorld[15] !== 1) fail('ANIMATION_GPU_VALUE', 'World transform must be affine');
    bounds.set(maxima);
    for (let t = 0; t < targetCount; t++) {
      weights[t] = f32(poseMorphs[morphStart + t]);
      for (let field = 0; field < 3; field++) bounds[field] += Math.abs(weights[t]) * morphMaxima[t * 3 + field];
    }
    // A conservative bound rejects f32 intermediate overflow before queue
    // writes, without scanning/deforming every vertex on the CPU each frame.
    for (const bound of bounds) if (bound > F32_BOUND) fail('ANIMATION_GPU_VALUE', 'Morph arithmetic exceeds the finite f32 profile');
    if (skin) {
      for (let i = 0; i < paletteCount; i++) palette[i] = f32(posePalette[paletteStart + i]);
      for (let i = 0; i < paletteCount; i += 16) {
        if (palette[i+3] || palette[i+7] || palette[i+11] || palette[i+15] !== 1) fail('ANIMATION_GPU_VALUE', 'Joint palette must be affine');
        for (let row = 0; row < 3; row++) {
          const scale = Math.abs(palette[i+row]) + Math.abs(palette[i+4+row]) + Math.abs(palette[i+8+row]);
          for (let field = 0; field < 3; field++) {
            const bound = (scale * bounds[field] + (field === 0 ? Math.abs(palette[i+12+row]) : 0)) * 1.001;
            if (bound > F32_BOUND) fail('ANIMATION_GPU_VALUE', 'Skin arithmetic exceeds the finite f32 profile');
          }
        }
      }
    }
    if (pose.version !== nextVersion || pose.disposed) fail('ANIMATION_GPU_CHANGED', 'Pose changed during GPU input capture');
    return nextVersion;
  }
  let pipeline, bindGroup, normalPipeline, normalBindGroup;
  try {
    prepare();
    const allocated = scoped(device, () => {
      const initial = [base, morphs, new Uint8Array(influenceData), palette, weights, null,
        new Uint32Array([vertexCount, targetCount, influenceCount, groupsX])];
      for (let i = 0; i < sizes.length; i++) {
        const buffer = device.createBuffer({label: `${label}/${i}`, size: sizes[i],
          usage: i === 6 ? UNIFORM : i === 5 ? STORAGE | VERTEX | COPY_SRC : STORAGE | COPY_DST,
          mappedAtCreation: initial[i] !== null});
        buffers.push(buffer);
        if (initial[i]) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(initial[i].buffer, initial[i].byteOffset, initial[i].byteLength)); buffer.unmap(); }
      }
      const module = device.createShaderModule({label, code: ANIMATION_DEFORM_WGSL});
      const deformReady = device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'deform'}});
      // A synchronous failure creating the optional pipeline must not leave the
      // already-started deformation pipeline rejection unobserved.
      deformReady.catch(() => {});
      const normalReady = flatNormals ? device.createComputePipelineAsync({label: `${label}/flat-normals`, layout: 'auto', compute: {
        module: device.createShaderModule({label: `${label}/flat-normals`, code: ANIMATION_FLAT_NORMALS_WGSL}), entryPoint: 'flat_normals',
      }}) : null;
      return Promise.all([deformReady, normalReady]);
    });
    [[pipeline, normalPipeline]] = await Promise.race([Promise.all([allocated.value, allocated.errors]), lost]);
    live();
    const bound = scoped(device, () => device.createBindGroup({label, layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}}))}));
    bindGroup = bound.value; await Promise.race([bound.errors, lost]); live();
    if (flatNormals) {
      const normalsBound = scoped(device, () => device.createBindGroup({label: `${label}/flat-normals`, layout: normalPipeline.getBindGroupLayout(0),
        entries: [{binding: 0, resource: {buffer: buffers[5]}}, {binding: 1, resource: {buffer: buffers[6]}}]}));
      normalBindGroup = normalsBound.value; await Promise.race([normalsBound.errors, lost]); live();
    }
  } catch (error) { release(); throw error; }
  const attributes = [{shaderLocation: 0, offset: 0, format: 'float32x3'}];
  if (source.normals || flatNormals) attributes.push({shaderLocation: 1, offset: 12, format: 'float32x3'});
  if (source.tangents) attributes.push({shaderLocation: 2, offset: 24, format: 'float32x4'});
  const vertexLayout = Object.freeze({arrayStride: 40, stepMode: 'vertex', attributes: Object.freeze(attributes.map(Object.freeze))});
  function update() {
    updateGpuAnimationDeformers([result]); return result;
  }
  const result = Object.freeze({vertexBuffer: buffers[5], vertexCount, vertexLayout, worldMatrix, node,
    bufferBytes: sizes.reduce((a, b) => a + b, 0),
    get inputCacheBytes() { return (lastPalette?.byteLength ?? 0) + (lastMorphs?.byteLength ?? 0); },
    execution: 'webgpu-compute-f32', update,
    async whenIdle() {
      live();
      await Promise.race([Promise.all([completion, device.queue.onSubmittedWorkDone()]), lost]);
      live(); return result;
    },
    get version() { return version; }, get poseVersion() { return poseVersion; },
    get disposed() { return disposed; }, get failed() { return terminal !== null; },
    dispose() { if (busy) fail('ANIMATION_REENTRANT', 'Cannot dispose during GPU submission'); if (!disposed) { disposed = true; release(); } },
  });
  batchStates.set(result, {
    device, prepare,
    changed() {
      if (version < 0) return true;
      for (let i = 0; i < lastPalette.length; i++) if (lastPalette[i] !== paletteWords[i]) return true;
      for (let i = 0; i < lastMorphs.length; i++) if (lastMorphs[i] !== morphWords[i]) return true;
      return false;
    },
    dispatches: flatNormals ? 2 : 1,
    bufferWrites: Number(!!skin) + Number(targetCount > 0),
    uploadedBytes: (skin ? palette.byteLength : 0) + (targetCount ? weights.byteLength : 0),
    lock() {
      if (busy) fail('ANIMATION_REENTRANT', 'GPU deformation cannot be reentered');
      busy = true; try { live(); } catch (error) { busy = false; throw error; }
    },
    unlock() { busy = false; },
    check(expected) {
      live(); fixed(worldMatrix, 16);
      if (pose.version !== expected) fail('ANIMATION_GPU_CHANGED', 'Pose changed during batched GPU input capture');
    },
    encode(encoder) {
      const pass = encoder.beginComputePass({label});
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
      if (flatNormals) {
        const normalPass = encoder.beginComputePass({label: `${label}/flat-normals`});
        normalPass.setPipeline(normalPipeline); normalPass.setBindGroup(0, normalBindGroup);
        normalPass.dispatchWorkgroups(normalGroupsX, normalGroupsY); normalPass.end();
      }
    },
    write() {
      if (skin) device.queue.writeBuffer(buffers[3], 0, palette);
      if (targetCount) device.queue.writeBuffer(buffers[4], 0, weights);
    },
    publish(nextVersion, submitted) {
      // A later batch must not hide an earlier unresolved error scope.
      if (submitted !== null) {
        completion = Promise.race([Promise.all([completion, submitted]), lost])
          .then(() => { if (terminal) throw terminal; }, error => { terminal ??= error; throw terminal; });
        completion.catch(() => {});
      }
      lastPalette.set(paletteWords.subarray(0, lastPalette.length));
      lastMorphs.set(morphWords.subarray(0, lastMorphs.length));
      copyWorld(worldMatrix, nextWorld); poseVersion = nextVersion; version++;
    },
    reject(error) { terminal ??= error; release(); },
  });
  try { update(); await result.whenIdle(); return result; }
  catch (error) { result.dispose(); throw error; }
}
