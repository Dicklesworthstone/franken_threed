/** r186 WebGLRenderer.renderTransmissionPass on the WebGL-surface program route.
 *
 * Route: a retained-JavaScript port (MIT, three.js authors) of r186
 * `renderTransmissionPass`, drawing with the application's own ShaderLib /
 * ShaderMaterial programs (three_program.mjs, compiled with `rows: 'gl'`):
 *
 * - one `WebGLRenderTarget` per camera id, created with r186's options
 *   (HalfFloat, LinearMipmapLinear, 4 samples, working color space) and resized
 *   every frame to the active viewport times `transmissionResolutionScale`;
 * - clear to the renderer clear color (white at alpha 0.5, premultiplied, when
 *   the clear alpha is below 1; a Color background clears to itself at alpha 1),
 *   then the frame's opaque list (background mesh included) with
 *   `onBeforeRender`/`onAfterRender` called again for each object, as r186's
 *   renderObjects does, through render-target program variants (linear output,
 *   no tone mapping);
 * - MSAA resolve and box-filtered mip generation (WebGL generateMipmap);
 * - DoubleSide transmissive objects again with `side = BackSide` (and the two
 *   `needsUpdate` bumps r186 makes), sampling the resolved opaque result while
 *   drawing into the multisampled attachment, then resolve and mips again.
 *
 * GPU storage: rgba16float color (4x MSAA) resolved into a mipmapped
 * rgba16float texture, depth24plus. Receivers bind the resolved texture as
 * `transmissionSamplerMap` with r186's sampler (trilinear, clamp).
 *
 * Stated difference: r186 invalidates the multisampled depth after the first
 * resolve (storeMultisampledDepthBuffer: false), leaving the back-face pass's
 * depth test against undefined contents; here the opaque depth is kept.
 * Frames never create pipelines: a missing record or a resized target is a
 * THREE_PROGRAM_TRANSMISSION_PREPARE boundary, raised before any source callback
 * of the pass runs. Explicit errors: scene.overrideMaterial is skipped exactly as
 * r186 skips it (no pass). No performance claim.
 */
import {createProgramMeshes} from './animation_program_mesh.mjs';

export class ThreeProgramTransmissionError extends Error {
  constructor(code, message) { super(`THREE_PROGRAM_TRANSMISSION_${code}: ${message}`); this.name = 'ThreeProgramTransmissionError'; this.code = 'THREE_PROGRAM_TRANSMISSION_' + code; }
}
const fail = (code, message) => { throw new ThreeProgramTransmissionError(code, message); };
const COLOR_FORMAT = 'rgba16float', DEPTH_FORMAT = 'depth24plus', SAMPLES = 4;

function scoped(device, operation) {
  device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory');
  let value, error;
  try { value = operation(); } catch (caught) { error = caught; }
  const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(values => {
    if (error) throw error;
    const reported = values.find(Boolean);
    if (reported) fail('DEVICE', reported.message || 'WebGPU operation failed');
  });
  return {value, error, errors};
}

/** items given to prepare/render: {object, material, group, side, gpu, compile(), refresh(), pack(bytes, camera, size)}
 * where compile() returns the render-target program variant for (material, object, side),
 * gpu is the program geometry residency it reads, and bindingOf(texture) resolves samplers. */
export function createThreeProgramTransmission({three: T, device, support, bindingOf, label = 'f3d-program-transmission', maxDraws = 4096}) {
  const targets = new Map(), records = new Map(), bindings = new Map(), lost = new Promise(() => {});
  const meshes = createProgramMeshes({device, format: COLOR_FORMAT, depthFormat: DEPTH_FORMAT, sampleCount: SAMPLES, maxDraws, label,
    fail: (code, message) => { throw new ThreeProgramTransmissionError(code.replace(/^ANIMATION_RENDER_/, ''), message); }, scoped, lost});
  const sampler = device.createSampler({label, magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
  const clearColor = new T.Color();
  let disposed = false, mipper = null, pending = [];
  const live = () => { if (disposed) fail('DISPOSED', 'Transmission owner is disposed'); };
  const ids = new WeakMap();
  let nextId = 0;
  const idOf = o => { let id = ids.get(o); if (id === undefined) ids.set(o, id = ++nextId); return id; };

  /** currentRenderState.state.transmissionRenderTarget[camera.id], sized as r186 sizes it. */
  function targetFor(camera, width, height) {
    let state = targets.get(camera.id);
    if (!state) {
      const target = new T.WebGLRenderTarget(1, 1, {generateMipmaps: true, type: T.HalfFloatType, minFilter: T.LinearMipmapLinearFilter,
        samples: SAMPLES, stencilBuffer: false, resolveDepthBuffer: false, resolveStencilBuffer: false,
        storeMultisampledDepthBuffer: false, storeMultisampledStencilBuffer: false, colorSpace: T.ColorManagement.workingColorSpace});
      targets.set(camera.id, state = {target, gpu: null});
    }
    state.target.setSize(width, height);
    return state;
  }
  /** GPU storage for a target's current size; true when it had to change. */
  function allocate(state) {
    const width = Math.max(1, Math.floor(state.target.width)), height = Math.max(1, Math.floor(state.target.height));
    if (state.gpu && state.gpu.width === width && state.gpu.height === height) return false;
    if (state.gpu) release(state);
    const mips = Math.floor(Math.log2(Math.max(width, height))) + 1;
    const msaa = device.createTexture({label: `${label}/msaa`, size: [width, height], format: COLOR_FORMAT, sampleCount: SAMPLES, usage: 16});
    const depth = device.createTexture({label: `${label}/depth`, size: [width, height], format: DEPTH_FORMAT, sampleCount: SAMPLES, usage: 16});
    const resolve = device.createTexture({label: `${label}/color`, size: [width, height], format: COLOR_FORMAT, mipLevelCount: mips, usage: 16 | 4});
    state.gpu = {width, height, mips, msaa, depth, resolve, msaaView: msaa.createView(), depthView: depth.createView(),
      levels: Array.from({length: mips}, (_, level) => resolve.createView({baseMipLevel: level, mipLevelCount: 1})),
      binding: {view: resolve.createView(), sampler, sampleType: 'float'}};
    bindings.set(state.target.texture, state.gpu.binding);
    return true;
  }
  function release(state) {
    state.gpu.msaa.destroy(); state.gpu.depth.destroy(); state.gpu.resolve.destroy();
    bindings.delete(state.target.texture); state.gpu = null;
  }

  function lookup(item) {
    const compiled = item.compile();
    const reflection = compiled.program.reflection;
    const values = item.refresh();
    const samplers = support.objectSamplers?.(item.material, item.sourceObject) ?? null;
    const textures = reflection.textures.map(t => {
      const value = samplers?.[t.name] ?? values?.[t.name]?.value, texture = t.element === null ? value : value?.[t.element];
      const b = texture ? (bindings.get(texture) ?? bindingOf(texture)) : bindingOf(null);
      return {view: b.view, sampler: b.sampler, sampleType: b.sampleType ?? 'float', texture: texture ?? null};
    });
    const raster = support.raster(item.material, {side: item.side, ...(item.topology ? {topology: item.topology} : {})});
    // rows:'gl' mirrors clip Y: GL's counter-clockwise front faces are clockwise here.
    const glRaster = {...raster, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw'};
    const geometry = support.geometrySnapshot(item.gpu, device);
    const key = [compiled.key, item.topology ?? 'triangles', idOf(item.gpu), geometry.signature, JSON.stringify(glRaster), ...textures.map(t => idOf(t.view) + ':' + idOf(t.sampler))].join('\u0001');
    return {key, compiled, textures, glRaster, entry: records.get(key)};
  }
  /** Build the records the last render() found missing (and any given now). */
  async function prepare(items = []) {
    live();
    const list = [...pending, ...items];
    pending = [];
    for (const item of list) {
      const {key, compiled, textures, glRaster, entry} = lookup(item);
      if (entry) continue;
      const record = await meshes.add(item.gpu, {program: compiled.program, textures, raster: glRaster, topology: item.topology ?? 'triangles',
        ...(item.stripIndexFormat ? {stripIndexFormat: item.stripIndexFormat} : {})});
      records.set(key, {record, textures});
    }
  }

  /** renderTransmissionPass for this frame. Source-visible effects (target size,
   * callbacks, side/needsUpdate toggles) happen now; the returned function
   * submits the pass (after this frame's uploads, before the main pass). */
  function render(opaqueItems, transmissiveItems, camera, {width, height, background, renderer, sourceScene}) {
    live();
    const state = targetFor(camera, width, height);
    if (allocate(state)) fail('PREPARE', 'Transmission target was (re)allocated; call prepare()');
    const back = transmissiveItems.filter(item => item.material.side === T.DoubleSide && item.object.layers.test(camera.layers));
    // Every record before any callback: a PREPARE boundary must not replay effects.
    const found = [...opaqueItems, ...back.map(item => item.backSide)].map(item => {
      const r = lookup(item);
      if (!r.entry) pending.push(item);
      return r;
    });
    if (pending.length) fail('PREPARE', 'Transmission pass programs need prepare()');
    meshes.begin();
    const draw = (item, r, commands) => {
      const object = item.object, geometry = object.geometry;
      const hooked = object.onBeforeRender !== T.Object3D.prototype.onBeforeRender || object.onAfterRender !== T.Object3D.prototype.onAfterRender;
      if (hooked) object.onBeforeRender(renderer, sourceScene, camera, geometry, item.material, item.group);
      if (item.material.onBeforeRender !== T.Material.prototype.onBeforeRender) item.material.onBeforeRender(renderer, sourceScene, camera, geometry, object, item.group);
      object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
      object.normalMatrix.getNormalMatrix(object.modelViewMatrix);
      const reflection = r.entry.record.reflection, bytes = new Uint8Array(reflection.uniformBufferSize);
      const current = item.pack(bytes, camera, [state.gpu.width, state.gpu.height, state.gpu.width, state.gpu.height], reflection);
      if (current.some((t, k) => (t ?? null) !== r.entry.textures[k].texture)) fail('PREPARE', 'Transmission pass textures changed; call prepare()');
      const start = item.group ? item.group.start : 0, count = item.group ? item.group.count : Number.MAX_SAFE_INTEGER;
      const instanceCount = object.isInstancedMesh ? object.count : geometry.isInstancedBufferGeometry ? geometry.instanceCount : 1;
      const command = {};
      meshes.stage(r.entry.record, {programUniforms: bytes, first: start, count, frontFaceCW: object.isMesh === true && object.matrixWorld.determinant() < 0, instanceCount}, command);
      commands.push(command);
      if (hooked) object.onAfterRender(renderer, sourceScene, camera, geometry, item.material, item.group);
    };
    const opaqueCommands = [], backCommands = [];
    opaqueItems.forEach((item, i) => draw(item, found[i], opaqueCommands));
    back.forEach((item, i) => {
      const material = item.material, currentSide = material.side;
      material.side = T.BackSide; material.needsUpdate = true;
      item.backSide.beforeDraw?.();
      draw(item.backSide, found[opaqueItems.length + i], backCommands);
      material.side = currentSide; material.needsUpdate = true;
      item.backSide.afterDraw?.();
    });
    // _this.clear() then background.render(scene): see the header for the color rules.
    let clear;
    if (background?.isColor) clear = [background.r, background.g, background.b, 1];
    else {
      renderer.getClearColor(clearColor);
      let alpha = renderer.getClearAlpha();
      if (alpha < 1) { clearColor.setRGB(1, 1, 1); alpha = 0.5; }
      clear = [clearColor.r * alpha, clearColor.g * alpha, clearColor.b * alpha, alpha];
    }
    const gpu = state.gpu;
    return () => submit(gpu, clear, opaqueCommands, backCommands);
  }
  function encodeDraws(pass, commands) {
    for (const c of commands) {
      if (!c.count || !c.instanceCount) continue;
      pass.setPipeline(c.pipeline);
      pass.setBindGroup(0, c.program.group, [c.program.offset]);
      if (c.record.textureGroup) pass.setBindGroup(1, c.record.textureGroup);
      c.vertexBuffers.forEach((b, slot) => pass.setVertexBuffer(slot, b));
      if (c.indexBuffer) { pass.setIndexBuffer(c.indexBuffer, c.indexFormat); pass.drawIndexed(c.count, c.instanceCount, c.first, 0, 0); }
      else pass.draw(c.count, c.instanceCount, c.first, c.firstInstance ?? 0);
    }
  }
  function submit(gpu, clear, opaqueCommands, backCommands) {
    meshes.write();
    const encoder = device.createCommandEncoder({label});
    const viewport = pass => pass.setViewport(0, 0, gpu.width, gpu.height, 0, 1);
    const back = backCommands.length > 0;
    const first = encoder.beginRenderPass({label,
      colorAttachments: [{view: gpu.msaaView, resolveTarget: gpu.levels[0], clearValue: clear, loadOp: 'clear', storeOp: back ? 'store' : 'discard'}],
      depthStencilAttachment: {view: gpu.depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: back ? 'store' : 'discard'}});
    viewport(first); encodeDraws(first, opaqueCommands); first.end();
    generateMips(encoder, gpu);
    if (back) {
      // The back faces sample the resolved texture, so this pass only writes the
      // multisampled attachment; an empty pass then resolves it.
      const second = encoder.beginRenderPass({label, colorAttachments: [{view: gpu.msaaView, loadOp: 'load', storeOp: 'store'}],
        depthStencilAttachment: {view: gpu.depthView, depthLoadOp: 'load', depthStoreOp: 'discard'}});
      viewport(second); encodeDraws(second, backCommands); second.end();
      encoder.beginRenderPass({label, colorAttachments: [{view: gpu.msaaView, resolveTarget: gpu.levels[0], loadOp: 'load', storeOp: 'discard'}]}).end();
      generateMips(encoder, gpu);
    }
    device.queue.submit([encoder.finish()]);
  }
  function generateMips(encoder, gpu) {
    if (!mipper) {
      const module = device.createShaderModule({label: `${label}/mips`, code: `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct V { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
  let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return V(vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(xy.x, 1.0 - xy.y));
}
@fragment fn fs(v: V) -> @location(0) vec4<f32> { return textureSampleLevel(src, smp, v.uv, 0.0); }`});
      const pipeline = device.createRenderPipeline({label, layout: 'auto', vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format: COLOR_FORMAT}]}});
      mipper = {pipeline, sampler: device.createSampler({label, minFilter: 'linear', magFilter: 'linear'})};
    }
    for (let level = 1; level < gpu.mips; level++) {
      const group = device.createBindGroup({layout: mipper.pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: gpu.levels[level - 1]}, {binding: 1, resource: mipper.sampler}]});
      const pass = encoder.beginRenderPass({label, colorAttachments: [{view: gpu.levels[level], loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store'}]});
      pass.setPipeline(mipper.pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
    }
  }
  return Object.freeze({
    prepare, render,
    /** r186's transmissionRenderTarget for a camera, created/sized as the pass does (uniform refresh reads it). */
    target(camera, width, height) { live(); const state = targetFor(camera, width, height); allocate(state); return state.target; },
    /** Receiver binding of a transmission target texture, or undefined. */
    binding: texture => bindings.get(texture),
    get pending() { return pending.length; },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const state of targets.values()) { if (state.gpu) release(state); state.target.dispose(); }
      targets.clear(); records.clear(); meshes.dispose(); pending = [];
    },
  });
}
