/** r186 PMREM environments for ShaderLib programs on the new WebGPU backend.
 *
 * Route: the application's own r186 `PMREMGenerator` (retained JavaScript, MIT,
 * three.js authors) runs unchanged against a recording host that implements the
 * renderer calls it makes (setRenderTarget / render / autoClear / xr). Each
 * recorded draw is one of PMREMGenerator's ShaderMaterials (CubemapToCubeUV,
 * EquirectToCubeUV, PMREMGGXConvolution) assembled as r186 WebGLProgram does
 * (three_program.mjs), compiled by essl_wgsl.mjs with `rows: 'gl'` so render
 * targets hold GL's bottom-up rows and the receiving `textureCubeUV` code
 * samples them with GL's uv convention, then replayed in recording order as one
 * WebGPU command buffer: one render pass per draw, the target's viewport and
 * scissor, `load` of everything outside them, a per-draw uniform slice (the
 * GGX passes reuse one material with different values), and the ping-pong /
 * cube-UV textures as rgba16float (HalfFloatType RGBA targets).
 *
 * Cache semantics follow r186 WebGLEnvironments.getPMREM: one cube-UV target per
 * source texture, generated once the source is complete (cube: six images;
 * equirect: image height > 0), never regenerated on later source edits (r186
 * regenerates only render-target textures by pmremVersion), released on the
 * source's `dispose` event. Render-target sources (pmremVersion) are explicit
 * errors here. No performance claim.
 */
import {threeProgramSources, threeProgramRaster, packThreeProgramUniforms} from './three_program.mjs';
import {compileEsslProgram} from './essl_wgsl.mjs';

export class ThreeProgramPMREMError extends Error {
  constructor(code, message) { super(`THREE_PMREM_${code}: ${message}`); this.name = 'ThreeProgramPMREMError'; this.code = 'THREE_PMREM_' + code; }
}
const fail = (code, message) => { throw new ThreeProgramPMREMError(code, message); };
const VERTEX = 1, FRAGMENT = 2, ALIGN = 256;

/** WebGLEnvironments source classification: 'pmrem' (equirect/cube needing
 * conversion), 'direct' (already a cube-UV or other texture, used as is). */
export function pmremSourceKind(T, texture) {
  const m = texture?.mapping;
  if (m === T.EquirectangularReflectionMapping || m === T.EquirectangularRefractionMapping) return 'equirect';
  if (m === T.CubeReflectionMapping || m === T.CubeRefractionMapping) return 'cube';
  return 'direct';
}
/** r186 readiness: equirect image height > 0; cube: all six faces assigned. */
export function pmremSourceComplete(T, texture) {
  const kind = pmremSourceKind(T, texture), image = texture.image;
  if (kind === 'equirect') return !!image && image.height > 0;
  if (kind === 'cube') { let n = 0; for (let i = 0; i < 6; i++) if (image?.[i] !== undefined) n++; return n === 6; }
  return true;
}

/** bindingOf(texture) -> {view, sampler} for a source texture (the scene's texture owner). */
export function createThreeProgramPMREM({three: T, device, bindingOf, label = 'f3d-pmrem'}) {
  const generator = new T.PMREMGenerator(createHost());
  const done = new Map(), outputs = new Map(), targets = new Map(), programs = new Map(), retired = [];
  let disposed = false, sampler = null, constant = null;

  function createHost() {
    let target = null, recording = null;
    return {
      autoClear: true, toneMapping: T.NoToneMapping, xr: {enabled: false},
      state: {buffers: {depth: {getReversed: () => false}}},
      getRenderTarget: () => target, getActiveCubeFace: () => 0, getActiveMipmapLevel: () => 0,
      setRenderTarget(t) { target = t; }, getClearColor: color => color.setRGB(0, 0, 0), clear() {}, compile() {},
      render(mesh) {
        if (!recording) fail('STATE', 'PMREMGenerator rendered outside a generation');
        if (!target) fail('STATE', 'PMREM draws need a render target');
        recording.push(record(target, mesh));
      },
      begin() { recording = []; }, end() { const r = recording; recording = null; return r; },
    };
  }
  function program(material, mesh) {
    // Render targets: no tone mapping, linear output (WebGLRenderer.getProgram).
    const sources = threeProgramSources(T, material, mesh, {toneMapping: T.NoToneMapping, outputColorSpace: T.LinearSRGBColorSpace});
    let entry = programs.get(sources.key);
    if (!entry) {
      const compiled = compileEsslProgram(sources.vertex, sources.fragment, {rows: 'gl'});
      entry = {compiled, pipeline: null};
      programs.set(sources.key, entry);
    }
    return entry;
  }
  /** Snapshot one draw at its source call: values, target rectangle, textures. */
  function record(target, mesh) {
    const material = mesh.material, entry = program(material, mesh), {reflection} = entry.compiled;
    const bytes = new Uint8Array(reflection.uniformBufferSize);
    const textures = packThreeProgramUniforms(T, reflection, material.uniforms, mesh, new T.OrthographicCamera(), bytes);
    const v = target.viewport, s = target.scissor;
    if (!target.scissorTest) fail('STATE', 'PMREM targets draw with their scissor rectangle');
    return {target, entry, bytes, textures, geometry: mesh.geometry, raster: threeProgramRaster(T, material),
      viewport: [v.x, v.y, v.z, v.w], scissor: [s.x, s.y, s.z, s.w]};
  }
  function targetTexture(target) {
    let t = targets.get(target);
    if (!t) {
      if (target.texture.type !== T.HalfFloatType || target.texture.format !== T.RGBAFormat) fail('FORMAT', 'Expected PMREM half-float RGBA targets');
      const texture = device.createTexture({label: `${label}/${target.texture.name}`, size: [target.width, target.height, 1], format: 'rgba16float', usage: 16 | 4});
      t = {texture, view: texture.createView()};
      targets.set(target, t);
    }
    return t;
  }
  function linearSampler() {
    // PMREM targets: LinearFilter min/mag, no mipmaps, ClampToEdge.
    return sampler ??= device.createSampler({label, magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
  }
  async function pipelineOf(entry, raster, geometry) {
    if (entry.pipeline) {
      if (entry.attributeSet !== entry.compiled.reflection.attributes.map(a => !!geometry.attributes[a.name]).join()) fail('FORMAT', 'PMREM geometry attributes changed');
      return entry.pipeline;
    }
    const {reflection} = entry.compiled;
    const uniformLayout = device.createBindGroupLayout({label, entries: [{binding: 0, visibility: VERTEX | FRAGMENT, buffer: {type: 'uniform'}}]});
    const textureLayout = device.createBindGroupLayout({label, entries: reflection.textures.flatMap(t => [
      {binding: t.textureBinding, visibility: VERTEX | FRAGMENT, texture: {sampleType: 'float', viewDimension: t.dimension}},
      {binding: t.samplerBinding, visibility: VERTEX | FRAGMENT, sampler: {type: 'filtering'}}])});
    // Locations the geometry lacks read GL's constant default attribute (0, 0, 0, 1).
    const buffers = reflection.attributes.map(a => geometry.attributes[a.name]
      ? {arrayStride: a.components * 4, stepMode: 'vertex', attributes: [{shaderLocation: a.location, offset: 0, format: a.components === 1 ? 'float32' : `float32x${a.components}`}]}
      : {arrayStride: 0, stepMode: 'vertex', attributes: [{shaderLocation: a.location, offset: 0, format: 'float32x4'}]});
    entry.attributeSet = reflection.attributes.map(a => !!geometry.attributes[a.name]).join();
    const pipeline = await device.createRenderPipelineAsync({label, layout: device.createPipelineLayout({label, bindGroupLayouts: [uniformLayout, textureLayout]}),
      vertex: {module: device.createShaderModule({label: `${label}/vertex`, code: entry.compiled.vertex}), entryPoint: 'f3d_vertex', buffers},
      fragment: {module: device.createShaderModule({label: `${label}/fragment`, code: entry.compiled.fragment}), entryPoint: 'f3d_fragment',
        targets: [{format: 'rgba16float', ...(raster.blend ? {blend: raster.blend} : {}), writeMask: raster.writeMask}]},
      // rows:'gl' mirrors clip Y, so GL's counter-clockwise front faces are clockwise here.
      primitive: {topology: 'triangle-list', cullMode: raster.cullMode, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw'}});
    entry.pipeline = {pipeline, uniformLayout, textureLayout};
    return entry.pipeline;
  }
  function vertexBuffers(geometry, attributes, owned) {
    return attributes.map(a => {
      const source = geometry.attributes[a.name];
      if (!source) return constant ??= (() => {
        const b = device.createBuffer({label: `${label}/default-attribute`, size: 16, usage: 32 | 8});
        device.queue.writeBuffer(b, 0, new Float32Array([0, 0, 0, 1]));
        return b;
      })();
      if (!(source.array instanceof Float32Array) || source.itemSize !== a.components || source.isInterleavedBufferAttribute) fail('FORMAT', 'Expected PMREM float attributes');
      const b = device.createBuffer({label: `${label}/${a.name}`, size: Math.max(4, source.array.byteLength), usage: 32 | 8});
      device.queue.writeBuffer(b, 0, source.array);
      owned.push(b);
      return b;
    });
  }
  /** Run r186 PMREMGenerator for one complete source and execute its passes. */
  async function generate(source) {
    if (disposed) fail('DISPOSED', 'PMREM owner is disposed');
    if (source.isRenderTargetTexture) fail('SOURCE', 'Render-target PMREM sources (pmremVersion) are not admitted yet');
    const kind = pmremSourceKind(T, source);
    if (kind === 'direct') fail('SOURCE', 'Expected an equirectangular or cube environment');
    const host = generator._renderer;
    let target, draws;
    host.begin();
    try { target = kind === 'equirect' ? generator.fromEquirectangular(source) : generator.fromCubemap(source); }
    finally { draws = host.end(); }
    target.texture.pmremVersion = source.pmremVersion;
    const owned = [];
    let submitted = false, failure = null;
    device.pushErrorScope('validation');
    try {
      const stride = draws.reduce((m, d) => Math.max(m, Math.ceil(d.bytes.byteLength / ALIGN) * ALIGN), ALIGN);
      const uniforms = device.createBuffer({label: `${label}/uniforms`, size: stride * draws.length, usage: 64 | 8});
      owned.push(uniforms);
      const staged = new Uint8Array(stride * draws.length);
      draws.forEach((d, i) => staged.set(d.bytes, i * stride));
      device.queue.writeBuffer(uniforms, 0, staged);
      const prepared = [], byTexture = new Map(draws.map(d => [d.target.texture, d.target]));
      for (const [i, d] of draws.entries()) {
        const {pipeline, uniformLayout, textureLayout} = await pipelineOf(d.entry, d.raster, d.geometry);
        const views = d.textures.map(t => {
          if (!t) fail('TEXTURE', 'PMREM pass without its input texture');
          const rt = byTexture.get(t);
          if (rt) return {view: targetTexture(rt).view, sampler: linearSampler()};
          const b = bindingOf(t);
          if (!b?.view || !b.sampler) fail('TEXTURE', 'PMREM source texture has no binding');
          return b;
        });
        const reflection = d.entry.compiled.reflection;
        prepared.push({d, pipeline, buffers: vertexBuffers(d.geometry, reflection.attributes, owned),
          uniformGroup: device.createBindGroup({label, layout: uniformLayout, entries: [{binding: 0, resource: {buffer: uniforms, offset: i * stride, size: reflection.uniformBufferSize}}]}),
          textureGroup: device.createBindGroup({label, layout: textureLayout, entries: reflection.textures.flatMap((t, k) => [
            {binding: t.textureBinding, resource: views[k].view}, {binding: t.samplerBinding, resource: views[k].sampler}])}),
          count: d.geometry.attributes.position.count});
      }
      const encoder = device.createCommandEncoder({label});
      for (const p of prepared) {
        const out = targetTexture(p.d.target), [x, y, w, h] = p.d.viewport, [sx, sy, sw, sh] = p.d.scissor;
        const pass = encoder.beginRenderPass({label, colorAttachments: [{view: out.view, loadOp: 'load', storeOp: 'store'}]});
        // rows:'gl': WebGPU row index equals GL's bottom-up window y, so GL rectangles apply unchanged.
        pass.setViewport(x, y, w, h, 0, 1);
        pass.setScissorRect(sx, sy, sw, sh);
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, p.uniformGroup); pass.setBindGroup(1, p.textureGroup);
        p.buffers.forEach((b, k) => pass.setVertexBuffer(k, b));
        pass.draw(p.count);
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
      submitted = true;
    } catch (error) { failure = error; }
    // Per-generation buffers die after their submitted use, or at once.
    const release = () => { for (const b of owned) b.destroy(); };
    if (submitted) device.queue.onSubmittedWorkDone().then(release, release); else release();
    const scoped = await device.popErrorScope();
    if (failure) throw failure;
    if (scoped) fail('GPU', scoped.message);
    const out = targetTexture(target);
    // Targets PMREMGenerator no longer holds (a resized ping-pong) are released.
    for (const [rt, t] of targets) if (rt !== target && rt !== generator._pingPongRenderTarget && ![...done.values()].some(e => e.target === rt)) { t.texture.destroy(); targets.delete(rt); }
    const entry = {target, binding: {view: out.view, sampler: linearSampler(), sampleType: 'float'}};
    done.set(source, entry); outputs.set(target.texture, entry);
    const onDispose = () => {
      source.removeEventListener('dispose', onDispose);
      if (done.get(source) !== entry) return;
      done.delete(source); outputs.delete(target.texture);
      // Live meshes may still bind it until the next preparation publishes.
      retired.push(target);
      target.dispose();
    };
    source.addEventListener('dispose', onDispose);
    return target.texture;
  }
  return Object.freeze({
    /** {state: 'ready', texture} | {state: 'incomplete'} | {state: 'needed'} | {state: 'direct'} */
    lookup(source) {
      const kind = pmremSourceKind(T, source);
      if (kind === 'direct') return {state: 'direct'};
      const entry = done.get(source);
      if (entry) return {state: 'ready', texture: entry.target.texture};
      if (source.isRenderTargetTexture) fail('SOURCE', 'Render-target PMREM sources (pmremVersion) are not admitted yet');
      return pmremSourceComplete(T, source) ? {state: 'needed'} : {state: 'incomplete'};
    },
    generate,
    /** The native binding of a generated cube-UV texture, or undefined. */
    binding: texture => outputs.get(texture)?.binding,
    /** Destroy textures of disposed sources once no published mesh uses them. */
    collect() { for (const rt of retired.splice(0)) { targets.get(rt)?.texture.destroy(); targets.delete(rt); } },
    dispose() {
      disposed = true;
      for (const t of targets.values()) t.texture.destroy();
      targets.clear(); done.clear(); outputs.clear(); constant?.destroy(); constant = null; generator.dispose();
    },
  });
}
