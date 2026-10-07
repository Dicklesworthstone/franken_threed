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

/** r186 WebGLEnvironments.getCube readiness for equirect sources: image height > 0. */
export function cubeSourceComplete(T, texture) { return pmremSourceKind(T, texture) === 'equirect' && !!texture.image && texture.image.height > 0; }

/** bindingOf(texture) -> {view, sampler} for a source texture (the scene's texture owner). */
export function createThreeProgramPMREM({three: T, device, bindingOf, label = 'f3d-pmrem'}) {
  const host = createHost();
  const generator = new T.PMREMGenerator(host);
  const done = new Map(), outputs = new Map(), targets = new Map(), programs = new Map(), retired = [], cubes = new Map();
  let disposed = false, sampler = null, constant = null, mipper = null;

  /** The renderer calls PMREMGenerator and CubeCamera make, recorded in order. */
  function createHost() {
    let target = null, face = 0, recording = null;
    return {
      isWebGLRenderer: true, coordinateSystem: T.WebGLCoordinateSystem,
      autoClear: true, toneMapping: T.NoToneMapping, xr: {enabled: false},
      state: {buffers: {depth: {getReversed: () => false}}},
      getRenderTarget: () => target, getActiveCubeFace: () => face, getActiveMipmapLevel: () => 0,
      setRenderTarget(t, activeCubeFace = 0, activeMipmapLevel = 0) {
        if (activeMipmapLevel !== 0) fail('STATE', 'Rendering into a target mip level is not admitted yet');
        target = t; face = activeCubeFace;
      },
      getClearColor: color => color.setRGB(0, 0, 0), clear() {}, compile() {},
      render(scene, camera) {
        if (!recording) fail('STATE', 'Offscreen draws outside a generation');
        if (!target) fail('STATE', 'Offscreen draws need a render target');
        if (!scene?.isMesh) fail('STATE', 'Only single-mesh offscreen renders are admitted');
        // WebGLRenderer.render: world matrices, then the object's view matrices.
        if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld();
        if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
        scene.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, scene.matrixWorld);
        scene.normalMatrix.getNormalMatrix(scene.modelViewMatrix);
        recording.push(record(target, face, scene, camera, this.autoClear));
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
  function record(target, face, mesh, camera, autoClear) {
    const material = mesh.material, entry = program(material, mesh), {reflection} = entry.compiled;
    const bytes = new Uint8Array(reflection.uniformBufferSize);
    const textures = packThreeProgramUniforms(T, reflection, material.uniforms, mesh, camera, bytes);
    const v = target.viewport, s = target.scissorTest ? target.scissor : target.viewport;
    // Source sampler state at the draw (fromEquirectangularTexture lowers
    // minFilter for the conversion and restores it afterwards).
    const samplers = textures.map(t => t ? samplerFor(t) : null);
    // autoClear clears the target (a scissored clear only touches the drawn rectangle).
    return {target, face, entry, bytes, textures, samplers, geometry: mesh.geometry, raster: threeProgramRaster(T, material),
      viewport: [v.x, v.y, v.z, v.w], scissor: [s.x, s.y, s.z, s.w], clear: autoClear && !target.scissorTest};
  }
  const FILTER = () => new Map([[T.NearestFilter, ['nearest', null]], [T.NearestMipmapNearestFilter, ['nearest', 'nearest']], [T.NearestMipmapLinearFilter, ['nearest', 'linear']],
    [T.LinearFilter, ['linear', null]], [T.LinearMipmapNearestFilter, ['linear', 'nearest']], [T.LinearMipmapLinearFilter, ['linear', 'linear']]]);
  const WRAP = () => new Map([[T.RepeatWrapping, 'repeat'], [T.ClampToEdgeWrapping, 'clamp-to-edge'], [T.MirroredRepeatWrapping, 'mirror-repeat']]);
  const samplerCache = new Map();
  /** WebGLTextures sampler parameters for a texture's current fields. */
  function samplerFor(t) {
    const filters = FILTER(), wraps = WRAP(), [mag] = filters.get(t.magFilter) ?? ['linear'], [min, mip] = filters.get(t.minFilter) ?? ['linear', 'linear'];
    const d = {magFilter: mag, minFilter: min, mipmapFilter: mip ?? 'nearest', addressModeU: wraps.get(t.wrapS) ?? 'clamp-to-edge', addressModeV: wraps.get(t.wrapT) ?? 'clamp-to-edge',
      addressModeW: 'clamp-to-edge', ...(mip ? {} : {lodMaxClamp: 0})};
    const key = JSON.stringify(d);
    let s = samplerCache.get(key);
    if (!s) samplerCache.set(key, s = device.createSampler({label, ...d}));
    return s;
  }
  /** WebGLTextures render-target storage: RGBA of the texture's type; 8-bit sRGB
   * targets are sRGB-encoded storage (SRGB8_ALPHA8); cube targets keep 6 layers. */
  function targetFormat(texture) {
    if (texture.format !== T.RGBAFormat) fail('FORMAT', 'Expected RGBA offscreen targets');
    if (texture.type === T.HalfFloatType) return 'rgba16float';
    if (texture.type === T.FloatType) return 'rgba32float';
    if (texture.type === T.UnsignedByteType) return T.ColorManagement.getTransfer(texture.colorSpace) === T.SRGBTransfer ? 'rgba8unorm-srgb' : 'rgba8unorm';
    fail('FORMAT', 'Unsupported offscreen target type');
  }
  function targetTexture(target) {
    let t = targets.get(target);
    if (!t) {
      const cube = target.isWebGLCubeRenderTarget === true, format = targetFormat(target.texture), tex = target.texture;
      const mips = tex.generateMipmaps && tex.minFilter !== T.NearestFilter && tex.minFilter !== T.LinearFilter ? Math.floor(Math.log2(Math.max(target.width, target.height))) + 1 : 1;
      if (format === 'rgba32float' && !device.features?.has?.('float32-filterable') && tex.minFilter !== T.NearestFilter) fail('FORMAT', 'Linear float32 targets need float32-filterable');
      const texture = device.createTexture({label: `${label}/${tex.name}`, size: [target.width, target.height, cube ? 6 : 1], format, mipLevelCount: mips, usage: 16 | 4});
      t = {texture, format, cube, mips, view: texture.createView({dimension: cube ? 'cube' : '2d'}),
        faces: Array.from({length: cube ? 6 : 1}, (_, f) => texture.createView({dimension: '2d', baseArrayLayer: f, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1}))};
      targets.set(target, t);
    }
    return t;
  }
  function linearSampler() {
    // PMREM targets: LinearFilter min/mag, no mipmaps, ClampToEdge.
    return sampler ??= device.createSampler({label, magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'});
  }
  async function pipelineOf(entry, raster, geometry, format) {
    entry.pipelines ??= new Map();
    const key = format + '|' + JSON.stringify(raster);
    if (entry.pipelines.has(key)) {
      if (entry.attributeSet !== entry.compiled.reflection.attributes.map(a => !!geometry.attributes[a.name]).join()) fail('FORMAT', 'PMREM geometry attributes changed');
      return entry.pipelines.get(key);
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
        targets: [{format, ...(raster.blend ? {blend: raster.blend} : {}), writeMask: raster.writeMask}]},
      // rows:'gl' mirrors clip Y, so GL's counter-clockwise front faces are clockwise here.
      primitive: {topology: 'triangle-list', cullMode: raster.cullMode, frontFace: raster.frontFace === 'ccw' ? 'cw' : 'ccw'}});
    const built = {pipeline, uniformLayout, textureLayout};
    entry.pipelines.set(key, built);
    return built;
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
  /** Replay recorded draws in order as one command buffer, then generate the
   * mips of targets that ask for them (WebGL generateMipmap; box-filtered). */
  async function execute(draws) {
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
        const out = targetTexture(d.target);
        const {pipeline, uniformLayout, textureLayout} = await pipelineOf(d.entry, d.raster, d.geometry, out.format);
        const views = d.textures.map((t, k) => {
          if (!t) fail('TEXTURE', 'Offscreen pass without its input texture');
          const rt = byTexture.get(t);
          if (rt) return {view: targetTexture(rt).view, sampler: linearSampler()};
          const b = bindingOf(t);
          if (!b?.view || !b.sampler) fail('TEXTURE', 'Offscreen source texture has no binding');
          return {view: b.view, sampler: d.samplers[k] ?? b.sampler};
        });
        const reflection = d.entry.compiled.reflection;
        prepared.push({d, out, pipeline, buffers: vertexBuffers(d.geometry, reflection.attributes, owned),
          uniformGroup: device.createBindGroup({label, layout: uniformLayout, entries: [{binding: 0, resource: {buffer: uniforms, offset: i * stride, size: reflection.uniformBufferSize}}]}),
          textureGroup: device.createBindGroup({label, layout: textureLayout, entries: reflection.textures.flatMap((t, k) => [
            {binding: t.textureBinding, resource: views[k].view}, {binding: t.samplerBinding, resource: views[k].sampler}])}),
          index: d.geometry.index, count: d.geometry.index ? d.geometry.index.count : d.geometry.attributes.position.count});
      }
      const encoder = device.createCommandEncoder({label});
      for (const p of prepared) {
        const [x, y, w, h] = p.d.viewport, [sx, sy, sw, sh] = p.d.scissor;
        const pass = encoder.beginRenderPass({label, colorAttachments: [{view: p.out.faces[p.out.cube ? p.d.face : 0],
          loadOp: p.d.clear ? 'clear' : 'load', clearValue: [0, 0, 0, 0], storeOp: 'store'}]});
        // rows:'gl': WebGPU row index equals GL's bottom-up window y, so GL rectangles apply unchanged.
        pass.setViewport(x, y, w, h, 0, 1);
        pass.setScissorRect(sx, sy, sw, sh);
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, p.uniformGroup); pass.setBindGroup(1, p.textureGroup);
        p.buffers.forEach((b, k) => pass.setVertexBuffer(k, b));
        if (p.index) {
          const array = p.index.array, u32 = array instanceof Uint32Array;
          const ib = device.createBuffer({label: `${label}/index`, size: Math.ceil(array.byteLength / 4) * 4, usage: 16 | 8});
          const data = u32 ? array : new Uint16Array(Math.ceil(array.length / 2) * 2);
          if (!u32) data.set(array);
          device.queue.writeBuffer(ib, 0, data, 0);
          owned.push(ib);
          pass.setIndexBuffer(ib, u32 ? 'uint32' : 'uint16'); pass.drawIndexed(p.count);
        } else pass.draw(p.count);
        pass.end();
      }
      for (const out of new Set(prepared.map(p => p.out))) if (out.mips > 1) generateMips(encoder, out);
      device.queue.submit([encoder.finish()]);
      submitted = true;
    } catch (error) { failure = error; }
    // Per-generation buffers die after their submitted use, or at once.
    const release = () => { for (const b of owned) b.destroy(); };
    if (submitted) device.queue.onSubmittedWorkDone().then(release, release); else release();
    const scoped = await device.popErrorScope();
    if (failure) throw failure;
    if (scoped) fail('GPU', scoped.message);
  }
  function generateMips(encoder, out) {
    mipper ??= new Map();
    let m = mipper.get(out.format);
    if (!m) {
      const module = device.createShaderModule({label: `${label}/mips`, code: `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct V { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
  let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return V(vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0), vec2<f32>(xy.x, 1.0 - xy.y));
}
@fragment fn fs(v: V) -> @location(0) vec4<f32> { return textureSampleLevel(src, smp, v.uv, 0.0); }`});
      const pipeline = device.createRenderPipeline({label, layout: 'auto', vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format: out.format}]}});
      m = {pipeline, sampler: device.createSampler({label, minFilter: 'linear', magFilter: 'linear'})};
      mipper.set(out.format, m);
    }
    for (let layer = 0; layer < (out.cube ? 6 : 1); layer++) for (let level = 1; level < out.mips; level++) {
      const view = l => out.texture.createView({dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1, baseMipLevel: l, mipLevelCount: 1});
      const group = device.createBindGroup({layout: m.pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: view(level - 1)}, {binding: 1, resource: m.sampler}]});
      const pass = encoder.beginRenderPass({label, colorAttachments: [{view: view(level), loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store'}]});
      pass.setPipeline(m.pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
    }
  }
  /** Run r186 PMREMGenerator for one complete source and execute its passes. */
  async function generate(source) {
    if (disposed) fail('DISPOSED', 'PMREM owner is disposed');
    if (source.isRenderTargetTexture) fail('SOURCE', 'Render-target PMREM sources (pmremVersion) are not admitted yet');
    const kind = pmremSourceKind(T, source);
    if (kind === 'direct') fail('SOURCE', 'Expected an equirectangular or cube environment');
    let target, draws;
    host.begin();
    try { target = kind === 'equirect' ? generator.fromEquirectangular(source) : generator.fromCubemap(source); }
    finally { draws = host.end(); }
    target.texture.pmremVersion = source.pmremVersion;
    await execute(draws);
    const out = targetTexture(target);
    // Targets PMREMGenerator no longer holds (a resized ping-pong) are released.
    for (const [rt, t] of targets) if (rt !== target && rt !== generator._pingPongRenderTarget && ![...done.values(), ...cubes.values()].some(e => e.target === rt)) { t.texture.destroy(); targets.delete(rt); }
    const entry = {target, binding: {view: out.view, sampler: linearSampler(), sampleType: 'float'}};
    done.set(source, entry); outputs.set(target.texture, entry);
    releaseWith(source, done, entry);
    return target.texture;
  }
  function releaseWith(source, map, entry) {
    const onDispose = () => {
      source.removeEventListener('dispose', onDispose);
      if (map.get(source) !== entry) return;
      map.delete(source); outputs.delete(entry.target.texture);
      // Live meshes may still bind it until the next preparation publishes.
      retired.push(entry.target);
      entry.target.dispose();
    };
    source.addEventListener('dispose', onDispose);
  }
  /** WebGLEnvironments.getCube: r186 WebGLCubeRenderTarget(image.height)
   * .fromEquirectangularTexture(renderer, texture), run unchanged against the
   * recording host (CubeCamera's six face renders), then mipmaps if requested. */
  async function convert(source) {
    if (disposed) fail('DISPOSED', 'PMREM owner is disposed');
    if (!cubeSourceComplete(T, source)) fail('SOURCE', 'Expected a ready equirectangular texture');
    const target = new T.WebGLCubeRenderTarget(source.image.height);
    let draws;
    host.begin();
    try { target.fromEquirectangularTexture(host, source); }
    finally { draws = host.end(); }
    await execute(draws);
    const out = targetTexture(target), tex = target.texture;
    const entry = {target, binding: {view: out.view, sampler: samplerFor(tex), sampleType: 'float'}};
    cubes.set(source, entry); outputs.set(tex, entry);
    releaseWith(source, cubes, entry);
    return tex;
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
    generate, convert,
    /** WebGLEnvironments.getCube for equirect sources: {state: 'ready', texture} |
     * {state: 'incomplete'} | {state: 'needed'}; the ready texture's mapping follows
     * the source's reflection/refraction mapping (mapTextureMapping). */
    lookupCube(source) {
      const entry = cubes.get(source);
      if (entry) {
        entry.target.texture.mapping = source.mapping === T.EquirectangularRefractionMapping ? T.CubeRefractionMapping : T.CubeReflectionMapping;
        return {state: 'ready', texture: entry.target.texture};
      }
      return cubeSourceComplete(T, source) ? {state: 'needed'} : {state: 'incomplete'};
    },
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
