/** Pinned public postprocessing passes with owned deformation programs.
 * Recording GPU bindings prove accepted poses, atlas bytes and uniforms;
 * these tests do not rasterize pixels or assert browser performance. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import * as T from '../../upstream/three.js/build/three.module.js';
import {WebGLMorphtargets} from '../../upstream/three.js/src/renderers/webgl/WebGLMorphtargets.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {createThreeProgramSupport} from './three_program.mjs';
import {transformThreePostProcessing} from './postprocessing_contracts.mjs';
import {renderThreePassScene} from './three_pass_state.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

// Use the compiler's existing exact-source transforms, including TAA's SSAA
// superclass. The renderer, scene bridge and shader owners are not replaced.
const passURLs = new Map();
async function pinnedPass(name) {
  const url = new URL(`../../upstream/three.js/examples/jsm/postprocessing/${name}.js`, import.meta.url);
  const source = await readFile(url, 'utf8');
  const code = transformThreePostProcessing(source, {name: `${name}.js`, transform: true,
    sourceHash: 'sha256:' + createHash('sha256').update(source).digest('hex')});
  const resolved = code.replace(/from '([^']+)'/g, (_, specifier) => `from ${JSON.stringify(specifier === 'three'
    ? new URL('../../upstream/three.js/build/three.module.js', import.meta.url).href
    : passURLs.get(new URL(specifier, url).href) ?? new URL(specifier, url).href)}`);
  const moduleURL = 'data:text/javascript;base64,' + Buffer.from(resolved + `\n//# sourceURL=f3d-test-${name}.mjs`).toString('base64');
  passURLs.set(url.href, moduleURL); return (await import(moduleURL))[name];
}
const SSAARenderPass = await pinnedPass('SSAARenderPass');
const TAARenderPass = await pinnedPass('TAARenderPass');
const OutlinePass = await pinnedPass('OutlinePass');

async function fixture(t, options = {}) {
  const device = geometryDevice(); device.features = new Set(['float32-filterable']);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = ({texture}, data) => {
    texture.pixels = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes) for (const draw of pass.draws) {
      draw.texturePixels = new Map();
      for (const {group} of draw.groups.values()) for (const entry of group.entries ?? []) {
        const texture = entry.resource?.texture;
        if (!texture) continue;
        assert.notEqual(texture.destroyed, true, 'each submitted sampler retains its accepted GPU resource');
        if (texture.pixels) draw.texturePixels.set(texture, texture.pixels.slice());
      }
    }
    submit(commands);
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [32, 32], format: configuration.format, usage: 16}),
  })};
  const Renderer = createWebGLRendererClass(T, {maxDraws: 64, maxBindings: 64, maxPendingRenders: 128, ...options});
  const renderer = new Renderer({canvas, device}); await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 5;
  const read = new T.WebGLRenderTarget(32, 32, {type: T.HalfFloatType}), write = read.clone();
  const programs = createThreeProgramSupport({three: T,
    state: () => ({toneMapping: T.NoToneMapping, outputColorSpace: T.LinearSRGBColorSpace})});
  t.after(async () => { renderer.dispose(); await renderer._disposePromise; programs.dispose(); read.dispose(); write.dispose(); });
  return {device, renderer, scene, camera, read, write,
    reflection: (object, material = object.material) => programs.compile(material, object).program.reflection};
}

function animatedMesh(t, {skin = false, morph = false, relative = false} = {}) {
  const geometry = new T.PlaneGeometry(1, 2), material = new T.MeshNormalMaterial({side: T.DoubleSide});
  const position = geometry.attributes.position;
  let target;
  if (morph) {
    target = position.clone(); geometry.morphTargetsRelative = relative;
    for (let i = 0; i < target.count; i++) target.setXYZ(i,
      relative ? .3 : position.getX(i) + .3, relative ? .1 : position.getY(i) + .1, 0);
    geometry.morphAttributes.position = [target];
    geometry.morphAttributes.normal = [geometry.attributes.normal.clone()];
  }
  if (skin) {
    const indices = [], weights = [];
    for (let i = 0; i < position.count; i++) {
      const weight = (position.getY(i) + 1) / 2;
      indices.push(0, 1, 0, 0); weights.push(1 - weight, weight, 0, 0);
    }
    geometry.setAttribute('skinIndex', new T.Uint16BufferAttribute(indices, 4));
    geometry.setAttribute('skinWeight', new T.Float32BufferAttribute(weights, 4));
  }
  const mesh = skin ? new T.SkinnedMesh(geometry, material) : new T.Mesh(geometry, material);
  let root, tip, skeleton;
  if (skin) {
    root = new T.Bone(); tip = new T.Bone(); root.position.y = -1; tip.position.y = 2; root.add(tip);
    skeleton = new T.Skeleton([root, tip]); mesh.add(root); mesh.bind(skeleton);
  }
  mesh.frustumCulled = false;
  t.after(() => { geometry.dispose(); material.dispose(); skeleton?.dispose(); });
  return {mesh, geometry, material, root, tip, skeleton, target};
}

function animatedInstances(t, relative = false) {
  const source = animatedMesh(t, {morph: true, relative});
  const mesh = new T.InstancedMesh(source.geometry, source.material, 3); mesh.frustumCulled = false;
  const weights = values => {
    values.forEach((value, index) => {
      source.mesh.morphTargetInfluences = Array.isArray(value) ? value : [value];
      mesh.setMorphAt(index, source.mesh);
    });
  };
  weights([.125, .5, .875]); mesh.morphTexture.needsUpdate = true;
  t.after(() => mesh.dispose());
  return {...source, mesh, weights};
}

const drain = renderer => renderer.waitForGPU();
const programDraws = device => device.snapshots.flat().filter(draw =>
  /boneTexture|morphTargetInfluences/.test(draw.pipeline.vertex.module.code));
function textureFloats(draw, predicate) {
  const entry = [...draw.texturePixels].find(([texture]) => predicate(texture));
  assert.ok(entry, 'the deformation sampler was uploaded before submission');
  return new Float32Array(entry[1].buffer, entry[1].byteOffset, entry[1].byteLength / 4);
}
const palette = draw => textureFloats(draw, texture => texture.format === 'rgba32float' && texture.width === 4 && texture.height === 4);
const atlas = draw => textureFloats(draw, texture => texture.format === 'rgba32float' && texture.height === 1);
const instanceWeights = draw => textureFloats(draw, texture => texture.format === 'r32float');
function uniformFloats(draw, reflection, name, count = 1) {
  const uniform = reflection.uniforms.find(value => value.name === name); assert.ok(uniform, `source uniform ${name}`);
  const {group, offsets} = draw.groups.get(0), resource = group.entries.find(entry => entry.binding === 0).resource;
  const bytes = draw.contents.get(resource.buffer), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const offset = (resource.offset ?? 0) + (offsets[0] ?? 0) + uniform.offset;
  return Array.from({length: count}, (_, i) => view.getFloat32(offset + i * (uniform.node.k === 'array' ? uniform.node.stride : 4), true));
}
function sourcePalette(mesh) {
  const reference = new T.Skeleton(mesh.skeleton.bones, mesh.skeleton.boneInverses);
  reference.update(); reference.computeBoneTexture();
  const result = reference.boneMatrices.slice(); reference.dispose(); return result;
}
function morphOracle() {
  const values = new Map(), source = WebGLMorphtargets({}, {maxTextureSize: 8192}, {});
  return object => {
    source.update(object, object.geometry, {getUniforms: () => ({setValue(_gl, name, value) { values.set(name, value); }})});
    return {pixels: values.get('morphTargetsTexture').image.data.slice(),
      base: values.get('morphTargetBaseInfluence'), influences: [...values.get('morphTargetInfluences') ?? []],
      weights: values.get('morphTexture')?.image.data.slice()};
  };
}

for (const relative of [false, true]) test(`SSAA retains ${relative ? 'relative' : 'absolute'} per-instance morph uploads from ordinary draws through queued animation and disposal`, async t => {
  const f = await fixture(t), a = animatedInstances(t, relative), reference = morphOracle(); f.scene.add(a.mesh);
  const initial = reference(a.mesh);
  assert.deepEqual([...initial.weights], relative ? [1, .125, 1, .5, 1, .875] : [.875, .125, .5, .5, .125, .875]);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(instanceWeights(programDraws(f.device).at(-1)), initial.weights);
  const start = f.device.snapshots.length;
  const pass = new SSAARenderPass(f.scene, f.camera); pass.sampleLevel = 1; t.after(() => pass.dispose());
  a.weights([.25, .75, .5]); // setMorphAt alone does not request a source upload.
  pass.render(f.renderer, f.write, f.read);
  a.mesh.count = 2; a.mesh.setMatrixAt(0, new T.Matrix4().makeTranslation(.75, 0, 0));
  a.mesh.instanceMatrix.needsUpdate = true; a.mesh.morphTexture.needsUpdate = true;
  const changed = reference(a.mesh); pass.render(f.renderer, f.write, f.read);
  a.weights([.875, .125, .25]); a.target.array.fill(99); a.mesh.instanceMatrix.array.fill(99); a.mesh.count = 1;
  a.mesh.dispose(); a.geometry.dispose(); a.material.dispose(); f.scene.clear();
  assert.equal(a.mesh.morphTexture, null);
  await drain(f.renderer);
  const draws = f.device.snapshots.slice(start).flat().filter(draw => draw.pipeline.vertex.module.code.includes('morphTexture'));
  assert.equal(draws.length, 4);
  for (const [index, draw] of draws.entries()) {
    assert.deepEqual(instanceWeights(draw), index < 2 ? initial.weights : changed.weights);
    assert.deepEqual(atlas(draw), initial.pixels);
    assert.equal(draw.args[1], index < 2 ? 3 : 2);
    const slot = draw.pipeline.vertex.buffers.findIndex(layout => layout.stepMode === 'instance' && layout.arrayStride === 64);
    assert.ok(slot >= 0);
    const matrix = new Float32Array(draw.contents.get(draw.streams.get(slot)).buffer);
    assert.equal(matrix[12], index < 2 ? 0 : .75, 'each sample keeps the matrices accepted alongside its influence rows');
  }
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('Outline retains per-instance morph depth beside source-supported ordinary morph masks', async t => {
  const f = await fixture(t), a = animatedInstances(t), b = animatedMesh(t, {morph: true});
  b.mesh.position.x = .75; b.mesh.morphTargetInfluences[0] = .25; f.scene.add(a.mesh, b.mesh);
  const first = a.mesh.morphTexture.image.data.slice();
  // r186's mask shader omits morphinstance_vertex, so it cannot compile for a
  // selected instanced morph. Keep that source limitation explicit; its depth
  // material supports the source instance shader and captured influence rows.
  const pass = new OutlinePass(new T.Vector2(32, 32), f.scene, f.camera, [b.mesh]); t.after(() => pass.dispose());
  const mask = f.reflection(b.mesh, pass.prepareMaskMaterial);
  pass.render(f.renderer, f.write, f.read);
  a.weights([.75, .25, .125]); a.mesh.morphTexture.needsUpdate = true;
  const changed = a.mesh.morphTexture.image.data.slice(); b.mesh.morphTargetInfluences[0] = .625;
  pass.render(f.renderer, f.write, f.read);
  a.mesh.dispose(); a.geometry.dispose(); b.geometry.dispose(); f.scene.clear(); await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, 4);
  assert.deepEqual(draws.map(draw => draw.pipeline.fragment.module.code.includes('depthTest')), [false, true, false, true]);
  assert.deepEqual(instanceWeights(draws[0]), first); assert.deepEqual(instanceWeights(draws[2]), changed);
  assert.deepEqual(uniformFloats(draws[1], mask, 'morphTargetInfluences'), [.25]);
  assert.deepEqual(uniformFloats(draws[3], mask, 'morphTargetInfluences'), [.625]);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('TAA retains accepted per-instance morph rows across hold and accumulation renders', async t => {
  const f = await fixture(t), a = animatedInstances(t, true); f.scene.add(a.mesh);
  const pass = new TAARenderPass(f.scene, f.camera); pass.sampleLevel = 0; pass.accumulate = true; t.after(() => pass.dispose());
  const first = a.mesh.morphTexture.image.data.slice(); pass.render(f.renderer, f.write, f.read);
  a.weights([.75, .25, .625]); a.mesh.morphTexture.needsUpdate = true;
  const second = a.mesh.morphTexture.image.data.slice(); pass.render(f.renderer, f.write, f.read);
  a.mesh.dispose(); a.geometry.dispose(); f.scene.clear(); await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, 3);
  [first, first, second].forEach((expected, index) => assert.deepEqual(instanceWeights(draws[index]), expected));
  assert.equal(pass.accumulateIndex, 2); assert.equal(f.renderer._passInstances.pendingBytes, 0);
});

test('instance morph weights keep one upload history across required passes, ordinary draws and target-count changes', async t => {
  const f = await fixture(t), a = animatedInstances(t), reference = morphOracle(); f.scene.add(a.mesh); f.renderer.setRenderTarget(f.write);
  const initial = reference(a.mesh); renderThreePassScene(f.renderer, f.scene, f.camera); await drain(f.renderer);
  a.weights([.75, .25, .5]); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(instanceWeights(programDraws(f.device).at(-1)), initial.weights,
    'ordinary rendering sees the version captured by the preceding pass');
  a.mesh.morphTexture.needsUpdate = true; const next = reference(a.mesh);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(instanceWeights(programDraws(f.device).at(-1)), next.weights);
  a.weights([.25, .875, .125]); renderThreePassScene(f.renderer, f.scene, f.camera);
  const oldTexture = a.mesh.morphTexture;
  for (const targets of Object.values(a.geometry.morphAttributes)) targets.push(targets[0].clone());
  a.geometry.morphAttributes.position[1].array[0] = 7;
  a.mesh.morphTexture = null; a.weights([[.25, .5], [.125, .75], [.5, .125]]); a.mesh.morphTexture.needsUpdate = true;
  const expanded = reference(a.mesh); renderThreePassScene(f.renderer, f.scene, f.camera);
  oldTexture.dispose(); a.mesh.dispose(); a.geometry.dispose(); f.scene.clear(); await drain(f.renderer);
  const draws = programDraws(f.device);
  assert.equal(draws.length, 5);
  assert.deepEqual(instanceWeights(draws[3]), next.weights); assert.deepEqual(atlas(draws[3]), initial.pixels);
  assert.deepEqual(instanceWeights(draws[4]), expanded.weights); assert.deepEqual(atlas(draws[4]), expanded.pixels);
  assert.equal(f.renderer._passInstances.pendingBytes, 0);
  assert.equal(f.renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0, 'disposed original textures and drained snapshots release all CPU inputs');
});

test('rejected instance morph FIFO capture does not become the next source upload baseline', async t => {
  const f = await fixture(t, {maxPendingRenders: 1}), a = animatedInstances(t); f.scene.add(a.mesh); f.renderer.setRenderTarget(f.write);
  const first = a.mesh.morphTexture.image.data.slice(); renderThreePassScene(f.renderer, f.scene, f.camera);
  const before = {...f.renderer._sampledTextureInputs.diagnostics};
  a.weights([.25, .75, .5]); a.mesh.morphTexture.needsUpdate = true;
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_LIMIT'});
  assert.deepEqual(f.renderer._sampledTextureInputs.diagnostics, before);
  a.weights([.875, .125, .375]); const accepted = a.mesh.morphTexture.image.data.slice();
  await drain(f.renderer); renderThreePassScene(f.renderer, f.scene, f.camera); await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, 2);
  assert.deepEqual(instanceWeights(draws[0]), first); assert.deepEqual(instanceWeights(draws[1]), accepted);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('one instance morph texture can be shared by multiple captured meshes', async t => {
  const f = await fixture(t), a = animatedInstances(t), b = new T.InstancedMesh(a.geometry, a.material, 3);
  b.frustumCulled = false; b.morphTexture = a.mesh.morphTexture; t.after(() => b.dispose());
  f.scene.add(a.mesh, b); f.renderer.setRenderTarget(f.write);
  const first = a.mesh.morphTexture.image.data.slice(); renderThreePassScene(f.renderer, f.scene, f.camera);
  a.weights([.25, .875, .625]); b.morphTexture.needsUpdate = true;
  const second = a.mesh.morphTexture.image.data.slice(); renderThreePassScene(f.renderer, f.scene, f.camera);
  a.weights([.75, .125, .5]); renderThreePassScene(f.renderer, f.scene, f.camera);
  a.mesh.dispose(); b.dispose(); f.scene.clear(); await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, 6);
  [first, first, second, second, second, second].forEach((expected, index) => assert.deepEqual(instanceWeights(draws[index]), expected));
  assert.equal(f.renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0);
});

test('a previously uploaded morph alias keeps its own version gate while capturing the shared Source history', async t => {
  const f = await fixture(t), a = animatedInstances(t), b = new T.InstancedMesh(a.geometry, a.material, 3);
  b.frustumCulled = false; b.morphTexture = a.mesh.morphTexture.clone(); t.after(() => b.dispose());
  f.scene.add(a.mesh); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  a.weights([.25, .875, .625]); b.morphTexture.needsUpdate = true;
  const expected = b.morphTexture.image.data.slice();
  f.scene.remove(a.mesh); f.scene.add(b); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.notEqual(a.mesh.morphTexture.version, b.morphTexture.version);
  a.weights([.75, .125, .5]); f.scene.remove(b); f.scene.add(a.mesh);
  renderThreePassScene(f.renderer, f.scene, f.camera);
  a.mesh.dispose(); b.dispose(); f.scene.clear(); await drain(f.renderer);
  assert.deepEqual(instanceWeights(programDraws(f.device).at(-1)), expected);
});

test('distinct simultaneous morph aliases refuse source draw-order effects before capture', async t => {
  const f = await fixture(t), a = animatedInstances(t), b = new T.InstancedMesh(a.geometry, a.material, 3);
  b.frustumCulled = false; b.morphTexture = a.mesh.morphTexture.clone(); b.renderOrder = -1; t.after(() => b.dispose());
  f.scene.add(a.mesh, b);
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), /Distinct instance morph aliases/);
  f.scene.remove(b);
  const material = new T.ShaderMaterial({uniforms: {otherTexture: {value: b.morphTexture}}});
  const probe = new T.Mesh(new T.PlaneGeometry(), material); f.scene.add(probe);
  t.after(() => { probe.geometry.dispose(); material.dispose(); });
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), /Instance morph material\/background aliases/);
  f.scene.remove(probe); f.scene.background = a.mesh.morphTexture;
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), /Instance morph material\/background aliases/);
  f.scene.background = null;
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('ordinary instance morph inputs keep the automatic texture owner callback boundary', async t => {
  const f = await fixture(t), a = animatedInstances(t); f.scene.add(a.mesh);
  let calls = 0; a.mesh.morphTexture.onUpdate = () => { calls++; };
  f.renderer.render(f.scene, f.camera);
  await assert.rejects(() => drain(f.renderer), {code: 'THREE_SCENE_HOOK'});
  assert.equal(calls, 0);
});

test('ordinary instance morph draws retain separately resident material inputs and required capture refuses that ownership', async t => {
  const f = await fixture(t), a = animatedInstances(t);
  const material = new T.ShaderMaterial({uniforms: {weights: {value: a.mesh.morphTexture}},
    vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform sampler2D weights; void main() { gl_FragColor = texture2D(weights, vec2(0.5)); }'});
  const geometry = new T.PlaneGeometry(), probe = new T.Mesh(geometry, material); probe.frustumCulled = false;
  t.after(() => { material.dispose(); geometry.dispose(); });
  f.scene.add(probe); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  f.scene.add(a.mesh); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(programDraws(f.device).length, 1, 'ordinary reuse keeps the existing sampled owner');
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('hidden, layered, culled and invisible-material instances do not seed influence uploads', async t => {
  const f = await fixture(t), a = animatedInstances(t); f.scene.add(a.mesh); f.renderer.setRenderTarget(f.write);
  a.mesh.visible = false; renderThreePassScene(f.renderer, f.scene, f.camera);
  a.mesh.visible = true; a.mesh.layers.set(1); renderThreePassScene(f.renderer, f.scene, f.camera);
  a.mesh.layers.set(0); a.mesh.frustumCulled = true; a.mesh.boundingSphere = new T.Sphere(new T.Vector3(), 1);
  a.mesh.position.x = 1000; renderThreePassScene(f.renderer, f.scene, f.camera);
  a.mesh.position.x = 0; a.material.visible = false; renderThreePassScene(f.renderer, f.scene, f.camera);
  assert.equal(f.renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0);
  await drain(f.renderer); assert.equal(programDraws(f.device).length, 0);
  a.weights([.25, .875, .625]); a.material.visible = true;
  const first = a.mesh.morphTexture.image.data.slice(); renderThreePassScene(f.renderer, f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(instanceWeights(programDraws(f.device).at(-1)), first,
    'the first selected draw reads current bytes even without a second needsUpdate');
});

test('instance morph captures refuse effectful descriptors and shader collisions before application callbacks', async t => {
  const f = await fixture(t), a = animatedInstances(t); f.scene.add(a.mesh); f.renderer.setRenderTarget(f.write);
  const texture = a.mesh.morphTexture; let calls = 0;
  for (const [object, key] of [[texture, 'onUpdate'], [texture, 'addEventListener'], [texture.image, 'data']]) {
    const previous = Object.getOwnPropertyDescriptor(object, key);
    Object.defineProperty(object, key, key === 'data' ? {get() { calls++; return new Float32Array(6); }, configurable: true} :
      {value() { calls++; }, writable: true, configurable: true});
    assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
    if (previous) Object.defineProperty(object, key, previous); else delete object[key];
  }
  texture.image.width = 1;
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
  texture.image.width = 2;
  const override = new T.ShaderMaterial({uniforms: {morphTexture: {value: texture}}}); t.after(() => override.dispose());
  f.scene.overrideMaterial = override;
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
  f.scene.overrideMaterial = null;
  assert.equal(calls, 0); assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._materialCaptureValues, 0);
  assert.equal(f.renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0);
  renderThreePassScene(f.renderer, f.scene, f.camera);
  assert.ok(f.renderer._passInstances.pendingBytes > 0);
  f.renderer.dispose(); await f.renderer._disposePromise;
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
  assert.equal(f.renderer._sampledTextureInputs.diagnostics.snapshotBytes, 0);
});

for (const accessor of [false, true]) test(`queued instance morph capture refuses a later ${accessor ? 'image accessor' : 'upload helper'} without invoking it`, async t => {
  const f = await fixture(t), a = animatedInstances(t); f.scene.add(a.mesh);
  renderThreePassScene(f.renderer, f.scene, f.camera);
  const object = accessor ? a.mesh.morphTexture.source : a.mesh.morphTexture, key = accessor ? 'data' : 'clearUpdateRanges';
  const previous = Object.getOwnPropertyDescriptor(object, key); let calls = 0;
  Object.defineProperty(object, key, accessor ? {get() { calls++; return previous.value; }, configurable: true} :
    {value() { calls++; }, writable: true, configurable: true});
  try { await assert.rejects(() => drain(f.renderer), {code: 'F3D_RENDERER_UNSUPPORTED'}); }
  finally { if (previous) Object.defineProperty(object, key, previous); else delete object[key]; }
  assert.equal(calls, 0); assert.equal(programDraws(f.device).length, 0);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

for (const morph of [false, true]) test(`SSAA retains queued ${morph ? 'morphed ' : ''}skinned poses, binds and shared-skeleton timing after disposal`, async t => {
  const f = await fixture(t), a = animatedMesh(t, {skin: true, morph}); f.scene.add(a.mesh);
  if (morph) a.mesh.bindMode = T.DetachedBindMode;
  const sibling = new T.SkinnedMesh(a.geometry, a.material); sibling.bind(a.skeleton, a.mesh.bindMatrix);
  sibling.bindMode = T.DetachedBindMode; sibling.frustumCulled = false; sibling.position.x = .75; f.scene.add(sibling);
  const pass = new SSAARenderPass(f.scene, f.camera); pass.sampleLevel = 1; t.after(() => pass.dispose());
  const reflection = f.reflection(a.mesh), originalIndices = a.geometry.attributes.skinIndex;
  const projections = [], render = f.renderer.render.bind(f.renderer);
  f.renderer.render = (scene, camera) => {
    if (scene === f.scene) projections.push(camera.projectionMatrix.elements.map(Math.fround));
    return render(scene, camera);
  };
  a.tip.rotation.z = .25;
  if (morph) { a.mesh.morphTargetInfluences[0] = .125; sibling.morphTargetInfluences[0] = .625; }
  pass.render(f.renderer, f.write, f.read);
  assert.equal(a.skeleton.boneTexture.version, 2, 'two samples update the shared source skeleton twice, not once per mesh');
  const first = sourcePalette(a.mesh);
  a.tip.rotation.z = -.625; a.mesh.bindMatrix.makeTranslation(.2, .1, 0);
  if (!morph) a.mesh.position.x = .25;
  if (morph) { a.mesh.morphTargetInfluences[0] = .875; sibling.morphTargetInfluences[0] = .25; }
  pass.render(f.renderer, f.write, f.read);
  assert.equal(a.skeleton.boneTexture.version, 4);
  const second = sourcePalette(a.mesh), bind = a.mesh.bindMatrix.elements.map(Math.fround);
  const inverse = a.mesh.bindMatrixInverse.elements.map(Math.fround);
  assert.equal(a.mesh.skeleton, a.skeleton); assert.equal(a.geometry.attributes.skinIndex, originalIndices);
  assert.equal(f.camera.view.enabled, false, 'the source pass restores its public camera after every sample sequence');
  f.camera.position.set(99, 99, 99); f.camera.fov = 15; f.camera.updateProjectionMatrix();
  a.tip.rotation.z = 2; a.skeleton.boneMatrices.fill(99); a.mesh.bindMatrix.identity();
  if (morph) { a.mesh.morphTargetInfluences.fill(99); sibling.morphTargetInfluences.fill(99); a.target.array.fill(99); }
  a.skeleton.dispose(); a.geometry.dispose(); a.material.dispose(); if (!morph) f.scene.clear();
  await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, 8);
  for (const [i, draw] of draws.entries()) {
    assert.deepEqual(palette(draw), i < 4 ? first : second);
    assert.deepEqual(uniformFloats(draw, reflection, 'projectionMatrix', 16), projections[Math.floor(i / 2)]);
  }
  for (const index of [4, 6]) {
    assert.deepEqual(uniformFloats(draws[index], reflection, 'bindMatrix', 16), bind);
    assert.deepEqual(uniformFloats(draws[index], reflection, 'bindMatrixInverse', 16), inverse);
  }
  if (morph) assert.deepEqual(draws.map(draw => uniformFloats(draw, reflection, 'morphTargetInfluences')[0]),
    [.125, .625, .125, .625, .875, .25, .875, .25]);
  assert.equal(a.tip.rotation.z, 2); assert.equal(a.mesh.skeleton, a.skeleton);
  assert.deepEqual(a.mesh.bindMatrixInverse.elements.map(Math.fround), inverse,
    'asynchronous submission never rewalks the attached public scene or recomputes its newer bind state');
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('SSAA automatic LOD retains each selected skinned morph pose and source hysteresis across queued camera changes', async t => {
  const lod = new T.LOD(), oracle = new T.LOD(), sourceUpdate = T.LOD.prototype.update; let updates = 0;
  T.LOD.prototype.update = function (camera) { if (Object.getPrototypeOf(this) === lod) updates++; return sourceUpdate.call(this, camera); };
  t.after(() => { T.LOD.prototype.update = sourceUpdate; });
  const f = await fixture(t), near = animatedMesh(t, {skin: true, morph: true}), far = animatedMesh(t, {skin: true, morph: true});
  lod.addLevel(near.mesh, 0); lod.addLevel(far.mesh, 10, .2); f.scene.add(lod);
  oracle.addLevel(new T.Object3D(), 0); oracle.addLevel(new T.Object3D(), 10, .2);
  const pass = new SSAARenderPass(f.scene, f.camera); pass.sampleLevel = 1; t.after(() => pass.dispose());
  const expected = [], selectedCounts = new Map([[near, 0], [far, 0]]);
  for (const [index, distance] of [5, 12, 9, 7].entries()) {
    f.camera.position.z = distance; f.camera.updateMatrixWorld();
    sourceUpdate.call(oracle, f.camera); sourceUpdate.call(oracle, f.camera);
    near.tip.rotation.z = .1 * (index + 1); far.tip.rotation.z = -.2 * (index + 1);
    near.mesh.morphTargetInfluences[0] = .125 * (index + 1); far.mesh.morphTargetInfluences[0] = .75 - .125 * index;
    pass.render(f.renderer, f.write, f.read);
    const selected = oracle.getCurrentLevel() ? far : near;
    assert.equal(lod.getCurrentLevel(), oracle.getCurrentLevel());
    assert.equal(updates, (index + 1) * 2, 'each source SSAA sample advances LOD exactly once');
    selectedCounts.set(selected, selectedCounts.get(selected) + 2);
    for (const object of [near, far]) assert.equal(object.skeleton.boneTexture?.version ?? 0, selectedCounts.get(object),
      'only the selected level updates its public skeleton');
    const value = {palette: sourcePalette(selected.mesh), influence: selected.mesh.morphTargetInfluences[0], reflection: f.reflection(selected.mesh)};
    expected.push(value, value);
  }
  f.camera.position.z = 99; lod.levels.reverse(); lod._currentLevel = 77; f.scene.clear();
  for (const object of [near, far]) {
    object.mesh.visible = true; object.tip.rotation.z = 2; object.skeleton.boneMatrices.fill(99);
    object.mesh.morphTargetInfluences.fill(99); object.target.array.fill(99);
    object.skeleton.dispose(); object.geometry.dispose(); object.material.dispose();
  }
  await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, expected.length);
  for (const [index, draw] of draws.entries()) {
    assert.deepEqual(palette(draw), expected[index].palette);
    assert.deepEqual(uniformFloats(draw, expected[index].reflection, 'morphTargetInfluences'), [expected[index].influence]);
  }
  assert.equal(updates, 8); assert.equal(lod.getCurrentLevel(), 77); assert.ok(near.mesh.visible && far.mesh.visible);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

for (const relative of [false, true]) test(`TAA preserves ${relative ? 'relative' : 'absolute'} morph atlas history across ordinary draws, count changes and disposal`, async t => {
  const f = await fixture(t), a = animatedMesh(t, {morph: true, relative}), reference = morphOracle(); f.scene.add(a.mesh);
  a.mesh.morphTargetInfluences[0] = .25; f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const first = reference(a.mesh), start = f.device.snapshots.length, expected = [];
  const pass = new TAARenderPass(f.scene, f.camera); pass.sampleLevel = 0; pass.accumulate = true; t.after(() => pass.dispose());
  const capture = repeats => {
    const value = reference(a.mesh), reflection = f.reflection(a.mesh);
    for (let i = 0; i < repeats; i++) expected.push({...value, reflection});
    pass.render(f.renderer, f.write, f.read);
  };
  a.target.array[0] = 77; a.target.needsUpdate = true; a.mesh.morphTargetInfluences[0] = .125; capture(2);
  a.geometry.morphAttributes.position = [a.target.clone()]; a.geometry.morphAttributes.position[0].array[0] = 88;
  a.mesh.morphTargetInfluences[0] = .75; capture(1);
  assert.deepEqual(expected[0].pixels, first.pixels); assert.deepEqual(expected[2].pixels, first.pixels);
  for (const list of Object.values(a.geometry.morphAttributes)) list.push(list[0].clone());
  a.mesh.updateMorphTargets(); a.mesh.morphTargetInfluences.splice(0, 2, .2, .6); capture(1);
  a.geometry.dispose(); a.geometry.morphAttributes.position[0].array[0] = 99;
  a.mesh.morphTargetInfluences.splice(0, 2, .5, .25); capture(1);
  assert.equal(expected[3].pixels[0], 88); assert.equal(expected[4].pixels[0], 99);
  a.mesh.morphTargetInfluences.fill(99); a.geometry.dispose(); a.material.dispose(); f.scene.clear();
  await drain(f.renderer);
  const draws = f.device.snapshots.slice(start).flat().filter(draw => draw.pipeline.vertex.module.code.includes('morphTargetInfluences'));
  assert.equal(draws.length, expected.length);
  for (const [index, draw] of draws.entries()) {
    const value = expected[index]; assert.deepEqual(atlas(draw), value.pixels);
    assert.deepEqual(uniformFloats(draw, value.reflection, 'morphTargetInfluences', value.influences.length), value.influences.map(Math.fround));
    assert.deepEqual(uniformFloats(draw, value.reflection, 'morphTargetBaseInfluence'), [Math.fround(value.base)]);
  }
  assert.equal(pass.accumulateIndex, 4); assert.equal(f.renderer._passInstances.pendingBytes, 0);
});

test('Outline retains animated depth and selected-mask draws while public selection and poses change', async t => {
  const f = await fixture(t), skin = animatedMesh(t, {skin: true}), morph = animatedMesh(t, {morph: true});
  morph.mesh.position.x = .75; f.scene.add(skin.mesh, morph.mesh);
  const pass = new OutlinePass(new T.Vector2(32, 32), f.scene, f.camera, [skin.mesh]); t.after(() => pass.dispose());
  skin.tip.rotation.z = .3; morph.mesh.morphTargetInfluences[0] = .125;
  pass.render(f.renderer, f.write, f.read);
  const first = sourcePalette(skin.mesh);
  assert.equal(skin.skeleton.boneTexture.version, 1, 'the selected skeleton updates only in the mask render');
  skin.tip.rotation.z = -.6; morph.mesh.morphTargetInfluences[0] = .75; pass.selectedObjects = [morph.mesh];
  pass.render(f.renderer, f.write, f.read);
  const second = sourcePalette(skin.mesh);
  assert.equal(skin.skeleton.boneTexture.version, 2, 'the next selection updates that skeleton only in the depth render');
  assert.equal(f.scene.overrideMaterial, null); assert.ok(skin.mesh.visible && morph.mesh.visible);
  const depth = f.reflection(morph.mesh, pass.depthMaterial), mask = f.reflection(morph.mesh, pass.prepareMaskMaterial);
  skin.tip.rotation.z = 99; morph.mesh.morphTargetInfluences[0] = 99;
  skin.skeleton.dispose(); skin.geometry.dispose(); morph.geometry.dispose(); f.scene.clear();
  await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, 4);
  assert.deepEqual(draws.map(draw => [draw.pipeline.vertex.module.code.includes('boneTexture'),
    draw.pipeline.fragment.module.code.includes('depthTest')]), [[false, false], [true, true], [true, false], [false, true]]);
  assert.deepEqual(palette(draws[1]), first); assert.deepEqual(palette(draws[2]), second);
  assert.deepEqual(uniformFloats(draws[0], depth, 'morphTargetInfluences'), [.125]);
  assert.deepEqual(uniformFloats(draws[3], mask, 'morphTargetInfluences'), [.75]);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('Outline LOD follows the retained pass visibility changes before source hysteresis selection', async t => {
  const f = await fixture(t), near = animatedMesh(t, {morph: true}), far = animatedMesh(t, {morph: true}), lod = new T.LOD();
  near.mesh.morphTargetInfluences[0] = .25; far.mesh.morphTargetInfluences[0] = .75;
  lod.addLevel(near.mesh, 0); lod.addLevel(far.mesh, 10, .2); f.scene.add(lod);
  const pass = new OutlinePass(new T.Vector2(32, 32), f.scene, f.camera, [near.mesh]); t.after(() => pass.dispose());
  const referenceScene = new T.Scene(), referenceLod = new T.LOD(), camera = f.camera.clone();
  const a = new T.Mesh(near.geometry, near.material), b = new T.Mesh(far.geometry, far.material);
  a.name = 'near'; b.name = 'far'; referenceLod.addLevel(a, 0); referenceLod.addLevel(b, 10, .2); referenceScene.add(referenceLod);
  const referencePass = new OutlinePass(new T.Vector2(32, 32), referenceScene, camera, [a]); t.after(() => referencePass.dispose());
  const expected = [];
  // Execute the retained pass itself, including its visibility cache restores.
  // The reference renderer only records source projection; no candidate capture
  // helper participates in this selection oracle.
  const recorder = {autoClear: true, getClearColor: color => color.set(0), getClearAlpha: () => 1,
    setClearColor() {}, setRenderTarget() {}, clear() {}, render(root, currentCamera) {
      if (root !== referenceScene) return;
      root.updateMatrixWorld(); currentCamera.updateMatrixWorld();
      const selected = [];
      root.traverseVisible(object => {
        if (!object.layers.test(currentCamera.layers)) return;
        if (object.isLOD && object.autoUpdate) object.update(currentCamera);
        if (object.isMesh) selected.push(object.name);
      });
      expected.push(selected);
    }};
  for (const distance of [5, 12, 9]) {
    f.camera.position.z = camera.position.z = distance;
    referencePass.render(recorder, f.write, f.read); pass.render(f.renderer, f.write, f.read);
    assert.equal(lod.getCurrentLevel(), referenceLod.getCurrentLevel());
    assert.deepEqual([near.mesh.visible, far.mesh.visible], [a.visible, b.visible]);
  }
  assert.deepEqual(expected, [['near'], ['near'], ['far'], ['far'], ['far'], ['near']],
    'temporarily hiding the far level removes its hysteresis before the mask draw');
  const depth = f.reflection(near.mesh, pass.depthMaterial), mask = f.reflection(near.mesh, pass.prepareMaskMaterial);
  near.mesh.morphTargetInfluences.fill(99); far.mesh.morphTargetInfluences.fill(99); f.scene.clear();
  await drain(f.renderer);
  const draws = programDraws(f.device); assert.equal(draws.length, expected.length);
  for (const [index, draw] of draws.entries()) assert.deepEqual(
    uniformFloats(draw, index % 2 ? mask : depth, 'morphTargetInfluences'), [expected[index][0] === 'near' ? .25 : .75]);
  assert.equal(f.renderer._passInstances.pendingBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('required animated captures update projected skeletons before material visibility and skip hidden, layered and culled meshes', async t => {
  const f = await fixture(t), a = animatedMesh(t, {skin: true}); f.scene.add(a.mesh); f.renderer.setRenderTarget(f.write);
  a.tip.rotation.z = .4; a.mesh.visible = false; renderThreePassScene(f.renderer, f.scene, f.camera);
  a.mesh.visible = true; a.mesh.layers.set(1); renderThreePassScene(f.renderer, f.scene, f.camera);
  a.mesh.layers.set(0); a.mesh.frustumCulled = true; a.mesh.boundingSphere = new T.Sphere(new T.Vector3(), 1);
  a.mesh.position.x = 1000; renderThreePassScene(f.renderer, f.scene, f.camera);
  assert.ok(a.skeleton.boneMatrices.every(value => value === 0)); assert.equal(a.skeleton.boneTexture, null);
  a.mesh.position.x = 0; a.material.visible = false; renderThreePassScene(f.renderer, f.scene, f.camera);
  assert.deepEqual(a.skeleton.boneMatrices, sourcePalette(a.mesh).subarray(0, a.skeleton.boneMatrices.length));
  assert.ok(a.skeleton.boneMatrices.some(value => value !== 0)); assert.equal(a.skeleton.boneTexture, null);
  await assert.doesNotReject(() => drain(f.renderer), 'projected invisible materials update the source skeleton without preparing a draw');
  assert.equal(programDraws(f.device).length, 0);
  a.material.visible = true; renderThreePassScene(f.renderer, f.scene, f.camera);
  assert.equal(a.skeleton.boneTexture.version, 1, 'the first drawn source mesh creates its palette lazily');
  await assert.doesNotReject(() => drain(f.renderer), 'making the material visible prepares its first indexed program draw');
  assert.equal(programDraws(f.device).length, 1);
});

test('required animated captures reject callbacks before source effects and release accepted state on renderer disposal', async t => {
  const f = await fixture(t), a = animatedMesh(t, {skin: true, morph: true}); f.scene.add(a.mesh); f.renderer.setRenderTarget(f.write);
  let calls = 0;
  a.geometry.attributes.position.addUpdateRange(0, 1); a.geometry.attributes.position.needsUpdate = true;
  for (const [object, method, original] of [[a.mesh, 'onBeforeRender', T.Object3D.prototype.onBeforeRender],
    [a.skeleton, 'update', T.Skeleton.prototype.update]]) {
    object[method] = () => { calls++; };
    assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
    object[method] = original;
  }
  assert.equal(calls, 0); assert.equal(a.skeleton.boneTexture, null);
  assert.ok(a.skeleton.boneMatrices.every(value => value === 0));
  assert.deepEqual(a.geometry.attributes.position.updateRanges, [{start: 0, count: 1}]);
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._materialCaptureValues, 0);
  const pass = new SSAARenderPass(f.scene, f.camera); pass.sampleLevel = 1; t.after(() => pass.dispose());
  pass.render(f.renderer, f.write, f.read);
  const sourceTexture = a.skeleton.boneTexture;
  assert.ok(sourceTexture); assert.ok(f.renderer._passInstances.pendingBytes > 0);
  f.renderer.dispose(); await f.renderer._disposePromise; await Promise.resolve();
  assert.equal(programDraws(f.device).length, 0); assert.equal(f.renderer._passInstances.pendingBytes, 0);
  assert.equal(f.renderer._materialCaptureValues, 0); assert.equal(a.skeleton.boneTexture, sourceTexture);
});
