/** Exact pinned cube graphs and recorded F3D commands; this is not a pixel oracle. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass, createPMREMGeneratorClass} from './three_renderer.mjs';
import {inspectThreeTexture} from './three_textures.mjs';
import {nativeTSLScenarios} from '../../tests/e2e/webgpu_renderer/scenarios_native_tsl.js';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const N = THREE.TSL, Renderer = createWebGPURendererClass(THREE, {maxDraws: 64, maxBindings: 64});
function cube(size = 2, value = 1, type = THREE.UnsignedByteType) {
  const Scalar = type === THREE.HalfFloatType ? Uint16Array : Uint8Array;
  const faces = Array.from({length: 6}, (_, i) => new THREE.DataTexture(
    new Scalar(size * size * 4).fill(value + i), size, size, THREE.RGBAFormat, type));
  const texture = new THREE.CubeTexture(faces); texture.type = type;
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.generateMipmaps = false; texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true; return texture;
}
function rig(t) {
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    coordinateSystem: THREE.WebGPUCoordinateSystem,
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: new THREE.NodeFrame()}, info: {calls: 1},
    _getSampledTextureInputs: () => ({layoutOf({texture}) {
      const shape = inspectThreeTexture(texture, THREE);
      return {texture: {sampleType: shape.sampleType, viewDimension: shape.viewDimension, multisampled: false}};
    }})};
  const owner = createThreeNodePrograms({three: THREE, renderer}); t.after(() => owner.dispose());
  const material = new THREE.NodeMaterial(); material.lights = false;
  material.envMapRotation = new THREE.Euler(.125, .25, .5);
  const object = new THREE.Mesh(new THREE.PlaneGeometry(), material), scene = new THREE.Scene();
  scene.add(object); scene.updateMatrixWorld();
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const context = {object, material, scene, camera, renderId: 1, frameId: 1};
  return {renderer, owner, material, object, scene, camera, context,
    compile: () => owner.compile(material, object, context)};
}
const sampled = draw => draw.textures.filter(binding => binding.isSampledTexture);
test('CPU cube graphs retain pinned WGSL, orientation and cube bind-group dimensions', t => {
  const f = rig(t), source = cube(), node = N.cubeTexture(source, N.vec3(.25, .5, 1)).level(0);
  f.material.colorNode = node.rgb;
  const descriptor = f.compile();
  assert.match(descriptor.fragment, /texture_cube<f32>/);
  assert.match(descriptor.fragment, /textureSampleLevel/);
  const builder = new THREE.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
  const draw = f.owner.captureDraw(descriptor, f.context), binding = sampled(draw)[0];
  assert.equal(binding.texture, source);
  assert.deepEqual(descriptor.groups[binding.group].entries[binding.binding].texture,
    {sampleType: 'float', viewDimension: 'cube', multisampled: false});
  assert.equal(draw.textures.filter(binding => binding.isSampler && !binding.isSampledTexture).length, 1);
  assert.ok(draw.uniforms.every(bytes => [...new Float32Array(bytes.buffer)].every(Number.isFinite)));
});
test('compatible cube replacements reuse bindings while alias and mapping changes require a new graph', t => {
  const f = rig(t), a = cube(), b = cube(2, 11), c = cube(2, 21);
  const first = N.cubeTexture(a, N.vec3(1, 0, 0)), second = N.cubeTexture(a, N.vec3(0, 1, 0));
  f.material.colorNode = first.add(second).rgb;
  const descriptor = f.compile(); assert.equal(sampled(f.owner.captureDraw(descriptor, f.context)).length, 1);
  first.value = b; second.value = b; assert.equal(f.compile(), descriptor);
  assert.equal(sampled(f.owner.captureDraw(descriptor, {...f.context, renderId: 2}))[0].texture, b);
  second.value = c;
  assert.throws(() => f.owner.captureDraw(descriptor, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const separate = f.compile(); assert.notEqual(separate, descriptor);
  assert.equal(sampled(f.owner.captureDraw(separate, {...f.context, renderId: 3})).length, 2);
  c.mapping = THREE.CubeRefractionMapping;
  assert.throws(() => f.owner.captureDraw(separate, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.notEqual(f.compile(), separate);
});
test('native cube admission refuses non-platform faces, malformed cubes, unsampled loads and mismatched node views', t => {
  const f = rig(t), good = cube();
  const cases = [
    () => N.cubeTexture(new THREE.CubeTexture(Array.from({length: 6}, () => ({width: 2, height: 2}))), N.vec3(1, 0, 0)),
    () => N.cubeTexture(new THREE.CubeTexture(good.image.slice(0, 5)), N.vec3(1, 0, 0)),
    () => N.cubeTexture(good, N.vec3(1, 0, 0)).setSampler(false),
    () => N.texture(good),
    () => N.cubeTexture(good.image[0], N.vec3(1, 0, 0)),
  ];
  for (const make of cases) {
    f.material.colorNode = make();
    assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  }
  f.material.colorNode = N.vec3(1); f.material.map = good;
  assert.throws(() => f.compile(), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});
function recording(t) {
  const device = geometryDevice(); device.textures = []; device.textureWrites = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => device.textureWrites.push({
    texture: destination.texture, level: destination.mipLevel ?? 0, layer: destination.origin?.[2] ?? 0,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const material = new THREE.NodeMaterial(); material.lights = false;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), material));
  return {device, renderer, material, scene, camera};
}
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
test('queued native cube draws preserve six face and authored mip histories through the shared texture owner', async t => {
  const f = recording(t); await f.renderer.init();
  const source = cube(); source.mipmaps = [cube(1, 21)]; source.minFilter = THREE.LinearMipmapLinearFilter;
  f.material.colorNode = N.cubeTexture(source, N.vec3(.25, .5, 1)).level(1)
    .mul(N.uniform(1).setName('cubeSurface'));
  f.renderer.render(f.scene, f.camera);
  source.image.forEach((face, i) => face.image.data.fill(11 + i));
  source.mipmaps[0].image.forEach((face, i) => face.image.data.fill(31 + i));
  source.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  source.image.forEach(face => face.image.data.fill(99));
  source.mipmaps[0].image.forEach(face => face.image.data.fill(99));
  await drain(f.renderer);
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.fragment.module.code.includes('cubeSurface'));
  assert.equal(draws.length, 2);
  const first = views(draws[0]).find(view => view.dimension === 'cube');
  const second = views(draws[1]).find(view => view.dimension === 'cube');
  assert.ok(first); assert.ok(second); assert.notEqual(first.texture, second.texture);
  assert.equal(first.texture.depthOrArrayLayers, 6); assert.equal(first.texture.mipLevelCount, 2);
  for (const [view, base, mip] of [[first, 1, 21], [second, 11, 31]]) {
    const writes = f.device.textureWrites.filter(write => write.texture === view.texture);
    assert.equal(writes.length, 12);
    for (let level = 0; level < 2; level++) for (let layer = 0; layer < 6; layer++) {
      const write = writes.find(write => write.level === level && write.layer === layer);
      assert.ok(write); assert.ok(write.bytes.every(value => value === (level ? mip : base) + layer));
    }
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
test('public native fromCubemap filters its source graph and retains queued conversion inputs through disposal', async t => {
  const f = recording(t); await f.renderer.init();
  const Generator = createPMREMGeneratorClass(THREE), generator = new Generator(f.renderer);
  assert.ok(generator instanceof THREE.PMREMGenerator);
  assert.equal(generator.fromCubemap, THREE.PMREMGenerator.prototype.fromCubemap);
  const source = cube(16, 0x3400, THREE.HalfFloatType), target = generator.fromCubemap(source);
  assert.ok(target instanceof THREE.RenderTarget); assert.equal(target.texture.isPMREMTexture, true);
  assert.equal(target.texture.mapping, THREE.CubeUVReflectionMapping);
  f.material.colorNode = N.vec4(N.pmremTexture(target.texture, N.vec3(0, 0, 1), N.float(.375)))
    .mul(N.uniform(1).setName('cubeAtlasSurface'));
  f.renderer.render(f.scene, f.camera);
  source.image.forEach(face => face.image.data.fill(0));
  generator.dispose(); await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
  const draws = f.device.snapshots.flat(), conversions = draws.filter(draw =>
    draw.pipeline.fragment.module.code.includes('texture_cube<f32>'));
  assert.equal(conversions.length, 1);
  const view = views(conversions[0]).find(view => view.dimension === 'cube');
  assert.ok(view);
  const writes = f.device.textureWrites.filter(write => write.texture === view.texture);
  assert.equal(writes.length, 6);
  for (let layer = 0; layer < 6; layer++) {
    const write = writes.find(write => write.layer === layer);
    assert.ok([...new Uint16Array(write.bytes.buffer)].every(value => value === 0x3400 + layer));
  }
  const consumers = draws.filter(draw => draw.pipeline.fragment.module.code.includes('cubeAtlasSurface'));
  assert.equal(consumers.length, 1);
  assert.ok(views(consumers[0]).some(view => view.texture.format === 'rgba16float' &&
    view.texture.width === 336 && view.texture.height === 64));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('registered cube scenario combines six source cube views with a disposed-generator PBR atlas', async t => {
  const f = recording(t); await f.renderer.init();
  const result = nativeTSLScenarios.native_tsl_cube_textures({
    ...THREE, PMREMGenerator: createPMREMGeneratorClass(THREE),
  }, f.renderer);
  t.after(() => result.atlas.dispose());
  f.renderer.render(result.scene, result.camera); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.map(draws => draws.filter(draw =>
    !draw.pipeline.label?.startsWith('Animation HDR output '))).filter(draws => draws.length).at(-1);
  assert.equal(draws.length, 8);
  const panels = draws.filter(draw => draw.pipeline.fragment.module.code.includes('texture_cube<f32>'));
  assert.equal(panels.length, 6);
  assert.ok(panels.every(draw => views(draw).some(view => view.dimension === 'cube' && view.texture.depthOrArrayLayers === 6)));
  const surfaces = draws.filter(draw => draw.pipeline.fragment.module.code.includes('fn roughnessToMip'));
  assert.equal(surfaces.length, 2);
  assert.ok(surfaces.every(draw => views(draw).some(view => view.texture.format === 'rgba16float' &&
    view.texture.width === result.atlas.width && view.texture.height === result.atlas.height && !view.texture.destroyed)));
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
