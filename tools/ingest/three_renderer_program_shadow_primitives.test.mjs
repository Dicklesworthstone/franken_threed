// Source WebGL points and wireframe casters through the public WebGPU facade.
// This recorder checks draw/buffer histories; the browser case checks pixels.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

async function fixture(t, object, pointLight = false) {
  const device = geometryDevice(); device.textures = []; device.features = new Set(['float32-filterable']);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = value => ({...value}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = value => ({...value, getBindGroupLayout: () => ({})});
  let configuration;
  const canvas = {width: 32, height: 24, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 24], format: configuration.format, usage: configuration.usage}); }
  })};
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device});
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.BasicShadowMap;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20);
  camera.position.z = 4; object.castShadow = true; object.frustumCulled = false;
  const light = new (pointLight ? T.PointLight : T.DirectionalLight)();
  light.position.set(1, 2, 3); light.castShadow = true; light.shadow.mapSize.set(16, 8);
  scene.add(object, light); await renderer.init();
  t.after(() => { renderer.dispose(); light.shadow.dispose(); object.geometry.dispose();
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) material.dispose();
    object.customDepthMaterial?.dispose(); object.customDistanceMaterial?.dispose(); });
  const render = async () => { renderer.render(scene, camera); await renderer.waitForGPU(); };
  const shadowDraws = () => device.submissions.flat(2).filter(pass => pass.desc.colorAttachments.length === 0).flatMap(pass => pass.draws);
  return {device, renderer, scene, camera, light, render, shadowDraws};
}

for (const pointLight of [false, true]) for (const indexed of [false, true])
test(`${pointLight ? 'point' : 'directional'} light retains ${indexed ? 'indexed' : 'nonindexed'} point shadow order and source hooks`, async t => {
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute([-1, 0, 0, -.5, 0, 0, 0, 0, 0, .5, 0, 0], 3));
  if (indexed) geometry.setIndex([3, 1, 2, 1]);
  geometry.setDrawRange(1, 3);
  const object = new T.Points(geometry, new T.PointsMaterial({size: 3}));
  const custom = new T.ShaderMaterial({vertexShader: 'void main(){ gl_PointSize=5.0; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
    fragmentShader: 'void main(){ if(gl_PointCoord.x<0.25) discard; gl_FragColor=vec4(1.0); }'});
  object[pointLight ? 'customDistanceMaterial' : 'customDepthMaterial'] = custom;
  const f = await fixture(t, object, pointLight), trace = [];
  object.onBeforeShadow = (renderer, caster, camera, shadowCamera, sourceGeometry, material) => {
    assert.equal(renderer, f.renderer); assert.equal(caster, object); assert.equal(camera, f.camera);
    assert.equal(sourceGeometry, geometry); assert.equal(material, custom); trace.push('before');
  };
  object.onAfterShadow = () => trace.push('after');
  await f.render();
  const faces = pointLight ? 6 : 1, perFace = indexed ? [[6, 2, 0, 1], [6, 1, 0, 1]] : [[6, 3, 0, 1]];
  assert.deepEqual(f.shadowDraws().map(draw => draw.args), Array.from({length: faces}, () => perFace).flat());
  assert.deepEqual(trace, Array.from({length: faces}, () => ['before', 'after']).flat());
  for (const draw of f.shadowDraws()) {
    assert.equal(draw.pipeline.primitive.topology, 'triangle-list');
    assert.equal(draw.pipeline.primitive.cullMode, 'none');
    assert.match(draw.pipeline.vertex.module.code, /f3d_PointSize = 5\.0/);
    assert.match(draw.pipeline.fragment.module.code, /f3d_PointCoord/);
  }
});

for (const indexed of [false, true]) for (const instanced of [false, true])
test(`${indexed ? 'indexed' : 'nonindexed'} ${instanced ? 'instanced' : 'ordinary'} wireframe shadows retain edge ranges and source group callbacks`, async t => {
  const plane = new T.PlaneGeometry(), geometry = indexed ? plane : plane.toNonIndexed();
  if (!indexed) plane.dispose();
  geometry.clearGroups(); geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 0); geometry.setDrawRange(1, 4);
  const material = new T.MeshBasicMaterial({wireframe: true, side: T.DoubleSide});
  const object = instanced ? new T.InstancedMesh(geometry, [material], 2) : new T.Mesh(geometry, [material]);
  if (instanced) object.setMatrixAt(1, new T.Matrix4().makeTranslation(.5, 0, 0));
  const f = await fixture(t, object), groups = [];
  object.onBeforeShadow = (_, __, ___, ____, sourceGeometry, depth, group) => {
    assert.equal(sourceGeometry, geometry); assert.equal(depth.wireframe, material.wireframe); groups.push(group);
  };
  await f.render();
  const draws = f.shadowDraws();
  assert.deepEqual(draws.map(draw => [draw.pipeline.primitive.topology, ...draw.args.slice(0, 3)]),
    [['line-list', 4, instanced ? 2 : 1, 2], ['line-list', 4, instanced ? 2 : 1, 6]]);
  assert.deepEqual(groups, geometry.groups);
  const first = draws[0], data = new Uint16Array(first.index.buffer.data);
  assert.deepEqual([...data].slice(0, 12), indexed ? [0, 2, 2, 1, 1, 0, 2, 3, 3, 1, 1, 2] : [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3]);
  assert.equal(first.pipeline.primitive.cullMode, 'none');
  material.wireframe = false; material.needsUpdate = true;
  await f.render();
  assert.equal(f.shadowDraws().at(-1).pipeline.primitive.topology, 'triangle-list');
});

test('indexed point color and shadow passes retain original IDs and requested position uploads', async t => {
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute([-1, 0, 0, -.5, 0, 0, 0, 0, 0, .5, 0, 0], 3));
  geometry.setIndex([2, 0, 2, 1]);
  const object = new T.Points(geometry, new T.PointsMaterial({size: 3}));
  const f = await fixture(t, object), colorDraws = () => f.device.snapshots.flat().filter(draw =>
    draw.pipeline.fragment.targets.length === 1 && draw.pipeline.vertex.module.code.includes('f3d_PointSize'));
  await f.render();
  assert.deepEqual(colorDraws().map(draw => draw.args), [[6, 1, 0, 2], [6, 1, 0, 0], [6, 1, 0, 2], [6, 1, 0, 1]]);
  const initial = colorDraws()[0], original = [...geometry.attributes.position.array];
  const positions = draw => [...new Float32Array(draw.contents.get(draw.streams.get(0)).buffer)].slice(0, 12);
  assert.deepEqual(positions(initial), original);
  geometry.attributes.position.setX(1, .75); geometry.attributes.position.needsUpdate = true;
  await f.render();
  const updated = colorDraws().at(-1); assert.deepEqual(positions(updated), [...geometry.attributes.position.array]);
  assert.equal(updated.streams.get(0), initial.streams.get(0), 'position updates retain native storage');
  geometry.attributes.position.setX(1, -100); await f.render();
  assert.equal(positions(colorDraws().at(-1))[3], .75, 'unrequested CPU edits stay GPU-stale');
  assert.deepEqual(positions(initial), original, 'earlier submitted bytes remain independently observable');
});
