/** Native Scene lifecycle against actual source objects and recorded GPU bytes.
 * Renderer.js r186 runs Scene-before after matrices/viewport selection, and
 * Scene-after after the accepted draws/output; compilation calls only before. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const draws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
function blocks(draw) {
  const result = [];
  for (const {group, offsets} of draw.groups.values()) {
    const layout = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding);
    for (const [index, entry] of layout.entries()) {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      result.push([...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)]);
    }
  }
  return result;
}
function hasColor(draw, color) {
  return blocks(draw).some(words => words.some((_, i) => color.every((value, j) => words[i + j] === value)));
}
function fixture(t, {linear = true} = {}) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.textures = []; device.textureWrites = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => {
    const pipeline = {...descriptor, getBindGroupLayout: () => ({entries: []})};
    device.pipelines.push(pipeline); return pipeline;
  };
  device.queue.writeTexture = (destination, data) => device.textureWrites.push({destination, data});
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device});
  if (linear) renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100);
  camera.position.z = 4;
  const material = new T.MeshBasicMaterial({color: new T.Color(.125, .375, .625)});
  const geometry = new T.PlaneGeometry(), mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; scene.add(mesh);
  t.after(() => { renderer.dispose(); geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, camera, material, mesh, canvas};
}

test('custom node-material subclasses select source graphs without authored slots and freeze queued uniforms', async t => {
  const f = fixture(t), colors = [[.25, .5, .75], [.625, .375, .125]];
  let setups = 0;
  class TintMaterial extends T.MeshBasicNodeMaterial {
    constructor() { super(); this.tint = T.TSL.uniform(new T.Color(...colors[0])); }
    setupDiffuseColor(builder) {
      setups++; assert.equal(builder.material, this);
      super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.assign(this.tint);
    }
  }
  class InheritedTintMaterial extends TintMaterial {}
  const material = new InheritedTintMaterial(); f.mesh.material = material;
  t.after(() => material.dispose());
  assert.equal(Object.keys(material).some(key => key.endsWith('Node') && material[key] != null), false);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  material.tint.value.setRGB(...colors[1]);
  f.renderer.render(f.scene, f.camera);
  assert.equal(setups, 1, 'a warm graph retains the source material setup');
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 2);
  for (const [index, color] of colors.entries()) assert.ok(hasColor(actual[index], color));
  assert.equal(setups, 1, 'deferred preparation does not replay setup');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const registered of [false, true]) for (const count of [4, 1025])
test(`custom instance morphs retain queued weight textures and ${count === 4 ? 'uniform' : 'attribute'} matrix history (registered=${registered})`, async t => {
  const f = fixture(t);
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256});
  f.device.queue.writeTexture = (destination, data) => f.device.textureWrites.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  let setups = 0;
  class Extension extends T.MeshBasicNodeMaterial {
    constructor() { super(); this.applicationGain = .375; }
    setupOutput(builder, output) {
      setups++;
      return super.setupOutput(builder, T.TSL.vec4(
        output.rgb.mul(T.TSL.materialReference('applicationGain', 'float')), output.a));
    }
  }
  class Application extends T.MeshBasicMaterial {
    constructor() { super(); this.type = 'InstanceMorphApplication'; this.applicationGain = .375; }
  }
  if (registered) f.renderer.library.addMaterial(Extension, 'InstanceMorphApplication');
  const material = new (registered ? Application : Extension)(), geometry = f.mesh.geometry;
  geometry.morphTargetsRelative = true;
  geometry.morphAttributes.position = [0, 1].map(axis => new T.Float32BufferAttribute(
    Array.from(geometry.attributes.position.array, (_, component) => component % 3 === axis ? .25 : 0), 3));
  const mesh = new T.InstancedMesh(geometry, material, count); mesh.frustumCulled = false;
  for (let index = 0; index < count; index++) {
    mesh.setMatrixAt(index, new T.Matrix4().makeTranslation(index + 1, 0, 0));
    mesh.setColorAt(index, new T.Color(.25, .5, .75));
  }
  const setWeights = offset => {
    for (let index = 0; index < count; index++)
      mesh.setMorphAt(index, {morphTargetInfluences: [.125 + offset + index * .125, .375 - offset]});
  };
  setWeights(0); mesh.morphTexture.needsUpdate = true;
  f.scene.remove(f.mesh); f.scene.add(mesh);
  t.after(() => { mesh.dispose(); material.dispose(); });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  setWeights(.5); mesh.morphTexture.needsUpdate = true;
  mesh.instanceMatrix.array[12] = 7; mesh.instanceMatrix.array[28] = 19;
  mesh.instanceMatrix.addUpdateRange(12, 1); mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.array[0] = .625; mesh.instanceColor.array[3] = .875;
  mesh.instanceColor.addUpdateRange(0, 1); mesh.instanceColor.needsUpdate = true;
  material.applicationGain = .875; mesh.count = 3;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  setWeights(99); mesh.count = 2;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  assert.equal(setups, 1);
  mesh.instanceMatrix.array.fill(99); mesh.instanceColor.array.fill(99); material.applicationGain = 99;
  await drain(f.renderer);
  const actual = draws(f.device);
  assert.deepEqual(actual.map(draw => draw.args[1]), [count, 3, 2]);
  const textureViews = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
    .map(entry => entry.resource).filter(resource => resource?.texture);
  const textures = actual.map(draw => textureViews(draw).find(view => view.texture.format === 'r32float').texture);
  const weights = texture => {
    const bytes = f.device.textureWrites.find(write => write.destination.texture === texture).bytes;
    return [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)];
  };
  assert.deepEqual(weights(textures[0]),
    Array.from({length: count}, (_, index) => [1, .125 + index * .125, .375]).flat());
  assert.deepEqual(weights(textures[1]),
    Array.from({length: count}, (_, index) => [1, .625 + index * .125, -.125]).flat());
  assert.notEqual(textures[0], textures[1]); assert.equal(textures[1], textures[2]);
  const streams = draw => draw.pipeline.vertex.buffers.flatMap((layout, slot) => {
    if (layout.stepMode !== 'instance') return [];
    const bytes = draw.contents.get(draw.streams.get(slot));
    return [{stride: layout.arrayStride, values: [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]}];
  });
  assert.deepEqual(actual.map(draw => streams(draw).find(stream => stream.stride === 12).values.slice(0, 4)),
    [[.25, .5, .75, .25], [.625, .5, .75, .25], [.625, .5, .75, .25]]);
  const matrices = actual.map(draw => count === 4 ? blocks(draw).find(words => words.length === count * 16)
    : streams(draw).find(stream => stream.stride === 64).values);
  assert.deepEqual(matrices.map(words => [words[12], words[28]]),
    [[1, 2], [7, count === 4 ? 19 : 2], [7, count === 4 ? 19 : 2]]);
  assert.ok(blocks(actual[0]).flat().includes(.375));
  assert.ok(actual.slice(1).every(draw => blocks(draw).flat().includes(.875)));
  assert.ok(actual.every(draw => draw.pipeline.vertex.module.code.includes('texture_2d_array<f32>')));
  assert.equal(setups, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const registered of [false, true])
test(`transparent lit material extensions retain queued source fog transitions (registered=${registered})`, async t => {
  const f = fixture(t), materialColors = [[.25, .5, .75], [.625, .375, .125], [.75, .25, .5], [.5, .75, .25]],
    fogColors = [[.1875, .3125, .4375], [.6875, .8125, .9375], [.125, .375, .625], [.875, .625, .375]],
    opacity = [.3125, .5625, .8125, .4375];
  let constructions = 0, setups = 0;
  class NativeFogMaterial extends T.MeshStandardNodeMaterial {
    constructor() { super(); constructions++; }
    setupDiffuseColor(builder) { setups++; super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.mulAssign(.5); }
  }
  class SourceFogMaterial extends T.MeshStandardMaterial { constructor() { super(); this.type = 'SourceFogMaterial'; } }
  const material = registered ? new SourceFogMaterial() : new NativeFogMaterial(), light = new T.AmbientLight(0xffffff, .375);
  if (registered) f.renderer.library.addMaterial(NativeFogMaterial, 'SourceFogMaterial');
  material.transparent = true; material.depthWrite = false; f.mesh.material = material; f.scene.add(light);
  t.after(() => { material.dispose(); light.dispose(); });
  const linear = new T.Fog(new T.Color(...fogColors[0]), 1.5, 8.5);
  const exponential = new T.FogExp2(new T.Color(...fogColors[2]), .09375);
  f.scene.fog = linear;
  await f.renderer.init();
  for (let index = 0; index < 4; index++) {
    material.color.setRGB(...materialColors[index]); material.opacity = opacity[index];
    if (index === 1) { linear.color.setRGB(...fogColors[1]); linear.near = 2.5; linear.far = 10.5; }
    if (index === 2) f.scene.fog = exponential;
    if (index === 3) { exponential.color.setRGB(...fogColors[3]); exponential.density = .21875; }
    f.renderer.render(f.scene, f.camera);
    assert.equal(setups, index < 2 ? 1 : 2,
      'source fog values remain live; the new fog graph reconstructs the material on the next render call');
    assert.equal(constructions, registered && index >= 2 ? 2 : 1,
      'source conversion is retained until the fog graph changes, while a NodeMaterial keeps its own instance');
  }
  linear.color.setRGB(99, 99, 99); linear.near = linear.far = 99;
  exponential.color.setRGB(99, 99, 99); exponential.density = 99;
  material.color.setRGB(99, 99, 99); material.opacity = .0625;
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 4);
  const fragments = actual.map(draw => draw.pipeline.fragment.module.code);
  assert.equal(fragments[0], fragments[1]); assert.equal(fragments[2], fragments[3]);
  assert.notEqual(fragments[0], fragments[2]);
  for (const index of [0, 1]) assert.match(fragments[index], /smoothstep\(/);
  for (const index of [2, 3]) assert.match(fragments[index], /exp\(/);
  for (const [index, draw] of actual.entries()) {
    assert.ok(hasColor(draw, materialColors[index])); assert.ok(hasColor(draw, fogColors[index]));
    const values = blocks(draw).flat(); assert.ok(values.includes(opacity[index]));
    for (const value of [[1.5, 8.5], [2.5, 10.5], [.09375], [.21875]][index]) assert.ok(values.includes(value));
    assert.ok(draw.pipeline.fragment.targets[0].blend);
    assert.equal(draw.pipeline.depthStencil.depthWriteEnabled, false);
  }
  assert.equal(setups, 2); assert.equal(constructions, registered ? 2 : 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('registered ordinary material extensions preserve conversion construction and original frame identity', async t => {
  const f = fixture(t), colors = [[.25, .5, .75], [.625, .375, .125]], updates = [];
  let constructions = 0, setups = 0;
  class SourceTintMaterial extends T.MeshBasicMaterial {
    constructor() {
      super(); this.type = 'SourceTintMaterial';
      this.tint = T.TSL.uniform(new T.Color(...colors[0])).onObjectUpdate(frame => {
        updates.push([frame.renderer, frame.scene, frame.camera, frame.object, frame.material]);
      });
    }
  }
  class NativeTintMaterial extends T.MeshBasicNodeMaterial {
    constructor() { super(); constructions++; }
    setupDiffuseColor(builder) {
      setups++; assert.equal(builder.material, material);
      super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.assign(this.tint);
    }
  }
  const material = new SourceTintMaterial(); f.mesh.material = material;
  t.after(() => material.dispose());
  f.renderer.library.addMaterial(NativeTintMaterial, 'SourceTintMaterial');
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  material.tint.value.setRGB(...colors[1]); f.renderer.render(f.scene, f.camera);
  material.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  assert.equal(constructions, 1, 'unchanged source program keys retain the conversion instance');
  assert.equal(setups, 1);
  material.dispose(); f.renderer.render(f.scene, f.camera);
  assert.equal(constructions, 2, 'disposal reconstructs through the source converter');
  assert.equal(setups, 2);
  assert.equal(updates.length, 4);
  for (const frame of updates) assert.deepEqual(frame, [f.renderer, f.scene, f.camera, f.mesh, material]);
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 4);
  assert.ok(hasColor(actual[0], colors[0]));
  for (const draw of actual.slice(1)) assert.ok(hasColor(draw, colors[1]));
  assert.equal(constructions, 2); assert.equal(updates.length, 4);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const registered of [false, true])
test(`transparent material extensions retain source sorting and queued color/opacity (registered=${registered})`, async t => {
  const f = fixture(t), order = []; let setups = 0;
  class HalfColorMaterial extends T.MeshBasicNodeMaterial {
    setupDiffuseColor(builder) {
      setups++; super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.mulAssign(.5);
    }
  }
  class SourceMaterial extends T.MeshBasicMaterial { constructor() { super(); this.type = 'TransparentSourceMaterial'; } }
  if (registered) f.renderer.library.addMaterial(HalfColorMaterial, 'TransparentSourceMaterial');
  const make = (color, opacity) => {
    const material = registered ? new SourceMaterial() : new HalfColorMaterial();
    material.transparent = true; material.depthWrite = false; material.opacity = opacity; material.color.setRGB(...color);
    t.after(() => material.dispose()); return material;
  };
  const nearColors = [[.25, .5, .75], [.625, .375, .125]], farColors = [[.125, .625, .375], [.75, .25, .5]];
  const nearOpacity = [.3125, .8125], farOpacity = [.4375, .6875];
  const near = f.mesh, far = new T.Mesh(f.mesh.geometry, make(farColors[0], farOpacity[0]));
  near.material = make(nearColors[0], nearOpacity[0]); near.position.z = 1; near.name = 'near';
  far.position.z = -1; far.name = 'far'; far.frustumCulled = false;
  const opaque = new T.Mesh(f.mesh.geometry, f.material); opaque.name = 'opaque'; opaque.renderOrder = 100;
  opaque.frustumCulled = false; f.scene.add(far, opaque);
  for (const object of [near, far, opaque]) object.onBeforeRender = () => order.push(object.name);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  near.material.color.setRGB(...nearColors[1]); near.material.opacity = nearOpacity[1]; near.position.z = -2;
  far.material.color.setRGB(...farColors[1]); far.material.opacity = farOpacity[1]; far.position.z = 2;
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(order, ['opaque', 'far', 'near', 'opaque', 'near', 'far'],
    'opaque objects precede transparent depth sorting, including when their renderOrder is larger');
  assert.equal(setups, 1, 'source material references let compatible transparent instances share construction');
  near.material.color.setRGB(1, 0, 0); far.material.color.setRGB(0, 1, 0);
  near.material.opacity = far.material.opacity = .0625;
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 6);
  for (const index of [0, 3]) {
    assert.ok(hasColor(actual[index], [.125, .375, .625]));
    assert.equal(actual[index].pipeline.fragment.targets[0].blend, undefined);
  }
  for (const [index, color, opacity] of [[1, farColors[0], farOpacity[0]], [2, nearColors[0], nearOpacity[0]],
    [4, nearColors[1], nearOpacity[1]], [5, farColors[1], farOpacity[1]]]) {
    assert.ok(hasColor(actual[index], color)); assert.ok(blocks(actual[index]).flat().includes(opacity));
    assert.ok(actual[index].pipeline.fragment.targets[0].blend);
    assert.equal(actual[index].pipeline.depthStencil.depthWriteEnabled, false);
  }
  assert.equal(setups, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const precompile of [false, true])
test(`transparent custom DoubleSide retains per-draw node lifecycle and source compilation sides (precompile=${precompile})`, async t => {
  const f = fixture(t), events = [], setupSides = [], colors = [[.1875, .4375, .6875], [.75, .5, .25]], opacity = [.3125, .8125];
  const observe = (phase, frame) => {
    assert.equal(frame.renderer, f.renderer); assert.equal(frame.scene, f.scene); assert.equal(frame.camera, f.camera);
    assert.equal(frame.object, f.mesh); assert.equal(frame.material, material);
    events.push([phase, frame.material.side, frame.renderer._isPreCompiling === true]);
  };
  class TransparentMaterial extends T.MeshBasicNodeMaterial {
    constructor() {
      super(); this.transparent = true; this.side = T.DoubleSide; this.opacity = opacity[0]; this.color.setRGB(...colors[0]);
      this.phase = T.TSL.uniform(.5).setName('transparentPhase').onObjectUpdate(frame => { observe('update', frame); return .5; });
      this.phase.updateBeforeType = this.phase.updateAfterType = 'object';
      this.phase.updateBefore = frame => observe('before', frame);
      this.phase.updateAfter = frame => observe('after', frame);
    }
    setupDiffuseColor(builder) {
      setupSides.push(builder.material.side); super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.mulAssign(this.phase);
    }
  }
  const material = new TransparentMaterial(); f.mesh.material = material; t.after(() => material.dispose());
  f.mesh.onBeforeRender = () => events.push(['object-before', material.side]);
  f.mesh.onAfterRender = () => events.push(['object-after', material.side]);
  const phases = (side, compiling) => ['before', 'update', 'after'].map(phase => [phase, side, compiling]);
  if (precompile) {
    await f.renderer.compileAsync(f.scene, f.camera);
    assert.deepEqual(events, [['object-before', T.DoubleSide], ['object-after', T.DoubleSide],
      ...phases(T.DoubleSide, true), ...phases(T.DoubleSide, true)],
    'source compilation selects both passes first, then prepares them with the restored public side');
    assert.deepEqual(setupSides, [T.DoubleSide], 'both queued passes share the source DoubleSide builder');
    assert.equal(f.device.submissions.length, 0); events.length = 0;
  } else await f.renderer.init();
  const expected = [['object-before', T.DoubleSide], ...phases(T.BackSide, false),
    ...phases(T.FrontSide, false), ['object-after', T.DoubleSide]];
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, expected, 'one object hook pair surrounds two physical draw lifecycle sequences');
  assert.equal(material.side, T.DoubleSide);
  material.color.setRGB(...colors[1]); material.opacity = opacity[1];
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, [...expected, ...expected]);
  assert.deepEqual(setupSides, precompile ? [T.DoubleSide] : [T.BackSide, T.FrontSide],
    'warm side passes retain their original source construction');
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 4);
  assert.deepEqual(actual.map(draw => [draw.pipeline.primitive.cullMode, draw.pipeline.primitive.frontFace]),
    [['back', 'cw'], ['back', 'ccw'], ['back', 'cw'], ['back', 'ccw']]);
  for (const [index, draw] of actual.entries()) {
    assert.ok(hasColor(draw, colors[Math.floor(index / 2)]));
    assert.ok(blocks(draw).flat().includes(opacity[Math.floor(index / 2)]));
  }
  assert.deepEqual(events, [...expected, ...expected], 'deferred GPU preparation does not repeat application callbacks');
  assert.equal(material.side, T.DoubleSide); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const override of [false, true]) for (const rejectPipeline of [false, true])
test(`custom material compilation prepares each graph after earlier pipeline effects (override=${override}, rejectPipeline=${rejectPipeline})`, async t => {
  const f = fixture(t), events = [], constructed = []; let stage = 0;
  class OrderedMaterial extends T.MeshBasicNodeMaterial {
    constructor(variant) {
      super(); this.variant = variant; this.transparent = true;
      this.phase = T.TSL.uniform(.5).onObjectUpdate(() => {
        events.push(`${variant}:update`); if (variant === 'first') stage = 2; return .5;
      });
      this.phase.updateBeforeType = this.phase.updateAfterType = 'object';
      this.phase.updateBefore = frame => {
        assert.equal(frame.renderer._isPreCompiling, true);
        events.push(`${variant}:before`); if (variant === 'first') stage = 1;
      };
      this.phase.updateAfter = frame => {
        assert.equal(frame.renderer._isPreCompiling, true);
        events.push(`${variant}:after`); if (variant === 'first') stage = 3;
      };
    }
    customProgramCacheKey() { return `ordered-material:${this.variant}`; }
    setupDiffuseColor(builder) {
      constructed.push([this.variant, stage]); events.push(`${this.variant}:setup`);
      super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.mulAssign(this.phase.mul(this.variant === 'first' ? .25 : .75));
    }
  }
  const first = new OrderedMaterial('first'), second = new OrderedMaterial('second');
  f.mesh.material = override ? f.material : first; f.mesh.renderOrder = 0;
  const object = new T.Mesh(f.mesh.geometry, override ? f.material : second);
  object.renderOrder = 1; object.frustumCulled = false; f.scene.add(object);
  t.after(() => { first.dispose(); second.dispose(); });
  for (const [mesh, name, material] of [[f.mesh, 'first', first], [object, 'second', second]]) {
    mesh.onBeforeRender = () => {
      events.push(`${name}:object-before`);
      if (override) f.scene.overrideMaterial = material;
    };
    mesh.onAfterRender = () => events.push(`${name}:object-after`);
  }
  await f.renderer.init();
  const createPipeline = f.device.createRenderPipeline;
  f.device.createRenderPipelineAsync = async descriptor => {
    assert.notEqual(f.renderer._isPreCompiling, true);
    // Native preparation may create several raster variants at this one
    // pipeline boundary; every variant must precede the object's after hook.
    if (events.at(-1) !== 'pipeline') events.push('pipeline');
    if (rejectPipeline) throw Error('custom material pipeline rejected');
    return createPipeline(descriptor);
  };
  const selection = ['first:object-before', 'first:object-after', 'second:object-before', 'second:object-after'];
  const firstPreparation = ['first:setup', 'first:before', 'first:update', 'pipeline'];
  if (rejectPipeline) {
    await assert.rejects(f.renderer.compileAsync(f.scene, f.camera), /custom material pipeline rejected/);
    assert.deepEqual(events, [...selection, ...firstPreparation]);
    assert.deepEqual(constructed, [['first', 0]], 'a failed earlier pipeline leaves the later graph constructor inactive');
    assert.equal(stage, 2, 'completed source updates remain, but after callbacks have not run');
  } else {
    await f.renderer.compileAsync(f.scene, f.camera);
    assert.deepEqual(events, [...selection, ...firstPreparation, 'first:after',
      'second:setup', 'second:before', 'second:update', 'pipeline', 'second:after']);
    assert.deepEqual(constructed, [['first', 0], ['second', 3]],
      'the second source constructor observes the first completed after callback');
  }
  assert.equal(f.device.submissions.length, 0); assert.equal(draws(f.device).length, 0);
  assert.notEqual(f.renderer._isPreCompiling, true); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a registered replacement for an ordinary built-in material selects the native source converter', async t => {
  const f = fixture(t), color = [.625, .125, .875];
  let constructions = 0;
  class RegisteredBasicMaterial extends T.MeshBasicNodeMaterial {
    constructor() { super(); constructions++; this.tint = T.TSL.uniform(new T.Color(...color)); }
    setupDiffuseColor(builder) {
      assert.equal(builder.material, f.material);
      super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.assign(this.tint);
    }
  }
  f.renderer.library.materialNodes.delete('MeshBasicMaterial');
  f.renderer.library.addMaterial(RegisteredBasicMaterial, 'MeshBasicMaterial');
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(constructions, 1); assert.ok(hasColor(draws(f.device)[0], color));
  f.renderer.library.materialNodes.set('MeshBasicMaterial', T.MeshBasicNodeMaterial);
  f.renderer.render(f.scene, f.camera);
  f.material.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  assert.equal(constructions, 1, 'restoring a mapping retains the original source builder');
  f.material.dispose(); f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 4);
  for (const draw of actual.slice(0, 3)) assert.ok(hasColor(draw, color));
  assert.ok(hasColor(actual[3], [.125, .375, .625]), 'disposal selects the restored source conversion');
});

test('new default-material owners retain a shared custom builder after registration restoration', async t => {
  const f = fixture(t), color = [.625, .125, .875];
  let constructions = 0;
  class RegisteredBasicMaterial extends T.MeshBasicNodeMaterial {
    constructor() { super(); constructions++; this.tint = T.TSL.uniform(new T.Color(...color)); }
    setupDiffuseColor(builder) { super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.assign(this.tint); }
  }
  f.renderer.library.materialNodes.set('MeshBasicMaterial', RegisteredBasicMaterial);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  f.renderer.library.materialNodes.set('MeshBasicMaterial', T.MeshBasicNodeMaterial);
  const material = new T.MeshBasicMaterial({color: new T.Color(.5, .375, .25)});
  const scene = new T.Scene(), object = new T.Mesh(f.mesh.geometry, material); object.frustumCulled = false; scene.add(object);
  t.after(() => material.dispose());
  f.renderer.render(scene, f.camera);
  assert.equal(constructions, 1, 'the new owner shares the still-live source builder');
  f.material.dispose(); material.dispose();
  f.renderer.render(scene, f.camera); await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 3);
  assert.ok(hasColor(actual[0], color)); assert.ok(hasColor(actual[1], color));
  assert.ok(hasColor(actual[2], [.5, .375, .25]), 'retiring the final owner releases the old registered graph');
});

test('changing a previously fixed material registration requires retiring its original program', async t => {
  const f = fixture(t), color = [.625, .125, .875]; let constructions = 0;
  class RegisteredBasicMaterial extends T.MeshBasicNodeMaterial {
    constructor() { super(); constructions++; this.tint = T.TSL.uniform(new T.Color(...color)); }
    setupDiffuseColor(builder) { super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.assign(this.tint); }
  }
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  f.renderer.library.materialNodes.set('MeshBasicMaterial', RegisteredBasicMaterial);
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_MATERIAL_CACHE'});
  assert.equal(constructions, 0, 'unsupported fixed-program migration must not execute the new converter');
  f.material.dispose(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(constructions, 1); assert.ok(hasColor(draws(f.device).at(-1), color));
});

test('registered inherited light nodes select native ordinary materials and capture each render once', async t => {
  const f = fixture(t), updates = [], color = [.25, .5, .75];
  class SourceAmbient extends T.AmbientLight {}
  class CameraAmbient extends T.AmbientLightNode {
    constructor(light) { super(light); this.updateType = 'render'; }
    update(frame) {
      super.update(frame); this.color.multiplyScalar(frame.camera.userData.gain);
      updates.push([frame.renderer, frame.scene, frame.camera, frame.material, frame.camera.userData.gain]);
    }
  }
  const material = new T.MeshLambertMaterial(), light = new SourceAmbient(new T.Color(...color));
  const second = new T.Mesh(f.mesh.geometry, material); second.frustumCulled = false;
  f.mesh.material = material; f.scene.add(second, light);
  t.after(() => { material.dispose(); light.dispose(); });
  f.renderer.library.addLight(CameraAmbient, SourceAmbient);
  await f.renderer.init();
  f.camera.userData.gain = 1; f.renderer.render(f.scene, f.camera);
  f.camera.userData.gain = 2; f.renderer.render(f.scene, f.camera);
  assert.deepEqual(updates, [
    [f.renderer, f.scene, f.camera, material, 1],
    [f.renderer, f.scene, f.camera, material, 2]
  ]);
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 4);
  for (const draw of actual.slice(0, 2)) assert.ok(hasColor(draw, color));
  for (const draw of actual.slice(2)) assert.ok(hasColor(draw, color.map(value => value * 2)));
  assert.equal(updates.length, 2, 'asynchronous preparation never replays light updates');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a registered built-in light replacement retains object-frequency updates through ordinary materials', async t => {
  const f = fixture(t), objects = [], color = [.125, .25, .375];
  class ObjectAmbient extends T.AmbientLightNode {
    constructor(light) { super(light); this.updateType = 'object'; this.colorNode.setGroup(T.TSL.objectGroup); }
    update(frame) {
      super.update(frame); this.color.multiplyScalar(frame.object.userData.gain); objects.push(frame.object);
    }
  }
  const material = new T.MeshLambertMaterial(), light = new T.AmbientLight(new T.Color(...color));
  f.mesh.material = material; f.mesh.userData.gain = 1; f.mesh.renderOrder = 0;
  const second = new T.Mesh(f.mesh.geometry, material); second.frustumCulled = false;
  second.userData.gain = 2; second.renderOrder = 1; f.scene.add(second, light);
  t.after(() => { material.dispose(); light.dispose(); });
  f.renderer.library.lightNodes.delete(T.AmbientLight);
  f.renderer.library.addLight(ObjectAmbient, T.AmbientLight);
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(objects, [f.mesh, second]);
  const actual = draws(f.device); assert.equal(actual.length, 2);
  assert.ok(hasColor(actual[0], color)); assert.ok(hasColor(actual[1], color.map(value => value * 2)));
  f.renderer.library.lightNodes.set(T.AmbientLight, T.AmbientLightNode);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(objects, [f.mesh, second, f.mesh, second]);
  assert.ok(hasColor(draws(f.device)[3], color.map(value => value * 2)));
  const next = fixture(t); next.mesh.material = material; next.mesh.userData.gain = 3; next.scene.add(light);
  await next.renderer.init(); next.renderer.render(next.scene, next.camera); await drain(next.renderer);
  assert.equal(objects.at(-1), next.mesh, 'another renderer retains the source light-identity cache');
  assert.ok(hasColor(draws(next.device)[0], color.map(value => value * 3)));
});

for (const Material of [T.MeshStandardNodeMaterial, T.MeshPhysicalNodeMaterial])
test(`custom ${Material.name} combines scene lighting and maps with queued texture and uniform versions`, async t => {
  const f = fixture(t), colors = [[.125, .375, .625], [.75, .5, .25]], pixels = [[32, 96, 192, 255], [224, 160, 64, 255]];
  let setups = 0;
  class TintPBRMaterial extends Material {
    constructor() { super(); this.tint = T.TSL.uniform(new T.Color(...colors[0])); }
    setupDiffuseColor(builder) { setups++; super.setupDiffuseColor(builder); T.TSL.diffuseColor.rgb.mulAssign(this.tint); }
  }
  const material = new TintPBRMaterial(), map = new T.DataTexture(new Uint8Array(pixels[0]), 1, 1);
  const light = new T.AmbientLight(new T.Color(.25, .5, .75));
  material.map = map; map.needsUpdate = true; f.mesh.material = material; f.scene.add(light);
  t.after(() => { material.dispose(); map.dispose(); light.dispose(); });
  f.device.queue.writeTexture = (destination, data) => f.device.textureWrites.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  await f.renderer.init();
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  material.tint.value.setRGB(...colors[1]); map.image.data.set(pixels[1]); map.needsUpdate = true; light.intensity = 2;
  f.renderer._nodes.nodeFrame.update(); f.renderer.render(f.scene, f.camera);
  assert.equal(setups, 1, 'uniform and map upload versions do not reconstruct the source shader');
  await drain(f.renderer);
  const actual = draws(f.device); assert.equal(actual.length, 2);
  assert.ok(hasColor(actual[0], colors[0])); assert.ok(hasColor(actual[1], colors[1]));
  assert.ok(hasColor(actual[0], [.25, .5, .75])); assert.ok(hasColor(actual[1], [.5, 1, 1.5]));
  const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
    .find(entry => entry.resource?.texture)?.resource.texture;
  const textures = actual.map(sampled); assert.ok(textures.every(Boolean));
  const writes = f.device.textureWrites.filter(write => textures.includes(write.destination.texture));
  assert.deepEqual(writes.map(write => [...write.bytes]), pixels);
  for (const [index, write] of writes.entries()) assert.equal(write.destination.texture, textures[index]);
  assert.equal(setups, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('unlit material subclasses observe the complete scene light list at source reconstruction boundaries', async t => {
  const f = fixture(t), selections = [];
  class SceneAwareBasicMaterial extends T.MeshBasicNodeMaterial {
    setup(builder) { selections.push(builder.lightsNode.getLights().map(light => light.id)); return super.setup(builder); }
  }
  const material = new SceneAwareBasicMaterial(), ambient = new T.AmbientLight(), directional = new T.DirectionalLight();
  f.mesh.material = material; f.scene.add(ambient);
  t.after(() => { material.dispose(); ambient.dispose(); directional.dispose(); });
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera);
  f.scene.add(directional); f.renderer.render(f.scene, f.camera);
  f.scene.remove(ambient); f.renderer.render(f.scene, f.camera);
  assert.deepEqual(selections, [[ambient.id], [ambient.id, directional.id], [directional.id]]);
  await drain(f.renderer); assert.equal(draws(f.device).length, 4);
  assert.equal(selections.length, 3, 'pipeline preparation does not repeat scene-sensitive setup');
});

test('custom PBR cache keys run at native draw capture and preserve queued uniform versions', async t => {
  const f = fixture(t), material = new T.MeshStandardMaterial(), events = [];
  f.mesh.material = material; t.after(() => material.dispose());
  let selected = 0;
  const colors = [[.125, .25, .375], [.5, .625, .75]];
  material.customProgramCacheKey = function () {
    assert.equal(this, material); events.push(selected);
    this.color.setRGB(...colors[selected]); return `variant-${selected}`;
  };
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, [0]);
  selected = 1; material.needsUpdate = true;
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, [0, 1, 1], 'comparison and replacement construction are separate source calls');
  await drain(f.renderer);
  assert.deepEqual(events, [0, 1, 1], 'asynchronous GPU preparation never replays a key');
  const actual = draws(f.device); assert.equal(actual.length, 2);
  assert.ok(hasColor(actual[0], colors[0])); assert.ok(hasColor(actual[1], colors[1]));
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(events, [0, 1, 1], 'a warm native render does not invoke the key');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native compilation invokes custom keys after source object selection hooks', async t => {
  const f = fixture(t), events = [];
  const a = new T.MeshStandardMaterial(), b = new T.MeshPhysicalMaterial();
  f.mesh.material = a; f.mesh.renderOrder = 1;
  const second = new T.Mesh(f.mesh.geometry, b); second.frustumCulled = false; second.renderOrder = 2; f.scene.add(second);
  t.after(() => { a.dispose(); b.dispose(); });
  for (const [object, material, name] of [[f.mesh, a, 'a'], [second, b, 'b']]) {
    object.onBeforeRender = () => events.push(`before-${name}`);
    object.onAfterRender = () => events.push(`after-${name}`);
    material.customProgramCacheKey = () => { events.push(`key-${name}`); return name; };
  }
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(events, ['before-a', 'after-a', 'before-b', 'after-b', 'key-a', 'key-b']);
  events.length = 0;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(events, ['before-a', 'after-a', 'before-b', 'after-b']);
  assert.equal(draws(f.device).length, 2);
});

for (const owner of ['material', 'object', 'override']) test(`key comparison ${owner} disposal preserves queued native draws without cache resurrection`, async t => {
  const f = fixture(t), material = owner === 'override' ? new T.MeshStandardNodeMaterial() : new T.MeshStandardMaterial();
  let calls = 0, disposeNow = false;
  if (owner === 'override') f.scene.overrideMaterial = material;
  else f.mesh.material = material;
  t.after(() => material.dispose());
  const disposeOwner = owner === 'material' ? material : owner === 'object' ? f.mesh : f.material;
  material.customProgramCacheKey = function () {
    calls++;
    if (disposeNow) { disposeNow = false; disposeOwner.dispatchEvent({type: 'dispose'}); }
    return 'same';
  };
  await f.renderer.init();
  const colors = [[.125, .25, .375], [.5, .625, .75], [.25, .375, .5]];
  material.color.setRGB(...colors[0]); f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 1);
  disposeNow = true; material.needsUpdate = true; material.color.setRGB(...colors[1]);
  f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 2); assert.equal(f.renderer._nodeProgramSupport.diagnostics.programs, 0);
  material.color.setRGB(...colors[2]); f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 3); assert.equal(f.renderer._nodeProgramSupport.diagnostics.programs, 1);
  await drain(f.renderer);
  assert.equal(calls, 3, 'deferred preparation does not resurrect or replay the disposed key owner');
  const actual = draws(f.device); assert.equal(actual.length, 3);
  for (const [index, color] of colors.entries()) assert.ok(hasColor(actual[index], color));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native Scene callbacks run synchronously around source object hooks and retain accepted state', async t => {
  const f = fixture(t), events = [], replacement = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial());
  t.after(() => { replacement.geometry.dispose(); replacement.material.dispose(); });
  await f.renderer.init();
  f.scene.onBeforeRender = function(renderer, scene, camera, target) {
    assert.equal(this, f.scene); assert.equal(renderer, f.renderer); assert.equal(scene, f.scene);
    assert.equal(camera, f.camera); assert.equal(target, null);
    assert.equal(camera.matrixWorld.elements[14], 4);
    assert.equal(renderer._nodes.nodeFrame.renderId, 1);
    events.push('scene-before');
    f.material.color.setRGB(.25, .5, .75);
    f.scene.background = new T.Color(.125, .25, .5);
  };
  f.mesh.onBeforeRender = () => events.push('object-before');
  f.mesh.onAfterRender = () => events.push('object-after');
  f.material.onBeforeRender = () => assert.fail('native source material hooks are inert');
  f.scene.onAfterRender = function(renderer, scene, camera, target) {
    assert.equal(this, f.scene); assert.equal(scene, f.scene); assert.equal(camera, f.camera); assert.equal(target, null);
    assert.equal(renderer._nodes.nodeFrame.renderId, 0);
    events.push('scene-after');
    f.material.color.setRGB(1, 0, 0);
    f.mesh.position.x = 20; f.mesh.updateMatrixWorld();
    f.scene.remove(f.mesh); f.scene.add(replacement);
  };
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after', 'scene-after']);
  await drain(f.renderer);
  assert.equal(draws(f.device).length, 1);
  assert.ok(hasColor(draws(f.device)[0], [.25, .5, .75]));
  assert.equal(blocks(draws(f.device)[0]).find(words => words.length === 20)[16], 0);
  assert.deepEqual(f.device.passes[0].desc.colorAttachments[0].clearValue, {r: .125, g: .25, b: .5, a: 1});
  assert.equal(events.length, 4, 'pipeline preparation never replays hooks');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('scene-before position edits do not add an automatic matrix update after the source boundary', async t => {
  const f = fixture(t), observations = [];
  await f.renderer.init();
  f.scene.onBeforeRender = () => {
    observations.push([f.mesh.matrixWorld.elements[12], f.camera.matrixWorld.elements[14]]);
    f.mesh.position.x = 1.5; f.camera.position.z = 8;
  };
  f.renderer.render(f.scene, f.camera);
  f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.deepEqual(observations, [[0, 4], [1.5, 8]]);
  const uniforms = draws(f.device).map(blocks);
  assert.deepEqual(uniforms.map(values => values.find(words => words.length === 20)[16]), [0, 1.5]);
  assert.deepEqual(uniforms.map(values => values.find(words => words.length === 32)[30]), [-4, -8]);
});

test('scene callbacks preserve selected targets and viewport while retaining public selector edits', async t => {
  const f = fixture(t), a = new T.RenderTarget(32, 24), b = new T.RenderTarget(12, 10);
  t.after(() => { a.dispose(); b.dispose(); });
  a.viewport.set(2, 3, 12, 10); a.scissor.set(3, 4, 8, 6); a.scissorTest = true;
  const events = [];
  await f.renderer.init(); f.renderer.setRenderTarget(a);
  f.scene.onBeforeRender = (renderer, scene, camera, target) => {
    events.push(['before', target, renderer.getRenderTarget()]);
    a.viewport.set(0, 0, 32, 24); a.scissor.set(0, 0, 32, 24);
    renderer.setRenderTarget(b);
  };
  f.scene.onAfterRender = (renderer, scene, camera, target) => events.push(['after', target, renderer.getRenderTarget()]);
  f.renderer.render(f.scene, f.camera);
  const request = f.renderer._pending[0];
  assert.equal(request.destination.target, a);
  assert.deepEqual(request.frame.viewport, [2, 3, 12, 10, 0, 1]);
  assert.deepEqual(request.frame.scissor, [3, 4, 8, 6]);
  assert.deepEqual(events, [['before', a, a], ['after', a, b]]);
  assert.equal(f.renderer.getRenderTarget(), b);
  await drain(f.renderer);
  assert.equal(f.device.passes[0].desc.colorAttachments[0].view.texture.width, 32);
  assert.equal(f.renderer.getRenderTarget(), b);
});

for (const initialized of [true, false]) test(`nested Scene before/after renders surround the parent once (initialized=${initialized})`, async t => {
  const f = fixture(t), events = [];
  const child = new T.Scene(), childMaterial = new T.MeshBasicMaterial();
  const childMesh = new T.Mesh(new T.PlaneGeometry(), childMaterial); childMesh.frustumCulled = false; child.add(childMesh);
  t.after(() => { childMesh.geometry.dispose(); childMaterial.dispose(); });
  let phase;
  child.onBeforeRender = () => events.push(phase);
  f.scene.onBeforeRender = renderer => {
    phase = 'before-child'; childMaterial.color.setRGB(.25, .375, .5);
    renderer.render(child, f.camera); events.push('parent-before-end');
  };
  f.scene.onAfterRender = renderer => {
    events.push('parent-after'); phase = 'after-child'; childMaterial.color.setRGB(.5, .625, .75);
    renderer.render(child, f.camera);
  };
  if (initialized) await f.renderer.init();
  if (initialized) f.renderer.render(f.scene, f.camera);
  else await f.renderer.renderAsync(f.scene, f.camera);
  await drain(f.renderer);
  assert.deepEqual(events, ['before-child', 'parent-before-end', 'parent-after', 'after-child']);
  const actual = draws(f.device); assert.equal(actual.length, 3);
  for (const [i, color] of [[0, [.25, .375, .5]], [1, [.125, .375, .625]], [2, [.5, .625, .75]]])
    assert.ok(hasColor(actual[i], color), `draw ${i} keeps its source color`);
  assert.equal(f.renderer._nodes.nodeFrame.renderId, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('the current Scene after-hook is selected after object callbacks', async t => {
  const f = fixture(t), calls = [];
  f.scene.onBeforeRender = () => { f.scene.onAfterRender = () => calls.push('before-installed'); };
  f.mesh.onAfterRender = () => { f.scene.onAfterRender = () => calls.push('object-installed'); };
  f.scene.onAfterRender = () => assert.fail('replaced after-hook');
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.deepEqual(calls, ['object-installed']);
  await drain(f.renderer); assert.deepEqual(calls, ['object-installed']);
});

for (const initialized of [true, false]) test(`a thrown Scene after-hook retains its accepted draw (initialized=${initialized})`, async t => {
  const f = fixture(t), error = new Error('scene after failed');
  let after = 0;
  f.scene.onAfterRender = () => { after++; throw error; };
  if (initialized) {
    await f.renderer.init();
    assert.throws(() => f.renderer.render(f.scene, f.camera), value => value === error);
    await drain(f.renderer);
  } else await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), value => value === error);
  assert.equal(after, 1); assert.equal(draws(f.device).length, 1);
  assert.equal(f.renderer._nodes.nodeFrame.renderId, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a thrown Scene before-hook preserves earlier nested work and never runs the parent after-hook', async t => {
  const f = fixture(t), child = new T.Scene(), material = new T.MeshBasicMaterial();
  child.add(new T.Mesh(new T.PlaneGeometry(), material));
  t.after(() => { child.children[0].geometry.dispose(); material.dispose(); });
  let after = 0; const error = new Error('scene before failed');
  child.onBeforeRender = () => {};
  f.scene.onBeforeRender = renderer => { renderer.render(child, f.camera); f.material.opacity = .25; throw error; };
  f.scene.onAfterRender = () => after++;
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), value => value === error);
  assert.equal(f.material.opacity, .25); assert.equal(after, 0); assert.equal(f.renderer._nodes.nodeFrame.renderId, 0);
  await drain(f.renderer); assert.equal(draws(f.device).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native compile initializes before Scene-before and never invokes Scene-after or duplicates source effects', async t => {
  const f = fixture(t), events = [];
  let initializedFrame;
  const init = f.renderer.init;
  f.renderer.init = async function(...args) {
    const result = await init.apply(this, args); initializedFrame ??= this._nodes.nodeFrame.frameId; return result;
  };
  f.scene.onBeforeRender = (renderer, scene, camera, target) => {
    assert.equal(renderer.initialized, true); assert.equal(scene, f.scene); assert.equal(camera, f.camera);
    assert.equal(target, null); assert.equal(camera.matrixWorld.elements[14], 4);
    assert.equal(renderer._nodes.nodeFrame.renderId, 1); events.push('scene-before');
    assert.equal(renderer._nodes.nodeFrame.frameId, initializedFrame + 1);
    f.material.color.setRGB(.25, .5, .75);
  };
  f.mesh.onBeforeRender = () => events.push('object-before');
  f.mesh.onAfterRender = () => events.push('object-after');
  f.scene.onAfterRender = () => events.push('scene-after');
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after']);
  assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodes.nodeFrame.renderId, 0);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after', 'scene-before', 'object-before', 'object-after', 'scene-after']);
  assert.ok(hasColor(draws(f.device)[0], [.25, .5, .75]));
});

test('a native after-hook can replace accepted geometry before deferred culling without invalidating its draw', async t => {
  const f = fixture(t);
  f.mesh.frustumCulled = true;
  f.scene.onAfterRender = () => { f.mesh.geometry = null; };
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(draws(f.device).length, 1); assert.equal(f.mesh.geometry, null);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('the native source dispatcher is selected before Scene-before and changes apply to the next render', async t => {
  const f = fixture(t); let calls = 0;
  const skip = () => { calls++; };
  f.scene.onBeforeRender = renderer => renderer.setRenderObjectFunction(skip);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 1); assert.equal(f.renderer.getRenderObjectFunction(), skip);
  await drain(f.renderer); assert.equal(draws(f.device).length, 1);
});

for (const initialized of [true, false]) test(`scene canvas changes select the backend surface after pre-hook raster capture (initialized=${initialized})`, async t => {
  const f = fixture(t); let configured;
  const previewCanvas = {width: 24, height: 16, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return f.device.createTexture({size: [24, 16], format: configured.format, usage: 16}); },
  })};
  const preview = new T.CanvasTarget(previewCanvas);
  f.renderer.setViewport(2, 3, 12, 10);
  f.scene.onBeforeRender = renderer => renderer.setCanvasTarget(preview);
  f.mesh.onBeforeRender = renderer => assert.equal(renderer.getCanvasTarget(), preview);
  if (initialized) { await f.renderer.init(); f.renderer.render(f.scene, f.camera); }
  else await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(f.renderer.getCanvasTarget(), preview);
  if (initialized) {
    assert.equal(f.renderer._pending[0].canvasTarget, preview);
    assert.equal(f.renderer._pending[0].canvas, previewCanvas);
    assert.deepEqual(f.renderer._pending[0].canvasSize, [24, 16]);
    assert.deepEqual(f.renderer._pending[0].frame.viewport, [2, 3, 12, 10, 0, 1]);
  }
  await drain(f.renderer);
  assert.equal(f.device.passes[0].desc.colorAttachments[0].view.texture.width, 24);
  assert.equal(f.renderer.getCanvasTarget(), preview);
});

for (const meshRoot of [false, true]) test(`native partial compilation preserves source root and Scene hook owner (meshRoot=${meshRoot})`, async t => {
  const f = fixture(t), target = new T.Scene(), events = [];
  const original = f.mesh.material, lambert = new T.MeshLambertMaterial();
  f.mesh.material = lambert;
  const unused = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial()); target.add(unused, new T.AmbientLight(0xffffff, .25));
  unused.onBeforeRender = () => assert.fail('targetScene contributes lights, not draw objects');
  t.after(() => { unused.geometry.dispose(); unused.material.dispose(); lambert.dispose(); f.mesh.material = original; });
  const root = meshRoot ? f.mesh : f.scene, owner = meshRoot ? target : f.scene;
  owner.onBeforeRender = function(renderer, scene, camera) {
    assert.equal(this, owner); assert.equal(scene, root); assert.equal(camera, f.camera); events.push('scene-before');
  };
  (meshRoot ? f.scene : target).onBeforeRender = () => assert.fail('wrong Scene hook owner');
  f.mesh.onBeforeRender = (renderer, scene) => { assert.equal(scene, owner); events.push('object-before'); };
  f.mesh.onAfterRender = (renderer, scene) => { assert.equal(scene, owner); events.push('object-after'); };
  owner.onAfterRender = () => assert.fail('compilation has no Scene after-hook');
  await f.renderer.compileAsync(root, f.camera, target);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after']);
  assert.equal(f.device.submissions.length, 0);
  assert.ok(f.device.pipelines.some(pipeline => pipeline.vertex?.entryPoint === 'main'),
    'the requested root compiles without accepting the targetScene mesh');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native Scene compilation passes the working target without changing the public selector and retains hook edits', async t => {
  const f = fixture(t, {linear: false}), next = new T.RenderTarget(12, 10);
  t.after(() => next.dispose());
  f.scene.onBeforeRender = (renderer, scene, camera, selected) => {
    assert.equal(selected.isPostProcessingRenderTarget, true);
    assert.equal(renderer.getRenderTarget(), null);
    renderer.setRenderTarget(next);
  };
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.renderer.getRenderTarget(), next); assert.equal(f.device.submissions.length, 0);
});

for (const output of ['canvas', 'target']) test(`Scene hook arguments retain the working target through ${output} output conversion`, async t => {
  const f = fixture(t, {linear: false}), target = output === 'target' ? new T.RenderTarget(20, 12) : null, events = [];
  if (target) { f.renderer.setOutputRenderTarget(target); t.after(() => target.dispose()); }
  f.scene.onBeforeRender = (renderer, scene, camera, selected) => events.push(['before', selected, renderer.getRenderTarget()]);
  f.scene.onAfterRender = (renderer, scene, camera, selected) => events.push(['after', selected, renderer.getRenderTarget()]);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.equal(events.length, 2);
  assert.equal(events[0][1]?.isPostProcessingRenderTarget, true);
  assert.equal(events[0][2], events[0][1]);
  assert.equal(events[1][1], events[0][1]); assert.equal(events[1][2], target);
  await drain(f.renderer);
  assert.equal(events.length, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a bare non-renderable Object3D root does not acquire Scene lifecycle callbacks', async t => {
  const f = fixture(t), root = new T.Group(); root.add(f.mesh);
  root.onBeforeRender = root.onAfterRender = () => assert.fail('root is not a Scene or draw object');
  await f.renderer.init();
  // The existing direct bridge may reject hooks on non-renderable roots; the
  // native Scene adapter must never reinterpret them as source Scene callbacks.
  try { f.renderer.render(root, f.camera); await drain(f.renderer); }
  catch (error) { assert.equal(error.code, 'THREE_SCENE_HOOK'); }
});
