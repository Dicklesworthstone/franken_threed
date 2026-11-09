import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {compileEsslProgram} from './essl_wgsl.mjs';
import {createGpuThreeTextures, createThreeTextureBindings} from './three_textures.mjs';
import {device} from './tests/volume_texture_test_fixture.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.0);}';
const fragment = (body, declarations) => '#version 300 es\nprecision highp float;precision highp int;out vec4 color;\n' + declarations + '\nvoid main(){' + body + '}';
const compile = (body, declarations = 'uniform highp usamplerCube image;') => compileEsslProgram(vertex, fragment(body, declarations));

test('integer cube sampling retains source dimensions and declares typed layered GPU images', () => {
  for (const prefix of ['i', 'u']) for (const call of ['texture(image,vec3(.2,.3,1.))',
    'texture(image,vec3(.2,.3,1.),1.5)', 'textureLod(image,vec3(.2,.3,1.),2.)']) {
    const result = compile(`color=vec4(${call});`, `uniform highp ${prefix}samplerCube image;`);
    assert.equal(result.reflection.integerSampling, true);
    assert.deepEqual(result.reflection.textures.map(t => [t.dimension, t.viewDimension, t.sampleType]),
      [['cube', '2d-array', prefix === 'i' ? 'sint' : 'uint']]);
    assert.match(result.fragment, new RegExp(`var image_texture: texture_2d_array<${prefix}32>`));
    assert.match(result.fragment, /textureLoad\(image, point, i32\(face\), level\)/);
    assert.doesNotMatch(result.fragment, /texture_cube|textureSample|nearest_index|control\.[xyz]/);
  }
});

test('cube implicit derivatives project before mip selection and source argument effects run once', () => {
  const {fragment: wgsl} = compile('color=vec4(texture(image,point(),clamp(bias(),-2.,2.)));',
    'uniform highp usamplerCube image;int calls=0;vec3 point(){calls++;return vec3(.2,.3,1.);}' +
    'float bias(){calls++;return .5;}');
  assert.equal((wgsl.match(/= point\(\);/g) ?? []).length, 1);
  assert.equal((wgsl.match(/bias\(\)/g) ?? []).length, 2, 'one declaration and one call');
  assert.match(wgsl, /\.xy - f3d_t\d+ \* f3d_t\d+\.z\) \* \(0.5 \/ f3d_t\d+\.z\)/);
  assert.ok(wgsl.indexOf('= point();') < wgsl.lastIndexOf('bias()'));
  assert.match(wgsl, /f3d_integer_cube_project\(dpdx\(f3d_t\d+\), f3d_t\d+\)/);
  assert.match(wgsl, /f3d_integer_cube_project\(dpdy\(f3d_t\d+\), f3d_t\d+\)/);
});

test('cube sampler arrays, nested helpers and textureSize share the acknowledged layered image', () => {
  const result = compile('color=vec4(tap(images[0])+tap(images[1]));',
    'uniform highp isamplerCube images[2];ivec4 sampleMap(highp isamplerCube a,vec3 p){' +
    'return textureLod(a,p,float(textureSize(a,0).x)/8.);}' +
    'ivec4 tap(highp isamplerCube a){return sampleMap(a,vec3(1.,.2,.3));}');
  assert.deepEqual(result.reflection.textures.map(t => t.samplingOffset), [0, 16]);
  assert.match(result.fragment, /a_texture: texture_2d_array<i32>/);
  assert.match(result.fragment, /vec2<i32>\(textureDimensions\(a_texture, 0\)\)/);
  assert.match(result.fragment, /images_0_sampler, f3d_u.f3d_sampling_0/);
  assert.match(result.fragment, /images_1_sampler, f3d_u.f3d_sampling_1/);
});

test('vertex cube implicit sampling stays at base LOD and explicit LOD keeps its source level', () => {
  for (const call of ['texture(image,vec3(1.,.2,.3))', 'textureLod(image,vec3(1.,.2,.3),2.)']) {
    const vs = vertex.replace('void main()', 'uniform highp usamplerCube image;void main()')
      .replace('vec4(position,1.0)', `vec4(position+vec3(float(${call}.r)*.01),1.)`);
    const result = compileEsslProgram(vs, fragment('color=vec4(1.);', ''));
    assert.doesNotMatch(result.vertex, /dpdx|dpdy/);
    assert.match(result.vertex, /f3d_integer_sample_cube_uint/);
    if (call.startsWith('texture(')) assert.match(result.vertex, /f3d_integer_sample_cube_uint\(image_texture, f3d_u.f3d_sampling_0, f3d_t\d+, f3d_t\d+, 0.0\)/);
  }
});

test('cube signatures reject projected/offset forms, wrong derivatives and nonportable bias', () => {
  for (const call of ['texture(image,vec2(.2))', 'textureLod(image,vec3(1.),1)',
    'textureGrad(image,vec3(1.),vec2(.2),vec2(.2))', 'textureProj(image,vec4(1.))',
    'textureGrad(image,vec3(1.),vec3(.2),vec3(.2))',
    'textureOffset(image,vec3(1.),ivec2(1))', 'textureLodOffset(image,vec3(1.),1.,ivec2(1))',
    'texelFetch(image,ivec3(0),0)', 'texture(image,vec3(1.),2.01)'])
    assert.throws(() => compile(`color=vec4(${call});`));
  const vs = vertex.replace('void main()', 'uniform highp usamplerCube image;void main()')
    .replace('vec4(position,1.0)', 'vec4(position+vec3(float(texture(image,vec3(1.),1.).r)),1.)');
  assert.throws(() => compileEsslProgram(vs, fragment('color=vec4(1.);', '')), /only valid in fragment/);
});

const cube = signed => {
  const faces = Array.from({length: 6}, (_, i) => new T.DataTexture(
    new (signed ? Int32Array : Uint32Array)(4).fill((0x81234567 + i * 0x10001) >>> 0), 2, 2));
  const texture = new T.CubeTexture(faces);
  Object.assign(texture, {type: signed ? T.IntType : T.UnsignedIntType, format: T.RedIntegerFormat,
    minFilter: T.NearestFilter, magFilter: T.NearestFilter, generateMipmaps: false});
  texture.needsUpdate = true;
  return texture;
};

test('integer cube alias views share one allocation, source history and disposal', async () => {
  for (const signed of [false, true]) {
    const d = device(), owner = createGpuThreeTextures(d, {three: T}), a = cube(signed), b = a.clone();
    owner.prepare([a, b]); await owner.whenIdle();
    const first = owner.binding(a), alias = owner.binding(b), native = owner.nativeTexture(a);
    assert.equal(first.view.dimension, 'cube'); assert.equal(first.cubeArrayView.dimension, '2d-array');
    assert.equal(first.cubeArrayView.arrayLayerCount, 6); assert.equal(first.cubeArrayView.texture, native);
    assert.equal(first.cubeArrayView, alias.cubeArrayView); assert.equal(d.textures.length, 1);
    const before = native.levels[0].slice();
    a.image[0].image.data.fill(0xffffffff); owner.update([a]);
    assert.deepEqual(native.levels[0], before, 'unmarked writes keep accepted GPU bytes');
    a.needsUpdate = true; owner.update([a]);
    assert.notDeepEqual(native.levels[0], before); assert.equal(owner.binding(a).cubeArrayView, first.cubeArrayView);
    a.dispose(); assert.equal(native.destroyed, false); b.dispose(); assert.equal(native.destroyed, true);
    owner.dispose();
  }
});

test('reflection selects cube aliases in scan, owned and borrowed bindings; defaults keep six layers', async () => {
  const d = device(), owner = createGpuThreeTextures(d, {three: T}), texture = cube(false);
  const reflected = compile('color=vec4(texture(image,vec3(1.)));').reflection.textures[0];
  const state = {three: T, textures: new Map(), autoTextures: true, ownedTextures: () => owner,
    textureScan: new Set(), pendingTextures: new WeakSet(), frameTextures: new Set(),
    fail(code, message) { throw new Error(`${code}: ${message}`); }};
  const bindings = createThreeTextureBindings(d, state);
  assert.equal(bindings.binding(texture, reflected).view, texture); assert.ok(state.textureScan.has(texture));
  owner.prepare([texture]); await owner.whenIdle(); state.textureScan = null;
  const selected = bindings.binding(texture, reflected);
  assert.equal(selected.view, owner.binding(texture).cubeArrayView);
  state.textures.set(texture, owner.binding(texture));
  assert.equal(bindings.binding(texture, reflected).view, selected.view);
  state.textures.set(texture, {...owner.binding(texture), cubeArrayView: undefined});
  assert.throws(() => bindings.binding(texture, reflected), /layered view/);
  const empty = bindings.placeholder(null, reflected);
  assert.equal(empty.view.dimension, '2d-array'); assert.equal(empty.texture.layers, 6);
  assert.equal(bindings.placeholder(null, reflected), empty);
  bindings.dispose(); assert.equal(empty.texture.destroyed, true); owner.dispose();
});
