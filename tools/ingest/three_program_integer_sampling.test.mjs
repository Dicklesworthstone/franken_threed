import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {compileEsslProgram} from './essl_wgsl.mjs';
import {packThreeProgramUniforms} from './three_program.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.0);}';
const fragment = (declarations, body) => '#version 300 es\nprecision highp float;precision highp int;out vec4 color;\n' + declarations + '\nvoid main(){' + body + '}';

test('integer normalized sampling lowers typed 2D, array and volume texels with mip selection', () => {
  for (const prefix of ['i', 'u']) for (const [suffix, coordinate, gradient] of [
    ['2D', 'vec2(.25)', 'vec2(.125)'], ['2DArray', 'vec3(.25,.75,1.6)', 'vec2(.125)'], ['3D', 'vec3(.25)', 'vec3(.125)'],
  ]) {
    const source = fragment(`uniform highp ${prefix}sampler${suffix} image;`,
      `${prefix}vec4 a=texture(image,${coordinate});${prefix}vec4 b=textureLod(image,${coordinate},1.5);` +
      `${prefix}vec4 c=textureGrad(image,${coordinate},${gradient},${gradient});color=vec4(a+b+c)/255.0;`);
    const result = compileEsslProgram(vertex, source);
    assert.equal(result.reflection.textures[0].sampleType, prefix === 'i' ? 'sint' : 'uint');
    assert.equal(result.reflection.textures[0].samplingOffset, 0);
    assert.match(result.fragment, /textureLoad\(image, point/);
    assert.match(result.fragment, /floor\(lod \+ 0.5\)/);
    assert.match(result.fragment, /lod > 0.5/);
    assert.doesNotMatch(result.fragment, /textureSample/);
    if (suffix === '2DArray') assert.match(result.fragment, /textureNumLayers\(image\)/);
  }
});

test('integer sampler function arguments and sampler arrays keep independent control bindings', () => {
  const result = compileEsslProgram(vertex, fragment(
    'uniform highp usampler2D maps[2];uvec4 sampleMap(highp usampler2D item, vec2 uv){return textureLod(item,uv,2.0);}',
    'color=vec4(sampleMap(maps[0],vec2(.2))+sampleMap(maps[1],vec2(.8)))/255.0;'));
  assert.deepEqual(result.reflection.textures.map(t => [t.element, t.samplingOffset]), [[0, 0], [1, 16]]);
  assert.match(result.fragment, /item_sampling: vec4<u32>/);
  assert.match(result.fragment, /maps_0_sampler, f3d_u.f3d_sampling_0/);
  assert.match(result.fragment, /maps_1_sampler, f3d_u.f3d_sampling_1/);
});

test('vertex implicit integer sampling stays at the base level without fragment derivatives', () => {
  const vs = '#version 300 es\nprecision highp float;precision highp int;uniform highp isampler2D data;in vec3 position;' +
    'void main(){gl_Position=vec4(position+vec3(float(texture(data,vec2(.25)).r)*.01),1.0);}';
  const result = compileEsslProgram(vs, fragment('', 'color=vec4(1.0);'));
  assert.match(result.vertex, /f3d_integer_sample_2d_int\(data_texture, f3d_u.f3d_sampling_0, f3d_t\d+, 0.0\)/);
  assert.doesNotMatch(result.vertex, /dpdx|dpdy/);
});

test('program packing snapshots integer sampler controls without disturbing authored uniforms', () => {
  const result = compileEsslProgram(vertex, fragment('uniform float exposure;uniform highp usampler2D maps[2];',
    'color=vec4(texture(maps[0],vec2(.25))+texture(maps[1],vec2(.5)))*exposure;'));
  const a = new T.DataTexture(new Uint32Array(4), 2, 2, T.RedIntegerFormat, T.UnsignedIntType);
  const b = a.clone();
  a.wrapS = T.RepeatWrapping; a.wrapT = T.MirroredRepeatWrapping; a.minFilter = T.NearestMipmapNearestFilter;
  b.wrapS = T.ClampToEdgeWrapping; b.wrapT = T.RepeatWrapping; b.minFilter = T.NearestFilter;
  const bytes = new Uint8Array(result.reflection.uniformBufferSize), object = new T.Mesh(), camera = new T.PerspectiveCamera();
  const textures = packThreeProgramUniforms(T, result.reflection, {maps: {value: [a, b]}, exposure: {value: .75}}, object, camera, bytes);
  assert.deepEqual(textures, [a, b]);
  const controls = i => Array.from(new Uint32Array(bytes.buffer, result.reflection.textures[i].samplingOffset, 4));
  assert.deepEqual(controls(0), [1, 2, 0, 1]); assert.deepEqual(controls(1), [0, 1, 0, 0]);
  assert.equal(new DataView(bytes.buffer).getFloat32(result.reflection.uniforms[0].offset, true), .75);
  const accepted = bytes.slice(); a.wrapS = T.ClampToEdgeWrapping; a.minFilter = T.NearestFilter;
  packThreeProgramUniforms(T, result.reflection, {maps: {value: [a, b]}}, object, camera, bytes);
  assert.deepEqual(controls(0), [0, 2, 0, 0]); assert.notDeepEqual(bytes, accepted);
  a.dispose(); b.dispose(); object.geometry.dispose(); object.material.dispose();
});

test('integer sampling validates source coordinate and derivative signatures before GPU use', () => {
  for (const call of ['texture(map,vec3(.1))','textureLod(map,vec2(.1),1)',
    'textureGrad(map,vec2(.1),vec3(.2),vec3(.2))','texture(map,vec2(.1),1)'])
    assert.throws(() => compileEsslProgram(vertex, fragment('uniform highp usampler2D map;', `color=vec4(${call});`)));
  assert.throws(() => compileEsslProgram(vertex, fragment('uniform highp isamplerCube map;',
    'color=vec4(texture(map,vec3(.1)));')), /layered cube view/);
});

test('installed sampler descriptors govern integer addressing after unrequested public edits', () => {
  const result = compileEsslProgram(vertex, fragment('uniform highp usampler2D image;', 'color=vec4(texture(image,vec2(1.25)));'));
  const texture = new T.DataTexture(new Uint32Array(4), 2, 2, T.RedIntegerFormat, T.UnsignedIntType);
  const object = new T.Mesh(), camera = new T.PerspectiveCamera(), bytes = new Uint8Array(result.reflection.uniformBufferSize);
  const samplerDescriptor = Object.freeze({addressModeU: 'repeat', addressModeV: 'mirror-repeat', lodMaxClamp: 3});
  texture.wrapS = texture.wrapT = T.ClampToEdgeWrapping; texture.minFilter = T.NearestFilter;
  packThreeProgramUniforms(T, result.reflection, {image: {value: texture}}, object, camera, bytes,
    {textureBindings: [{samplerDescriptor}]});
  assert.deepEqual(Array.from(new Uint32Array(bytes.buffer, result.reflection.textures[0].samplingOffset, 4)), [1, 2, 0, 1]);
  assert.throws(() => packThreeProgramUniforms(T, result.reflection, {image: {value: texture}}, object, camera, bytes,
    {textureBindings: [{}]}), /bound sampler descriptor/);
  const fetch = compileEsslProgram(vertex, fragment('uniform highp usampler2D image;', 'color=vec4(texelFetch(image,ivec2(0),0));'));
  assert.doesNotThrow(() => packThreeProgramUniforms(T, fetch.reflection, {image: {value: texture}}, object, camera,
    new Uint8Array(fetch.reflection.uniformBufferSize), {textureBindings: [{}]}));
  texture.dispose(); object.geometry.dispose(); object.material.dispose();
});
