import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram} from './essl_wgsl.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.0);}';
const fragment = (body, declarations) => '#version 300 es\nprecision highp float;precision highp int;out vec4 color;\n' + declarations + '\nvoid main(){' + body + '}';
const compile = (body, declarations = 'uniform sampler2D image;') => compileEsslProgram(vertex, fragment(body, declarations));

test('projected texture families divide coordinates once and preserve explicit gradients and LOD', () => {
  for (const [sampler, coordinates, gradient, offset, swizzle, divisor] of [
    ['sampler2D', 'vec3(.5,.75,2.)', 'vec2(.125)', 'ivec2(1,-1)', 'xy', 'z'],
    ['sampler2D', 'vec4(.5,.75,99.,2.)', 'vec2(.125)', 'ivec2(1,-1)', 'xy', 'w'],
    ['sampler3D', 'vec4(.5,.75,1.,2.)', 'vec3(.125)', 'ivec3(1,-1,0)', 'xyz', 'w'],
  ]) for (const [name, tail, builtin] of [
    ['textureProj', '', 'textureSample'], ['textureProj', ',1.5', 'textureSampleBias'],
    ['textureProjLod', ',1.5', 'textureSampleLevel'], ['textureProjGrad', `,${gradient},${gradient}`, 'textureSampleGrad'],
    ['textureProjOffset', `,${offset}`, 'textureSample'], ['textureProjOffset', `,${offset},1.5`, 'textureSampleBias'],
    ['textureProjLodOffset', `,1.5,${offset}`, 'textureSampleLevel'],
    ['textureProjGradOffset', `,${gradient},${gradient},${offset}`, 'textureSampleGrad'],
  ]) {
    const {fragment: wgsl} = compile(`color=${name}(image,${coordinates}${tail});`, `uniform highp ${sampler} image;`);
    assert.match(wgsl, new RegExp(`${builtin}\\(image_texture, image_sampler, \\(f3d_t\\d+\\.${swizzle} / f3d_t\\d+\\.${divisor}\\)`));
    assert.equal((wgsl.match(/ \/ /g) ?? []).length, 1, name + ' only projects the coordinates');
    if (name.includes('Grad')) assert.match(wgsl, /, vec[23]<f32>\(0.125\), vec[23]<f32>\(0.125\)/);
  }
});

test('offset sampling keeps array layers separate and selects base LOD in vertex programs', () => {
  const declarations = 'uniform highp sampler2DArray image;';
  for (const [expression, builtin] of [
    ['textureOffset(image,vec3(.2,.3,1.6),ivec2(1,-1))', 'textureSample'],
    ['textureOffset(image,vec3(.2,.3,1.6),ivec2(1,-1),2.)', 'textureSampleBias'],
    ['textureLodOffset(image,vec3(.2,.3,1.6),2.,ivec2(1,-1))', 'textureSampleLevel'],
    ['textureGradOffset(image,vec3(.2,.3,1.6),vec2(.1),vec2(.2),ivec2(1,-1))', 'textureSampleGrad'],
  ]) {
    const {fragment: wgsl} = compile(`color=${expression};`, declarations);
    assert.match(wgsl, new RegExp(`${builtin}\\(image_texture, image_sampler, f3d_t\\d+\\.xy, i32\\(floor\\(f3d_t\\d+\\.z \\+ 0.5\\)\\)`));
    assert.match(wgsl, /vec2<i32>\(1, \(-1\)\)/);
  }
  for (const expression of ['textureOffset(image,vec2(.25),ivec2(1))', 'textureProjOffset(image,vec3(.25,.5,2.),ivec2(-1))']) {
    const source = vertex.replace('void main()', 'uniform sampler2D image;void main()').replace('vec4(position,1.0)', expression);
    const {vertex: wgsl} = compileEsslProgram(source, fragment('color=vec4(1.);', ''));
    assert.match(wgsl, /textureSampleLevel\(image_texture, image_sampler, .*, 0.0, vec2<i32>/);
    assert.doesNotMatch(wgsl, /textureSample\(|dpdx|dpdy/);
  }
});

test('texelFetchOffset preserves integer payload types and does not offset array layers', () => {
  for (const prefix of ['', 'i', 'u']) for (const [dimension, coordinate, offset] of [
    ['2D', 'ivec2(2,3)', 'ivec2(-1,1)'], ['3D', 'ivec3(2,3,1)', 'ivec3(-1,1,0)'],
    ['2DArray', 'ivec3(2,3,1)', 'ivec2(-1,1)'],
  ]) {
    const result = compile(`color=vec4(texelFetchOffset(image,${coordinate},0,${offset}));`, `uniform highp ${prefix}sampler${dimension} image;`);
    assert.equal(result.reflection.textures[0].sampleType, prefix === 'i' ? 'sint' : prefix === 'u' ? 'uint' : 'float');
    assert.match(result.fragment, /textureLoad\(image_texture, \(f3d_t\d+(?:\.xy)? \+ vec[23]<i32>/);
    if (dimension === '2DArray') assert.match(result.fragment, /\)\), f3d_t\d+\.z, 0\)/);
    assert.doesNotMatch(result.fragment, /textureSample/);
  }
});

test('named local offsets remain compile-time constants and side effects run once', () => {
  const {fragment: wgsl} = compile(`const ivec2 base=ivec2(stepSize,-stepSize); const ivec2 delta=base.yx+ivec2(1);
    color=textureProjOffset(image,point(),delta,float(++calls));`,
    'uniform sampler2D image;const int stepSize=1;int calls=0;vec3 point(){calls++;return vec3(.2,.4,2.);}');
  assert.match(wgsl, /const base: vec2<i32>/); assert.match(wgsl, /const delta: vec2<i32>/);
  assert.equal((wgsl.match(/= point\(\);/g) ?? []).length, 1);
  assert.match(wgsl, /textureSampleBias\(image_texture, image_sampler, .*, f32\(calls\), delta\)/);
  const captured = compile('color=textureProjOffset(image,vec3(.2,.4,2.),ivec2(1,-1),float(++calls));',
    'uniform sampler2D image;int calls=0;').fragment;
  assert.match(captured, /textureSampleBias\(image_texture, image_sampler, .*, vec2<i32>\(1, \(-1\)\)\)/,
    'offset arguments remain constants even when argument sequencing captures their first translation');
});

test('legacy projected-gradient extension maps to the same typed sampling path', () => {
  const result = compileEsslProgram('attribute vec3 position;void main(){gl_Position=vec4(position,1.);}',
    '#extension GL_EXT_shader_texture_lod : enable\nprecision highp float;uniform sampler2D image;void main(){' +
    'gl_FragColor=texture2DProjGradEXT(image,vec4(.5,.75,99.,2.),vec2(.125),vec2(.25));}');
  assert.match(result.fragment, /textureSampleGrad\(image_texture, image_sampler, \(f3d_t\d+\.xy \/ f3d_t\d+\.w\)/);
});

test('sampling signatures reject invalid coordinates, gradients, offsets, dimensions and vertex bias', () => {
  for (const expression of ['textureOffset(image,vec2(.2))', 'textureOffset(image,vec3(.2),ivec2(1))',
    'textureOffset(image,vec2(.2),vec2(1.))', 'textureLodOffset(image,vec2(.2),1,ivec2(1))',
    'textureProj(image,vec2(.2))', 'textureGradOffset(image,vec2(.2),vec3(.1),vec3(.1),ivec2(1))',
    'textureOffset(image,vec2(.2),ivec2(1),1)', 'texelFetchOffset(image,ivec2(1),1.,ivec2(1))',
    'texelFetchOffset(image,ivec2(1),1,ivec3(1))']) assert.throws(() => compile(`color=${expression};`));
  assert.throws(() => compile('ivec2 offset=ivec2(1);color=textureOffset(image,vec2(.2),offset);'), /constant expression/);
  assert.throws(() => compile('color=textureOffset(image,vec3(.2),ivec3(1));', 'uniform samplerCube image;'), /cube samplers/);
  assert.throws(() => compile('color=textureProj(image,vec4(.2));', 'uniform highp sampler2DArray image;'), /2D or 3D sampler/);
  assert.throws(() => compile('color=texelFetchOffset(image,ivec2(1),0,ivec2(1));', 'uniform samplerCube image;'), /non-shadow/);
  assert.throws(() => compile('color=vec4(textureGradOffset(image,vec2(.2),vec2(.1),vec2(.1),ivec2(1)));',
    'uniform highp isampler2D image;'), /Integer projected\/offset/);
  for (const expression of ['textureOffset(image,vec2(.2),ivec2(1),1.)', 'textureProj(image,vec3(.2),1.)']) {
    const source = vertex.replace('void main()', 'uniform sampler2D image;void main()').replace('vec4(position,1.0)', expression);
    assert.throws(() => compileEsslProgram(source, fragment('color=vec4(1.);', '')), /only valid in fragment/);
  }
});
