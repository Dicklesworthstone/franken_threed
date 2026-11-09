import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram} from './essl_wgsl.mjs';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createThreeProgramSupport, packThreeProgramUniforms} from './three_program.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.);}';
const fragment = (body, declarations = 'uniform highp sampler2DShadow image;') =>
  '#version 300 es\nprecision highp float;precision highp int;out vec4 color;'+declarations+'void main(){'+body+'}';
const compile = (expression, declarations) => compileEsslProgram(vertex, fragment('color=vec4('+expression+');', declarations));
const forms = [
  ['textureLod', 'vec3(.2,.3,.4)', ''],
  ['textureLodOffset', 'vec3(.2,.3,.4)', ',ivec2(1,-1)'],
  ['textureProjLod', 'vec4(.4,.6,.8,2.)', ''],
  ['textureProjLodOffset', 'vec4(.4,.6,.8,2.)', ',ivec2(1,-1)'],
];

test('all four source 2D shadow explicit-LOD forms use level zero in either stage', () => {
  for (const [name, point, offset] of forms) {
    const expression = name+'(image,'+point+',0.'+offset+')';
    const fs = compile(expression);
    const vs = compileEsslProgram(vertex.replace('void main()', 'uniform highp sampler2DShadow image;void main()')
      .replace('vec4(position,1.)', 'vec4(position*'+expression+',1.)'), fragment('color=vec4(1.);', ''));
    for (const shader of [fs.fragment, vs.vertex]) {
      assert.match(shader, /textureSampleCompareLevel\(image_texture, image_sampler,/);
      assert.doesNotMatch(shader, /textureSampleCompare\(|dpdx|dpdy/);
      assert.match(shader, /f3d_u\.f3d_depth_reference_clamp_0\.x != 0u/);
      assert.equal((shader.match(/\.xyz \/ .*\.w/g) ?? []).length, Number(name.includes('Proj')));
      if (offset) assert.match(shader, /, vec2<i32>\(1, \(-1\)\)\)/);
    }
    assert.equal(fs.reflection.textures[0].sampleType, 'depth');
    assert.equal(fs.reflection.textures[0].dimension, '2d');
  }
});

test('constant-zero LOD proof accepts safe aliases and refuses nonzero or runtime levels', () => {
  for (const lod of ['0.', '-0.', '+(0.)', 'float(0)', 'zero', 'float(integerZero)']) {
    const shader = compile('textureLod(image,vec3(.3),'+lod+')',
      'uniform highp sampler2DShadow image;const float zero=-(0.);const int integerZero=0;').fragment;
    assert.match(shader, /textureSampleCompareLevel/);
  }
  const local = compileEsslProgram(vertex, fragment(
    'const float zero=float(0);const float alias=zero;color=vec4(textureLod(image,vec3(.3),alias));'));
  assert.match(local.fragment, /textureSampleCompareLevel/);
  for (const lod of ['1.', '-1.', '.00001', 'level', 'float(0u)', '0.+0.', 'clamp(level,0.,0.)']) {
    assert.throws(() => compile('textureLod(image,vec3(.3),'+lod+')',
      'uniform highp sampler2DShadow image;uniform float level;'), /proven constant zero/);
  }
  assert.throws(() => compileEsslProgram(vertex, fragment('float zero=0.;color=vec4(textureLod(image,vec3(.3),zero));')),
    /proven constant zero/);
});

test('explicit comparison overloads enforce source dimensions, coordinate types and constant offsets', () => {
  for (const expression of [
    'textureLod(image,vec2(.3),0.)', 'textureLod(image,vec3(.3),0)',
    'textureLod(image,vec3(.3),0.,1.)', 'textureLodOffset(image,vec3(.3),0.)',
    'textureLodOffset(image,vec3(.3),0.,vec2(1.))',
    'textureProjLod(image,vec3(.3),0.)', 'textureProjLodOffset(image,vec4(.3),0.,ivec3(1))',
  ]) assert.throws(() => compile(expression));
  for (const sampler of ['samplerCubeShadow', 'sampler2DArrayShadow'])
    assert.throws(() => compile('textureLod(image,vec4(.3),0.)', 'uniform highp '+sampler+' image;'), /sampler2DShadow/);
  assert.throws(() => compileEsslProgram(vertex, fragment(
    'ivec2 offset=ivec2(1);color=vec4(textureLodOffset(image,vec3(.3),0.,offset));')), /constant expression/);
  for (const expression of ['texture(image,vec3(.3),0.)',
    'textureGrad(image,vec3(.3),vec2(.1),vec2(.1))',
    'textureProjGrad(image,vec4(.3),vec2(.1),vec2(.1))'])
    assert.throws(() => compile(expression), /not supported/);
});

test('projected explicit comparison evaluates a coordinate once and retains named offsets', () => {
  const {fragment: shader} = compileEsslProgram(vertex, fragment(
    'const ivec2 offset=ivec2(stepSize,-stepSize);float sample=textureProjLodOffset(image,point(),zero,offset);color=vec4(sample,float(calls),0.,1.);',
    'uniform highp sampler2DShadow image;const float zero=0.;const int stepSize=1;int calls=0;'+
    'vec4 point(){calls++;return vec4(.4,.6,.8,2.);}'));
  assert.equal((shader.match(/= point\(\);/g) ?? []).length, 1);
  assert.equal((shader.match(/\.xyz \/ .*\.w/g) ?? []).length, 1);
  assert.match(shader, /const offset: vec2<i32>/);
  assert.match(shader, /textureSampleCompareLevel\(.*offset\)/);
});

test('explicit comparison helpers and sampler arrays preserve independent source depth domains', () => {
  const result = compile('sampleDepth(images[0],vec4(.4,.6,.8,2.))+sampleDepth(images[1],vec4(.4,.6,.8,2.))',
    'uniform highp sampler2DShadow images[2];float sampleDepth(sampler2DShadow item,vec4 p){return textureProjLod(item,p,0.);}');
  assert.match(result.fragment, /item_depth_reference_clamp: u32/);
  assert.match(result.fragment, /textureSampleCompareLevel\(item_texture, item_sampler,/);
  const objects = [new T.DepthTexture(4,4,T.UnsignedIntType), new T.DepthTexture(4,4,T.FloatType)];
  const uniforms = {images:{value:objects}}, object = new T.Mesh(), camera = new T.PerspectiveCamera();
  const bytes = new Uint8Array(result.reflection.uniformBufferSize);
  packThreeProgramUniforms(T,result.reflection,uniforms,object,camera,bytes,
    {textureBindings:[{depthReferenceClamp:true},{depthReferenceClamp:false}]});
  assert.deepEqual(result.reflection.textures.map(t => new DataView(bytes.buffer).getUint32(t.depthReferenceClampOffset,true)),[1,0]);
  for (const texture of objects) texture.dispose(); object.geometry.dispose(); object.material.dispose();
});

test('public ShaderMaterial source preparation admits base-level comparison without changing resource binding', () => {
  const material = new T.ShaderMaterial({glslVersion:T.GLSL3,
    uniforms:{image:{value:new T.DepthTexture(4,4)}},
    vertexShader:'void main(){gl_Position=vec4(position,1.);}',
    fragmentShader:'uniform highp sampler2DShadow image;out vec4 color;void main(){if(gl_FragCoord.x>4.)color=vec4(textureLod(image,vec3(.2,.3,.4),0.));else color=vec4(textureProjLodOffset(image,vec4(.4,.6,.8,2.),0.,ivec2(1,-1)));}'});
  const object=new T.Mesh(new T.PlaneGeometry(),material);
  const support=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  try {
    const program=support.compile(material,object).program;
    assert.equal(program.reflection.textures.length,1);
    assert.equal(program.reflection.textures[0].sampleType,'depth');
    assert.equal((program.fragment.match(/textureSampleCompareLevel\(/g)??[]).length,2);
    assert.doesNotMatch(program.fragment,/textureSampleCompare\(/);
  } finally {support.dispose();material.uniforms.image.value.dispose();material.dispose();object.geometry.dispose();}
});
