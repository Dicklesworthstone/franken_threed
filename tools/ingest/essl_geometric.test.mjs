import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram} from './essl_wgsl.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.);}';
const fragment = (body, declarations = '') => '#version 300 es\nprecision highp float;out vec4 color;\n' + declarations + '\nvoid main(){' + body + '}';
const compile = (body, declarations) => compileEsslProgram(vertex, fragment(body, declarations));

test('scalar geometric operations lower without vector-only WGSL overloads', () => {
  const {fragment: wgsl} = compile(`float a=normalize(inputValue.x);
    float b=faceforward(inputValue.x,inputValue.y,inputValue.z);
    float c=reflect(inputValue.x,inputValue.y);
    float d=refract(inputValue.x,inputValue.y,inputValue.z);
    color=vec4(a,b,c,d);`, 'uniform vec3 inputValue;');
  assert.match(wgsl, /sign\(f3d_u\.inputValue\.x\)/);
  for (const name of ['faceforward', 'reflect', 'refract']) {
    assert.match(wgsl, new RegExp(`fn f3d_${name}_scalar\\(`));
    assert.equal((wgsl.match(new RegExp(`fn f3d_${name}_scalar\\(`, 'g')) ?? []).length, 1);
  }
  assert.doesNotMatch(wgsl, /\b(normalize|faceForward|reflect|refract)\(f3d_u/);
});

test('scalar helpers use the source geometric definitions and guard total internal reflection', () => {
  const {fragment: wgsl} = compile(`color=vec4(faceforward(1.,-.5,1.),reflect(.5,1.),refract(.5,1.,2.),1.);`);
  assert.match(wgsl, /select\(-n, n, nref \* i < 0\.0\)/);
  assert.match(wgsl, /i - 2\.0 \* \(n \* i\) \* n/);
  assert.match(wgsl, /let k = 1\.0 - eta \* eta \* \(1\.0 - d \* d\);/);
  assert.ok(wgsl.indexOf('if (k < 0.0)') < wgsl.indexOf('sqrt(k)'), 'negative discriminants return before sqrt');
});

test('constant scalar normalization remains a legal WGSL constant in globals and local offsets', () => {
  const {fragment: wgsl} = compile(`const float localSign=normalize(3.);
    color=vec4(direction,localSign,normalize(direction),1.);`, 'const float direction=normalize(-2.);');
  assert.match(wgsl, /const direction: f32 = sign\(\(-2\.0\)\);/);
  assert.match(wgsl, /const localSign: f32 = sign\(3\.0\);/);
});

test('geometric float/vector signatures stay distinct from WGSL integer and broadcast overloads', () => {
  for (const expression of [
    'normalize(1)', 'normalize(mat2(1.))', 'length(ivec2(1))', 'distance(vec2(1.),1.)',
    'dot(ivec2(1),ivec2(2))', 'dot(vec2(1.),vec3(1.))', 'cross(vec2(1.),vec2(1.))',
    'faceforward(1.,vec2(1.),1.)', 'reflect(vec3(1.),1.)', 'refract(vec2(1.),vec2(1.),vec2(.5))',
    'refract(1.,1.,1)', 'refract(1.,1.)',
  ]) assert.throws(() => compile(`color=vec4(${expression});`), /ESSL/, expression);
  for (const type of ['float', 'vec2', 'vec3', 'vec4']) {
    const {fragment: wgsl} = compile(`float a=length(x)+distance(x,y)+dot(x,y);
      ${type} b=normalize(x)+faceforward(x,y,z)+reflect(x,y)+refract(x,y,.5);
      color=vec4(a)+vec4(${type === 'float' ? 'b' : 'b.x'});`, `uniform ${type} x,y,z;`);
    assert.match(wgsl, /@fragment/);
    if (type !== 'float') for (const name of ['normalize', 'faceForward', 'reflect', 'refract'])
      assert.match(wgsl, new RegExp(`\\b${name}\\(`));
  }
});

test('ESSL 1.00 and vertex programs use the same scalar geometric lowering', () => {
  const vs = 'attribute vec3 position;varying vec4 result;void main(){' +
    'result=vec4(normalize(-2.),faceforward(1.,-.5,1.),reflect(-1.,1.),refract(-1.,1.,.5));' +
    'gl_Position=vec4(position,1.);}';
  const result = compileEsslProgram(vs, 'precision highp float;varying vec4 result;void main(){gl_FragColor=result;}');
  assert.match(result.vertex, /sign\(\(-2\.0\)\)/);
  assert.match(result.vertex, /fn f3d_refract_scalar\(/);
  assert.doesNotMatch(result.fragment, /fn f3d_refract_scalar\(/, 'helpers are owned only by the stage using them');
});

test('scalar helper arguments containing user calls are evaluated once', () => {
  const {fragment: wgsl} = compile(`float a=faceforward(tick(1.),tick(-.5),tick(1.));
    float b=reflect(tick(-1.),tick(1.));float c=refract(tick(-1.),tick(1.),tick(.5));
    color=vec4(a,b,c,float(calls));`, 'int calls=0;float tick(float value){calls++;return value;}');
  assert.equal((wgsl.match(/\btick\(/g) ?? []).length - 1, 8);
});
