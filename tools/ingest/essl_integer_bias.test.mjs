import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram} from './essl_wgsl.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.);}';
const fragment = (body, declarations = '') => '#version 300 es\nprecision highp float;precision highp int;out vec4 color;\n' + declarations + '\nvoid main(){' + body + '}';
const compile = (expression, sampler = 'usampler2D', declarations = '') => compileEsslProgram(vertex,
  fragment(`color=vec4(${expression});`, `uniform highp ${sampler} image;uniform float bias;${declarations}`));

test('portable integer bias covers ordinary, offset and projected sampler overloads', () => {
  for (const prefix of ['i', 'u']) for (const dimension of ['2D', '3D', '2DArray']) {
    const sampler = prefix + 'sampler' + dimension, n = dimension === '3D' ? 3 : 2;
    const p = dimension === '2D' ? 'vec2(.2,.7)' : 'vec3(.2,.7,1.6)', delta = `ivec${n}(1)`;
    const expressions = [`texture(image,${p},1.)`, `textureOffset(image,${p},${delta},-2.)`];
    if (dimension !== '2DArray') expressions.push('textureProj(image,vec4(.2,.7,.6,2.),1.5)',
      `textureProjOffset(image,vec4(.2,.7,.6,2.),${delta},-1.5)`);
    for (const expression of expressions) {
      const source = compile(expression, sampler).fragment;
      assert.match(source, /textureLoad\(image, point/);
      assert.equal((source.match(/dpdx\(/g) ?? []).length, 1);
      assert.equal((source.match(/dpdy\(/g) ?? []).length, 1);
      assert.match(source, /log2\(.*\) \+ /);
      assert.doesNotMatch(source, /textureSample/);
      if (dimension === '2DArray') assert.match(source, /floor\(p.z \+ 0.5\)/);
    }
  }
});

test('bias proof accepts literal signs, parentheses, safe const aliases and explicit bounded clamps', () => {
  for (const value of ['0.', '(-2.)', '+(2.)', 'float(1)', 'LIMIT', '(-LIMIT)',
    'clamp(bias,-2.,2.)', 'clamp(bias,-LIMIT,LIMIT)', '-clamp(bias,-2.,1.)',
    'clamp(bias,0.,0.)']) assert.doesNotThrow(() => compile(`texture(image,vec2(.5),${value})`, 'usampler2D', 'const float LIMIT=2.;'));
  const source = compileEsslProgram(vertex, fragment('const float lo=-2.;const float hi=-lo;' +
    'const float local=1.5;color=vec4(texture(image,vec2(.5),clamp(bias,lo,hi))+texture(image,vec2(.5),local));',
    'uniform highp usampler2D image;uniform float bias;')).fragment;
  assert.match(source, /clamp\(f3d_u.bias, lo, hi\)/);
});

test('unproven bias and source-invalid signatures refuse before GPU compilation', () => {
  for (const value of ['bias', '2.01', '-3.', 'clamp(bias,-3.,2.)', 'clamp(bias,-2.,3.)',
    'clamp(bias,2.,-2.)', 'clamp(bias,-bias,2.)', 'clamp(bias,clamp(bias,-2.,-2.),2.)', '1.+1.', 'LIMIT',
    'float(-1u)', 'float(16777216)+1.-float(16777216)']) assert.throws(() => compile(`texture(image,vec2(.5),${value})`,
      'usampler2D', 'const float LIMIT=1.+1.;'), /bias must be proven within/);
  for (const call of ['texture(image,vec2(.5),1)', 'texture(image,vec2(.5),vec2(1.))'])
    assert.throws(() => compile(call), /float bias/);
  assert.throws(() => compile('textureLod(image,vec2(.5),1.,1.)'), /expects 3 arguments/);
  assert.throws(() => compile('textureGrad(image,vec2(.5),vec2(.1),vec2(.1),1.)'), /expects 4 arguments/);
  assert.throws(() => compile('textureProj(image,vec4(1.),1.)', 'usampler2DArray'), /2D or 3D sampler/);
  assert.throws(() => compile('textureOffset(image,vec2(.5),ivec2(int(bias)),1.)'), /constant expression/);
});

test('bias proof respects user-defined clamp and mutable shadowing of const aliases', () => {
  assert.throws(() => compile('texture(image,vec2(.5),clamp(bias,-2.,2.))', 'usampler2D',
    'float clamp(float value,float lo,float hi){return value;}'), /bias must be proven within/);
  assert.throws(() => compile('texture(image,vec2(.5),LIMIT)', 'usampler2D',
    'const float LIMIT=clamp(1.,-2.,2.);float clamp(float value,float lo,float hi){return 99.;}'), /bias must be proven within/);
  assert.throws(() => compile('texture(image,vec2(.5),float(1))', 'usampler2D',
    'float float(int value){return 99.;}'), /bias must be proven within/);
  assert.throws(() => compileEsslProgram(vertex, fragment('float LIMIT=bias;' +
    'color=vec4(texture(image,vec2(.5),LIMIT));', 'uniform highp usampler2D image;uniform float bias;const float LIMIT=2.;')),
    /bias must be proven within/);
});

test('integer bias preserves argument effects once and rejects vertex-stage bias', () => {
  const source = compile('textureProjOffset(image,point(),ivec2(1),clamp(nextBias(),-2.,2.))', 'usampler2D',
    'int calls=0;vec4 point(){calls=calls*10+1;return vec4(.2,.7,0.,2.);}' +
    'float nextBias(){calls=calls*10+2;return bias;}').fragment;
  assert.equal((source.match(/= point\(\)/g) ?? []).length, 1);
  assert.equal((source.match(/nextBias\(\)/g) ?? []).length, 2, 'one definition and one call');
  assert.ok(source.indexOf('= point()') < source.lastIndexOf('nextBias()'));
  const vs = vertex.replace('void main()', 'uniform highp usampler2D image;void main()')
    .replace('vec4(position,1.)', 'vec4(texture(image,vec2(.5),1.))');
  assert.throws(() => compileEsslProgram(vs, fragment('color=vec4(1.);')), /only valid in fragment/);
});
