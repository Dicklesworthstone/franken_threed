import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram} from './essl_wgsl.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.);}';
const fragment = (body, declarations) => '#version 300 es\nprecision highp float;precision highp int;out vec4 color;\n' + declarations + '\nvoid main(){' + body + '}';
const compile = (expression, sampler, declarations = '') => compileEsslProgram(vertex,
  fragment(`color=vec4(${expression});`, `uniform highp ${sampler} image;${declarations}`));

test('integer projected sampling preserves signed payloads, projective coordinates and explicit gradients', () => {
  for (const prefix of ['i', 'u']) for (const dimension of ['2D', '3D']) {
    const n = dimension === '2D' ? 2 : 3, sampler = `${prefix}sampler${dimension}`, gradient = `vec${n}(.25)`;
    for (const size of dimension === '2D' ? [3, 4] : [4]) for (const [name, extra] of [
      ['textureProj', ''], ['textureProjLod', ',1.5'], ['textureProjGrad', `,${gradient},${gradient}`],
      ['textureProjOffset', `,ivec${n}(1,-1${n === 3 ? ',2' : ''})`],
      ['textureProjLodOffset', `,1.5,ivec${n}(1)`],
      ['textureProjGradOffset', `,${gradient},${gradient},ivec${n}(-1)`],
    ]) {
      const result = compile(`${name}(image,vec${size}(2.)${extra})`, sampler), wgsl = result.fragment;
      assert.equal(result.reflection.integerSampling, true);
      assert.equal(result.reflection.textures[0].sampleType, prefix === 'i' ? 'sint' : 'uint');
      assert.match(wgsl, new RegExp(`\\.?(?:${'xyz'.slice(0, n)}) / f3d_t\\d+\\.${size === 4 ? 'w' : 'z'}`));
      assert.match(wgsl, /textureLoad\(image, point/); assert.doesNotMatch(wgsl, /textureSample/);
      if (name.includes('Grad')) {
        assert.doesNotMatch(wgsl, /dpdx|dpdy/);
        assert.match(wgsl, new RegExp(`vec${n}<f32>\\(0\\.25\\) \\* f3d_t\\d+`));
      }
    }
  }
});

test('integer offset families wrap chosen mip texels while preserving array layers', () => {
  for (const prefix of ['i', 'u']) for (const dimension of ['2D', '3D', '2DArray']) {
    const n = dimension === '3D' ? 3 : 2, coordinates = dimension === '2D' ? 2 : 3;
    for (const expression of [
      `textureOffset(image,vec${coordinates}(.25),ivec${n}(-1))`,
      `textureLodOffset(image,vec${coordinates}(.25),2.,ivec${n}(1))`,
      `textureGradOffset(image,vec${coordinates}(.25),vec${n}(.125),vec${n}(.25),ivec${n}(1))`,
    ]) {
      const wgsl = compile(expression, `${prefix}sampler${dimension}`).fragment;
      assert.match(wgsl, /let coordinate = p \* f32\(size\) \+ f32\(delta\);/);
      assert.match(wgsl, /wrapped = extent - abs\(coordinate - period \* floor\(coordinate \/ period\) - extent\);/);
      assert.match(wgsl, /return min\(i32\(floor\(wrapped\)\), i32\(size\) - 1\);/);
      assert.match(wgsl, /textureDimensions\(image, level\)/);
      assert.match(wgsl, /floor\(lod \+ 0\.5\)/);
      assert.match(wgsl, /lod > 0\.5/);
      assert.match(wgsl, new RegExp(`delta: vec${n}<i32>`));
      if (dimension === '2DArray') {
        assert.match(wgsl, /floor\(p.z \+ 0.5\)/);
        assert.doesNotMatch(wgsl, /delta\.z/);
      }
    }
  }
});

test('projected coordinates and gradient calls are evaluated once in source argument order', () => {
  const wgsl = compile('textureProjGradOffset(image,nextPosition(),nextX(),nextY(),ivec2(1,-1))', 'usampler2D', `
    int trace=0;
    vec4 nextPosition(){trace=trace*10+1;return vec4(.5,1.5,9.,2.);}
    vec2 nextX(){trace=trace*10+2;return vec2(.25,0.);}
    vec2 nextY(){trace=trace*10+3;return vec2(0.,.25);}`).fragment;
  const main = /fn main_\(\) \{([\s\S]*?)^\}/m.exec(wgsl)[1];
  for (const name of ['nextPosition', 'nextX', 'nextY']) assert.equal(main.match(new RegExp(name + '\\(\\)', 'g'))?.length, 1);
  assert.ok(main.indexOf('nextPosition()') < main.indexOf('nextX()'));
  assert.ok(main.indexOf('nextX()') < main.indexOf('nextY()'));
  assert.equal(main.match(/\.xy \/ f3d_t\d+\.w/g)?.length, 1);
});

test('integer sampler arrays and sampler parameters retain independent wrap and mip controls', () => {
  const result = compileEsslProgram(vertex, fragment('color=vec4(tap(images[0],vec4(2.))+tap(images[1],vec4(3.)));', `
    uniform highp isampler2D images[2];
    ivec4 tap(highp isampler2D image,vec4 p){return textureProjLodOffset(image,p,1.5,ivec2(-1,2));}`));
  assert.deepEqual(result.reflection.textures.map(texture => [texture.element, texture.samplingOffset]), [[0, 0], [1, 16]]);
  assert.match(result.fragment, /image_sampling: vec4<u32>/);
  for (const index of [0, 1]) assert.ok(result.fragment.includes(`images_${index}_sampler, f3d_u.f3d_sampling_${index}`));
});

test('vertex implicit projected offsets select the base level without derivatives', () => {
  const source = '#version 300 es\nprecision highp float;precision highp int;in vec3 position;uniform highp usampler3D image;' +
    'void main(){gl_Position=vec4(position+vec3(textureProjOffset(image,vec4(.5,.75,1.,2.),ivec3(1,-1,2)).xyz)*.01,1.);}';
  const {vertex: wgsl} = compileEsslProgram(source, fragment('color=vec4(1.);', ''));
  assert.doesNotMatch(wgsl, /dpdx|dpdy/);
  assert.match(wgsl, /f3d_integer_sample_3d_uint_offset\(image_texture, f3d_u.f3d_sampling_0, f3d_t\d+, 0.0, vec3<i32>\(1, \(-1\), 2\)\)/);
});

test('integer projection and offsets reject invalid ESSL shapes and retain explicit unsupported boundaries', () => {
  for (const [sampler, expression, pattern] of [
    ['usampler2D', 'textureProj(image,vec2(.5))', /coordinates/],
    ['isampler3D', 'textureProj(image,vec3(.5))', /coordinates/],
    ['usampler2DArray', 'textureProj(image,vec4(.5))', /2D or 3D/],
    ['usamplerCube', 'textureOffset(image,vec3(.5),ivec3(1))', /cube/],
    ['usampler2D', 'textureProjLod(image,vec3(.5),1)', /float LOD/],
    ['isampler3D', 'textureProjGrad(image,vec4(.5),vec2(.25),vec2(.25))', /derivatives/],
    ['usampler2DArray', 'textureGradOffset(image,vec3(.5),vec3(.25),vec3(.25),ivec2(1))', /derivatives/],
    ['usampler2D', 'textureOffset(image,vec2(.5),ivec3(1))', /ivec2/],
    ['usampler2DArray', 'textureOffset(image,vec3(.5),ivec3(1))', /ivec2/],
    ['usampler2D', 'textureProjOffset(image,vec4(.5),ivec2(1),3.)', /bias must be proven/],
  ]) assert.throws(() => compile(expression, sampler), pattern, expression);
  assert.throws(() => compileEsslProgram(vertex, fragment('ivec2 delta=ivec2(1);color=vec4(textureOffset(image,vec2(.5),delta));',
    'uniform highp usampler2D image;')), /constant expression/);
  assert.throws(() => compileEsslProgram(vertex, fragment('ivec4 value=textureProj(image,vec3(.5));color=vec4(value);',
    'uniform highp usampler2D image;')), /Type mismatch/);
});
