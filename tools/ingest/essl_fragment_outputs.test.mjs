/** ESSL fragment output interfaces, not a rasterization or GPU timing oracle. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram, compileEsslStage} from './essl_wgsl.mjs';

const vertex = '#version 300 es\nin vec3 position;void main(){gl_Position=vec4(position,1.);}';
const fragment = (declarations, body = '') => '#version 300 es\nprecision highp float;precision highp int;\n' +
  declarations + '\nvoid main(){' + body + '}';
const compile = (declarations, body) => compileEsslProgram(vertex, fragment(declarations, body));

test('fragment output arrays flatten into consecutive typed attachment locations', () => {
  const result = compile('layout(location=1) out vec4 colors[2];layout(location=0) out vec2 velocity;',
    'colors[0]=vec4(.25);colors[1]=vec4(.75);velocity=vec2(.5);');
  assert.deepEqual(result.reflection.outputs, [
    {name: 'colors[0]', glslType: 'vec4', location: 1}, {name: 'colors[1]', glslType: 'vec4', location: 2},
    {name: 'velocity', glslType: 'vec2', location: 0},
  ]);
  assert.match(result.fragment, /var<private> colors: array<vec4<f32>, 2>;/);
  assert.match(result.fragment, /@location\(1\) f3d_o1: vec4<f32>,/);
  assert.match(result.fragment, /@location\(2\) f3d_o2: vec4<f32>,/);
  assert.match(result.fragment, /out\.f3d_o1 = colors\[0\];/);
  assert.match(result.fragment, /out\.f3d_o2 = colors\[1\];/);
  assert.doesNotMatch(result.fragment, /@location\([^)]*\) \w+: array</);
});

test('one implicit array output starts at zero for float, signed and unsigned element shapes', () => {
  for (const type of ['float', 'vec2', 'vec3', 'vec4', 'int', 'ivec2', 'ivec3', 'ivec4', 'uint', 'uvec2', 'uvec3', 'uvec4']) {
    const result = compile(`out ${type} values[2];`);
    assert.deepEqual(result.reflection.outputs, [
      {name: 'values[0]', glslType: type, location: 0}, {name: 'values[1]', glslType: type, location: 1},
    ]);
    assert.equal(compileEsslStage('fragment', fragment(`out ${type} values[2];`)), result.fragment);
  }
});

test('arrays retain whole-value copies, inout helpers and constant direct subscript rules', () => {
  const result = compile(`const uint count=2u;out vec4 values[count];
    void update(inout vec4 copy[2], int selected) {copy[selected]+=vec4(.125);}`, `
    values=vec4[2](vec4(.25),vec4(.5));
    update(values,1);
    const int last=values.length()-1;
    (values)[last].xy=values[0u].yx;
    int component=2;values[1][component]=.75;
    {vec4 values[2];int i=1;values[i]=vec4(.5);}`);
  assert.match(result.fragment, /fn update\(copy: ptr<function, array<vec4<f32>, 2>>, selected_in: i32\)/);
  assert.match(result.fragment, /values\[1\]\.x =/);
  assert.match(result.fragment, /values\[0u\]/);
  assert.match(result.fragment, /out\.f3d_o1 = values\[1\];/);
});

test('generated array members cannot collide with user output names', () => {
  const result = compile('layout(location=0) out vec4 colors[2];layout(location=2) out vec4 colors_e0;layout(location=3) out vec4 f3d_o0;',
    'colors[0]=vec4(.25);colors[1]=vec4(.5);colors_e0=vec4(.75);f3d_o0=vec4(1.);');
  const members = [...result.fragment.matchAll(/@location\(\d+\) (\w+):/g)].map(match => match[1]);
  assert.equal(new Set(members).size, 4);
  assert.deepEqual(members, ['f3d_o0', 'f3d_o1', 'colors_e0', 'f3d_o0_']);
});

test('fragment output arrays reject dynamic and out-of-range direct reads and writes', () => {
  for (const access of ['values[selector]=vec4(1.);', 'vec4 x=values[selector];',
    '(values)[selector].xy=vec2(1.);', 'update(values[selector]);', 'values[-1]=vec4(1.);',
    'values[values.length()]=vec4(1.);', 'values[.5]=vec4(1.);', 'values[false]=vec4(1.);']) {
    assert.throws(() => compile('out vec4 values[2];uniform int selector;void update(out vec4 x){x=vec4(1.);}', access),
      /Fragment output array index/, access);
  }
});

test('fragment output location validation includes the full array range', () => {
  assert.throws(() => compile('layout(location=2) out vec4 a[2];layout(location=3) out vec4 b;'), /location 3 is already occupied/);
  assert.throws(() => compile('layout(location=3) out vec4 b;layout(location=2) out vec4 a[2];'), /location 3 is already occupied/);
  assert.throws(() => compile('layout(location=0) out vec4 a[2],b[2];'), /location 0 is already occupied/);
  assert.throws(() => compile('out vec4 a[2];out vec4 b;'), /Multiple fragment outputs require explicit locations/);
  assert.throws(() => compile('layout(location=.5) out vec4 a[2];'), /location must be a nonnegative integer/);
  assert.throws(() => compile('out vec4 a[4097];'), /expansion exceeds 4096/);
});

test('fragment output type validation excludes matrices, booleans and nested arrays', () => {
  for (const declaration of ['out mat2 invalid;', 'out mat3 invalid[2];', 'out bvec2 invalid[2];',
    'out float[2] invalid[2];', 'out vec4 invalid[2.5];', 'centroid out vec4 invalid[2];'])
    assert.throws(() => compile(declaration), /Fragment output/);
  assert.throws(() => compileEsslStage('fragment', 'precision highp float;out vec4 colors[2];void main(){}'),
    /Fragment output arrays require ESSL 3.00/);
});

test('single ordinary and ESSL 1.00 fragment output contracts are unchanged', () => {
  assert.deepEqual(compile('out vec4 color;', 'color=vec4(.5);').reflection.outputs,
    [{name: 'color', glslType: 'vec4', location: 0}]);
  const legacy = compileEsslProgram('attribute vec3 position;void main(){gl_Position=vec4(position,1.);}',
    'precision highp float;void main(){gl_FragColor=vec4(.5);}');
  assert.deepEqual(legacy.reflection.outputs, [{name: 'gl_FragColor', glslType: 'vec4', location: 0}]);
});
