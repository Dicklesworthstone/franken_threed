/** Structural shader constants through production compilation and reflection. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram} from './essl_wgsl.mjs';

const vertex = '#version 300 es\nvoid main(){gl_Position=vec4(0.,0.,0.,1.);}';
const compile = (declarations, body = '') => compileEsslProgram(vertex,
  '#version 300 es\nprecision highp float;precision highp int;\n' + declarations + '\nvoid main(){' + body + '}');

test('builtin-derived extents reach uniform, sampler and fragment output arrays', () => {
  const p = compile('const int count=max(2,clamp(3,1,4));uniform vec4 weights[count];' +
    'uniform sampler2D images[count];out vec4 colors[count];',
  'colors[0]=weights[0];colors[1]=texture(images[min(count-1,1)],vec2(.5));colors[2]=weights[2];');
  assert.equal(p.reflection.outputs.length, 3);
  assert.equal(p.reflection.textures.length, 3);
  assert.match(p.fragment, /images_1_texture, images_1_sampler/);
  assert.match(p.fragment, /colors: array<vec4<f32>, 3>/);
});

test('constant structs, arrays, matrix columns and swizzles select concrete resource bindings', () => {
  const p = compile(`struct Choices {ivec3 indices;bool enabled;};
    const Choices choices=Choices(ivec3(0,1,2),true);
    const int order[3]=int[3](2,0,1);
    const mat2 basis=mat2(2.0);
    uniform sampler2D images[3];out vec4 colors[3];`, `
    const ivec2 selected=choices.indices.zy;
    const bool enabled=any(bvec2(false,choices.enabled));
    colors[enabled ? selected.x : selected.y]=texture(images[order[1]],vec2(.5));
    colors[int(basis[0].x)-1]=texture(images[selected.y],vec2(.25));
    colors[0]=texture(images[int(mat2(2.0)[1].y)],vec2(.75));`);
  assert.match(p.fragment, /out\.f3d_o2 = colors\[2\]/);
  for (const index of [0, 1, 2]) assert.match(p.fragment, new RegExp(`images_${index}_texture, images_${index}_sampler`));
});

test('uniform block arrays and sampler parameters share builtin constant selection', () => {
  const p = compile(`uniform Samples {vec4 color;} source[max(1,2)];
    uniform sampler2D images[2];out vec4 color;
    vec4 sampleColor(sampler2D values[max(1,2)]) {return texture(values[clamp(-2,0,1)],vec2(.5));}`,
  'const bool last=all(equal(ivec2(2),ivec2(2)));color=source[last ? 1 : 0].color+sampleColor(images);');
  assert.equal(p.reflection.blocks.length, 2);
  assert.match(p.fragment, /textureSample\(f3d_samplers_\d+_0_texture, f3d_samplers_\d+_0_sampler/);
  assert.match(p.fragment, /fn sampleColor\(/);
});

test('source constant array lengths do not require constant array contents', () => {
  const p = compile('uniform sampler2D images[3];out vec4 colors[images.length()];',
    'const int last=images.length()-1;colors[last]=texture(images[last],vec2(.5));');
  assert.equal(p.reflection.outputs.length, 3);
  assert.match(p.fragment, /images_2_texture, images_2_sampler/);
});

test('texture offsets materialize closed selection builtins as WGSL constant vectors', () => {
  const p = compile('uniform sampler2D image;out vec4 color;', `
    const ivec2 offset=ivec2(mix(vec2(-1.,1.),vec2(1.,-1.),bvec2(true,false)));
    color=textureOffset(image,vec2(.5),offset);`);
  assert.match(p.fragment, /textureSample\([^\n]*vec2<i32>\(1, 1\)\)/);
});

test('switch labels use evaluated builtin values and detect equivalent duplicate labels', () => {
  const p = compile('uniform int selector;out vec4 color;',
    'switch(selector){case max(1,2):color=vec4(.25);break;case ivec2(3,4).y:color=vec4(.75);break;default:color=vec4(1.);}');
  assert.match(p.fragment, /case 2:/);
  assert.match(p.fragment, /case 4:/);
  assert.throws(() => compile('uniform int selector;out vec4 color;',
    'switch(selector){case min(2,3):break;case 2:break;}'), /Duplicate switch case/);
  assert.throws(() => compile('uniform int selector;out vec4 color;',
    'switch(selector){case selector:break;}'), /case needs a constant integer/);
});

test('constant arithmetic preserves integer division, unsigned wrap and float boundaries', () => {
  const expressions = new Map([
    ['7/2', 3], ['int(7.0/2.0)', 3], ['int(0xffffffffu+3u)', 2],
    ['int(0x80000000u>>30u)', 2], ['int(16777216.0+1.0)-16777215', 1],
    ['int(roundEven(2.5))', 2], ['int(floor(-1.25))+4', 2],
    ['int(fract(-1.25)*4.0)', 3], ['int(mod(-1.0,3.0))+1', 3],
    ['int(step(2.0,1.0))+1', 1], ['int(ceil(1.25))', 2],
    ['int(trunc(-1.25))+3', 2], ['abs(-3)', 3], ['sign(-2)+3', 2],
  ]);
  for (const [expression, count] of expressions)
    assert.equal(compile(`out vec4 colors[${expression}];`).reflection.outputs.length, count, expression);
});

test('constant component operators preserve Boolean selection and unsigned vector types', () => {
  const p = compile(`const uvec2 selector=(uvec2(0xffffffffu,0x80000000u)+uvec2(2u,0u))>>uvec2(0u,31u);
    const bvec2 selected=not(lessThan(ivec2(2,3),ivec2(1,3)));
    out vec4 colors[2];`, 'colors[all(selected) ? int(selector.y) : 0]=vec4(1.);');
  assert.match(p.fragment, /colors\[1\] = vec4<f32>\(1\.0\)/);
});

test('dynamic, effectful and shadowed builtin expressions never become structural constants', () => {
  for (const expression of ['max(runtime,2)', 'true ? 2 : runtime', '(false && runtime>0) ? 1 : 2',
    '(runtime=2)', '(runtime++,2)', 'userValue()', 'int(sin(1.0))+2'])
    assert.throws(() => compile(`uniform int runtime;int userValue(){return 2;}out vec4 colors[${expression}];`),
      /constant integer expression/, expression);
  assert.throws(() => compile('int max(int a,int b);out vec4 colors[max(1,2)];int max(int a,int b){return a+b;}'),
    /constant integer expression/);
  assert.throws(() => compile('out vec4 colors[2];', 'int mutable=1;colors[max(mutable,0)]=vec4(1.);'),
    /Fragment output array index must be a constant/);
  assert.throws(() => compile('uniform sampler2D image;uniform ivec2 runtime;out vec4 color;',
    'color=textureOffset(image,vec2(.5),clamp(runtime,ivec2(-1),ivec2(1)));'), /Texture offset must be a constant/);
});

test('invalid extents and aggregate construction stay bounded before output expansion', () => {
  for (const expression of ['2.0', 'true', '1/0', '1<<32', 'ivec2(1,2).z', '(int[2](1,2))[2]', 'clamp(1,3,2)'])
    assert.throws(() => compile(`out vec4 colors[${expression}];`), /constant integer expression/, expression);
  assert.throws(() => compile('out vec4 colors[int[4097](1)[0]];'), /constant integer expression/);
  assert.throws(() => compile('out vec4 colors[max(4096,4097)];'), /Fragment output expansion exceeds/);
});
