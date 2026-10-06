/** ESSL -> WGSL compiler: semantic lowering and reflection on small programs.
 * WGSL validity of the full r186 example corpus is checked separately with Naga
 * and Chrome (see ESSL_WGSL.md); rendered parity runs in tests/e2e/webgpu_renderer.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram, compileEsslStage, EsslError} from './essl_wgsl.mjs';
import {preprocess} from './essl_preprocess.mjs';

const V = (body, decls = '') => `#version 300 es\nprecision highp float;\nin vec3 position;\nuniform mat4 mvp;\n${decls}\nvoid main() {\n${body}\n}\n`;
const F = (body, decls = '') => `#version 300 es\nprecision highp float;\nlayout(location = 0) out vec4 color;\n${decls}\nvoid main() {\n${body}\n}\n`;
const program = (vs, fs) => compileEsslProgram(vs, fs);
const minimalVS = V('gl_Position = mvp * vec4(position, 1.0);');

test('preprocessor: function-like macros, nested conditionals, defined() and #error', () => {
  const {tokens} = preprocess('#define SQ(a) ((a) * (a))\n#define N 3\n#if defined(N) && N > 2\nfloat x = SQ(N + 1);\n#elif 1\nbad\n#else\nbad\n#endif\n#ifdef MISSING\nbad\n#endif');
  assert.equal(tokens.map(t => t.v).join(' '), 'float x = ( ( 3 + 1 ) * ( 3 + 1 ) ) ;');
  assert.throws(() => preprocess('#if 1\n#error stop here\n#endif'), /stop here/);
  assert.throws(() => preprocess('#if 1\n'), EsslError);
});

test('clip depth remaps GL [-w,w] to WebGPU [0,w]; gl_FragCoord is GL bottom-left; dFdy keeps GL sign', () => {
  const r = program(minimalVS, F('color = vec4(gl_FragCoord.xy, dFdy(gl_FragCoord.y), 1.0);'));
  assert.match(r.vertex, /out\.f3d_position = vec4<f32>\(f3d_Position\.xy, \(f3d_Position\.z \+ f3d_Position\.w\) \* 0\.5, f3d_Position\.w\);/);
  assert.match(r.fragment, /f3d_u\.f3d_target\.y - input\.f3d_frag_position\.y/);
  assert.match(r.fragment, /\(-dpdy\(/);
  assert.equal(typeof r.reflection.targetOffset, 'number');
  assert.equal(compileEsslProgram(minimalVS, F('color = vec4(1.0);'), {clipDepth: 'webgpu'}).vertex.includes('out.f3d_position = f3d_Position;'), true);
});

test('vector equality reduces with all/any; mod uses floor; integer shifts take u32 counts', () => {
  const r = program(minimalVS, F(`vec3 a = vec3(1.0); vec3 b = a;
  int i = 7; int j = i << 2; ivec2 k = ivec2(i) >> 1;
  color = vec4(a == b ? 1.0 : 0.0, a != b ? 1.0 : 0.0, mod(-1.5, 1.0), float(j + k.x));`));
  assert.match(r.fragment, /all\(\(?a\)? == \(?b\)?\)|all\(a == b\)/);
  assert.match(r.fragment, /any\(a != b\)/);
  assert.match(r.fragment, /floor\(/);
  assert.match(r.fragment, /i << u32\(2\)/);
  assert.match(r.fragment, /vec2<u32>\(u32\(1\)\)/);
});

test('out/inout parameters copy through temporaries in source order; swizzle stores split', () => {
  const r = program(minimalVS, F(`vec4 v = vec4(0.0); float s = 1.0;
  float r = bump(s) + s;
  split(v.xz, s);
  v.yw = vec2(2.0, 3.0);
  color = v * r;`, `float bump(inout float x) { x += 1.0; return x; }
void split(out vec2 o, float k) { o = vec2(k, -k); }`));
  // bump() result and its writeback are hoisted before the trailing read of s.
  assert.match(r.fragment, /var f3d_t\d+: f32 = s;\s+let f3d_t\d+ = bump\(&f3d_t\d+\);\s+s = f3d_t\d+;/);
  assert.match(r.fragment, /fn bump\(x: ptr<function, f32>\) -> f32/);
  assert.match(r.fragment, /v\.x = f3d_t\d+\.x;\s+v\.z = f3d_t\d+\.y;/);
  assert.match(r.fragment, /v\.y = f3d_t\d+\.x;\s+v\.w = f3d_t\d+\.y;/);
});

test('side-effecting ?: and && become statements; pure ones stay select/short-circuit', () => {
  const r = program(minimalVS, F(`float x = 0.0; bool c = x > 0.5;
  float y = c ? tick(x) : 2.0;
  bool z = c && tick(x) > 1.0;
  float w = c ? 1.0 : 2.0;
  color = vec4(x, y, w, z ? 1.0 : 0.0);`, `float tick(inout float v) { v += 1.0; return v; }`));
  assert.match(r.fragment, /if \(c\) \{/);
  assert.match(r.fragment, /select\(2\.0, 1\.0, c\)/);
});

test('uniform layout: bool -> u32, scalar/vec2 arrays -> vec4 elements, structs align 16, mat3 columns', () => {
  const r = program(V('gl_Position = mvp * vec4(position, 1.0);', 'struct L { vec3 c; bool on; float w[2]; };\nuniform L lights[2];\nuniform bool flag;\nuniform float k[3];\nuniform mat3 n;'),
    F('color = vec4(0.0);'));
  const by = Object.fromEntries(r.reflection.uniforms.map(u => [u.name, u]));
  assert.deepEqual(['mvp', 'lights', 'flag', 'k', 'n'], r.reflection.uniforms.map(u => u.name));
  assert.equal(by.mvp.offset, 0);
  assert.equal(by.lights.offset, 64);
  assert.equal(by.lights.node.stride, 48, 'vec3 + u32 (packed at 12) + 2 x vec4 -> 48');
  assert.deepEqual(by.lights.node.elem.fields.map(f => [f.name, f.node.offset]), [['c', 0], ['on', 12], ['w', 16]]);
  assert.equal(by.lights.node.elem.fields[2].node.stride, 16);
  assert.equal(by.flag.offset, 160); assert.equal(by.flag.node.s, 'bool');
  assert.equal(by.k.offset, 176); assert.equal(by.k.node.stride, 16);
  assert.equal(by.n.offset, 224); assert.equal(by.n.node.colStride, 16);
  assert.equal(r.reflection.uniformBufferSize, 272);
  assert.match(r.vertex, /struct L_u \{/);
});

test('combined samplers split into texture/sampler bindings, also as function parameters', () => {
  const r = program(minimalVS, F('color = tap(map, vec2(0.5)) + texture(shadows[1], vec3(0.5, 0.5, 0.25)) * vec4(1.0);',
    'uniform sampler2D map;\nuniform sampler2DShadow shadows[2];\nvec4 tap(sampler2D s, vec2 uv) { return texture(s, uv); }'));
  assert.deepEqual(r.reflection.textures.map(t => [t.name, t.element, t.textureBinding, t.samplerBinding, t.sampleType]),
    [['map', null, 0, 1, 'float'], ['shadows', 0, 2, 3, 'depth'], ['shadows', 1, 4, 5, 'depth']]);
  assert.match(r.fragment, /fn tap\(s_texture: texture_2d<f32>, s_sampler: sampler, uv_in: vec2<f32>\)/);
  assert.match(r.fragment, /textureSampleCompare\(shadows_1_texture, shadows_1_sampler/);
  assert.throws(() => program(minimalVS, F('int i = 1; color = texture(shadows[i], vec3(0.5));', 'uniform sampler2DShadow shadows[2];')), /constant/);
});

test('varyings link by name with matching locations; arrays and matrices use one location per slot', () => {
  const vs = V('vUv = position.xy; vArr[0] = vec4(1.0); vArr[1] = vec4(2.0); vM = mat3(1.0); vI = 3; gl_Position = mvp * vec4(position, 1.0);',
    'out vec2 vUv;\nout vec4 vArr[2];\nout mat3 vM;\nflat out int vI;');
  const fs = F('color = vec4(vUv, vM[1].y, float(vI)) + vArr[1];', 'in vec2 vUv;\nin vec4 vArr[2];\nin mat3 vM;\nflat in int vI;');
  const r = program(vs, fs);
  assert.deepEqual(r.reflection.varyings.map(v => [v.name, v.location, v.flat]), [['vUv', 0, false], ['vArr', 1, false], ['vM', 3, false], ['vI', 6, true]]);
  assert.match(r.fragment, /@location\(2\) vArr_e1: vec4<f32>/);
  assert.match(r.fragment, /@location\(6\) @interpolate\(flat\) vI: i32/);
  assert.throws(() => program(vs, F('color = vec4(vMissing);', 'in float vMissing;')), /not written/);
  // Declared but unused (ShaderLib does this): GL links, and so do we.
  assert.doesNotMatch(program(vs, F('color = vec4(1.0);', 'in float vMissing;')).fragment, /vMissing_|location\(\d+\) vMissing/);
});

test('ESSL 1.00 sources: gl_FragColor and texture2D; texture() in vertex stages samples level 0', () => {
  const r = program('attribute vec3 position; uniform sampler2D h; varying float v; void main(){ v = texture2D(h, position.xy).r; gl_Position = vec4(position, 1.0); }',
    'precision mediump float; uniform sampler2D h; varying float v; void main(){ gl_FragColor = texture2D(h, vec2(v)); }');
  assert.match(r.vertex, /textureSampleLevel\(h_texture, h_sampler, [^,]+, 0\.0\)/);
  assert.match(r.fragment, /@location\(0\) f3d_FragColor: vec4<f32>/);
  assert.equal(r.reflection.textures.length, 1);
});

test('invalid ESSL fails explicitly instead of being dropped', () => {
  assert.match(program(minimalVS, F('color = vec4(1);')).fragment, /vec4<f32>\(f32\(1\)\)/, 'constructors convert');
  assert.throws(() => program(minimalVS, F('float x = 1;')), /Type mismatch/);
  assert.throws(() => program(minimalVS, F('color = vec4(gl_PointCoord, 0.0, 1.0);')), /point primitives/);
  assert.throws(() => compileEsslStage('fragment', F('undefinedCall();')), /Unknown function/);
});

test('points: each vertex becomes an instanced quad of gl_PointSize pixels; gl_PointCoord has GL upper-left origin', () => {
  const r = compileEsslProgram(V('gl_PointSize = 8.0; gl_Position = mvp * vec4(position, 1.0);'), F('color = vec4(gl_PointCoord, 0.0, 1.0);'), {points: true, maxPointSize: 64});
  assert.equal(r.reflection.points, true);
  assert.equal(typeof r.reflection.targetOffset, 'number', 'viewport size needed for the pixel-to-NDC scale');
  assert.match(r.vertex, /@builtin\(vertex_index\) f3d_corner_index: u32/);
  assert.match(r.vertex, /clamp\(f3d_PointSize, 1\.0, 64\.0\)/);
  assert.match(r.vertex, /f3d_corner \* f3d_size \/ f3d_u\.f3d_target\.zw \* f3d_p\.w/);
  assert.match(r.vertex, /out\.f3d_point_coord = vec2<f32>\(0\.5 \+ 0\.5 \* f3d_corner\.x, 0\.5 - 0\.5 \* f3d_corner\.y\)/);
  assert.match(r.vertex, /select\(vec4<f32>\(2\.0, 2\.0, 2\.0, 1\.0\)/, 'a clipped center drops the whole point');
  assert.match(r.fragment, /f3d_PointCoord = input\.f3d_point_coord;/);
});
