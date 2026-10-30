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

test('struct varyings preserve member values, matrix columns and neighboring locations', () => {
  const type = 'struct Surface { float weight; vec3 normal; mat2x3 frame; };';
  const r = program(V('lead = 1.0; payload = Surface(2.0, position, mat2x3(1.0)); tail = vec2(3.0); gl_Position = vec4(position, 1.0);',
    type + 'out float lead; out Surface payload; out vec2 tail;'),
  F('color = vec4(payload.normal.x, payload.frame[1].z, payload.weight, tail.x + lead);',
    type + 'in Surface payload; in vec2 tail; in float lead;'));
  assert.deepEqual(r.reflection.varyings.map(v => [v.name, v.location, v.flat]),
    [['lead', 0, false], ['payload', 1, false], ['tail', 5, false]]);
  for (const stage of [r.vertex, r.fragment]) {
    assert.match(stage, /@location\(1\) f3d_v1: f32,/);
    assert.match(stage, /@location\(2\) f3d_v2: vec3<f32>,/);
    assert.match(stage, /@location\(3\) f3d_v3: vec3<f32>,/);
    assert.match(stage, /@location\(4\) f3d_v4: vec3<f32>,/);
    assert.match(stage, /@location\(5\) tail: vec2<f32>,/);
    assert.doesNotMatch(stage, /@interpolate\(flat\)/);
  }
  // Execute the emitted inter-stage copy assignments with distinct column data.
  // This checks packing/unpacking, not rasterization or WGSL validation.
  const value = {weight: -0, normal: [2, 3, 5], frame: [[7, 11, 13], [17, 19, 23]]};
  const packed = {}, unpacked = {frame: [[], []]};
  const pack = r.vertex.split('\n').filter(line => /^\s*out\.f3d_v\d+ = payload\./.test(line)).join('\n');
  const unpack = r.fragment.split('\n').filter(line => /^\s*payload\..* = input\.f3d_v\d+;/.test(line)).join('\n');
  assert.equal(pack.split('\n').length, 4); assert.equal(unpack.split('\n').length, 4);
  Function('payload', 'out', pack)(value, packed);
  Function('payload', 'input', unpack)(unpacked, packed);
  assert.deepEqual(unpacked, value);
});

test('flat struct varyings keep integer and float members in one flat interface', () => {
  const type = 'struct Tag { vec2 uv; int kind; uvec2 ids; };';
  const r = program(V('tag = Tag(position.xy, 4, uvec2(5u, 6u)); gl_Position = vec4(position, 1.0);', type + 'flat out Tag tag;'),
    F('color = vec4(tag.uv, float(tag.kind), float(tag.ids.y));', type + 'flat in Tag tag;'));
  assert.equal(r.reflection.varyings[0].flat, true);
  for (const stage of [r.vertex, r.fragment]) {
    assert.match(stage, /@location\(0\) @interpolate\(flat\) f3d_v0: vec2<f32>,/);
    assert.match(stage, /@location\(1\) @interpolate\(flat\) f3d_v1: i32,/);
    assert.match(stage, /@location\(2\) @interpolate\(flat\) f3d_v2: vec2<u32>,/);
  }
  assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', type + 'out Tag tag;'), F('color = vec4(1.0);')),
    /Integer struct varyings require flat/);
});

test('centroid interpolation survives parsing for struct and ordinary varyings', () => {
  const type = 'struct Sample { vec2 uv; float weight; };';
  for (const flat of [false, true]) {
    const q = flat ? 'flat centroid ' : 'centroid ';
    const r = program(V('value = Sample(position.xy, 1.0); normal = position; gl_Position = vec4(position, 1.0);',
      type + q + 'out Sample value; ' + q + 'out vec3 normal;'),
    F('color = vec4(value.uv, value.weight + normal.z, 1.0);', type + q + 'in Sample value; ' + q + 'in vec3 normal;'));
    assert.deepEqual(r.reflection.varyings.map(v => [v.flat, v.centroid]), [[flat, true], [flat, true]]);
    const interpolation = flat ? '@interpolate(flat)' : '@interpolate(perspective, centroid)';
    for (const stage of [r.vertex, r.fragment]) assert.equal(stage.split(interpolation).length - 1, 3);
  }
});

test('struct varying slots escape fields and avoid user varying name collisions', () => {
  const type = 'struct Data { float alias; mat2 matrix; };';
  const vs = V('payload = Data(1.0, mat2(2.0)); payload_f0 = 3.0; f3d_v0 = 4.0; gl_Position = vec4(position, 1.0);',
    type + 'out Data payload; out float payload_f0; out float f3d_v0;');
  const fs = F('color = vec4(payload.alias, payload.matrix[1], payload_f0 + f3d_v0);',
    type + 'in Data payload; in float payload_f0; in float f3d_v0;');
  const r = program(vs, fs);
  assert.match(r.vertex, /out\.f3d_v0 = payload\.alias_;/);
  assert.match(r.fragment, /payload\.alias_ = input\.f3d_v0;/);
  assert.match(r.fragment, /@location\(3\) payload_f0: f32,/);
  assert.match(r.fragment, /@location\(4\) f3d_v0_: f32,/);
  const declarations = /struct F3DFragmentIn \{([\s\S]*?)\};/.exec(r.fragment)[1];
  const names = [...declarations.matchAll(/\s(\w+): /g)].map(m => m[1]);
  assert.equal(new Set(names).size, names.length);
  assert.match(compileEsslStage('vertex', vs), /@location\(2\) f3d_v2: vec2<f32>,/);
  assert.match(compileEsslStage('fragment', fs), /payload\.matrix\[1\] = input\.f3d_v2;/);
});

test('struct stage interfaces enforce ESSL shape, stage and link requirements', () => {
  for (const field of ['bool enabled;', 'bvec2 enabled;', 'float weights[2];', 'sampler2D image;']) {
    assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', `struct Bad { ${field} }; out Bad value;`), F('color = vec4(1.0);')),
      /struct varyings require numeric/);
  }
  const type = 'struct Data { vec2 uv; };';
  assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', type + 'struct Outer { Data inner; }; out Outer value;'), F('color = vec4(1.0);')),
    /struct varyings require numeric/);
  assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', type + 'out Data value[2];'), F('color = vec4(1.0);')), /Arrays of structs/);
  assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', type + 'in Data value;'), F('color = vec4(1.0);')), /vertex outputs or fragment inputs/);
  assert.throws(() => program(minimalVS, F('color = vec4(1.0);', type + 'out Data value;')), /vertex outputs or fragment inputs/);
  assert.throws(() => compileEsslStage('vertex', type + 'varying Data value; void main() { gl_Position = vec4(1.0); }'), /requires ESSL 3.00/);
  assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', 'struct Bad { centroid float value; }; out Bad data;'), F('color = vec4(1.0);')),
    /Qualifiers are not allowed on struct members/);
  for (const q of ['flat ', 'centroid ']) {
    assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', type + q + 'out Data value;'),
      F('color = vec4(value.uv, 0.0, 1.0);', type + 'in Data value;')), /interpolation differs/);
  }
  assert.throws(() => program(V('gl_Position = vec4(position, 1.0);', type + 'out Data value;'),
    F('color = vec4(value.uv, 1.0);', 'struct Data { vec3 uv; }; in Data value;')), /differs between declarations/);
});

test('unused unlinked struct inputs disappear, while used ones remain link errors', () => {
  const decl = 'struct Data { vec2 uv; }; in Data missing;';
  assert.doesNotMatch(program(minimalVS, F('color = vec4(1.0);', decl)).fragment, /@location\(\d+\) f3d_v/);
  assert.throws(() => program(minimalVS, F('color = vec4(missing.uv, 0.0, 1.0);', decl)), /not written by the vertex stage/);
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

test("rows:'gl' stores GL's bottom-up rows for render targets: clip Y mirrored, gl_FragCoord and dFdy unflipped", () => {
  const vs = V('gl_Position = mvp * vec4(position, 1.0);'), fs = F('color = vec4(gl_FragCoord.xy, dFdy(gl_FragCoord.y), 1.0);');
  const gl = compileEsslProgram(vs, fs, {rows: 'gl'}), canvas = compileEsslProgram(vs, fs);
  assert.match(gl.vertex, /out\.f3d_position\.y = -out\.f3d_position\.y;/);
  assert.doesNotMatch(canvas.vertex, /out\.f3d_position\.y = -/);
  assert.match(gl.fragment, /f3d_FragCoord = input\.f3d_frag_position;/);
  assert.match(canvas.fragment, /f3d_u\.f3d_target\.y - input\.f3d_frag_position\.y/);
  assert.match(gl.fragment, /[^-]dpdy\(/); assert.match(canvas.fragment, /\(-dpdy\(/);
  assert.equal(gl.reflection.rows, 'gl');
  assert.throws(() => compileEsslProgram(vs, fs, {rows: 'up'}), /rows must be/);
});

test('gl_DepthRange is the GL default range (near 0, far 1, diff 1) in both stages', () => {
  const r = program(V('gl_Position = mvp * vec4(position, gl_DepthRange.far);'), F('color = vec4(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff, 1.0);'));
  for (const m of [r.vertex, r.fragment]) {
    assert.match(m, /var<private> f3d_DepthRange: F3DDepthRange = F3DDepthRange\(0\.0, 1\.0, 1\.0\);/);
    assert.match(m, /struct F3DDepthRange \{/);
  }
  assert.match(r.fragment, /f3d_DepthRange\.diff/);
});


// Test seam for scalar control flow only: execute the emitted integer helper
// body with JS arithmetic. WGSL case blocks break implicitly; the adapter adds
// those breaks and removes scalar type syntax. This is not WGSL/GPU validation.
function scalarSwitch(body, {unsigned = false} = {}) {
  const wgsl = compileEsslStage('fragment', F('color = vec4(float(probe(' + (unsigned ? '0u' : '0') + ', 0)));',
    'int probe(' + (unsigned ? 'uint' : 'int') + ' key, int gate) {' + body + '}'));
  const signature = /fn probe\(([^)]*)\) -> i32 \{/.exec(wgsl);
  assert.ok(signature);
  const start = signature.index + signature[0].length;
  let end = start, depth = 1;
  for (; depth; end++) { if (wgsl[end] === '{') depth++; else if (wgsl[end] === '}') depth--; }
  const source = wgsl.slice(start, end - 1), braces = [];
  let code = '';
  for (const c of source) {
    if (c === '{') braces.push(/\b(?:case [^{}:]+|default):\s*$/.test(code));
    if (c === '}' && braces.pop()) code += '\nbreak;\n';
    code += c;
  }
  code = code.replace(/\bcase ([^:]+):\s*\{/g, (_, labels) => labels.split(',').map(label =>
    label.trim() === 'default' ? 'default:' : 'case ' + label.trim() + ':').join('\n') + '{')
    .replace(/\bvar (\w+): (?:i32|u32)\s*;/g, 'let $1 = 0;')
    .replace(/\bvar (\w+): bool\s*;/g, 'let $1 = false;')
    .replace(/\b(?:var|let) (\w+): (?:i32|u32|bool)\s*=/g, 'let $1 =')
    .replace(/\b(\d+)u\b/g, '$1');
  const params = signature[1].split(',').map(parameter => parameter.trim().split(':')[0]);
  return {wgsl, run: Function('i32', 'u32', ...params, code).bind(null, value => value | 0, value => value >>> 0)};
}

test('switch fallthrough preserves default placement, empty labels and a single selector evaluation', () => {
  const {run, wgsl} = scalarSwitch('int n=key; int value=0; switch(n++) {' +
    'case 0:value+=1; case 1:case 2:value+=10; default:value+=100; case 3:value+=1000;break;' +
    'case 4:value=7;} return value*10+n;');
  const reference = key => {
    let n=key, value=0;
    switch(n++) {case 0:value+=1; case 1:case 2:value+=10; default:value+=100; case 3:value+=1000;break;case 4:value=7;}
    return value*10+n;
  };
  for (const key of [-1,0,1,2,3,4,5]) assert.equal(run(key,0),reference(key));
  assert.equal((wgsl.match(/switch \(/g) ?? []).length,1);
});

test('switch fallthrough keeps conditional breaks inside nested if and block scopes', () => {
  const {run} = scalarSwitch('int value=0; switch(key) {' +
    'case 0:value+=1;if(gate>0){{if(gate>1)break;value+=10;}}' +
    'case 1:value+=100;if(gate<0)break;default:value+=1000;} return value;');
  const reference = (key,gate) => {
    let value=0;
    switch(key) {case 0:value+=1;if(gate>0){{if(gate>1)break;value+=10;}}
      case 1:value+=100;if(gate<0)break;default:value+=1000;}
    return value;
  };
  for (const key of [0,1,2]) for (const gate of [-1,0,1,2]) assert.equal(run(key,gate),reference(key,gate));
});

test('switch labels share declarations while mutable initializers and shadowed names stay source-correct', () => {
  const {run,wgsl} = scalarSwitch('int hits=0;int value=80;int result=0;switch(key){' +
    'case 0:int value=value+(++hits);case 1:if(key==1)value=20;result=value;break;' +
    'default:{int value=7;result=value;}}return result*100+hits*10+value;');
  assert.deepEqual([run(0,0),run(1,0),run(2,0)],[8190,2080,780]);
  assert.match(wgsl,/var f3d_t\d+_value: i32;/);
  assert.equal((wgsl.match(/hits = \(hits \+ 1\);/g) ?? []).length,1);
  const constants = scalarSwitch('int value=0;switch(key){case 0:const int label=2;' +
    'case label:value=label;break;default:value=7;}return value;');
  assert.deepEqual([constants.run(0,0),constants.run(2,0),constants.run(5,0)],[2,2,7]);
});

test('switch suffixes stop at both-branch returns and correctly consume nested switch breaks', () => {
  const allReturn = scalarSwitch('switch(key){case 0:if(gate>0){return 1;}else{return 2;}gate=999;' +
    'case 1:return 3;default:return 4;}');
  assert.deepEqual([allReturn.run(0,1),allReturn.run(0,0),allReturn.run(1,0),allReturn.run(2,0)],[1,2,3,4]);
  assert.doesNotMatch(allReturn.wgsl,/gate = 999;/,'unreachable suffix statements are not emitted');
  const nested = scalarSwitch('int value=0;switch(key){case 0:switch(gate){case 0:value=1;break;default:value=2;}' +
    'case 1:value+=10;break;default:value=50;}return value;');
  assert.deepEqual([nested.run(0,0),nested.run(0,1),nested.run(1,0),nested.run(2,0)],[11,12,10,50]);
  const nestedReturn = scalarSwitch('switch(key){case 0:switch(gate){case 0:return 1;default:return 2;}' +
    'case 1:return 3;default:return 4;}');
  assert.deepEqual([nestedReturn.run(0,0),nestedReturn.run(0,1),nestedReturn.run(1,0)],[1,2,3]);
  assert.equal((nestedReturn.wgsl.match(/return 3;/g) ?? []).length,1);
});

test('switch fallthrough retains outer continue and inner-loop break targets without a dispatch loop', () => {
  const wgsl = compileEsslStage('fragment', F('int value=0;for(int i=0;i<3;i++){' +
    'switch(i){case 0:if(value==0)continue;value+=1;case 1:while(value<2){value++;break;}' +
    'value+=10;default:value+=100;}}color=vec4(float(value));'));
  assert.equal((wgsl.match(/\bloop \{/g) ?? []).length,3,'one outer loop and the inner loop in two case suffixes');
  assert.equal((wgsl.match(/\bcontinue;/g) ?? []).length,1);
  assert.equal((wgsl.match(/\bcontinuing \{/g) ?? []).length,1,'the original for step is the only continuing block');
  assert.match(wgsl,/value = \(value \+ 10\);[\s\S]*value = \(value \+ 100\);/);
});

test('switch unsigned selectors, discard, scope diagnostics and compact non-fallthrough cases', () => {
  const {run} = scalarSwitch('int value=0;switch(key){case 0u:value+=1;case 1u:value+=2;break;}return value;', {unsigned:true});
  assert.deepEqual([run(0,0),run(1,0),run(2,0)],[3,2,0]);
  const discarded = compileEsslStage('fragment', F('switch(0){case 0:discard;case 1:color=vec4(1.0);break;}'));
  assert.equal((discarded.match(/\bdiscard;/g) ?? []).length,1);
  const compact = scalarSwitch('int value=0;switch(key){case 0:value+=11;break;case 1:value+=22;break;' +
    'default:value+=33;}return value;').wgsl;
  for (const n of [11,22,33]) assert.equal(compact.split('value = (value + ' + n + ');').length-1,1);
  for (const body of ['switch(0){case 0:int value=1;case 1:int value=2;}', 'switch(0){case 0:break;case 0:break;}',
    'switch(0){default:break;default:break;}', 'switch(0){case 1u:break;}'])
    assert.throws(() => compileEsslStage('fragment', F(body)), EsslError);
  assert.throws(() => compileEsslStage('fragment', F('switch(0){case 0:int value=1;break;}color=vec4(float(value));')),/Undeclared identifier/);
  const cases = Array.from({length:400},(_,i) => 'case ' + i + ':value+=1;').join('');
  assert.throws(() => compileEsslStage('fragment', F('int value=0;switch(0){' + cases + '}')),/fallthrough expansion exceeds/);
});
