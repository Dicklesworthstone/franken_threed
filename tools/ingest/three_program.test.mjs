/** ShaderMaterial program assembly, WebGL raster state and uniform packing.
 * Rendered parity against upstream WebGLRenderer: tests/e2e/webgpu_renderer
 * scenario shader_material (--surface webgl --reference webgl).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {threeProgramSources, threeProgramRaster, packThreeProgramUniforms, createThreeProgramSupport} from './three_program.mjs';
import {compileEsslProgram} from './essl_wgsl.mjs';
const T = await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js'), 'build/three.module.js')));

const state = {toneMapping: T.NoToneMapping, outputColorSpace: T.SRGBColorSpace};
const shader = (extra = {}) => new T.ShaderMaterial({
  uniforms: {tint: {value: new T.Color(1, 0.5, 0.25)}, k: {value: [1, 2, 3]}, on: {value: true}, fogColor: {value: new T.Color()}, fogNear: {value: 0}, fogFar: {value: 0}},
  vertexShader: '#include <common>\nvoid main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: 'uniform vec3 tint; uniform float k[3]; uniform bool on;\nvoid main() { gl_FragColor = linearToOutputTexel(vec4(tint * k[2], on ? 1.0 : 0.0)); }',
  ...extra,
});

test('ShaderMaterial sources follow r186 WebGLProgram: version, precision, defines, includes, conversion macros', () => {
  const m = shader({defines: {FOO: 2, OFF: false}}), mesh = new T.Mesh(new T.BoxGeometry(), m);
  const {vertex, fragment} = threeProgramSources(T, m, mesh, state);
  assert.ok(vertex.startsWith('#version 300 es\n#define attribute in\n#define varying out\n#define texture2D texture\nprecision highp float;'));
  assert.match(vertex, /#define SHADER_TYPE ShaderMaterial\n#define SHADER_NAME \n#define FOO 2\n#define HAS_NORMAL/);
  assert.doesNotMatch(vertex, /OFF|#include/);
  assert.match(vertex, /#define PI 3\.141592653589793/, '<common> resolved from the live ShaderChunk');
  assert.match(fragment, /layout\(location = 0\) out highp vec4 pc_fragColor;\n#define gl_FragColor pc_fragColor/);
  assert.match(fragment, /vec4 linearToOutputTexel\( vec4 value \) \{\n\treturn sRGBTransferOETF/);
  // Transparent double-sided second pass: the source flips the side define.
  assert.match(threeProgramSources(T, m, mesh, {...state, side: T.BackSide}).vertex, /#define FLIP_SIDED/);
  // Raw programs get only the SHADER_* and custom defines.
  const raw = new T.RawShaderMaterial({vertexShader: 'void main(){}', fragmentShader: 'void main(){}', defines: {A: 1}});
  assert.equal(threeProgramSources(T, raw, new T.Mesh(new T.BufferGeometry(), raw), state).vertex,
    '#define SHADER_TYPE RawShaderMaterial\n#define SHADER_NAME \n#define A 1\nvoid main(){}');
  assert.throws(() => threeProgramSources(T, shader({lights: true}), mesh, state), {code: 'THREE_PROGRAM_LIGHTS'});
});

test('raster state uses WebGLState: blend table, BACK culling with frontFace flips', () => {
  const additive = threeProgramRaster(T, shader({blending: T.AdditiveBlending, transparent: true}));
  assert.deepEqual(additive.blend, {color: {operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one'}, alpha: {operation: 'add', srcFactor: 'one', dstFactor: 'one'}});
  assert.equal(threeProgramRaster(T, shader()).blend, null, 'opaque NormalBlending disables blending');
  assert.deepEqual([threeProgramRaster(T, shader({side: T.BackSide})).cullMode, threeProgramRaster(T, shader({side: T.BackSide})).frontFace], ['back', 'cw']);
  assert.equal(threeProgramRaster(T, shader({side: T.BackSide}), {frontFaceCW: true}).frontFace, 'ccw');
  assert.equal(threeProgramRaster(T, shader({side: T.DoubleSide})).cullMode, 'none');
  assert.throws(() => threeProgramRaster(T, shader({blending: T.SubtractiveBlending, transparent: true})), {code: 'THREE_PROGRAM_MATERIAL'});
});

test('uniform packing: built-ins from camera/object, material values, bools, arrays, fog refresh, untouched zeros', () => {
  const m = shader({fog: true}), mesh = new T.Mesh(new T.BoxGeometry(), m);
  mesh.position.set(1, 2, 3); mesh.updateMatrixWorld();
  const camera = new T.PerspectiveCamera(50, 1, 0.1, 10); camera.position.z = 5; camera.updateMatrixWorld();
  mesh.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld);
  const fog = new T.Fog(0x336699, 1, 9);
  const {vertex, fragment} = threeProgramSources(T, m, mesh, {...state, fog});
  const {reflection} = compileEsslProgram(vertex, fragment);
  const bytes = new Uint8Array(reflection.uniformBufferSize), view = new DataView(bytes.buffer);
  packThreeProgramUniforms(T, reflection, m, mesh, camera, bytes, {fog});
  const at = name => reflection.uniforms.find(u => u.name === name);
  const f = (o) => view.getFloat32(o, true);
  assert.deepEqual([12, 13, 14].map(i => f(at('modelMatrix').offset + i * 4)), [1, 2, 3]);
  assert.equal(f(at('modelViewMatrix').offset + 14 * 4), 3 - 5);
  assert.deepEqual([0, 1, 2].map(i => f(at('tint').offset + i * 4)), [1, 0.5, 0.25]);
  assert.deepEqual([0, 1, 2].map(i => f(at('k').offset + i * at('k').node.stride)), [1, 2, 3]);
  assert.equal(view.getUint32(at('on').offset, true), 1);
  assert.equal(view.getUint32(at('isOrthographic').offset, true), 0);
  assert.equal(m.uniforms.fogFar.value, 9, 'fog uniforms refreshed into the material, as WebGLRenderer does');
});

test('program support caches compiled programs by exact source text', () => {
  const support = createThreeProgramSupport({three: T, state: () => state});
  const m = shader(), a = new T.Mesh(new T.BoxGeometry(), m), b = new T.Mesh(new T.BoxGeometry(), m);
  assert.equal(support.compile(m, a), support.compile(m, b));
  assert.notEqual(support.compile(m, a), support.compile(m, new T.Mesh(new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(9), 3)), m)),
    'HAS_NORMAL differs, so the program differs');
});
