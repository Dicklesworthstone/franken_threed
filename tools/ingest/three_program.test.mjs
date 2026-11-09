/** ShaderMaterial program assembly, WebGL raster state and uniform packing.
 * Rendered parity against upstream WebGLRenderer: tests/e2e/webgpu_renderer
 * scenario shader_material (--surface webgl --reference webgl).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {parse} from 'acorn';
import {simple} from 'acorn-walk';
import {threeProgramSources, threeProgramRaster, packThreeProgramUniforms, createThreeProgramSupport} from './three_program.mjs';
import {compileEsslProgram} from './essl_wgsl.mjs';
const upstream = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(upstream, 'build/three.module.js')));

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
  // Byte-identical to r186 WebGLProgram (empty extension line after #version).
  assert.ok(vertex.startsWith('#version 300 es\n\n#define attribute in\n#define varying out\n#define texture2D texture\nprecision highp float;\n\tprecision highp int;'));
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
  // Wireframe draws the same program over an edge index (the caller's lines topology).
  assert.equal(threeProgramSources(T, shader({wireframe: true}), mesh, state).fragment, threeProgramSources(T, shader(), mesh, state).fragment);
  assert.throws(() => threeProgramSources(T, shader({stencilWrite: true}), mesh, state), {code: 'THREE_PROGRAM_MATERIAL'});
});

test('raster state uses WebGLState: blend table, BACK culling with frontFace flips', () => {
  const additive = threeProgramRaster(T, shader({blending: T.AdditiveBlending, transparent: true}));
  assert.deepEqual(additive.blend, {color: {operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one'}, alpha: {operation: 'add', srcFactor: 'one', dstFactor: 'one'}});
  assert.equal(threeProgramRaster(T, shader()).blend, null, 'opaque NormalBlending disables blending');
  assert.deepEqual([threeProgramRaster(T, shader({side: T.BackSide})).cullMode, threeProgramRaster(T, shader({side: T.BackSide})).frontFace], ['back', 'cw']);
  assert.equal(threeProgramRaster(T, shader({side: T.BackSide}), {frontFaceCW: true}).frontFace, 'ccw');
  assert.equal(threeProgramRaster(T, shader({side: T.DoubleSide})).cullMode, 'none');
  const untested = threeProgramRaster(T, shader({depthTest: false, depthWrite: true}));
  assert.equal(untested.depthCompare, 'always');
  assert.equal(untested.depthWriteEnabled, false, 'a depth-disabled backdrop cannot occlude later scene geometry');
  assert.equal(threeProgramRaster(T, shader({depthTest: true, depthWrite: false})).depthWriteEnabled, false);
  assert.equal(threeProgramRaster(T, shader()).depthWriteEnabled, true);
  assert.throws(() => threeProgramRaster(T, shader({blending: T.SubtractiveBlending, transparent: true})), {code: 'THREE_PROGRAM_MATERIAL'});
});

test('uniform packing: built-ins from camera/object, material values, bools, arrays, fog refresh, untouched zeros', () => {
  const m = shader({fog: true}), mesh = new T.Mesh(new T.BoxGeometry(), m);
  mesh.position.set(1, 2, 3); mesh.updateMatrixWorld();
  const camera = new T.PerspectiveCamera(50, 1, 0.1, 10); camera.position.z = 5; camera.updateMatrixWorld();
  mesh.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld);
  const fog = new T.Fog(0x336699, 1, 9);
  const support = createThreeProgramSupport({three: T, state: () => ({...state, pixelRatio: 1, height: 100})});
  const {program: {reflection}} = support.compile(m, mesh, {fog});
  const bytes = new Uint8Array(reflection.uniformBufferSize), view = new DataView(bytes.buffer);
  support.pack(reflection, support.refresh(m, {fog}), mesh, camera, bytes, {});
  const at = name => reflection.uniforms.find(u => u.name === name);
  const f = (o) => view.getFloat32(o, true);
  assert.deepEqual([12, 13, 14].map(i => f(at('modelMatrix').offset + i * 4)), [1, 2, 3]);
  assert.equal(f(at('modelViewMatrix').offset + 14 * 4), 3 - 5);
  assert.deepEqual([0, 1, 2].map(i => f(at('tint').offset + i * 4)), [1, 0.5, 0.25]);
  assert.deepEqual([0, 1, 2].map(i => f(at('k').offset + i * at('k').node.stride)), [1, 2, 3]);
  assert.equal(view.getUint32(at('on').offset, true), 1);
  assert.equal(view.getUint32(at('isOrthographic').offset, true), 0);
  assert.equal(m.uniforms.fogFar.value, 9, 'fog uniforms refreshed into the material, as WebGLRenderer does');
  // Unlit fog color is converted to the canvas output space (getUnlitUniformColorSpace).
  assert.ok(Math.abs(m.uniforms.fogColor.value.r - 0x33 / 255) < 1e-4);
});

test('ShaderLib materials: refreshed uniform clones, light state wiring, PointsMaterial sizes', () => {
  const support = createThreeProgramSupport({three: T, state: () => ({...state, pixelRatio: 2, height: 300})});
  const points = new T.PointsMaterial({size: 6, color: 0x00ff00}), obj = new T.Points(new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(3), 3)), points);
  const u = support.refresh(points);
  assert.notEqual(u, T.ShaderLib.points.uniforms, 'a per-material clone');
  assert.equal(u.size.value, 12); assert.equal(u.scale.value, 150);
  assert.equal(support.compile(points, obj).program.reflection.points, true);
  const lambert = new T.MeshLambertMaterial({color: 0xff0000});
  const light = new T.DirectionalLight(0xffffff, 2); light.position.set(0, 1, 0); light.updateMatrixWorld(); light.target.updateMatrixWorld();
  support.setLights([light, new T.AmbientLight(0x404040, 1)]);
  const {program} = support.compile(lambert, new T.Mesh(new T.BoxGeometry(), lambert));
  assert.ok(program.reflection.uniforms.some(x => x.name === 'directionalLights' && x.node.n === 1), 'NUM_DIR_LIGHTS = 1');
  const lu = support.refresh(lambert);
  assert.equal(lu.directionalLights.value.length, 1); assert.equal(lu.directionalLights.value[0].color.r, 2);
});

test('program support caches compiled programs by exact source text', () => {
  const support = createThreeProgramSupport({three: T, state: () => state});
  const m = shader(), a = new T.Mesh(new T.BoxGeometry(), m), b = new T.Mesh(new T.BoxGeometry(), m);
  assert.equal(support.compile(m, a), support.compile(m, b));
  assert.notEqual(support.compile(m, a), support.compile(m, new T.Mesh(new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(9), 3)), m)),
    'HAS_NORMAL differs, so the program differs');
});

// Execute unchanged r186 program/uniform ownership, source assembly and disposal
// against recording GL allocation calls. Source keys/clones/acquisition are the
// real WebGLPrograms; this is an ownership oracle, not a GPU shader oracle.
async function sourceProgramOwner() {
  const {WebGLPrograms} = await import(pathToFileURL(path.join(upstream, 'src/renderers/webgl/WebGLPrograms.js')));
  const {WebGLProperties} = await import(pathToFileURL(path.join(upstream, 'src/renderers/webgl/WebGLProperties.js')));
  const {WebGLLights} = await import(pathToFileURL(path.join(upstream, 'src/renderers/webgl/WebGLLights.js')));
  const source = fs.readFileSync(path.join(upstream, 'src/renderers/WebGLRenderer.js'), 'utf8');
  const names = new Set(['getProgram', 'updateCommonMaterialProperties', 'materialNeedsLights',
    'onMaterialDispose', 'deallocateMaterial', 'releaseMaterialProgramReferences']), bodies = [];
  simple(parse(source, {ecmaVersion: 'latest', sourceType: 'module'}), {FunctionDeclaration(node) {
    if (names.has(node.id?.name)) bodies.push(source.slice(node.start, node.end));
  }});
  assert.equal(bodies.length, names.size);
  const gl = {VERTEX_SHADER: 'vertex', FRAGMENT_SHADER: 'fragment', createProgram: () => ({}), createShader: type => ({type}),
    shaderSource(shader, text) { shader.source = text; }, compileShader() {}, attachShader() {}, bindAttribLocation() {}, linkProgram() {}, deleteProgram() {}};
  const renderer = {...state, getContext: () => gl, getRenderTarget: () => null, state: {buffers: {depth: {getReversed: () => false}}},
    shadowMap: {enabled: false, type: T.PCFShadowMap}}, environments = {get: texture => texture ?? null};
  const extensions = {has: () => false}, capabilities = {precision: 'highp', getMaxPrecision: value => value};
  const clipping = {numPlanes: 0, numIntersection: 0, uniform: {value: null}}, properties = WebGLProperties();
  const programCache = WebGLPrograms(renderer, environments, extensions, capabilities, {releaseStatesOfProgram() {}}, clipping);
  const currentRenderState = {state: {lights: WebGLLights(extensions), shadowsArray: [], lightProbeGridArray: []}};
  return Function('properties', 'currentRenderState', 'programCache', 'environments', 'clipping', '_this', '_emptyScene',
    'const _nodesHandler = null;\n' + bodies.join('\n') + '\nreturn {compile(material, object) { return getProgram(material, _emptyScene, object); },' +
    'uniformsFor(material) { return properties.get(material).uniforms; }};')(
    properties, currentRenderState, programCache, environments, clipping, renderer, new T.Scene());
}

test('shader hooks retain replaced uniform dictionaries and fresh variant inputs like r186', async t => {
  for (const owner of [await sourceProgramOwner(), createThreeProgramSupport({three: T, state: () => state})]) {
    t.after(() => owner.dispose?.());
    for (const material of [shader(), new T.MeshBasicMaterial()]) {
      const mesh = new T.Mesh(new T.BoxGeometry(), material), inputs = [], outputs = [];
      t.after(() => { material.dispose(); mesh.geometry.dispose(); });
      material.onBeforeCompile = parameters => {
        inputs.push(parameters.uniforms);
        parameters.uniforms = {...parameters.uniforms, injected: {value: outputs.length + 1}};
        outputs.push(parameters.uniforms);
      };
      owner.compile(material, mesh);
      assert.equal(owner.uniformsFor(material), outputs[0], 'the callback-selected dictionary owns the program uniforms');
      owner.compile(material, mesh);
      assert.equal(outputs.length, 1, 'a cached program does not replay the callback');
      material.defines = {SECOND: 1}; material.needsUpdate = true;
      owner.compile(material, mesh);
      assert.equal(owner.uniformsFor(material), outputs[1]);
      assert.equal(inputs[1].injected, undefined, 'a new variant does not inherit the preceding hook dictionary');
      if (material.isShaderMaterial) assert.equal(inputs[1], material.uniforms);
      else assert.notEqual(inputs[1], inputs[0], 'ShaderLib supplies a fresh clone for a new variant');
      material.defines = {}; material.needsUpdate = true;
      owner.compile(material, mesh);
      assert.equal(outputs.length, 2);
      assert.equal(owner.uniformsFor(material), outputs[1], 'r186 keeps the latest dictionary when selecting an older program');
      material.dispose();
      owner.compile(material, mesh);
      assert.equal(outputs.length, 3, 'a new residency reruns the hook after material disposal');
      assert.equal(owner.uniformsFor(material), outputs[2]);
    }
  }
});

test('shader hooks pack replaced numeric and sampler uniforms without changing material.uniforms', t => {
  const support = createThreeProgramSupport({three: T, state: () => state}), material = shader();
  const mesh = new T.Mesh(new T.BoxGeometry(), material), camera = new T.PerspectiveCamera();
  const texture = new T.DataTexture(new Uint8Array([31, 63, 127, 255]), 1, 1), original = material.uniforms;
  const selected = {tint: {value: new T.Color(0.25, 0.5, 0.75)}, map: {value: texture}};
  t.after(() => { material.dispose(); mesh.geometry.dispose(); texture.dispose(); support.dispose?.(); });
  material.onBeforeCompile = parameters => {
    parameters.uniforms = selected;
    parameters.fragmentShader = 'uniform vec3 tint; uniform sampler2D map; void main() { gl_FragColor = vec4(tint, 1.0) * texture2D(map, vec2(0.5)); }';
  };
  const {program: {reflection}} = support.compile(material, mesh), bytes = new Uint8Array(reflection.uniformBufferSize);
  const pack = () => support.pack(reflection, support.refresh(material), mesh, camera, bytes);
  assert.deepEqual(pack(), [texture]);
  const offset = reflection.uniforms.find(u => u.name === 'tint').offset;
  assert.deepEqual(Array.from(new Float32Array(bytes.buffer, offset, 3)), [0.25, 0.5, 0.75]);
  selected.tint.value.setRGB(0.75, 0.25, 0.5); pack();
  assert.deepEqual(Array.from(new Float32Array(bytes.buffer, offset, 3)), [0.75, 0.25, 0.5]);
  assert.equal(material.uniforms, original);
  assert.equal(original.map, undefined, 'the renderer must not publish the replacement onto the material');
});

test('shader hooks preserve the prior owner on callback failure and restore default uniform ownership', async t => {
  for (const owner of [await sourceProgramOwner(), createThreeProgramSupport({three: T, state: () => state})]) {
    const material = shader(), mesh = new T.Mesh(new T.BoxGeometry(), material), original = material.uniforms;
    t.after(() => { material.dispose(); mesh.geometry.dispose(); owner.dispose?.(); });
    const marker = new Error('shader hook failed');
    let reject = false, calls = 0, selected;
    material.onBeforeCompile = parameters => {
      calls++;
      parameters.uniforms = {...parameters.uniforms, injected: {value: calls}};
      if (reject) throw marker;
      selected = parameters.uniforms;
    };
    owner.compile(material, mesh);
    material.defines = {SECOND: 1}; material.needsUpdate = true; reject = true;
    assert.throws(() => owner.compile(material, mesh), error => error === marker);
    assert.equal(owner.uniformsFor(material), selected, 'a failed callback does not publish its dictionary');
    reject = false; owner.compile(material, mesh);
    assert.equal(calls, 3, 'a failed variant remains retryable');
    assert.equal(owner.uniformsFor(material), selected);
    material.onBeforeCompile = T.Material.prototype.onBeforeCompile; material.needsUpdate = true;
    owner.compile(material, mesh);
    assert.equal(owner.uniformsFor(material), original, 'restoring the default hook installs its source dictionary');
  }
});

test('shader hooks release their disposal listeners and keep application textures alive on owner teardown', () => {
  const support = createThreeProgramSupport({three: T, state: () => state}), material = shader();
  const mesh = new T.Mesh(new T.BoxGeometry(), material), texture = new T.Texture(), foreign = () => {};
  let textureDisposals = 0;
  texture.addEventListener('dispose', () => { textureDisposals++; });
  material.addEventListener('dispose', foreign);
  material.onBeforeCompile = parameters => { parameters.uniforms = {map: {value: texture}}; };
  support.compile(material, mesh);
  assert.equal(material._listeners.dispose.length, 2);
  support.dispose(); support.dispose();
  assert.deepEqual(material._listeners.dispose, [foreign]);
  assert.equal(textureDisposals, 0, 'the shader owner does not own the application sampler texture');
  material.dispose(); mesh.geometry.dispose(); texture.dispose();
});

test('shader hooks publish uniforms only after source assembly succeeds and retry failed variants', async t => {
  for (const owner of [await sourceProgramOwner(), createThreeProgramSupport({three: T, state: () => state})]) {
    const material = shader(), mesh = new T.Mesh(new T.BoxGeometry(), material);
    t.after(() => { material.dispose(); mesh.geometry.dispose(); owner.dispose?.(); });
    let failAssembly = false, calls = 0;
    material.onBeforeCompile = parameters => {
      parameters.uniforms = {...parameters.uniforms, injected: {value: ++calls}};
      if (failAssembly) parameters.fragmentShader += '\n#include <f3d_missing_hook_chunk>';
    };
    owner.compile(material, mesh);
    const prior = owner.uniformsFor(material);
    material.defines = {SECOND: 1}; material.needsUpdate = true; failAssembly = true;
    assert.throws(() => owner.compile(material, mesh), /Can not resolve #include <f3d_missing_hook_chunk>/);
    assert.equal(owner.uniformsFor(material), prior);
    material.defines = {}; material.needsUpdate = true;
    owner.compile(material, mesh);
    assert.equal(owner.uniformsFor(material), prior, 'returning to the healthy variant must not bind failed values');
    assert.equal(calls, 2);
    material.defines = {SECOND: 1}; material.needsUpdate = true; failAssembly = false;
    owner.compile(material, mesh);
    assert.equal(calls, 3, 'failed source assembly did not cache a hook result');
    assert.equal(owner.uniformsFor(material).injected.value, 3);
  }
});

test('shader hooks bracket synchronous application callbacks and restore nesting after exceptions', () => {
  const renderer = {_shaderEncodedOutput: true, _programCompileCallbackDepth: 2};
  const support = createThreeProgramSupport({three: T, state: () => ({...state, renderer})}), material = shader();
  const mesh = new T.Mesh(new T.BoxGeometry(), material), marker = new Error('application shader callback');
  let variant = 'a', rejectKey = false, rejectBody = false;
  material.customProgramCacheKey = () => {
    assert.equal(renderer._programCompileCallbackDepth, 3);
    if (rejectKey) throw marker;
    return variant;
  };
  material.onBeforeCompile = (parameters, value) => {
    assert.equal(value, renderer); assert.equal(renderer._programCompileCallbackDepth, 3);
    if (rejectBody) throw marker;
  };
  support.compile(material, mesh); assert.equal(renderer._programCompileCallbackDepth, 2);
  rejectKey = true;
  assert.throws(() => support.compile(material, mesh), error => error === marker);
  assert.equal(renderer._programCompileCallbackDepth, 2);
  rejectKey = false; variant = 'b'; rejectBody = true;
  assert.throws(() => support.compile(material, mesh), error => error === marker);
  assert.equal(renderer._programCompileCallbackDepth, 2);
  rejectBody = false;
  material.customProgramCacheKey = () => ({toString() {
    assert.equal(renderer._programCompileCallbackDepth, 3, 'cache-key coercion is still application callback work');
    return 'coerced';
  }});
  support.compile(material, mesh); assert.equal(renderer._programCompileCallbackDepth, 2);
  support.dispose(); material.dispose(); mesh.geometry.dispose();
});

test('aggregate equality: live ShaderMaterial uniforms and unseen runtime variants use the compiler path', t => {
  const value = () => ({uv: new T.Vector2(2, 3), gains: [5, 7], active: true, flags: [[true, false, true], [false, true, false]]});
  const material = shader({uniforms: {left: {value: value()}, right: {value: value()}},
    fragmentShader: 'struct Payload { vec2 uv; float gains[2]; bool active; bvec3 flags[2]; };\n' +
      'uniform Payload left; uniform Payload right;\nvoid main() { gl_FragColor = vec4(left == right ? 1.0 : 0.0); }'});
  const mesh = new T.Mesh(new T.BoxGeometry(), material), camera = new T.PerspectiveCamera();
  t.after(() => { material.dispose(); mesh.geometry.dispose(); });
  const support = createThreeProgramSupport({three: T, state: () => state});
  const first = support.compile(material, mesh), {program} = first, {reflection} = program;
  assert.match(program.fragment, /fn f3d_equal_Payload\(/);
  assert.match(program.fragment, /f3d_equal_Payload\(f3d_from_u_Payload\(f3d_u.left\), f3d_from_u_Payload\(f3d_u.right\)\)/);
  assert.match(program.fragment, /v\[i\]\.xyz != vec3<u32>\(0u\)/);
  const bytes = new Uint8Array(reflection.uniformBufferSize), view = new DataView(bytes.buffer);
  const pack = () => support.pack(reflection, support.refresh(material), mesh, camera, bytes, {});
  const uniform = name => reflection.uniforms.find(u => u.name === name);
  const right = uniform('right'), gains = right.node.fields.find(f => f.name === 'gains').node;
  const active = right.node.fields.find(f => f.name === 'active').node;
  const flags = right.node.fields.find(f => f.name === 'flags').node;
  pack();
  assert.equal(view.getFloat32(gains.offset + gains.stride, true), 7);
  assert.equal(view.getUint32(active.offset, true), 1);
  assert.deepEqual([0, 4, 8, flags.stride, flags.stride + 4, flags.stride + 8].map(offset => view.getUint32(flags.offset + offset, true)), [1, 0, 1, 0, 1, 0]);
  material.uniforms.right.value.gains[1] = 11; material.uniforms.right.value.active = false; material.uniforms.right.value.flags[1][2] = true;
  pack();
  assert.equal(view.getFloat32(gains.offset + gains.stride, true), 11);
  assert.equal(view.getUint32(active.offset, true), 0);
  assert.equal(view.getUint32(flags.offset + flags.stride + 8, true), 1);
  assert.equal(support.compile(material, mesh), first, 'uniform edits preserve the compiled program');
  material.fragmentShader = material.fragmentShader.replace('left == right', 'left != right'); material.needsUpdate = true;
  const second = support.compile(material, mesh);
  assert.notEqual(second, first, 'an unvisited runtime shader variant is compiled');
  assert.match(second.program.fragment, /!f3d_equal_Payload\(/);
});

test('matrix-array packing retains live values and compiler-only nested layouts', t => {
  // Global mat2[] values use the r186 material upload contract. The nested
  // struct arrays below probe compiler/packer layout only: this nested fixture
  // has not established source pixel parity.
  const material = shader({glslVersion: T.GLSL3,
    uniforms: {lead: {value: 17}, frames: {value: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])},
      groups: {value: [{lead: 19, frames: [21, 22, 23, 24, 25, 26, 27, 28], tail: 29},
        {lead: 31, frames: [41, 42, 43, 44, 45, 46, 47, 48], tail: 49}]}, tail: {value: 59}},
    fragmentShader: `uniform float lead; uniform mat2 frames[2];
      struct FrameSet { float lead; mat2 frames[2]; float tail; };
      uniform FrameSet groups[2]; uniform float tail;
      out vec4 color;
      void main() { mat2 copy[2] = frames; FrameSet selected = groups[1];
        color = vec4(copy[1][1].y, selected.frames[1][1].x, selected.tail, lead + tail); }`});
  const mesh = new T.Mesh(new T.BoxGeometry(), material), camera = new T.PerspectiveCamera();
  const support = createThreeProgramSupport({three: T, state: () => state});
  t.after(() => { support.dispose(); material.dispose(); mesh.geometry.dispose(); });
  const compiled = support.compile(material, mesh), {reflection} = compiled.program;
  const by = Object.fromEntries(reflection.uniforms.map(u => [u.name, u]));
  const bytes = new Uint8Array(reflection.uniformBufferSize), view = new DataView(bytes.buffer);
  const pack = () => support.pack(reflection, support.refresh(material), mesh, camera, bytes, {});
  const f32 = offset => view.getFloat32(offset, true);
  const readFrames = (node, base = 0) => Array.from({length: 8}, (_, i) =>
    f32(base + node.offset + Math.floor(i / 4) * node.stride + Math.floor(i % 4 / 2) * node.elem.colStride + i % 2 * 4));
  pack();
  assert.equal(by.frames.offset % 16, 0);
  assert.equal(by.frames.node.stride, 16); assert.equal(by.frames.node.elem.colStride, 8);
  assert.deepEqual(readFrames(by.frames.node), [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(f32(by.lead.offset), 17); assert.equal(f32(by.tail.offset), 59);
  const fields = Object.fromEntries(by.groups.node.elem.fields.map(f => [f.name, f.node]));
  assert.equal(fields.frames.offset, 16); assert.equal(by.groups.node.stride, 64);
  for (let i = 0; i < 2; i++) {
    const base = by.groups.offset + i * by.groups.node.stride, value = material.uniforms.groups.value[i];
    assert.deepEqual(readFrames(fields.frames, base), value.frames);
    assert.equal(f32(base + fields.lead.offset), value.lead);
    assert.equal(f32(base + fields.tail.offset), value.tail);
    assert.deepEqual([...bytes.slice(base + 4, base + 16)], Array(12).fill(0), 'alignment padding is not source matrix data');
  }
  material.uniforms.frames.value[7] = 61;
  material.uniforms.groups.value[1].frames[6] = 67;
  material.uniforms.groups.value[1].tail = 71;
  pack();
  assert.equal(readFrames(by.frames.node)[7], 61);
  const second = by.groups.offset + by.groups.node.stride;
  assert.equal(readFrames(fields.frames, second)[6], 67);
  assert.equal(f32(second + fields.tail.offset), 71);
  assert.equal(support.compile(material, mesh), compiled, 'matrix values remain data, not program variants');
});

test('rectangular ESSL matrix arrays pack each column and leave element padding clear', () => {
  // r186 WebGLUniforms only uploads square matrix types. This exercises the
  // compiler/packing API's ESSL type support, not an upstream material claim.
  const {reflection} = compileEsslProgram(
    '#version 300 es\nin vec3 position; void main() { gl_Position = vec4(position, 1.0); }',
    '#version 300 es\nprecision highp float; uniform float lead; uniform mat3x2 frames[2]; uniform float tail;' +
      'out vec4 color; void main() { color = vec4(frames[1][2], lead, tail); }');
  const by = Object.fromEntries(reflection.uniforms.map(u => [u.name, u]));
  const mesh = new T.Mesh(), camera = new T.PerspectiveCamera(), bytes = new Uint8Array(reflection.uniformBufferSize);
  const values = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  packThreeProgramUniforms(T, reflection, {lead: {value: 17}, frames: {value: values}, tail: {value: 19}}, mesh, camera, bytes);
  const view = new DataView(bytes.buffer), at = by.frames.offset;
  assert.equal(at, 16); assert.equal(by.frames.node.stride, 32); assert.equal(by.frames.node.elem.colStride, 8);
  for (let i = 0; i < 2; i++) {
    assert.deepEqual(Array.from({length: 6}, (_, j) => view.getFloat32(at + i * 32 + j * 4, true)), [...values.slice(i * 6, i * 6 + 6)]);
    assert.deepEqual([...bytes.slice(at + i * 32 + 24, at + (i + 1) * 32)], Array(8).fill(0));
  }
  assert.equal(view.getFloat32(by.tail.offset, true), 19);
  mesh.geometry.dispose(); mesh.material.dispose();
});

test('UniformsGroup converts source std140 rectangular matrix arrays into compact draw data', t => {
  // r186 accepts raw typed-array UniformsGroup data even though its ordinary
  // WebGLUniforms setters do not accept non-square matrix uniform types.
  const source = new Float32Array(32), values = Array.from({length: 12}, (_, i) => i + 1);
  source[0] = 17; source[28] = 19;
  for (let i = 0; i < 2; i++) for (let c = 0; c < 3; c++) for (let row = 0; row < 2; row++)
    source[4 + i * 12 + c * 4 + row] = values[i * 6 + c * 2 + row];
  const group = new T.UniformsGroup().setName('Matrices').add(new T.Uniform(source));
  const material = new T.RawShaderMaterial({glslVersion: T.GLSL3, uniformsGroups: [group],
    vertexShader: 'precision highp float; in vec3 position; void main() { gl_Position = vec4(position, 1.0); }',
    fragmentShader: `precision highp float;
      layout(std140) uniform Matrices { float lead; mat3x2 frames[2]; float tail; } matrices;
      out vec4 color; void main() { mat3x2 copy[2] = matrices.frames;
        color = vec4(copy[1][2], matrices.lead, matrices.tail); }`});
  const mesh = new T.Mesh(new T.BoxGeometry(), material), camera = new T.PerspectiveCamera();
  const support = createThreeProgramSupport({three: T, state: () => state});
  t.after(() => { support.dispose(); group.dispose(); material.dispose(); mesh.geometry.dispose(); });
  const {program: {reflection}} = support.compile(material, mesh);
  const by = Object.fromEntries(reflection.uniforms.map(u => [u.name, u]));
  const bytes = new Uint8Array(reflection.uniformBufferSize), view = new DataView(bytes.buffer);
  const pack = () => support.pack(reflection, support.refresh(material), mesh, camera, bytes, {material});
  const frames = by.matrices_frames, valueAt = i => view.getFloat32(frames.offset + Math.floor(i / 6) * frames.node.stride + i % 6 * 4, true);
  pack();
  assert.equal(group.__size, 128, 'the retained source group keeps std140 column/array padding');
  assert.equal(reflection.uniformBufferSize, 96, 'each draw uses the reflected WGSL layout');
  assert.equal(frames.offset, 16); assert.equal(frames.node.stride, 32);
  assert.deepEqual(values.map((_, i) => valueAt(i)), values);
  assert.equal(view.getFloat32(by.matrices_lead.offset, true), 17);
  assert.equal(view.getFloat32(by.matrices_tail.offset, true), 19);
  source[25] = 23;
  pack(); assert.equal(valueAt(11), 12, 'a source group is uploaded once per render frame');
  support.beginFrame(); pack(); assert.equal(valueAt(11), 23);
});

test('PMREM: r186 PMREMGenerator passes record against the host and compile with GL rows', async () => {
  const {createThreeProgramPMREM, pmremSourceComplete, pmremSourceKind} = await import('./three_program_pmrem.mjs');
  const cube = new T.CubeTexture([0, 1, 2, 3, 4].map(() => ({width: 32, height: 32})));
  assert.equal(pmremSourceKind(T, cube), 'cube');
  assert.equal(pmremSourceComplete(T, cube), false, 'five faces: r186 renders without the environment');
  cube.images[5] = {width: 32, height: 32};
  assert.equal(pmremSourceComplete(T, cube), true);
  const equirect = new T.DataTexture(new Uint8Array(4), 1, 1); equirect.mapping = T.EquirectangularReflectionMapping;
  assert.equal(pmremSourceKind(T, equirect), 'equirect');
  assert.equal(pmremSourceKind(T, new T.Texture()), 'direct');
  // Without a device the generation fails at the GPU step, after recording.
  const calls = [];
  const device = new Proxy({}, {get: (_, k) => k === 'pushErrorScope' ? () => {} : k === 'popErrorScope' ? async () => null : k === 'createBuffer' ? () => { throw new Error('no-gpu'); } : (...a) => calls.push(k)});
  const pmrem = createThreeProgramPMREM({three: T, device, bindingOf: () => null});
  assert.deepEqual(pmrem.lookup(cube), {state: 'needed'});
  await assert.rejects(pmrem.generate(cube), /no-gpu/);
  const rt = new T.WebGLRenderTarget(4, 4); rt.texture.mapping = T.CubeReflectionMapping;
  assert.deepEqual(pmrem.lookup(rt.texture), {state: 'incomplete'});
  rt.texture.needsPMREMUpdate = true;
  assert.deepEqual(pmrem.lookup(rt.texture), {state: 'needed'}, 'GPU-authored versions bypass CPU image readiness, as r186 does');
});

test('WebGLLights shadow state: r186 sorting, shadow uniforms, maps by light type, receiver sources', () => {
  const support = createThreeProgramSupport({three: T, state: () => ({...state, pixelRatio: 1, height: 1})});
  const dir = new T.DirectionalLight(), spot = new T.SpotLight(), point = new T.PointLight(), plain = new T.DirectionalLight();
  for (const l of [dir, spot, point]) l.castShadow = true;
  dir.shadow.bias = -0.001; point.shadow.camera.far = 42;
  point.shadow.map = {texture: {format: T.RGBAFormat}, depthTexture: new T.CubeDepthTexture(8)};
  support.setLights([plain, dir, spot, point]);
  const lights = support.refresh(new T.MeshStandardMaterial());
  assert.equal(lights.directionalLights.value.length, 2);
  assert.equal(lights.directionalLightShadows.value.length, 1, 'only casting lights get shadow uniforms');
  assert.equal(lights.directionalLightShadows.value[0].shadowBias, -0.001);
  assert.equal(lights.pointLightShadows.value[0].shadowCameraFar, 42);
  const samplers = support.shadowSamplers(new T.MeshLambertMaterial());
  assert.equal(samplers.pointShadowMap[0], point.shadow.map.depthTexture, 'depth texture, as WebGLLights selects it');
  assert.equal(samplers.directionalShadowMap[0], null, 'not yet rendered');
  assert.equal(support.shadowSamplers(new T.MeshBasicMaterial()), null, 'unlit materials bind no shadow maps');
  const {program} = support.compile(new T.MeshLambertMaterial(), new T.Mesh(new T.BoxGeometry()), {shadows: true});
  assert.ok(program.reflection.textures.some(t => t.name === 'pointShadowMap' && t.comparison && t.dimension === 'cube'));
  assert.ok(program.reflection.uniforms.some(u => u.name === 'receiveShadow'));
});
