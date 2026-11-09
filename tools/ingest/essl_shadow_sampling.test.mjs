/** Projected comparison sampling through real ESSL/source-program preparation.
 * The native pipeline device records bindings; it does not execute WGSL. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {compileEsslProgram} from './essl_wgsl.mjs';
import {createThreeProgramSupport, packThreeProgramUniforms} from './three_program.mjs';
import {createProgramMeshes} from './animation_program_mesh.mjs';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const vertex = '#version 300 es\nprecision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.);}';
const fragment = (body, declarations = 'uniform highp sampler2DShadow image;') =>
  '#version 300 es\nprecision highp float;precision highp int;out vec4 color;\n' + declarations + '\nvoid main(){' + body + '}';
const compile = (expression, declarations) => compileEsslProgram(vertex, fragment(`color=vec4(${expression});`, declarations));

for (const offset of [false, true]) test(`projected shadow ${offset ? 'offset ' : ''}sampling divides the comparison reference with the coordinates`, () => {
  const name = offset ? 'textureProjOffset' : 'textureProj';
  const result = compile(`${name}(image,vec4(.4,1.2,.6,2.)${offset ? ',ivec2(1,-1)' : ''})`);
  assert.match(result.fragment, /let (f3d_t\d+) = (f3d_t\d+)\.xyz \/ \2\.w;/);
  const projected = /let (f3d_t\d+) = f3d_t\d+\.xyz \/ f3d_t\d+\.w;/.exec(result.fragment)[1];
  assert.match(result.fragment, new RegExp(`textureSampleCompare\\(image_texture, image_sampler, ${projected}\\.xy, select\\(${projected}\\.z, clamp\\(${projected}\\.z, 0.0, 1.0\\), f3d_u.f3d_depth_reference_clamp_0.x != 0u\\)`));
  if (offset) assert.match(result.fragment, /, vec2<i32>\(1, \(-1\)\)\)/);
  assert.deepEqual(result.reflection.textures, [{name: 'image', element: null,
    glslType: 'sampler2DShadow', dimension: '2d', sampleType: 'depth', comparison: true,
    textureBinding: 0, samplerBinding: 1, depthReferenceClampOffset: 0}]);
  assert.match(result.fragment, /var image_texture: texture_depth_2d;/);
  assert.match(result.fragment, /var image_sampler: sampler_comparison;/);
});

test('projected vertex comparisons use the source base level without implicit derivatives', () => {
  for (const expression of ['textureProj(image,vec4(.4,1.2,.6,2.))',
    'textureProjOffset(image,vec4(.4,1.2,.6,2.),ivec2(-1,1))']) {
    const source = '#version 300 es\nprecision highp float;in vec3 position;uniform highp sampler2DShadow image;\n' +
      `void main(){gl_Position=vec4(position*${expression},1.);}`;
    const result = compileEsslProgram(source, fragment('color=vec4(1.);', ''));
    assert.match(result.vertex, /textureSampleCompareLevel\(image_texture, image_sampler, f3d_t\d+\.xy, select\(f3d_t\d+\.z,/);
    assert.doesNotMatch(result.vertex, /textureSampleCompare\(|dpdx|dpdy/);
    assert.equal(result.reflection.textures[0].comparison, true);
  }
});

test('projected shadow helpers and sampler arrays retain single coordinate evaluation and constant offsets', () => {
  const result = compile('tap(images[1],point())', `uniform highp sampler2DShadow images[2];
    int calls=0;vec4 point(){calls++;return vec4(.4,1.2,float(calls)*.6,2.);}
    float tap(highp sampler2DShadow image,vec4 p){
      const int stepSize=1;const ivec2 delta=ivec2(stepSize,-stepSize);
      return textureProjOffset(image,p,delta);
    }`);
  assert.match(result.fragment, /fn tap\(image_texture: texture_depth_2d, image_sampler: sampler_comparison, image_depth_reference_clamp: u32, p_in: vec4<f32>\)/);
  const pointResult = /let (f3d_t\d+) = point\(\);/.exec(result.fragment)?.[1];
  assert.ok(pointResult, 'the helper coordinate is evaluated before the call');
  assert.match(result.fragment, new RegExp(`tap\\(images_1_texture, images_1_sampler, f3d_u.f3d_depth_reference_clamp_1.x, ${pointResult}\\)`));
  assert.equal((result.fragment.match(/point\(\)/g) ?? []).length, 2, 'one declaration and one coordinate call');
  assert.match(result.fragment, /textureSampleCompare\(image_texture, image_sampler, f3d_t\d+\.xy, select\(f3d_t\d+\.z, clamp\(f3d_t\d+\.z, 0.0, 1.0\), image_depth_reference_clamp != 0u\), delta\)/);
  const direct = compile('textureProjOffset(image,point(),ivec2(1,-1))',
    'uniform highp sampler2DShadow image;int calls=0;vec4 point(){calls++;return vec4(.4,1.2,.6,2.);}');
  assert.equal((direct.fragment.match(/= point\(\);/g) ?? []).length, 1);
});

test('ordinary depth comparisons carry the same format-specific reference control through nested helpers', () => {
  for (const [type, coordinate, reference] of [
    ['sampler2DShadow', 'vec3(.4,.6,-1.)', 'z'],
    ['samplerCubeShadow', 'vec4(.2,.4,.6,2.)', 'w'],
    ['sampler2DArrayShadow', 'vec4(.4,.6,1.,2.)', 'w'],
  ]) {
    const result = compile('outer(images[0])+outer(images[1])', `uniform highp ${type} images[2];
      float inner(highp ${type} item){return texture(item,${coordinate});}
      float outer(highp ${type} item){return inner(item);}`);
    assert.deepEqual(result.reflection.textures.map(t => t.depthReferenceClampOffset), [0, 16]);
    assert.match(result.fragment, /inner\(item_texture, item_sampler, item_depth_reference_clamp\)/);
    assert.match(result.fragment, /outer\(images_0_texture, images_0_sampler, f3d_u.f3d_depth_reference_clamp_0.x\)/);
    assert.match(result.fragment, /outer\(images_1_texture, images_1_sampler, f3d_u.f3d_depth_reference_clamp_1.x\)/);
    assert.match(result.fragment, new RegExp(`select\\(f3d_t\\d+\\.${reference}, clamp\\(f3d_t\\d+\\.${reference}, 0.0, 1.0\\), item_depth_reference_clamp != 0u\\)`));
  }
  const offset = compile('textureOffset(image,vec3(.5,.5,-1.),ivec2(1,-1))');
  assert.match(offset.fragment, /textureSampleCompare\(image_texture, image_sampler, f3d_t\d+\.xy, select\(f3d_t\d+\.z, clamp\(f3d_t\d+\.z, 0.0, 1.0\), f3d_u.f3d_depth_reference_clamp_0.x != 0u\),/);
});

test('vertex shadow offsets use base-level comparisons with typed constant offsets and unchanged array layers', () => {
  for (const [type, coordinate] of [['sampler2DShadow','vec3(.5,.5,-1.)'],['sampler2DArrayShadow','vec4(.5,.5,1.,-1.)']]) {
    const source = '#version 300 es\nprecision highp float;precision highp int;in vec3 position;' +
      `uniform highp ${type} image;void main(){const ivec2 delta=ivec2(-8,7);` +
      `gl_Position=vec4(position*textureOffset(image,${coordinate},delta),1.);}`;
    const result = compileEsslProgram(source, fragment('color=vec4(1.);',''));
    assert.match(result.vertex,/textureSampleCompareLevel\(image_texture, image_sampler,/);
    assert.match(result.vertex,/, delta\)/);assert.doesNotMatch(result.vertex,/textureSampleCompare\(|dpdx|dpdy/);
    if(type==='sampler2DArrayShadow')assert.match(result.vertex,/\.xy, i32\(floor\(f3d_t\d+\.z \+ 0.5\)\), select\(f3d_t\d+\.w,/);
  }
  for(const expression of ['textureOffset(image,vec3(.5),vec2(1.))','textureOffset(image,vec3(.5),ivec3(1))',
    'textureOffset(image,vec4(.5),ivec2(1))','textureOffset(image,vec3(.5))','textureOffset(image,vec3(.5),ivec2(1),1.)'])
    assert.throws(()=>compile(expression),/ESSL line/);
  assert.throws(()=>compile('textureOffset(image,vec3(.5),delta)','uniform highp sampler2DShadow image;uniform ivec2 delta;'),/constant expression/);
  assert.throws(()=>compile('textureOffset(image,vec4(.5),ivec2(1))','uniform highp samplerCubeShadow image;'),/2D or 2D-array/);
});

test('projected shadow overloads require ES3 2D depth comparisons and complete typed arguments', () => {
  for (const expression of ['textureProj(image)', 'textureProj(image,vec3(.5))',
    'textureProj(image,ivec4(1))', 'textureProjOffset(image,vec4(.5))',
    'textureProjOffset(image,vec4(.5),vec2(1.))', 'textureProjOffset(image,vec4(.5),ivec3(1))'])
    assert.throws(() => compile(expression), /ESSL line/);
  for (const sampler of ['samplerCubeShadow', 'sampler2DArrayShadow'])
    assert.throws(() => compile('textureProj(image,vec4(.5))', `uniform highp ${sampler} image;`), /sampler2DShadow/);
  assert.throws(() => compileEsslProgram(vertex, fragment('ivec2 delta=ivec2(1);color=vec4(textureProjOffset(image,vec4(.5),delta));')),
    /constant expression/);
  assert.throws(() => compile('textureProjOffset(image,vec4(.5),delta)',
    'uniform highp sampler2DShadow image;uniform ivec2 delta;'), /constant expression/);
  for (const name of ['textureProj', 'texture2DProj'])
    assert.throws(() => compileEsslProgram('attribute vec3 position;void main(){gl_Position=vec4(position,1.);}',
      `precision highp float;uniform sampler2DShadow image;void main(){gl_FragColor=vec4(${name}(image,vec4(.5)));}`), /requires ESSL 3.00/);
});

test('projected shadow bias, explicit gradients and nonzero explicit LOD never become an implicit or level-zero approximation', () => {
  for (const expression of ['textureProj(image,vec4(.5),1.)', 'textureProjOffset(image,vec4(.5),ivec2(1),1.)'])
    assert.throws(() => compile(expression), /with bias is not supported/);
  for (const expression of ['textureProjLod(image,vec4(.5),.5)',
    'textureProjLodOffset(image,vec4(.5),1.,ivec2(1))',
    'textureProjGrad(image,vec4(.5),vec2(.1),vec2(.2))',
    'textureProjGradOffset(image,vec4(.5),vec2(.1),vec2(.2),ivec2(1))'])
    assert.throws(() => compile(expression), /explicit LOD or gradients|proven constant zero/);
  for (const expression of ['textureProj(image,vec4(.5),1.)', 'textureProjOffset(image,vec4(.5),ivec2(1),1.)']) {
    const source = vertex.replace('void main()', 'uniform highp sampler2DShadow image;void main()')
      .replace('vec4(position,1.)', `vec4(position*${expression},1.)`);
    assert.throws(() => compileEsslProgram(source, fragment('color=vec4(1.);', '')), /only valid in fragment shaders/);
  }
});

test('ordinary shadow bias and nonzero explicit LOD or gradients never silently discard source arguments', () => {
  assert.throws(()=>compile('texture(image,vec3(.5),1.)'),/with bias is not supported/);
  for(const expression of ['textureLod(image,vec3(.5),.5)','textureLod(image,vec3(.5),2.)',
    'textureLodOffset(image,vec3(.5),.5,ivec2(1))','textureGrad(image,vec3(.5),vec2(.1),vec2(.2))',
    'textureGradOffset(image,vec3(.5),vec2(.1),vec2(.2),ivec2(1))'])
    assert.throws(()=>compile(expression),/explicit LOD or gradients|proven constant zero/);
  for(const expression of ['texture(image)','texture(image,vec2(.5))','texture(image,ivec3(1))'])
    assert.throws(()=>compile(expression),/needs vec3 coordinates/);
});

async function pinnedThree(t) {
  const source = process.env.F3D_THREE_ROOT
    ? pathToFileURL(path.join(process.env.F3D_THREE_ROOT, 'build/three.module.js'))
    : new URL('../../upstream/three.js/build/three.module.js', import.meta.url);
  if (!fs.existsSync(source)) {
    assert.equal(process.env.F3D_THREE_ROOT, undefined, 'F3D_THREE_ROOT must contain the pinned Three build');
    t.skip('Requires the pinned upstream checkout'); return null;
  }
  const T = await import(source.href);
  assert.equal(T.REVISION, '186');
  return T;
}

test('comparison controls use admitted image formats and do not disturb authored or integer uniforms', async t => {
  const T = await pinnedThree(t); if (!T) return;
  const result = compile('textureProj(images[0],vec4(.4,.6,-1.,2.)) + texture(images[1],vec3(.5,.5,2.)) + float(texture(bits,vec2(.5)).x)*exposure',
    'uniform float exposure;uniform highp sampler2DShadow images[2];uniform highp usampler2D bits;');
  const fixed = new T.DepthTexture(2, 2, T.UnsignedIntType), floating = new T.DepthTexture(2, 2, T.FloatType);
  const integer = new T.DataTexture(new Uint32Array(4), 2, 2, T.RedIntegerFormat, T.UnsignedIntType);
  const object = new T.Mesh(), camera = new T.PerspectiveCamera(), bytes = new Uint8Array(result.reflection.uniformBufferSize);
  const uniforms = {images: {value: [fixed, floating]}, bits: {value: integer}, exposure: {value: .75}};
  const bindings = [{depthReferenceClamp: true}, {depthReferenceClamp: false},
    {samplerDescriptor: {addressModeU: 'repeat', addressModeV: 'mirror-repeat', lodMaxClamp: 2}}];
  const controls = i => Array.from(new Uint32Array(bytes.buffer, result.reflection.textures[i].depthReferenceClampOffset, 4));
  try {
    assert.deepEqual(result.reflection.textures.map(t => t.depthReferenceClampOffset ?? t.samplingOffset), [16, 32, 48]);
    assert.deepEqual(packThreeProgramUniforms(T, result.reflection, uniforms, object, camera, bytes, {textureBindings: bindings}), [fixed, floating, integer]);
    assert.deepEqual(controls(0), [1, 0, 0, 0]); assert.deepEqual(controls(1), [0, 0, 0, 0]);
    assert.deepEqual(Array.from(new Uint32Array(bytes.buffer, 48, 4)), [1, 2, 0, 1]);
    assert.equal(new DataView(bytes.buffer).getFloat32(result.reflection.uniforms[0].offset, true), .75);
    // Public type edits are not the authority for the retained native image.
    fixed.type = T.FloatType; floating.type = T.UnsignedIntType;
    packThreeProgramUniforms(T, result.reflection, uniforms, object, camera, bytes, {textureBindings: bindings});
    assert.deepEqual(controls(0), [1, 0, 0, 0]); assert.deepEqual(controls(1), [0, 0, 0, 0]);
    bindings[0] = {}; bindings[1] = {depthReferenceClamp: true};
    packThreeProgramUniforms(T, result.reflection, uniforms, object, camera, bytes, {textureBindings: bindings});
    assert.deepEqual(controls(0), [0, 0, 0, 0], 'private floating shadows default to unclamped references');
    assert.deepEqual(controls(1), [1, 0, 0, 0]);
  } finally {fixed.dispose(); floating.dispose(); integer.dispose(); object.geometry.dispose(); object.material.dispose();}
});

test('pinned raw and converted material programs bind native depth views and comparison samplers', async t => {
  const T = await pinnedThree(t); if (!T) return;
  const device = geometryDevice(); device.features = new Set();
  const support = createThreeProgramSupport({three: T, state: () => ({toneMapping: T.NoToneMapping, outputColorSpace: T.LinearSRGBColorSpace})});
  const owner = createProgramMeshes({device, format: 'rgba8unorm', depthFormat: null, sampleCount: 1,
    maxDraws: 4, label: 'projected-shadow', lost: device.lost,
    fail(code, message) {throw Object.assign(Error(message), {code});},
    scoped(_device, operation) {return {value: operation(), errors: Promise.resolve()};}});
  const geometry = new T.PlaneGeometry(2, 2), depth = new T.DepthTexture(8, 8, T.FloatType);
  depth.compareFunction = T.LessEqualCompare;
  // The renderer's native target owner supplies these GPU boundary objects.
  // Source target residency/orientation is covered by the renderer integration.
  const view = {format: 'depth32float'}, sampler = {compare: 'less-equal', minFilter: 'nearest', magFilter: 'nearest'};
  try {
    for (const raw of [false, true]) {
      const material = new (raw ? T.RawShaderMaterial : T.ShaderMaterial)({
        ...(raw ? {glslVersion: T.GLSL3} : {}), depthTest: false, depthWrite: false, blending: T.NoBlending,
        vertexShader: (raw ? 'precision highp float;in vec3 position;' : '') + 'void main(){gl_Position=vec4(position,1.);}',
        fragmentShader: 'precision highp float;uniform highp sampler2DShadow image;' + (raw ? 'out vec4 color;' : '') +
          `void main(){${raw ? 'color' : 'gl_FragColor'}=vec4(textureProjOffset(image,vec4(.4,1.2,.6,2.),ivec2(1,-1)));}`,
        uniforms: {image: {value: depth}}});
      let gpu;
      try {
        const mesh = new T.Mesh(geometry, material), {program} = support.compile(material, mesh);
        gpu = createGpuProgramGeometry(device, geometry, program.reflection.attributes);
        const record = await owner.add(gpu, {program, raster: support.raster(material),
          textures: [{view, sampler, sampleType: 'depth'}]});
        for (const pipeline of Object.values(record.pipelines)) {
          assert.match(pipeline.fragment.module.code, /textureSampleCompare\(/);
          assert.deepEqual(pipeline.layout.bindGroupLayouts[1].entries, [
            {binding: 0, visibility: 3, texture: {sampleType: 'depth', viewDimension: '2d'}},
            {binding: 1, visibility: 3, sampler: {type: 'comparison'}}]);
        }
        assert.equal(record.textureGroup.entries[0].resource, view);
        assert.equal(record.textureGroup.entries[1].resource, sampler);
      } finally {gpu?.dispose(); material.dispose();}
    }
    await owner.whenIdle();
  } finally {owner.dispose(); support.dispose(); geometry.dispose(); depth.dispose();}
});
