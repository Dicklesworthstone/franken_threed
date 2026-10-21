import assert from 'node:assert/strict';
import test from 'node:test';
import {ENVIRONMENT_UNIFORM_BYTES, environmentLightingWgsl, packAnimationEnvironment} from './animation_environment_receiver.mjs';

const reject = (code, message) => { throw Object.assign(new Error(message), {code}); };
const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
function fixture() {
  const device = {}, snapshot = Object.freeze({profile: 'f3d-animation-environment-v1', version: 1,
    diffuseView: {}, specularView: {}, brdfView: {}, sampler: {}, mipLevelCount: 8});
  let calls = 0, disposed = false;
  const map = {sample(d) { assert.equal(d, device); calls++; if (disposed) throw Error('disposed'); return snapshot; },
    async whenIdle() {}, dispose() { disposed = true; }};
  return {device, map, snapshot, get calls() { return calls; }};
}

test('scattering is a live flag in the existing 64-byte environment uniform', () => {
  const f = fixture(), out = new Float32Array(ENVIRONMENT_UNIFORM_BYTES / 4).fill(9);
  const multiple = packAnimationEnvironment(f.device, {map: f.map, scattering: 'multiple'}, out, reject);
  assert.deepEqual([...out], [1,0,0,0,0,1,0,0,0,0,1,0,1,7,1,0]);
  const copy = out.slice();
  const single = packAnimationEnvironment(f.device, {map: f.map, scattering: 'single'}, out, reject);
  assert.deepEqual([...out], [1,0,0,0,0,1,0,0,0,0,1,0,1,7,0,0]);
  assert.equal(copy[14], 1, 'earlier upload bytes are independent');
  assert.equal(single.snapshot, multiple.snapshot); assert.equal(single.map, multiple.map);
  single.check(); multiple.check();
  f.map.dispose(); assert.throws(() => multiple.check(), /disposed/);
});

test('omitting scattering retains the original uniform bytes and clears a prior multiple flag', () => {
  const f = fixture(), out = new Float32Array(16).fill(42);
  packAnimationEnvironment(f.device, {map: f.map, scattering: 'multiple'}, out, reject);
  packAnimationEnvironment(f.device, {map: f.map, intensity: 3, rotation: identity}, out, reject);
  assert.deepEqual([...out], [1,0,0,0,0,1,0,0,0,0,1,0,3,7,0,0]);
});

for (const scattering of [null, false, true, 0, 1, '', 'auto', 'MULTIPLE', {}, ['multiple']])
test(`invalid scattering ${JSON.stringify(scattering)} rejects before sampling or packing`, () => {
  const f = fixture(), out = new Float32Array(16).fill(42);
  assert.throws(() => packAnimationEnvironment(f.device, {map: f.map, scattering}, out, reject),
    {code: 'ANIMATION_RENDER_ENVIRONMENT'});
  assert.equal(f.calls, 0); assert.deepEqual([...out], Array(16).fill(42));
});

test('multiple mode retains validation and snapshot lifetime checks atomically', () => {
  const f = fixture(), out = new Float32Array(16).fill(42);
  for (const settings of [{intensity: -1}, {intensity: Infinity}, {rotation: identity.map(x => x * 2)}, {extra: 1}]) {
    assert.throws(() => packAnimationEnvironment(f.device, {map: f.map, scattering: 'multiple', ...settings}, out, reject),
      {code: 'ANIMATION_RENDER_ENVIRONMENT'});
    assert.deepEqual([...out], Array(16).fill(42));
  }
  let current = f.snapshot;
  const map = {sample: () => current, whenIdle: async () => {}};
  const packed = packAnimationEnvironment(f.device, {map, scattering: 'multiple'}, out, reject);
  current = Object.freeze({...current});
  assert.throws(packed.check, {code: 'ANIMATION_RENDER_ENVIRONMENT'});
});

// Execute the scalar, component-independent arithmetic emitted by the production
// WGSL generator. This is deliberately NOT a WGSL compiler or native GPU test.
// Vector constructors become scalars; the only two vec2 operations are expanded
// explicitly. All compensation, mixing and substrate expressions remain verbatim.
function scalarShader(physical = false) {
  const source = environmentLightingWgsl(2, physical);
  const body = /fn environment_multiple_scattering\([^]*?-> vec3<f32> \{([^]*?)\n\}/.exec(source)?.[1];
  const branch = /if \(environment_info.options.z == 1.0\) \{([^]*?)\n  \}/.exec(source)?.[1];
  assert.ok(body && branch, 'production shader contains both computations');
  const scalar = text => text.replaceAll('vec3<f32>', 'Number');
  const transformed = scalar(branch)
    .replace('max(dfg, vec2<f32>(0.0))', '({x: max(dfg.x, 0), y: max(dfg.y, 0)})')
    .replace('positive_dfg / max(1.0, positive_dfg.x + positive_dfg.y)',
      '({x: positive_dfg.x / max(1, positive_dfg.x + positive_dfg.y), y: positive_dfg.y / max(1, positive_dfg.x + positive_dfg.y)})');
  const functions = `const max=Math.max, clamp=(x,a,b)=>Math.min(b,Math.max(a,x)),
    mix=(a,b,t)=>a*(1-t)+b*t, select=(a,b,p)=>p?b:a;
    function environment_multiple_scattering(f0,f90,dfg){${scalar(body)}}`;
  return {
    multiple: new Function('f0','f90','dfg', functions + 'return environment_multiple_scattering(f0,f90,dfg);'),
    lighting: new Function('base','metallic','dfg','radiance','irradiance','dielectric_f0','specular_f90','intensity',
      functions + 'const environment_info={options:{x:intensity}};' + transformed),
    source,
  };
}

// Independent oracle: explicitly sum successive microfacet bounces rather than
// copying the implementation's closed-form division. Each bounce loses 1-E of
// its angular energy, then reflects with the hemisphere-averaged Schlick Fresnel.
function bounceSum(f0, f90, a, b) {
  const single = f0 * a + f90 * b;
  // Numerical integration of 2*cos(theta)*sin(theta)*F(theta) gives 1/21 for
  // Schlick's grazing term. The exact integral is used for the independent sum.
  const average = (20 * f0 + f90) / 21;
  let term = single, sum = 0;
  for (let bounce = 0; bounce < 10000; bounce++) {
    term *= average * (1 - a - b); sum += term;
    if (Math.abs(term) < 1e-15) break;
  }
  return sum;
}
function close(actual, expected, tolerance = 2e-12) {
  assert.ok(Number.isFinite(actual), `nonfinite result ${actual}`);
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
}

test('emitted repeated-bounce formula agrees with an independent geometric series', () => {
  const {multiple} = scalarShader();
  for (const f0 of [0, .001, .04, .25, .8, 1]) for (const f90 of [0, .25, 1])
    for (const [a,b] of [[.3,.02],[.5,.25],[.001,.949],[1,0],[0,1],[.0001,.0009]])
      close(multiple(f0, f90, {x:a,y:b}), bounceSum(f0,f90,a,b), 5e-5);
});

test('a white metallic furnace retains unit energy at every DFG pair', () => {
  const {lighting} = scalarShader();
  for (let ai=0;ai<=32;ai++) for (let bi=0;bi<=32-ai;bi++) {
    const dfg={x:ai/32,y:bi/32};
    close(lighting(1,1,dfg,1,1,.04,1,1),1);
  }
});

test('colored dielectrics and metals stay finite, nonnegative and energy bounded', () => {
  for (const physical of [false,true]) {
    const {lighting}=scalarShader(physical);
    for (const base of [0,.02,.4,1]) for (const metallic of [0,.25,.5,.75,1])
      for (const f0 of [0,.04,.6,1]) for (const f90 of [0,.3,1])
        for (const dfg of [{x:.3,y:.01},{x:.1,y:.85},{x:1,y:0},{x:0,y:1},{x:0,y:0},{x:1.01,y:.02}]) {
          const result=lighting(base,metallic,dfg,1,1,f0,f90,1);
          assert.ok(Number.isFinite(result)&&result>=-1e-14&&result<=1+1e-12, `${physical}: ${result}`);
        }
  }
});

test('partial metalness mixes completed lobes, with only a dielectric diffuse substrate', () => {
  const {lighting}=scalarShader(true), dfg={x:.38,y:.07};
  for (const base of [.05,.3,.9]) for (const f0 of [0,.04,.8]) {
    const metal=lighting(base,1,dfg,3,.4,f0,.7,1), dielectric=lighting(base,0,dfg,3,.4,f0,.7,1);
    for (const m of [.1,.5,.8]) close(lighting(base,m,dfg,3,.4,f0,.7,1), metal*m+dielectric*(1-m));
  }
});

test('multiple bounces sample diffuse irradiance, not the rough specular radiance', () => {
  const {lighting}=scalarShader(), dfg={x:.4,y:.03}, base=.8;
  const multi=bounceSum(base,1,dfg.x,dfg.y), single=base*dfg.x+dfg.y;
  close(lighting(base,1,dfg,0,2,.04,1,1),2*multi);
  close(lighting(base,1,dfg,2,0,.04,1,1),2*single);
  close(lighting(base,1,dfg,4,2,.04,1,3),3*(4*single+2*multi));
});

test('zero physical specular remains disabled; HDR intensity is applied exactly once', () => {
  const {lighting}=scalarShader(true), dfg={x:.4,y:.1};
  close(lighting(.6,0,dfg,100,2,0,0,3),3*2*.6);
  close(lighting(0,1,dfg,100,2,0,0,3),0);
  close(lighting(.6,.5,dfg,100,2,.04,1,0),0);
});

test('unit-energy DFG needs no correction; excess and negative coefficients are bounded only in multiple mode', () => {
  const {multiple,lighting}=scalarShader();
  close(multiple(.3,1,{x:.6,y:.4}),0);
  close(lighting(.8,1,{x:2,y:1},3,1,.04,1,1),3*(.8*2/3+1/3));
  close(lighting(.8,1,{x:-.1,y:1},3,1,.04,1,1),3);
});

test('shader variants preserve bindings, explicit LOD, singularity guards and legacy branch', () => {
  for (const physical of [false,true]) for (const group of [1,2]) {
    const code=environmentLightingWgsl(group,physical);
    assert.deepEqual([...code.matchAll(/@binding\((\d+)\)/g)].map(m=>+m[1]),[4,5,6,7,8]);
    assert.equal((code.match(/textureSampleLevel\(/g)||[]).length,3);
    assert.doesNotMatch(code,/textureSample\(/);
    assert.ok(code.indexOf('if (!pbr)')<code.indexOf('if (environment_info.options.z'));
    assert.ok(code.indexOf('if (dot(view, view)')<code.indexOf('if (environment_info.options.z'));
    const legacy=code.slice(code.indexOf('  let f0 = mix('));
    assert.match(legacy,/return \(radiance \* specular \+ irradiance \* diffuse\) \* environment_info.options.x/);
    assert.doesNotMatch(legacy,/bounded_dfg|multiple_scattering/);
    assert.match(code,physical?/let f90 = specular_f90;/:/let f90 = 1.0;/);
  }
});
