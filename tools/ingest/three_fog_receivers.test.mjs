/** Host contract tests. Fog/camera/packing and receiver implementation are real;
 * the small source classes and recording renderer are explicit fixtures. These
 * tests do not construct Three.js, compile WGSL, submit a GPU or assert pixels.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {inspectThreeFog, threeFogDescriptor, withThreeFogReceivers} from './three_fog.mjs';
import {packAnimationFog} from './animation_fog.mjs';

class Color { constructor(r = .1, g = .2, b = .3) { Object.assign(this, {r, g, b, isColor: true}); } }
class Fog { constructor() { this.isFog = true; this.color = new Color(); this.near = 2; this.far = 20; } }
class FogExp2 { constructor() { this.isFogExp2 = true; this.color = new Color(); this.density = .125; } }
class Camera { constructor() {
  this.isPerspectiveCamera = true; this.coordinateSystem = 2000;
  this.projectionMatrix = {elements: [2,0,0,0, 0,3,0,0, .2,.1,-1.02,-1, 0,0,-.202,0]};
} }
const three = {REVISION: '186', Color, Fog, FogExp2, Camera, WebGLCoordinateSystem: 2000, WebGPUCoordinateSystem: 2001};
const nativeFog = () => ({type:'linear', color:[.1,.2,.3], near:2, far:20, depthFromClip:[0,0,0,1]});
function recording() {
  const calls = [], registrations = [], fault = {index: -1, terminal: false, hook: null};
  let disposed = false, failed = false, drawCallCount = 0, waits = 0;
  const renderer = {
    calls, registrations, fault,
    get disposed() { return disposed; }, get failed() { return failed; },
    get drawCallCount() { return drawCallCount; }, get waits() { return waits; },
    allocatedBytes: 304, bundleDiagnostics: Object.freeze({builds: 1}),
    addMesh(gpu, options) { registrations.push({gpu, options}); return Promise.resolve({gpu, options}); },
    render(frame) {
      if (calls.length === fault.index) { failed = fault.terminal; throw new Error('injected submit failure'); }
      // Snapshot the production packer's actual f32 bytes as a native renderer
      // would. Retain receiver flags to test composition with nested owners.
      calls.push({frame, packed: packAnimationFog(frame.fog)});
      drawCallCount = frame.draws.length;
      fault.hook?.(frame);
    },
    async whenIdle() { waits++; }, dispose() { disposed = true; },
  };
  return renderer;
}
const draw = (id, receiveFog) => ({mesh: {id}, receiveFog});
const ids = calls => calls.flatMap(c => c.frame.draws.map(d => d.mesh.id));
const near = (a,b) => assert.ok(Math.abs(a-b) < 1e-9 * Math.max(1,Math.abs(a),Math.abs(b)), `${a} != ${b}`);

for (const Kind of [Fog, FogExp2]) test(`${Kind.name} is a live, independent linear-space snapshot`, () => {
  const source = new Kind(), before = structuredClone(source), a = inspectThreeFog(source, three);
  assert.ok(Object.isFrozen(a)); assert.ok(Object.isFrozen(a.color));
  source.color.r = 2;
  if (Kind === Fog) source.near = 3; else source.density = .5;
  const b = inspectThreeFog(source, three);
  assert.equal(a.color[0], before.color.r); assert.equal(b.color[0], 2);
  assert.equal(a.type, Kind === Fog ? 'linear' : 'exp2');
  assert.equal(Kind === Fog ? b.near : b.density, Kind === Fog ? 3 : .5);
});

for (const [name, matrix, coordinateSystem, orthographic] of [
  ['GL perspective', [2,0,0,0, 0,3,0,0, .2,.1,-1.02,-1, 0,0,-.202,0], 2000, false],
  ['GPU perspective', [2,0,0,0, 0,3,0,0, .2,.1,-1.01,-1, 0,0,-.101,0], 2001, false],
  ['GL orthographic', [.2,0,0,0, 0,.3,0,0, 0,0,-.02,0, .1,.2,-1.002,1], 2000, true],
  ['GPU orthographic', [.2,0,0,0, 0,.3,0,0, 0,0,-.01,0, .1,.2,-.001,1], 2001, true],
]) test(`${name} agrees with independent native clip multiplication`, () => {
  const camera = new Camera();
  Object.assign(camera, {coordinateSystem, isPerspectiveCamera: !orthographic, isOrthographicCamera: orthographic});
  camera.projectionMatrix.elements = matrix;
  const before = matrix.slice(), descriptor = threeFogDescriptor(new Fog(), camera, three);
  for (const z of [-.2, -2, -20, -200]) {
    const v = [3,-4,z,1];
    const clip = Array.from({length:4}, (_, r) => v.reduce((n,x,c) => n+x*matrix[c*4+r], 0));
    if (coordinateSystem === 2000) clip[2] = .5*(clip[2]+clip[3]);
    near(clip.reduce((n,x,i) => n+x*descriptor.depthFromClip[i],0), -z);
  }
  assert.deepEqual(matrix, before);
});

test('null reset never consults camera but retains pinned-module admission', () => {
  const trap = new Proxy({}, {get() { throw new Error('unexpected read'); }});
  assert.equal(inspectThreeFog(null, three), null);
  assert.equal(threeFogDescriptor(null, trap, three), null);
});

for (const [name, change] of [
  ['plain spoof', () => ({color: new Color(), near:2, far:20, isFog:true})],
  ['subclass', () => new (class CustomFog extends Fog {})()],
  ['foreign Color', f => { f.color = {r:0,g:0,b:0}; return f; }],
  ['negative color', f => { f.color.r = -1; return f; }],
  ['NaN color', f => { f.color.b = NaN; return f; }],
  ['collapsed f32 edges', f => { f.near = 1; f.far = 1+Number.EPSILON; return f; }],
  ['infinite near', f => { f.near = Infinity; return f; }],
  ['negative density', () => Object.assign(new FogExp2(), {density:-1})],
  ['fog accessor', f => { Object.defineProperty(f, 'near', {get() { throw new Error('getter ran'); }}); return f; }],
  ['color accessor', f => { Object.defineProperty(f.color, 'r', {get() { throw new Error('getter ran'); }}); return f; }],
]) test(`source rejects ${name} before rendering`, () => {
  assert.throws(() => inspectThreeFog(change(new Fog()), three), e => /^(THREE_FOG|ANIMATION_FOG)_/.test(e.code));
});

test('source module revision and camera family/convention fail closed', () => {
  assert.throws(() => inspectThreeFog(new Fog(), {...three, REVISION:'185'}), {code:'THREE_FOG_SOURCE'});
  for (const changes of [{coordinateSystem: 999}, {isArrayCamera:true}, {reversedDepth:true}, {isPerspectiveCamera:false}])
    assert.throws(() => threeFogDescriptor(new Fog(), Object.assign(new Camera(), changes), three), {code:'THREE_FOG_CAMERA'});
  const camera = new Camera(); camera.projectionMatrix.elements.fill(0);
  assert.throws(() => threeFogDescriptor(new Fog(), camera, three), {code:'THREE_FOG_CAMERA'});
});

test('mixed transparent receivers retain order, original clears and later attachment loads', () => {
  const core = recording(), owner = withThreeFogReceivers(core), fog = nativeFog();
  const input = [true,true,false,true,false,false].map((v,i) => ({...draw(i,v), receiveShadow:i%2===0, receiveEnvironment:i%3===0}));
  const frame = {draws:input, fog, colorView:{}, depthView:{}, resolveView:{}, loadOp:'clear', depthLoadOp:'clear', renderBundles:false};
  assert.equal(owner.render(frame), owner);
  assert.deepEqual(ids(core.calls), [0,1,2,3,4,5]);
  assert.deepEqual(core.calls.map(c=>c.frame.draws.length), [2,1,1,2]);
  assert.deepEqual(core.calls.map(c=>c.packed[11]), [1,0,1,0]);
  for (const [i,{frame:part}] of core.calls.entries()) {
    assert.equal(part.loadOp, i?'load':'clear'); assert.equal(part.depthLoadOp, i?'load':'clear');
    for (const key of ['colorView','depthView','resolveView','renderBundles']) assert.equal(part[key], frame[key]);
    for (const d of part.draws) {
      assert.ok(!Object.hasOwn(d,'receiveFog'));
      assert.equal(d.receiveShadow, d.mesh.id%2===0); assert.equal(d.receiveEnvironment, d.mesh.id%3===0);
    }
  }
  assert.ok(input.every(d=>Object.hasOwn(d,'receiveFog')));
  assert.equal(owner.drawCount, 6); assert.equal(owner.drawCallCount,6); assert.equal(owner.colorPassCount,4);
});

test('all 511 bounded receiver patterns partition only adjacent state changes', () => {
  for (let n=0; n<=8; n++) for (let bits=0; bits<2**n; bits++) {
    const flags = Array.from({length:n},(_,i)=>!!(bits&(1<<i)));
    const core = recording(), owner = withThreeFogReceivers(core,8);
    owner.render({fog:nativeFog(),draws:flags.map((flag,i)=>draw(i,flag))});
    const transitions = flags.reduce((s,x,i)=>s+Number(i===0||x!==flags[i-1]),0);
    assert.equal(core.calls.length, Math.max(1,transitions));
    assert.deepEqual(ids(core.calls), Array.from({length:n},(_,i)=>i));
    for (const c of core.calls) for (const d of c.frame.draws) assert.equal(c.packed[11]!==0, flags[d.mesh.id]);
  }
});

test('null/omitted fog coalesces all flags; empty frame still clears exactly once', () => {
  for (const fog of [null,undefined]) for (const draws of [[],[draw(0,true),draw(1,false),draw(2,true)]]) {
    const core=recording(), owner=withThreeFogReceivers(core);
    owner.render({fog,draws,loadOp:'clear',depthLoadOp:'clear'});
    assert.equal(core.calls.length,1); assert.equal(core.calls[0].frame.loadOp,'clear');
    assert.deepEqual([...core.calls[0].packed],Array(12).fill(0));
  }
});

test('frame fog snapshots survive mutations between spans and later frames reset state', () => {
  const core=recording(), owner=withThreeFogReceivers(core), fog=nativeFog();
  const draws=[draw(0,true),draw(1,false),draw(2,true)];
  core.fault.hook=()=>{fog.color[0]=9;fog.depthFromClip[0]=7;fog.near=3;draws.length=0;};
  owner.render({fog,draws});
  assert.deepEqual(core.calls[0].packed,core.calls[2].packed);
  assert.equal(owner.drawCount,3);
  core.fault.hook=null;
  owner.render({fog:{type:'exp2',color:[1,2,3],density:.25,depthFromClip:[0,0,0,1]},draws:[draw(3,true)]});
  assert.equal(core.calls.at(-1).packed[11],2);
  owner.render({fog:null,draws:[draw(4,true)]});
  assert.equal(core.calls.at(-1).packed[11],0);
});

test('flag and fog preflight covers even disabled/late spans before any renderer call', () => {
  const core=recording(), owner=withThreeFogReceivers(core,3);
  for (const bad of [undefined,null,1,'false']) {
    assert.throws(()=>owner.render({fog:nativeFog(),draws:[draw(0,true),draw(1,bad)]}),{code:'THREE_FOG_FRAME'});
    assert.equal(core.calls.length,0); assert.equal(owner.failed,false);
  }
  assert.throws(()=>owner.render({fog:{...nativeFog(),far:2},draws:[draw(0,false)]}),{code:'ANIMATION_FOG_VALUE'});
  assert.throws(()=>owner.render({fog:nativeFog(),draws:Array.from({length:4},(_,i)=>draw(i,true))}),{code:'THREE_FOG_LIMIT'});
  assert.equal(core.calls.length,0);
  owner.render({fog:null,draws:[draw(0,false)]}); assert.equal(core.calls.length,1);
});

test('draw iterators cannot extend the declared capacity', () => {
  const core=recording(), owner=withThreeFogReceivers(core,2), draws=[draw(0,true),draw(1,false)];
  draws[Symbol.iterator]=()=>{throw new Error('unexpected iterator');};
  owner.render({fog:nativeFog(),draws}); assert.deepEqual(ids(core.calls),[0,1]);
});

test('post-prefix failure is terminal and retains previous complete-frame counts', async () => {
  const core=recording(), owner=withThreeFogReceivers(core);
  owner.render({fog:nativeFog(),draws:[draw(0,true)]});
  core.fault.index=2;
  assert.throws(()=>owner.render({fog:nativeFog(),draws:[draw(1,true),draw(2,false)]}),/injected submit failure/);
  assert.equal(owner.failed,true); assert.equal(owner.disposed,true);
  assert.equal(owner.drawCount,1); assert.equal(owner.colorPassCount,1);
  assert.throws(()=>owner.render({draws:[]}),/injected submit failure/);
  await assert.rejects(owner.whenIdle(),/injected submit failure/);
});

test('pre-submit failures remain retryable unless the underlying renderer failed', () => {
  for (const terminal of [false,true]) {
    const core=recording(), owner=withThreeFogReceivers(core);
    core.fault.index=0;core.fault.terminal=terminal;
    assert.throws(()=>owner.render({fog:nativeFog(),draws:[draw(0,true)]}),/injected submit failure/);
    assert.equal(owner.failed,terminal); assert.equal(owner.disposed,terminal);
    if (!terminal) {core.fault.index=-1;owner.render({draws:[]});assert.equal(owner.colorPassCount,1);}
  }
});

test('nested receiver diagnostics, resource forwarding, reentrancy and disposal', async () => {
  const core=recording(); core.colorPassCount=3;
  const owner=withThreeFogReceivers(core), gpu={}, options={};
  assert.deepEqual(await owner.addMesh(gpu,options),{gpu,options});
  assert.equal(owner.allocatedBytes,304); assert.equal(owner.bundleDiagnostics,core.bundleDiagnostics);
  core.fault.hook=()=>{
    assert.throws(()=>owner.render({draws:[]}),{code:'THREE_FOG_REENTRANT'});
    assert.throws(()=>owner.dispose(),{code:'THREE_FOG_REENTRANT'});
  };
  owner.render({fog:nativeFog(),draws:[draw(0,true),draw(1,false)]});
  assert.equal(owner.colorPassCount,6);
  assert.equal(await owner.whenIdle(),owner);assert.equal(core.waits,1);
  owner.dispose();assert.throws(()=>owner.render({draws:[]}),{code:'THREE_FOG_DISPOSED'});
});

for (const capacity of [0,-1,1.5,Infinity,Number.MAX_SAFE_INTEGER+1])
  test(`invalid receiver capacity ${capacity}`,()=>assert.throws(()=>withThreeFogReceivers(recording(),capacity),{code:'THREE_FOG_LIMIT'}));


test('legacy scene and direct-fog calls agree without interpreting fog metadata as a scene', () => {
  const fog = new Fog(), camera = new Camera(), scene = {fog};
  fog.fog = 'unrelated metadata';
  const direct = threeFogDescriptor(fog, camera, three);
  assert.deepEqual(threeFogDescriptor(scene, camera, three), direct);
  scene.fog = new FogExp2();
  assert.equal(threeFogDescriptor(scene, camera, three).type, 'exp2');
  scene.fog = null;
  const trap = new Proxy({}, {get() { throw new Error('unexpected camera access'); }});
  assert.equal(threeFogDescriptor(scene, trap, three), null);
  assert.throws(() => threeFogDescriptor(scene, trap, {...three, REVISION:'185'}), {code:'THREE_FOG_SOURCE'});
});

test('source projection admission preserves f32/error codes and rejects conflicting fog tags', () => {
  const fog = new Fog(), camera = new Camera();
  for (const source of [fog, {fog}]) {
    camera.projectionMatrix.elements[0] = 1e100;
    assert.throws(() => threeFogDescriptor(source, camera, three), {code:'THREE_FOG_CAMERA'});
    camera.projectionMatrix.elements[0] = 2;
    fog.isFogExp2 = true;
    assert.throws(() => threeFogDescriptor(source, camera, three), {code:'THREE_FOG_SOURCE'});
    delete fog.isFogExp2;
  }
});
