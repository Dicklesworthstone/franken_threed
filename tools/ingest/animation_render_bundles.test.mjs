import assert from 'node:assert/strict';
import test from 'node:test';
import {createAnimationRenderBundleCache, encodeAnimationDraws} from './animation_render_bundles.mjs';

// Models bundle isolation and records actual host encoding calls. It is NOT a
// native GPU implementation or shader/pixel oracle.
export function bundleDevice() {
  const device = {encoders: [], calls: 0};
  function encoder(target) {
    let pipeline, index = null;
    const groups = new Map(), vertices = new Map();
    function draw(indexed, args) {
      device.calls++;
      assert.ok(pipeline && groups.has(0) && vertices.has(0), 'all required state was rebound');
      if (indexed) assert.ok(index, 'index rebound');
      target.draws.push({pipeline, groups: new Map(groups), vertices: new Map(vertices), index, indexed, args});
    }
    return {
      setPipeline(value) { pipeline = value; },
      setBindGroup(slot, group, offsets = []) { groups.set(slot, {group, offsets: [...offsets]}); },
      setVertexBuffer(slot, buffer) { vertices.set(slot, buffer); },
      setIndexBuffer(buffer, format) { index = {buffer, format}; },
      draw(...args) { draw(false, args); }, drawIndexed(...args) { draw(true, args); },
      executeBundles(bundles) {
        if (device.executeError) throw device.executeError;
        target.executions.push(bundles);
        for (const bundle of bundles) target.draws.push(...bundle.draws);
        pipeline = undefined; index = null; groups.clear(); vertices.clear();
      },
      finish() { if (device.finishError) throw device.finishError; return target; },
    };
  }
  device.createRenderBundleEncoder = descriptor => {
    if (device.createError) throw device.createError;
    const record = {descriptor, draws: []}; device.encoders.push(record); return encoder(record);
  };
  device.pass = () => {
    const record = {draws: [], executions: []}; return {...encoder(record), record};
  };
  return device;
}
const draw = (overrides = {}) => ({pipeline: {}, first: 0, count: 3,
  vertexBuffers: [{}], indexBuffer: null, indexFormat: null,
  record: {surfaceBuffer: null, textureGroup: null, lit: false, alphaMode: 'OPAQUE'}, ...overrides});
const parameters = () => ({format: 'rgba8unorm', depthFormat: 'depth32float', sampleCount: 1,
  bindGroup: {}, stride: 256, instancing: false});

test('reuses structural commands without capturing changing buffer contents or versions', () => {
  const device = bundleDevice(), cache = createAnimationRenderBundleCache(device, parameters()),
    commands = [draw(), draw()], pass = device.pass();
  assert.equal(cache.execute(pass, commands, 2), 2);
  for (let frame = 0; frame < 12; frame++) {
    commands[0].vertexBuffers[0].contents = frame;
    commands[0].record.version = frame;
    cache.execute(pass, commands, 2);
  }
  assert.equal(device.encoders.length, 1); assert.equal(device.calls, 2);
  assert.deepEqual(cache.diagnostics, {builds:1,reuses:12,executions:13,evictions:0,encodedDrawCalls:2,
    cachedBundles:1,cachedDraws:2,maxBundles:4,maxDraws:1024});
  assert.equal(pass.record.draws.length, 26);
});

for (const property of ['pipeline','first','count','vertex','extraVertex','indexBuffer','indexFormat','surfaceBuffer','textureGroup','lightGroup','lit','blend'])
test(`recorded dependency ${property} invalidates a bundle`, () => {
  const device = bundleDevice(), cache = createAnimationRenderBundleCache(device, parameters()), command = draw();
  command.record.lit = true;
  const pass = device.pass(), light = {};
  cache.execute(pass, [command], 1, light);
  const old = pass.record.executions[0][0];
  let nextLight = light;
  if (['pipeline','indexBuffer','indexFormat'].includes(property)) command[property] = property === 'indexFormat' ? 'uint32' : {};
  else if (['first','count'].includes(property)) command[property]++;
  else if (property === 'vertex') command.vertexBuffers[0] = {};
  else if (property === 'extraVertex') command.vertexBuffers.push({});
  else if (property === 'lightGroup') nextLight = {};
  else if (property === 'lit') command.record.lit = false;
  else if (property === 'blend') command.record.alphaMode = 'BLEND';
  else command.record[property] = {};
  cache.execute(pass, [command], 1, nextLight);
  assert.equal(cache.diagnostics.builds, 2);
  assert.notEqual(pass.record.executions[1][0], old);
});

test('ordering, list length and arena offsets are part of the recorded schedule', () => {
  const device = bundleDevice(), config = parameters(), cache = createAnimationRenderBundleCache(device, config),
    a = draw(), b = draw(), pass = device.pass();
  cache.execute(pass, [a,b], 2); cache.execute(pass, [b,a], 2); cache.execute(pass, [a], 1);
  assert.equal(cache.diagnostics.builds, 3);
  assert.deepEqual(pass.record.draws[0].groups.get(0).offsets, [0]);
  assert.deepEqual(pass.record.draws[1].groups.get(0).offsets, [256]);
  // Reused objects are copied: later mutations do not silently alter an old key.
  a.count=6;cache.execute(pass, [a], 1);a.count=3;cache.execute(pass, [a], 1);
  assert.equal(cache.diagnostics.reuses, 1);
});

test('instancing preserves native calls, firstInstance and BLEND order', () => {
  const device = bundleDevice(), cache = createAnimationRenderBundleCache(device, {...parameters(),instancing:true});
  const a=draw({indexBuffer:{},indexFormat:'uint16'}), b={...a,record:{...a.record,alphaMode:'BLEND'}};
  const pass=device.pass();
  assert.equal(cache.execute(pass,[a,a,b,b,a],5),4);
  assert.deepEqual(pass.record.draws.map(d=>d.args),[[3,2,0,0,0],[3,1,0,0,2],[3,1,0,0,3],[3,1,0,0,4]]);
  cache.execute(pass,[a,a,b,b,a],5);assert.equal(device.calls,4);
});

test('all state is rebound for direct draws after both a bundle and an empty executeBundles', () => {
  const device=bundleDevice(),config=parameters(),cache=createAnimationRenderBundleCache(device,config),
    a=draw({indexBuffer:{},indexFormat:'uint16'}),b=draw({indexBuffer:{},indexFormat:'uint32'}),pass=device.pass();
  a.record={...a.record,lit:true,textureGroup:{},surfaceBuffer:{}};
  b.record={...b.record,lit:true,textureGroup:{},surfaceBuffer:{}};
  const lightA={},lightB={};cache.execute(pass,[a],1,lightA);
  encodeAnimationDraws(pass,[b],1,{...config,lightGroup:lightB});
  pass.executeBundles([]);encodeAnimationDraws(pass,[a],1,{...config,lightGroup:lightA});
  assert.deepEqual(pass.record.draws.map(d=>d.pipeline),[a.pipeline,b.pipeline,a.pipeline]);
  assert.equal(pass.record.draws[1].groups.get(2).group,lightB);
  assert.equal(pass.record.draws[1].vertices.get(1),b.record.surfaceBuffer);
  assert.equal(pass.record.draws[1].index.buffer,b.indexBuffer);
});

test('cache is bounded, least-recently-used entries evict, and retirement drops borrowed resources', () => {
  const device=bundleDevice(),cache=createAnimationRenderBundleCache(device,{...parameters(),maxBundles:2,maxDraws:2}),
    pass=device.pass(),a=draw(),b=draw(),c=draw();
  for(const d of [a,b,a,c,a])cache.execute(pass,[d],1);
  assert.equal(cache.diagnostics.evictions,1);assert.equal(cache.diagnostics.reuses,2);
  cache.execute(pass,[b],1);assert.equal(cache.diagnostics.builds,4);
  assert.ok(cache.diagnostics.cachedDraws<=4);
  assert.throws(()=>cache.execute(pass,[a,b,c],3),RangeError);
  cache.clear();assert.equal(cache.diagnostics.cachedBundles,0);
  cache.execute(pass,[a],1);cache.dispose();assert.equal(cache.diagnostics.cachedDraws,0);
  assert.throws(()=>cache.execute(pass,[a],1),/disposed/);
});

test('empty frames allocate nothing and attachment layouts are fixed per device cache', () => {
  for(const [format,depthFormat,sampleCount] of [[null,'depth32float',1],['rgba8unorm',null,4]]){
    const device=bundleDevice(),cache=createAnimationRenderBundleCache(device,{...parameters(),format,depthFormat,sampleCount}),pass=device.pass();
    assert.equal(cache.execute(pass,[],0),0);assert.equal(device.encoders.length,0);assert.equal(pass.record.executions.length,0);
    cache.execute(pass,[draw()],1);
    assert.deepEqual(device.encoders[0].descriptor.colorFormats,format===null?[]:[format]);
    assert.equal(device.encoders[0].descriptor.depthStencilFormat,depthFormat??undefined);
    assert.equal(device.encoders[0].descriptor.sampleCount,sampleCount);
  }
});

test('bundle creation, finish and execution failures never publish a reusable entry or replay directly', () => {
  for(const property of ['createError','finishError','executeError']){
    const device=bundleDevice(),cache=createAnimationRenderBundleCache(device,parameters()),pass=device.pass(),error=new Error(property);
    device[property]=error;
    assert.throws(()=>cache.execute(pass,[draw()],1),e=>e===error);
    assert.equal(cache.diagnostics.cachedBundles,0);assert.equal(pass.record.draws.length,0);
  }
  assert.throws(()=>createAnimationRenderBundleCache({},parameters()),TypeError);
  for(const options of [{maxBundles:0},{maxBundles:65},{maxDraws:0}])
    assert.throws(()=>createAnimationRenderBundleCache(bundleDevice(),{...parameters(),...options}),RangeError);
});
