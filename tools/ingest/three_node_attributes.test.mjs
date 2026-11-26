import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodeAttributes, inspectThreeNodeSharedAttribute} from './three_node_attributes.mjs';
import {createThreeNodeBundleGeometry} from './three_node_bundle_geometry.mjs';
import SourceWebGPUAttributeUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUAttributeUtils.js';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function sourceUploads(attribute) {
  const entries = new Map(), device = geometryDevice();
  const backend = {device, get(owner) {
    if (!entries.has(owner)) entries.set(owner, {});
    return entries.get(owner);
  }, delete(owner) { entries.delete(owner); }};
  const utils = new SourceWebGPUAttributeUtils(backend), owner = attribute.isInterleavedBufferAttribute ? attribute.data : attribute;
  return {create() {
    const usage = globalThis.GPUBufferUsage;
    globalThis.GPUBufferUsage ??= {INDEX: 16};
    try { return utils.createAttribute(attribute, 32); }
    finally { if (usage === undefined) delete globalThis.GPUBufferUsage; else globalThis.GPUBufferUsage = usage; }
  }, update: () => utils.updateAttribute(attribute),
    dispose: () => utils.destroyAttribute(attribute), bytes: () => new Uint8Array(entries.get(owner).buffer.data)};
}

for (const Type of [Uint8Array, Uint16Array, Int8Array, Int16Array]) for (const dynamic of [false, true])
test(`${Type.name} private node inputs enforce source upload alignment before consuming any stream (dynamic=${dynamic})`, () => {
  const values = Array.from({length: 16}, (_, index) => index + 1), alignment = 4 / Type.BYTES_PER_ELEMENT;
  const owner = new T.InterleavedBuffer(new Type(values), 4), attribute = new T.InterleavedBufferAttribute(owner, 4, 0);
  const referenceOwner = new T.InterleavedBuffer(new Type(values), 4), reference = new T.InterleavedBufferAttribute(referenceOwner, 4, 0);
  const earlier = new T.BufferAttribute(new Float32Array([1, 2, 3, 4]), 1);
  if (dynamic) owner.setUsage(T.DynamicDrawUsage);
  const oracle = sourceUploads(reference), histories = createThreeNodeAttributes(T), descriptor = {};
  const view = histories.register(descriptor, new T.BufferGeometry(), [
    {name: 'earlier', attribute: earlier}, {name: 'generated', attribute}]);
  const capture = renderId => { histories.capture(descriptor, renderId).install(); return view.attributes.generated.data.array; };
  try {
    owner.addUpdateRange(1, 1); referenceOwner.addUpdateRange(1, 1);
    const first = capture(1); oracle.create();
    assert.deepEqual(new Uint8Array(first.buffer), oracle.bytes(), 'first uploads ignore unaligned ranges');
    assert.deepEqual(owner.updateRanges, referenceOwner.updateRanges);
    for (const [index, range] of [{start: 1, count: alignment}, {start: 0, count: 1}].entries()) {
      earlier.array[0] = 31 + index; earlier.addUpdateRange(0, 1); earlier.needsUpdate = true;
      for (const buffer of [owner, referenceOwner]) {
        buffer.clearUpdateRanges(); buffer.array.fill(21 + index, 0, alignment);
        buffer.addUpdateRange(range.start, range.count);
        if (!dynamic) buffer.needsUpdate = true;
      }
      const prior = capture(index + 1), earlierView = view.attributes.earlier.array;
      assert.throws(() => oracle.update(), {code: 'ERR_ASSERTION'});
      assert.throws(() => capture(index + 2), /four-byte offsets and sizes/);
      assert.deepEqual(owner.updateRanges, [range]); assert.deepEqual(referenceOwner.updateRanges, [range]);
      assert.deepEqual(earlier.updateRanges, [{start: 0, count: 1}], 'a later invalid stream cannot consume an earlier upload');
      assert.equal(view.attributes.earlier.array, earlierView); assert.equal(view.attributes.generated.data.array, prior);
      for (const buffer of [owner, referenceOwner]) { buffer.clearUpdateRanges(); buffer.addUpdateRange(0, alignment); }
      const accepted = capture(index + 2); oracle.update();
      assert.deepEqual(new Uint8Array(accepted.buffer), oracle.bytes());
      assert.equal(view.attributes.earlier.array[0], 31 + index); assert.deepEqual(earlier.updateRanges, []);
    }
    assert.deepEqual([...first], values, 'rejected and repaired uploads preserve earlier captures');
    assert.equal(owner.array.constructor, Type);
  } finally { histories.dispose(); oracle.dispose(); }
});

for (const [Type, Promoted, word] of [[Uint8Array, Uint32Array, 233], [Uint16Array, Uint32Array, 65003],
  [Int8Array, Int32Array, -119], [Int16Array, Int32Array, -32003]])
test(`${Type.name} public node inputs match source first-upload promotion and queued partial histories`, () => {
  const values = Array.from({length: 8}, (_, index) => word + index), original = new Type(values);
  const source = new T.BufferAttribute(original, 2), reference = new T.BufferAttribute(new Type(values), 2);
  const oracle = sourceUploads(reference), histories = createThreeNodeAttributes(T), captures = createThreeNodeBundleGeometry(T);
  const geometry = new T.BufferGeometry().setAttribute('ordinary', source), descriptor = {};
  const view = histories.register(descriptor, geometry, [{name: 'generated', attribute: source, shared: true}]);
  const leases = [], capture = renderId => {
    histories.capture(descriptor, renderId).install();
    const lease = captures.capture(view, renderId, ['ordinary', 'generated']); leases.push(lease);
    histories.validate(descriptor);
    return lease.geometry.attributes.generated;
  };
  const equal = actual => {
    assert.equal(actual.array.constructor, Promoted);
    assert.deepEqual(new Uint8Array(actual.array.buffer), oracle.bytes());
    assert.equal(source.array.constructor, reference.array.constructor);
    assert.deepEqual(source.array, reference.array);
    assert.deepEqual(source.updateRanges, reference.updateRanges);
  };
  try {
    source.addUpdateRange(1, 1); reference.addUpdateRange(1, 1);
    assert.equal(source.array, original, 'registering a graph does not upload');
    const first = capture(1); oracle.create(); equal(first);
    assert.notEqual(source.array, original); assert.equal(source.version, 0);
    assert.equal(leases[0].geometry.attributes.ordinary, first, 'aliases share one upload owner');
    for (const attribute of [source, reference]) {
      attribute.array[0] = 7; attribute.array[1] = 11; attribute.array[4] = 17; attribute.needsUpdate = true;
    }
    const sameRender = capture(1);
    assert.deepEqual(sameRender.array, first.array, 'one source render has one upload boundary');
    assert.deepEqual(source.updateRanges, [{start: 1, count: 1}]);
    const second = capture(2); oracle.update(); equal(second);
    assert.equal(second.array[0], word); assert.equal(second.array[1], 11); assert.equal(second.array[4], word + 4);
    source.array.fill(19); reference.array.fill(19);
    const stale = capture(3); equal(stale);
    assert.deepEqual(stale.array, second.array, 'unmarked CPU edits do not enter GPU history');
    assert.deepEqual([...first.array], values, 'later uploads do not rewrite accepted generations');
    geometry.dispose(); oracle.dispose();
    const reused = capture(4); oracle.create(); equal(reused);
    assert.deepEqual([...reused.array], Array(8).fill(19));
    assert.equal(histories.diagnostics.allocatedBytes, 0, 'the common capture owner charges the snapshots');
  } finally {
    for (const lease of leases) lease.release(); histories.dispose(); captures.dispose();
  }
  assert.equal(captures.diagnostics.allocatedBytes, 0);
});

for (const Type of [Uint8Array, Uint16Array, Int8Array, Int16Array])
test(`${Type.name} public interleaved node inputs preserve source storage and aligned update history`, () => {
  const values = Array.from({length: 16}, (_, index) => index + 1), original = new Type(values);
  const owner = new T.InterleavedBuffer(original, 4), source = new T.InterleavedBufferAttribute(owner, 4, 0);
  const referenceOwner = new T.InterleavedBuffer(new Type(values), 4), reference = new T.InterleavedBufferAttribute(referenceOwner, 4, 0);
  const oracle = sourceUploads(reference), histories = createThreeNodeAttributes(T), captures = createThreeNodeBundleGeometry(T);
  const geometry = new T.BufferGeometry(), descriptor = {}, leases = [];
  const view = histories.register(descriptor, geometry, [{name: 'generated', attribute: source, shared: true}]);
  const capture = renderId => {
    histories.capture(descriptor, renderId).install();
    const lease = captures.capture(view, renderId, ['generated']); leases.push(lease);
    return lease.geometry.attributes.generated.data.array;
  };
  try {
    const first = capture(1); oracle.create();
    assert.equal(owner.array, original); assert.equal(referenceOwner.array.constructor, Type);
    assert.deepEqual(new Uint8Array(first.buffer), oracle.bytes());
    for (const buffer of [owner, referenceOwner]) {
      buffer.array.fill(7, 0, 4); buffer.array.fill(19, 8, 12); buffer.addUpdateRange(8, 4); buffer.needsUpdate = true;
    }
    const changed = capture(2); oracle.update();
    assert.deepEqual(new Uint8Array(changed.buffer), oracle.bytes());
    assert.deepEqual([...changed.subarray(0, 4)], values.slice(0, 4));
    assert.deepEqual([...changed.subarray(8, 12)], Array(4).fill(19));
    assert.deepEqual([...first], values); assert.equal(owner.array, original);
    assert.deepEqual(owner.updateRanges, []); histories.validate(descriptor);
  } finally {
    for (const lease of leases) lease.release(); histories.dispose(); captures.dispose();
  }
});

test('native integer promotion charges its 32-bit snapshot before changing public arrays', () => {
  const original = new Uint8Array(8), attribute = new T.BufferAttribute(original, 2);
  const captures = createThreeNodeBundleGeometry(T, {maxBytes: 8});
  try {
    const geometry = new T.BufferGeometry().setAttribute('value', attribute);
    assert.throws(() => captures.capture(geometry, 1), /byte budget/);
    assert.equal(attribute.array, original); assert.equal(attribute.version, 0);
    assert.equal(captures.diagnostics.allocatedBytes, 0);
  } finally { captures.dispose(); }
});

test('public node attributes share budgets, first-upload gates and disposal with ordinary geometry',()=>{
  const histories=createThreeNodeAttributes(T),captures=createThreeNodeBundleGeometry(T,{maxBytes:192});
  const source=new T.BufferAttribute(new Float32Array(12).fill(.25),3);
  const geometry=new T.BufferGeometry().setAttribute('color',source),other=new T.BufferGeometry().setAttribute('color',source);
  const descriptor={},secondDescriptor={},selection=[{name:'nodeColor',attribute:source,shared:true}];
  const view=histories.register(descriptor,geometry,selection),secondView=histories.register(secondDescriptor,other,selection);
  const leases=[];
  const capture=(selected,renderId)=>{const lease=captures.capture(selected,renderId,['color','nodeColor']);leases.push(lease);return lease.geometry;};
  try{
    assert.equal(view.attributes.nodeColor,source);assert.equal(secondView.attributes.nodeColor,source);
    const packet=histories.capture(descriptor,1);packet.install();
    assert.equal(packet.bytes,0);assert.equal(histories.diagnostics.allocatedBytes,0);
    const first=capture(view,1);assert.equal(first.attributes.color,first.attributes.nodeColor);
    assert.equal(captures.diagnostics.allocatedBytes,48,'two shader names retain one source allocation');
    source.array.fill(.875);source.array.fill(.5,0,3);source.addUpdateRange(0,3);source.needsUpdate=true;
    const sameRender=capture(secondView,1);
    assert.deepEqual([...sameRender.attributes.nodeColor.array],Array(12).fill(.25));
    assert.deepEqual(source.updateRanges,[{start:0,count:3}],'a second render object cannot consume ranges after the source upload gate');
    const next=capture(secondView,2);
    assert.deepEqual([...next.attributes.nodeColor.array],[...Array(3).fill(.5),...Array(9).fill(.25)]);
    assert.deepEqual(source.updateRanges,[]);
    source.array.fill(.75);geometry.dispose();
    const reused=capture(secondView,3);
    assert.deepEqual([...reused.attributes.nodeColor.array],Array(12).fill(.75),'source geometry disposal invalidates shared residency');
    assert.deepEqual([...first.attributes.nodeColor.array],Array(12).fill(.25),'accepted generations survive disposal');
    assert.equal(histories.diagnostics.allocatedBytes,0);
  }finally{for(const lease of leases)lease.release();histories.dispose();captures.dispose();}
  assert.equal(captures.diagnostics.allocatedBytes,0);
});

test('generated instance colors retain queued versions, partial uploads and unmarked stale bytes',()=>{
  const histories=createThreeNodeAttributes(T),descriptor={},geometry=new T.PlaneGeometry();
  const colors=new T.InstancedBufferAttribute(new Float32Array([.125,.25,.5,.75,.875,1]),3);
  const view=histories.register(descriptor,geometry,[{name:'nodeColor',attribute:colors}]);
  const originalArray=colors.array,first=histories.capture(descriptor,1);
  colors.array.set([.5,.625,.75,.25,.375,.5]);
  const stale=histories.capture(descriptor,2);
  colors.addUpdateRange(3,3);colors.needsUpdate=true;
  const partial=histories.capture(descriptor,3);
  assert.equal(colors.updateRanges.length,0);
  colors.array.fill(0);
  first.install();assert.deepEqual([...view.attributes.nodeColor.array],[.125,.25,.5,.75,.875,1]);
  stale.install();assert.deepEqual([...view.attributes.nodeColor.array],[.125,.25,.5,.75,.875,1]);
  partial.install();assert.deepEqual([...view.attributes.nodeColor.array],[.125,.25,.5,.25,.375,.5]);
  assert.equal(colors.array,originalArray);assert.deepEqual([...colors.array],[0,0,0,0,0,0]);
  assert.equal(colors.version,1);assert.notEqual(view.attributes.nodeColor,colors);
  assert.equal(view.attributes.position,geometry.attributes.position);
  const generatedColor=view.attributes.nodeColor,position=geometry.attributes.position.clone();
  geometry.setAttribute('position',position);
  assert.equal(view.attributes.position,position,'ordinary attribute replacement remains live');
  assert.equal(view.attributes.nodeColor,generatedColor,'generated streams keep their private owners');
  assert.equal(first.bytes,24);assert.equal(histories.byteLength(descriptor),24);
  histories.release(descriptor);assert.equal(histories.diagnostics.allocatedBytes,24);
  const replacement={};const next=histories.register(replacement,geometry,[{name:'nodeColor',attribute:colors}]);
  histories.capture(replacement,4).install();assert.equal(next.attributes.nodeColor.array[0],.125,'program replacement preserves unmarked stale bytes');
  geometry.dispose();assert.equal(histories.diagnostics.allocatedBytes,0);
  first.install();assert.equal(view.attributes.nodeColor.array[0],.125,'queued snapshots survive compiler retirement');
  histories.dispose();histories.release(replacement);assert.equal(histories.diagnostics.allocatedBytes,0);
});

test('four generated matrix columns share one captured owner and its source upload history',()=>{
  const histories=createThreeNodeAttributes(T),descriptor={},other={},geometry=new T.PlaneGeometry();
  const matrix=new T.InstancedInterleavedBuffer(new Float32Array(32),16,1);
  new T.Matrix4().makeTranslation(1,2,3).toArray(matrix.array,0);
  new T.Matrix4().makeTranslation(4,5,6).toArray(matrix.array,16);
  matrix.addUpdateRange(12,1);
  const attributes=[0,4,8,12].map((offset,index)=>({name:`column${index}`,attribute:new T.InterleavedBufferAttribute(matrix,4,offset)}));
  const view=histories.register(descriptor,geometry,attributes);
  const shared=histories.register(other,geometry,attributes);
  const first=histories.capture(descriptor,1);
  assert.equal(matrix.updateRanges.length,1,'first source creation leaves ranges intact');
  assert.equal(first.bytes,128);assert.equal(histories.diagnostics.allocatedBytes,128);
  matrix.array[12]=7;matrix.array[28]=8;matrix.needsUpdate=true;
  const changed=histories.capture(descriptor,2),sameCall=histories.capture(other,2);
  assert.equal(matrix.updateRanges.length,0);
  first.install();assert.equal(view.attributes.column3.data.array[12],1);
  changed.install();sameCall.install();
  assert.equal(view.attributes.column3.data.array[12],7);assert.equal(view.attributes.column3.data.array[28],4);
  assert.equal(shared.attributes.column3.data.array[12],7);assert.equal(shared.attributes.column3.data.array[28],4);
  assert.equal(new Set(Object.values(view.attributes).filter(a=>a.isInterleavedBufferAttribute).map(a=>a.data)).size,1);
  histories.release(descriptor);assert.equal(histories.diagnostics.allocatedBytes,128);
  histories.release(other);assert.equal(histories.diagnostics.allocatedBytes,128);
  geometry.dispose();assert.equal(histories.diagnostics.allocatedBytes,0);histories.dispose();
});

test('dynamic generated attributes update without flags once per logical render',()=>{
  const histories=createThreeNodeAttributes(T),descriptor={};
  const color=new T.InstancedBufferAttribute(new Float32Array([.25,.5,.75]),3).setUsage(T.DynamicDrawUsage);
  const view=histories.register(descriptor,new T.PlaneGeometry(),[{name:'nodeColor',attribute:color}]);
  const first=histories.capture(descriptor,1);color.array[0]=.5;
  const same=histories.capture(descriptor,1),next=histories.capture(descriptor,2);
  first.install();assert.equal(view.attributes.nodeColor.array[0],.25);
  same.install();assert.equal(view.attributes.nodeColor.array[0],.25);
  next.install();assert.equal(view.attributes.nodeColor.array[0],.5);
  color.array[0]=.75;color.addUpdateRange(0,0);
  const empty=histories.capture(descriptor,3);empty.install();
  assert.equal(view.attributes.nodeColor.array[0],.5,'zero WebGPU range does not upload remaining bytes');
  assert.equal(color.updateRanges.length,0);histories.dispose();
});

for(const [Type,otherType,word] of [[Uint32Array,Int32Array,0xf0000001],[Int32Array,Uint32Array,-0x70000001]])
test(`${Type.name} generated interleaved inputs preserve queued partial updates and signedness`,()=>{
  const histories=createThreeNodeAttributes(T),descriptor={},geometry=new T.PlaneGeometry();
  const source=new T.InstancedInterleavedBuffer(new Type([word,11,word+2,13,word+4,15,word+6,17]),4,1);
  const attributes=[0,2].map((offset,i)=>({name:`input${i}`,attribute:new T.InterleavedBufferAttribute(source,2,offset)}));
  const view=histories.register(descriptor,geometry,attributes),first=histories.capture(descriptor,1);
  source.array[0]=word+20;source.array[4]=word+40;
  const stale=histories.capture(descriptor,2);
  source.addUpdateRange(4,1);source.needsUpdate=true;
  const partial=histories.capture(descriptor,3);
  assert.deepEqual(source.updateRanges,[]);assert.equal(partial.bytes,32);
  source.array.fill(0);
  first.install();assert.equal(view.attributes.input0.data.array.constructor,Type);
  assert.deepEqual([...view.attributes.input0.data.array],[word,11,word+2,13,word+4,15,word+6,17]);
  stale.install();assert.equal(view.attributes.input0.data.array[0],word);
  partial.install();assert.equal(view.attributes.input0.data.array[0],word);assert.equal(view.attributes.input0.data.array[4],word+40);
  assert.equal(view.attributes.input0.data,view.attributes.input1.data);
  assert.deepEqual([...source.array],Array(8).fill(0));
  const replacement={};const next=histories.register(replacement,geometry,attributes);
  histories.capture(replacement,4).install();assert.equal(next.attributes.input0.data.array[4],word+40);
  source.array=new otherType(source.array.buffer);
  assert.throws(()=>histories.capture(descriptor,5),/identity or layout changed/);
  histories.dispose();
});

test('generated attribute history validates the whole selection before consuming ranges',()=>{
  const histories=createThreeNodeAttributes(T),descriptor={};
  const left=new T.InstancedBufferAttribute(new Float32Array([1,2,3]),3);
  const right=new T.InstancedBufferAttribute(new Float32Array([4,5,6]),3);
  histories.register(descriptor,new T.PlaneGeometry(),[{name:'left',attribute:left},{name:'right',attribute:right}]);
  histories.capture(descriptor,1);
  left.addUpdateRange(0,1);left.needsUpdate=true;right.addUpdateRange(3,1);right.needsUpdate=true;
  assert.throws(()=>histories.capture(descriptor,2),/bounded source update ranges/);
  assert.equal(left.updateRanges.length,1);
  right.clearUpdateRanges();right.array=new Float32Array(6);
  assert.throws(()=>histories.capture(descriptor,2),/complete source stride/);
  assert.equal(left.updateRanges.length,1);histories.dispose();
  const small=createThreeNodeAttributes(T,{maxBytes:8});
  assert.throws(()=>small.register({},new T.PlaneGeometry(),[{name:'left',attribute:left}]),/byte budget/);
  assert.equal(small.diagnostics.owners,0);small.dispose();
});

test('shared Float16Array inputs accept native vectors and reject invalid scalar, normalization and alignment profiles',
  {skip: typeof Float16Array !== 'function'}, () => {
    const reject = (code, message) => { throw Object.assign(new Error(message), {code}); };
    for (const components of [2, 3, 4]) {
      const attribute = new T.BufferAttribute(new Float16Array(components * 4), components);
      assert.equal(inspectThreeNodeSharedAttribute(T, attribute, reject).array, attribute.array);
      attribute.normalized = true;
      assert.throws(() => inspectThreeNodeSharedAttribute(T, attribute, reject), {code: 'ATTRIBUTE'});
      attribute.normalized = false; attribute.isFloat16BufferAttribute = true;
      assert.throws(() => inspectThreeNodeSharedAttribute(T, attribute, reject), {code: 'ATTRIBUTE'});
    }
    for (const attribute of [new T.BufferAttribute(new Float16Array(4), 1),
      new T.InterleavedBufferAttribute(new T.InterleavedBuffer(new Float16Array(12), 3), 3, 0),
      new T.InterleavedBufferAttribute(new T.InterleavedBuffer(new Float16Array(24), 6), 3, 1)])
      assert.throws(() => inspectThreeNodeSharedAttribute(T, attribute, reject), {code: 'ATTRIBUTE'});
    const owner = new T.InterleavedBuffer(new Float16Array(32), 8);
    const attribute = new T.InterleavedBufferAttribute(owner, 3, 4);
    assert.equal(inspectThreeNodeSharedAttribute(T, attribute, reject).owner, owner);
  });
