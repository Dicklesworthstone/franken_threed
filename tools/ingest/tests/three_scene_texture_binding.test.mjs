/** Production source binding policy + production shared texture owner.
 * The existing source-descriptor and recording-device fixtures are boundary
 * doubles. These are not full scene traversal or executed WGSL/pixel tests.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuThreeTextures,createThreeTextureBindings} from '../three_textures.mjs';
import {T,texture,device} from './volume_texture_test_fixture.mjs';

function setup({d=device(),autoTextures=true,limits={}}={}) {
  let owner=null;
  const state={textures:new Map(),autoTextures,textureScan:null,frameTextures:null,pendingTextures:new WeakSet(),
    ownedTextures(){return owner??=createGpuThreeTextures(d,{three:T,...limits});},
    fail(code,message){throw Object.assign(new Error(message),{code:'THREE_SCENE_'+code});},
  };
  const bindings=createThreeTextureBindings(d,state);
  return {d,state,bindings,get owner(){return owner;},
    scan(values){state.textureScan=new Set();try{for(const t of values)bindings.binding(t);return state.textureScan;}finally{state.textureScan=null;}},
    prepare(values){state.pendingTextures=new WeakSet();const requested=this.scan(values);owner?.prepare(requested);owner?.retain(requested);return requested;},
    update(values){state.frameTextures=new Set();try{for(const t of values)bindings.binding(t);owner?.update(state.frameTextures);}finally{state.frameTextures=null;}},
    dispose(){bindings.dispose();owner?.dispose();},
  };
}
const ordinary=()=>{const t=new T.DataTexture(new Uint8Array([1,2,3,4]),1,1);t.needsUpdate=true;return t;};
const morph=()=>{const t=texture('array',2,2,3,4,4);t.isF3DMorphTexture=true;return t;};

for(const kind of ['array','volume','morph']) {
  const make=()=>kind==='morph'?morph():texture(kind);
  test(`${kind}: scans are metadata-only and populate the shared owner's request set`,()=>{
    const s=setup(),t=make(),requested=s.scan([t,t]);
    assert.deepEqual([...requested],[t]);assert.equal(s.d.textures.length,0);assert.equal(s.d.writes.length,0);
    s.owner.prepare(requested);const b=s.bindings.binding(t);
    assert.equal(b.view,s.owner.binding(t).view);assert.equal(b.sampler,s.owner.binding(t).sampler);
    assert.equal(b.viewDimension,kind==='volume'?'3d':'2d-array');
    assert.equal(s.owner.diagnostics.textures,1);assert.equal(s.d.textures.length,1);
    s.dispose();assert.ok(s.d.textures.every(t=>t.destroyed));
  });
  test(`${kind}: source versions preserve stale CPU edits and shared-source residency`,()=>{
    const s=setup(),t=make(),clone=t.clone();s.prepare([t,clone]);
    assert.equal(s.d.textures.length,1);assert.equal(s.d.writes.length,1);
    const gpu=s.owner.nativeTexture(t),before=gpu.levels[0].slice();
    new Uint8Array(t.image.data.buffer).fill(7);s.update([t,clone]);
    assert.deepEqual(gpu.levels[0],before);assert.equal(s.d.writes.length,1);
    t.needsUpdate=true;s.update([t,clone]);assert.ok(gpu.levels[0].every(v=>v===7));
    assert.equal(s.d.writes.length,2);assert.equal(s.bindings.binding(clone).view,s.bindings.binding(t).view);
    s.owner.retain([clone]);assert.equal(gpu.destroyed,false);clone.dispose();assert.equal(gpu.destroyed,true);s.dispose();
  });
  test(`${kind}: automatic ownership can be disabled and borrowed bindings retain priority`,()=>{
    const s=setup({autoTextures:false}),t=make();
    assert.throws(()=>s.bindings.binding(t),{code:'THREE_SCENE_TEXTURE'});assert.equal(s.owner,null);
    const b={view:{},sampler:{},version:t.version,sourceVersion:t.source.version};s.state.textures.set(t,b);
    assert.equal(s.bindings.binding(t),b);assert.equal(s.owner,null);
    // Caller ownership also bypasses the automatic owner's hook restriction.
    t.onUpdate=()=>{};assert.equal(s.bindings.binding(t),b);
    t.needsUpdate=true;assert.throws(()=>s.bindings.binding(t),{code:'THREE_SCENE_TEXTURE'});
    b.version=t.version;assert.throws(()=>s.bindings.binding(t),{code:'THREE_SCENE_TEXTURE'});
    b.sourceVersion=t.source.version;assert.equal(s.bindings.binding(t),b);s.dispose();assert.equal(s.d.textures.length,0);
  });
  test(`${kind}: automatic upload callbacks are not silently discarded`,()=>{
    const s=setup(),t=make();t.onUpdate=()=>assert.fail('automatic scene callbacks are not admitted');
    assert.throws(()=>s.scan([t]),{code:'THREE_SCENE_HOOK'});assert.equal(s.owner,null);assert.equal(s.d.textures.length,0);s.dispose();
  });
}

test('array frame updates upload only selected layers through the shared owner',()=>{
  const s=setup(),t=texture('array',2,2,3);s.prepare([t]);
  const gpu=s.owner.nativeTexture(t),before=gpu.levels[0].slice();t.image.data.fill(91);t.addLayerUpdate(1);t.needsUpdate=true;
  s.update([t]);assert.equal(s.d.writes.length,2);assert.deepEqual(s.d.writes[1].dest.origin,[0,0,1]);
  assert.deepEqual(gpu.levels[0].slice(0,4),before.slice(0,4));
  assert.deepEqual([...gpu.levels[0].slice(4,8)],[91,91,91,91]);assert.deepEqual(gpu.levels[0].slice(8),before.slice(8));
  assert.equal(t.layerUpdates.size,0);assert.equal(s.d.textures.length,1);s.dispose();
});

for(const kind of ['array','volume'])test(`${kind}: generated mipmaps now use the same owner from scene bindings`,()=>{
  const s=setup(),t=texture(kind,4,4,4);t.generateMipmaps=true;t.minFilter=T.LinearMipmapLinearFilter;
  s.prepare([t]);assert.equal(s.owner.inspect(t).levels,3);assert.ok(s.d.passes.length>0);
  const expected=kind==='volume'?3:8;
  assert.equal(s.d.passes.length,expected);assert.equal(s.d.textures.length,1);
  const b=s.bindings.binding(t);assert.equal(b.sampler.lodMaxClamp,2);
  t.needsUpdate=true;s.update([t]);assert.equal(s.d.passes.length,expected*2);assert.equal(s.d.textures.length,1);s.dispose();
});

test('morph arrays and ordinary maps participate in one aggregate memory budget',()=>{
  const m=morph(),map=ordinary(),s=setup({limits:{maxTextureBytes:192}});
  const requested=s.scan([m,map]);assert.equal(s.d.textures.length,0);
  assert.throws(()=>s.owner.prepare(requested),{code:'THREE_TEXTURE_LIMIT'});assert.equal(s.d.textures.length,0);s.dispose();
  const ok=setup({limits:{maxTextureBytes:196}});ok.prepare([m,map]);assert.equal(ok.owner.diagnostics.textureBytes,196);
  assert.equal(ok.owner.diagnostics.textures,2);assert.equal(ok.bindings.binding(m).sampleType,'unfilterable-float');ok.dispose();
});

test('late invalid array input cannot allocate or upload an earlier valid source during scanning',()=>{
  const s=setup(),good=morph(),bad=texture();bad.image.data=new Uint8Array(1);
  assert.throws(()=>s.scan([good,bad]),{code:'THREE_TEXTURE_STORAGE'});
  assert.equal(s.d.textures.length,0);assert.equal(s.d.writes.length,0);assert.equal(s.owner.failed,false);s.dispose();
});

test('structural mutations and source disposal become renderer-recognized preparation boundaries',()=>{
  const s=setup(),t=texture();s.prepare([t]);const original=s.bindings.binding(t).view;
  t.wrapR=T.RepeatWrapping;
  assert.throws(()=>s.bindings.binding(t),{code:'THREE_SCENE_PREPARE'});assert.equal(original.texture.destroyed,false);
  s.prepare([t]);assert.equal(original.texture.destroyed,true);assert.equal(s.bindings.binding(t).sampler.addressModeW,'repeat');
  t.dispose();assert.throws(()=>s.bindings.binding(t),{code:'THREE_SCENE_PREPARE'});s.prepare([t]);s.dispose();
});

test('source count and independent native volume dimensions are enforced before residency',()=>{
  const s=setup({limits:{maxTextures:1}});const requested=s.scan([texture(),morph()]);
  assert.throws(()=>s.owner.prepare(requested),{code:'THREE_TEXTURE_LIMIT'});assert.equal(s.d.textures.length,0);s.dispose();
  const small=setup({d:device({limits:{maxTextureDimension3D:2}})});
  assert.throws(()=>small.scan([texture('volume',3,1,1)]),{code:'THREE_TEXTURE_VALUE'});assert.equal(small.d.textures.length,0);small.dispose();
});

for(const kind of ['array','volume'])test(`${kind}: loading defaults have the right dimension and arrival requires preparation`,()=>{
  const s=setup(),t=kind==='array'?new T.DataArrayTexture(null,1,1,1):new T.Data3DTexture(null,1,1,1);
  s.prepare([t]);const before=s.bindings.binding(t),dimension=kind==='array'?'2d-array':'3d';
  assert.equal(before.view.dimension,dimension);assert.equal(s.d.writes.length,0);assert.equal(s.owner.diagnostics.textures,0);
  t.image.data=new Uint8Array([11,22,33,44]);t.needsUpdate=true;
  assert.throws(()=>s.bindings.binding(t),{code:'THREE_SCENE_PREPARE'});assert.equal(s.d.writes.length,0);
  s.prepare([t]);const after=s.bindings.binding(t);assert.notEqual(after.view,before.view);assert.equal(after.view.dimension,dimension);
  assert.deepEqual([...after.view.texture.levels[0]],[11,22,33,44]);s.dispose();assert.ok(s.d.textures.every(t=>t.destroyed));
});

test('default cache keys include dimensionality and integer sampler type',()=>{
  const s=setup();
  for(const dimension of ['2d','2d-array','3d','cube'])for(const [prefix,sampleType,format] of [['','float','rgba8unorm'],['u','uint','rgba8uint'],['i','sint','rgba8sint']]){
    const reflection={dimension,glslType:prefix+'sampler2D'},b=s.bindings.placeholder(null,reflection);
    assert.equal(b,s.bindings.placeholder(null,reflection));assert.equal(b.view.dimension,dimension);assert.equal(b.sampleType,sampleType);
    assert.equal(b.texture.desc.format,format);assert.equal(b.texture.layers,dimension==='cube'?6:1);
    assert.ok(b.texture.levels[0].every(v=>v===0));
  }
  assert.equal(s.d.textures.length,12);assert.equal(s.d.writes.length,0);assert.equal(s.owner,null);
  s.dispose();assert.ok(s.d.textures.every(t=>t.destroyed));
  assert.throws(()=>s.bindings.placeholder(),{code:'THREE_SCENE_DISPOSED'});
});

test('shadow inspection defaults have depth storage and comparison samplers',()=>{
  const s=setup(),b=s.bindings.placeholder(null,{dimension:'2d',comparison:true,glslType:'sampler2DShadow'});
  assert.equal(b.sampleType,'depth');assert.equal(b.texture.desc.format,'depth32float');assert.equal(b.sampler.compare,'less-equal');
  assert.throws(()=>s.bindings.placeholder(null,{dimension:'3d',comparison:true}),{code:'THREE_SCENE_TEXTURE'});
  assert.throws(()=>s.bindings.placeholder(null,{dimension:'cube-array'}),{code:'THREE_SCENE_TEXTURE'});
  assert.equal(s.d.textures.length,1);s.dispose();
});

test('failed default view or sampler creation releases the partially allocated texture',()=>{
  for(const part of ['view','sampler']){
    const d=device(),s=setup({d}),sentinel=new Error('injected '+part+' failure');
    if(part==='sampler')d.createSampler=()=>{throw sentinel;};
    else {const create=d.createTexture;d.createTexture=desc=>{const t=create(desc);t.createView=()=>{throw sentinel;};return t;};}
    assert.throws(()=>s.bindings.placeholder(),e=>e===sentinel);assert.equal(d.textures[0].destroyed,true);s.dispose();
  }
});

test('native shared-owner failures and device loss propagate to the scene resource owner',async()=>{
  for(const type of ['write','validation','loss']){
    const s=setup(),t=morph();s.prepare([t]);await s.owner.whenIdle();
    if(type==='loss'){s.d.loss.resolve({message:'lost'});await assert.rejects(s.owner.whenIdle(),{code:'THREE_TEXTURE_LOST'});}
    else{
      t.needsUpdate=true;
      if(type==='write'){s.d.failWrite=true;assert.throws(()=>s.update([t]),/injected write failure/);}
      else{s.d.scopeError={message:'invalid texture'};s.update([t]);await assert.rejects(s.owner.whenIdle(),{code:'THREE_TEXTURE_DEVICE'});}
    }
    assert.equal(s.owner.failed,true);assert.equal(s.owner.diagnostics.textureBytes,0);assert.ok(s.d.textures.every(t=>t.destroyed));s.dispose();
  }
});

test('ordinary 2D bindings retain their established source-version and loading behavior',()=>{
  const s=setup(),a=ordinary();s.prepare([a]);assert.equal(s.bindings.binding(a).view.dimension,'2d');
  const loading=new T.DataTexture(null,1,1);s.prepare([a,loading]);
  assert.equal(s.bindings.binding(loading).view.dimension,'2d');assert.equal(s.owner.diagnostics.textures,1);
  s.state.frameTextures=new Set();s.bindings.binding(loading);s.bindings.binding(a);
  assert.deepEqual([...s.state.frameTextures],[a]);s.dispose();
});

test('allocation-only preparation reserves version-zero data without upload or source acknowledgment',async()=>{
  const d=device(),p=createGpuThreeTextures(d,{three:T}),t=new T.DataTexture(new Uint8Array([5,6,7,8]),1,1);
  let callbacks=0;t.onUpdate=()=>callbacks++;
  assert.throws(()=>p.prepare([t]),{code:'THREE_TEXTURE_NOT_READY'});
  p.prepare([t],{initialize:true});const b=p.binding(t),native=p.nativeTexture(t);
  assert.equal(t.version,0);assert.equal(t.source.version,0);assert.equal(b.sourceVersion,-1);
  assert.equal(d.writes.length,0);assert.equal(callbacks,0);assert.ok(native.levels[0].every(v=>v===0));
  p.update([t]);assert.equal(d.writes.length,0);
  t.needsUpdate=true;p.update([t]);await p.whenIdle();
  assert.equal(p.binding(t).view,b.view);assert.deepEqual([...native.levels[0]],[5,6,7,8]);
  assert.equal(callbacks,1);assert.equal(p.binding(t).sourceVersion,t.source.version);p.dispose();
});

test('allocation-only preparation still validates the full request and its peak budget',()=>{
  const d=device(),p=createGpuThreeTextures(d,{three:T,maxTextureBytes:4});
  const t=new T.DataTexture(new Uint8Array(4),1,1),bad=new T.DataTexture(new Uint8Array(1),1,1);
  assert.throws(()=>p.prepare([t,bad],{initialize:true}),{code:'THREE_TEXTURE_STORAGE'});
  assert.equal(d.textures.length,0);
  p.prepare([t],{initialize:true});const storage=p.nativeTexture(t);
  t.wrapS=T.RepeatWrapping;
  assert.throws(()=>p.prepare([t],{initialize:true}),{code:'THREE_TEXTURE_LIMIT'});
  assert.equal(storage.destroyed,false);assert.equal(d.writes.length,0);p.dispose();
});
