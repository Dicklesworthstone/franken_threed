/** Real two-level CPU raycasting. The sceneBvh:false route is the linear mesh
 * reference; both routes use the existing triangle arithmetic, not GPU doubles. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationRaycaster} from './animation_raycast.mjs';

const identity=()=>new Float64Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
const triangle=()=>new Float32Array([-1,-1,0,1,-1,0,0,1,0]);
const ray=(x=0,y=0,z=2,extra={})=>({origin:[x,y,z],direction:[0,0,-1],...extra});
function fixture(t,count,place=(i,d)=>{d.worldMatrix[12]=i*4;},options={}) {
  const pose={nodeCount:count,version:0,disposed:false};
  const entries=Array.from({length:count},(_,node)=>{
    const deformer={node,vertexCount:3,positions:triangle(),worldMatrix:identity(),
      version:0,poseVersion:0,disposed:false,failed:false};
    place(node,deformer);
    return {deformer,source:{node,mesh:1000+node,primitive:7,material:2}};
  });
  const tree=createAnimationRaycaster(pose,entries,options);
  const linear=createAnimationRaycaster(pose,entries,{...options,sceneBvh:false});
  t.after(()=>{tree.dispose();linear.dispose();});
  const compare=(input,settings)=>{
    const expected=linear.raycast(input,settings),actual=tree.raycast(input,settings);
    assert.deepEqual(actual,expected);return actual;
  };
  const advance=change=>{
    pose.version++;
    for(const {deformer:d}of entries){if(change(d)){d.version++;}d.poseVersion=pose.version;}
  };
  return {pose,entries,tree,linear,compare,advance};
}

test('scene hierarchy rejects most meshes in a 1024-draw grid without changing hits',t=>{
  const {compare,tree,linear}=fixture(t,1024,(i,d)=>{
    d.worldMatrix[12]=(i%32)*4;d.worldMatrix[13]=Math.floor(i/32)*4;
  });
  const hits=compare(ray(60,64));assert.equal(hits.length,1);assert.equal(hits[0].drawIndex,527);
  assert.equal(linear.lastQuery.meshesTested,1024);assert.ok(tree.lastQuery.meshesTested<=4);
  assert.ok(tree.lastQuery.sceneBoxesTested<=25);assert.equal(tree.lastQuery.sceneRebuilds,1);
  assert.equal(tree.lastQuery.refittedMeshes,1024);
  compare(ray(60,64));assert.equal(tree.lastQuery.sceneRefits,0);assert.equal(tree.lastQuery.refittedMeshes,0);
  assert.equal(tree.lastQuery.sceneRebuilds,0);
});

test('root miss performs one scene box test and no mesh traversal',t=>{
  const {compare,tree}=fixture(t,257);
  assert.deepEqual(compare(ray(-100)),[]);
  assert.equal(tree.lastQuery.meshesTested,0);assert.equal(tree.lastQuery.trianglesTested,0);
  assert.equal(tree.lastQuery.sceneBoxesTested,1);
});

test('global nearest-first traversal prunes farther meshes even with reverse source order',t=>{
  const {compare,tree,linear}=fixture(t,256,(i,d)=>{d.worldMatrix[14]=i-256;});
  const hits=compare(ray(),{firstHitOnly:true,maxHits:1});
  assert.equal(hits[0].drawIndex,255);assert.equal(hits[0].distance,3);
  assert.equal(linear.lastQuery.meshesTested,256);assert.ok(tree.lastQuery.meshesTested<=4);
  assert.ok(tree.lastQuery.trianglesTested<=4);
});

test('coincident and shared-boundary hits retain source draw/face order and inclusive intervals',t=>{
  const {compare,entries}=fixture(t,33,()=>{});
  for(const entry of entries){entry.deformer.worldMatrix[0]=-2;entry.deformer.worldMatrix[5]=3;}
  const hits=compare(ray(0,0,2,{near:2,far:2}));
  assert.equal(hits.length,33);assert.deepEqual(hits.map(h=>h.drawIndex),Array.from({length:33},(_,i)=>i));
  assert.equal(compare(ray(),{firstHitOnly:true})[0].drawIndex,0);
  assert.equal(compare(ray(0,3),{firstHitOnly:true})[0].drawIndex,0);
  assert.deepEqual(compare(ray(0,0,2,{far:1.99})),[]);
});

test('world and local deformation changes refit existing hierarchy without rebuilding',t=>{
  const {compare,tree,advance}=fixture(t,65);
  const old=compare(ray())[0];
  advance(d=>{if(d.node!==64)return false;d.worldMatrix[12]=0;d.worldMatrix[14]=1;return true;});
  const next=compare(ray(),{firstHitOnly:true})[0];assert.equal(next.drawIndex,64);assert.equal(next.distance,1);
  assert.equal(tree.lastQuery.refittedMeshes,1);assert.equal(tree.lastQuery.sceneRefits,1);assert.equal(tree.lastQuery.sceneRebuilds,0);
  assert.equal(old.distance,2);assert.equal(old.drawIndex,0);
  advance(d=>{if(d.node!==64)return false;for(let i=0;i<3;i++)d.positions[i*3]+=2;return true;});
  assert.equal(compare(ray(),{firstHitOnly:true})[0].drawIndex,0);
  assert.equal(tree.lastQuery.sceneRebuilds,0);
});

test('changing subsets rebuilds in reusable storage; equivalent subset order does not',t=>{
  const {compare,tree}=fixture(t,64);
  const a=Array.from({length:32},(_,i)=>i),b=Array.from({length:32},(_,i)=>i+32);
  compare(ray(0),{drawIndices:a});const bytes=tree.bufferBytes;
  compare(ray(0),{drawIndices:a.toReversed()});assert.equal(tree.lastQuery.sceneRebuilds,0);
  compare(ray(128),{drawIndices:b});assert.equal(tree.lastQuery.sceneRebuilds,1);
  const fullBytes=tree.bufferBytes;assert.ok(fullBytes>bytes);
  for(let i=0;i<12;i++)compare(ray(i%2?128:0),{drawIndices:i%2?b:a});
  assert.equal(tree.bufferBytes,fullBytes);
  assert.deepEqual(compare(ray(),{drawIndices:[]}),[]);assert.equal(tree.lastQuery.sceneBoxesTested,0);
  assert.equal(compare(ray(),{drawIndices:[0]})[0].drawIndex,0);assert.equal(tree.lastQuery.sceneBoxesTested,0);
  compare(ray(),{drawIndices:a});assert.equal(tree.lastQuery.sceneRebuilds,1);
});

test('small-selection refits invalidate a previously cached large scene hierarchy',t=>{
  const {compare,tree,advance}=fixture(t,32);
  compare(ray());
  advance(d=>{if(d.node!==31)return false;d.worldMatrix[12]=0;d.worldMatrix[14]=1;return true;});
  assert.equal(compare(ray(),{drawIndices:[31]})[0].distance,1);
  assert.equal(compare(ray(),{firstHitOnly:true})[0].drawIndex,31);
  assert.equal(tree.lastQuery.sceneRefits,1);assert.equal(tree.lastQuery.sceneRebuilds,0);
});

test('excluded meshes may remain stale or unavailable, but selected meshes must be current',t=>{
  const {tree,entries,pose}=fixture(t,16);
  pose.version=1;entries[0].deformer.poseVersion=1;
  for(let i=1;i<entries.length;i++)Object.defineProperty(entries[i].deformer,'positions',{get(){throw Error('inactive deformer accessed');}});
  assert.equal(tree.raycast(ray(),{drawIndices:[0]})[0].drawIndex,0);
  assert.deepEqual(tree.raycast(ray(),{drawIndices:[]}),[]);
  const previous=tree.lastQuery;
  assert.throws(()=>tree.raycast(ray(),{drawIndices:[1]}),{code:'ANIMATION_PICK_STALE'});
  assert.equal(tree.lastQuery,previous);
  entries[1].deformer.poseVersion=1;
  assert.throws(()=>tree.raycast(ray(),{drawIndices:[1]}),/inactive deformer accessed/);
});

test('failed active refits do not hide malformed geometry behind stale scene bounds',t=>{
  const {compare,tree,entries}=fixture(t,32);
  compare(ray());const previous=tree.lastQuery,d=entries[31].deformer;
  d.worldMatrix[12]=0;d.worldMatrix[14]=1;d.positions[0]=NaN;d.version++;
  assert.throws(()=>tree.raycast(ray()),{code:'ANIMATION_PICK_VALUE'});assert.equal(tree.lastQuery,previous);
  d.positions[0]=-1;
  assert.equal(compare(ray(),{firstHitOnly:true})[0].drawIndex,31);
  assert.equal(tree.lastQuery.sceneRefits,1);
});

test('rejected hit budgets preserve last published stats and leave the hierarchy recoverable',t=>{
  const {compare,tree}=fixture(t,32,()=>{});
  compare(ray(),{firstHitOnly:true});const previous=tree.lastQuery;
  assert.throws(()=>tree.raycast(ray(),{maxHits:1}),{code:'ANIMATION_PICK_LIMIT'});
  assert.equal(tree.lastQuery,previous);assert.equal(compare(ray()).length,32);
});

test('hierarchy storage is budgeted before use, lazy, bounded and released on dispose',t=>{
  // Nine one-triangle meshes: 172 reserved bytes per mesh (including build
  // scratch), plus 7 scene nodes * 60 and 9 order indices * 4.
  const required=9*172+7*60+9*4;
  assert.throws(()=>fixture(t,9,undefined,{maxBytes:required-1}),{code:'ANIMATION_PICK_LIMIT'});
  const {tree,entries}=fixture(t,9,undefined,{maxBytes:required});
  assert.equal(tree.bufferBytes,9*12);
  tree.raycast(ray());assert.equal(tree.bufferBytes,9*148+7*60+9*4);assert.ok(tree.bufferBytes<=required);
  tree.dispose();assert.equal(tree.bufferBytes,0);assert.ok(entries.every(e=>!e.deformer.disposed));
  assert.throws(()=>tree.raycast(ray()),{code:'ANIMATION_PICK_DISPOSED'});
  assert.throws(()=>fixture(t,9,undefined,{sceneBvh:1}),{code:'ANIMATION_PICK_OPTION'});
});

test('randomized rays, affine transforms, subsets and pose updates match complete linear hit snapshots',t=>{
  let seed=0x6d2b79f5;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  const {compare,advance}=fixture(t,96,(i,d)=>{
    const m=d.worldMatrix;m[0]=(random()<0.5?-1:1)*(0.3+random()*2);m[5]=0.3+random()*2;
    m[4]=(random()-0.5)*2;m[8]=random()-0.5;m[9]=random()-0.5;
    m[12]=(random()-0.5)*24;m[13]=(random()-0.5)*24;m[14]=-random()*20;
  });
  for(let round=0;round<4;round++) {
    for(let i=0;i<180;i++) {
      const settings={firstHitOnly:i%2===0};
      if(i%3===0)settings.drawIndices=Array.from({length:96},(_,j)=>j).filter(()=>random()<0.5).reverse();
      const origin=[(random()-0.5)*24,(random()-0.5)*24,2+random()*2];
      const direction=[(random()-0.5)*0.1,(random()-0.5)*0.1,-1];
      compare({origin,direction,near:i%7===0?5:0,far:i%5===0?15:Infinity},settings);
    }
    advance(d=>{if(random()>0.25)return false;d.worldMatrix[12]+=(random()-0.5)*8;d.worldMatrix[14]+=(random()-0.5)*3;return true;});
  }
});
