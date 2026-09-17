import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer,AnimationPoseError} from './animation_runtime.mjs';
import {createAnimationController} from './animation_controller.mjs';

const track=(node,path,values,times=[0,2])=>({node,path,times,values});
const definition=(extra={})=>({format:'f3d-animation-v1',nodes:[{}],clips:[
  {name:'walk',channels:[track(0,'translation',[0,0,0,2,4,6])]},
  {name:'run',channels:[track(0,'translation',[10,0,0,14,0,0])]},
],...extra});
const make=extra=>createAnimationController(createAnimationPlayer(definition(extra)));
const close=(a,b,e=1e-10)=>assert.ok(Math.abs(a-b)<=e,`${a} != ${b}`);
const events=c=>c.events.map(({type,count,direction})=>({type,count,direction}));
const state=a=>[a.time,a.weight,a.playing,a.paused,a.finished,a.completedTraversals,a.effectiveWeight];

test('independent actions can share a clip; explicit updates publish stable output storage',()=>{
  const c=make(),a=c.createAction(0).play(),b=c.createAction(0).play().seek(1),out=c.pose.worldMatrices;
  assert.notEqual(a,b);c.update(0.5);close(a.time,0.5);close(b.time,1.5);close(c.pose.translations[0],1);
  assert.equal(c.pose.worldMatrices,out);assert.equal(c.time,0.5);assert.equal(c.actionCount,2);
  a.pause();c.update(0.25);close(a.time,0.5);close(b.time,1.75);close(c.pose.translations[0],1.125);
  a.play();b.stop();c.update(0.25);close(a.time,0.75);close(c.pose.translations[0],0.75);
  a.stop();c.update(0);assert.deepEqual([...c.pose.translations],[0,0,0]);
});

for(const clamp of [false,true])test(`once completion at the exact endpoint, clamp=${clamp}`,()=>{
  const c=make(),a=c.createAction(0,{loop:'once',clampWhenFinished:clamp}).play();
  c.update(2);assert.equal(a.finished,true);assert.equal(a.playing,false);assert.equal(a.time,2);
  close(c.pose.translations[0],clamp?2:0);assert.equal(c.events.length,1);
  assert.equal(c.events[0].action,a);assert.equal(c.events[0].type,'finished');
  const saved=c.events;c.update(5);assert.equal(c.events.length,0);assert.equal(saved.length,1);
  a.play();assert.equal(a.time,0);assert.equal(a.finished,false);c.update(0.5);close(a.time,0.5);
});

for(const speed of [1,-1])test(`repeat finite traversals, exact boundaries and aggregated events, speed=${speed}`,()=>{
  const c=make(),a=c.createAction(0,{timeScale:speed,repetitions:3,clampWhenFinished:true}).play();
  assert.equal(a.time,speed>0?0:2);
  c.update(2);assert.equal(a.time,speed>0?0:2);assert.equal(a.completedTraversals,1);
  assert.deepEqual(events(c),[{type:'loop',count:1,direction:speed}]);
  c.update(10);assert.equal(a.time,speed>0?2:0);assert.equal(a.completedTraversals,3);
  assert.deepEqual(events(c),[{type:'loop',count:1,direction:speed},{type:'finished',count:undefined,direction:speed}]);
  c.update(1);assert.equal(c.events.length,0);
});

for(const speed of [1,-1])test(`pingpong reflects at each endpoint without repeating events, speed=${speed}`,()=>{
  const c=make(),a=c.createAction(0,{loop:'pingpong',timeScale:speed}).play();
  const expected=speed>0?[2,1,0,1,2]:[0,1,2,1,0];
  for(const [i,dt] of [2,1,1,1,1].entries()){
    c.update(dt);close(a.time,expected[i]);assert.equal(c.events.length,i%2===0?1:0);
    const count=a.completedTraversals;c.update(0);assert.equal(c.events.length,0);assert.equal(a.completedTraversals,count);
  }
  // Reversing the clock mid-leg retraces that leg.
  c.update(0.5);const before=a.time;a.setTimeScale(-speed);c.update(0.25);
  close(a.time,before+(speed>0?0.25:-0.25));
});

for(const repetitions of [1,2,3,4])test(`finite pingpong stops on traversal ${repetitions}`,()=>{
  const c=make(),a=c.createAction(0,{loop:'pingpong',repetitions,clampWhenFinished:true}).play();
  c.update(100);assert.equal(a.finished,true);assert.equal(a.time,repetitions%2?2:0);
  assert.equal(a.completedTraversals,repetitions);
  const finish=c.events.at(-1);assert.equal(finish.type,'finished');assert.equal(finish.direction,repetitions%2?1:-1);
  if(repetitions>1)assert.equal(c.events[0].count,repetitions-1);
});

test('large looping steps have constant event count and correct residual phase',()=>{
  const c=make(),a=c.createAction(0,{loop:'repeat'}).play();
  c.update(2000000.25);close(a.time,0.25);assert.equal(c.events.length,1);assert.equal(c.events[0].count,1000000);
  a.setLoop('pingpong').reset();c.update(2000002.5);close(a.time,1.5);
  assert.equal(c.events[0].count,1000001);
  a.setLoop('once').reset();c.update(Number.MAX_VALUE/2);assert.equal(a.finished,true);
});

test('seek resets traversal budget without events and speed zero pauses only playback',()=>{
  const c=make(),a=c.createAction(0,{repetitions:2}).play();c.update(2.5);
  a.seek(1);assert.equal(a.completedTraversals,0);c.update(0);assert.equal(c.events.length,0);close(c.pose.translations[0],1);
  a.setTimeScale(0).fadeTo(0.5,1);c.update(0.5);assert.equal(a.time,1);close(a.weight,0.75);
  a.setTimeScale(-1);c.update(0.5);close(a.time,0.5);close(a.weight,0.5);
});

test('crossfades advance both playheads and stop the outgoing action',()=>{
  const c=make(),walk=c.createAction(0).play(),run=c.createAction(1);
  c.update(0.25);c.crossFade(walk,run,1);c.update(0.5);
  close(walk.time,0.75);close(run.time,0.5);close(walk.weight,0.5);close(run.weight,0.5);
  close(c.pose.translations[0],0.5*0.75+0.5*11);
  c.update(0.5);assert.equal(walk.playing,false);close(run.weight,1);close(c.pose.translations[0],12);
  c.update(0.5);close(c.pose.translations[0],13);
});

test('interrupted fades start from the current weights; paused actions still fade',()=>{
  const c=make(),a=c.createAction(0).play(),b=c.createAction(1);
  c.crossFade(a,b,2);c.update(0.5);close(a.weight,0.75);close(b.weight,0.25);
  c.crossFade(b,a,1);a.pause();const t=a.time;c.update(0.5);
  close(a.time,t);close(a.weight,0.875);close(b.weight,0.125);
  c.update(0.5);assert.equal(b.playing,false);close(a.weight,1);
  a.play();c.crossFade(a,b,0);assert.equal(a.playing,false);close(b.weight,1);c.update(0);
  close(c.pose.translations[0],10);
});

test('fade-out stops playback at the fade boundary rather than advancing the whole large delta',()=>{
  const c=make(),a=c.createAction(0).play().fadeTo(0,0.5,{stopWhenDone:true});
  c.update(100);assert.equal(a.playing,false);assert.equal(c.events.length,0);close(c.pose.translations[0],0);
});

test('node-masked additive actions use rest-relative deltas after normal actions',()=>{
  const c=make({nodes:[{translation:[10,0,0]},{}],clips:[
    {channels:[track(0,'translation',[20,0,0,20,0,0])]},
    {channels:[track(0,'translation',[14,0,0,14,0,0]),track(1,'translation',[0,8,0,0,8,0])]},
  ]});
  const mask=[0.5,1];c.createAction(0).play();const add=c.createAction(1,{mode:'additive',mask}).play();
  mask.fill(0);c.update(0.5);close(c.pose.translations[0],22);close(c.pose.translations[4],8);
  add.setWeight(0.5);c.update(0);close(c.pose.translations[0],21);close(c.pose.translations[4],4);
  add.setMask([0,1]);c.update(0);close(c.pose.translations[0],20);
});

test('failed pose evaluation rolls back clocks, fades, completion and events together',()=>{
  const c=make({nodes:[{},{}],skins:[{joints:[1]}],instances:[{node:0,skin:0}],clips:[
    {channels:[track(0,'scale',[1,1,1,-1,1,1])]},
  ]}),a=c.createAction(0,{loop:'once',clampWhenFinished:true}).play();
  c.update(0.5);a.fadeTo(0.5,1);const before=state(a),t=c.time,version=c.pose.version,eventList=c.events,world=[...c.pose.worldMatrices];
  // Full weight at t=1 is singular. Also check a failure late in pose input validation.
  a.setWeight(1);const still=state(a);
  assert.throws(()=>c.update(0.5),{code:'ANIMATION_SINGULAR_MESH'});
  assert.deepEqual(state(a),still);assert.equal(c.time,t);assert.equal(c.pose.version,version);
  a.fadeTo(0.5,1);const fading=state(a);
  assert.throws(()=>c.update(0.25,{rootMatrix:[]}),AnimationPoseError);
  assert.deepEqual(state(a),fading);assert.equal(c.events,eventList);assert.deepEqual([...c.pose.worldMatrices],world);
  c.update(0.25);close(a.time,0.75);close(a.weight,0.875);assert.equal(c.time,t+0.25);
  assert.equal(before[0],0.5);
});

test('action changes cannot reenter an update from a root matrix getter',()=>{
  const c=make(),a=c.createAction(0).play();
  assert.throws(()=>c.update(1,{get rootMatrix(){a.stop();return null;}}),{code:'ANIMATION_REENTRANT'});
  assert.equal(a.playing,true);assert.equal(a.time,0);assert.equal(c.time,0);
});

test('events are immutable snapshots in action creation order',()=>{
  const c=make(),a=c.createAction(0,{loop:'once'}).play(),b=c.createAction(1,{repetitions:2}).play();
  c.update(4);assert.deepEqual(c.events.map(e=>[e.type,e.action]),[['finished',a],['loop',b],['finished',b]]);
  assert.equal(Object.isFrozen(c.events),true);assert.equal(Object.isFrozen(c.events[0]),true);
  const old=c.events;c.update(0);assert.equal(old.length,3);assert.equal(c.events.length,0);
});

test('zero-duration clips finish only once, including reverse and held playback',()=>{
  const c=make({clips:[{channels:[track(0,'translation',[3,4,5],[0])]}]}),a=c.createAction(0,{timeScale:-1,clampWhenFinished:true}).play();
  c.update(0);assert.equal(a.finished,false);close(c.pose.translations[0],3);
  c.update(1);assert.equal(a.finished,true);assert.equal(c.events.length,1);assert.equal(c.events[0].direction,-1);
  c.update(1);assert.equal(c.events.length,0);close(c.pose.translations[0],3);
});

test('action disposal releases capacity; controllers borrow rather than dispose players',()=>{
  const c=make(),a=c.createAction(0).play();a.dispose();assert.equal(c.actionCount,0);
  assert.throws(()=>a.play(),{code:'ANIMATION_ACTION_DISPOSED'});c.update(0);close(c.pose.translations[0],0);
  for(let i=0;i<256;i++)c.createAction(0);assert.throws(()=>c.createAction(0),{code:'ANIMATION_ACTION_LIMIT'});
  c.dispose();assert.equal(a.disposed,true);assert.equal(c.actionCount,0);assert.equal(c.pose.disposed,false);
  c.pose.sample(1);close(c.pose.translations[0],1);
  assert.throws(()=>c.update(0),{code:'ANIMATION_CONTROLLER_DISPOSED'});
});

test('invalid creation, controls and updates never partially change action state',()=>{
  const c=make(),a=c.createAction(0).play();
  for(const options of [{weight:-1},{weight:NaN},{timeScale:Infinity},{loop:'invalid'},{repetitions:0},{repetitions:1.5},
    {clampWhenFinished:1},{mode:'invalid'},{mask:[]},{mask:[2]},{mask:[NaN]}]){
    assert.throws(()=>c.createAction(0,options),AnimationPoseError);assert.equal(c.actionCount,1);
  }
  const before=state(a);
  for(const operation of [()=>a.seek(-1),()=>a.seek(3),()=>a.setWeight(-1),()=>a.setTimeScale(NaN),()=>a.setLoop('bad'),
    ()=>a.fadeTo(1,-1),()=>a.fadeTo(NaN,1),()=>a.setMask([2]),()=>a.setClampWhenFinished(1),()=>a.setBlendMode('bad'),
    ()=>c.update(-1),()=>c.update(Infinity),()=>c.update(NaN),()=>c.crossFade(a,a,1),()=>c.crossFade(a,{},1)]){
    assert.throws(operation,AnimationPoseError);assert.deepEqual(state(a),before);assert.equal(c.time,0);
  }
  assert.throws(()=>c.createAction(99),{code:'ANIMATION_INDEX'});
  a.setTimeScale(Number.MAX_VALUE);assert.throws(()=>c.update(2),AnimationPoseError);assert.equal(c.time,0);
});

test('many-step pingpong follows the analytic triangular wave in both directions',()=>{
  for(const speed of [0.5,1,3,-0.5,-1,-3]) {
    const c=make(),a=c.createAction(0,{loop:'pingpong',timeScale:speed}).play();let elapsed=0;
    for(let i=0;i<500;i++) {
      const dt=[0,0.125,0.25,0.5,1,4.125][i%6];elapsed+=dt;c.update(dt);
      const travel=elapsed*Math.abs(speed),phase=travel%4,forward=phase<=2?phase:4-phase;
      close(a.time,speed>0?forward:2-forward);close(c.pose.translations[0],a.time);
      assert.equal(a.completedTraversals,Math.floor(travel/2));
    }
  }
});
