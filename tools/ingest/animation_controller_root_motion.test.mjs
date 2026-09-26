import test from "node:test";
import assert from "node:assert/strict";
import { createAnimationPlayer } from "./animation_runtime.mjs";
import { createAnimationController } from "./animation_controller.mjs";
import { extractAnimationRootMotion, createAnimationRootMotionTrack, applyAnimationRootMotion } from "./animation_root_motion.mjs";

const channel = (node, path, values, times = [0,2], interpolation = "LINEAR") => ({node,path,values,times,interpolation});
const cues = [{name:"left",time:0.5},{name:"right",time:1.5},{name:"end",time:2}];
const fields = ["translations","rotations","scales","morphWeights","worldMatrices","jointMatrices"];
const bytes = pose => fields.map(field => Buffer.from(pose[field].buffer).toString("hex"));
const state = a => [a.time,a.weight,a.timeScale,a.playing,a.paused,a.finished,a.warping,a.completedTraversals,a.startTime];
const error = code => e => e?.code === code;
const close = (a,b,epsilon=1e-10) => {
  assert.equal(a.length,b.length);
  a.forEach((v,i) => assert.ok(Math.abs(v-b[i]) <= epsilon, `${v} != ${b[i]} at ${i}`));
};
function definition({interpolation="LINEAR",singular=false,duration=2}={}) {
  const root = interpolation === "CUBICSPLINE" ?
    channel(0,"translation",[0,0,2, 10,4,-3, 0,0,2, 12,4,2, 18,8,1, 12,4,2],[0,2],interpolation) :
    channel(0,"translation",duration ? [10,4,-3,18,6,1] : [10,4,-3],duration ? [0,duration] : [0],interpolation);
  return {format:"f3d-animation-v1",nodes:[
    {translation:[10,4,-3],weights:[0.2,0.4]}, {parent:0,translation:[0,1,0]},
    {translation:[4,0,0]}, {translation:[-4,0,0]},
  ],skins:[{joints:[0,1]}],instances:[{node:2,skin:0},{node:3,skin:0}],clips:[{name:"walk",channels:[root,
    channel(1,"rotation",[0,0,0,1,0,0,Math.SQRT1_2,Math.SQRT1_2]),
    channel(0,"weights",[0.2,0.4,0.8,-0.2]),
    channel(2,"scale",[1,1,1,singular?0:2,1,1]),
  ]}]};
}
function setup(actionOptions={}, defOptions={}, controllerOptions={}) {
  const pose = createAnimationPlayer(definition(defOptions));
  const controller = createAnimationController(pose,controllerOptions);
  const take = extractAnimationRootMotion(pose,{node:0});
  const [clip] = pose.addClips([take.clip]);
  const action = controller.createAction(clip,{rootMotion:take.rootMotion,...actionOptions}).play();
  return {pose,controller,action,take,clip};
}
const step = (c,a,dt,placement=c.rootMotionMatrix) => c.update(dt,{rootMotionAction:a,rootMatrix:placement});
function snapshot({pose,controller,action}) {
  return {bytes:bytes(pose),version:pose.version,time:controller.time,state:state(action),
    motion:action.rootMotionDelta,root:controller.rootMotionMatrix,events:controller.events,markers:controller.markerEvents};
}
function unchanged(s,before) {
  assert.deepEqual(bytes(s.pose),before.bytes); assert.equal(s.pose.version,before.version);
  assert.equal(s.controller.time,before.time); assert.deepEqual(state(s.action),before.state);
  assert.equal(s.action.rootMotionDelta,before.motion); assert.equal(s.controller.rootMotionMatrix,before.root);
  assert.equal(s.controller.events,before.events); assert.equal(s.controller.markerEvents,before.markers);
}

test("real clip extraction and live installation leave an animated source pose untouched", () => {
  const pose = createAnimationPlayer(definition());
  pose.sample(0.75); const before = bytes(pose), version = pose.version, clips = pose.clips;
  const saved = pose.snapshotClips();
  const take = extractAnimationRootMotion(pose,{node:0});
  assert.deepEqual(bytes(pose),before); assert.equal(pose.version,version); assert.equal(pose.clips,clips);
  assert.deepEqual(pose.snapshotClips(),saved);
  const [id] = pose.addClips([take.clip]);
  assert.equal(id,1); assert.equal(pose.clipVersion,1); assert.equal(pose.version,version);
  pose.sample(1,{clip:id}); close(Array.from(pose.translations.slice(0,3)),[10,5,-3]);
  assert.equal(pose.clips[0],clips[0]);
});

test("one blend publishes locomotion, local bob, world transforms and distinct skin palettes", () => {
  const s=setup({markers:cues}); const arrays=fields.map(f=>s.pose[f]);
  assert.deepEqual(s.action.rootMotionDelta,[0,0,0]); assert.equal(s.controller.rootMotionMatrix,null);
  step(s.controller,s.action,0.5);
  assert.equal(s.pose.version,1); assert.equal(s.pose.mode,"blend");
  close(s.action.rootMotionDelta,[2,0,1]); close(s.controller.rootMotionMatrix.slice(12,15),[2,0,1]);
  close(Array.from(s.pose.translations.slice(0,3)),[10,4.5,-3]);
  close(Array.from(s.pose.worldMatrices.slice(12,15)),[12,4.5,-2]);
  close(Array.from(s.pose.worldMatrices.slice(44,47)),[6,0,1]);
  close(Array.from(s.pose.morphWeights),[0.35,0.25]);
  assert.notDeepEqual(s.pose.jointMatrices.slice(0,32),s.pose.jointMatrices.slice(32));
  assert.deepEqual(s.controller.markerEvents.map(e=>e.name),["left"]);
  fields.forEach((f,i)=>assert.equal(s.pose[f],arrays[i]));
  assert.ok(Object.isFrozen(s.action.rootMotionDelta)); assert.ok(Object.isFrozen(s.controller.rootMotionMatrix));
  const prior=s.action.rootMotionDelta; step(s.controller,s.action,0.5);
  close(s.controller.rootMotionMatrix.slice(12,15),[4,0,2]); assert.equal(s.pose.version,2);
  assert.deepEqual(prior,[2,0,1]); assert.notEqual(prior,s.action.rootMotionDelta);
});

test("forward repeat locomotion continues across wrap seams without snapping to the clip start", () => {
  const s=setup(); step(s.controller,s.action,5.25);
  close(s.action.rootMotionDelta,[21,0,10.5]); close(s.controller.rootMotionMatrix.slice(12,15),[21,0,10.5]);
  assert.equal(s.action.time,1.25); assert.equal(s.action.completedTraversals,2);
  step(s.controller,s.action,0.75);
  close(s.action.rootMotionDelta,[3,0,1.5]); close(s.controller.rootMotionMatrix.slice(12,15),[24,0,12]);
  assert.equal(s.action.time,0);
});

test("reverse repeat, exact reverse seams and finite terminal arrival preserve signed displacement", () => {
  for (const speed of [-1,1]) {
    const s=setup({timeScale:speed,repetitions:3,clampWhenFinished:true});
    step(s.controller,s.action,2); close(s.action.rootMotionDelta,[8*speed,0,4*speed]);
    step(s.controller,s.action,100); close(s.action.rootMotionDelta,[16*speed,0,8*speed]);
    close(s.controller.rootMotionMatrix.slice(12,15),[24*speed,0,12*speed]);
    assert.equal(s.action.finished,true); assert.equal(s.action.completedTraversals,3);
    const placement=s.controller.rootMotionMatrix; step(s.controller,s.action,100);
    assert.deepEqual(s.action.rootMotionDelta,[0,0,0]); assert.deepEqual(s.controller.rootMotionMatrix,placement);
  }
});

test("ping-pong reflections cancel full round trips and preserve a partial return", () => {
  for (const speed of [-1,1]) {
    const s=setup({loop:"pingpong",timeScale:speed}); step(s.controller,s.action,4);
    assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
    step(s.controller,s.action,2.5); close(s.action.rootMotionDelta,[6*speed,0,3*speed]);
    s.action.setTimeScale(-speed); step(s.controller,s.action,0.25);
    close(s.action.rootMotionDelta,[speed,0,speed/2]);
  }
});

test("finite ping-pong never adds a repeat stride, even across many legs", () => {
  for (const repetitions of [1,2,3,100]) {
    const s=setup({loop:"pingpong",repetitions,clampWhenFinished:true});
    step(s.controller,s.action,1000); close(s.action.rootMotionDelta,repetitions%2?[8,0,4]:[0,0,0]);
    assert.equal(s.action.finished,true);
  }
});

test("STEP and CUBICSPLINE in-place clips plus displacement reconstruct authored root positions", () => {
  for (const interpolation of ["STEP","LINEAR","CUBICSPLINE"]) {
    const s=setup({loop:"once",clampWhenFinished:true},{interpolation});
    const oracle=createAnimationPlayer(definition({interpolation}));
    let last=[10,4,-3];
    for (let i=1;i<=16;i++) {
      const time=i/8; oracle.sample(time); step(s.controller,s.action,0.125);
      const expected=Array.from(oracle.translations.slice(0,3));
      close(Array.from(s.pose.worldMatrices.slice(12,15)),expected);
      close(s.action.rootMotionDelta,[expected[0]-last[0],0,expected[2]-last[2]]);
      assert.equal(s.pose.translations[0],10); assert.equal(s.pose.translations[2],-3);
      last=expected;
    }
  }
});

test("reversing rate ramps use both travel segments but net root displacement telescopes", () => {
  for (const loop of ["repeat","pingpong"]) {
    const s=setup({loop}); s.action.seek(0.5).warp(4,-4,1);
    step(s.controller,s.action,1); close(s.action.rootMotionDelta,[0,0,0]);
    assert.equal(s.action.time,0.5);
  }
  const finite=setup({loop:"once",clampWhenFinished:true}); finite.action.seek(1.5).warp(4,-4,1);
  step(finite.controller,finite.action,1); close(finite.action.rootMotionDelta,[2,0,1]);
  assert.equal(finite.action.finished,true);
});

test("scheduled starts freeze motion and late zero-host-delta updates catch up correctly", () => {
  const s=setup(); s.action.startAt(2);
  step(s.controller,s.action,1); assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
  step(s.controller,s.action,1); assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
  assert.equal(s.controller.events[0].type,"started");
  step(s.controller,s.action,0.5); close(s.action.rootMotionDelta,[2,0,1]);
  s.action.stop().startAt(0).play(); step(s.controller,s.action,0);
  close(s.action.rootMotionDelta,[10,0,5]); assert.equal(s.action.completedTraversals,1);
});

test("pause, halt and stop freeze motion; zero pose weight alone does not", () => {
  const s=setup(); s.action.pause(); step(s.controller,s.action,1);
  assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
  s.action.play().halt(1); step(s.controller,s.action,1); close(s.action.rootMotionDelta,[2,0,1]);
  step(s.controller,s.action,1); assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
  s.action.setTimeScale(1).setWeight(0); step(s.controller,s.action,0.5);
  close(s.action.rootMotionDelta,[2,0,1]); assert.equal(s.pose.translations[0],10);
  s.action.stop(); step(s.controller,s.action,1); assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
});

test("fading to stop includes only motion up to the fade deadline", () => {
  const s=setup(); s.action.fadeTo(0,0.75,{stopWhenDone:true});
  step(s.controller,s.action,100); close(s.action.rootMotionDelta,[3,0,1.5]);
  assert.equal(s.action.playing,false); assert.equal(s.action.time,0);
  step(s.controller,s.action,1); assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
});

test("seek/reset/sync/track replacement reposition without synthesizing movement", () => {
  const s=setup(); const b=s.controller.createAction(s.clip).play().seek(1.5);
  for (const op of [()=>s.action.seek(1),()=>s.action.reset(),()=>s.action.syncWith(b),
    ()=>s.action.setRootMotion(s.take.rootMotion)]) {
    op(); step(s.controller,s.action,0); assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
  }
  step(s.controller,s.action,0.25); close(s.action.rootMotionDelta,[1,0,0.5]);
});

test("crossfades expose independent unweighted motion and use only the explicitly selected action", () => {
  const s=setup(); const long=structuredClone(s.take.clip);
  long.channels.forEach(ch=>ch.times=ch.times.map(t=>t*2));
  const motion={...s.take.rootMotion,duration:4,times:[0,4]};
  const [id]=s.pose.addClips([long]); const b=s.controller.createAction(id,{rootMotion:motion});
  s.action.seek(0.5); s.controller.crossFade(s.action,b,1,{sync:true,warp:true});
  step(s.controller,s.action,0.5);
  close(s.action.rootMotionDelta,[1.75,0,0.875]); close(b.rootMotionDelta,[1.75,0,0.875]);
  close(s.controller.rootMotionMatrix.slice(12,15),[1.75,0,0.875]);
  // After crossfade, choosing the incoming action avoids guessing fade/collision policy.
  step(s.controller,b,0.5); close(b.rootMotionDelta,[1.25,0,0.625]);
  close(s.controller.rootMotionMatrix.slice(12,15),[3,0,1.5]);
});

test("collecting motion without automatic placement leaves the existing pose path unchanged", () => {
  const s=setup(); s.controller.update(0.5);
  close(s.action.rootMotionDelta,[2,0,1]); assert.equal(s.controller.rootMotionMatrix,null);
  close(Array.from(s.pose.worldMatrices.slice(12,15)),[10,4.5,-3]);
  const placement=applyAnimationRootMotion(null,[100,0,20]);
  s.controller.update(0.5,{rootMatrix:placement}); assert.equal(s.controller.rootMotionMatrix,null);
  close(Array.from(s.pose.worldMatrices.slice(12,15)),[110,5,17]);
});

test("automatic placement uses caller rotation and scale and never silently accumulates omitted placement", () => {
  const s=setup(); const m=[0,0,-2,0,0,3,0,0,4,0,0,0,10,20,30,1], before=m.slice();
  step(s.controller,s.action,0.5,m); close(s.controller.rootMotionMatrix.slice(12,15),[14,20,26]);
  assert.deepEqual(m,before);
  s.controller.update(0.5,{rootMotionAction:s.action});
  close(s.controller.rootMotionMatrix.slice(12,15),[2,0,1]);
});

test("marker overflow rolls back motion, placement, clocks and both event streams; retry commits once", () => {
  const s=setup({markers:cues},{},{maxMarkerEvents:1}); step(s.controller,s.action,0.5);
  s.action.fadeTo(0.5,4).warpTo(2,2); const before=snapshot(s);
  assert.throws(()=>step(s.controller,s.action,2),error("ANIMATION_MARKER_LIMIT")); unchanged(s,before);
  s.controller.setMarkerEventLimit(20); step(s.controller,s.action,2);
  close(s.action.rootMotionDelta,[12,0,6]); assert.equal(s.pose.version,before.version+1);
});

test("invalid placement or motion overflow cannot consume a pending schedule or successful snapshots", () => {
  const s=setup(); step(s.controller,s.action,0.25);
  s.action.startAt(1).warpTo(2,2).fadeTo(0.5,2); const before=snapshot(s);
  assert.throws(()=>step(s.controller,s.action,2,[]),error("ANIMATION_ROOT_MOTION_SHAPE")); unchanged(s,before);
  s.action.setRootMotion({...s.take.rootMotion,values:[0,0,0,Number.MAX_VALUE,0,0]});
  const changed=snapshot(s);
  assert.throws(()=>step(s.controller,s.action,10),error("ANIMATION_ROOT_MOTION_VALUE")); unchanged(s,changed);
  s.action.setRootMotion(s.take.rootMotion); step(s.controller,s.action,2);
  assert.equal(s.controller.events[0].type,"started"); assert.equal(s.action.scheduled,false);
});

test("singular real skin palettes roll back completed root motion and support a successful retry", () => {
  const s=setup({loop:"once",clampWhenFinished:true,markers:cues},{singular:true});
  step(s.controller,s.action,0.5); const before=snapshot(s);
  assert.throws(()=>step(s.controller,s.action,1.5),error("ANIMATION_SINGULAR_MESH")); unchanged(s,before);
  s.action.setWeight(0); step(s.controller,s.action,1.5);
  close(s.action.rootMotionDelta,[6,0,3]); close(s.controller.rootMotionMatrix.slice(12,15),[8,0,4]);
  assert.equal(s.action.finished,true); assert.equal(s.controller.events.at(-1).type,"finished");
});

test("detached production output leaves successful motion and placement snapshots unadvanced", () => {
  const s=setup(); step(s.controller,s.action,0.5);
  const before=snapshot(s);
  structuredClone(s.pose.worldMatrices.buffer,{transfer:[s.pose.worldMatrices.buffer]});
  assert.throws(()=>step(s.controller,s.action,1),error("ANIMATION_OUTPUT_STORAGE"));
  assert.equal(s.action.rootMotionDelta,before.motion); assert.equal(s.controller.rootMotionMatrix,before.root);
  assert.equal(s.pose.version,before.version); assert.deepEqual(state(s.action),before.state);
});

test("root input getters cannot reenter controls or expose pending displacement", () => {
  const s=setup(); const before=snapshot(s), old=s.action.rootMotionTrack;
  assert.throws(()=>s.action.setRootMotion({...s.take.rootMotion,get values(){s.action.seek(1);return [];}}),error("ANIMATION_REENTRANT"));
  assert.equal(s.action.rootMotionTrack,old); unchanged(s,before);
  assert.throws(()=>s.controller.update(1,{rootMotionAction:s.action,get rootMatrix(){s.action.stop();return null;}}),error("ANIMATION_REENTRANT"));
  unchanged(s,before);
  s.controller.update(1,{get rootMotionAction(){assert.equal(s.action.rootMotionDelta,before.motion);return s.action;}});
  close(s.action.rootMotionDelta,[4,0,2]);
});

test("track creation/replacement copies inputs and updates live budgets atomically", () => {
  const s=setup(); assert.equal(s.controller.rootMotionComponents,8);
  const old=s.action.rootMotionTrack;
  assert.throws(()=>s.action.setRootMotion({...s.take.rootMotion,duration:3}),error("ANIMATION_ROOT_MOTION_CLIP"));
  assert.equal(s.action.rootMotionTrack,old); assert.equal(s.controller.rootMotionComponents,8);
  const input=structuredClone(s.take.rootMotion); s.action.setRootMotion(input); input.values.fill(0);
  step(s.controller,s.action,0.5); close(s.action.rootMotionDelta,[2,0,1]);
  const previous=s.action.rootMotionDelta;
  s.action.setRootMotion(null); assert.equal(s.controller.rootMotionComponents,0); assert.equal(s.action.rootMotionTrack,null);
  assert.equal(s.action.rootMotionDelta,previous);
  s.controller.update(1); assert.deepEqual(s.action.rootMotionDelta,[0,0,0]);
  s.action.setRootMotion(s.take.rootMotion); assert.equal(s.controller.rootMotionComponents,8);
  s.action.dispose(); assert.equal(s.controller.rootMotionComponents,0);
  assert.throws(()=>s.action.setRootMotion(null),error("ANIMATION_ACTION_DISPOSED"));
});

test("unknown, disposed, cross-controller or unconfigured motion drivers reject before any pose publication", () => {
  const s=setup(); const b=s.controller.createAction(s.clip); const other=setup();
  const dead=s.controller.createAction(s.clip,{rootMotion:s.take.rootMotion}); dead.dispose();
  const before=snapshot(s);
  for (const driver of [{},dead,other.action]) {
    assert.throws(()=>step(s.controller,driver,1),error("ANIMATION_ACTION_DISPOSED")); unchanged(s,before);
  }
  assert.throws(()=>step(s.controller,b,1),error("ANIMATION_ROOT_MOTION_ACTION")); unchanged(s,before);
  assert.throws(()=>s.controller.createAction(s.clip,{rootMotion:{}})); assert.equal(s.controller.actionCount,2);
});

test("zero-duration root channels and zero-duration clips never invent a stride", () => {
  const def=definition(); def.clips=[{channels:[channel(0,"translation",[10,4,-3],[0])]}];
  const pose=createAnimationPlayer(def), controller=createAnimationController(pose);
  const take=extractAnimationRootMotion(pose,{node:0}), [id]=pose.addClips([take.clip]);
  const a=controller.createAction(id,{rootMotion:take.rootMotion,clampWhenFinished:true}).play();
  for (const delta of [0,1,1]) {step(controller,a,delta); assert.deepEqual(a.rootMotionDelta,[0,0,0]);}
  assert.equal(a.finished,true); a.play(); step(controller,a,1); assert.deepEqual(a.rootMotionDelta,[0,0,0]);
});

test("huge loop counts use arithmetic root motion while retaining aggregate legacy events", {timeout:1000}, () => {
  const s=setup(); step(s.controller,s.action,2000000000.25);
  close(s.action.rootMotionDelta,[8000000001,0,4000000000.5]);
  assert.equal(s.controller.events.length,1); assert.equal(s.controller.events[0].count,1000000000);
});

test("failed extraction refuses child or matrix roots without modifying a real player", () => {
  const s=setup(); const before=snapshot(s);
  assert.throws(()=>extractAnimationRootMotion(s.pose,{node:1}),error("ANIMATION_ROOT_MOTION_NODE")); unchanged(s,before);
  const def=definition(); delete def.nodes[2].translation; def.nodes[2].matrix=applyAnimationRootMotion(null,[4,0,0]);
  def.clips[0].channels=def.clips[0].channels.filter(c=>c.node!==2);
  const matrixPose=createAnimationPlayer(def);
  assert.throws(()=>extractAnimationRootMotion(matrixPose,{node:2}),error("ANIMATION_ROOT_MOTION_NODE"));
});

test("component ceiling is shared across actions, checked before values access, and released by replacement/disposal", {timeout:10000}, () => {
  const s=setup(); s.action.setRootMotion(null);
  const n=1048576, times=Float64Array.from({length:n},(_,i)=>2*i/(n-1)), values=new Float64Array(n*3);
  const input={format:"f3d-root-translation-v1",duration:2,times,values};
  const actions=Array.from({length:4},()=>s.controller.createAction(s.clip,{rootMotion:input}));
  assert.equal(s.controller.rootMotionComponents,16777216);
  let valuesRead=false;
  const tooMuch={...s.take.rootMotion,get values(){valuesRead=true;throw new Error("must preflight");}};
  assert.throws(()=>s.action.setRootMotion(tooMuch),error("ANIMATION_ROOT_MOTION_LIMIT")); assert.equal(valuesRead,false);
  assert.equal(s.controller.rootMotionComponents,16777216); assert.equal(s.action.rootMotionTrack,null);
  actions[0].setRootMotion(s.take.rootMotion); assert.equal(s.controller.rootMotionComponents,12582920);
  s.action.setRootMotion(s.take.rootMotion); assert.equal(s.controller.rootMotionComponents,12582928);
  actions[1].dispose(); assert.equal(s.controller.rootMotionComponents,8388624);
  s.controller.dispose(); assert.equal(s.controller.rootMotionComponents,0); assert.equal(s.controller.rootMotionMatrix,null);
  assert.equal(s.pose.disposed,false); assert.throws(()=>s.action.setRootMotion(null),error("ANIMATION_CONTROLLER_DISPOSED"));
});

test("large and partitioned updates agree for reverse/finite/warped locomotion and real pose bytes", () => {
  for (const loop of ["repeat","pingpong","once"]) for (const speed of [-2,-1,1,2])
    for (const reverse of [false,true]) for (const repetitions of [1,2,Infinity]) {
      const a=setup({loop,timeScale:speed,repetitions,clampWhenFinished:true});
      const b=setup({loop,timeScale:speed,repetitions,clampWhenFinished:true});
      if (reverse) {a.action.warp(speed,-speed,2);b.action.warp(speed,-speed,2);}
      step(a.controller,a.action,4); const sum=[0,0,0];
      for(let i=0;i<16;i++) {step(b.controller,b.action,0.25);b.action.rootMotionDelta.forEach((v,j)=>sum[j]+=v);}
      close(a.action.rootMotionDelta,sum); assert.deepEqual(a.controller.rootMotionMatrix,b.controller.rootMotionMatrix);
      assert.deepEqual(bytes(a.pose),bytes(b.pose),JSON.stringify({loop,speed,reverse,repetitions}));
    }
});
