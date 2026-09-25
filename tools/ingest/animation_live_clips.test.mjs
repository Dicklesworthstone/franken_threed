import test from "node:test";
import assert from "node:assert/strict";
import {createAnimationPlayer, AnimationPoseError} from "./animation_runtime.mjs";
import {createAnimationController} from "./animation_controller.mjs";
import {createAnimationRecorder} from "./animation_recorder.mjs";

const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const fields = ["translations", "rotations", "scales", "morphWeights", "worldMatrices", "jointMatrices"];
function motion(duration = 1, end = 4) {
  return {name: "motion", channels: [
    {node: 0, path: "translation", times: [0,duration], values: [0,0,0,end,0,0]},
  ]};
}
function rig(clips = [motion()]) {
  return {format: "f3d-animation-v1", nodes: [
    {}, {parent: 0, translation: [0,1,0]},
    {translation: [2,0,0], weights: Array(7).fill(0)}, {translation: [-3,0,0]},
  ], skins: [{joints: [0,1], inverseBindMatrices: [...identity(), ...identity()]}],
  instances: [{node: 2, skin: 0}, {node: 3, skin: 0}], clips};
}
function setup(t, clips) {
  const pose = createAnimationPlayer(rig(clips)), controller = createAnimationController(pose);
  t.after(() => {controller.dispose(); pose.dispose();});
  return {pose, controller};
}
function actionState(action) {
  return Object.fromEntries(["clip","duration","time","weight","timeScale","warping","effectiveTimeScale",
    "scheduled","startTime","playing","paused","finished","completedTraversals","loop","repetitions",
    "effectiveWeight","disposed"].map(k => [k, action[k]]));
}
function controllerState(controller) {
  return {time: controller.time, actionCount: controller.actionCount,
    events: controller.events.map(({action, ...event}) => ({...event, clip: action.clip}))};
}
function equalPose(a, b) {
  for (const field of fields) assert.deepEqual(a[field], b[field], field);
}
function nearPose(a, b) {
  for (const field of fields) for (let i = 0; i < a[field].length; i++)
    assert.ok(Math.abs(a[field][i] - b[field][i]) < 3e-6, `${field}[${i}]`);
}
const error = code => ({code: "ANIMATION_" + code});

test("an existing clipless controller plays its player's first installed clip", t => {
  const {pose, controller} = setup(t, []), buffers = fields.map(f => pose[f]);
  controller.update(.2);
  const version = pose.version, time = controller.time;
  const [index] = pose.addClips([motion()]);
  assert.equal(controller.time, time); assert.equal(pose.version, version);
  assert.equal(controller.actionCount, 0);
  const action = controller.createAction(index).play();
  controller.update(.25);
  assert.equal(pose.translations[0], 1); assert.equal(action.time, .25);
  for (let i = 0; i < fields.length; i++) assert.equal(pose[fields[i]], buffers[i]);
});

for (const loop of ["repeat", "pingpong", "once"]) for (const timeScale of [-1, 1])
  for (const clampWhenFinished of [false, true])
    test(`late clips leave ${loop}/${timeScale}/${clampWhenFinished} clocks, loops and finish state intact`, t => {
      const live = setup(t), reference = setup(t, [motion(), motion(2,8)]);
      const options = {loop, timeScale, clampWhenFinished, repetitions: 3};
      const a = live.controller.createAction(0, options).play();
      const b = reference.controller.createAction(0, options).play();
      live.controller.update(.35); reference.controller.update(.35);
      a.warpTo(-timeScale, .8); b.warpTo(-timeScale, .8);
      a.fadeTo(.25, .6); b.fadeTo(.25, .6);
      const before = actionState(a), clock = controllerState(live.controller), arrays = fields.map(f => live.pose[f].slice());
      assert.deepEqual(live.pose.addClips([motion(2,8)]), [1]);
      assert.deepEqual(actionState(a), before); assert.deepEqual(controllerState(live.controller), clock);
      for (let i = 0; i < fields.length; i++) assert.deepEqual(live.pose[fields[i]], arrays[i]);
      for (const delta of [.1,.2,.7,.3,1.5,3,.1]) {
        live.controller.update(delta); reference.controller.update(delta);
        equalPose(live.pose, reference.pose);
        assert.deepEqual(actionState(a), actionState(b));
        assert.deepEqual(controllerState(live.controller), controllerState(reference.controller));
      }
    });

for (const sync of [false, true]) for (const warp of [false, true])
  test(`crossfade into an installed clip retains normal controls (sync=${sync}, warp=${warp})`, t => {
    const live = setup(t), reference = setup(t, [motion(), motion(2,12)]);
    const a = live.controller.createAction(0).play(), b = reference.controller.createAction(0).play();
    live.controller.update(.4); reference.controller.update(.4);
    const [index] = live.pose.addClips([motion(2,12)]);
    const next = live.controller.createAction(index), refNext = reference.controller.createAction(1);
    live.controller.crossFade(a, next, .5, {sync, warp});
    reference.controller.crossFade(b, refNext, .5, {sync, warp});
    for (const delta of [0,.125,.125,.25,.125,2.2]) {
      live.controller.update(delta); reference.controller.update(delta);
      equalPose(live.pose, reference.pose);
      assert.deepEqual(actionState(a), actionState(b)); assert.deepEqual(actionState(next), actionState(refNext));
      assert.deepEqual(controllerState(live.controller), controllerState(reference.controller));
    }
  });

test("paused actions and scheduled starts are not disturbed by successful or failed installation", t => {
  const {pose, controller} = setup(t), a = controller.createAction(0).play().pause();
  const scheduled = controller.createAction(0).startAt(2).play();
  controller.update(.5);
  const saved = [actionState(a), actionState(scheduled), controllerState(controller)];
  pose.addClips([motion(2,8)]);
  assert.throws(() => pose.addClips([motion(3), {channels: [null]}]), AnimationPoseError);
  assert.deepEqual([actionState(a), actionState(scheduled), controllerState(controller)], saved);
  controller.update(1.75);
  assert.equal(a.time, 0); assert.equal(scheduled.time, .25);
  assert.deepEqual(controller.events.map(e => [e.type,e.time]), [["started",2]]);
  const [index] = pose.addClips([motion(3,12)]);
  const newAction = controller.createAction(index).startAt(3).play();
  controller.update(1);
  assert.equal(newAction.time, .25); assert.equal(newAction.duration, 3);
});

test("a newly installed failing action rolls back the existing controller and can be stopped", t => {
  const {pose, controller} = setup(t), first = controller.createAction(0).play();
  controller.update(.25);
  const [index] = pose.addClips([{channels: [{node: 2, path: "scale", times: [0], values: [0,0,0]}]}]);
  const broken = controller.createAction(index, {clampWhenFinished: true}).play();
  const before = controllerState(controller), actionBefore = actionState(first), poseBefore = pose.snapshotLocalPose();
  assert.throws(() => controller.update(.1), error("SINGULAR_MESH"));
  assert.deepEqual(controllerState(controller), before); assert.deepEqual(actionState(first), actionBefore);
  assert.deepEqual(pose.snapshotLocalPose(), poseBefore); assert.equal(broken.finished, false);
  broken.stop(); controller.update(.25); assert.equal(first.time, .5); assert.equal(pose.translations[0], 2);
});

for (const interpolation of ["LINEAR", "STEP"]) test(`record/install/play/snapshot/reinstall ${interpolation} motion on real controllers`, t => {
  const {pose, controller} = setup(t), oldAction = controller.createAction(0).play();
  const recorder = createAnimationRecorder(pose, {interpolation, tracks: [
    {node: 0,path: "translation"}, {node: 1,path: "rotation"}, {node: 2,path: "weights"},
  ]});
  t.after(() => recorder.dispose());
  const expected = [];
  for (let i=0; i<=4; i++) {
    controller.update(i ? .25 : 0);
    const angle = i * .3;
    pose.edit([{node: 1,rotation: [0,0,Math.sin(angle/2),Math.cos(angle/2)]},
      {node: 2,weights: Array.from({length: 7}, (_, j) => (i+j)/16)}]);
    recorder.capture(10+i/4);
    expected.push(Object.fromEntries(fields.map(f => [f,pose[f].slice()])));
  }
  const take = recorder.finish(), before = pose.snapshotLocalPose(), time = controller.time;
  const [index] = pose.addClips([take.clip]);
  assert.deepEqual(pose.snapshotLocalPose(), before); assert.equal(controller.time, time);
  // Discard caller-owned capture data: persistence now uses the installed copy.
  take.clip.channels.length = 0;
  const saved = pose.snapshotClips([index]);
  assert.deepEqual(saved.indices, [index]); assert.equal(saved.clipVersion, 1);
  const reloaded = setup(t, []);
  const [reloadedIndex] = reloaded.pose.addClips(saved.clips);
  oldAction.stop(); controller.createAction(index, {loop: "once",clampWhenFinished: true}).play();
  reloaded.controller.createAction(reloadedIndex, {loop: "once",clampWhenFinished: true}).play();
  const buffers = fields.map(f => pose[f]);
  for (let i=0; i<=4; i++) {
    controller.update(i ? .25 : 0); reloaded.controller.update(i ? .25 : 0);
    nearPose(pose, expected[i]); equalPose(pose, reloaded.pose);
    assert.notDeepEqual(pose.jointMatrices.slice(0,32),pose.jointMatrices.slice(32,64));
    for(let j=0;j<fields.length;j++) assert.equal(pose[fields[j]],buffers[j]);
  }
});

test("a running recorder survives clip-registry changes without mistaking them for pose edits", t => {
  const {pose} = setup(t), recorder = createAnimationRecorder(pose, {tracks: [{node: 0,path: "translation"}]});
  t.after(() => recorder.dispose()); pose.sample(.5); const version = pose.version;
  recorder.capture(0); pose.addClips([motion(2,10)]); pose.snapshotClips(); recorder.capture(1);
  assert.equal(pose.version,version); const result = recorder.finish();
  assert.deepEqual(result.clip.channels[0].values,[2,0,0,2,0,0]);
});

test("snapshots select and reorder owned clips without leaking private bindings or buffers", t => {
  const {pose} = setup(t); pose.addClips([motion(2,8),motion(3,12)]);
  pose.sample(.25); const before = pose.snapshotLocalPose(), version = pose.version, meta = pose.clips;
  const input = [2,0], saved = pose.snapshotClips(input);
  input[0] = 1;
  assert.equal(saved.format,"f3d-animation-clips-v1"); assert.equal(saved.nodeCount,4);
  assert.deepEqual(saved.indices,[2,0]); assert.equal(saved.clipVersion,1);
  assert.deepEqual(saved.clips,[motion(3,12),motion()].map(c => ({...c,channels:c.channels.map(ch=>({...ch,interpolation:"LINEAR"}))})));
  assert.ok(Object.isFrozen(saved)); assert.ok(Object.isFrozen(saved.indices));
  assert.equal(pose.version,version); assert.equal(pose.clips,meta); assert.deepEqual(pose.snapshotLocalPose(),before);
  saved.clips[0].channels[0].values.fill(999); saved.clips[1].channels.length=0;
  pose.sample(1,{clip:2}); assert.equal(pose.translations[0],4);
  const old = pose.snapshotClips(); pose.addClips([motion(4,16)]);
  assert.equal(old.clips.length,3); assert.equal(old.clipVersion,1); assert.equal(pose.snapshotClips().clips.length,4);
});

test("snapshots retain exact Float64 keys, signed zero, cubic tangents and quantized rotation admission", t => {
  const t1=.12345678912345678;
  const authored = {name:"precision",channels:[
    {node:0,path:"translation",interpolation:"CUBICSPLINE",times:[0,t1],values:[-0,.1,.2, -0,.2,.3, .7,.8,.9, 1,2,3, 4,5,6, 7,8,9]},
    {node:1,path:"rotation",interpolation:"LINEAR",quantizedRotation:true,times:[0,1],values:[0,0,0,1.005, 0,0,.603,.804]},
  ]};
  const {pose}=setup(t,[authored]); const first=pose.snapshotClips();
  assert.deepEqual(first.clips,[authored]); assert.equal(first.clips[0].channels[0].times[1],t1);
  assert.ok(Object.is(first.clips[0].channels[0].values[0],-0));
  const replay=setup(t,first.clips); pose.sample(.4); replay.pose.sample(.4); equalPose(pose,replay.pose);
  const [index]=replay.pose.addClips(first.clips); replay.pose.sample(.4,{clip:index}); equalPose(pose,replay.pose);
});

test("public output poisoning and source edits cannot alter installed keyframe snapshots", t => {
  const input=motion(), {pose}=setup(t,[input]); const expected=pose.snapshotClips();
  input.channels[0].values.fill(888); for(const f of fields) pose[f].fill(777);
  assert.deepEqual(pose.snapshotClips(),expected);
});

test("every snapshot owns its arrays, which remain usable after player disposal", t => {
  const {pose}=setup(t); pose.addClips([motion()]);
  const a=pose.snapshotClips(), b=pose.snapshotClips(); pose.dispose();
  a.clips[0].channels[0].times[0]=999;
  a.clips[0].channels[0].values[0]=999;
  assert.equal(a.clips[1].channels[0].times[0],0); assert.equal(b.clips[0].channels[0].times[0],0);
  const replay=setup(t,b.clips); replay.pose.sample(.5); assert.equal(replay.pose.translations[0],2);
});

test("clipless and explicit empty snapshots are legal and keep both revisions unchanged", t => {
  const {pose}=setup(t,[]); const a=pose.snapshotClips();
  assert.deepEqual(a.indices,[]); assert.deepEqual(a.clips,[]); assert.equal(a.clipVersion,0);
  pose.addClips([motion()]); const version=pose.version;
  assert.deepEqual(pose.snapshotClips([]).clips,[]); assert.equal(pose.version,version); assert.equal(pose.clipVersion,1);
});

for(const selection of [null,{},"0",new Uint32Array([0]),[-1],[.5],[1],["0"],[NaN],[0,0],Array(1)])
  test(`invalid snapshot selection ${String(selection)} is recoverable`, t => {
    const {pose}=setup(t); const before=pose.snapshotClips(), state=pose.snapshotLocalPose();
    assert.throws(()=>pose.snapshotClips(selection),AnimationPoseError);
    assert.deepEqual(pose.snapshotClips(),before); assert.deepEqual(pose.snapshotLocalPose(),state);
  });

for(const operation of [p=>p.addClips([motion()]),p=>p.snapshotClips(),p=>p.sample(0),p=>p.dispose()])
  test("snapshot selectors cannot reenter pose publication, clip installation or disposal", t => {
    const {pose}=setup(t), selector=[]; selector.length=1;
    Object.defineProperty(selector,0,{get(){operation(pose);return 0;}});
    const before=pose.snapshotClips();
    assert.throws(()=>pose.snapshotClips(selector),error("REENTRANT"));
    assert.deepEqual(pose.snapshotClips(),before);
  });

test("installation getters cannot observe partially installed clips through snapshots", t => {
  const {pose}=setup(t); const before=pose.snapshotClips();
  assert.throws(()=>pose.addClips([{get channels(){pose.snapshotClips();return [];}}]),error("REENTRANT"));
  assert.deepEqual(pose.snapshotClips(),before);
});

test("detaching output during selector reads prevents a snapshot from being published", t => {
  const {pose}=setup(t), selector=[]; selector.length=1;
  Object.defineProperty(selector,0,{get(){structuredClone(pose.worldMatrices.buffer,{transfer:[pose.worldMatrices.buffer]});return 0;}});
  assert.throws(()=>pose.snapshotClips(selector),error("OUTPUT_STORAGE"));
  assert.equal(pose.clipVersion,0); assert.equal(pose.version,0);
});

test("disposed players reject clip snapshots", () => {
  const pose=createAnimationPlayer(rig());pose.dispose();
  assert.throws(()=>pose.snapshotClips(),error("DISPOSED"));
});
