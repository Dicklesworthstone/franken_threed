import test from "node:test";
import assert from "node:assert/strict";
import { createAnimationPlayer } from "./animation_runtime.mjs";
import { createAnimationController } from "./animation_controller.mjs";

const cues = [{ name: "start", time: 0 }, { name: "left", time: 0.25 },
  { name: "right", time: 0.75 }, { name: "end", time: 1 }];
const error = (code) => (e) => e?.code === code;
const summary = (controller) => controller.markerEvents.map(({ name, direction, traversal }) => [name, direction, traversal]);
function clip(duration = 1, singular = false) {
  const channel = (node, path, a, b) => ({ node, path,
    times: duration ? [0, duration] : [0], values: duration ? [...a, ...b] : a });
  return { name: "motion", channels: [
    channel(0, "translation", [0, 0, 0], [8, 2, -1]),
    channel(1, "rotation", [0, 0, 0, 1], [0, 0, Math.SQRT1_2, Math.SQRT1_2]),
    channel(0, "weights", [0.2, 0.4], [0.8, -0.2]),
    channel(2, "scale", [1, 1, 1], singular ? [0, 0, 0] : [2, 1, 1]),
  ] };
}
function definition(clips = [clip()]) {
  return { format: "f3d-animation-v1", nodes: [
    { weights: [0.2, 0.4] }, { parent: 0, translation: [0, 1, 0] },
    { translation: [4, 0, 0] }, { translation: [-4, 0, 0] },
  ], skins: [{ joints: [0, 1] }], instances: [{ node: 2, skin: 0 }, { node: 3, skin: 0 }], clips };
}
function setup(options = {}, clips) {
  const pose = createAnimationPlayer(definition(clips));
  return { pose, controller: createAnimationController(pose, options) };
}
const fields = ["translations", "rotations", "scales", "morphWeights", "worldMatrices", "jointMatrices"];
const poseBytes = (pose) => fields.map((f) => Buffer.from(pose[f].buffer).toString("hex"));
const actionState = (action) => [action.time, action.weight, action.timeScale, action.warping,
  action.playing, action.paused, action.finished, action.completedTraversals, action.startTime];

test("cues publish with real skinned/morph poses, and do not replace legacy events", () => {
  const { pose, controller } = setup();
  const arrays = fields.map((f) => pose[f]);
  const a = controller.createAction(0, { markers: cues, loop: "once", clampWhenFinished: true }).play();
  assert.deepEqual(controller.markerEvents, []);
  controller.update(0.25);
  assert.deepEqual(summary(controller), [["left", 1, 0]]);
  assert.equal(pose.translations[0], 2);
  assert.ok(Object.isFrozen(controller.markerEvents)); assert.ok(Object.isFrozen(controller.markerEvents[0]));
  assert.equal(controller.markerEvents[0].action, a);
  const retained = controller.markerEvents;
  controller.update(0.25); assert.deepEqual(controller.markerEvents, []);
  assert.equal(retained[0].name, "left");
  controller.update(0.5);
  assert.deepEqual(summary(controller), [["right", 1, 0], ["end", 1, 0]]);
  assert.deepEqual(controller.events, [{ type: "finished", action: a, direction: 1 }]);
  assert.equal(pose.translations[0], 8);
  for (let i = 0; i < fields.length; i++) assert.equal(pose[fields[i]], arrays[i]);
  assert.notDeepEqual(pose.jointMatrices.slice(0, 32), pose.jointMatrices.slice(32));
  controller.update(100); assert.deepEqual(controller.markerEvents, []);
});

test("reverse repeating actions emit endpoint cues and retain legacy loop direction", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: cues, timeScale: -1 }).play();
  controller.update(1);
  assert.deepEqual(summary(controller), [["right", -1, 0], ["left", -1, 0], ["start", -1, 0], ["end", -1, 1]]);
  assert.deepEqual(controller.events, [{ type: "loop", action: a, count: 1, direction: -1 }]);
  controller.update(0); assert.deepEqual(controller.markerEvents, []);
});

test("large ping-pong updates report each crossed cue in its local direction", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: cues, loop: "pingpong" }).play();
  controller.update(2.25);
  assert.deepEqual(summary(controller), [["left", 1, 0], ["right", 1, 0], ["end", 1, 0],
    ["right", -1, 1], ["left", -1, 1], ["start", -1, 1], ["left", 1, 2]]);
  assert.deepEqual(controller.events, [{ type: "loop", action: a, count: 2, direction: 1 }]);
});

test("finite ping-pong completion emits no departure after the terminal endpoint", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: cues, loop: "pingpong", repetitions: 2 }).play();
  controller.update(100);
  assert.deepEqual(summary(controller).slice(-3), [["right", -1, 1], ["left", -1, 1], ["start", -1, 1]]);
  assert.equal(controller.markerEvents.length, 6);
  assert.equal(a.finished, true); assert.equal(a.time, 0);
  assert.deepEqual(controller.events.map((e) => [e.type, e.count, e.direction]),
    [["loop", 1, 1], ["finished", undefined, -1]]);
});

test("reversing warps retain outward and returning cue crossings, including the turning point", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: [
    { name: "a", time: 0.25 }, { name: "b", time: 0.5 }, { name: "c", time: 0.75 },
  ] }).play().seek(0.25).warp(2, -2, 1);
  controller.update(1);
  assert.deepEqual(summary(controller), [["b", 1, 0], ["c", 1, 0], ["b", -1, 0], ["a", -1, 0]]);
  assert.equal(a.time, 0.25); assert.deepEqual(controller.events, []);
});

test("reversing warps through repeat boundaries preserve the two legacy loop summaries", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: cues }).play().seek(0.75).warp(4, -4, 1);
  controller.update(1);
  assert.deepEqual(summary(controller), [["end", 1, 0], ["start", 1, 1], ["left", 1, 1], ["right", 1, 1],
    ["left", -1, 1], ["start", -1, 1], ["end", -1, 2], ["right", -1, 2]]);
  assert.equal(a.time, 0.75);
  assert.deepEqual(controller.events.map(({ type, count, direction }) => [type, count, direction]),
    [["loop", 1, 1], ["loop", 1, -1]]);
});

test("scheduled starts freeze cues, exact starts do not synthesize zero markers, late starts catch up", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: cues }).startAt(2).play();
  controller.update(1); assert.deepEqual(controller.markerEvents, []);
  controller.update(1); assert.deepEqual(controller.markerEvents, []);
  assert.deepEqual(controller.events, [{ type: "started", action: a, time: 2 }]);
  controller.update(2.25);
  assert.equal(controller.markerEvents.length, 9); assert.equal(a.time, 0.25);
  a.stop().startAt(0).play();
  controller.update(0); // nonzero catch-up despite zero host delta
  assert.equal(controller.markerEvents.length, 17);
  assert.equal(controller.markerEvents.at(-1).name, "left");
  assert.equal(a.completedTraversals, 4);
});

test("pause, zero speed and stopped actions emit nothing; zero weight alone does not stop the clock", () => {
  const { controller, pose } = setup();
  const a = controller.createAction(0, { markers: cues }).play();
  a.pause(); controller.update(1); assert.deepEqual(controller.markerEvents, []); assert.equal(a.time, 0);
  a.play().setTimeScale(0); controller.update(1); assert.deepEqual(controller.markerEvents, []);
  a.setTimeScale(1).setWeight(0); controller.update(0.25);
  assert.deepEqual(summary(controller), [["left", 1, 0]]); assert.equal(pose.translations[0], 0);
  a.stop(); controller.update(1); assert.deepEqual(controller.markerEvents, []);
});

test("seek, reset and sync are repositioning, not synthetic traversals", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: cues }).play().seek(0.75);
  controller.update(0); assert.deepEqual(controller.markerEvents, []);
  a.setTimeScale(-1); controller.update(0.5);
  assert.deepEqual(summary(controller), [["left", -1, 0]]);
  const b = controller.createAction(0).play().seek(0.75);
  a.syncWith(b); controller.update(0); assert.deepEqual(controller.markerEvents, []);
  a.reset(); controller.update(0); assert.deepEqual(controller.markerEvents, []);
});

test("fade-to-stop emits only cues reached before the fade deadline, not the full host delta", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: [...cues, { name: "cut", time: 0.5 }] }).play();
  a.fadeTo(0, 0.5, { stopWhenDone: true });
  controller.update(2);
  assert.deepEqual(summary(controller), [["left", 1, 0], ["cut", 1, 0]]);
  assert.equal(a.playing, false); assert.equal(a.time, 0);
});

test("synchronized warped crossfades emit source and destination cues from their own clocks", () => {
  const { controller } = setup({}, [clip(1), clip(2)]);
  const a = controller.createAction(0, { markers: [{ name: "a", time: 0.5 }] }).play().seek(0.25);
  const b = controller.createAction(1, { markers: [{ name: "b", time: 1 }] });
  controller.crossFade(a, b, 1, { sync: true, warp: true });
  controller.update(0.5);
  assert.deepEqual(summary(controller), [["a", 1, 0], ["b", 1, 0]]);
  assert.equal(controller.markerEvents[0].action, a); assert.equal(controller.markerEvents[1].action, b);
  assert.equal(a.time, 0.6875); assert.equal(b.time, 1.375);
});

test("action creation order, not global cue time, determines multi-action event blocks", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: [{ name: "later", time: 0.75 }] }).play();
  const b = controller.createAction(0, { markers: [{ name: "earlier", time: 0.25 }] }).play();
  controller.update(1);
  assert.deepEqual(controller.markerEvents.map((e) => [e.name, e.action]), [["later", a], ["earlier", b]]);
});

test("aggregate event overflow rolls back every action, all pose bytes and both event snapshots", () => {
  const { pose, controller } = setup({ maxMarkerEvents: 1 });
  const a = controller.createAction(0, { markers: [{ name: "a", time: 0.25 }] }).play();
  const b = controller.createAction(0, { markers: [{ name: "b", time: 0.75 }] }).play();
  controller.update(0.25);
  const previous = controller.markerEvents, legacy = controller.events, bytes = poseBytes(pose), version = pose.version;
  const states = [actionState(a), actionState(b)];
  assert.throws(() => controller.update(1), error("ANIMATION_MARKER_LIMIT"));
  assert.equal(controller.time, 0.25); assert.equal(pose.version, version);
  assert.deepEqual(poseBytes(pose), bytes); assert.deepEqual([actionState(a), actionState(b)], states);
  assert.equal(controller.markerEvents, previous); assert.equal(controller.events, legacy);
  assert.equal(controller.setMarkerEventLimit(2), controller);
  controller.update(1);
  assert.deepEqual(summary(controller), [["a", 1, 1], ["b", 1, 0]]);
  assert.equal(controller.time, 1.25);
});

test("overflow cannot half-consume a pending start, fade or rate warp", () => {
  const { pose, controller } = setup({ maxMarkerEvents: 1 });
  const a = controller.createAction(0, { markers: cues }).startAt(0.5).play().fadeTo(0.5, 4).warpTo(2, 2);
  const state = actionState(a), bytes = poseBytes(pose);
  assert.throws(() => controller.update(2), error("ANIMATION_MARKER_LIMIT"));
  assert.deepEqual(actionState(a), state); assert.equal(controller.time, 0); assert.equal(pose.version, 0);
  assert.deepEqual(poseBytes(pose), bytes); assert.deepEqual(controller.markerEvents, []);
  controller.setMarkerEventLimit(32).update(2);
  assert.equal(controller.events[0].type, "started"); assert.equal(controller.events[0].time, 0.5);
  const reference = setup({ maxMarkerEvents: 32 });
  reference.controller.createAction(0, { markers: cues }).startAt(0.5).play().fadeTo(0.5, 4).warpTo(2, 2);
  reference.controller.update(2);
  assert.deepEqual(poseBytes(pose), poseBytes(reference.pose));
  assert.deepEqual(summary(controller), summary(reference.controller));
});

test("singular skinned-pose failure publishes no finish or cue and can be retried", () => {
  const { pose, controller } = setup({}, [clip(1, true)]);
  const a = controller.createAction(0, { markers: cues, loop: "once", clampWhenFinished: true }).play();
  controller.update(0.25);
  const previous = controller.markerEvents, legacy = controller.events, bytes = poseBytes(pose), version = pose.version;
  const state = actionState(a);
  assert.throws(() => controller.update(0.75), error("ANIMATION_SINGULAR_MESH"));
  assert.equal(controller.markerEvents, previous); assert.equal(controller.events, legacy);
  assert.deepEqual(actionState(a), state); assert.equal(pose.version, version); assert.deepEqual(poseBytes(pose), bytes);
  a.setWeight(0); controller.update(0.75); // rest pose is invertible; finish/cues can now commit
  assert.deepEqual(summary(controller), [["right", 1, 0], ["end", 1, 0]]);
  assert.equal(controller.events[0].type, "finished");
});

test("detached production output rejects queued cues without advancing any action", () => {
  const { pose, controller } = setup();
  const a = controller.createAction(0, { markers: cues }).play();
  controller.update(0.25);
  const previous = controller.markerEvents, state = actionState(a);
  structuredClone(pose.worldMatrices.buffer, { transfer: [pose.worldMatrices.buffer] });
  assert.throws(() => controller.update(0.5), error("ANIMATION_OUTPUT_STORAGE"));
  assert.equal(controller.markerEvents, previous); assert.deepEqual(actionState(a), state);
  assert.equal(controller.time, 0.25);
});

test("marker replacement is copied, frozen, atomic and does not move a playhead", () => {
  const { controller } = setup();
  const a = controller.createAction(0, { markers: cues }).play();
  const old = a.markers;
  const input = [{ name: "new", time: 0.5 }];
  assert.equal(a.setMarkers(input), a); input[0].name = "changed";
  assert.equal(a.markers[0].name, "new"); assert.equal(old.length, 4); assert.equal(a.time, 0);
  const current = a.markers;
  assert.throws(() => a.setMarkers([{ name: "valid", time: 0 }, { name: "bad", time: 2 }]), error("ANIMATION_MARKER_VALUE"));
  assert.equal(a.markers, current); assert.ok(Object.isFrozen(current));
  controller.update(0.5); assert.equal(controller.markerEvents[0].name, "new");
  a.setMarkers([]); controller.update(1); assert.deepEqual(controller.markerEvents, []);
});

test("failed marker admission consumes no action slot, including after live clip installation", () => {
  const { pose, controller } = setup({}, []);
  const [id] = pose.addClips([clip()]);
  assert.throws(() => controller.createAction(id, { markers: [{ name: "bad", time: 2 }] }), error("ANIMATION_MARKER_VALUE"));
  assert.equal(controller.actionCount, 0);
  const a = controller.createAction(id, { markers: cues }).play();
  pose.addClips([clip(2)]); controller.update(0.25);
  assert.equal(controller.markerEvents[0].action, a); assert.equal(a.clip, id);
  assert.equal(pose.clipVersion, 2); assert.equal(pose.translations[0], 2);
});

test("marker input getters cannot reenter configuration, and frame getters cannot replace pending tracks", () => {
  const { pose, controller } = setup();
  const a = controller.createAction(0, { markers: cues }).play();
  const markers = a.markers;
  assert.throws(() => a.setMarkers([{ get name() { a.seek(0.8); return "bad"; }, time: 0 }]), error("ANIMATION_REENTRANT"));
  assert.equal(a.markers, markers); assert.equal(a.time, 0);
  assert.throws(() => controller.update(0.25, { get rootMatrix() { a.setMarkers([]); return null; } }), error("ANIMATION_REENTRANT"));
  assert.equal(a.time, 0); assert.equal(pose.version, 0); assert.deepEqual(controller.markerEvents, []);
  controller.update(0.25); assert.deepEqual(summary(controller), [["left", 1, 0]]);
});

test("zero-duration actions fire their one completion cue, including after replay", () => {
  for (const timeScale of [-1, 1]) {
    const { controller } = setup({}, [clip(0)]);
    const a = controller.createAction(0, { markers: [{ name: "instant", time: 0 }], timeScale, clampWhenFinished: true }).play();
    controller.update(0); assert.deepEqual(controller.markerEvents, []);
    controller.update(1); assert.deepEqual(summary(controller), [["instant", timeScale, 0]]);
    controller.update(1); assert.deepEqual(controller.markerEvents, []);
    a.play(); controller.update(1); assert.equal(controller.markerEvents.length, 1);
  }
});

test("very small and very large clip clocks preserve cue boundaries", () => {
  for (const duration of [1e-300, 1e300]) {
    const { controller } = setup({}, [clip(duration)]);
    controller.createAction(0, { loop: "once", markers: [{ name: "middle", time: duration / 2 }] }).play();
    controller.update(duration / 2); assert.equal(controller.markerEvents[0].name, "middle");
    controller.update(duration / 2); assert.deepEqual(controller.markerEvents, []);
  }
});

test("event limits can change without losing prior snapshots; lifetime guards remain effective", () => {
  const { controller, pose } = setup();
  const a = controller.createAction(0, { markers: cues }).play();
  assert.equal(controller.maxMarkerEvents, 4096);
  controller.update(1); const previous = controller.markerEvents;
  controller.setMarkerEventLimit(1); assert.equal(controller.markerEvents, previous);
  for (const value of [0, 65537, 1.1, NaN, Infinity, "2", null]) {
    assert.throws(() => controller.setMarkerEventLimit(value), error("ANIMATION_MARKER_LIMIT"));
    assert.equal(controller.maxMarkerEvents, 1);
  }
  controller.dispose(); assert.deepEqual(controller.markerEvents, []); assert.equal(pose.disposed, false);
  assert.throws(() => a.setMarkers([]), error("ANIMATION_CONTROLLER_DISPOSED"));
  assert.throws(() => controller.setMarkerEventLimit(2), error("ANIMATION_CONTROLLER_DISPOSED"));
  assert.equal(previous.length, 4);
});

test("controller marker options are bounded and reject unknown configuration", () => {
  const pose = createAnimationPlayer(definition());
  for (const input of [null, [], 1, { typo: true }, { maxMarkerEvents: 0 }, { maxMarkerEvents: Infinity }, { maxMarkerEvents: null }])
    assert.throws(() => createAnimationController(pose, input));
  assert.equal(createAnimationController(pose, { maxMarkerEvents: 65536 }).maxMarkerEvents, 65536);
});

// Dyadic times isolate event partitioning from unavoidable floating arithmetic
// differences. Warps use the real signed integrator and test both sides of turns.
test("large updates and partitioned updates produce the same ordered cue path", () => {
  for (const loop of ["repeat", "pingpong", "once"]) for (const speed of [-2, -1, 1, 2])
    for (const reverse of [false, true]) for (const repetitions of [1, 2, Infinity]) {
      const a = setup(), b = setup();
      for (const { controller } of [a, b]) {
        const action = controller.createAction(0, { markers: cues, loop, repetitions, timeScale: speed,
          clampWhenFinished: true }).play();
        if (reverse) action.warp(speed, -speed, 2);
      }
      a.controller.update(4);
      const parts = [];
      for (let i = 0; i < 16; i++) { b.controller.update(0.25); parts.push(...summary(b.controller)); }
      assert.deepEqual(summary(a.controller), parts, JSON.stringify({ loop, speed, reverse, repetitions }));
      assert.deepEqual(poseBytes(a.pose), poseBytes(b.pose));
    }
});
