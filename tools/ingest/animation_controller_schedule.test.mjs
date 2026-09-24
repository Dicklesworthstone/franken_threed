import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationController } from "./animation_controller.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

function make(duration = 20) {
  return createAnimationController(createAnimationPlayer({
    format: "f3d-animation-v1", nodes: [{}],
    clips: [{ channels: [{ node: 0, path: "translation", times: [0, duration], values: [0, 0, 0, duration, 0, 0] }] }],
  }));
}
const close = (a, b, epsilon = 1e-10) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
const state = (c, a) => ({
  time: c.time, version: c.pose.version, events: c.events, pose: [...c.pose.worldMatrices],
  actionTime: a.time, weight: a.weight, speed: a.timeScale, playing: a.playing,
  paused: a.paused, finished: a.finished, startTime: a.startTime,
  scheduled: a.scheduled, warping: a.warping, traversals: a.completedTraversals,
  effectiveWeight: a.effectiveWeight, effectiveTimeScale: a.effectiveTimeScale,
});

test("scheduled actions freeze pose contribution, fades and warps until the exact deadline", () => {
  const c = make(), a = c.createAction(0).startAt(2).play().warpTo(3, 2).fadeTo(0.5, 2);
  assert.equal(a.playing, true);
  assert.equal(a.scheduled, true);
  assert.equal(a.startTime, 2);
  assert.equal(a.effectiveWeight, 0);
  assert.equal(a.effectiveTimeScale, 0);
  c.update(1);
  close(a.time, 0);
  close(a.timeScale, 1);
  close(a.weight, 1);
  close(c.pose.translations[0], 0);
  assert.equal(c.events.length, 0);
  c.update(1);
  assert.equal(a.scheduled, false);
  assert.equal(a.startTime, null);
  close(a.time, 0);
  close(a.weight, 1);
  assert.deepEqual(c.events, [{ type: "started", action: a, time: 2 }]);
  assert.ok(Object.isFrozen(c.events[0]));
  c.update(1);
  close(a.time, 1.5);
  close(a.timeScale, 2);
  close(a.weight, 0.75);
  close(c.pose.translations[0], 1.125);
  assert.deepEqual(c.events, []);
});

test("crossing a deadline advances only the post-start part of that update", () => {
  const c = make(), a = c.createAction(0).startAt(2).play();
  c.update(3.25);
  close(a.time, 1.25);
  close(c.pose.translations[0], 1.25);
  assert.equal(c.time, 3.25);
  assert.deepEqual(c.events, [{ type: "started", action: a, time: 2 }]);
});

test("late starts catch up from the specified deadline even on a zero-delta update", () => {
  const c = make();
  c.update(5);
  const a = c.createAction(0).startAt(2).play().warpTo(3, 2);
  c.update(0);
  close(a.time, 7); // Two ramp seconds (area 4) followed by one second at 3.
  close(c.pose.translations[0], 7);
  assert.equal(c.time, 5);
  assert.equal(a.warping, false);
  assert.deepEqual(c.events, [{ type: "started", action: a, time: 2 }]);
  c.update(0);
  assert.deepEqual(c.events, []);
  close(a.time, 7);
});

test("assigning a deadline does not play an action; play can later catch up", () => {
  const c = make(), a = c.createAction(0).startAt(1);
  c.update(3);
  assert.equal(a.playing, false);
  assert.equal(a.scheduled, true);
  assert.deepEqual(c.events, []);
  a.play();
  c.update(0);
  close(a.time, 2);
  assert.equal(c.events[0].time, 1);
});

test("a started event precedes aggregated loop and finish records within the same update", () => {
  const c = make(2), a = c.createAction(0, { repetitions: 3, clampWhenFinished: true }).startAt(1).play();
  c.update(10);
  assert.equal(a.time, 2);
  assert.equal(a.finished, true);
  assert.deepEqual(c.events, [
    { type: "started", action: a, time: 1 },
    { type: "loop", action: a, count: 2, direction: 1 },
    { type: "finished", action: a, direction: 1 },
  ]);
  const previous = c.events;
  c.update(1);
  assert.equal(c.events.length, 0);
  assert.equal(previous.length, 3);
});

test("start effects retain action creation order, not a new global timestamp sort", () => {
  const c = make(2), a = c.createAction(0).startAt(2).play(), b = c.createAction(0).startAt(1).play();
  c.update(4.25);
  assert.deepEqual(c.events.map((e) => [e.type, e.action, e.time]), [
    ["started", a, 2], ["loop", a, undefined],
    ["started", b, 1], ["loop", b, undefined],
  ]);
});

test("reverse starts use the same controller clock and existing reverse clip endpoint", () => {
  const c = make(2), a = c.createAction(0, { timeScale: -2, loop: "once", clampWhenFinished: true }).startAt(3).play();
  c.update(3.25);
  close(a.time, 1.5);
  assert.equal(c.events[0].type, "started");
  c.update(1);
  assert.equal(a.time, 0);
  assert.equal(a.finished, true);
  assert.deepEqual(c.events, [{ type: "finished", action: a, direction: -1 }]);
});

test("paused starts release on time and advance fades/warps, but not the playhead", () => {
  const c = make(), a = c.createAction(0).startAt(1).play().pause().warpTo(3, 2).fadeTo(0, 2);
  c.update(2);
  assert.equal(a.scheduled, false);
  assert.equal(a.paused, true);
  close(a.time, 0);
  close(a.timeScale, 2);
  close(a.weight, 0.5);
  a.play();
  c.update(0.5);
  close(a.time, 1.125);
});

test("cancelStart resumes an armed action without retroactive catch-up or a started event", () => {
  const c = make(), a = c.createAction(0).play().seek(2).startAt(10);
  c.update(3);
  assert.equal(a.cancelStart(), a);
  assert.equal(a.scheduled, false);
  close(a.effectiveWeight, 1);
  c.update(0.5);
  close(a.time, 2.5);
  assert.deepEqual(c.events, []);
  a.stop().startAt(20).cancelStart();
  c.update(1);
  assert.equal(a.playing, false);
});

test("reset and stop cancel pending starts along with ramps", () => {
  for (const command of ["reset", "stop"]) {
    const c = make(), a = c.createAction(0).startAt(10).play().warpTo(3, 2);
    a[command]();
    assert.equal(a.scheduled, false);
    assert.equal(a.warping, false);
    c.update(1);
    close(a.time, command === "reset" ? 1 : 0);
    assert.deepEqual(c.events, []);
  }
});

test("replaying a finished action preserves an explicitly assigned new deadline", () => {
  const c = make(2), a = c.createAction(0, { loop: "once", clampWhenFinished: true }).play();
  c.update(2);
  assert.equal(a.finished, true);
  a.startAt(4).play();
  assert.equal(a.time, 0);
  assert.equal(a.startTime, 4);
  assert.equal(a.finished, false);
  c.update(1);
  assert.equal(a.scheduled, true);
  c.update(1.5);
  close(a.time, 0.5);
  assert.deepEqual(c.events, [{ type: "started", action: a, time: 4 }]);
});

test("rescheduling a playing action preserves its local phase and supports seeking while waiting", () => {
  const c = make(), a = c.createAction(0).play();
  c.update(1);
  a.startAt(4);
  c.update(2);
  close(a.time, 1);
  close(c.pose.translations[0], 0);
  a.seek(5);
  c.update(2);
  close(a.time, 6);
  assert.equal(c.events[0].time, 4);
});

test("an immediate crossfade cancels the target's schedule rather than fading to silence", () => {
  const c = make(), a = c.createAction(0).play().seek(2), b = c.createAction(0).startAt(100).play().seek(4);
  c.crossFade(a, b, 2);
  assert.equal(b.scheduled, false);
  c.update(1);
  close(a.weight, 0.5);
  close(b.weight, 0.5);
  close(c.pose.translations[0], 4);
  assert.deepEqual(c.events, []);
});

test("a not-yet-started action cannot silently serve as a crossfade source", () => {
  const c = make(), a = c.createAction(0).startAt(10).play(), b = c.createAction(0).play();
  const beforeA = state(c, a), beforeB = state(c, b);
  assert.throws(() => c.crossFade(a, b, 1), { code: "ANIMATION_ACTION_FADE" });
  assert.deepEqual(state(c, a), beforeA);
  assert.deepEqual(state(c, b), beforeB);
});

test("failed pose publication leaves the start pending and emits it exactly once on retry", () => {
  const c = make(), a = c.createAction(0).startAt(1).play().warpTo(3, 2).fadeTo(0.5, 2);
  c.update(0.5);
  const before = state(c, a);
  assert.throws(() => c.update(1, { rootMatrix: [] }), { code: "ANIMATION_SHAPE" });
  assert.deepEqual(state(c, a), before);
  assert.equal(c.events, before.events);
  c.update(1);
  close(a.time, 0.625);
  close(a.timeScale, 1.5);
  close(a.weight, 0.875);
  assert.deepEqual(c.events, [{ type: "started", action: a, time: 1 }]);
  c.update(0);
  assert.deepEqual(c.events, []);
});

test("invalid deadlines and reentrant cancellation leave a scheduled action unchanged", () => {
  const c = make(), a = c.createAction(0).startAt(1).play();
  const before = state(c, a);
  for (const value of [-1, NaN, Infinity, "1", null, undefined]) {
    assert.throws(() => a.startAt(value), { code: "ANIMATION_ACTION_VALUE" });
    assert.deepEqual(state(c, a), before);
  }
  assert.throws(() => c.update(2, { get rootMatrix() { a.cancelStart(); } }), { code: "ANIMATION_REENTRANT" });
  assert.deepEqual(state(c, a), before);
  a.dispose();
  assert.throws(() => a.startAt(2), { code: "ANIMATION_ACTION_DISPOSED" });
  c.update(2);
  assert.deepEqual(c.events, []);
});

test("zero-duration scheduled clips start at zero delta but only finish on movement", () => {
  const c = createAnimationController(createAnimationPlayer({
    format: "f3d-animation-v1", nodes: [{}],
    clips: [{ channels: [{ node: 0, path: "translation", times: [0], values: [3, 0, 0] }] }],
  }));
  const a = c.createAction(0, { clampWhenFinished: true }).startAt(0).play();
  c.update(0);
  assert.deepEqual(c.events, [{ type: "started", action: a, time: 0 }]);
  assert.equal(a.finished, false);
  close(c.pose.translations[0], 3);
  c.update(1);
  assert.deepEqual(c.events, [{ type: "finished", action: a, direction: 1 }]);
});

test("scheduled warped playback publishes matching morph weights and joint palettes once", () => {
  const pose = createAnimationPlayer({
    format: "f3d-animation-v1", nodes: [{ weights: [0] }, {}],
    skins: [{ joints: [1] }], instances: [{ node: 0, skin: 0 }],
    clips: [{ channels: [
      { node: 1, path: "translation", times: [0, 2], values: [0, 0, 0, 2, 0, 0] },
      { node: 0, path: "weights", times: [0, 2], values: [0, 1] },
    ] }],
  });
  const c = createAnimationController(pose), a = c.createAction(0).startAt(1).play().warpTo(3, 2);
  const joints = pose.jointMatrices, weights = pose.morphWeights;
  c.update(2);
  close(a.time, 1.5);
  close(joints[12], 1.5);
  close(weights[0], 0.75);
  assert.equal(pose.jointMatrices, joints);
  assert.equal(pose.morphWeights, weights);
  assert.equal(pose.version, 1);
});

test("large and subdivided updates agree for scheduled warped and fading actions", () => {
  for (const start of [0, 0.125, 1.75, 4, 10]) {
    for (const speed of [2, -2]) {
      const run = (steps) => {
        const c = make(2), a = c.createAction(0, { timeScale: speed, loop: "pingpong" })
          .startAt(start).play().warpTo(-speed, 2).fadeTo(0.25, 3);
        const starts = [], counts = [0, 0];
        for (let step = 0; step < steps; step++) {
          c.update(6 / steps);
          for (const event of c.events) {
            if (event.type === "started") starts.push(event.time);
            else if (event.type === "loop") counts[event.direction > 0 ? 0 : 1] += event.count;
          }
        }
        return { c, a, starts, counts };
      };
      const large = run(1), small = run(96);
      close(large.a.time, small.a.time);
      close(large.a.timeScale, small.a.timeScale);
      close(large.a.weight, small.a.weight);
      close(large.c.pose.translations[0], small.c.pose.translations[0]);
      assert.equal(large.a.scheduled, small.a.scheduled);
      assert.deepEqual(large.starts, small.starts);
      assert.deepEqual(large.counts, small.counts);
    }
  }
});
