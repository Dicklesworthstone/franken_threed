import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationController } from "./animation_controller.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

function make(duration = 20) {
  const pose = createAnimationPlayer({
    format: "f3d-animation-v1",
    nodes: [{}],
    clips: [{ channels: [{ node: 0, path: "translation", times: [0, duration], values: [0, 0, 0, duration, 0, 0] }] }],
  });
  return createAnimationController(pose);
}
const close = (a, b, epsilon = 1e-10) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
const events = (c) => c.events.map(({ type, count, direction }) => ({ type, count, direction }));
const snapshot = (c, a) => ({
  time: c.time, events: c.events, version: c.pose.version,
  pose: [...c.pose.worldMatrices], actionTime: a.time, speed: a.timeScale,
  weight: a.weight, playing: a.playing, paused: a.paused, finished: a.finished,
  warping: a.warping, traversals: a.completedTraversals,
});

test("linear warps integrate elapsed clip time and apply the endpoint to the remaining delta", () => {
  const c = make(), a = c.createAction(0).play();
  const output = c.pose.worldMatrices;
  assert.equal(a.warp(1, 3, 2), a);
  c.update(0.5);
  close(a.timeScale, 1.5);
  close(a.time, 0.625);
  close(c.pose.translations[0], 0.625);
  assert.equal(a.warping, true);
  c.update(2.5);
  close(a.time, 7); // integral over [0,2] = 4, then one second at 3.
  close(a.timeScale, 3);
  assert.equal(a.warping, false);
  assert.equal(c.pose.worldMatrices, output);
  assert.equal(c.pose.version, 2); // Only one pose publication per update.
});

test("interruption starts from the current speed; cancellation and immediate warps are chainable", () => {
  const c = make(), a = c.createAction(0).play().warpTo(3, 2);
  c.update(1);
  close(a.time, 1.5);
  assert.equal(a.warpTo(0, 2), a);
  c.update(1);
  close(a.time, 3);
  close(a.timeScale, 1);
  assert.equal(a.stopWarping(), a);
  c.update(1);
  close(a.time, 4);
  assert.equal(a.warping, false);
  a.warpTo(10, 2).setTimeScale(2);
  c.update(1);
  close(a.time, 6);
  assert.equal(a.warping, false);
  a.warp(100, -2, 0);
  close(a.timeScale, -2);
  c.update(1);
  close(a.time, 4);
});

for (const sign of [1, -1]) {
  test(`halt keeps the pose and weight without advancing after its boundary, direction=${sign}`, () => {
    const c = make(), a = c.createAction(0, { timeScale: sign, weight: 0.75 }).play();
    const initial = a.time;
    a.halt(2);
    c.update(20);
    close(a.time, initial + sign);
    assert.equal(a.timeScale, 0);
    assert.equal(a.playing, true);
    assert.equal(a.warping, false);
    assert.equal(a.weight, 0.75);
    const time = a.time;
    c.update(20);
    close(a.time, time);
    a.setTimeScale(sign);
    c.update(0.5);
    close(a.time, time + sign * 0.5);
  });
}

test("paused actions keep progressing wall-time fades and warps without moving the playhead", () => {
  const c = make(), a = c.createAction(0).play().warpTo(3, 2).fadeTo(0.5, 2).pause();
  c.update(1);
  close(a.time, 0);
  close(a.timeScale, 2);
  close(a.weight, 0.75);
  assert.equal(a.effectiveTimeScale, 0);
  a.play();
  c.update(1);
  close(a.time, 2.5);
  close(a.timeScale, 3);
  close(a.weight, 0.5);
  assert.equal(a.effectiveTimeScale, 3);
});

test("reversing ramps preserve both traversal directions instead of cancelling their displacement", () => {
  const c = make(2), a = c.createAction(0).play().seek(0.5).warp(4, -4, 2);
  c.update(2);
  close(a.time, 0.5);
  assert.equal(a.completedTraversals, 2);
  assert.deepEqual(events(c), [
    { type: "loop", count: 1, direction: 1 },
    { type: "loop", count: 1, direction: -1 },
  ]);
  assert.ok(c.events.every(Object.isFrozen));
});

test("finite playback finishes before a warp reversal and does not resume in the same update", () => {
  const c = make(2), a = c.createAction(0, { loop: "once", clampWhenFinished: true }).play().seek(0.5).warp(4, -4, 2);
  c.update(3);
  assert.equal(a.finished, true);
  assert.equal(a.time, 2);
  assert.equal(a.timeScale, -4);
  assert.equal(a.effectiveTimeScale, 0);
  assert.deepEqual(events(c), [{ type: "finished", count: undefined, direction: 1 }]);
});

test("same-direction crossings across the ramp endpoint are one bounded event record", () => {
  const c = make(2), a = c.createAction(0).play().warp(2, 4, 2);
  c.update(1000002);
  assert.equal(c.events.length, 1);
  assert.equal(c.events[0].count, 2000003);
  assert.equal(a.completedTraversals, 2000003);
  assert.equal(a.time, 0);
});

test("fade-out clips warp advancement and events at the fade boundary", () => {
  const c = make(2), a = c.createAction(0).play().warp(0, 4, 2).fadeTo(0, 0.5, { stopWhenDone: true });
  c.update(100);
  assert.equal(a.playing, false);
  assert.equal(a.warping, false);
  close(a.timeScale, 1);
  assert.deepEqual(c.events, []);
  close(c.pose.translations[0], 0);
});

test("failed pose evaluation rolls back warp progress, clocks, events and output together", () => {
  const c = make(2), a = c.createAction(0).play().warpTo(3, 2).fadeTo(0.5, 2);
  c.update(0.25);
  const before = snapshot(c, a);
  assert.throws(() => c.update(2, { rootMatrix: [] }), { code: "ANIMATION_SHAPE" });
  assert.deepEqual(snapshot(c, a), before);
  assert.equal(c.events, before.events);
  c.update(0.25);
  close(a.time, 0.625);
  close(a.timeScale, 1.5);
  close(a.weight, 0.875);
});

test("invalid warp configuration leaves the previous warp and controls intact", () => {
  const c = make(), a = c.createAction(0).play().warpTo(3, 2);
  c.update(0.25);
  const before = snapshot(c, a);
  for (const args of [[NaN, 1, 1], [1, Infinity, 1], [1, 2, -1], [1, 2, NaN], ["1", 2, 1]]) {
    assert.throws(() => a.warp(...args), { code: "ANIMATION_ACTION_VALUE" });
    assert.deepEqual(snapshot(c, a), before);
  }
  assert.throws(() => a.warpTo(2, Infinity), { code: "ANIMATION_ACTION_VALUE" });
  assert.throws(() => a.halt(-1), { code: "ANIMATION_ACTION_VALUE" });
  assert.deepEqual(snapshot(c, a), before);
  c.update(0.25);
  close(a.timeScale, 1.5);
});

test("zero delta consumes no warp; reset and stop cancel ramps without restoring speed", () => {
  const c = make(), a = c.createAction(0).play().warpTo(3, 2);
  c.update(0);
  close(a.timeScale, 1);
  assert.equal(a.warping, true);
  c.update(1);
  a.reset();
  assert.equal(a.warping, false);
  close(a.timeScale, 2);
  a.warpTo(5, 1).stop();
  assert.equal(a.warping, false);
  close(a.timeScale, 2);
  assert.equal(a.effectiveTimeScale, 0);
});

test("overflowing playback rolls back instead of publishing an invalid warped clock", () => {
  const c = make(), a = c.createAction(0).play().warp(1e308, 1e308, 10);
  const before = snapshot(c, a);
  assert.throws(() => c.update(10), { code: "ANIMATION_ACTION_VALUE" });
  assert.deepEqual(snapshot(c, a), before);
});

test("extreme finite ramp endpoints do not overflow the average or reversal ratio", () => {
  const c = make(), a = c.createAction(0).play().seek(1).warp(1e308, -1e308, 1e-308);
  c.update(1e-308);
  close(a.time, 1);
  close(a.timeScale, -1e308);
  const d = make(), b = d.createAction(0).play().warp(Number.MIN_VALUE, 0, 1e308);
  d.update(1e308);
  close(b.time / (Number.MIN_VALUE * 1e308), 0.5, 1e-12);
});

test("zero-duration clips finish once on the first nonzero warped movement", () => {
  const pose = createAnimationPlayer({ format: "f3d-animation-v1", nodes: [{}],
    clips: [{ channels: [{ node: 0, path: "translation", times: [0], values: [3, 0, 0] }] }] });
  const c = createAnimationController(pose), a = c.createAction(0, { clampWhenFinished: true }).play().warp(0, -1, 1);
  c.update(0);
  assert.equal(a.finished, false);
  c.update(0.5);
  assert.equal(a.finished, true);
  assert.deepEqual(events(c), [{ type: "finished", count: undefined, direction: -1 }]);
  c.update(10);
  assert.deepEqual(c.events, []);
  close(pose.translations[0], 3);
});

test("a caller getter cannot mutate or dispose a warp during pose publication", () => {
  const c = make(), a = c.createAction(0).play().warpTo(3, 2);
  const before = snapshot(c, a);
  assert.throws(() => c.update(1, { get rootMatrix() { a.stopWarping(); } }), { code: "ANIMATION_REENTRANT" });
  assert.deepEqual(snapshot(c, a), before);
  a.dispose();
  assert.throws(() => a.halt(1), { code: "ANIMATION_ACTION_DISPOSED" });
});

for (const loop of ["once", "repeat", "pingpong"]) {
  test(`large and subdivided updates agree through ramp completion and reversals (${loop})`, () => {
    let seed = 173;
    const random = () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 4294967296);
    for (let i = 0; i < 80; i++) {
      const start = random() * 8 - 4, end = random() * 8 - 4;
      const ramp = 0.1 + random() * 3, total = ramp + random() * 2;
      const time = random() * 2, repetitions = i % 2 ? Infinity : 3;
      const run = (steps) => {
        const c = make(2), a = c.createAction(0, { loop, repetitions, clampWhenFinished: i % 3 === 0 }).play().seek(time).warp(start, end, ramp);
        const counts = [0, 0], finishes = [];
        for (let step = 0; step < steps; step++) {
          c.update(total / steps);
          for (const event of c.events) {
            if (event.type === "loop") counts[event.direction > 0 ? 0 : 1] += event.count;
            else finishes.push(event.direction);
          }
        }
        return { a, c, counts, finishes };
      };
      const large = run(1), small = run(64);
      close(large.a.time, small.a.time, 1e-9);
      close(large.a.timeScale, small.a.timeScale, 1e-9);
      close(large.c.pose.translations[0], small.c.pose.translations[0], 1e-9);
      assert.equal(large.a.completedTraversals, small.a.completedTraversals);
      assert.equal(large.a.finished, small.a.finished);
      assert.deepEqual(large.counts, small.counts);
      assert.deepEqual(large.finishes, small.finishes);
    }
  });
}
