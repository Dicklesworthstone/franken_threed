import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationController } from "./animation_controller.mjs";
import { createAnimationPlayer } from "./animation_runtime.mjs";

function make(durations = [2, 4]) {
  return createAnimationController(createAnimationPlayer({
    format: "f3d-animation-v1", nodes: [{}],
    clips: durations.map((duration) => ({ channels: [{ node: 0, path: "translation",
      times: duration === 0 ? [0] : [0, duration],
      values: duration === 0 ? [1, 0, 0] : [0, 0, 0, 1, 0, 0],
    }] })),
  }));
}
const close = (a, b, epsilon = 1e-10) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
const state = (c, a) => ({
  time: c.time, version: c.pose.version, events: c.events, pose: [...c.pose.worldMatrices],
  actionTime: a.time, speed: a.timeScale, weight: a.weight, playing: a.playing,
  paused: a.paused, finished: a.finished, scheduled: a.startTime,
  warping: a.warping, traversals: a.completedTraversals,
});

for (const speed of [1, -1, 0]) {
  test(`setDuration sets seconds per traversal without resetting time, speed=${speed}`, () => {
    const c = make(), a = c.createAction(0, { timeScale: speed }).play().seek(0.5).warpTo(speed, 10);
    assert.equal(a.duration, 2);
    assert.equal(a.setDuration(4), a);
    assert.equal(a.warping, false);
    close(a.time, 0.5);
    close(a.timeScale, speed < 0 ? -0.5 : 0.5);
    c.update(0.5);
    close(a.time, speed < 0 ? 0.25 : 0.75);
  });
}

test("invalid durations do not cancel an existing ramp or mutate clocks", () => {
  const c = make([2, 0]), a = c.createAction(0).play().warpTo(3, 2);
  const before = state(c, a);
  for (const duration of [0, -1, NaN, Infinity, "1", Number.MIN_VALUE]) {
    assert.throws(() => a.setDuration(duration));
    assert.deepEqual(state(c, a), before);
  }
  const still = c.createAction(1).play();
  assert.throws(() => still.setDuration(1), { code: "ANIMATION_ACTION_DURATION" });
});

test("syncWith copies normalized local phase and rate, not raw seconds", () => {
  const c = make(), a = c.createAction(0, { timeScale: 1.5 }).play().seek(0.5), b = c.createAction(1);
  assert.equal(b.syncWith(a), b);
  close(b.time, 1);
  close(b.timeScale, 3);
  assert.equal(b.playing, false);
  b.play();
  c.update(0.25);
  close(a.time / a.duration, b.time / b.duration);
  close(c.pose.translations[0], a.time / a.duration);
});

test("synchronization preserves target fades, weight, pause and schedule while clearing its ramp", () => {
  const c = make(), a = c.createAction(0).play().seek(0.5), b = c.createAction(1, { weight: 0.6 })
    .startAt(3).play().pause().warpTo(4, 2).fadeTo(0.2, 2);
  const source = state(c, a);
  b.syncWith(a);
  assert.deepEqual(state(c, a), source);
  close(b.time, 1);
  close(b.timeScale, 2);
  assert.equal(b.warping, false);
  assert.equal(b.paused, true);
  assert.equal(b.startTime, 3);
  close(b.weight, 0.6);
  c.update(4);
  close(b.time, 1);
  close(b.weight, 0.4);
});

test("syncWith starts a fresh traversal budget and clears finished without activating", () => {
  const c = make(), a = c.createAction(0).play().seek(0.5), b = c.createAction(1, { loop: "once", clampWhenFinished: true }).play();
  c.update(4);
  assert.equal(b.finished, true);
  a.seek(0.5);
  b.syncWith(a);
  assert.equal(b.finished, false);
  assert.equal(b.playing, false);
  assert.equal(b.completedTraversals, 0);
  close(b.time, 1);
  b.play();
  c.update(0.25);
  close(b.time, 1.5);
});

test("syncWith preserves the descending ping-pong leg", () => {
  const c = make(), a = c.createAction(0, { loop: "pingpong" }).play(), b = c.createAction(1, { loop: "pingpong" });
  c.update(2.5);
  close(a.time, 1.5);
  b.syncWith(a).play();
  c.update(0.5);
  close(a.time, 1);
  close(b.time, 2);
  c.update(1.5);
  close(a.time / a.duration, b.time / b.duration);
});

test("one-shot synchronization maps the local direction even between different loop modes", () => {
  const c = make(), a = c.createAction(0, { loop: "pingpong" }).play(), b = c.createAction(1);
  c.update(2.5);
  b.syncWith(a).play();
  close(b.timeScale, -2);
  c.update(0.25);
  close(a.time, 1.25);
  close(b.time, 2.5);
});

test("self-synchronization is a no-op; foreign and disposed actions are rejected", () => {
  const c = make(), a = c.createAction(0).play().warpTo(3, 2);
  c.update(0.5);
  const before = state(c, a);
  a.syncWith(a);
  assert.deepEqual(state(c, a), before);
  const foreign = make().createAction(0), disposed = c.createAction(1);
  disposed.dispose();
  for (const other of [foreign, disposed, null, {}]) {
    assert.throws(() => a.syncWith(other), { code: "ANIMATION_ACTION_DISPOSED" });
    assert.deepEqual(state(c, a), before);
  }
});

for (const durations of [[1e-200, 1e200], [1e200, 1e-200]]) {
  test(`duration scaling avoids overflowing or underflowing an intermediate ratio: ${durations}`, () => {
    const c = make(durations), a = c.createAction(0, { timeScale: durations[0] }).seek(durations[0] / 2), b = c.createAction(1);
    b.syncWith(a);
    close(b.time / durations[1], 0.5);
    close(b.timeScale / durations[1], 1);
  });
}

test("same-duration synchronization retains representable subnormal phase", () => {
  const c = make([1e308, 1e308]), a = c.createAction(0).seek(Number.MIN_VALUE), b = c.createAction(1);
  b.syncWith(a);
  assert.equal(b.time, Number.MIN_VALUE);
});

test("unrepresentable scaled rates and zero clip durations fail before either action changes", () => {
  const c = make([1e-200, 1e200, 0]), a = c.createAction(0, { timeScale: 1e200 }).play(), b = c.createAction(1).startAt(10).play().warpTo(3, 2);
  const beforeA = state(c, a), beforeB = state(c, b);
  assert.throws(() => b.syncWith(a), { code: "ANIMATION_ACTION_DURATION" });
  assert.throws(() => c.crossFade(a, b, 1, { sync: true, warp: true }), { code: "ANIMATION_ACTION_DURATION" });
  assert.deepEqual(state(c, a), beforeA);
  assert.deepEqual(state(c, b), beforeB);
  const zero = c.createAction(2);
  assert.throws(() => b.syncWith(zero), { code: "ANIMATION_ACTION_DURATION" });
  assert.throws(() => zero.syncWith(b), { code: "ANIMATION_ACTION_DURATION" });
});

for (const speed of [1, -1]) {
  test(`phase-aligned warped crossfades keep unequal-length clips in step, direction=${speed}`, () => {
    const c = make(), a = c.createAction(0, { timeScale: speed }).play().seek(0.75), b = c.createAction(1, { timeScale: speed });
    assert.equal(c.crossFade(a, b, 1, { sync: true, warp: true }), c);
    close(b.time, 1.5);
    close(b.timeScale, speed * 2);
    close(a.timeScale, speed);
    for (let i = 0; i < 7; i++) {
      c.update(0.125);
      close(a.time / a.duration, b.time / b.duration);
      close(a.timeScale / a.duration, b.timeScale / b.duration);
      close(c.pose.translations[0], b.time / b.duration);
    }
    c.update(0.125);
    assert.equal(a.playing, false);
    assert.equal(b.warping, false);
    close(b.timeScale, speed);
    close(b.weight, 1);
  });
}

test("warped crossfades can align descending ping-pong clips through reflection", () => {
  const c = make(), a = c.createAction(0, { loop: "pingpong" }).play(), b = c.createAction(1, { loop: "pingpong" });
  c.update(3.5);
  close(a.time, 0.5);
  c.crossFade(a, b, 2, { sync: true, warp: true });
  for (let i = 0; i < 7; i++) {
    c.update(0.25);
    close(a.time / a.duration, b.time / b.duration);
    close(c.pose.translations[0], b.time / b.duration);
  }
});

test("a warp-only crossfade preserves phase offsets and accounts for opposite ping-pong legs", () => {
  const c = make(), a = c.createAction(0, { loop: "pingpong" }).play(), b = c.createAction(1, { loop: "pingpong" });
  c.update(2.5);
  b.seek(1);
  c.crossFade(a, b, 1, { warp: true });
  close(a.time, 1.5);
  close(b.time, 1);
  close(b.timeScale, -2);
  c.update(0.25);
  // Local directions agree even though the timeScale signs differ.
  close(a.time / a.duration - b.time / b.duration, 0.5);
});

test("sync without warping aligns only phase and retains the target's configured speed", () => {
  const c = make(), a = c.createAction(0).play().seek(0.5), b = c.createAction(1, { timeScale: 0.5 });
  c.crossFade(a, b, 2, { sync: true });
  close(b.time, 1);
  close(b.timeScale, 0.5);
  assert.equal(b.warping, false);
  c.update(0.5);
  close(a.time, 1);
  close(b.time, 1.25);
});

test("interrupting a phase-aligned transition preserves current weights and local phase", () => {
  const c = make(), a = c.createAction(0).play().seek(0.5), b = c.createAction(1);
  c.crossFade(a, b, 2, { sync: true, warp: true });
  c.update(0.5);
  const timeA = a.time, timeB = b.time, speedA = a.timeScale, speedB = b.timeScale;
  c.crossFade(b, a, 1, { sync: true, warp: true });
  close(a.time, timeA); close(b.time, timeB);
  close(a.timeScale, speedA); close(b.timeScale, speedB);
  close(a.weight, 0.75); close(b.weight, 0.25);
  c.update(0.5);
  close(a.weight, 0.875); close(b.weight, 0.125);
  close(a.time / a.duration, b.time / b.duration);
  c.update(0.5);
  assert.equal(b.playing, false);
  close(a.weight, 1);
});

test("zero-time synchronized crossfade aligns immediately without events or residual ramps", () => {
  const c = make(), a = c.createAction(0).play().seek(0.75), b = c.createAction(1).startAt(10);
  c.crossFade(a, b, 0, { sync: true, warp: true });
  assert.equal(a.playing, false);
  assert.equal(b.scheduled, false);
  close(b.time, 1.5);
  close(b.timeScale, 1);
  assert.equal(b.warping, false);
  c.update(0);
  close(c.pose.translations[0], 0.375);
  assert.deepEqual(c.events, []);
});

test("scheduled targets start immediately in synchronized crossfades", () => {
  const c = make(), a = c.createAction(0).play().seek(0.5), b = c.createAction(1).startAt(100).play().pause();
  c.crossFade(a, b, 1, { sync: true, warp: true });
  assert.equal(b.scheduled, false);
  assert.equal(b.paused, false);
  c.update(0.5);
  close(a.time / a.duration, b.time / b.duration);
  assert.deepEqual(c.events, []);
});

test("invalid crossfade options and reentrant getters preserve both actions", () => {
  const c = make(), a = c.createAction(0).play().warpTo(3, 2), b = c.createAction(1).startAt(10).play();
  const beforeA = state(c, a), beforeB = state(c, b);
  for (const options of [null, 1, [], { sync: 1 }, { warp: null }, { unknown: true },
    { get sync() { a.stop(); } }]) {
    assert.throws(() => c.crossFade(a, b, 1, options));
    assert.deepEqual(state(c, a), beforeA);
    assert.deepEqual(state(c, b), beforeB);
  }
});

test("advanced crossfades reject incompatible loops or a frozen source without affecting ordinary fades", () => {
  const c = make([2, 0]), a = c.createAction(0).play(), b = c.createAction(0, { loop: "pingpong" });
  assert.throws(() => c.crossFade(a, b, 1, { sync: true }), { code: "ANIMATION_ACTION_FADE" });
  a.pause();
  const same = c.createAction(0);
  assert.throws(() => c.crossFade(a, same, 1, { warp: true }), { code: "ANIMATION_ACTION_FADE" });
  const zero = c.createAction(1);
  assert.throws(() => c.crossFade(a, zero, 1, { sync: true }), { code: "ANIMATION_ACTION_DURATION" });
  c.crossFade(a, zero, 1);
  c.update(0.5);
  assert.equal(zero.finished, true);
});

test("failed pose publication rolls back synchronized ramps, event lists and all clocks", () => {
  const c = make(), a = c.createAction(0).play().seek(0.5), b = c.createAction(1);
  c.crossFade(a, b, 2, { sync: true, warp: true });
  c.update(0.25);
  const beforeA = state(c, a), beforeB = state(c, b);
  assert.throws(() => c.update(1, { rootMatrix: [] }), { code: "ANIMATION_SHAPE" });
  assert.deepEqual(state(c, a), beforeA);
  assert.deepEqual(state(c, b), beforeB);
  c.update(0.25);
  close(a.time / a.duration, b.time / b.duration);
});

for (const loop of ["once", "repeat", "pingpong"]) {
  test(`large and subdivided synchronized transitions agree through reversals and completion (${loop})`, () => {
    for (const speeds of [[1, 1], [-1, -1], [4, -2], [0, 2]]) {
      for (const total of [0, 0.5, 1, 2.5, 8]) {
        const run = (steps) => {
          const c = make(), a = c.createAction(0, { loop, timeScale: speeds[0], clampWhenFinished: true }).play().seek(1),
            b = c.createAction(1, { loop, timeScale: speeds[1], clampWhenFinished: true });
          c.crossFade(a, b, 2, { sync: true, warp: true });
          const eventCounts = [0, 0, 0, 0], finishes = [];
          for (let i = 0; i < steps; i++) {
            c.update(total / steps);
            for (const e of c.events) {
              const index = e.action === a ? 0 : 2;
              if (e.type === "loop") eventCounts[index + (e.direction < 0 ? 1 : 0)] += e.count;
              else if (e.type === "finished") finishes.push([index, e.direction]);
            }
          }
          return { c, a, b, eventCounts, finishes };
        };
        const big = run(1), small = run(64);
        for (const key of ["a", "b"]) {
          close(big[key].time, small[key].time);
          close(big[key].timeScale, small[key].timeScale);
          close(big[key].weight, small[key].weight);
          assert.equal(big[key].playing, small[key].playing);
          assert.equal(big[key].finished, small[key].finished);
          assert.equal(big[key].completedTraversals, small[key].completedTraversals);
        }
        close(big.c.pose.translations[0], small.c.pose.translations[0]);
        assert.deepEqual(big.eventCounts, small.eventCounts);
        assert.deepEqual(big.finishes, small.finishes);
      }
    }
  });
}

test("reversing warp preserves a loop exactly at its endpoint without snapping early ordinary frames", () => {
  const c = make(), a = c.createAction(0).play().seek(1).warp(4, -1, 2);
  c.update(2);
  assert.equal(a.time, 2); // Reverse repeat wraps to the end at zero.
  assert.deepEqual(c.events.map(({ type, count, direction }) => [type, count, direction]),
    [["loop", 2, 1], ["loop", 1, -1]]);
  const d = make(), b = d.createAction(0).play();
  d.update(2 - 4 * Number.EPSILON);
  assert.equal(b.completedTraversals, 0);
  assert.deepEqual(d.events, []);
});

test("a small reversed tail must not become negative through cancellation", () => {
  for (let i = 0; i < 1000; i++) {
    const from = 1 + i / 17, to = -(10 ** (-(i % 25))), duration = 1 + (i % 7) / 11;
    const c = make([1000]), a = c.createAction(0).play().seek(1).warp(from, to, duration);
    c.update(duration);
    close(a.time, 1 + duration * (from + to) * 0.5, 1e-10);
    close(a.timeScale, to);
  }
});
