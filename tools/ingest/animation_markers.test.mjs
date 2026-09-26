import test from "node:test";
import assert from "node:assert/strict";
import { AnimationMarkerError, animationMarkerEventLimit, createAnimationMarkerTrack } from "./animation_markers.mjs";

const action = Object.freeze({ clip: 7 });
const markers = [
  { name: "end", time: 1 }, { name: "left", time: 0.25 },
  { name: "start", time: 0 }, { name: "right", time: 0.75 },
];
const movement = (overrides = {}) => ({
  start: 0, end: 1, direction: 1, loop: "once", crossings: 1,
  finished: true, traversal: 0, ...overrides,
});
function collect(input, move, { duration = 1, maxEvents = 4096, output = [] } = {}) {
  createAnimationMarkerTrack(input, duration).append(output, action, move, maxEvents);
  return output;
}
const tuples = (events) => events.map(({ name, direction, traversal }) => [name, direction, traversal]);
const error = (code) => (e) => e instanceof AnimationMarkerError && e.code === `ANIMATION_MARKER_${code}`;

test("tracks copy and freeze source records, sorted without mutating their input", () => {
  const input = structuredClone(markers);
  const track = createAnimationMarkerTrack(input, 1);
  assert.deepEqual(input, markers);
  assert.deepEqual(track.markers.map((m) => m.name), ["start", "left", "right", "end"]);
  input[0].name = "changed"; input[1].time = 1; input.length = 0;
  assert.equal(track.markers[3].name, "end");
  assert.equal(track.markers[1].time, 0.25);
  assert.ok(Object.isFrozen(track)); assert.ok(Object.isFrozen(track.markers));
  assert.ok(track.markers.every(Object.isFrozen));
  assert.throws(() => { track.markers[0].time = 1; }, TypeError);
});

test("forward intervals exclude departure and include arrival", () => {
  const out = collect(markers, movement({ start: 0.25, end: 0.75, crossings: 0, finished: false }));
  assert.deepEqual(tuples(out), [["right", 1, 0]]);
  assert.deepEqual(collect(markers, movement({ start: 0.25, end: 0.25, crossings: 0, finished: false })), []);
  assert.equal(out[0].action, action); assert.equal(out[0].time, 0.75);
  assert.equal(out[0].type, "marker"); assert.ok(Object.isFrozen(out[0]));
});

test("reverse intervals exclude departure and include arrival", () => {
  const out = collect(markers, movement({ start: 0.75, end: 0.25, direction: -1, crossings: 0, finished: false }));
  assert.deepEqual(tuples(out), [["left", -1, 0]]);
  assert.deepEqual(tuples(collect(markers, movement({ start: 1, end: 0, direction: -1 }))),
    [["right", -1, 0], ["left", -1, 0], ["start", -1, 0]]);
});

test("simultaneous cues retain declaration order in both directions", () => {
  const input = [ { name: "a", time: 0.5 }, { name: "b", time: 0.5 }, { name: "a", time: 0.5 } ];
  for (const direction of [-1, 1]) {
    assert.deepEqual(collect(input, movement({ start: direction > 0 ? 0 : 1,
      end: direction > 0 ? 1 : 0, direction })).map((e) => e.name), ["a", "b", "a"]);
  }
});

test("repeat emits both distinct wrap endpoints with the correct traversal", () => {
  assert.deepEqual(tuples(collect(markers, movement({ end: 0, loop: "repeat", finished: false }))),
    [["left", 1, 0], ["right", 1, 0], ["end", 1, 0], ["start", 1, 1]]);
  assert.deepEqual(tuples(collect(markers, movement({ start: 1, end: 1, direction: -1,
    loop: "repeat", finished: false, traversal: 9 }))),
    [["right", -1, 9], ["left", -1, 9], ["start", -1, 9], ["end", -1, 10]]);
});

test("finite repeat completion stops at the last endpoint, without a phantom wrap", () => {
  assert.deepEqual(tuples(collect(markers, movement({ loop: "repeat", crossings: 2 }))), [
    ["left", 1, 0], ["right", 1, 0], ["end", 1, 0],
    ["start", 1, 1], ["left", 1, 1], ["right", 1, 1], ["end", 1, 1],
  ]);
});

test("ping-pong endpoints fire only on arrival, with local rather than clock direction", () => {
  assert.deepEqual(tuples(collect(markers, movement({ end: 0.25, loop: "pingpong", crossings: 2, finished: false }))), [
    ["left", 1, 0], ["right", 1, 0], ["end", 1, 0],
    ["right", -1, 1], ["left", -1, 1], ["start", -1, 1], ["left", 1, 2],
  ]);
  assert.deepEqual(tuples(collect(markers, movement({ start: 1, end: 1, direction: -1,
    loop: "pingpong", crossings: 2 }))), [
    ["right", -1, 0], ["left", -1, 0], ["start", -1, 0],
    ["left", 1, 1], ["right", 1, 1], ["end", 1, 1],
  ]);
});

test("starting at an endpoint emits no duplicate departure cue", () => {
  assert.deepEqual(tuples(collect(markers, movement({ start: 1, end: 0.75,
    loop: "pingpong", crossings: 1, finished: false }))), [["right", -1, 1]]);
  assert.deepEqual(tuples(collect(markers, movement({ start: 1, end: 0.25,
    loop: "repeat", crossings: 1, finished: false }))), [["start", 1, 1], ["left", 1, 1]]);
});

test("zero-duration cues emit only at the one natural completion", () => {
  const input = [{ name: "instant", time: -0 }];
  const out = collect(input, movement({ end: 0 }), { duration: 0 });
  assert.deepEqual(tuples(out), [["instant", 1, 0]]); assert.ok(Object.is(out[0].time, -0));
  assert.deepEqual(collect(input, movement({ end: 0, crossings: 0, finished: false }), { duration: 0 }), []);
});

test("event-budget failure does not append even the first interval", () => {
  const output = [Object.freeze({ sentinel: true })];
  assert.throws(() => collect(markers, movement({ loop: "repeat", crossings: 3 }),
    { maxEvents: 6, output }), error("LIMIT"));
  assert.deepEqual(output, [{ sentinel: true }]);
  collect(markers, movement(), { maxEvents: 4, output });
  assert.equal(output.length, 4);
});

test("huge repeat and ping-pong traversals are rejected before per-loop work", { timeout: 1000 }, () => {
  for (const loop of ["repeat", "pingpong"]) {
    for (const input of [markers, [{ name: "edge", time: 0 }], [{ name: "edge", time: 1 }]]) {
      assert.throws(() => collect(input, movement({ loop, crossings: Number.MAX_SAFE_INTEGER,
        end: loop === "pingpong" ? 1 : 0, finished: false })), error("LIMIT"));
    }
    assert.deepEqual(collect([], movement({ loop, crossings: Number.MAX_SAFE_INTEGER })), []);
  }
});

test("all endpoint-only ping-pong cases admit exactly their reported count", () => {
  for (const direction of [-1, 1]) for (const time of [0, 1]) for (const crossings of [1, 2, 3, 100]) {
    const move = movement({ loop: "pingpong", start: direction > 0 ? 0 : 1,
      end: (direction > 0) === (crossings % 2 === 1) ? 1 : 0, direction, crossings });
    const input = [{ name: "edge", time }];
    const out = collect(input, move);
    if (out.length) {
      assert.equal(collect(input, move, { maxEvents: out.length }).length, out.length);
      if (out.length > 1) assert.throws(() => collect(input, move, { maxEvents: out.length - 1 }), error("LIMIT"));
    }
  }
});

for (const value of [null, {}, "x", [null], Array(1), [{ time: 0 }], [{ name: "" , time: 0 }],
  [{ name: "a".repeat(257), time: 0 }], [{ name: "a", time: NaN }], [{ name: "a", time: Infinity }],
  [{ name: "a", time: -1 }], [{ name: "a", time: 1.01 }], [{ name: "a", time: "0" }],
  [{ name: "a", time: 0, data: {} }], Array(4097).fill({ name: "a", time: 0 })]) {
  test(`refuses malformed marker input ${JSON.stringify(value)?.slice(0, 90)}`, () => {
    assert.throws(() => createAnimationMarkerTrack(value, 1), AnimationMarkerError);
  });
}

test("maximum marker count and name length are admitted exactly", () => {
  assert.equal(createAnimationMarkerTrack(Array(4096).fill({ name: "x".repeat(256), time: 1 }), 1).markers.length, 4096);
  for (const duration of [-1, Infinity, NaN, "1", null])
    assert.throws(() => createAnimationMarkerTrack([], duration), error("VALUE"));
  for (const limit of [0, -1, 65537, 1.5, NaN, Infinity, "2", null])
    assert.throws(() => animationMarkerEventLimit(limit), error("LIMIT"));
  assert.equal(animationMarkerEventLimit(1), 1);
  assert.equal(animationMarkerEventLimit(65536), 65536);
});

// An independent, deliberately slow path-walking oracle. It does not use the
// implementation's binary searches, full-leg counts or ping-pong parity formula.
function walk(input, start, distance, direction, loop, available, traversal) {
  const events = [];
  let time = start, crossings = 0, finished = false, dir = direction;
  const emit = (from, to, includeStart = false) => {
    const sorted = input.map((v, index) => ({ ...v, index })).sort((a, b) =>
      dir * (a.time - b.time) || a.index - b.index);
    for (const item of sorted) if ((includeStart ? dir * (item.time - from) >= 0 : dir * (item.time - from) > 0) &&
        dir * (to - item.time) >= 0)
      events.push({ type: "marker", action, name: item.name, time: item.time, direction: dir, traversal: traversal + crossings });
  };
  while (distance > 0) {
    const edge = dir > 0 ? 1 : 0, remaining = Math.abs(edge - time);
    if (distance < remaining) { const next = time + dir * distance; emit(time, next); time = next; break; }
    emit(time, edge); time = edge; distance -= remaining; crossings++;
    if (crossings === available) { finished = true; break; }
    if (loop === "pingpong") dir = -dir;
    else { time = dir > 0 ? 0 : 1; emit(time, time, true); }
  }
  return { events, move: { start, end: time, direction, loop, crossings, finished, traversal } };
}

test("12000 deterministic randomized paths agree with independent walking oracle", () => {
  let seed = 0x73637470;
  const rand = (n) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
  for (let i = 0; i < 12000; i++) {
    const loop = ["repeat", "pingpong", "once"][rand(3)];
    const input = Array.from({ length: 1 + rand(8) }, (_, n) => ({ name: `cue${n}`, time: rand(9) / 8 }));
    const reference = walk(input, rand(9) / 8, rand(257) / 8, rand(2) ? 1 : -1,
      loop, loop === "once" ? 1 : rand(2) ? Infinity : 1 + rand(16), rand(100));
    const actual = collect(input, reference.move);
    assert.deepEqual(actual, reference.events, JSON.stringify({ i, input, ...reference.move }));
    if (actual.length) assert.equal(collect(input, reference.move, { maxEvents: actual.length }).length, actual.length);
    if (actual.length > 1) assert.throws(() => collect(input, reference.move, { maxEvents: actual.length - 1 }), error("LIMIT"));
  }
});
