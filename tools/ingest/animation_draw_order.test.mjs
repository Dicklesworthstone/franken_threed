import assert from "node:assert/strict";
import test from "node:test";
import { createAnimationDrawOrder } from "./animation_draw_order.mjs";

const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function entry(name, z, alphaMode = "BLEND") {
  const worldMatrix = identity();
  worldMatrix[14] = z;
  return { mesh: { name }, deformer: { worldMatrix }, alphaMode };
}
const names = (draws) => draws.map((mesh) => mesh.name);
const code = (error) => error.code === "ANIMATION_SCENE_SORT";

test("opaque and mask retain source order before far-to-near alpha draws", () => {
  const entries = [
    entry("near", 0.2),
    entry("opaque-a", 0.9, "OPAQUE"),
    entry("far", 0.8),
    entry("mask", 0.1, "MASK"),
    entry("opaque-b", 0.9, "OPAQUE"),
    entry("middle", 0.5),
  ];
  const before = entries.slice(),
    sorter = createAnimationDrawOrder(entries);
  assert.deepEqual(names(sorter.order(identity())), [
    "opaque-a",
    "mask",
    "opaque-b",
    "far",
    "middle",
    "near",
  ]);
  assert.deepEqual(entries, before);
  assert.equal(sorter.order(identity())[0], entries[1].mesh);
});

test("perspective depth uses homogeneous division rather than clip z alone", () => {
  const camera = identity();
  camera[10] = -10 / 9;
  camera[14] = -10 / 9;
  camera[11] = -1;
  camera[15] = 0;
  const entries = [entry("near", -2), entry("far", -9), entry("middle", -5)];
  assert.deepEqual(names(createAnimationDrawOrder(entries).order(camera)), [
    "far",
    "middle",
    "near",
  ]);
});

test("sort uses current uploaded transforms on every call, not initial pose", () => {
  const entries = [entry("a", 0.2), entry("b", 0.8)],
    sorter = createAnimationDrawOrder(entries);
  assert.deepEqual(names(sorter.order(identity())), ["b", "a"]);
  entries[0].deformer.worldMatrix[14] = 0.9;
  assert.deepEqual(names(sorter.order(identity())), ["a", "b"]);
});

test("each camera recomputes ordering without mutating matrices", () => {
  const entries = [entry("a", 0.2), entry("b", 0.8)],
    sorter = createAnimationDrawOrder(entries);
  const forward = identity(),
    reverse = identity();
  reverse[10] = -1;
  reverse[14] = 1;
  const copied = reverse.slice();
  assert.deepEqual(names(sorter.order(forward)), ["b", "a"]);
  assert.deepEqual(names(sorter.order(reverse)), ["a", "b"]);
  assert.deepEqual(names(sorter.order(forward)), ["b", "a"]);
  assert.deepEqual(reverse, copied);
});

test("view rotation includes world X/Y in the sort key", () => {
  const entries = [entry("a", 0.5), entry("b", 0.5)];
  entries[0].deformer.worldMatrix[12] = 3;
  entries[1].deformer.worldMatrix[13] = 7;
  const camera = identity();
  camera[2] = 1;
  camera[6] = 1;
  assert.deepEqual(names(createAnimationDrawOrder(entries).order(camera)), ["b", "a"]);
});

test("ties return to immutable insertion order after depths cross", () => {
  const entries = [entry("a", 0.8), entry("b", 0.2), entry("c", 0.8)],
    sorter = createAnimationDrawOrder(entries);
  assert.deepEqual(names(sorter.order(identity())), ["a", "c", "b"]);
  entries.forEach((item) => {
    item.deformer.worldMatrix[14] = 0.5;
  });
  assert.deepEqual(names(sorter.order(identity())), ["a", "b", "c"]);
});

test("registration snapshots alpha classification and input list, not pose state", () => {
  const a = entry("a", 0.1),
    b = entry("b", 0.9),
    entries = [a, b],
    sorter = createAnimationDrawOrder(entries);
  a.alphaMode = "OPAQUE";
  entries.reverse();
  entries.pop();
  assert.deepEqual(names(sorter.order(identity())), ["b", "a"]);
});

test("centers on or behind the camera plane are retained deterministically", () => {
  const camera = identity();
  camera[11] = -1;
  camera[15] = 0;
  const entries = [entry("eye", 0), entry("behind", 1), entry("front", -1)];
  const sorter = createAnimationDrawOrder(entries),
    first = sorter.order(camera);
  assert.equal(first.length, 3);
  assert.deepEqual(first, sorter.order(camera));
  assert.equal(new Set(first).size, 3);
  camera[14] = 1;
  assert.equal(sorter.order(camera)[0], entries[0].mesh);
});

test("sort validation fails without changing future valid ordering", () => {
  const entries = [entry("a", 0.1), entry("b", 0.9)],
    sorter = createAnimationDrawOrder(entries);
  for (const bad of [
    null,
    [],
    new Float64Array(15),
    [...identity().slice(0, 15), NaN],
    identity().map(String),
  ]) {
    assert.throws(() => sorter.order(bad), code);
  }
  entries[0].deformer.worldMatrix[3] = 1;
  assert.throws(() => sorter.order(identity()), code);
  entries[0].deformer.worldMatrix[3] = 0;
  assert.deepEqual(names(sorter.order(identity())), ["b", "a"]);
});

test("fixed typed matrices work and shared or detached storage is rejected", () => {
  const item = entry("typed", 0.5);
  item.deformer.worldMatrix = Float32Array.from(item.deformer.worldMatrix);
  const sorter = createAnimationDrawOrder([item]);
  assert.equal(sorter.order(Float64Array.from(identity()))[0], item.mesh);
  assert.throws(() => sorter.order(new Float64Array(new SharedArrayBuffer(128))), code);
  const detached = Float64Array.from(identity());
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  assert.throws(() => sorter.order(detached), code);
});

test("empty and opaque-only scenes do not require a sorting camera or share output arrays", () => {
  assert.deepEqual(createAnimationDrawOrder([]).order(null), []);
  const sorter = createAnimationDrawOrder([entry("opaque", 0, "OPAQUE")]);
  const a = sorter.order(null),
    b = sorter.order(null);
  assert.notEqual(a, b);
  a.length = 0;
  assert.equal(b.length, 1);
});

test("invalid registrations fail before they can become scene draws", () => {
  for (const entries of [null, new Array(4097).fill({}), [{}], [entry("x", 0, "UNKNOWN")]]) {
    assert.throws(() => createAnimationDrawOrder(entries), code);
  }
});
