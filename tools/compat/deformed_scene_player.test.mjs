import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Execute the real player with controllable mixer and scene-submission boundaries.
// Scene preparation/capture are covered separately in deformed_scene_adapter.test.
const dataUrl = (s) => "data:text/javascript;base64," + Buffer.from(s).toString("base64");
const dependencyUrl = dataUrl(`
export async function renderDeformedScene(...args) { return args[1].submit(...args); }
export function prepareDeformedScene() { throw new Error('unused boundary'); }
`);
const source = (await readFile(new URL("./deformed_scene_player.mjs", import.meta.url), "utf8"))
  .replaceAll('"./deformed_scene_adapter.mjs"', JSON.stringify(dependencyUrl));
const { createDeformedScenePlayer: create } = await import(dataUrl(source));
const flush = async () => { for (let i = 0; i < 16; i++) await Promise.resolve(); };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
function clock() {
  let next = 0;
  const pending = new Map(), canceled = [];
  return {
    pending, canceled,
    requestFrame(callback) { const id = next++; pending.set(id, callback); return id; },
    cancelFrame(id) { canceled.push(id); pending.delete(id); },
    fire(time) { const [id, callback] = pending.entries().next().value ?? []; assert.equal(typeof callback, "function"); pending.delete(id); callback(time); return callback; },
  };
}
function setup(overrides = {}) {
  const raf = clock();
  const state = { draws: [], updates: [], seeks: [], frames: [], errors: [] };
  const scene = { position: 0, async submit(host, source, camera, context, wasm, options) {
    state.draws.push({ position: source.position, context, options, width: context?.canvas?.width });
    if (state.failure) throw state.failure;
    if (state.gate) await state.gate.promise;
    return state.result ?? { admitted: ["mesh"], refused: [], pose: source.position };
  } };
  const mixer = {
    time: 0,
    update(delta) { state.updates.push(delta); this.time += delta; scene.position = this.time; },
    setTime(value) { state.seeks.push(value); this.time = value; scene.position = value; },
  };
  const config = {
    bridgeHost: { executePacket() {} }, scene, camera: { isCamera: true }, wasmModule: {}, mixers: [mixer],
    requestFrame: raf.requestFrame, cancelFrame: raf.cancelFrame,
    onFrame: (result) => { state.frames.push(result); }, onError: (error) => { state.errors.push(error); },
    ...overrides,
  };
  return { player: create(config), config, scene, mixer, state, raf };
}
const code = (value) => (error) => error.code === value;

test("seek and step update real mixer-bound scene values before submission", async () => {
  const { player, state, mixer } = setup();
  await player.seek(2); await player.step(0.5); await player.render();
  assert.deepEqual(state.seeks, [2]); assert.deepEqual(state.updates, [0.5]);
  assert.deepEqual(state.draws.map((draw) => draw.position), [2, 2.5, 2.5]);
  assert.equal(player.time, 2.5); assert.equal(mixer.time, 2.5); assert.equal(player.isPlaying, false);
});

test("RAF milliseconds become frame-rate-independent seconds with no startup jump", async () => {
  const { player, state, raf } = setup(); player.play();
  raf.fire(1000); await flush(); raf.fire(1250); await flush(); raf.fire(2000); await flush();
  assert.deepEqual(state.updates, [0, 0.25, 0.75]); assert.equal(player.time, 1);
  assert.equal(raf.pending.size, 1); player.pause();
});

test("pause is quiet and resume excludes elapsed paused wall time", async () => {
  const { player, state, raf } = setup(); player.play(); raf.fire(0); await flush();
  raf.fire(100); await flush(); player.pause();
  const count = state.draws.length; await flush();
  assert.equal(raf.pending.size, 0); assert.equal(state.draws.length, count);
  player.play(); raf.fire(50_000); await flush(); raf.fire(50_100); await flush();
  assert.deepEqual(state.updates, [0, 0.1, 0, 0.1]); player.pause();
});

test("reverse rates and explicit seeks feed the existing mixer instead of custom interpolation", async () => {
  const { player, state, mixer } = setup(); await player.seek(3);
  player.playbackRate = -2; await player.step(0.5);
  assert.equal(mixer.time, 2); assert.equal(player.time, 2);
  player.playbackRate = 0; await player.step(8); assert.equal(player.time, 2);
  assert.deepEqual(state.updates, [-1, 0]);
});

test("GPU backpressure prevents overlapping draws and retains elapsed animation time", async () => {
  const { player, state, raf } = setup(); state.gate = deferred();
  player.play(); raf.fire(0); await flush();
  assert.equal(state.draws.length, 1); assert.equal(raf.pending.size, 0); assert.equal(player.isRendering, true);
  assert.throws(() => player.step(1), code("SCENE_PLAYER_BUSY"));
  player.invalidate(); player.invalidate(); assert.equal(raf.pending.size, 0);
  state.gate.resolve(); await player.whenIdle(); await flush();
  assert.equal(raf.pending.size, 1); state.gate = null;
  raf.fire(2000); await flush(); assert.deepEqual(state.updates, [0, 2]); player.pause();
});

test("paused invalidations coalesce to one draw without advancing mixers", async () => {
  const { player, state, raf } = setup(); player.invalidate(); player.invalidate(); player.invalidate();
  assert.equal(raf.pending.size, 1); raf.fire(100); await flush();
  assert.equal(state.draws.length, 1); assert.deepEqual(state.updates, []); assert.equal(raf.pending.size, 0);
});

test("canceled RAF tickets cannot render after pause or restart", async () => {
  const { player, state, raf } = setup(); player.play();
  const stale = raf.pending.values().next().value; player.pause(); player.play();
  stale(100); await flush(); assert.equal(state.draws.length, 0); assert.equal(raf.pending.size, 1);
  raf.fire(200); await flush(); assert.equal(state.draws.length, 1); player.pause();
});

test("scheduled renderer failures halt and report exactly once", async () => {
  const { player, state, raf } = setup(); state.failure = new Error("device lost");
  player.play(); raf.fire(0); await flush();
  assert.equal(player.isPlaying, false); assert.equal(raf.pending.size, 0);
  assert.equal(player.lastError, state.failure); assert.deepEqual(state.errors, [state.failure]);
  assert.equal(state.frames.length, 0);
});

test("admission refusals halt instead of spinning a blank-frame loop", async () => {
  const { player, state, raf } = setup(); state.result = { admitted: [], refused: [{ uuid: "mesh", reason: "UNSUPPORTED_MATERIAL" }] };
  player.play(); raf.fire(0); await flush();
  assert.equal(player.lastError.code, "SCENE_PLAYER_ADMISSION");
  assert.equal(player.lastError.result, state.result); assert.equal(player.isPlaying, false);
  assert.equal(state.errors.length, 1); assert.equal(raf.pending.size, 0);
});

test("manual failures reject to the caller and permit an explicit retry", async () => {
  const { player, state } = setup(); const failure = new Error("native failed"); state.failure = failure;
  await assert.rejects(player.render(), (error) => error === failure);
  assert.equal(player.isRendering, false); assert.equal(state.errors.length, 0);
  state.failure = null; await player.render(); assert.equal(state.frames.length, 1);
});

test("async onFrame is part of backpressure", async () => {
  const gate = deferred(); const { player, state, raf } = setup({ onFrame: () => gate.promise });
  player.play(); raf.fire(0); await flush();
  assert.equal(state.draws.length, 1); assert.equal(player.isRendering, true); assert.equal(raf.pending.size, 0);
  gate.resolve(); await player.whenIdle(); await flush(); assert.equal(raf.pending.size, 1); player.pause();
});

test("dispose during submission suppresses callbacks and preserves borrowed resources", async () => {
  const { player, state, raf, config, mixer } = setup(); state.gate = deferred(); let disposed = 0;
  config.bridgeHost.dispose = mixer.dispose = () => { disposed++; };
  player.play(); raf.fire(0); await flush(); const pending = player.whenIdle(); player.dispose();
  state.gate.resolve(); await pending; await flush();
  assert.equal(disposed, 0); assert.equal(state.frames.length, 0); assert.equal(state.errors.length, 0);
  assert.equal(raf.pending.size, 0); assert.equal(player.isDisposed, true); player.dispose();
  assert.throws(() => player.play(), code("SCENE_PLAYER_DISPOSED"));
});

test("disposal before the submission microtask performs no mixer or render work", async () => {
  const { player, state, raf } = setup(); player.play(); raf.fire(0); player.dispose(); await flush();
  assert.equal(state.draws.length, 0); assert.equal(state.updates.length, 0); assert.equal(state.errors.length, 0);
});

test("each player has an independent clock and scene", async () => {
  const a = setup(), b = setup(); await a.player.seek(1); await b.player.seek(5);
  await a.player.step(0.25); await b.player.step(1);
  assert.equal(a.scene.position, 1.25); assert.equal(b.scene.position, 6);
  assert.equal(a.player.time, 1.25); assert.equal(b.player.time, 6);
});

test("duplicate mixers update once and caller array changes cannot change ownership", async () => {
  const { config, mixer, state } = setup(); const mixers = [mixer, mixer];
  const player = create({ ...config, mixers }); mixers.push({ update() { throw new Error("not owned"); } });
  await player.step(1); await player.seek(3);
  assert.deepEqual(state.updates, [1]); assert.deepEqual(state.seeks, [3]);
});

test("manual/offscreen operation needs no browser scheduler and resize is read per frame", async () => {
  const context = { canvas: { width: 64, height: 32 } };
  const { player, state } = setup({ requestFrame: null, cancelFrame: null, canvasContext: context });
  await player.render(); context.canvas.width = 320; await player.render();
  assert.deepEqual(state.draws.map((draw) => draw.width), [64, 320]);
  assert.throws(() => player.play(), code("SCENE_PLAYER_SCHEDULER")); assert.equal(player.isPlaying, false);
});

test("finite delta caps are opt-in and invalid timestamps halt cleanly", async () => {
  const { player, state, raf } = setup({ maxDeltaSeconds: 0.25 });
  player.play(); raf.fire(100); await flush(); raf.fire(10_100); await flush();
  assert.deepEqual(state.updates, [0, 0.25]); raf.fire(10_000); await flush();
  assert.equal(player.lastError.code, "SCENE_PLAYER_TIMESTAMP"); assert.equal(raf.pending.size, 0);
});

test("invalid inputs and time overflow fail before render submission", async () => {
  const { player, config, state } = setup();
  assert.throws(() => create({ ...config, playbackRate: NaN }), code("SCENE_PLAYER_RATE"));
  assert.throws(() => create({ ...config, mixers: [{}] }), code("SCENE_PLAYER_MIXER"));
  assert.throws(() => create({ ...config, maxDeltaSeconds: 0 }), code("SCENE_PLAYER_DELTA"));
  assert.throws(() => player.seek(-1), code("SCENE_PLAYER_TIME"));
  assert.throws(() => { player.playbackRate = Infinity; }, code("SCENE_PLAYER_RATE"));
  player.playbackRate = Number.MAX_VALUE;
  await assert.rejects(player.step(2), code("SCENE_PLAYER_DELTA")); assert.equal(state.draws.length, 0);
});

test("scheduler and error-callback failures cannot leave active scheduling behind", async () => {
  const error = new Error("scheduler"); const { player } = setup({ requestFrame() { throw error; } });
  assert.throws(() => player.play(), (e) => e === error); assert.equal(player.isPlaying, false);
  const callbackError = new Error("error callback");
  const other = setup({ onError() { throw callbackError; } }); other.state.failure = error;
  other.player.play(); other.raf.fire(0); await flush();
  assert.ok(other.player.lastError instanceof AggregateError);
  assert.deepEqual(other.player.lastError.errors, [error, callbackError]); assert.equal(other.raf.pending.size, 0);
});

test("pause then whenIdle enables safe scrubbing without competing pose writes", async () => {
  const { player, state, raf } = setup(); state.gate = deferred(); player.play(); raf.fire(0); await flush();
  player.pause(); assert.throws(() => player.seek(3), code("SCENE_PLAYER_BUSY"));
  const waiting = player.whenIdle(); state.gate.resolve(); await waiting; state.gate = null;
  await player.seek(3); assert.equal(state.draws.at(-1).position, 3); assert.equal(raf.pending.size, 0);
});
