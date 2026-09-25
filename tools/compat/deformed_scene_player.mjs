/** AnimationMixer -> live scene pose -> Rust deformation -> native scene draw. */
import { renderDeformedScene } from "./deformed_scene_adapter.mjs";
export { prepareDeformedScene, renderDeformedScene } from "./deformed_scene_adapter.mjs";

function requireThat(condition, code, detail) {
  if (!condition) {
    const error = new Error(`${code}: ${detail}`);
    error.code = code;
    throw error;
  }
}

/**
 * Create an opt-in player for the existing opaque MeshBasicMaterial native slice.
 * Supply already-configured AnimationMixers (e.g. mixer.clipAction(clip).play()).
 * Their actions, interpolation, loops and blending stay owned by Three; this
 * player supplies seconds, updates the scene/palettes and submits actual draws.
 *
 * play()/pause() control an RAF clock. Paused players draw only on invalidate()
 * or explicit render()/step()/seek(); there is no idle polling. The first frame
 * after play/resume samples delta zero, so a paused interval never jumps the pose.
 * playbackRate may be negative. maxDeltaSeconds defaults to Infinity (no elapsed
 * time discarded under GPU backpressure); choose a finite cap explicitly to
 * suppress long background-tab gaps. Mixer.timeScale still has its own semantics.
 *
 * render(), step(seconds) and seek(seconds) require paused, idle state. After
 * pause(), await whenIdle() before seeking if a GPU submission was in progress.
 * Manual operations return their renderer result/rejection. Scheduled failures
 * halt playback and call onError, if provided; lastError always records failure.
 * onFrame may be async and is included in backpressure. No two frames overlap.
 *
 * Scene, camera, mixers, host, Wasm and canvas are borrowed. dispose() cancels
 * scheduling but neither frees borrowed resources nor cancels a submitted GPU
 * command. No callback or replacement frame is emitted after disposal. Context
 * dimensions are read by the scene renderer on each draw, including after resize.
 * RAF functions may be injected for another host; they must be asynchronous and
 * use millisecond timestamps, like requestAnimationFrame. Manual mode needs none.
 *
 * Example:
 *   const player = createDeformedScenePlayer({ bridgeHost, scene, camera,
 *     wasmModule, canvasContext, mixers: [mixer], onError: console.error });
 *   player.play();
 *   // Scrub: player.pause(); await player.whenIdle(); await player.seek(1.5);
 */
export function createDeformedScenePlayer({
  bridgeHost, scene, camera, wasmModule, canvasContext = null,
  mixers = [], renderOptions = {}, playbackRate = 1, maxDeltaSeconds = Infinity,
  requestFrame = typeof globalThis.requestAnimationFrame === "function"
    ? globalThis.requestAnimationFrame.bind(globalThis) : null,
  cancelFrame = typeof globalThis.cancelAnimationFrame === "function"
    ? globalThis.cancelAnimationFrame.bind(globalThis) : null,
  onFrame = null, onError = null,
} = {}) {
  requireThat(typeof bridgeHost?.executePacket === "function" && scene && camera?.isCamera,
    "SCENE_PLAYER_INPUT", "a scene, camera and initialized bridge host are required");
  requireThat(Array.isArray(mixers) && mixers.every((m) => typeof m?.update === "function" && typeof m?.setTime === "function"),
    "SCENE_PLAYER_MIXER", "mixers must implement update(seconds) and setTime(seconds)");
  requireThat(Number.isFinite(playbackRate), "SCENE_PLAYER_RATE", "playbackRate must be finite");
  requireThat(maxDeltaSeconds > 0 && (Number.isFinite(maxDeltaSeconds) || maxDeltaSeconds === Infinity),
    "SCENE_PLAYER_DELTA", "maxDeltaSeconds must be positive");
  for (const callback of [onFrame, onError])
    requireThat(callback === null || typeof callback === "function", "SCENE_PLAYER_CALLBACK", "callbacks must be functions");
  const activeMixers = [...new Set(mixers)];
  const options = { ...renderOptions };
  let playing = false, disposed = false, busy = false, dirty = false;
  let scheduled = null, generation = 0, lastTimestamp = null;
  let time = 0, rate = playbackRate, lastError = null, inFlight = null;

  function alive() {
    requireThat(!disposed, "SCENE_PLAYER_DISPOSED", "player has been disposed");
  }
  function manual() {
    alive();
    requireThat(!playing && !busy, "SCENE_PLAYER_BUSY", "pause and await whenIdle() before a manual operation");
  }
  function cancelScheduled() {
    const ticket = scheduled;
    scheduled = null;
    if (ticket !== null) cancelFrame(ticket.id);
  }
  function halt() {
    playing = false;
    dirty = false;
    lastTimestamp = null;
    generation++;
    cancelScheduled();
  }
  function scheduler() {
    requireThat(typeof requestFrame === "function" && typeof cancelFrame === "function",
      "SCENE_PLAYER_SCHEDULER", "an asynchronous requestFrame/cancelFrame pair is required");
  }
  async function report(error) {
    if (disposed || onError === null) return;
    try { await onError(error); }
    catch (callbackError) { lastError = new AggregateError([error, callbackError], "scene player and onError failed"); }
  }
  function advance(delta) {
    const scaled = delta * rate;
    requireThat(Number.isFinite(scaled) && Number.isFinite(time + scaled), "SCENE_PLAYER_DELTA", "animation time overflow");
    for (const mixer of activeMixers) mixer.update(scaled);
    time += scaled;
  }
  function schedule() {
    if (disposed || busy || scheduled !== null || (!playing && !dirty)) return;
    scheduler();
    const ticket = { generation, id: null };
    scheduled = ticket;
    try {
      ticket.id = requestFrame((timestamp) => {
        if (scheduled !== ticket || ticket.generation !== generation || disposed) return;
        scheduled = null;
        let delta = 0;
        try {
          requireThat(Number.isFinite(timestamp), "SCENE_PLAYER_TIMESTAMP", "RAF timestamp must be finite milliseconds");
          if (playing) {
            requireThat(lastTimestamp === null || timestamp >= lastTimestamp, "SCENE_PLAYER_TIMESTAMP", "RAF clock moved backwards");
            if (lastTimestamp !== null) delta = Math.min((timestamp - lastTimestamp) / 1000, maxDeltaSeconds);
            lastTimestamp = timestamp;
          }
          const animate = playing;
          dirty = false;
          void submit(animate ? () => advance(delta) : null).catch(report);
        } catch (error) {
          lastError = error;
          halt();
          void report(error);
        }
      });
    } catch (error) {
      scheduled = null;
      throw error;
    }
  }
  function submit(update) {
    busy = true;
    // Enter update/render in a microtask so inFlight exists before mixer events
    // can re-enter pause()/whenIdle(). A disposed queued frame does no work.
    const operation = Promise.resolve().then(async () => {
      alive();
      if (update) update();
      alive();
      const result = await renderDeformedScene(bridgeHost, scene, camera, canvasContext, wasmModule, options);
      if (result?.refused?.length) {
        const error = new Error("SCENE_PLAYER_ADMISSION: scene renderer refused visible content");
        error.code = "SCENE_PLAYER_ADMISSION";
        error.result = result;
        throw error;
      }
      if (!disposed && onFrame !== null) await onFrame(result);
      return result;
    });
    inFlight = operation.then((result) => {
      busy = false;
      inFlight = null;
      // Scheduling errors also halt the loop instead of leaving isPlaying true.
      try { schedule(); }
      catch (error) { lastError = error; halt(); throw error; }
      return result;
    }, (error) => {
      busy = false;
      inFlight = null;
      lastError = error;
      halt();
      throw error;
    });
    return inFlight;
  }

  const player = {
    get isPlaying() { return playing; },
    get isDisposed() { return disposed; },
    get isRendering() { return busy; },
    get time() { return time; },
    get lastError() { return lastError; },
    get playbackRate() { return rate; },
    set playbackRate(value) {
      alive();
      requireThat(Number.isFinite(value), "SCENE_PLAYER_RATE", "playbackRate must be finite");
      rate = value;
    },
    play() {
      alive(); scheduler();
      if (!playing) {
        playing = true; generation++; lastTimestamp = null;
        cancelScheduled();
        try { schedule(); }
        catch (error) { lastError = error; halt(); throw error; }
      }
      return player;
    },
    pause() { alive(); halt(); return player; },
    invalidate() {
      alive(); scheduler(); dirty = true;
      try { schedule(); }
      catch (error) { lastError = error; halt(); throw error; }
      return player;
    },
    render() {
      manual(); cancelScheduled(); dirty = false;
      return submit(null);
    },
    step(deltaSeconds) {
      manual();
      requireThat(Number.isFinite(deltaSeconds), "SCENE_PLAYER_DELTA", "step must be finite seconds");
      cancelScheduled(); dirty = false;
      return submit(() => advance(deltaSeconds));
    },
    seek(seconds) {
      manual();
      requireThat(Number.isFinite(seconds) && seconds >= 0, "SCENE_PLAYER_TIME", "seek must be nonnegative finite seconds");
      cancelScheduled(); dirty = false;
      return submit(() => {
        for (const mixer of activeMixers) mixer.setTime(seconds);
        time = seconds;
      });
    },
    whenIdle() { return inFlight ?? Promise.resolve(); },
    dispose() {
      if (!disposed) { halt(); disposed = true; }
    },
  };
  return player;
}
