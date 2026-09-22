/**
 * Device-backed scene lifetime for WebGpuBridgeHost.
 *
 * The caller owns scheduling (there is no RAF/timer loop here). Rust/Wasm owns
 * scene compilation and packet production; this module only owns browser host
 * resources, cancellation, and publication of the latest scene generation.
 *
 * A scene factory receives {width, height, capabilities, signal, generation}
 * and returns {packet: Uint8Array, frame?: fn, dispose?: fn}. `frame` receives
 * the same context plus `time` and returns a packet, or null for no GPU work.
 * Packets must be owned bytes, not borrowed Wasm memory or shared buffers.
 * `dispose` must release CPU-side scene ownership, not submit more GPU work.
 */
export class WebGpuSceneSession {
  constructor({
    host,
    canvas,
    requiredProfile = {},
    maxRecoveryAttempts = 1,
    onState = () => {},
    onError = () => {},
  }) {
    for (const method of [
      "negotiateAndCreateDevice",
      "executePacket",
      "clearDeviceResources",
      "destroyDevice",
    ]) {
      if (typeof host?.[method] !== "function") throw new TypeError(`host.${method} is required`);
    }
    if (!Number.isSafeInteger(maxRecoveryAttempts) || maxRecoveryAttempts < 0) {
      throw new RangeError("maxRecoveryAttempts must be a non-negative safe integer");
    }
    if (typeof onState !== "function" || typeof onError !== "function") {
      throw new TypeError("onState and onError must be functions");
    }
    this.context = canvas?.getContext?.("webgpu");
    if (!this.context) throw new Error("A WebGPU canvas context is required");
    this.host = host;
    this.canvas = canvas;
    this.profile = requiredProfile;
    this.maxRecoveryAttempts = maxRecoveryAttempts;
    this.onState = onState;
    this.onError = onError;
    this.state = "idle";
    this.generation = 0;
    this.submittedFrames = 0;
    this.recoveryAttempts = 0;
    this.lastError = null;
    this.factory = null;
    this.size = null;
    this.scene = null;
    this.controller = null;
    this.observedDevice = null;
    this.tail = Promise.resolve();
    this.disposed = new WeakMap();
    this.disposals = new Set();
    this.closePromise = null;
  }

  snapshot() {
    return Object.freeze({
      state: this.state,
      generation: this.generation,
      width: this.size?.width ?? this.canvas.width,
      height: this.size?.height ?? this.canvas.height,
      submittedFrames: this.submittedFrames,
      recoveryAttempts: this.recoveryAttempts,
      lastError: this.lastError,
    });
  }

  load(factory, { width = this.canvas.width, height = this.canvas.height } = {}) {
    this.assertOpen();
    if (typeof factory !== "function") throw new TypeError("A scene factory is required");
    this.validateSize(width, height);
    return this.replace(factory, { width, height }, "loading", true);
  }

  reload() {
    this.assertOpen();
    if (!this.factory) throw new Error("No scene has been loaded");
    return this.replace(this.factory, this.size, "loading", true);
  }

  resize(width, height) {
    this.assertOpen();
    this.validateSize(width, height);
    if (!this.factory) throw new Error("Load a scene before resizing it");
    if (this.state === "ready" && width === this.size.width && height === this.size.height) {
      return Promise.resolve(this.snapshot());
    }
    return this.replace(this.factory, { width, height }, "loading", true);
  }

  /** Submit one caller-scheduled frame. Concurrent calls never interleave host work. */
  render(time = 0) {
    this.assertOpen();
    if (!Number.isFinite(time)) throw new TypeError("Frame time must be finite");
    if (this.state !== "ready") throw new Error("The scene is not ready");
    const token = this.token();
    return this.enqueue(async () => {
      this.check(token);
      const scene = this.scene;
      const frame = scene?.frame;
      if (!frame) return this.snapshot();
      try {
        const packet = await waitFor(
          Promise.resolve().then(() => {
            this.check(token);
            return frame.call(scene, { ...this.sceneContext(token), time });
          }),
          token.signal,
        );
        this.check(token);
        if (packet != null) {
          validatePacket(packet);
          await waitFor(this.host.executePacket(packet, this.context), token.signal);
          this.check(token);
          this.submittedFrames++;
        }
        return this.snapshot();
      } catch (error) {
        this.failCurrent(token, error);
        throw error;
      }
    });
  }

  /** Cancel immediately, then join known CPU ownership cleanup. Late factories
   * are disposed when they settle; an uncooperative producer cannot block close. */
  close() {
    if (this.closePromise) return this.closePromise;
    this.generation++;
    this.state = "closed";
    this.observedDevice = null;
    // Publish the close promise before dispatching abort or observer callbacks:
    // those callbacks may synchronously request close again.
    this.closePromise = this.enqueue(async () => {
      const scene = this.scene;
      this.scene = null;
      await this.dispose(scene);
      while (this.disposals.size) await Promise.all([...this.disposals]);
      return this.snapshot();
    });
    this.controller?.abort();
    try {
      this.host.destroyDevice();
    } catch (error) {
      this.report(error);
    }
    try {
      this.context.unconfigure();
    } catch (error) {
      this.report(error);
    }
    this.emit();
    return this.closePromise;
  }

  replace(factory, size, state, resetRecovery) {
    const previousController = this.controller;
    this.controller = new AbortController();
    this.generation++;
    this.factory = factory;
    this.size = { ...size };
    this.state = state;
    this.lastError = null;
    if (resetRecovery) this.recoveryAttempts = 0;
    const token = this.token();
    // Install the new generation before abort listeners run. A listener may
    // close or replace this operation, and that newer decision must win.
    previousController?.abort();
    this.emit();
    return this.enqueue(async () => {
      let candidate = null;
      try {
        this.check(token);
        const previous = this.scene;
        this.scene = null;
        const disposal = this.dispose(previous);
        // GPU completion, unlike executePacket's validation receipt, permits
        // destruction of the preceding scene's resident buffers and textures.
        const completion = this.host.device?.queue.onSubmittedWorkDone();
        if (completion) await waitFor(completion, token.signal);
        this.check(token);
        this.clearResidency();
        await waitFor(disposal, token.signal);
        this.check(token);
        if (!this.host.device) {
          await waitFor(this.host.negotiateAndCreateDevice(this.profile), token.signal, () =>
            this.host.destroyDevice(),
          );
          this.check(token);
        }
        this.watchDevice();
        this.validateSize(size.width, size.height);
        const build = Promise.resolve().then(() => {
          this.check(token);
          return factory(this.sceneContext(token));
        });
        candidate = await waitFor(build, token.signal, null, (late) => this.dispose(late));
        this.check(token);
        if (!candidate || typeof candidate !== "object")
          throw new TypeError("Scene factory must return a scene object");
        if (candidate.frame !== undefined && typeof candidate.frame !== "function")
          throw new TypeError("scene.frame must be a function");
        if (candidate.dispose !== undefined && typeof candidate.dispose !== "function")
          throw new TypeError("scene.dispose must be a function");
        validatePacket(candidate.packet);
        this.canvas.width = size.width;
        this.canvas.height = size.height;
        this.context.configure({
          device: this.host.device,
          format: this.host.capabilityRecord.preferredCanvasFormat,
          alphaMode: "premultiplied",
        });
        await waitFor(this.host.executePacket(candidate.packet, this.context), token.signal);
        this.check(token);
        this.scene = candidate;
        candidate = null;
        this.submittedFrames++;
        this.state = "ready";
        this.emit();
        return this.snapshot();
      } catch (error) {
        await this.dispose(candidate);
        this.failCurrent(token, error);
        throw error;
      }
    });
  }

  watchDevice() {
    const device = this.host.device;
    if (device === this.observedDevice) return;
    this.observedDevice = device;
    device.lost
      .then((info) => {
        if (this.state === "closed" || this.observedDevice !== device) return;
        this.observedDevice = null;
        const error = new Error(
          `WebGPU device lost: ${info?.message || info?.reason || "unknown reason"}`,
        );
        const generation = this.generation;
        this.report(error);
        if (this.state === "closed" || this.generation !== generation) return;
        if (this.recoveryAttempts >= this.maxRecoveryAttempts || !this.factory) {
          this.generation++;
          this.state = "failed";
          this.controller?.abort();
          this.emit();
          return;
        }
        this.recoveryAttempts++;
        // Recompile/rebuild for negotiated successor capabilities; never replay
        // old numeric resource IDs or assume a replacement device is identical.
        this.replace(this.factory, this.size, "recovering", false).catch(() => {});
      })
      .catch((error) => this.report(error));
  }

  clearResidency() {
    const resources = new Set();
    for (const value of this.host.buffers?.values() ?? []) resources.add(value.buffer ?? value);
    for (const value of this.host.textures?.values() ?? []) resources.add(value.texture ?? value);
    for (const resource of resources) {
      try {
        resource.destroy?.();
      } catch (error) {
        this.report(error);
      }
    }
    this.host.clearDeviceResources();
  }

  async dispose(scene) {
    if (!scene || typeof scene !== "object") return;
    if (this.disposed.has(scene)) return this.disposed.get(scene);
    const cleanup = Promise.resolve()
      .then(() => {
        if (typeof scene.dispose === "function") return scene.dispose();
      })
      .catch((error) => this.report(error));
    this.disposed.set(scene, cleanup);
    this.disposals.add(cleanup);
    cleanup.then(() => this.disposals.delete(cleanup));
    return cleanup;
  }

  validateSize(width, height) {
    const limit = this.host.device?.limits.maxTextureDimension2D ?? 0xffffffff;
    for (const [name, value] of Object.entries({ width, height })) {
      if (!Number.isSafeInteger(value) || value < 1 || value > limit) {
        throw new RangeError(`${name} must be a positive integer no greater than ${limit}`);
      }
    }
  }

  sceneContext(token) {
    return {
      ...this.size,
      capabilities: this.host.capabilityRecord,
      signal: token.signal,
      generation: token.generation,
    };
  }

  token() {
    return { generation: this.generation, signal: this.controller.signal };
  }
  check(token) {
    if (token.signal.aborted || token.generation !== this.generation || this.state === "closed")
      throw cancelled();
  }
  assertOpen() {
    if (this.state === "closed") throw new Error("Scene session is closed");
  }
  enqueue(action) {
    const result = this.tail.then(action);
    this.tail = result.catch(() => {});
    return result;
  }
  failCurrent(token, error) {
    if (!token.signal.aborted && token.generation === this.generation && this.state !== "closed") {
      this.report(error);
      if (token.signal.aborted || token.generation !== this.generation || this.state === "closed")
        return;
      this.state = "failed";
      this.controller.abort();
      this.emit();
    }
  }
  report(error) {
    this.lastError = error instanceof Error ? error.message : String(error);
    try {
      this.onError(error);
    } catch {
      /* Observers do not own the renderer. */
    }
  }
  emit() {
    try {
      this.onState(this.snapshot());
    } catch (error) {
      this.report(error);
    }
  }
}

function cancelled() {
  const error = new Error("Scene operation was superseded or the session was closed");
  error.name = "AbortError";
  return error;
}

function validatePacket(packet) {
  if (
    !(packet instanceof Uint8Array) ||
    packet.byteLength < 16 ||
    (typeof SharedArrayBuffer !== "undefined" && packet.buffer instanceof SharedArrayBuffer)
  ) {
    throw new TypeError(
      "Scene packets must be non-detached, non-shared Uint8Array values with a complete header",
    );
  }
}

/** Abandon uncooperative producers without publishing them or leaking late scenes. */
function waitFor(promise, signal, onAbort = null, onLateValue = null) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const abort = () => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", abort);
      try {
        onAbort?.();
      } catch {
        /* Cancellation must still settle. */
      }
      reject(cancelled());
    };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    Promise.resolve(promise).then(
      (value) => {
        if (settled) {
          if (onLateValue) Promise.resolve(onLateValue(value)).catch(() => {});
          return;
        }
        settled = true;
        signal.removeEventListener("abort", abort);
        resolve(value);
      },
      (error) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}
