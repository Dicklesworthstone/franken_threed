import { WebGpuBridgeHost } from "./bridge_runtime.js";

// Real device requests, rendering and loss. Packet generation stays in Rust;
// device replacement and publication guards are owned by the JS host bridge.
export async function testDeviceReplacement(buildPacket) {
  const host = new WebGpuBridgeHost();
  const devices = new Set();
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const resourceMaps = ["buffers", "textures", "pipelines", "bindGroups", "bundles", "bufferEpochs"];
  const assertEmpty = () => {
    for (const name of resourceMaps) assert(host[name].size === 0, `${name} retained old-device residency`);
  };

  async function render() {
    let observed = 0;
    const completion = host.executePacket(buildPacket(true), null, () => {
      assert(!host.errorScopeActive, "Submission observer ran inside shared device error scopes");
      assert(host.buffers.has(40) && host.buffers.has(41), "Submission observer preceded packet execution");
      observed++;
    });
    assert(observed === 1, "Submission observer did not run synchronously exactly once");
    await completion;
    const red = await host.readbackBuffer(40, 256 * 64);
    const blue = await host.readbackBuffer(41, 256 * 64);
    const offset = 32 * 256 + 32 * 4;
    assert(red[offset] > 200 && red[offset + 2] < 50 &&
      blue[offset + 2] > 200 && blue[offset] < 50, "Recreated device did not render red-A/blue-B");
    assert(red.deviceGeneration === host.deviceGeneration &&
      blue.deviceGeneration === host.deviceGeneration, "Readback has the wrong device generation");
  }

  try {
    // Both begin in the same task, so the first adapter continuation is obsolete.
    const requests = await Promise.allSettled([
      host.negotiateAndCreateDevice(), host.negotiateAndCreateDevice(),
    ]);
    if (host.device) devices.add(host.device);
    assert(requests[0].status === "rejected" && /superseded/.test(requests[0].reason.message),
      "Older device request published after a newer request started");
    assert(requests[1].status === "fulfilled", "Latest device request failed");
    await render();

    // Fail during scoped command decoding, before submit. A failed preparation
    // must not tell the adapter to publish uploads or clear update ranges.
    const malformed = buildPacket(true).slice();
    new DataView(malformed.buffer, malformed.byteOffset, malformed.byteLength).setUint16(16, 0xffff, true);
    let invalidObserved = false;
    const invalid = await host.executePacket(malformed, null, () => { invalidObserved = true; }).then(
      () => null, error => error,
    );
    assert(invalid && !invalidObserved, "Malformed packet notified submission despite never reaching submit");

    const asyncObserver = await host.executePacket(buildPacket(true), null,
      () => host.executePacket(malformed)).then(() => null, error => error);
    assert(asyncObserver instanceof TypeError && /synchronously/.test(asyncObserver.message),
      "Async submission observer was accepted or its nested rejection was left unowned");

    const firstDevice = host.device;
    const firstGeneration = host.deviceGeneration;
    await host.negotiateAndCreateDevice();
    devices.add(host.device);
    assert(host.device !== firstDevice && host.deviceGeneration > firstGeneration,
      "Device replacement did not change device identity and generation");
    assertEmpty();
    const missing = await host.readbackBuffer(40, 256 * 64).then(
      () => null, error => error,
    );
    assert(missing && /unknown bufferId/.test(missing.message), "Old-device buffer remained readable after replacement");
    await render();

    // A real delayed loss of the retired device must leave the new renderer usable.
    const replacement = host.device;
    firstDevice.destroy();
    await firstDevice.lost;
    assert(host.device === replacement && host.buffers.size > 0,
      "Retired device loss invalidated its successor");
    await render();

    // Mapping starts before host-owned destruction. Observe rejection immediately to avoid
    // an unhandled promise rejection while awaiting the real device.lost signal.
    const pending = host.readbackBuffer(40, 256 * 64).then(
      () => ({ published: true }), error => ({ published: false, error }),
    );
    const beforeLoss = host.deviceGeneration;
    host.destroyDevice();
    await replacement.lost;
    const outcome = await pending;
    assert(!outcome.published, "Readback published after its GPU device was destroyed");
    assert(host.device === null && host.capabilityRecord === null &&
      host.deviceGeneration > beforeLoss, "Active device loss did not invalidate the host");
    assertEmpty();

    await host.negotiateAndCreateDevice();
    devices.add(host.device);
    await render();

    // Loss outside the host is observable through the real device.lost signal.
    // A map may finish before that signal; once notified, residency must be gone.
    const externallyLost = host.device;
    externallyLost.destroy();
    await externallyLost.lost;
    assert(host.device === null && host.capabilityRecord === null,
      "External device loss notification did not invalidate the host");
    assertEmpty();
    await host.negotiateAndCreateDevice();
    devices.add(host.device);
    await render();
    return "Real GPU devices: latest request wins; replacement clears residency; retired loss preserves successor; owned destruction rejects pending readback; external loss notification clears residency; recreated device renders Rust red-A/blue-B packet";
  } finally {
    host.destroyDevice();
    for (const device of devices) device.destroy();
  }
}
