import {RendererInspector} from '/upstream/three.js/examples/jsm/inspector/RendererInspector.js';

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function duration(value, label) {
  check(typeof value === 'number' && Number.isFinite(value) && value >= 0,
    `${label}: expected a finite nonnegative GPU duration, got ${String(value)}`);
}

function stopFrameClock(renderer) {
  // The harness owns logical frames. Stop each renderer's background RAF before
  // attaching the actual upstream Inspector, so wall time cannot change UIDs.
  if (renderer._animation) renderer._animation.stop();
  else {
    renderer._animationStarted = false;
    if (renderer._animationHandle !== null)
      renderer._animationHost?.cancelAnimationFrame(renderer._animationHandle);
    renderer._animationHandle = null;
    if (renderer.inspector.isRunning) renderer.inspector.finish();
  }
}

export const nativeInspectorTimestampScenarios = {
  native_inspector_timestamps(T, renderer) {
    const backend = renderer.backend, N = T.TSL;
    check(backend.device.features.has('timestamp-query'),
      'This correctness proof requires actual timestamp-query support on the source and candidate devices');
    check(backend.hasTimestamp === true, 'The backend must expose actual timestamp-query capability');
    stopFrameClock(renderer);
    const inspector = new RendererInspector();
    const resolveInspector = inspector.resolveTimestamp.bind(inspector);
    // Only defer automatic scheduling. At readback checkpoints, invoke the
    // unchanged upstream resolver and inspect its real published frame stats.
    inspector.resolveTimestamp = () => {};
    renderer.inspector = inspector;
    check(inspector.getRenderer() === renderer, 'Inspector must attach to the exercised renderer');
    backend.trackTimestamp = true;
    const passCounts = {render: 0, compute: 0}, beginPasses = {};
    const queryStarts = new WeakMap(), repeatedEnds = {render: 0, compute: 0};
    for (const [type, method] of [['render', 'beginRenderPass'], ['compute', 'beginComputePass']]) {
      const beginPass = beginPasses[method] = GPUCommandEncoder.prototype[method];
      GPUCommandEncoder.prototype[method] = function (descriptor) {
        const writes = descriptor?.timestampWrites;
        if (writes) {
          check(writes.querySet instanceof GPUQuerySet && writes.querySet.type === 'timestamp',
            `${type}: timing must use a real GPU timestamp query set`);
          check(Number.isInteger(writes.endOfPassWriteIndex), `${type}: the real GPU pass must write an end timestamp`);
          if (writes.beginningOfPassWriteIndex !== undefined) {
            check(Number.isInteger(writes.beginningOfPassWriteIndex) &&
              writes.beginningOfPassWriteIndex !== writes.endOfPassWriteIndex,
              `${type}: the first GPU pass must write distinct beginning and end timestamps`);
            let ends = queryStarts.get(writes.querySet);
            if (!ends) queryStarts.set(writes.querySet, ends = new Set());
            ends.add(writes.endOfPassWriteIndex);
          } else {
            check(queryStarts.get(writes.querySet)?.has(writes.endOfPassWriteIndex),
              `${type}: a continuation must update the end of an already started query`);
            repeatedEnds[type]++;
          }
          passCounts[type]++;
        }
        return beginPass.call(this, descriptor);
      };
    }

    const scene = new T.Scene(); scene.background = new T.Color(0x172334);
    const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20);
    camera.position.z = 5;
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = N.vec3(N.uv().x.mul(.6).add(.2), N.uv().y.mul(.5).add(.2), .4);
    const geometry = new T.PlaneGeometry(2.5, 1.8), mesh = new T.Mesh(geometry, material);
    scene.add(mesh);
    const count = 16, values = new T.StorageBufferAttribute(new Uint32Array(count), 1);
    const element = N.storage(values, 'uint', count).element(N.instanceIndex);
    const kernel = N.Fn(() => { element.addAssign(N.instanceIndex.add(N.uint(1))); })().compute(count);
    kernel.name = 'Inspector timestamp compute';

    const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
    const target = renderer.getRenderTarget();
    let frame = 0, resolvedThrough = 0;
    renderer.render = (...args) => {
      if (args[0] !== scene || args[1] !== camera) return render(...args);
      // Source compileAsync advances NodeFrame too; the manual logical
      // sequence begins only after the harness completes that preparation.
      if (frame === 0) renderer._nodes.nodeFrame.frameId = 0;
      frame++;
      renderer.info.reset();
      renderer._nodes.nodeFrame.update();
      renderer.info.frame = renderer._nodes.nodeFrame.frameId;
      check(renderer.info.frame === frame, 'Logical timestamp frame must not advance on background RAF');
      inspector.begin();
      try {
        // The native grouped call uses two real passes sharing one query pair.
        // It must retain one Inspector UID while the later call gets its own.
        mesh.position.x = ((frame % 5) - 2) / 16;
        renderer.compute([kernel, kernel]);
        render(...args);
        renderer.compute(kernel);
        return render(...args);
      } finally { inspector.finish(); }
    };
    renderer.readRenderTargetPixelsAsync = async (...args) => {
      const pixels = await read(...args);
      if (args[0] !== target) return pixels;
      const records = inspector.frames.filter(record => record.frameId > resolvedThrough);
      const expectedFrames = records.map(record => record.frameId);
      check(records.length === frame - resolvedThrough, 'Inspector must retain every logical frame');
      for (const type of ['compute', 'render']) {
        check(passCounts[type] >= 2 * frame, `${type}: actual GPU passes must write timestamps for both calls`);
        if (type === 'compute') {
          const native = backend.isF3DBackend === true;
          check(passCounts.compute === frame * (native ? 3 : 2),
            'Grouped compute must execute every real timestamped pass');
          check(repeatedEnds.compute === (native ? frame : 0),
            'Each native grouped compute call must rewrite its query end in a second real pass');
        }
        const pending = type === 'render' ? renderer.resolveTimestampsAsync() : renderer.resolveTimestampsAsync(type);
        check(pending instanceof Promise, `${type}: timestamp resolution must remain asynchronous`);
        const total = await pending;
        duration(total, `${type} total`);
        check(renderer.info[type].timestamp === total, `${type}: public info must publish the resolved total`);
        check(JSON.stringify(backend.getTimestampFrames(type)) === JSON.stringify(expectedFrames),
          `${type}: query results must retain their submitted logical frames`);
        for (const record of records) {
          const stats = record[type === 'render' ? 'renders' : 'computes'];
          check(stats.length === 2, `${type}, frame ${record.frameId}: Inspector must record both calls`);
          check(new Set(stats.map(stat => stat.uid)).size === stats.length,
            `${type}, frame ${record.frameId}: separate calls must have distinct query UIDs`);
          let sum = 0;
          for (const stat of stats) {
            check(stat.uid.endsWith(`:f${record.frameId}`), `${type}: query UID lost its source frame`);
            check(backend.hasTimestampQuery(stat.uid) === true, `${type}: missing completed query ${stat.uid}`);
            const value = backend.getTimestamp(stat.uid);
            duration(value, stat.uid); sum += value;
          }
          if (record.frameId === frame)
            check(Math.abs(total - sum) < 1e-9, `${type}: total must sum the latest frame's actual query durations`);
        }
      }
      await resolveInspector();
      for (const record of records) {
        check(record.resolvedRender && record.resolvedCompute,
          `Inspector frame ${record.frameId}: both query types must resolve`);
        for (const stat of [...record.renders, ...record.computes]) {
          check(stat.gpuNotAvailable !== true, `Inspector must consume the real query for ${stat.uid}`);
          check(stat.gpu === backend.getTimestamp(stat.uid), `Inspector published a different duration for ${stat.uid}`);
          duration(stat.gpu, `Inspector ${stat.uid}`);
        }
      }
      resolvedThrough = frame;
      const actual = new Uint32Array(await renderer.getArrayBufferAsync(values));
      check(actual.length === count, 'Timestamped compute must preserve the storage layout');
      for (let i = 0; i < count; i++)
        check(actual[i] === 3 * frame * (i + 1),
          `Timestamped compute value ${i}, frame ${frame}: expected ${3 * frame * (i + 1)}, got ${actual[i]}`);
      check(values.array.every(value => value === 0), 'GPU compute must not rewrite the public CPU array');
      const colors = new Set();
      for (let i = 0; i < pixels.length; i += 4)
        colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
      check(colors.size > 24, 'Timestamped renders must still produce visible nonconstant output');
      return pixels;
    };
    const dispose = renderer.dispose.bind(renderer);
    renderer.dispose = () => {
      for (const [method, beginPass] of Object.entries(beginPasses)) GPUCommandEncoder.prototype[method] = beginPass;
      kernel.dispose(); geometry.dispose(); material.dispose(); return dispose();
    };
    return {scene, camera};
  },
};
