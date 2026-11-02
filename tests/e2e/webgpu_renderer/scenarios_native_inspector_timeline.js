import {RendererInspector} from '/upstream/three.js/examples/jsm/inspector/RendererInspector.js';
import {Timeline} from '/upstream/three.js/examples/jsm/inspector/tabs/Timeline.js';

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function stopFrameClock(renderer) {
  if (renderer._animation) renderer._animation.stop();
  else {
    renderer._animationStarted = false;
    if (renderer._animationHandle !== null)
      renderer._animationHost?.cancelAnimationFrame(renderer._animationHandle);
    renderer._animationHandle = null;
    if (renderer.inspector.isRunning) renderer.inspector.finish();
  }
}

export const nativeInspectorTimelineScenarios = {
  native_inspector_timeline(T, renderer) {
    stopFrameClock(renderer);
    const inspector = new RendererInspector(); renderer.inspector = inspector;
    renderer.backend.trackTimestamp = true;
    const timeline = new Timeline(); timeline.setRenderer(renderer);
    timeline.content.style.cssText = 'display:block;width:900px;height:480px';
    document.body.appendChild(timeline.content);
    const methods = ['beginRender', 'draw', 'finishRender', 'beginCompute', 'compute', 'finishCompute',
      'beginBundle', 'finishBundle', 'addBundle'];
    const backend = renderer.backend, native = backend.isF3DBackend === true;
    const prototype = Object.getPrototypeOf(backend);
    for (const method of methods)
      check(typeof Object.getOwnPropertyDescriptor(prototype, method)?.value === 'function',
        `Timeline must discover the real backend operation ${method} on its prototype`);

    const N = T.TSL, scene = new T.Scene(); scene.name = 'Timeline scene';
    scene.background = new T.Color(0x172334);
    const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20);
    camera.name = 'Timeline camera'; camera.position.z = 5;
    const directGeometry = new T.PlaneGeometry(1.3, 1.5); directGeometry.name = 'Direct geometry';
    const directMaterial = new T.MeshBasicNodeMaterial(); directMaterial.name = 'Direct material';
    directMaterial.colorNode = N.vec3(N.uv().x.mul(.6).add(.2), N.uv().y.mul(.5).add(.2), .4);
    const direct = new T.Mesh(directGeometry, directMaterial); direct.name = 'Direct mesh';
    direct.frustumCulled = false;
    const bundledGeometry = new T.PlaneGeometry(1.1, 1.2); bundledGeometry.name = 'Bundled geometry';
    const bundledMaterial = new T.MeshBasicNodeMaterial({color: 0xe4a74d}); bundledMaterial.name = 'Bundled material';
    const bundled = new T.Mesh(bundledGeometry, bundledMaterial); bundled.name = 'Bundled mesh';
    bundled.position.x = .85; bundled.frustumCulled = false;
    const group = new T.BundleGroup(); group.name = 'Timeline bundle'; group.add(bundled);
    const count = 16, values = new T.StorageBufferAttribute(new Uint32Array(count), 1);
    const element = N.storage(values, 'uint', count).element(N.instanceIndex);
    const kernel = N.Fn(() => { element.addAssign(N.instanceIndex.add(N.uint(1))); })().compute(count);
    kernel.name = 'Timeline kernel';

    const compile = renderer.compileAsync.bind(renderer);
    renderer.compileAsync = async (...args) => {
      const result = await compile(...args);
      if (args[0] === scene && args[1] === camera) {
        // Pinned Timeline's createNodeBuilder detail assumes a non-null render
        // object, but the source compute builder receives null. Prepare this
        // real kernel before recording on both routes; retain its GPU increment.
        renderer._nodes.nodeFrame.frameId = 0; renderer.info.frame = 0;
        inspector.begin();
        try {
          renderer.compute(kernel);
          const prepared = new Uint32Array(await renderer.getArrayBufferAsync(values));
          for (let i = 0; i < count; i++)
            check(prepared[i] === i + 1, `Timeline kernel preparation value ${i}: got ${prepared[i]}`);
        } finally { inspector.finish(); }
      }
      return result;
    };
    const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
    const target = renderer.getRenderTarget();
    let frame = 0, originals = null, stoppedCounts = null;
    renderer.render = (...args) => {
      if (args[0] !== scene || args[1] !== camera) return render(...args);
      // compileAsync has its own NodeFrame advance. Begin the manual sequence
      // after that preparation and keep the real Inspector methods unchanged.
      if (frame === 0) renderer._nodes.nodeFrame.frameId = 0;
      frame++; renderer.info.reset(); renderer._nodes.nodeFrame.update();
      renderer.info.frame = renderer._nodes.nodeFrame.frameId;
      check(renderer.info.frame === frame, 'Background RAF must not change Timeline frame attribution');
      inspector.begin();
      try {
        if (frame === 1) {
          // Both objects are deliberately absent during compileAsync. The first
          // candidate render and following compute must finish through its cold
          // queue after the synchronous Inspector frame has already finished.
          scene.add(direct, group);
          timeline.recordButton.click();
          check(timeline.isRecording, 'The real Record control must start Timeline recording');
          originals = new Map(timeline.originalMethods);
          for (const method of methods) {
            check(originals.has(method), `Timeline did not wrap ${method}`);
            check(backend[method] !== originals.get(method), `Recording did not install its ${method} wrapper`);
          }
        }
        direct.position.x = -.85 + (frame % 3) / 32;
        directGeometry.setDrawRange(0, 3);
        const result = render(...args);
        if (frame === 1) {
          if (native)
            check(renderer._drain !== null, 'The first native Timeline render must exercise queued preparation');
          directGeometry.setDrawRange(0, 6);
        }
        renderer.compute(kernel);
        return result;
      } finally { inspector.finish(); }
    };
    renderer.readRenderTargetPixelsAsync = async (...args) => {
      const pixels = await read(...args);
      if (args[0] !== target) return pixels;
      await inspector.resolveTimestamp();
      check(renderer.info.frame === frame && inspector.currentFrame === null &&
        inspector.currentRender === null && inspector.currentCompute === null,
      'Queued backend operations must restore the completed Inspector frame context');
      const expectedFrames = Math.min(frame, 13);
      check(timeline.frames.length === expectedFrames, `Expected ${expectedFrames} useful Timeline frames, got ${timeline.frames.length}`);
      for (const [index, record] of timeline.frames.entries()) {
        check(record.id === index + 1, 'Timeline must retain each accepted source frame ID');
        check(Number.isFinite(record.fps) && record.fps >= 0 && Number.isFinite(record.triangles),
          `Timeline frame ${record.id} must expose finite statistics`);
        const calls = method => record.calls.filter(call => call.method === method);
        for (const method of ['beginRender', 'finishRender', 'beginCompute', 'compute', 'finishCompute', 'addBundle'])
          check(calls(method).length === 1, `Frame ${record.id}: expected one actual ${method} operation, got ${calls(method).length}`);
        const begin = calls('beginRender')[0];
        check(begin.target.camera === camera && begin.target.renderTarget === target,
          `Frame ${record.id}: Timeline lost the actual camera or target`);
        check(begin.details.scene === scene.name && begin.details.camera === camera.name,
          `Frame ${record.id}: queued render details lost their Inspector context`);
        check(typeof calls('beginCompute')[0].details.compute === 'string', 'Compute context must have a usable name');
        const compute = calls('compute')[0];
        check(compute.details.node === kernel.name && compute.details.dispatch === count && compute.details.bindings > 0,
          `Frame ${record.id}: Timeline must describe the actual kernel, bindings and dispatch`);
        const draws = calls('draw'), directDraw = draws.find(call => call.target.object === direct);
        const build = record.id === 1 ? 1 : 0;
        // Native keeps the existing whole-pass bundle cache, so warm replay
        // also retains the ordinary mesh. Timeline must expose physical work.
        if (build || !native) {
          check(directDraw?.target.material === directMaterial && directDraw.target.geometry === directGeometry,
            `Frame ${record.id}: Timeline draw identities must be the actual source objects`);
          check(directDraw.details.object === direct.name && directDraw.details.material === directMaterial.name &&
            directDraw.details.geometry === directGeometry.name && directDraw.details.triangles === 1,
          `Frame ${record.id}: Timeline must retain the captured direct draw range and labels`);
        }
        check(calls('beginBundle').length === build && calls('finishBundle').length === build,
          `Frame ${record.id}: warm bundle replay must not create another bundle`);
        check(draws.length === (native ? 2 * build : 1 + build) &&
          draws.filter(call => call.target.object === bundled).length === build,
          `Frame ${record.id}: warm bundle replay must not fabricate draw calls`);
        check(record.triangles === (build ? 3 : native ? 0 : 1), 'Timeline triangles must describe recorded draws');
      }
      const actual = new Uint32Array(await renderer.getArrayBufferAsync(values));
      check(actual.length === count, 'Timeline compute must preserve storage layout');
      for (let i = 0; i < count; i++)
        check(actual[i] === (frame + 1) * (i + 1), `Timeline compute value ${i}, frame ${frame}: got ${actual[i]}`);
      check(values.array.every(value => value === 0), 'Timeline instrumentation must not replace GPU compute with CPU writes');
      if (frame === 13) {
        timeline.recordButton.click();
        check(!timeline.isRecording && timeline.originalMethods.size === 0, 'The real Stop control must end recording');
        for (const [method, original] of originals)
          check(backend[method] === original, `Stopping Timeline did not restore ${method}`);
        timeline.selectFrame(12);
        check(timeline.timelineContent.childElementCount > 0 && timeline.timelineContent.textContent.includes(scene.name) &&
          timeline.frameInfo.textContent.includes('13'),
        'The real Timeline must display the selected frame and render details');
        stoppedCounts = timeline.frames.map(record => record.calls.length);
      } else if (frame > 13) {
        check(JSON.stringify(timeline.frames.map(record => record.calls.length)) === JSON.stringify(stoppedCounts),
          'Rendering after Stop must not append any Timeline calls');
        for (const [method, original] of originals)
          check(backend[method] === original, `Stopped Timeline replaced ${method} again`);
      }
      return pixels;
    };
    const dispose = renderer.dispose.bind(renderer);
    renderer.dispose = () => {
      timeline.stopRecording(); timeline.graph.dispose(); timeline.content.remove();
      kernel.dispose(); directGeometry.dispose(); directMaterial.dispose();
      bundledGeometry.dispose(); bundledMaterial.dispose(); return dispose();
    };
    return {scene, camera};
  },
};
