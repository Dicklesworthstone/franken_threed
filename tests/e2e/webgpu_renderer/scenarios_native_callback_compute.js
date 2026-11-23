// Source compute submits before its enclosing render encoder. Each callback
// writes an allocation not read by an earlier draw in that pending pass. The
// next draw consumes it without a CPU round trip or a different renderer.
async function callbackCompute(T, renderer, interleaved) {
  const N = T.TSL, scene = new T.Scene(), target = renderer.getRenderTarget();
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 20); camera.position.z = 5;
  const records = [], events = [];
  let frame = 0;
  function record(tag) {
    const attribute = new T.StorageBufferAttribute(new Float32Array(4), 4);
    const storage = N.storage(attribute, 'vec4', 1), gain = N.uniform(0);
    const kernel = N.Fn(() => {
      storage.element(0).x.assign(gain);
      storage.element(0).y.assign(tag);
      storage.element(0).z.assign(.375);
      storage.element(0).w.addAssign(1);
    })().compute(1, [1]);
    const value = {attribute, storage, gain, kernel, tag}; records.push(value); return value;
  }
  const first = record(.25), second = interleaved ? record(.5) : null;
  const final = interleaved ? record(.75) : null;
  // Compile the compute pipelines before entering synchronous draw callbacks.
  // Each warmup is a real dispatch and remains included in exact readback counts.
  for (const value of records) await renderer.computeAsync(value.kernel);
  const geometry = new T.PlaneGeometry(interleaved ? 1 : 2, 2);
  function mesh(value, x) {
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = value.storage.element(0).xyz;
    const object = new T.Mesh(geometry, material); object.frustumCulled = false;
    object.position.x = x; scene.add(object); return object;
  }
  const left = mesh(first, interleaved ? -.5 : 0), right = second ? mesh(second, .5) : null;
  left.renderOrder = 0; if (right) right.renderOrder = 1;
  const gainAt = index => index >= 13 ? .75 : index >= 8 ? .5 : .25;
  function dispatch(value, gain) {
    value.gain.value = gain; renderer.compute(value.kernel);
  }
  left.onBeforeRender = function(actualRenderer, actualScene, actualCamera, actualGeometry, material) {
    if (frame === 0) return;
    if (this !== left || actualRenderer !== renderer || actualScene !== scene || actualCamera !== camera ||
        actualGeometry !== geometry || material !== left.material) throw Error('Before hook lost its source receiver or arguments');
    events.push('left-before'); dispatch(first, gainAt(frame));
  };
  left.onAfterRender = function(actualRenderer, actualScene, actualCamera) {
    if (frame === 0) return;
    if (this !== left || actualRenderer !== renderer || actualScene !== scene || actualCamera !== camera)
      throw Error('After hook lost its source receiver or arguments');
    events.push('left-after'); if (second) dispatch(second, .125);
  };
  if (right) {
    right.onBeforeRender = () => {
      if (frame === 0) return;
      events.push('right-before'); dispatch(second, 1 - gainAt(frame));
    };
    right.onAfterRender = () => {
      if (frame === 0) return;
      events.push('right-after'); dispatch(final, gainAt(frame));
    };
  }
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  renderer.render = (...args) => {
    if (args[0] !== scene || args[1] !== camera) return render(...args);
    frame++; events.length = 0;
    const result = render(...args);
    const expected = interleaved ? ['left-before', 'left-after', 'right-before', 'right-after'] : ['left-before', 'left-after'];
    if (events.length !== expected.length || events.some((event, index) => event !== expected[index]))
      throw Error(`Render callbacks changed their synchronous source order at frame ${frame}: ${events}`);
    return result;
  };
  let initial = null;
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] !== target) return pixels;
    for (const value of records) {
      const gpu = new Float32Array(await renderer.getArrayBufferAsync(value.attribute));
      const gain = value === second ? 1 - gainAt(frame) : gainAt(frame);
      const count = 1 + frame * (value === second ? 2 : 1);
      const expected = [gain, value.tag, .375, count];
      if (gpu.length !== 4 || gpu.some((number, index) => number !== expected[index]))
        throw Error(`Callback compute result diverged at frame ${frame}: ${Array.from(gpu)} vs ${expected}`);
      if (value.attribute.array.some(number => number !== 0))
        throw Error('Callback compute copied GPU state into the public CPU array');
    }
    if (frame === 1) initial = pixels.slice();
    if (frame >= 8 && initial && pixels.every((value, index) => value === initial[index]))
      throw Error('Callback-produced values did not update the rendered image');
    return pixels;
  };
  return {scene, camera};
}

export const callbackComputeScenarios = {
  native_callback_compute_before: (T, renderer) => callbackCompute(T, renderer, false),
  native_callback_compute_interleaved: (T, renderer) => callbackCompute(T, renderer, true),
};
