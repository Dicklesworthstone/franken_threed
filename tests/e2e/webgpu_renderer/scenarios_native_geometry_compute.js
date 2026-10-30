// Geometry setup follows the pinned jelly example: a lazy geometryNode creates
// GPU positions and returns a compute kernel whose onInit dispatches a second
// kernel. Algebraic deformation makes every GPU position independently testable.
function setup(T, renderer, geometry, controls) {
  const N = T.TSL, original = geometry.attributes.position, count = original.count;
  const base = Array.from(original.array), positions = new T.StorageBufferAttribute(count, 3);
  const counters = new T.StorageBufferAttribute(new Uint32Array(count), 1);
  geometry.setAttribute('storagePosition', positions);
  const input = N.storage(original, 'vec3', count).toReadOnly().element(N.instanceIndex);
  const output = N.storage(positions, 'vec3', count).element(N.instanceIndex);
  const counter = N.storage(counters, 'uint', count).element(N.instanceIndex);
  const initialize = N.Fn(() => { output.assign(input); counter.assign(N.uint(0)); })().compute(count);
  const kernel = N.Fn(() => {
    output.assign(N.vec3(input.x.mul(controls.stretch).add(controls.shift),
      input.y.add(input.x.mul(controls.shear)), input.z));
    counter.addAssign(controls.tick);
  })().compute(count);
  const state = {positions, counters, base, count, kernel, initCalls: 0};
  kernel.onInit(() => { state.initCalls++; renderer.compute(initialize); });
  return state;
}

function scenario(T, renderer, shared) {
  const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0x172334);
  const camera = new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 20);
  camera.position.z = 5;
  // This ordinary draw must remain before automatic compute in the same scene.
  const backdrop = new T.MeshBasicNodeMaterial();
  backdrop.colorNode = N.mix(N.vec3(.025, .06, .11), N.vec3(.12, .18, .28), N.uv().y);
  const background = new T.Mesh(new T.PlaneGeometry(4.6, 3.4), backdrop);
  background.position.z = -1; background.renderOrder = -2; scene.add(background);
  const companion = new T.Mesh(new T.BoxGeometry(.3, .3, .3),
    new T.MeshBasicNodeMaterial({color: 0x87bfa5}));
  companion.position.set(-1.8, 1.25, 0); companion.renderOrder = -1; scene.add(companion);

  const geometry = new T.PlaneGeometry(1.25, 1, 4, 3);
  const controls = {stretch: N.uniform(1), shift: N.uniform(0), shear: N.uniform(0), tick: N.uniform(0, 'uint')};
  const material = new T.MeshBasicNodeMaterial({side: T.DoubleSide});
  material.colorNode = N.vec3(N.uv().x.mul(.7).add(.2), N.uv().y.mul(.6).add(.15), .2);
  material.positionNode = N.attribute('storagePosition');
  let state = null, setupCalls = 0;
  if (shared) {
    state = setup(T, renderer, geometry, controls);
    material.geometryNode = state.kernel;
  } else {
    material.geometryNode = N.Fn(({geometry, renderer}) => {
      if (++setupCalls !== 1) throw Error('geometryNode recreated its compute/storage state');
      state = setup(T, renderer, geometry, controls);
      return state.kernel;
    })();
  }
  const mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false;
  mesh.position.x = shared ? -.85 : 0; scene.add(mesh);
  if (shared) {
    const second = new T.Mesh(geometry, material); second.frustumCulled = false;
    second.position.set(.85, -.3, 0); second.renderOrder = 1; scene.add(second);
  }

  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  const target = renderer.getRenderTarget(); let frame = 0, firstPixels = null;
  renderer.render = (...args) => {
    if (args[0] === scene && args[1] === camera) {
      // Match source FRAME scheduling without depending on the background RAF.
      renderer._nodes.nodeFrame.update(); frame++; controls.tick.value = 1;
      controls.shift.value = ((frame % 5) - 2) / 32;
      if (frame === 8) { controls.stretch.value = 1.5; controls.shear.value = .25; }
      if (frame === 10) { controls.stretch.value = .75; controls.shear.value = -.25; }
      if (frame === 12) { controls.stretch.value = 1; controls.shear.value = .5; }
      if (frame === 13) { controls.stretch.value = 1.25; controls.shear.value = 0; }
    }
    return render(...args);
  };
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] !== target) return pixels;
    if (!state || state.initCalls !== 1 || geometry.attributes.storagePosition !== state.positions)
      throw Error('Automatic geometry compute must initialize once and preserve storage identity');
    const positions = new Float32Array(await renderer.getArrayBufferAsync(state.positions));
    const counters = new Uint32Array(await renderer.getArrayBufferAsync(state.counters));
    if (positions.length !== state.count * 4 || counters.length !== state.count)
      throw Error('Automatic geometry compute lost the source padded vec3 storage layout');
    for (let i = 0; i < state.count; i++) {
      const expected = [state.base[i * 3] * controls.stretch.value + controls.shift.value,
        state.base[i * 3 + 1] + state.base[i * 3] * controls.shear.value, state.base[i * 3 + 2]];
      if (expected.some((value, channel) => Math.abs(positions[i * 4 + channel] - value) > 1e-6))
        throw Error(`Automatic geometry compute has incorrect GPU position at vertex ${i}, frame ${frame}`);
      if (counters[i] !== frame)
        throw Error(`Shared geometry compute must run once per FRAME: vertex ${i}, expected ${frame}, got ${counters[i]}`);
    }
    if (state.positions.array.some(value => value !== 0) || state.counters.array.some(value => value !== 0))
      throw Error('Geometry compute must write GPU storage without rewriting CPU arrays');
    const colors = new Set();
    for (let i = 0; i < pixels.length; i += 4) colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
    if (colors.size < 24) throw Error('Computed geometry must produce visible, nonconstant output');
    if (frame === 1) firstPixels = pixels.slice();
    if (frame === 8 && firstPixels && pixels.every((value, i) => value === firstPixels[i]))
      throw Error('Live automatic geometry compute did not change the rendered output');
    return pixels;
  };
  return {scene, camera};
}

export const geometryComputeScenarios = {
  native_geometry_compute_fn(T, renderer) { return scenario(T, renderer, false); },
  native_geometry_compute_shared(T, renderer) { return scenario(T, renderer, true); },
};
