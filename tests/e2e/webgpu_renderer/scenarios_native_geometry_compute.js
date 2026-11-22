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

// The first draw reads storage before a later object's geometryNode updates it.
// Source WebGPU submits that compute encoder before the pending parent render,
// so both draws must observe the updated allocation, including on cold setup.
function overlappingStorageScenario(T, renderer) {
  const N = T.TSL, scene = new T.Scene();
  const camera = new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 20);
  camera.position.z = 5;
  const attribute = new T.StorageBufferAttribute(new Float32Array(4), 4);
  const writable = N.storage(attribute, 'vec4', 1).element(0);
  const readable = N.storage(attribute, 'vec4', 1).toReadOnly().element(0);
  const shift = N.uniform(0), tint = N.uniform(.25), tick = N.uniform(0);
  const kernel = N.Fn(() => {
    writable.assign(N.vec4(shift, shift.mul(-.5), tint, writable.w.add(tick)));
  })().compute(1);
  let initializations = 0;
  kernel.onInit(() => { initializations++; });
  for (let index = 0; index < 2; index++) {
    const material = new T.MeshBasicNodeMaterial({side: T.DoubleSide});
    material.positionNode = N.positionLocal.add(N.vec3(readable.xy, 0));
    material.colorNode = N.vec3(index ? .75 : .25, readable.z, N.uv().y.mul(.5).add(.25));
    if (index === 1) material.geometryNode = kernel;
    const mesh = new T.Mesh(new T.PlaneGeometry(1.25, 1, 4, 3), material);
    mesh.position.set(index ? .85 : -.85, index ? -.25 : .25, 0);
    mesh.renderOrder = index; mesh.frustumCulled = false; scene.add(mesh);
  }
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  const target = renderer.getRenderTarget(); let frame = 0, firstPixels = null;
  renderer.render = (...args) => {
    if (args[0] === scene && args[1] === camera) {
      renderer._nodes.nodeFrame.update(); frame++; tick.value = 1;
      shift.value = ((frame % 5) - 2) / 8;
      tint.value = .25 + (frame % 3) / 8;
    }
    return render(...args);
  };
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] !== target) return pixels;
    const values = new Float32Array(await renderer.getArrayBufferAsync(attribute));
    const expected = [shift.value, -.5 * shift.value, tint.value, frame];
    if (initializations !== 1 || values.length !== 4 ||
        values.some((value, index) => value !== expected[index]))
      throw Error(`Overlapping geometry compute must publish one shared update per frame (${frame})`);
    if (attribute.array.some(value => value !== 0))
      throw Error('Overlapping geometry compute must retain GPU-only results');
    if (frame === 1) firstPixels = pixels.slice();
    if (frame === 8 && firstPixels && pixels.every((value, index) => value === firstPixels[index]))
      throw Error('Overlapping storage reads must display the live computed values');
    return pixels;
  };
  return {scene, camera};
}

// Texture reads obey the same queue boundary as buffer reads. The first mesh
// samples an existing allocation before the second mesh triggers its next
// compute write; both pending draws must see that write after submission.
// Keep a single mip: the source regenerates upper levels at later sample uses.
async function overlappingTextureScenario(T, renderer) {
  const N = T.TSL, scene = new T.Scene();
  scene.background = new T.Color(0x172334);
  const camera = new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 20);
  camera.position.z = 5;
  const texture = new T.StorageTexture(4, 4), source = texture.source;
  texture.type = T.UnsignedByteType; texture.format = T.RGBAFormat;
  texture.colorSpace = T.NoColorSpace; texture.generateMipmaps = false;
  texture.minFilter = texture.magFilter = T.NearestFilter;
  const counters = new T.StorageBufferAttribute(new Uint32Array(16), 1);
  const counter = N.storage(counters, 'uint', 16).element(N.instanceIndex);
  const tint = N.uniform(.125), tick = N.uniform(0, 'uint');
  const kernel = N.Fn(() => {
    const x = N.instanceIndex.mod(4), y = N.instanceIndex.div(4);
    N.textureStore(texture, N.uvec2(x, y),
      N.vec4(x.toFloat().mul(.125).add(.125), tint, y.toFloat().mul(.125).add(.25), 1));
    counter.addAssign(tick);
  })().compute(16);
  let initializations = 0;
  kernel.onInit(() => { initializations++; });
  await renderer.computeAsync(kernel);
  for (let index = 0; index < 2; index++) {
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = N.texture(texture, N.uv()).rgb;
    if (index === 1) material.geometryNode = kernel;
    const mesh = new T.Mesh(new T.PlaneGeometry(1.5, 1.5), material);
    mesh.position.x = index ? .9 : -.9;
    mesh.renderOrder = index; mesh.frustumCulled = false; scene.add(mesh);
  }
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  const target = renderer.getRenderTarget(); let frame = 0, firstPixels = null;
  renderer.render = (...args) => {
    if (args[0] === scene && args[1] === camera) {
      renderer._nodes.nodeFrame.update(); frame++; tick.value = 1;
      tint.value = .125 + (frame % 4) * .125;
    }
    return render(...args);
  };
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] !== target) return pixels;
    const values = new Uint32Array(await renderer.getArrayBufferAsync(counters));
    if (initializations !== 1 || values.length !== 16 || values.some(value => value !== frame))
      throw Error(`Shared texture geometry compute must run once per frame (${frame})`);
    if (texture.source !== source || texture.image.data !== undefined || counters.array.some(value => value !== 0))
      throw Error('Shared texture compute must retain source identity and GPU-only results');
    const colors = new Set();
    for (let i = 0; i < pixels.length; i += 4) colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
    if (colors.size < 16) throw Error('Shared texture draws must display the computed grid');
    if (frame === 1) firstPixels = pixels.slice();
    if (frame === 8 && firstPixels && pixels.every((value, index) => value === firstPixels[index]))
      throw Error('Pending texture consumers must display the next computed image');
    return pixels;
  };
  return {scene, camera};
}

export const geometryComputeScenarios = {
  native_geometry_compute_fn(T, renderer) { return scenario(T, renderer, false); },
  native_geometry_compute_shared(T, renderer) { return scenario(T, renderer, true); },
  native_geometry_compute_overlap(T, renderer) { return overlappingStorageScenario(T, renderer); },
  native_geometry_compute_texture_overlap(T, renderer) { return overlappingTextureScenario(T, renderer); },
};
