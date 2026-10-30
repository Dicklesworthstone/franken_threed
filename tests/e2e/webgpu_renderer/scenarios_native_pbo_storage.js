// The pinned cloth/birds examples mark native storage with setPBO(true) so
// their separate WebGL fallback can read it too. Keep that authored hint.
function world(T) {
  const scene = new T.Scene(); scene.background = new T.Color(.025, .045, .08);
  const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20); camera.position.z = 5;
  return {scene, camera};
}

function execute(renderer, result, kernels, update, verify) {
  const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
  const read = renderer.readRenderTargetPixelsAsync.bind(renderer), target = renderer.getRenderTarget();
  let warmed = false, frame = 0, initial = null;
  renderer.compileAsync = async (...args) => {
    if (args[0] === result.scene && !warmed) { await renderer.computeAsync(kernels); warmed = true; }
    return compile(...args);
  };
  renderer.render = (...args) => {
    if (args[0] === result.scene && args[1] === result.camera) { update(++frame); renderer.compute(kernels); }
    return render(...args);
  };
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] !== target) return pixels;
    await verify(frame);
    const colors = new Set();
    for (let i = 0; i < pixels.length; i += 4) colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
    if (colors.size < 24) throw Error('PBO storage must drive visible geometry and color');
    if (frame === 1) initial = pixels.slice();
    if (frame >= 8 && initial && pixels.every((value, i) => value === initial[i]))
      throw Error('Native PBO compute did not change the rendered output');
    return pixels;
  };
  return result;
}

export const pboStorageScenarios = {
  native_pbo_cloth(T, renderer) {
    const N = T.TSL, result = world(T), geometry = new T.PlaneGeometry(2.8, 2, 4, 3);
    const original = geometry.attributes.position.array.slice(), count = geometry.attributes.position.count;
    const positions = N.instancedArray(original.slice(), 'vec3').setPBO(true);
    const velocities = N.instancedArray(count, 'vec3'), pairs = [], rest = [], adjacency = new Int32Array(count * 4);
    const degrees = new Uint8Array(count), width = 5;
    const edge = (a, b) => {
      const index = pairs.length / 2; pairs.push(a, b);
      rest.push(Math.hypot(original[a * 3] - original[b * 3], original[a * 3 + 1] - original[b * 3 + 1]));
      adjacency[a * 4 + degrees[a]++] = index + 1; adjacency[b * 4 + degrees[b]++] = -index - 1;
    };
    for (let i = 0; i < count; i++) {
      if (i % width + 1 < width) edge(i, i + 1);
      if (i + width < count) edge(i, i + width);
    }
    const ids = N.instancedArray(new Uint32Array(pairs), 'uvec2').setPBO(true);
    const lengths = N.instancedArray(new Float32Array(rest), 'float');
    const forces = N.instancedArray(rest.length, 'vec3').setPBO(true);
    const links = N.instancedArray(adjacency, 'ivec4').setPBO(true);
    const wind = N.uniform(.4), gravity = N.uniform(-3), dt = N.float(1 / 32);
    const springKernel = N.Fn(() => {
      const pair = ids.element(N.instanceIndex), delta = positions.element(pair.y).sub(positions.element(pair.x));
      const length = delta.length();
      forces.element(N.instanceIndex).assign(delta.mul(length.sub(lengths.element(N.instanceIndex)).mul(25).div(length.max(.0001))));
    })().compute(rest.length, [32]);
    const vertexKernel = N.Fn(() => {
      N.If(N.instanceIndex.greaterThanEqual(width), () => {
        const total = N.vec3(wind, gravity, 0).toVar(), neighbors = links.element(N.instanceIndex);
        for (let i = 0; i < 4; i++) {
          const signed = neighbors.element(i);
          N.If(signed.notEqual(0), () => {
            const force = forces.element(N.uint(signed.abs().sub(1)));
            total.addAssign(force.mul(signed.greaterThan(0).select(1, -1)));
          });
        }
        const velocity = velocities.element(N.instanceIndex);
        velocity.assign(velocity.add(total.mul(dt)).mul(.98));
        positions.element(N.instanceIndex).addAssign(velocity.mul(dt));
      });
    })().compute(count, [32]);
    const material = new T.MeshBasicNodeMaterial({side: T.DoubleSide});
    material.positionNode = positions.element(N.vertexIndex);
    material.colorNode = N.vec3(N.uv().x.mul(.6).add(.2), N.uv().y.mul(.5).add(.25),
      forces.element(0).length().mul(.1).add(.15));
    const mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; result.scene.add(mesh);
    return execute(renderer, result, [springKernel, vertexKernel], frame => {
      if (frame === 8) wind.value = -1.1;
      if (frame === 10) gravity.value = -1.5;
      if (frame === 12) for (const node of [positions, ids, forces, links]) node.setPBO(false);
      if (frame === 13) for (const node of [positions, ids, forces, links]) node.setPBO(true);
    }, async frame => {
      const gpu = new Float32Array(await renderer.getArrayBufferAsync(positions.value));
      if (gpu.length !== count * 4 || gpu.some(value => !Number.isFinite(value))) throw Error('Cloth lost padded native vec3 storage');
      for (let i = 0; i < width; i++) for (let c = 0; c < 3; c++)
        if (gpu[i * 4 + c] !== original[i * 3 + c]) throw Error('Compute moved a fixed cloth vertex');
      if (frame >= 7 && Math.abs(gpu[(count - 1) * 4 + 1] - original[(count - 1) * 3 + 1]) < .01)
        throw Error('Spring and vertex kernels did not deform the GPU cloth');
      if (positions.value.itemSize !== 4) throw Error('Cloth lost source WebGPU storage padding');
      for (let i = 0; i < count; i++) for (let c = 0; c < 4; c++)
        if (positions.value.array[i * 4 + c] !== (c === 3 ? 0 : original[i * 3 + c]))
          throw Error('Cloth compute overwrote the padded source CPU array');
    });
  },

  native_pbo_instances(T, renderer) {
    const N = T.TSL, result = world(T), count = 8;
    const positions = N.instancedArray(count, 'vec3').setPBO(true), phases = N.instancedArray(count, 'float').setPBO(true);
    const tick = N.uniform(0), shift = N.uniform(0);
    const kernel = N.Fn(() => {
      const id = N.float(N.instanceIndex), phase = id.div(count).add(tick.div(32)).fract();
      positions.element(N.instanceIndex).assign(N.vec3(id.mod(4).mul(.8).sub(1.2).add(shift),
        id.div(4).floor().mul(.8).sub(.4).add(phase.mul(6.283185307).sin().mul(.25)), 0));
      phases.element(N.instanceIndex).assign(phase);
    })().compute(count, [8]);
    const material = new T.MeshBasicNodeMaterial();
    material.positionNode = N.positionLocal.add(positions.toAttribute());
    const phase = phases.element(N.instanceIndex);
    material.colorNode = N.vec3(phase.mul(.7).add(.2), N.uv().x.mul(.7).add(.15), N.float(1).sub(phase).mul(.6).add(.2));
    const mesh = new T.Mesh(new T.PlaneGeometry(.4, .4), material); mesh.count = count; mesh.frustumCulled = false; result.scene.add(mesh);
    return execute(renderer, result, kernel, frame => {
      tick.value = frame;
      if (frame === 8) shift.value = .125;
      if (frame === 10) shift.value = -.125;
      if (frame === 12) { positions.setPBO(false); phases.setPBO(false); }
      if (frame === 13) { positions.setPBO(true); phases.setPBO(true); }
    }, async frame => {
      const gpu = new Float32Array(await renderer.getArrayBufferAsync(positions.value));
      const colors = new Float32Array(await renderer.getArrayBufferAsync(phases.value));
      if (gpu.length !== count * 4 || colors.length !== count) throw Error('Instanced storage lost its source layout');
      for (let i = 0; i < count; i++) {
        if (Math.abs(gpu[i * 4] - ((i % 4) * .8 - 1.2 + shift.value)) > 1e-6 ||
            Math.abs(colors[i] - ((i / count + frame / 32) % 1)) > 1e-6)
          throw Error('Generated vertex and fragment inputs did not observe current GPU storage');
      }
      if (positions.value.array.some(value => value !== 0) || phases.value.array.some(value => value !== 0))
        throw Error('PBO storage must not introduce a CPU readback path');
    });
  },
};
