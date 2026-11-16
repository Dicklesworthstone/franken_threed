// Fragment and compute kernels share one mixed-layout record. Atomic results
// are checked as exact GPU integers; pixels depend only on stable companion
// fields, never on invocation order or the return value of atomicAdd.
function atomicRecords(T, renderer, floatBacking) {
  const N = T.TSL, scene = new T.Scene(), target = renderer.getRenderTarget();
  const width = target.width, height = target.height, count = 4;
  if (width % count !== 0) throw Error('Atomic record fixture needs equal screen-space buckets');
  const schema = N.struct({hits: {type: 'uint', atomic: true}, balance: {type: 'int', atomic: true},
    tag: 'uint', gain: 'float', color: 'vec3', padding: 'float'}, 'FragmentRecord');
  if (schema.getLength() !== 8) throw Error('Atomic fixture lost the pinned mixed-field stride');
  const array = floatBacking ? new Float32Array(8) : new Uint32Array(8);
  const words = new Uint32Array(array.buffer), floats = new Float32Array(array.buffer);
  words.set([3, (-50) >>> 0, 0x12340000]);
  floats[3] = .5; floats.set([.2, .7, .35], 4);
  const Attribute = floatBacking ? T.StorageInstancedBufferAttribute : T.StorageBufferAttribute;
  const attribute = new Attribute(array, 8), storage = N.storage(attribute, schema, 1).toAtomic();
  const step = N.uniform(1, 'uint'), material = new T.MeshBasicNodeMaterial();
  material.colorNode = N.Fn(() => {
    const index = N.uint(N.screenCoordinate.x.div(width / count)).min(count - 1);
    const record = storage;
    N.atomicAdd(record.get('hits'), step);
    N.atomicAdd(record.get('balance'), N.int(-2));
    return record.get('color').mul(record.get('gain')).add(N.vec3(index.toFloat().mul(.125), 0, 0));
  })();
  const mesh = new T.Mesh(new T.PlaneGeometry(2, 2), material); mesh.frustumCulled = false; scene.add(mesh);
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 20); camera.position.z = 5;
  const kernel = N.Fn(() => {
    const record = storage;
    N.atomicAdd(record.get('hits'), N.uint(7));
    N.atomicAdd(record.get('balance'), N.int(11));
  })().compute(1, [1]);
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  let frame = 0, totalStep = 0, initial = null;
  renderer.render = (...args) => {
    if (args[0] !== scene || args[1] !== camera) return render(...args);
    frame++;
    if (frame === 8) step.value = 2;
    if (frame === 10) {
      floats[3] = .875; attribute.addUpdateRange(3, 1); attribute.needsUpdate = true;
    }
    if (frame === 13) step.value = 3;
    totalStep += step.value;
    const result = render(...args);
    renderer.compute(kernel);
    return result;
  };
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] !== target) return pixels;
    const data = await renderer.getArrayBufferAsync(attribute), gpu = new Uint32Array(data);
    const signed = new Int32Array(data), gpuFloats = new Float32Array(data), fragments = width * height;
    if (gpu.length !== words.length) throw Error('Atomic struct readback lost its padded source layout');
    if (gpu[0] !== 3 + fragments * totalStep + 7 * frame || signed[1] !== -50 + frame * (11 - 2 * fragments))
      throw Error(`Atomic fragment/compute ordering diverged at frame ${frame}: ${gpu[0]}, ${signed[1]}`);
    if (gpu[2] !== 0x12340000 || gpuFloats[3] !== (frame >= 10 ? .875 : .5))
      throw Error('Partial host uploads damaged GPU-owned counters or companion fields');
    if (words[0] !== 3 || words[1] !== ((-50) >>> 0))
      throw Error('GPU atomic writes were copied into the public CPU array');
    const colors = new Set();
    for (let i = 0; i < pixels.length; i += 4) colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
    if (colors.size !== count) throw Error('Every atomic record must supply its visible companion color');
    if (frame === 1) initial = pixels.slice();
    if (frame >= 10 && initial && pixels.every((value, i) => value === initial[i]))
      throw Error('The requested companion-field upload did not change the rendered result');
    return pixels;
  };
  return {scene, camera};
}

// Exactly one fragment writes ordinary members. Other invocations touch only
// the atomic counter, so this does not depend on racing non-atomic writes or
// reads. A later compute dispatch and render pass consume the produced fields.
function ordinaryMembers(T, renderer, floatBacking) {
  const N = T.TSL, scene = new T.Scene(), consumer = new T.Scene(), target = renderer.getRenderTarget();
  const width = target.width, height = target.height;
  const schema = N.struct({hits: {type: 'uint', atomic: true}, balance: {type: 'int', atomic: true},
    tag: 'uint', gain: 'float', color: 'vec3', padding: 'float', basis: 'mat3'}, 'WritableFragmentRecord');
  if (schema.getLength() !== 20) throw Error('Writable record fixture lost its padded source stride');
  const array = floatBacking ? new Float32Array(20) : new Uint32Array(20);
  const words = new Uint32Array(array.buffer), floats = new Float32Array(array.buffer);
  words.set([3, (-50) >>> 0, 0x12340000]); floats[6] = .375;
  const Attribute = floatBacking ? T.StorageInstancedBufferAttribute : T.StorageBufferAttribute;
  const attribute = new Attribute(array, 20), storage = N.storage(attribute, schema, 1).toAtomic();
  const step = N.uniform(.25), material = new T.MeshBasicNodeMaterial();
  material.colorNode = N.Fn(() => {
    N.If(N.screenCoordinate.x.lessThan(1).and(N.screenCoordinate.y.lessThan(1)), () => {
      storage.get('gain').assign(step);
      storage.get('tag').addAssign(N.uint(3));
      storage.get('color').xy.assign(N.vec2(step, N.float(1).sub(step)));
      storage.get('basis').assign(N.mat3(1));
      storage.get('basis').element(0).assign(N.vec3(step, .25, .75));
      storage.get('basis').element(1).yz.assign(N.vec2(.125, .375));
      storage.get('basis').element(N.uint(1)).x.addAssign(.5);
    });
    N.atomicAdd(storage.get('hits'), N.uint(1));
    return N.vec3(.125, .25, .5);
  })();
  const geometry = new T.PlaneGeometry(2, 2), mesh = new T.Mesh(geometry, material);
  mesh.frustumCulled = false; scene.add(mesh);
  const displayed = new T.MeshBasicNodeMaterial();
  displayed.colorNode = N.select(N.screenCoordinate.x.lessThan(width / 2),
    storage.get('color').mul(storage.get('gain')),
    N.vec3(storage.get('basis').element(0).x, storage.get('basis').element(1).x, storage.get('basis').element(2).z.mul(.5)));
  const displayedMesh = new T.Mesh(geometry, displayed); displayedMesh.frustumCulled = false; consumer.add(displayedMesh);
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 20); camera.position.z = 5;
  const kernel = N.Fn(() => {
    storage.get('tag').addAssign(N.uint(7));
    storage.get('gain').mulAssign(.5);
    storage.get('basis').element(2).z.addAssign(.125);
  })().compute(1, [1]);
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  let frame = 0, initial = null;
  renderer.render = (...args) => {
    if (args[0] !== scene || args[1] !== camera) return render(...args);
    frame++;
    if (frame === 8) step.value = .5;
    if (frame === 13) step.value = .75;
    if (frame === 10) {
      floats[6] = .875; attribute.addUpdateRange(6, 1); attribute.needsUpdate = true;
    }
    render(...args); renderer.compute(kernel); return render(consumer, camera);
  };
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] !== target) return pixels;
    const data = await renderer.getArrayBufferAsync(attribute), gpu = new Uint32Array(data), gpuFloats = new Float32Array(data);
    if (gpu.length !== 20 || gpu[0] !== 3 + width * height * frame || gpu[1] !== ((-50) >>> 0) || gpu[2] !== 0x12340000 + 10 * frame)
      throw Error(`Ordinary fragment record ordering diverged at frame ${frame}: ${Array.from(gpu.slice(0, 3))}`);
    const expected = new Map([[3, step.value * .5], [4, step.value], [5, 1 - step.value], [6, frame >= 10 ? .875 : .375],
      [7, 0], [8, step.value], [9, .25], [10, .75], [11, 0], [12, .5], [13, .125], [14, .375], [15, 0],
      [16, 0], [17, 0], [18, 1.125], [19, 0]]);
    for (const [index, value] of expected) if (gpuFloats[index] !== value)
      throw Error(`Ordinary fragment field ${index} differs at frame ${frame}: ${gpuFloats[index]} vs ${value}`);
    if (words[0] !== 3 || words[2] !== 0x12340000 || floats[3] !== 0)
      throw Error('Fragment record stores were copied into the public CPU array');
    if (frame === 1) initial = pixels.slice();
    if (frame >= 8 && initial && pixels.every((value, i) => value === initial[i]))
      throw Error('The fragment-produced values did not change the later rendered image');
    return pixels;
  };
  return {scene, camera};
}

export const atomicStructScenarios = {
  native_atomic_structs: (T, renderer) => atomicRecords(T, renderer, false),
  native_atomic_structs_float_backing: (T, renderer) => atomicRecords(T, renderer, true),
  native_atomic_struct_member_stores: (T, renderer) => ordinaryMembers(T, renderer, false),
  native_atomic_struct_member_stores_float_backing: (T, renderer) => ordinaryMembers(T, renderer, true),
};
