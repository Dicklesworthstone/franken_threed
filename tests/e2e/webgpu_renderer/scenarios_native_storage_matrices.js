// Identical source/candidate kernels write column-aligned matrices; graphics
// reads the same storage for per-instance geometry and fragment color.
function matrices(T, renderer, dimension) {
  const N = T.TSL, scene = new T.Scene();
  scene.background = new T.Color(0x182435);
  const stride = dimension === 3 ? 12 : 4;
  const attribute = new T.StorageBufferAttribute(new Float32Array(2 * stride), stride);
  const storage = N.storage(attribute, `mat${dimension}`, 2), phase = N.uniform(.5);
  const kernel = N.Fn(() => {
    const shear = N.instanceIndex.toFloat().mul(.125);
    storage.element(N.instanceIndex).assign(dimension === 2
      ? N.mat2(phase, .125, shear, .75)
      : N.mat3(phase, .125, 0, shear, .75, 0, 0, 0, 1));
  })().compute(2, [2]);
  const matrix = storage.element(N.instanceIndex), material = new T.MeshBasicNodeMaterial();
  const transformed = dimension === 2
    ? N.vec3(matrix.mul(N.positionLocal.xy), N.positionLocal.z)
    : matrix.mul(N.positionLocal);
  material.positionNode = transformed.add(N.vec3(N.instanceIndex.toFloat().mul(1.6).sub(.8), 0, 0));
  const color = dimension === 2 ? N.vec3(matrix.mul(N.vec2(.8, .4)), .5) : matrix.mul(N.vec3(.8, .4, .5));
  material.colorNode = color;
  const mesh = new T.Mesh(new T.PlaneGeometry(1.4, 1.7), material);
  mesh.count = 2; mesh.frustumCulled = false; scene.add(mesh);
  const camera = new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 20); camera.position.z = 5;
  const compile = renderer.compileAsync.bind(renderer);
  renderer.compileAsync = async (...args) => { await renderer.computeAsync(kernel); return compile(...args); };
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    frame++;
    if (frame === 8 || frame === 13) {
      phase.value = frame === 8 ? .875 : .375;
      renderer.compute(kernel);
    }
    if (frame === 10) mesh.count = 1;
    if (frame === 12) mesh.count = 2;
    return render(...args);
  };
  return {scene, camera};
}

export const storageMatrixScenarios = {
  native_storage_mat2: (T, renderer) => matrices(T, renderer, 2),
  native_storage_mat3: (T, renderer) => matrices(T, renderer, 3),
};
