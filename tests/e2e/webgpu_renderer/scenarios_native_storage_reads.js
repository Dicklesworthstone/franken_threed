// Identical source/candidate compute producers and graphics texel consumers.
// Every authored mip has distinct values; no automatic mip generation is used.
function storageReads(T, renderer, array, integer) {
  const N = T.TSL, scene = new T.Scene(), kernels = [], meshes = [];
  scene.background = new T.Color(0x182435);
  const texture = array ? new T.StorageArrayTexture(4, 4, 2) : new T.StorageTexture(4, 4);
  texture.type = integer ? T.UnsignedIntType : array ? T.FloatType : T.UnsignedByteType;
  texture.format = T.RGBAFormat; texture.colorSpace = T.NoColorSpace;
  texture.minFilter = texture.magFilter = T.NearestFilter;
  texture.generateMipmaps = true; texture.mipmapsAutoUpdate = false;
  const phase = N.uniform(integer ? 64 : .25, integer ? 'uint' : 'float');
  for (let mip = 0; mip < 3; mip++) {
    const size = 4 >> mip;
    kernels.push(N.Fn(() => {
      const layer = N.instanceIndex.div(size * size);
      const coordinates = N.uvec2(N.instanceIndex.mod(size), N.instanceIndex.div(size).mod(size));
      let output = N.storageTexture(texture).toWriteOnly().setMipLevel(mip);
      if (array) output = output.depth(layer);
      const value = integer ? N.uvec4(32 + mip * 64, phase, 192 - mip * 48, 255)
        : N.vec4(.125 + mip * .25, phase, layer.toFloat().mul(.125).add(.25), 1);
      N.textureStore(output, coordinates, value);
    })().compute(size * size * (array ? 2 : 1)));
    let input = N.storageTexture(texture, N.ivec2(0, 0)).setSampler(false).toReadOnly().setMipLevel(mip);
    if (array) input = input.depth(N.int(1));
    const color = integer ? N.vec3(input.rgb).div(255) : input.rgb;
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = color;
    material.positionNode = N.positionLocal.add(N.vec3(color.r.mul(.2), 0, 0));
    const mesh = new T.Mesh(new T.PlaneGeometry(.9, 1.4), material);
    mesh.position.x = (mip - 1) * 1.2; mesh.frustumCulled = false;
    scene.add(mesh); meshes.push(mesh);
  }
  const camera = new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 20);
  camera.position.z = 5;
  const compile = renderer.compileAsync.bind(renderer);
  renderer.compileAsync = async (...args) => { await renderer.computeAsync(kernels); return compile(...args); };
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    frame++;
    if (frame === 8 || frame === 13) {
      phase.value = integer ? frame === 8 ? 192 : 96 : frame === 8 ? .75 : .375;
      renderer.compute(kernels);
    }
    if (frame === 10) meshes[1].visible = false;
    if (frame === 12) meshes[1].visible = true;
    return render(...args);
  };
  return {scene, camera};
}

export const storageReadScenarios = {
  native_storage_read_mips(T, renderer) { return storageReads(T, renderer, false, false); },
  native_storage_read_array_mips(T, renderer) { return storageReads(T, renderer, true, false); },
  native_storage_read_uint_mips(T, renderer) { return storageReads(T, renderer, false, true); },
};
