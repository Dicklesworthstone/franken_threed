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

// Ordinary texture painting follows the pinned partial-update example: each
// copy changes GPU storage while the material and its CPU image remain intact.
// The same public calls and controls execute on the source and candidate.
function sampledCopies(T, renderer, imageDestination, aliases, bundled = false, preserveHistory = false) {
  const scene = new T.Scene(), width = 8, height = 8;
  const group = bundled ? new T.BundleGroup() : scene;
  if (bundled) { group.static = true; scene.add(group); }
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const offset = (y * width + x) * 4;
    pixels.set([20 + x * 23, 30 + y * 21, 190 - x * 13 - y * 7, 255], offset);
  }
  let destination, ready = Promise.resolve();
  if (imageDestination) {
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
    const image = new Image(); image.src = canvas.toDataURL('image/png'); ready = image.decode();
    destination = new T.Texture(image);
  } else destination = new T.DataTexture(pixels, width, height);
  destination.generateMipmaps = false;
  destination.minFilter = destination.magFilter = imageDestination ? T.LinearFilter : T.NearestFilter;
  destination.colorSpace = imageDestination ? T.SRGBColorSpace : T.NoColorSpace;
  destination.needsUpdate = true;
  const alias = aliases ? destination.clone() : null;
  const sourceBytes = Uint8Array.of(240, 20, 40, 255, 20, 230, 60, 255,
    30, 40, 220, 255, 220, 210, 20, 255);
  const source = new T.DataTexture(sourceBytes, 2, 2);
  source.colorSpace = destination.colorSpace; source.needsUpdate = true;
  const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 10); camera.position.z = 3;
  const add = (texture, x, size) => {
    const mesh = new T.Mesh(new T.PlaneGeometry(size, 2.4), new T.MeshBasicMaterial({map: texture}));
    mesh.position.x = x; mesh.frustumCulled = false; group.add(mesh);
  };
  if (alias) { add(destination, -1, 1.8); add(alias, 1, 1.8); }
  else add(destination, 0, 3.2);
  const compile = renderer.compileAsync.bind(renderer);
  renderer.compileAsync = async (...args) => { await ready; return compile(...args); };
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    frame++;
    if (frame === 2 && preserveHistory) {
      // A copy initializes storage only when its Texture version requests it.
      // Keep both an already uploaded alias and changed CPU/source identities
      // GPU-stale until the explicit needsUpdate at frame 10.
      pixels.fill(77);
      destination.source = new T.Source({width, height, data: pixels});
    }
    if (frame === 2 || frame === 7)
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(frame === 2 ? 1 : 4, 1));
    if (frame === 8) {
      sourceBytes.set([240, 120, 20, 255], 8); sourceBytes.set([20, 120, 240, 255], 12);
      source.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination,
        new T.Box2(new T.Vector2(0, 1), new T.Vector2(2, 2)), new T.Vector2(2, 5));
    }
    if (frame === 10) {
      if (!imageDestination) {
        for (let i = 0; i < pixels.length; i += 4) pixels.set([150, 60 + i % 128, 90, 255], i);
        // Native r186 deliberately ignores DataTexture's WebGL row ranges.
        destination.addUpdateRange(0, 4);
      }
      if (bundled) {
        destination.source = new T.Source({width, height, data: pixels});
        destination.minFilter = destination.magFilter = T.LinearFilter;
        destination.flipY = true;
      }
      destination.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(5, 5));
    }
    if (frame === 12) {
      destination.dispose();
      if (bundled) group.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(0, 6));
    }
    if (frame === 13) {
      source.flipY = true; source.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(5, 0));
    }
    return render(...args);
  };
  return {scene, camera};
}

export const storageReadScenarios = {
  native_storage_read_mips(T, renderer) { return storageReads(T, renderer, false, false); },
  native_storage_read_array_mips(T, renderer) { return storageReads(T, renderer, true, false); },
  native_storage_read_uint_mips(T, renderer) { return storageReads(T, renderer, false, true); },
  native_sampled_copy_data(T, renderer) { return sampledCopies(T, renderer, false, false); },
  native_sampled_copy_image(T, renderer) { return sampledCopies(T, renderer, true, false); },
  native_sampled_copy_aliases(T, renderer) { return sampledCopies(T, renderer, false, true, false, true); },
  native_sampled_copy_bundle(T, renderer) { return sampledCopies(T, renderer, false, false, true); },
  native_sampled_copy_history(T, renderer) { return sampledCopies(T, renderer, false, false, false, true); },
};
