// Public native loading states; the same factories execute on pinned r186 and F3D.
function cubePixels(T, size, phase = 0) {
  return Array.from({length: 6}, (_, face) => {
    const data = new Uint16Array(size * size * 4);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const rgba = [.08 + ((face + phase) % 6) * .12 + x / size * .1,
        .12 + ((face * 3 + phase) % 5) * .13 + y / size * .12,
        .18 + ((face * 2 + phase * 3) % 5) * .11, 1];
      data.set(rgba.map(T.DataUtils.toHalfFloat), (y * size + x) * 4);
    }
    return new T.DataTexture(data, size, size, T.RGBAFormat, T.HalfFloatType);
  });
}
function layerPixels(phase = 0) {
  const data = new Uint8Array(2 * 2 * 3 * 4);
  for (let layer = 0; layer < 3; layer++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++)
    data.set([35 + layer * 63 + phase * 17, 41 + y * 89 + phase * 31, 57 + x * 71 + phase * 19, 255],
      ((layer * 2 + y) * 2 + x) * 4);
  return data;
}
export const nativeTextureLoadingScenarios = {
  native_layered_texture_loading(T, renderer) {
    const N = T.TSL, scene = new T.Scene(), camera = new T.OrthographicCamera(-3.2, 3.2, 2.4, -2.4, .1, 20);
    scene.background = new T.Color(0x162335); camera.position.z = 4;
    const cube = new T.CubeTexture(); cube.type = T.HalfFloatType; cube.colorSpace = T.LinearSRGBColorSpace;
    cube.minFilter = cube.magFilter = T.NearestFilter; cube.generateMipmaps = false;
    const array = new T.DataArrayTexture(null, 2, 2, 3);
    const volume = new T.Data3DTexture(layerPixels(), 2, 2, 3);
    volume.source.dataReady = false; volume.needsUpdate = true;
    const geometry = new T.PlaneGeometry(.88, .88), textures = [cube, array, volume];
    const add = (node, x, y) => {
      const material = new T.MeshBasicNodeMaterial(); material.colorNode = node.rgb;
      const mesh = new T.Mesh(geometry, material); mesh.position.set(x, y, 0); scene.add(mesh);
    };
    const u = N.uv().x.mul(1.4).sub(.7), v = N.uv().y.mul(1.4).sub(.7);
    const directions = [N.vec3(1, v, u), N.vec3(-1, v, u), N.vec3(u, 1, v),
      N.vec3(u, -1, v), N.vec3(u, v, 1), N.vec3(u, v, -1)];
    directions.forEach((direction, face) => add(N.cubeTexture(cube, direction), face - 2.5, 1.4));
    for (let layer = 0; layer < 3; layer++) {
      add(N.texture(array, N.uv()).depth(layer), (layer - 1) * 2, 0);
      add(N.texture3D(volume, N.vec3(N.uv(), (layer + .5) / 3)), (layer - 1) * 2, -1.4);
    }
    let frame = -1;
    scene.onBeforeRender = () => {
      ++frame;
      if (frame === 3) {
        cube.images = cubePixels(T, 2); array.image.data = layerPixels(); volume.image.data = layerPixels();
        volume.source.dataReady = true; // Every accepted loading version stays unchanged.
      }
      if (frame === 8) for (const texture of textures) texture.needsUpdate = true;
      if (frame === 10) {
        cube.images = cubePixels(T, 2, 1); array.image.data = layerPixels(1); volume.image.data = layerPixels(1);
        for (const texture of textures) { texture.source.dataReady = false; texture.needsUpdate = true; }
      }
      if (frame === 12) for (const texture of textures) texture.source.dataReady = true;
      if (frame === 13) for (const texture of textures) texture.needsUpdate = true;
      if (frame === 20) for (const texture of textures) {
        texture.dispose(); texture.source.dataReady = false; texture.needsUpdate = true;
      }
    };
    return {scene, camera};
  },
  async native_pmrem_texture_warmup(T, renderer) {
    const generator = new T.PMREMGenerator(renderer);
    await generator.compileCubemapShader(); await generator.compileEquirectangularShader();
    const cube = new T.CubeTexture(cubePixels(T, 16)); cube.type = T.HalfFloatType;
    cube.minFilter = cube.magFilter = T.LinearFilter; cube.colorSpace = T.LinearSRGBColorSpace;
    cube.generateMipmaps = false; cube.needsUpdate = true;
    const data = new Uint16Array(64 * 32 * 4);
    for (let y = 0; y < 32; y++) for (let x = 0; x < 64; x++)
      data.set([.1 + x / 64, .15 + y / 32, .45 + Math.sin(x / 64 * Math.PI * 2) * .2, 1].map(T.DataUtils.toHalfFloat),
        (y * 64 + x) * 4);
    const panorama = new T.DataTexture(data, 64, 32, T.RGBAFormat, T.HalfFloatType);
    panorama.mapping = T.EquirectangularReflectionMapping; panorama.colorSpace = T.LinearSRGBColorSpace;
    panorama.minFilter = panorama.magFilter = T.LinearFilter; panorama.needsUpdate = true;
    const outputs = [generator.fromCubemap(cube), generator.fromEquirectangular(panorama)];
    // Sampling the two real CubeUV atlases covers both producers without
    // introducing unrelated material-environment selection into this profile.
    const scene = new T.Scene(), camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20);
    camera.position.z = 4; scene.background = new T.Color(0x162335);
    const geometry = new T.PlaneGeometry(3.8, 1.2);
    for (const [index, output] of outputs.entries()) {
      const material = new T.MeshBasicNodeMaterial(); material.colorNode = T.TSL.texture(output.texture, T.TSL.uv()).rgb;
      const mesh = new T.Mesh(geometry, material); mesh.position.y = index === 0 ? .72 : -.72; scene.add(mesh);
    }
    return {scene, camera};
  },
};
