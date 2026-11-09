// Real PMREM producers followed by structurally equivalent material consumers.
// Source r186 shares the first environment graph for each builder cache key.
async function makeScene(T, renderer, physical = false, reverse = false) {
  const cube = new T.CubeTexture(Array.from({length: 6}, (_, face) => {
    const data = new Uint16Array(16 * 16 * 4);
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++)
      data.set([.08 + face * .12 + x / 160, .12 + (face * 3 % 5) * .13 + y / 128,
        .18 + (face * 2 % 5) * .11, 1].map(T.DataUtils.toHalfFloat), (y * 16 + x) * 4);
    return new T.DataTexture(data, 16, 16, T.RGBAFormat, T.HalfFloatType);
  }));
  cube.type = T.HalfFloatType; cube.minFilter = cube.magFilter = T.LinearFilter;
  cube.colorSpace = T.LinearSRGBColorSpace; cube.generateMipmaps = false; cube.needsUpdate = true;
  const data = new Uint16Array(64 * 32 * 4);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 64; x++)
    data.set([.12 + x / 64, .15 + y / 32, .45 + Math.sin(x / 64 * Math.PI * 2) * .2, 1]
      .map(T.DataUtils.toHalfFloat), (y * 64 + x) * 4);
  const panorama = new T.DataTexture(data, 64, 32, T.RGBAFormat, T.HalfFloatType);
  panorama.mapping = T.EquirectangularReflectionMapping; panorama.colorSpace = T.LinearSRGBColorSpace;
  panorama.minFilter = panorama.magFilter = T.LinearFilter; panorama.needsUpdate = true;
  const generator = new T.PMREMGenerator(renderer);
  const outputs = [generator.fromCubemap(cube), generator.fromEquirectangular(panorama)];
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 4 / 3, .1, 30);
  camera.position.z = 6; scene.background = new T.Color(0x162335);
  const Material = physical ? T.MeshPhysicalNodeMaterial : T.MeshStandardNodeMaterial, materials = [];
  for (let i = 0; i < 4; i++) {
    const material = new Material({metalness: .75, roughness: [.15, .35, .55, .8][i],
      envMap: outputs[(i < 2 ? 0 : 1) ^ Number(reverse)].texture, envMapIntensity: .75});
    if (physical) { material.clearcoat = .5; material.clearcoatRoughness = .25; material.sheen = .25; }
    const mesh = new T.Mesh(new T.SphereGeometry(.75, 24, 16), material);
    mesh.position.set(i % 2 ? 1 : -1, i < 2 ? 1 : -1, 0); scene.add(mesh); materials.push(material);
  }
  let frame = -1;
  scene.onBeforeRender = () => {
    frame++;
    if (frame === 7) {
      materials[0].envMap = outputs[Number(!reverse)].texture;
      materials[0].needsUpdate = true; // Same source cache key retains its constructed environment.
    }
    if (frame === 8) materials[0].dispose(); // Other render objects still retain that builder state.
    if (frame === 10) materials[2].envMapIntensity = 1.25;
    if (frame === 12) { materials[3].color.setRGB(.75, .3, .15); materials[3].roughness = .25; }
  };
  return {scene, camera};
}

export const nativeEnvironmentCacheScenarios = {
  native_pbr_environment_cache: (T, renderer) => makeScene(T, renderer),
  native_physical_environment_cache: (T, renderer) => makeScene(T, renderer, true),
  native_pbr_environment_cache_panorama_first: (T, renderer) => makeScene(T, renderer, false, true),
};
