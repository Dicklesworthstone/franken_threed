// Same source inputs execute on the pinned reference and the new backend.
function map(T, channel, kind) {
  const pixels = new Uint8Array(4 * 4 * 4);
  for (let i = 0; i < 16; i++) {
    const n = (i + Math.floor(i / 4)) % 4;
    pixels.set(kind === 'metal'
      ? [255, 255 - n * 55, [20, 180, 80, 250][n], 255]
      : [255 - n * 50, [45, 225, 100, 180][n], 255, 255], i * 4);
  }
  const texture = new T.DataTexture(pixels, 4, 4);
  texture.channel = channel; texture.wrapS = texture.wrapT = T.RepeatWrapping;
  texture.needsUpdate = true; return texture;
}
function camera(T) {
  const c = new T.PerspectiveCamera(50, 320 / 240, .1, 50);
  c.position.set(0, 0, 4); return c;
}
function lights(T, scene) {
  const sun = new T.DirectionalLight(0xffedda, 2.3); sun.position.set(-2, 3, 4);
  const point = new T.PointLight(0x77aaff, 9, 0, 2); point.position.set(2, -.5, 2);
  scene.add(sun, point, new T.HemisphereLight(0x657799, 0x332211, .7));
}
export const pbrMapScenarios = {
  pbr_maps_independent(T) {
    const scene = new T.Scene(); scene.background = new T.Color(0x121820);
    const geometry = new T.SphereGeometry(.47, 48, 32);
    for (const name of ['uv1', 'uv2', 'uv3']) geometry.setAttribute(name, geometry.attributes.uv.clone());
    for (let i = 0; i < 6; i++) {
      const metal = map(T, 1, 'metal'), rough = map(T, 2, 'rough');
      metal.repeat.set(2, 1); metal.offset.set(.125, .25);
      rough.repeat.set(1, 2); rough.rotation = .35;
      const C = i === 5 ? T.MeshStandardNodeMaterial : i === 3 ? T.MeshPhysicalMaterial : T.MeshStandardMaterial;
      const material = new C({color: i < 3 ? 0xa65a2e : 0x286a9a, metalness: .85, roughness: .8,
        metalnessMap: i === 1 ? null : metal, roughnessMap: i === 2 ? null : i === 4 ? metal : rough});
      if (i === 3) { material.clearcoat = .5; material.clearcoatRoughness = .3; material.ior = 2; }
      const mesh = new T.Mesh(geometry, material);
      mesh.position.set((i % 3 - 1) * 1.05, i < 3 ? .55 : -.55, 0); scene.add(mesh);
    }
    lights(T, scene); return {scene, camera: camera(T)};
  },
  pbr_maps_updates(T, renderer) {
    const result = pbrMapScenarios.pbr_maps_independent(T), render = renderer.render.bind(renderer);
    let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) for (const mesh of result.scene.children.filter(o => o.isMesh)) {
        const m = mesh.material; m.metalness = .55; m.roughness = .95;
        if (m.metalnessMap) { m.metalnessMap.offset.set(.3, .6); m.metalnessMap.rotation = -.5; }
        if (m.roughnessMap) {
          m.roughnessMap.repeat.set(3, 1); m.roughnessMap.rotation = .7;
          for (let i = 1; i < m.roughnessMap.image.data.length; i += 4) m.roughnessMap.image.data[i] = 255 - m.roughnessMap.image.data[i];
          m.roughnessMap.needsUpdate = true;
        }
      }
      return render(...args);
    };
    return result;
  },
  pbr_maps_layers(T) {
    const scene = new T.Scene(); scene.background = new T.Color(0x121820); scene.fog = new T.Fog(0x121820, 3, 9);
    const geometry = new T.SphereGeometry(.65, 48, 32);
    for (const name of ['uv1', 'uv2', 'uv3']) geometry.setAttribute(name, geometry.attributes.uv.clone());
    const material = new T.MeshPhysicalMaterial({color: 0xa86a40, roughness: .75, metalness: .7,
      clearcoat: .7, clearcoatRoughness: .4, specularIntensity: .9, ior: 1.8});
    for (const [i, name] of ['map', 'emissiveMap', 'aoMap', 'clearcoatMap', 'clearcoatRoughnessMap',
      'alphaMap', 'specularIntensityMap', 'specularColorMap', 'metalnessMap', 'roughnessMap'].entries()) {
      material[name] = map(T, i % 4, name === 'metalnessMap' ? 'metal' : 'rough');
      material[name].offset.set(i * .07, i * .11);
    }
    for (const name of ['normalMap', 'clearcoatNormalMap']) {
      const texture = new T.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1);
      texture.channel = 3; texture.needsUpdate = true; material[name] = texture;
    }
    material.emissive.setRGB(.02, .01, .03); material.aoMapIntensity = .3;
    const mesh = new T.InstancedMesh(geometry, material, 2);
    mesh.setMatrixAt(0, new T.Matrix4().makeTranslation(-.75, 0, 0));
    mesh.setMatrixAt(1, new T.Matrix4().makeTranslation(.75, 0, 0));
    scene.add(mesh); lights(T, scene); return {scene, camera: camera(T)};
  },
};
