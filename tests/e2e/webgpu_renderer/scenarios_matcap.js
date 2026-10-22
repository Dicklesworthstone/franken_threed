// The same public material/texture inputs execute on both native renderers.
function texture(T, kind = 'matcap', variant = 0) {
  const size = 16, pixels = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const u = x / (size - 1), v = y / (size - 1), i = (y * size + x) * 4;
    if (kind === 'normal') pixels.set([100 + Math.round(u * 55), 105 + Math.round(v * 45), 245, 255], i);
    else if (kind === 'alpha') pixels.set([(x + y) % 5 < 2 ? 95 : 240, 255, 0, 255], i);
    else if (kind === 'map') pixels.set([140 + Math.round(u * 100), 150 + Math.round(v * 90), 220, 255], i);
    else pixels.set(variant
      ? [220 - Math.round(v * 170), 40 + Math.round(u * 170), 70 + Math.round(v * 170), x < 4 ? 0 : 255]
      : [35 + Math.round(u * 200), 45 + Math.round(v * 170), 210 - Math.round(u * 110), y > 12 ? 0 : 255], i);
  }
  const result = new T.DataTexture(pixels, size, size);
  result.minFilter = result.magFilter = T.LinearFilter;
  result.wrapS = result.wrapT = T.RepeatWrapping;
  if (kind === 'matcap' || kind === 'map') result.colorSpace = T.SRGBColorSpace;
  result.needsUpdate = true; return result;
}
function camera(T, orthographic = false) {
  const result = orthographic
    ? new T.OrthographicCamera(-2.1, 2.1, 1.575, -1.575, .1, 50)
    : new T.PerspectiveCamera(48, 320 / 240, .1, 50);
  result.position.set(.7, .35, 4.4); result.lookAt(0, 0, 0); return result;
}
function materials(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x182435);
  const geometry = new T.SphereGeometry(.56, 48, 32);
  const generated = texture(T); generated.channel = 3;
  generated.center.set(.5, .5); generated.rotation = .18;
  // Matcap coordinates are generated in view space: channel 3 needs no uv3.
  const materials = [
    new T.MeshMatcapMaterial({color: 0xc48652}),
    new T.MeshMatcapNodeMaterial({color: 0xffffff, matcap: generated}),
    new T.MeshMatcapMaterial({color: 0xa8cce8, matcap: texture(T, 'matcap', 1)}),
    new T.MeshMatcapMaterial({color: 0xe8c894, matcap: texture(T), map: texture(T, 'map'),
      normalMap: texture(T, 'normal'), normalScale: new T.Vector2(.7, .55), alphaMap: texture(T, 'alpha'),
      transparent: true, opacity: .9, alphaTest: .42}),
  ];
  const meshes = materials.map((material, i) => {
    const mesh = new T.Mesh(geometry, material);
    mesh.position.set(i % 2 ? .72 : -.72, i < 2 ? .65 : -.65, 0);
    mesh.rotation.set(.15 * i, .25 * i, -.1 * i); scene.add(mesh); return mesh;
  });
  return {scene, camera: camera(T), meshes};
}
export const matcapScenarios = {
  native_matcap_materials(T) { return materials(T); },
  native_matcap_updates(T, renderer) {
    const result = materials(T), render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) {
        const [fallback, node, edited, replaced] = result.meshes.map(mesh => mesh.material);
        fallback.matcap = texture(T, 'matcap', 1); fallback.needsUpdate = true;
        node.matcap = null; node.needsUpdate = true;
        edited.matcap.matrixAutoUpdate = false;
        edited.matcap.matrix.setUvTransform(.17, -.13, .8, 1.1, -.4, .5, .5);
        const pixels = edited.matcap.image.data;
        for (let i = 0; i < pixels.length; i += 4) pixels[i] = 255 - pixels[i];
        edited.matcap.needsUpdate = true;
        replaced.matcap = texture(T, 'matcap', 1);
        replaced.matcap.offset.set(.13, .07); replaced.matcap.rotation = .5;
        replaced.map.offset.set(.2, .1); replaced.normalMap.repeat.set(2, 1);
      }
      return render(...args);
    };
    return result;
  },
  native_matcap_instanced(T, renderer) {
    const scene = new T.Scene(); scene.background = new T.Color(0x182435);
    const material = new T.MeshMatcapNodeMaterial({matcap: texture(T), normalMap: texture(T, 'normal'),
      normalScale: new T.Vector2(.45, .6)});
    const mesh = new T.InstancedMesh(new T.SphereGeometry(.48, 48, 32), material, 3);
    for (let i = 0; i < 3; i++) {
      const rotation = new T.Quaternion().setFromEuler(new T.Euler(.2 * i, -.3 * i, .15));
      mesh.setMatrixAt(i, new T.Matrix4().compose(new T.Vector3((i - 1) * 1.15, i === 1 ? -.35 : .35, i * -.15),
        rotation, new T.Vector3(1, .75 + i * .2, 1.2 - i * .15)));
      mesh.setColorAt(i, new T.Color(i === 0 ? 0xe8a878 : i === 1 ? 0x88c4ed : 0xb6df86));
    }
    mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true; scene.add(mesh);
    const result = {scene, camera: camera(T, true)}, render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) {
        result.camera.position.set(-.7, .9, 4.5); result.camera.lookAt(0, 0, 0);
        mesh.rotation.y = .3;
      }
      return render(...args);
    };
    return result;
  },
};
