// The same public scene executes on pinned r186 and the owned native renderer.
// Large translations cancel on the CPU before model/view uniforms reach f32.
function largeScene(T, renderer, {orthographic = false, toggles = false} = {}) {
  const origin = new T.Vector3(2 ** 28, -(2 ** 27), 2 ** 26);
  const scene = new T.Scene(); scene.background = new T.Color(0x172131);
  const camera = orthographic ? new T.OrthographicCamera(-2.3, 2.3, 1.725, -1.725, .1, 30)
    : new T.PerspectiveCamera(45, 4 / 3, .1, 30);
  camera.position.copy(origin).add(new T.Vector3(.375, .25, 5.6)); camera.lookAt(origin);
  renderer.highPrecision = true;
  const root = new T.Group(); root.position.copy(origin); scene.add(root);
  const geometry = new T.SphereGeometry(.48, 24, 16);
  const materials = [new T.MeshBasicMaterial({color: 0xe88742}),
    new T.MeshNormalMaterial(), new T.MeshStandardMaterial({color: 0x6dc8b2, roughness: .43})];
  const meshes = materials.map((material, index) => {
    const mesh = new T.Mesh(geometry, material);
    mesh.position.set((index - 1) * 1.28, .45, 0);
    mesh.scale.set(.83 + index * .09, 1.16 - index * .08, .67 + index * .14);
    mesh.rotation.set(.13 + index * .18, -.24 + index * .35, .17);
    root.add(mesh); return mesh;
  });
  const nativeMaterial = new T.MeshBasicNodeMaterial();
  nativeMaterial.colorNode = T.TSL.normalView.mul(.27).add(T.TSL.vec3(.4, .5, .55));
  const nativeMesh = new T.Mesh(new T.BoxGeometry(.5, .4, .35), nativeMaterial);
  nativeMesh.position.set(-1.65, -.72, 0); nativeMesh.rotation.set(.2, .4, -.1); root.add(nativeMesh);
  const sprite = new T.Sprite(new T.SpriteMaterial({color: 0x6d92ef}));
  sprite.position.set(-.73, -.72, 0); sprite.scale.set(.4, .3, 1); root.add(sprite);
  const pointGeometry = new T.BufferGeometry().setAttribute('position', new T.Float32BufferAttribute([
    -.2, -.1, 0, .05, .12, 0, .26, -.06, 0,
  ], 3));
  const points = new T.Points(pointGeometry, new T.PointsMaterial({color: 0xe7cc62, size: .09}));
  points.position.set(.25, -.72, 0); root.add(points);
  const lineGeometry = new T.BufferGeometry().setAttribute('position', new T.Float32BufferAttribute([
    -.3, -.14, 0, -.13, .16, 0, .12, -.04, 0, .3, .13, 0,
  ], 3));
  const line = new T.Line(lineGeometry, new T.LineBasicMaterial({color: 0xe993b2}));
  line.position.set(1.3, -.72, 0); root.add(line);
  scene.add(new T.AmbientLight(0xdde8ff, .8));
  const light = new T.DirectionalLight(0xffd3a7, 2);
  light.position.copy(origin).add(new T.Vector3(-2, 3, 4)); light.target.position.copy(origin);
  scene.add(light, light.target);
  let frame = -1;
  scene.onBeforeRender = () => {
    ++frame; // compileAsync observes the same Scene hook before the first draw.
    if (toggles && frame === 7) pointGeometry.attributes.position.array[0] += .3;
    if (toggles && frame === 8) renderer.highPrecision = false;
    if (toggles && frame === 9) root.traverse(object => { if (object.material) object.material.needsUpdate = true; });
    if (toggles && frame === 10) renderer.contextNode.needsUpdate = true;
    if (toggles && frame === 12) {
      renderer.highPrecision = true; renderer.contextNode.needsUpdate = true;
      pointGeometry.attributes.position.needsUpdate = true;
    }
    if (frame === 13) {
      camera.position.x += .3125; camera.position.y -= .1875; camera.lookAt(origin);
      meshes[1].rotation.y += .37; meshes[2].position.x -= .21875;
    }
  };
  return {scene, camera};
}

export const highPrecisionScenarios = {
  native_high_precision(T, renderer) { return largeScene(T, renderer); },
  native_high_precision_orthographic(T, renderer) { return largeScene(T, renderer, {orthographic: true}); },
  native_high_precision_toggles(T, renderer) { return largeScene(T, renderer, {toggles: true}); },
};
