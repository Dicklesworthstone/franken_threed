// The same authored ClippingGroup hierarchy runs on both WebGPU renderers.
function clippingScene(T, renderer, coverage) {
  const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0x263a4d);
  const camera = new T.PerspectiveCamera(43, 4 / 3, .1, 20);
  camera.position.set(2.3, 1.7, 5.8); camera.lookAt(0, .15, 0);
  const outer = new T.ClippingGroup();
  outer.clippingPlanes = [new T.Plane(new T.Vector3(0, 1, 0), .4)];
  outer.clipShadows = true; scene.add(outer);
  const inner = new T.ClippingGroup();
  inner.clippingPlanes = [new T.Plane(new T.Vector3(1, 0, 0), .68),
    new T.Plane(new T.Vector3(0, -1, 0), .2)];
  inner.clipIntersection = true; inner.clipShadows = false; outer.add(inner);
  const right = new T.ClippingGroup();
  right.clippingPlanes = [new T.Plane(new T.Vector3(-1, .25, 0).normalize(), .95)];
  right.clipShadows = true; scene.add(right);

  // Shared material/geometry must retain different inherited clipping contexts.
  const material = new T.MeshStandardNodeMaterial({roughness: .42, side: T.DoubleSide,
    alphaToCoverage: coverage});
  material.colorNode = N.normalLocal.mul(.25).add(N.vec3(.35, .48, .45));
  material.forceSinglePass = true;
  const geometry = new T.SphereGeometry(.74, 40, 28), leftMesh = new T.Mesh(geometry, material);
  const rightMesh = new T.Mesh(geometry, material);
  leftMesh.position.set(-.8, .2, 0); rightMesh.position.set(.85, .2, -.15);
  leftMesh.castShadow = rightMesh.castShadow = true; inner.add(leftMesh); right.add(rightMesh);
  const box = new T.Mesh(new T.BoxGeometry(.6, .9, .6),
    new T.MeshPhongMaterial({color: 0xd29b48, shininess: 35, alphaToCoverage: coverage}));
  box.position.set(-.05, .15, -.7); box.rotation.y = .4; box.castShadow = true; outer.add(box);
  const glass = new T.Mesh(new T.PlaneGeometry(.65, 1.45),
    new T.MeshBasicNodeMaterial({color: 0xb877c9, transparent: true, opacity: .35,
      depthWrite: false, alphaToCoverage: coverage}));
  glass.position.set(.5, .15, .8); glass.rotation.z = .18; right.add(glass);
  const ground = new T.Mesh(new T.PlaneGeometry(5, 4),
    new T.MeshStandardNodeMaterial({color: 0x9b9c91, roughness: 1}));
  ground.rotation.x = -Math.PI / 2; ground.position.y = -.65; ground.receiveShadow = true; scene.add(ground);
  const light = new T.DirectionalLight(0xffe1bd, 2.4); light.position.set(-2, 4, 3);
  light.castShadow = true; light.shadow.mapSize.set(128, 128);
  Object.assign(light.shadow.camera, {left: -3, right: 3, top: 3, bottom: -3, near: .1, far: 15});
  light.shadow.camera.updateProjectionMatrix(); light.shadow.normalBias = .025;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
  scene.add(light, new T.HemisphereLight(0xbdd5ee, 0x4d3827, .6));
  const update = frame => {
    if (frame === 8) {
      outer.enabled = false; inner.clipIntersection = false;
      inner.clippingPlanes[0].constant = .9; right.clippingPlanes[0].constant = 1.15;
      camera.position.set(-1.4, 1.1, 5.2); camera.lookAt(0, .15, 0);
    }
    if (frame === 18) {
      outer.enabled = true; inner.clipIntersection = true; inner.clipShadows = true;
      outer.clippingPlanes.push(new T.Plane(new T.Vector3(0, 0, 1), .45));
      outer.clippingPlanes[0].constant = .25; inner.clippingPlanes[0].constant = .65;
      right.clippingPlanes[0].constant = .9; right.clipShadows = false;
      leftMesh.rotation.z = .3; rightMesh.rotation.y = -.6; glass.position.x = .75;
      camera.position.set(1.7, 1.35, 5.4); camera.lookAt(0, .1, 0);
    }
  };
  return {scene, camera, update, outer, inner, right};
}

export const nativeClippingScenarios = {
  native_clipping_groups(T, renderer) {
    const result = clippingScene(T, renderer, false), render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && args[1] === result.camera) result.update(++frame);
      return render(...args);
    };
    return result;
  },
  native_clipping_msaa(T, renderer) {
    if (renderer.samples !== 4) throw new Error('Clipping coverage requires source constructor antialias');
    const result = clippingScene(T, renderer, true), scenePass = T.TSL.pass(result.scene, result.camera);
    const pipeline = new T.RenderPipeline(renderer, scenePass);
    const scene = new T.Scene(), camera = new T.PerspectiveCamera(), render = renderer.render.bind(renderer);
    let frame = 0;
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      result.update(++frame); return pipeline.render();
    };
    return {scene, camera, source: result, scenePass, pipeline};
  },
};
