function stage(T) {
  const scene = new T.Scene(), camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 12);
  camera.position.z = 5; camera.updateMatrixWorld();
  return {scene, camera};
}

function depthColor(T, image) {
  const d = image.r;
  return T.TSL.vec3(d, d.mul(.3), T.TSL.oneMinus(d));
}

function viewport(T, renderer, {samples = 0, arrayCamera = false} = {}) {
  const result = stage(T), {scene} = result;
  const target = renderer.getRenderTarget(); target.samples = samples; renderer.setRenderTarget(target);
  if (arrayCamera) {
    const children = [-.35, .35].map((x, i) => {
      const camera = result.camera.clone(); camera.position.x = x; camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
      camera.viewport = new T.Vector4(i * 64, 0, 64, 96); return camera;
    });
    result.camera = new T.ArrayCamera(children);
  }
  const prefix = new T.Mesh(new T.BoxGeometry(1.4, 1.4, 1), new T.MeshBasicMaterial({color: 0x448833}));
  prefix.renderOrder = -2; prefix.position.set(-.3, .15, 1); scene.add(prefix);
  const material = new T.MeshBasicNodeMaterial({depthTest: false, depthWrite: false});
  material.colorNode = depthColor(T, T.TSL.viewportDepthTexture());
  const display = new T.Mesh(new T.PlaneGeometry(4, 3), material);
  display.renderOrder = 1; display.frustumCulled = false; scene.add(display);
  const later = new T.Mesh(new T.PlaneGeometry(2.3, .25), new T.MeshBasicMaterial({color: 0xf3a327}));
  later.position.set(0, -.2, -.3); later.renderOrder = 2; scene.add(later);
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    if (args[0] === scene) {
      frame++; prefix.position.x = -.6 + .035 * frame;
      prefix.rotation.y = frame * .025;
    }
    return render(...args);
  };
  return result;
}

export const nativeDepthSnapshotScenarios = {
  native_viewport_depth: (T, renderer) => viewport(T, renderer),
  native_viewport_depth_msaa: (T, renderer) => viewport(T, renderer, {samples: 4}),
  native_viewport_depth_array_camera: (T, renderer) => viewport(T, renderer, {arrayCamera: true}),

  native_public_depth_float(T, renderer) {
    const result = stage(T), producer = stage(T);
    const target = new T.RenderTarget(128, 96, {depthTexture: new T.DepthTexture(128, 96, T.FloatType)});
    const snapshot = new T.DepthTexture(128, 96, T.FloatType);
    const prefix = new T.Mesh(new T.BoxGeometry(1.5, 1.4, 1), new T.MeshBasicMaterial({color: 0x448833}));
    prefix.position.z = 1; producer.scene.add(prefix);
    const material = new T.MeshBasicNodeMaterial({depthTest: false, depthWrite: false});
    material.colorNode = depthColor(T, T.TSL.texture(snapshot));
    result.scene.add(new T.Mesh(new T.PlaneGeometry(4, 3), material));
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] !== result.scene) return render(...args);
      frame++; prefix.position.x = -.6 + frame * .035; prefix.rotation.y = frame * .025;
      const output = renderer.getRenderTarget();
      renderer.setRenderTarget(target); render(producer.scene, producer.camera);
      renderer.copyFramebufferToTexture(snapshot);
      renderer.setRenderTarget(output); return render(...args);
    };
    return result;
  },
};
