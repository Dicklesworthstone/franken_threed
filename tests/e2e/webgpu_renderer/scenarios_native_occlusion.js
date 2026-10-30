// Observe genuine completed GPU queries from source object callbacks. Status
// panels make undefined/visible/occluded results part of the pixel comparison.
function stage(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x17232d);
  const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20);
  camera.position.z = 5; camera.updateMatrixWorld();
  return {scene, camera};
}

function panel(T, scene, x) {
  const material = new T.MeshBasicMaterial({color: 0x665588, depthTest: false, depthWrite: false});
  const mesh = new T.Mesh(new T.PlaneGeometry(.7, .32), material);
  mesh.position.set(x, -1, 1); mesh.renderOrder = 100; scene.add(mesh);
  return value => material.color.setHex(value === undefined ? 0x665588 : value ? 0xee4488 : 0x33cc77);
}

function observe(renderer, object, expected, label, setPanel) {
  const value = renderer.isOccluded(object);
  if (value !== expected)
    throw new Error(`${label}: expected occlusion ${String(expected)}, got ${String(value)}`);
  setPanel(value);
}

function outside(renderer, object) {
  if (renderer.isOccluded(object) !== null)
    throw new Error('isOccluded outside an active source render context must return null');
}

export const nativeOcclusionScenarios = {
  native_occlusion_array_camera(T, renderer) {
    const result = stage(T), {scene} = result;
    const children = [-.2, .2].map((x, index) => {
      const child = result.camera.clone(); child.position.x = x;
      child.lookAt(0, 0, 0); child.updateMatrixWorld();
      child.viewport = new T.Vector4(index * 64, 0, 64, 96);
      return child;
    });
    const camera = result.camera = new T.ArrayCamera(children);
    const occluder = new T.Mesh(new T.PlaneGeometry(2.6, 1.4), new T.MeshBasicMaterial({color: 0x425a83}));
    occluder.position.set(0, .25, .7); occluder.renderOrder = -10; scene.add(occluder);
    const mesh = new T.Mesh(new T.PlaneGeometry(.7, .7), new T.MeshBasicMaterial({color: 0xefac45}));
    mesh.position.y = .25; mesh.occlusionTest = true; mesh.frustumCulled = false; scene.add(mesh);
    const setPanel = panel(T, scene, 0);
    let frame = 0, calls = 0;
    mesh.onBeforeRender = (r, selectedScene, selectedCamera) => {
      if (frame === 0) return;
      calls++;
      if (selectedScene !== scene || selectedCamera !== camera)
        throw new Error('ArrayCamera occlusion callbacks must retain the parent scene/camera');
      // The source ArrayCamera branch submits geometry but never opens a
      // query. Its allocated, unused query slots still resolve/map with normal
      // GPU latency; after that first map the empty object result is false,
      // even though the mesh is fully hidden in both views.
      observe(r, mesh, frame <= 2 ? undefined : false, `ArrayCamera frame ${frame}`, setPanel);
    };
    const render = renderer.render.bind(renderer);
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++; calls = 0; outside(renderer, mesh);
      const value = render(...args);
      if (calls !== 1) throw new Error(`ArrayCamera callbacks: expected 1, got ${calls}`);
      outside(renderer, mesh); return value;
    };
    return result;
  },

  native_occlusion_groups(T, renderer) {
    const result = stage(T), {scene} = result;
    const occluder = new T.Mesh(new T.PlaneGeometry(1.15, 1.15), new T.MeshBasicMaterial({color: 0x425a83}));
    occluder.position.set(-.7, .25, .7); occluder.renderOrder = -10; scene.add(occluder);
    const geometry = new T.PlaneGeometry(.7, .7);
    geometry.clearGroups(); geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 1);
    const materials = [new T.MeshBasicMaterial({color: 0xefac45}), new T.MeshBasicMaterial({color: 0x49afd4})];
    let frame = 0, calls = 0;
    const meshes = [-.7, .7].map((x, index) => {
      const mesh = new T.Mesh(geometry, materials); mesh.position.set(x, .25, 0);
      mesh.renderOrder = index; mesh.occlusionTest = true;
      const setPanel = panel(T, scene, x);
      mesh.onBeforeRender = r => {
        if (frame === 0) return; // compileAsync has callbacks but no query submission.
        calls++;
        const occluderRight = frame >= 10 && frame <= 14;
        const expected = frame <= 2 ? undefined : index === (occluderRight ? 1 : 0);
        observe(r, mesh, expected, `grouped mesh ${index}, frame ${frame}`, setPanel);
      };
      scene.add(mesh); return mesh;
    });
    const render = renderer.render.bind(renderer);
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++; calls = 0; outside(renderer, meshes[0]);
      if (frame === 8) occluder.position.x = .7;
      if (frame === 13) occluder.position.x = -.7;
      const value = render(...args);
      if (calls !== 4) throw new Error(`Grouped occlusion callbacks: expected 4, got ${calls}`);
      outside(renderer, meshes[0]); return value;
    };
    return result;
  },

  native_occlusion_batch_shared_targets(T, renderer) {
    const result = stage(T), {scene, camera} = result;
    const originalTarget = renderer.getRenderTarget();
    const alternateTarget = new T.RenderTarget(128, 96, {type: T.UnsignedByteType});
    alternateTarget.texture.colorSpace = T.LinearSRGBColorSpace;
    renderer.initRenderTarget(alternateTarget);
    const occluder = new T.Mesh(new T.PlaneGeometry(2.2, 1.2), new T.MeshBasicMaterial({color: 0x425a83}));
    occluder.position.set(0, .25, .7); occluder.renderOrder = -10; scene.add(occluder);
    const material = new T.MeshBasicMaterial(), batch = new T.BatchedMesh(4, 16, 24, material);
    const geometry = new T.PlaneGeometry(.65, .65), geometryId = batch.addGeometry(geometry);
    for (const [index, x] of [-.6, .6].entries()) {
      const id = batch.addInstance(geometryId);
      batch.setMatrixAt(id, new T.Matrix4().makeTranslation(x, .25, 0));
      batch.setColorAt(id, new T.Color(index ? 0x49afd4 : 0xefac45));
    }
    batch.frustumCulled = false; batch.perObjectFrustumCulled = false;
    batch.occlusionTest = true; scene.add(batch);
    const setPanel = panel(T, scene, 0);
    let frame = 0, lane = 'prepare', calls = 0;
    batch.setCustomSort(list => {
      list.sort((a, b) => b.index - a.index);
      if (frame === 0) return;
      calls++;
      // Equal-shaped targets share source RenderContext state. Each main draw
      // observes the preceding completed alternate-target query even though
      // this main draw's batch is visible. Two render calls happen per frame.
      const expected = frame === 1 ? undefined : !(frame >= 9 && frame <= 12);
      observe(renderer, batch, expected, `batch ${lane}, frame ${frame}`, setPanel);
    });
    const render = renderer.render.bind(renderer);
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++; calls = 0; outside(renderer, batch);
      try {
        lane = 'alternate'; renderer.setRenderTarget(alternateTarget);
        occluder.position.x = frame >= 8 && frame <= 11 ? 3 : 0;
        render(scene, camera);
        lane = 'main'; renderer.setRenderTarget(originalTarget); occluder.position.x = 3;
        const value = render(...args);
        if (calls !== 2) throw new Error(`Batch source selections: expected 2, got ${calls}`);
        outside(renderer, batch); return value;
      } finally { renderer.setRenderTarget(originalTarget); lane = 'idle'; }
    };
    const dispose = renderer.dispose.bind(renderer);
    renderer.dispose = () => { alternateTarget.dispose(); batch.dispose(); geometry.dispose(); return dispose(); };
    return result;
  },
};
