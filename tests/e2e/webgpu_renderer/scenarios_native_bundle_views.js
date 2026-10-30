// The pinned ordinary-target BundleGroup encoder cannot set viewports. Keep
// child viewports absent at recording and observe inherited pass state during
// execution, including the selection retained across warm source edits.
function stage(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x17232d);
  const children = [-.22, .27].map(x => {
    const child = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20);
    child.position.set(x, 0, 5); child.lookAt(0, 0, 0); child.updateMatrixWorld();
    return child;
  });
  const camera = new T.ArrayCamera(children); camera.layers.enableAll();
  return {scene, camera};
}

function check(value, expected, label) {
  if (value !== expected) throw new Error(`${label}: expected ${expected}, got ${value}`);
}

export const nativeBundleViewScenarios = {
  native_bundle_array_camera(T, renderer) {
    const result = stage(T), {scene, camera} = result;
    const fixed = new T.BundleGroup(), dynamic = new T.BundleGroup(); dynamic.static = false;
    scene.add(fixed, dynamic);
    const frozen = new T.Mesh(new T.PlaneGeometry(.8, .9), new T.MeshBasicMaterial({color: 0xeb8844}));
    frozen.position.set(-.75, .2, 0); frozen.frustumCulled = false; fixed.add(frozen);
    let frame = 0, localUpdates = 0, fixedCalls = 0, dynamicCalls = 0;
    const material = new T.MeshBasicNodeMaterial({depthTest: false});
    material.colorNode = T.TSL.uniform(new T.Color()).onObjectUpdate(({camera: selected}) => {
      if (selected !== camera) throw new Error('Bundle updates must retain the parent ArrayCamera');
      if (frame) localUpdates++;
      return new T.Color().setRGB(frame < 10 ? .2 : .65, .65, frame < 10 ? .9 : .2);
    });
    const live = new T.Mesh(new T.PlaneGeometry(.85, .7), material);
    live.position.set(.7, .25, .1); live.frustumCulled = false; dynamic.add(live);
    frozen.onBeforeRender = () => { if (frame) fixedCalls++; };
    live.onBeforeRender = () => { if (frame) dynamicCalls++; };
    const ordinaryMaterial = new T.MeshBasicNodeMaterial(); ordinaryMaterial.colorNode = T.TSL.color(0x54799b);
    const ordinary = new T.Mesh(new T.PlaneGeometry(3.2, .3), ordinaryMaterial);
    ordinary.position.y = -1; ordinary.frustumCulled = false;
    ordinary.onBeforeRender = () => {
      // Bundles have already been recorded. Their viewport-less commands will
      // execute after these ordinary draws, inheriting the final selected view.
      camera.cameras[0].viewport = new T.Vector4(0, 0, 96, 96);
      camera.cameras[1].viewport = new T.Vector4(32, 8, 96, 80);
    };
    scene.add(ordinary);
    const render = renderer.render.bind(renderer);
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++; localUpdates = 0; fixedCalls = 0; dynamicCalls = 0;
      const rebuild = frame === 1 || frame === 8 || frame === 13;
      if (rebuild) for (const child of camera.cameras) child.viewport = undefined;
      if (frame === 7) {
        frozen.position.x = -.2; frozen.material.color.setHex(0xdb3988);
        live.layers.set(2); camera.cameras[0].layers.enable(2); camera.cameras[1].layers.set(3);
      }
      if (frame === 8) dynamic.needsUpdate = true;
      if (frame === 10) {
        camera.cameras[0].position.x = .5; camera.cameras[0].updateMatrixWorld();
        live.position.y = .55;
      }
      if (frame === 12) live.visible = false;
      if (frame === 13) {
        fixed.needsUpdate = true; dynamic.needsUpdate = true;
        live.visible = true; live.layers.set(0); camera.cameras[1].layers.set(0);
      }
      const value = render(...args);
      check(localUpdates, 1, `bundle node updates at frame ${frame}`);
      check(fixedCalls, frame === 1 || frame === 13 ? 1 : 0, `static bundle callbacks at frame ${frame}`);
      check(dynamicCalls, rebuild ? 1 : 0, `dynamic bundle callbacks at frame ${frame}`);
      return value;
    };
    return result;
  },

  native_bundle_batch_array_camera(T, renderer) {
    const result = stage(T), {scene, camera} = result;
    const group = new T.BundleGroup(); scene.add(group);
    let frame = 0, updates = 0, selections = 0;
    const material = new T.MeshBasicNodeMaterial({transparent: true, opacity: .48,
      side: T.DoubleSide, depthWrite: false});
    material.colorNode = T.TSL.uniform(1).onObjectUpdate(({camera: selected}) => {
      if (selected !== camera) throw new Error('Batched bundle updates must use the parent camera');
      if (frame) updates++;
      return frame < 10 ? 1 : .75;
    });
    const batch = new T.BatchedMesh(4, 96, 144, material), geometry = new T.BoxGeometry(1.1, .9, .45);
    const geometryId = batch.addGeometry(geometry), instances = [-.32, .3].map((x, index) => {
      const id = batch.addInstance(geometryId);
      batch.setMatrixAt(id, new T.Matrix4().makeTranslation(x, index ? -.12 : .15, index * .18));
      batch.setColorAt(id, new T.Color(index ? 0x48a9db : 0xeb864b));
      return id;
    });
    batch.frustumCulled = false; batch.perObjectFrustumCulled = false;
    batch.setCustomSort(list => {
      if (frame) selections++;
      list.sort((a, b) => frame < 8 ? b.index - a.index : a.index - b.index);
    });
    group.add(batch);
    const render = renderer.render.bind(renderer);
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++; updates = 0; selections = 0;
      if (frame === 7) {
        batch.setVisibleAt(instances[1], false); batch.layers.set(2);
        camera.cameras[0].layers.enable(2); camera.cameras[1].layers.set(3);
      }
      if (frame === 8) group.needsUpdate = true;
      if (frame === 10) {
        batch.setMatrixAt(instances[0], new T.Matrix4().makeTranslation(.2, .3, .1));
        batch.setColorAt(instances[0], new T.Color(0x97d65d));
        camera.cameras[0].position.x = .6; camera.cameras[0].updateMatrixWorld();
      }
      if (frame === 13) {
        group.needsUpdate = true; batch.setVisibleAt(instances[1], true);
        batch.layers.set(0); camera.cameras[1].layers.set(0);
      }
      const value = render(...args);
      check(updates, 2, `batch bundle updates per material side at frame ${frame}`);
      check(selections, frame === 1 || frame === 8 || frame === 13 ? 1 : 0,
        `batch bundle source selection at frame ${frame}`);
      return value;
    };
    const dispose = renderer.dispose.bind(renderer);
    renderer.dispose = () => { batch.dispose(); geometry.dispose(); return dispose(); };
    return result;
  },
};
