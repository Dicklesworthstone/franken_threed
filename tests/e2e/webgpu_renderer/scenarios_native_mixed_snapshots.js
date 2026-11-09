function stage(T, renderer) {
  const scene = new T.Scene(), camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 12);
  camera.position.z = 5; renderer.sortObjects = false;
  return {scene, camera};
}

function ordinaryInstances(T, renderer, capacity) {
  const result = stage(T, renderer), {scene} = result;
  const geometry = new T.PlaneGeometry(.65, .8), material = new T.MeshBasicMaterial();
  const instances = new T.InstancedMesh(geometry, material, capacity); instances.count = 2; instances.frustumCulled = false;
  instances.setMatrixAt(0, new T.Matrix4().makeTranslation(-.8, 0, 0));
  instances.setMatrixAt(1, new T.Matrix4().makeTranslation(.8, 0, 0));
  instances.setColorAt(0, new T.Color(.125, .375, .625)); instances.setColorAt(1, new T.Color(.25, .5, .75));
  const position = geometry.attributes.position, matrices = instances.instanceMatrix, colors = instances.instanceColor;
  const initialPosition = position.array.slice(), initialMatrices = matrices.array.slice(), initialColors = colors.array.slice();
  const authored = new T.MeshBasicNodeMaterial(); authored.colorNode = T.TSL.vec3(.625, .375, .125);
  const native = new T.Mesh(new T.PlaneGeometry(.4, .4), authored); native.position.y = .9;
  const ordinary = new T.Mesh(new T.PlaneGeometry(.4, .4), new T.MeshBasicMaterial({color: 0x446633})); ordinary.position.y = -.9;
  const render = renderer.render.bind(renderer); let frame = 0, before = 0, after = 0;
  renderer.render = (...args) => {
    if (args[0] !== scene) return render(...args);
    frame++; scene.add(instances); position.array.set(initialPosition); matrices.array.set(initialMatrices); colors.array.set(initialColors);
    // Ordinary materials retain object uniforms on SHARED refresh. A source
    // FULL refresh reads all small matrices but only requested large ranges.
    matrices.array[12] = -.8 + frame / 100;
    if (frame >= 7) { position.array[0] = .1; colors.array[0] = 99; matrices.array[28] = 99; }
    if (frame === 7) {
      instances.onBeforeRender = () => { before++; }; instances.onAfterRender = () => { after++; };
    }
    if (frame >= 7 && frame < 10 || frame >= 12) scene.add(native, ordinary);
    else scene.remove(native, ordinary);
    if (frame === 10) {
      position.array[0] = -.25; position.addUpdateRange(0, 1); position.needsUpdate = true;
      colors.array[0] = .875; colors.addUpdateRange(0, 1); colors.needsUpdate = true;
      matrices.addUpdateRange(12, 1); matrices.needsUpdate = true;
    }
    if (frame === 12) {
      position.array[1] = .25; position.addUpdateRange(1, 1); position.needsUpdate = true;
      colors.array[1] = .25; colors.addUpdateRange(1, 1); colors.needsUpdate = true;
      matrices.array[28] = .65; matrices.addUpdateRange(28, 1); matrices.needsUpdate = true;
    }
    renderer._nodes.nodeFrame.update();
    const value = render(...args), callbacks = Math.max(0, frame - 6);
    if (before !== callbacks || after !== callbacks) throw Error('Instance callbacks must run once per selected draw');
    position.array.fill(99); matrices.array.fill(99); colors.array.fill(99);
    return value;
  };
  return result;
}

export const nativeMixedSnapshotScenarios = {
  native_ordinary_instance_snapshots: (T, renderer) => ordinaryInstances(T, renderer, 3),
  native_large_instance_snapshots: (T, renderer) => ordinaryInstances(T, renderer, 1025),
  native_mixed_accepted_inputs(T, renderer) {
    const result = stage(T, renderer), {scene} = result;
    const geometry = new T.PlaneGeometry(1, 1), material = new T.MeshBasicMaterial();
    const ordinary = new T.Mesh(geometry, material); ordinary.frustumCulled = false;
    const authored = new T.MeshBasicNodeMaterial(); authored.colorNode = T.TSL.vec3(.25, .5, .75);
    const native = new T.Mesh(new T.PlaneGeometry(.7, .7), authored); native.position.set(.9, .4, 0);
    const invalid = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial());
    const render = renderer.render.bind(renderer); let frame = 0;
    // compileAsync sees an empty scene. The first accepted mixed render must
    // retain its inputs even while its shader pipeline is still being built.
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++; scene.remove(invalid); scene.add(ordinary, native); ordinary.geometry = geometry;
      ordinary.position.set(-.7 + frame / 100, -.2, 0);
      material.color.setRGB(frame % 2 ? .125 : .75, .375, .625);
      geometry.attributes.position.setX(0, -.5 + (frame % 3) / 10);
      geometry.attributes.position.needsUpdate = true;
      const value = render(...args);
      ordinary.geometry = null; ordinary.position.set(20, 20, 20); ordinary.updateMatrixWorld();
      material.color.setRGB(1, 0, 0); geometry.attributes.position.setX(0, 99);
      scene.remove(ordinary); scene.add(invalid);
      return value;
    };
    return result;
  },

  native_mixed_source_updates(T, renderer) {
    const result = stage(T, renderer), {scene} = result;
    const geometry = new T.PlaneGeometry(.75, .8), position = geometry.attributes.position, initial = position.array.slice();
    const shared = new T.PlaneGeometry(.75, .8); shared.setAttribute('position', position);
    const material = new T.MeshBasicMaterial();
    const before = new T.Mesh(geometry, material), later = new T.Mesh(shared, material);
    before.frustumCulled = later.frustumCulled = false;
    let frame = 0, updates = 0;
    const tint = T.TSL.uniform(.5).onObjectUpdate(() => {
      updates++; material.color.setRGB(.625, .375, .125);
      before.matrixWorld.elements[12] = 9; later.matrixWorld.elements[12] = .8 + frame / 100;
      position.setX(0, .1); position.needsUpdate = true;
      return .5;
    });
    tint.updateAfterType = 'object'; tint.updateAfter = () => material.color.setRGB(.75, .5, .25);
    const authored = new T.MeshBasicNodeMaterial(); authored.colorNode = T.TSL.vec3(tint);
    const native = new T.Mesh(new T.PlaneGeometry(.5, .5), authored); native.position.y = -.7;
    const render = renderer.render.bind(renderer);
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++; scene.add(before, native, later);
      before.geometry = geometry; later.geometry = shared;
      before.position.set(-1.1, .25, 0); later.position.set(1.1, .25, 0);
      material.color.setRGB(.125, .375, .625); position.array.set(initial); position.needsUpdate = true;
      const value = render(...args);
      if (updates !== frame) throw Error('NodeFrame updates must complete once per accepted render');
      before.geometry = later.geometry = null; before.matrixWorld.elements[12] = later.matrixWorld.elements[12] = 99;
      material.color.setRGB(1, 0, 0); position.array.fill(99);
      return value;
    };
    return result;
  },
};
