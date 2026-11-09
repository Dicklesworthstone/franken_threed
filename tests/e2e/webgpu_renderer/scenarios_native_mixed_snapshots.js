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

function ordinaryDeformation(T, renderer, skinned, materialKeys = false) {
  const result = stage(T, renderer), {scene} = result;
  const geometry = new T.PlaneGeometry(1.5, 1.7, 4, 6), position = geometry.attributes.position;
  const material = materialKeys ? new T.MeshStandardMaterial({color: 0x59a6cf, roughness: 0}) : new T.MeshBasicMaterial({color: 0x59a6cf});
  if (materialKeys) scene.add(new T.AmbientLight(0xffffff, 1));
  const initial = position.array.slice();
  geometry.morphTargetsRelative = !skinned;
  geometry.morphAttributes.position = [0, 1].map(target => new T.Float32BufferAttribute(Array.from(initial, (value, i) =>
    (skinned ? value : 0) + (i % 3 === target ? Math.sin(initial[i - i % 3 + 1] * 2.4) * .35 : 0)), 3));
  const targets = geometry.morphAttributes.position, targetData = targets.map(attribute => attribute.array.slice());
  let root, tip, updates = 0;
  const mesh = skinned ? new T.SkinnedMesh(geometry, material) : new T.Mesh(geometry, material); mesh.frustumCulled = false;
  if (skinned) {
    const indices = [], weights = [];
    for (let i = 0; i < position.count; i++) {
      const weight = Math.round((position.getY(i) + .85) / 1.7 * 255);
      indices.push(0, 1, 0, 0); weights.push(255 - weight, weight, 0, 0);
    }
    geometry.setAttribute('skinIndex', new T.Uint16BufferAttribute(indices, 4));
    geometry.setAttribute('skinWeight', new T.Uint8BufferAttribute(weights, 4, true));
    root = new T.Bone(); tip = new T.Bone(); root.position.y = -.85; tip.position.y = 1.7;
    root.add(tip); mesh.add(root); mesh.bind(new T.Skeleton([root, tip]));
    const update = mesh.skeleton.update; mesh.skeleton.update = function () { updates++; return update.call(this); };
  }
  const authored = new T.MeshBasicNodeMaterial(); authored.colorNode = T.TSL.vec3(.625, .375, .125);
  const native = new T.Mesh(new T.PlaneGeometry(.4, .4), authored); native.position.set(1.2, .9, 0);
  const render = renderer.render.bind(renderer); let frame = 0, before = 0, after = 0;
  renderer.render = (...args) => {
    if (args[0] !== scene) return render(...args);
    frame++; scene.add(mesh); position.array.set(initial);
    targets.forEach((attribute, i) => attribute.array.set(targetData[i]));
    if (frame >= 7) {
      position.array[0] += .3;
      for (let i = 0; i < targets[0].array.length; i += 3) targets[0].array[i] += materialKeys && frame >= 12 ? .7 : .35;
    }
    mesh.morphTargetInfluences = frame < 7 ? [.25, .5] : frame < 8 ? [.5, -.25] : frame < 10 ? [.85, .25] : frame < 12 ? [.75, .15] : [.1, .9];
    mesh.static = frame === 8 || frame === 9;
    if (skinned) { root.rotation.z = Math.sin(frame / 7) * .15; tip.rotation.z = Math.sin(frame / 11) * .5; }
    if (frame === 7) {
      mesh.onBeforeRender = () => { before++; }; mesh.onAfterRender = () => { after++; };
      if (materialKeys) geometry.dispose();
    }
    if (materialKeys && frame === 8) material.roughness = .5;
    if (frame >= 7 && frame < 10 || frame >= 12) scene.add(native); else scene.remove(native);
    if (frame === 10) {
      if (!materialKeys) { position.array[1] -= .2; position.addUpdateRange(1, 1); position.needsUpdate = true; }
      targets[0].needsUpdate = true; // Target edits do not rebuild the existing source morph graph.
      if (materialKeys) material.needsUpdate = true;
    }
    if (frame === 12) geometry.dispose();
    if (materialKeys && frame === 12) { material.roughness = .75; material.needsUpdate = true; }
    if (frame === 13) {
      if (materialKeys) targets[0] = targets[0].clone();
      material.needsUpdate = true;
    }
    if (frame === 18) { material.alphaHash = true; material.needsUpdate = true; }
    renderer._nodes.nodeFrame.update(); const value = render(...args), callbacks = Math.max(0, frame - 6);
    if (before !== callbacks || after !== callbacks || skinned && updates !== frame)
      throw Error('Deformation callbacks and skeleton updates must retain source multiplicity');
    position.array.fill(99); targets.forEach(attribute => attribute.array.fill(99)); mesh.morphTargetInfluences.fill(99);
    if (skinned) mesh.skeleton.boneMatrices.fill(99);
    return value;
  };
  return result;
}

export const nativeMixedSnapshotScenarios = {
  native_morph_material_keys: (T, renderer) => ordinaryDeformation(T, renderer, false, true),
  native_ordinary_morph_snapshots: (T, renderer) => ordinaryDeformation(T, renderer, false),
  native_ordinary_skin_morph_snapshots: (T, renderer) => ordinaryDeformation(T, renderer, true),
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
