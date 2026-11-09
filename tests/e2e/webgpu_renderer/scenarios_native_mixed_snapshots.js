function stage(T, renderer) {
  const scene = new T.Scene(), camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 12);
  camera.position.z = 5; renderer.sortObjects = false;
  return {scene, camera};
}

export const nativeMixedSnapshotScenarios = {
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
