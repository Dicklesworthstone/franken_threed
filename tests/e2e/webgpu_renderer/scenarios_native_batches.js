// The source BatchedMesh owns ranges, visibility, sorting and matrix/color data.
function stage(T, perspective = false) {
  const scene = new T.Scene(); scene.background = new T.Color(0x182535);
  const camera = perspective ? new T.PerspectiveCamera(43, 4 / 3, .1, 40)
    : new T.OrthographicCamera(-2.2, 2.2, 1.65, -1.65, .1, 30);
  camera.position.set(perspective ? 4.5 : 0, perspective ? 3.7 : 0, perspective ? 6 : 5);
  camera.lookAt(0, perspective ? .4 : 0, 0);
  return {scene, camera};
}

function transform(T, x, y, z, angle = 0, scale = 1) {
  return new T.Matrix4().compose(new T.Vector3(x, y, z),
    new T.Quaternion().setFromEuler(new T.Euler(angle * .7, angle, angle * .3)),
    new T.Vector3(scale * .9, scale * 1.2, scale * .8));
}

function batch(T, {indexed = true, colors = true, material = new T.MeshBasicMaterial()} = {}) {
  const mesh = new T.BatchedMesh(12, 2048, 4096, material);
  const geometries = [new T.BoxGeometry(.55, .55, .55), new T.SphereGeometry(.33, 16, 10)]
    .map(geometry => indexed ? geometry : geometry.toNonIndexed());
  const geometryIds = geometries.map(geometry => mesh.addGeometry(geometry));
  const ids = [];
  for (let i = 0; i < 6; i++) {
    const id = mesh.addInstance(geometryIds[i % 2]); ids.push(id);
    mesh.setMatrixAt(id, transform(T, (i % 3 - 1) * 1.1, i < 3 ? .6 : -.55, (i % 3) * .15, .2 + i * .28));
    if (colors) mesh.setColorAt(id, new T.Color().setHSL(.03 + i * .14, .7, .52));
  }
  mesh.frustumCulled = false;
  return {mesh, ids, geometryIds, material};
}

function light(T, scene) {
  const key = new T.DirectionalLight(0xffe3c4, 2.4); key.position.set(-3, 4, 5);
  scene.add(key, new T.HemisphereLight(0xadcfff, 0x514134, .6)); return key;
}

export const nativeBatchScenarios = {
  native_batches_indexed(T) {
    const result = stage(T), source = batch(T, {material: new T.MeshStandardMaterial({roughness: .4})});
    source.mesh.setVisibleAt(source.ids[4], false);
    source.mesh.setMatrixAt(source.ids[5], transform(T, 50, 0, 0));
    // The native BatchedMesh backend draws its own ranges, not this mesh range.
    source.mesh.geometry.setDrawRange(0, 0);
    result.scene.add(source.mesh); light(T, result.scene); return result;
  },

  native_batches_nonindexed_updates(T, renderer) {
    const result = stage(T), material = new T.MeshBasicNodeMaterial({transparent: true,
      opacity: .58, side: T.DoubleSide, depthWrite: false});
    material.colorNode = T.TSL.color(0xffffff);
    const source = batch(T, {indexed: false, colors: false, material});
    let initializedColors = false;
    source.mesh.setCustomSort(function(list) {
      if (!initializedColors) {
        // The first batch color allocation changes the source shader graph.
        // Source onBeforeRender must run before selecting that graph.
        for (const [i, id] of source.ids.entries()) this.setColorAt(id, new T.Color().setHSL(.03 + i * .14, .7, .52));
        initializedColors = true;
      }
      list.sort((a, b) => b.z - a.z);
    });
    source.mesh.setVisibleAt(source.ids[3], false);
    source.mesh.setMatrixAt(source.ids[5], transform(T, 50, 0, 0));
    result.scene.add(source.mesh);
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && ++frame === 8) {
        source.mesh.deleteInstance(source.ids[1]);
        const replacement = source.mesh.addInstance(source.geometryIds[0]);
        if (replacement !== source.ids[1]) throw new Error('Source batch did not reuse the freed instance ID');
        source.mesh.setVisibleAt(source.ids[2], false); source.mesh.setVisibleAt(source.ids[3], true);
        source.mesh.setGeometryIdAt(source.ids[4], source.geometryIds[1]);
        for (const [index, id] of [source.ids[0], replacement, source.ids[3], source.ids[4]].entries()) {
          source.mesh.setMatrixAt(id, transform(T, (index - 1.5) * .36, (index % 2 ? -.16 : .16), index * .3, .3 + index * .25, 1.6));
          source.mesh.setColorAt(id, new T.Color().setHSL(.08 + index * .22, .7, .52));
        }
        source.mesh.setCustomSort(function(list) { list.sort((a, b) => b.index - a.index); });
      }
      return render(...args);
    };
    return result;
  },

  native_batches_targets(T, renderer) {
    const result = stage(T), writer = stage(T), source = batch(T);
    writer.scene.add(source.mesh);
    const targets = Array.from({length: 3}, () => new T.RenderTarget(128, 96,
      {depthBuffer: false, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
    for (const [i, target] of targets.entries()) {
      renderer.initRenderTarget(target);
      const panel = new T.Mesh(new T.PlaneGeometry(1.32, .99), new T.MeshBasicMaterial({map: target.texture}));
      panel.position.x = (i - 1) * 1.43; result.scene.add(panel);
    }
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && (++frame === 1 || frame === 8)) {
        const previous = renderer.getRenderTarget(), previousMip = renderer.getActiveMipmapLevel();
        try {
          for (const [i, target] of targets.entries()) {
            for (const [j, id] of source.ids.entries()) {
              source.mesh.setVisibleAt(id, (j + i) % 3 !== 0);
              source.mesh.setMatrixAt(id, transform(T, (j % 3 - 1) * 1.1 + i * .08,
                (j < 3 ? .6 : -.55) + i * .08, (j % 3) * .12, (frame === 8 ? .7 : .15) + i * .6 + j * .2));
              source.mesh.setColorAt(id, new T.Color().setHSL(.04 + j * .12 + i * .18 + frame * .01, .7, .55));
            }
            source.mesh.setCustomSort(i === 1 ? function(list) { list.sort((a, b) => b.index - a.index); } : null);
            renderer.setRenderTarget(target); render(writer.scene, writer.camera);
          }
        } finally { renderer.setRenderTarget(previous, 0, previousMip); }
        // Every cold request must retain the accepted texture version and draw
        // ranges even when the public batch changes before preparation ends.
        for (const id of source.ids) {
          source.mesh.setMatrixAt(id, transform(T, 99, 99, 99));
          source.mesh.setColorAt(id, new T.Color(0)); source.mesh.setVisibleAt(id, false);
        }
      }
      return render(...args);
    };
    return result;
  },

  native_batches_shadows(T, renderer) {
    const result = stage(T, true), source = batch(T, {material: new T.MeshStandardNodeMaterial({roughness: .65})});
    source.mesh.castShadow = true; source.mesh.receiveShadow = true;
    for (const [index, id] of source.ids.entries())
      source.mesh.setMatrixAt(id, transform(T, (index % 3 - 1) * 1.25, .5 + (index % 2) * .35,
        index < 3 ? -.75 : .75, .15 + index * .3, 1.25));
    const ground = new T.Mesh(new T.PlaneGeometry(8, 8), new T.MeshStandardNodeMaterial({color: 0x9caba7, roughness: 1}));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
    result.scene.add(source.mesh, ground);
    const key = light(T, result.scene); key.castShadow = true;
    key.shadow.mapSize.set(128, 128); key.shadow.bias = -.001; key.shadow.normalBias = .02;
    Object.assign(key.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4, near: .2, far: 20});
    key.shadow.camera.updateProjectionMatrix();
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && args[1] === result.camera && ++frame === 8) {
        source.mesh.setVisibleAt(source.ids[2], false);
        source.mesh.setMatrixAt(source.ids[0], transform(T, -.8, 1.1, -.3, .85, 1.6));
        source.mesh.setColorAt(source.ids[4], new T.Color(0x477ce8));
        key.position.x = -1.8;
      }
      return render(...args);
    };
    return result;
  },
};
