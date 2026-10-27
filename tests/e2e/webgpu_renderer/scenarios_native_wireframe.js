// Wireframe uses the source triangle-edge index while retaining native TSL.
function stage(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x162332);
  const camera = new T.OrthographicCamera(-2.2, 2.2, 1.65, -1.65, .1, 20);
  camera.position.set(0, 0, 5);
  const key = new T.DirectionalLight(0xffe4c8, 2.4); key.position.set(-2, 3, 4);
  scene.add(key, new T.HemisphereLight(0xa8d3ff, 0x493a2c, .65));
  return {scene, camera};
}

function surface(T, {ordinary = false, color = 0xe49b58} = {}) {
  const material = ordinary ? new T.MeshStandardMaterial({color, roughness: .5, wireframe: true})
    : new T.MeshStandardNodeMaterial({color, roughness: .5, wireframe: true});
  const N = T.TSL, phase = N.uniform(.2);
  material.positionNode = N.positionLocal.add(N.vec3(N.positionLocal.y.mul(4).add(phase).sin().mul(.14), 0, 0));
  return {material, phase};
}

export const nativeWireframeScenarios = {
  native_wireframe_materials(T) {
    const result = stage(T);
    for (const [i, geometry] of [new T.SphereGeometry(.82, 16, 10), new T.BoxGeometry(1.25, 1.25, 1, 4, 4, 3).toNonIndexed()].entries()) {
      const {material} = surface(T, {ordinary: i === 0, color: i ? 0x60b6cf : 0xe49b58});
      const mesh = new T.Mesh(geometry, material);
      mesh.position.x = i ? 1 : -1; mesh.rotation.set(.24, i ? -.42 : .3, .08);
      result.scene.add(mesh);
    }
    return result;
  },

  native_wireframe_targets(T, renderer) {
    const result = stage(T), writer = stage(T), source = surface(T, {ordinary: true});
    const geometry = new T.SphereGeometry(1.15, 14, 10), mesh = new T.Mesh(geometry, source.material);
    mesh.rotation.set(.3, .45, .15); writer.scene.add(mesh);
    const targets = Array.from({length: 3}, () => new T.RenderTarget(128, 96,
      {depthBuffer: true, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
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
            source.material.wireframe = i !== 1;
            source.phase.value = (frame === 8 ? 1.2 : .1) + i * .8;
            geometry.setDrawRange([0, 60, 30][i], [360, 270, 180][i]);
            renderer.setRenderTarget(target); render(writer.scene, writer.camera);
          }
        } finally { renderer.setRenderTarget(previous, 0, previousMip); }
        // Accepted draws retain both topology and source ranges through cold
        // pipeline preparation; wireframe multiplies those ranges by two.
        source.material.wireframe = false; source.phase.value = 9;
        geometry.setDrawRange(0, 0);
      }
      return render(...args);
    };
    return result;
  },

  native_wireframe_instances(T, renderer) {
    const result = stage(T), source = surface(T, {color: 0xffffff});
    const mesh = new T.InstancedMesh(new T.BoxGeometry(.55, .65, .45, 2, 2, 2), source.material, 4);
    const transform = new T.Object3D();
    for (let i = 0; i < 4; i++) {
      transform.position.set((i - 1.5) * .95, .65, .1 * (i % 2));
      transform.rotation.set(.2, .2 + i * .25, -.1); transform.updateMatrix();
      mesh.setMatrixAt(i, transform.matrix); mesh.setColorAt(i, new T.Color().setHSL(.05 + i * .2, .65, .55));
    }
    mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
    mesh.frustumCulled = false; result.scene.add(mesh);

    const geometry = new T.InstancedBufferGeometry().copy(new T.IcosahedronGeometry(.37, 0));
    geometry.setAttribute('instanceOffset', new T.InstancedBufferAttribute(new Float32Array([
      -1.4, -.7, 0, -.45, -.7, 0, .5, -.7, 0, 1.45, -.7, 0,
    ]), 3));
    geometry.setAttribute('instanceTint', new T.InstancedBufferAttribute(new Float32Array([
      .2, .8, .6, .8, .3, .6, .4, .5, .9, .8, .7, .2,
    ]), 3));
    geometry.instanceCount = 4;
    const material = new T.MeshBasicNodeMaterial({wireframe: true}), N = T.TSL;
    material.colorNode = N.attribute('instanceTint', 'vec3');
    material.positionNode = N.positionLocal.add(N.attribute('instanceOffset', 'vec3'));
    const custom = new T.Mesh(geometry, material); custom.frustumCulled = false; result.scene.add(custom);
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && ++frame === 8) {
        mesh.count = 3; geometry.instanceCount = 2; source.phase.value = 1.6;
        mesh.setColorAt(1, new T.Color(0xb9d747)); mesh.instanceColor.needsUpdate = true;
      }
      return render(...args);
    };
    return result;
  },

  native_wireframe_mixed_shadows(T, renderer) {
    const result = stage(T);
    result.camera = new T.PerspectiveCamera(43, 4 / 3, .1, 30);
    result.camera.position.set(4.5, 3.4, 6); result.camera.lookAt(0, .65, 0);
    const colors = new Uint8Array(8 * 8 * 4), normals = new Uint8Array(8 * 8 * 4);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const index = (y * 8 + x) * 4;
      colors.set([45 + x * 24, 190 - y * 18, 70 + y * 18, 255], index);
      normals.set([128 + Math.round(Math.sin(x) * 45), 128 + Math.round(Math.cos(y) * 40), 240, 255], index);
    }
    const matcap = new T.DataTexture(colors, 8, 8), normalMap = new T.DataTexture(normals, 8, 8);
    for (const texture of [matcap, normalMap]) {
      texture.minFilter = texture.magFilter = T.LinearFilter;
      texture.generateMipmaps = false; texture.needsUpdate = true;
    }
    matcap.colorSpace = T.SRGBColorSpace;
    const wire = new T.Mesh(new T.BoxGeometry(.9, 1.55, .9, 3, 6, 3),
      new T.MeshStandardMaterial({color: 0xe49b58, roughness: .6, wireframe: true}));
    wire.position.set(-1.2, .8, 0); wire.rotation.y = .35;
    const normal = new T.Mesh(new T.SphereGeometry(.62, 24, 16),
      new T.MeshNormalMaterial({normalMap, normalScale: new T.Vector2(.4, -.65)}));
    normal.position.set(.15, .65, -.45);
    const capped = new T.Mesh(new T.TorusKnotGeometry(.38, .13, 64, 12), new T.MeshMatcapMaterial({matcap}));
    capped.position.set(1.35, .8, .3); capped.rotation.set(.25, -.2, .1);
    for (const mesh of [wire, normal, capped]) { mesh.castShadow = true; result.scene.add(mesh); }
    const ground = new T.Mesh(new T.PlaneGeometry(8, 8), new T.MeshStandardMaterial({color: 0x9caba7, roughness: 1}));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; result.scene.add(ground);
    const key = result.scene.children.find(object => object.isDirectionalLight);
    key.position.set(-3, 6, 4); key.castShadow = true; key.shadow.mapSize.set(128, 128);
    key.shadow.bias = -.001; key.shadow.normalBias = .02;
    Object.assign(key.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4, near: .2, far: 20});
    key.shadow.camera.updateProjectionMatrix();
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
    return result;
  },
};
