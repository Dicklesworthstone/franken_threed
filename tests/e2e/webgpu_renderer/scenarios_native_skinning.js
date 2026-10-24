// The same source skeletons and public pose updates execute on both renderers.
function rig(T, {color = 0xd98a43, detached = false, paletteSize = 2, ordinary = false} = {}) {
  const height = 1.8, geometry = new T.CylinderGeometry(.21, .27, height, 24, 24);
  const positions = geometry.attributes.position, indices = [], weights = [];
  for (let i = 0; i < positions.count; i++) {
    const weight = T.MathUtils.clamp((positions.getY(i) + height / 2) / height, 0, 1);
    // Use the LAST bone, so the texture fallback cannot pass by truncating its palette.
    indices.push(0, paletteSize - 1, 0, 0); weights.push(1 - weight, weight, 0, 0);
  }
  geometry.setAttribute('skinIndex', new T.Uint16BufferAttribute(indices, 4));
  geometry.setAttribute('skinWeight', new T.Float32BufferAttribute(weights, 4));
  const material = ordinary ? new T.MeshStandardMaterial({color, roughness: .4})
    : new T.MeshStandardNodeMaterial({color, roughness: .4});
  if (!ordinary) material.colorNode = T.TSL.uniform(new T.Color(color)).mul(
    T.TSL.uv().y.mul(14).sin().mul(.12).add(.86));
  const mesh = new T.SkinnedMesh(geometry, material), root = new T.Bone(), tip = new T.Bone();
  root.position.y = -height / 2; tip.position.y = height; root.add(tip); mesh.add(root);
  const bones = [root, ...Array.from({length: paletteSize - 2}, () => new T.Bone()), tip];
  const skeleton = new T.Skeleton(bones); mesh.bind(skeleton);
  if (detached) { mesh.remove(root); mesh.bindMode = T.DetachedBindMode; }
  return {mesh, root, tip, skeleton, geometry, material};
}

function pose(value, phase) {
  value.root.rotation.set(.16 + phase * .1, .12, -.16 - phase * .2);
  value.tip.rotation.set(.3 - phase * .15, .2 + phase * .1, .6 + phase * .45);
}

function scene(T, {shadows = false} = {}) {
  const result = new T.Scene(); result.background = new T.Color(0x172536);
  const light = new T.DirectionalLight(0xffd5ac, 2.7); light.position.set(-2.5, 4, 3);
  light.castShadow = shadows;
  light.shadow.mapSize.set(128, 128);
  Object.assign(light.shadow.camera, {left: -3, right: 3, top: 3, bottom: -3, near: .1, far: 15});
  light.shadow.camera.updateProjectionMatrix(); light.shadow.normalBias = .015;
  result.add(light, light.target, new T.HemisphereLight(0xb6d7ff, 0x64513b, .75));
  const camera = new T.PerspectiveCamera(43, 320 / 240, .1, 20);
  camera.position.set(.2, .7, 5.5); camera.lookAt(0, -.05, 0);
  return {scene: result, camera, light};
}

function pairedScene(T, shadows = false) {
  const result = scene(T, {shadows});
  const attached = rig(T, {color: 0xe69a49}), detached = rig(T, {color: 0x52a7c9, detached: true, ordinary: true});
  attached.mesh.position.x = -.72; detached.mesh.position.set(.78, .08, -.15);
  attached.mesh.rotation.y = .25; detached.mesh.rotation.y = -.3;
  attached.mesh.scale.set(1.1, .95, .8); detached.mesh.scale.set(.85, 1.05, 1.1);
  result.scene.add(attached.mesh, detached.mesh, detached.root);
  for (const [i, value] of [attached, detached].entries()) { pose(value, i ? -.65 : .25); value.mesh.castShadow = shadows; }
  if (shadows) {
    const ground = new T.Mesh(new T.PlaneGeometry(7, 6), new T.MeshStandardNodeMaterial({color: 0x9ca8ae, roughness: 1}));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -1.02; ground.receiveShadow = true;
    result.scene.add(ground); result.camera.position.set(2.4, 2.1, 5.2); result.camera.lookAt(0, -.2, 0);
  }
  return {...result, attached, detached};
}

function animate(renderer, result, update) {
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    if (args[0] === result.scene && args[1] === result.camera) {
      frame++;
      if (frame === 8 || frame === 18) update(frame);
    }
    return render(...args);
  };
  return result;
}

export const nativeSkinningScenarios = {
  native_tsl_skinning(T) {
    return pairedScene(T);
  },
  native_tsl_skinning_updates(T, renderer) {
    const result = pairedScene(T);
    return animate(renderer, result, frame => {
      const phase = frame === 8 ? -.9 : .85;
      pose(result.attached, phase); pose(result.detached, -phase);
      // Attached and detached bindings respond differently to public mesh transforms.
      result.attached.mesh.position.y = frame === 8 ? .2 : -.08;
      result.detached.mesh.position.set(frame === 8 ? .55 : 1, .15, -.2);
      result.detached.mesh.rotation.y = frame === 8 ? .4 : -.6;
    });
  },
  native_tsl_skinning_shadows(T, renderer) {
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
    const result = pairedScene(T, true);
    return animate(renderer, result, frame => {
      pose(result.attached, frame === 8 ? -.8 : .65);
      pose(result.detached, frame === 8 ? .7 : -.5);
      result.light.position.x = frame === 8 ? 1.5 : -1.5;
    });
  },
  native_tsl_skinning_large_palette(T, renderer) {
    const result = scene(T);
    const paletteSize = Math.floor(renderer.backend.capabilities.getUniformBufferLimit() / 64) + 1;
    const large = rig(T, {paletteSize, color: 0xa479da});
    large.mesh.scale.setScalar(1.3); result.scene.add(large.mesh); pose(large, .2);
    return animate(renderer, {...result, large, paletteSize}, frame => pose(large, frame === 8 ? -.8 : .9));
  },
  native_tsl_skinning_targets(T, renderer) {
    const writer = scene(T), value = rig(T, {color: 0x5bc6a7}); writer.scene.add(value.mesh);
    writer.camera.position.set(0, .1, 3.2); writer.camera.lookAt(0, 0, 0);
    const targets = Array.from({length: 3}, () => new T.RenderTarget(96, 128,
      {generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
    const result = {scene: new T.Scene(), camera: new T.OrthographicCamera(-2.2, 2.2, 1.65, -1.65, .1, 10)};
    result.scene.background = new T.Color(0x15202d); result.camera.position.z = 4;
    for (const [i, target] of targets.entries()) {
      renderer.initRenderTarget(target);
      const panel = new T.Mesh(new T.PlaneGeometry(1.35, 1.8), new T.MeshBasicMaterial({map: target.texture}));
      panel.position.x = (i - 1) * 1.45; result.scene.add(panel);
    }
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => {
      const previous = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
      try {
        for (const target of targets) { renderer.setRenderTarget(target); await compile(writer.scene, writer.camera); }
      } finally { renderer.setRenderTarget(previous, face, mip); }
      return compile(...args);
    };
    let captured = false;
    renderer.render = (...args) => {
      if (args[0] === result.scene && !captured) {
        captured = true;
        const previous = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
        try {
          for (const [i, target] of targets.entries()) {
            pose(value, [-1.2, .15, 1.15][i]); writer.scene.updateMatrixWorld(true);
            // Explicit public updates give each same-frame draw a distinct pose.
            value.skeleton.update(); value.material.roughness = [.15, .5, .9][i];
            renderer.setRenderTarget(target); render(writer.scene, writer.camera);
          }
        } finally { renderer.setRenderTarget(previous, face, mip); }
        // Queued native draws must retain all three accepted palette snapshots.
        value.skeleton.boneMatrices.fill(0); pose(value, 4);
      }
      return render(...args);
    };
    return {...result, writer, value, targets};
  },
};
