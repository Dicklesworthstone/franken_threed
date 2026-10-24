// Ordinary source morph attributes and influences, including morph-before-skin.
function geometry(T, relative = false) {
  const base = new T.BoxGeometry(.9, 1.3, .65, 8, 12, 6), targets = [], v = new T.Vector3();
  for (let targetIndex = 0; targetIndex < 2; targetIndex++) {
    const target = base.clone(), positions = target.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      v.fromBufferAttribute(positions, i);
      if (targetIndex === 0) {
        const angle = v.y * 1.6, x = v.x, z = v.z;
        v.set(x * Math.cos(angle) - z * Math.sin(angle), v.y * 1.1,
          x * Math.sin(angle) + z * Math.cos(angle));
      } else {
        v.x = v.x * (.6 + (v.y + .65) * .3) + Math.sin(v.y * 2.8) * .3;
        v.z *= 1.35; v.y *= .85;
      }
      positions.setXYZ(i, v.x, v.y, v.z);
    }
    target.computeVertexNormals(); targets.push(target);
  }
  base.morphTargetsRelative = relative;
  for (const name of ['position', 'normal']) {
    base.morphAttributes[name] = targets.map(target => {
      const attribute = target.attributes[name].clone();
      if (relative) for (let i = 0; i < attribute.array.length; i++) attribute.array[i] -= base.attributes[name].array[i];
      return attribute;
    });
  }
  return base;
}

function material(T, color, ordinary = false) {
  if (ordinary) return new T.MeshStandardMaterial({color, roughness: .35});
  const result = new T.MeshStandardNodeMaterial({roughness: .35});
  result.colorNode = T.TSL.uniform(new T.Color(color)); return result;
}

function stage(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x1c2a3a);
  const light = new T.DirectionalLight(0xffd2a7, 2.8); light.position.set(-2, 3, 4);
  scene.add(light, new T.HemisphereLight(0xc3dcff, 0x5a4330, .7));
  const camera = new T.PerspectiveCamera(43, 4 / 3, .1, 20);
  camera.position.set(.25, .5, 4.8); camera.lookAt(0, 0, 0);
  return {scene, camera};
}

function paired(T, skinned = false) {
  const result = stage(T), meshes = [], rigs = [];
  for (let i = 0; i < 2; i++) {
    const g = geometry(T, !!i), m = material(T, i ? 0x57aec8 : 0xe89952, !!i);
    let mesh;
    if (skinned) {
      const indices = [], weights = [];
      for (let j = 0; j < g.attributes.position.count; j++) {
        const weight = T.MathUtils.clamp((g.attributes.position.getY(j) + .65) / 1.3, 0, 1);
        indices.push(0, 1, 0, 0); weights.push(1 - weight, weight, 0, 0);
      }
      g.setAttribute('skinIndex', new T.Uint16BufferAttribute(indices, 4));
      g.setAttribute('skinWeight', new T.Float32BufferAttribute(weights, 4));
      mesh = new T.SkinnedMesh(g, m);
      const root = new T.Bone(), tip = new T.Bone(); root.position.y = -.65; tip.position.y = 1.3;
      root.add(tip); mesh.add(root); mesh.bind(new T.Skeleton([root, tip]));
      root.rotation.x = .2; tip.rotation.set(.25, .15, i ? -.7 : .75); rigs.push({root, tip});
    } else mesh = new T.Mesh(g, m);
    mesh.morphTargetInfluences[0] = i ? -.2 : .6; mesh.morphTargetInfluences[1] = i ? .85 : .35;
    mesh.position.x = i ? .8 : -.8; mesh.rotation.set(.12, i ? -.4 : .4, i ? -.08 : .08);
    mesh.scale.set(1.05, .9, 1.1); result.scene.add(mesh); meshes.push(mesh);
  }
  return {...result, meshes, rigs};
}

function instanced(T, count = 6) {
  const result = stage(T), g = geometry(T, true), m = material(T, 0xffffff);
  const mesh = new T.InstancedMesh(g, m, count), template = new T.Mesh(g, m);
  const transform = new T.Object3D(), columns = count === 2 ? 1 : 3, rows = Math.ceil(count / columns);
  const weights = [[0, 0], [.85, .1], [-.3, .9], [.25, 1.1], [1.15, -.2], [.55, .65]];
  for (let i = 0; i < count; i++) {
    transform.position.set((i % columns - (columns - 1) / 2) * 1.25, (rows - 1) * .8 - Math.floor(i / columns) * 1.6, 0);
    transform.rotation.set(.1, (i % 3 - 1) * .35, (i % 2 ? -1 : 1) * .08);
    transform.scale.set(.86 + (i % 2) * .12, .84, .8 + (i % 3) * .1); transform.updateMatrix();
    mesh.setMatrixAt(i, transform.matrix);
    mesh.setColorAt(i, new T.Color().setHSL(.06 + i * .12, .62, .56));
    template.morphTargetInfluences = weights[i]; mesh.setMorphAt(i, template);
  }
  mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true; mesh.morphTexture.needsUpdate = true;
  result.scene.add(mesh); result.camera.position.set(.2, .25, count === 2 ? 4.9 : 6.5); result.camera.lookAt(0, 0, 0);
  const writeWeights = (values, upload = true) => {
    for (let i = 0; i < count; i++) { template.morphTargetInfluences = values[i % values.length]; mesh.setMorphAt(i, template); }
    if (upload) mesh.morphTexture.needsUpdate = true;
  };
  return {...result, mesh, template, writeWeights};
}

export const nativeMorphScenarios = {
  native_tsl_morph_instances(T, renderer) {
    const result = instanced(T), render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene) {
        if (++frame === 8) result.writeWeights([[.2, .95], [1.1, -.2], [-.35, .4], [.7, .15], [.1, 1.2], [.8, .6]]);
        // Public CPU writes without needsUpdate must leave the uploaded weights unchanged.
        if (frame === 18) result.writeWeights([[3, -2]], false);
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_morph_instances_targets(T, renderer) {
    const writer = instanced(T, 2); writer.camera.aspect = 3 / 4; writer.camera.updateProjectionMatrix();
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
      try { for (const target of targets) { renderer.setRenderTarget(target); await compile(writer.scene, writer.camera); } }
      finally { renderer.setRenderTarget(previous, face, mip); }
      return compile(...args);
    };
    let captured = false;
    renderer.render = (...args) => {
      if (args[0] === result.scene && !captured) {
        captured = true;
        const previous = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
        const poses = [[[.05, .1], [.9, -.2]], [[.85, .1], [-.3, 1]], [[.15, 1.15], [1.1, .3]]];
        try {
          for (const [i, target] of targets.entries()) {
            writer.writeWeights(poses[i]); renderer.setRenderTarget(target); render(writer.scene, writer.camera);
          }
        } finally { renderer.setRenderTarget(previous, face, mip); }
        writer.writeWeights([[4, -3]], false);
      }
      return render(...args);
    };
    return {...result, writer, targets};
  },
  native_tsl_morph(T, renderer) {
    const result = paired(T), render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && ( ++frame === 8 || frame === 18 )) {
        result.meshes[0].morphTargetInfluences = frame === 8 ? [.2, .9] : [1.1, .4];
        result.meshes[1].morphTargetInfluences[0] = frame === 8 ? .8 : -.35;
        result.meshes[1].morphTargetInfluences[1] = frame === 8 ? .1 : 1.15;
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_morph_skinning(T, renderer) {
    const result = paired(T, true), render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && ++frame === 12) {
        for (const [i, mesh] of result.meshes.entries()) {
          mesh.morphTargetInfluences[0] = i ? .95 : .25; mesh.morphTargetInfluences[1] = i ? -.3 : .8;
          result.rigs[i].tip.rotation.z = i ? .65 : -.55; result.rigs[i].root.rotation.y = i ? -.3 : .2;
        }
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_morph_targets(T, renderer) {
    const writer = stage(T), mesh = new T.Mesh(geometry(T), material(T, 0x9b7bd9));
    mesh.rotation.set(.1, .35, 0); writer.scene.add(mesh);
    writer.camera.position.set(.2, .3, 3.2); writer.camera.lookAt(0, 0, 0);
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
      try { for (const target of targets) { renderer.setRenderTarget(target); await compile(writer.scene, writer.camera); } }
      finally { renderer.setRenderTarget(previous, face, mip); }
      return compile(...args);
    };
    let captured = false;
    renderer.render = (...args) => {
      if (args[0] === result.scene && !captured) {
        captured = true;
        const previous = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
        try {
          for (const [i, target] of targets.entries()) {
            mesh.morphTargetInfluences = [[.05, .1], [.8, .1], [.15, 1.1]][i];
            renderer.setRenderTarget(target); render(writer.scene, writer.camera);
          }
        } finally { renderer.setRenderTarget(previous, face, mip); }
        mesh.morphTargetInfluences.fill(4);
      }
      return render(...args);
    };
    return {...result, writer, mesh, targets};
  },
};
