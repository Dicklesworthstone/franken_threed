// Native TSL graphs are authored once and executed unchanged by both renderers.
function camera(T, orthographic = false) {
  const result = orthographic
    ? new T.OrthographicCamera(-2.2, 2.2, 1.65, -1.65, .1, 30)
    : new T.PerspectiveCamera(48, 320 / 240, .1, 30);
  result.position.set(.4, .25, 4.4); result.lookAt(0, 0, 0); return result;
}
function scene(T) {
  const result = new T.Scene(); result.background = new T.Color(0x162335); return result;
}
export const nativeTSLScenarios = {
  native_tsl_uniforms(T, renderer) {
    const {uniform, vec3, positionLocal} = T.TSL;
    const phase = uniform(.2), tint = uniform(new T.Color(0xd97c42));
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = tint;
    material.positionNode = positionLocal.add(vec3(0, positionLocal.x.mul(5).add(phase).sin().mul(.16), 0));
    const result = {scene: scene(T), camera: camera(T)};
    const geometry = new T.PlaneGeometry(1.2, 1.35, 16, 12);
    for (let i = 0; i < 2; i++) {
      const mesh = new T.Mesh(geometry, material);
      mesh.position.set(i ? .8 : -.8, i ? -.15 : .15, i ? -.15 : 0);
      mesh.rotation.set(i ? .2 : -.12, i ? -.3 : .18, i ? -.15 : .1);
      mesh.scale.set(i ? .9 : 1, i ? 1.1 : .9, 1); result.scene.add(mesh);
    }
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) { phase.value = 1.4; tint.value.setRGB(.18, .65, .9); }
      return render(...args);
    };
    return result;
  },
  native_tsl_fragment(T) {
    const {attribute, float, uv, vec4} = T.TSL;
    const geometry = new T.PlaneGeometry(1.55, 1.55, 6, 6);
    const heat = new Float32Array(geometry.attributes.position.count);
    for (let i = 0; i < heat.length; i++) heat[i] = (i % 7) / 6;
    geometry.setAttribute('heat', new T.BufferAttribute(heat, 1));
    const material = new T.MeshBasicNodeMaterial({transparent: true, depthWrite: false});
    material.fragmentNode = vec4(uv().x.mul(.7).add(.1), uv().y.mul(.6).add(.12),
      attribute('heat', 'float').mul(.7).add(.1), float(.6));
    const result = {scene: scene(T), camera: camera(T)};
    for (let i = 0; i < 2; i++) {
      const mesh = new T.Mesh(geometry, material);
      mesh.position.set(i ? .5 : -.5, i ? .2 : -.15, i ? -.2 : .15);
      mesh.rotation.set(i ? .12 : -.16, i ? -.2 : .15, i ? .25 : -.2);
      mesh.scale.set(i ? .85 : 1.05, i ? 1.1 : .85, 1); result.scene.add(mesh);
    }
    return result;
  },
  native_tsl_object_targets(T, renderer) {
    const {uniform, vec3, positionLocal} = T.TSL;
    const gain = uniform(.35);
    const tint = uniform(new T.Color()).onObjectUpdate(frame => frame.object.userData.tint);
    const shift = uniform(0).onObjectUpdate(frame => frame.object.userData.shift);
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = tint.mul(gain);
    material.positionNode = positionLocal.add(vec3(0, shift, 0));
    const writer = scene(T), meshes = [];
    for (let i = 0; i < 2; i++) {
      const mesh = new T.Mesh(new T.PlaneGeometry(.9, 1.05, 2, 2), material);
      mesh.position.x = i ? .58 : -.58; mesh.rotation.z = i ? -.1 : .1;
      mesh.userData.tint = new T.Color(i ? 0x65b6ed : 0xeaa259);
      mesh.userData.shift = i ? -.17 : .17; writer.add(mesh); meshes.push(mesh);
    }
    const writeCamera = new T.OrthographicCamera(-1.2, 1.2, .9, -.9, .1, 10);
    writeCamera.position.z = 3;
    const targets = Array.from({length: 2}, () => new T.RenderTarget(96, 72,
      {depthBuffer: false, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
    const result = {scene: scene(T), camera: camera(T, true)};
    for (const [i, target] of targets.entries()) {
      renderer.initRenderTarget(target);
      const panel = new T.Mesh(new T.PlaneGeometry(1.85, 1.4), new T.MeshBasicMaterial({map: target.texture}));
      panel.position.x = i ? 1 : -1; result.scene.add(panel);
    }
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => {
      const previous = renderer.getRenderTarget(), previousMip = renderer.getActiveMipmapLevel();
      try {
        for (const target of targets) { renderer.setRenderTarget(target); await compile(writer, writeCamera); }
      } finally { renderer.setRenderTarget(previous, 0, previousMip); }
      return compile(...args);
    };
    let frame = 0;
    renderer.render = (...args) => {
      frame++;
      if (frame === 1 || frame === 8) {
        if (frame === 8) for (const [i, mesh] of meshes.entries()) {
          mesh.userData.tint.setRGB(i ? .7 : .15, i ? .25 : .8, i ? .6 : .35);
          mesh.userData.shift *= -1;
        }
        for (const [i, target] of targets.entries()) {
          gain.value = frame === 1 ? (i ? .85 : .35) : (i ? .25 : .7);
          renderer.setRenderTarget(target); render(writer, writeCamera);
        }
        renderer.setRenderTarget(null);
        // Queued draws must retain their own values after this later mutation.
        gain.value = .95;
      }
      return render(...args);
    };
    return result;
  },
};
