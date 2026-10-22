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
function sampledTexture(T, variant = 0, colorSpace = T.SRGBColorSpace, alpha = false) {
  const size = 16, pixels = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const color = variant === 0 ? [30 + x * 12, 35 + y * 11, 210 - x * 8]
      : variant === 1 ? [210 - y * 10, 40 + x * 10, 40 + y * 10]
        : [40 + y * 10, 220 - x * 11, 50 + x * 9];
    pixels.set([...color, alpha ? 100 + ((x + y) % 4) * 45 : 255], (y * size + x) * 4);
  }
  const result = new T.DataTexture(pixels, size, size);
  result.minFilter = result.magFilter = T.LinearFilter;
  result.wrapS = result.wrapT = T.RepeatWrapping;
  result.colorSpace = colorSpace; result.generateMipmaps = false; result.needsUpdate = true; return result;
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
  native_tsl_textures(T) {
    const {texture, uv, vec2, vec3, positionLocal} = T.TSL;
    const geometry = new T.PlaneGeometry(1.2, 1.1, 12, 10);
    const secondary = geometry.attributes.uv.clone();
    for (let i = 0; i < secondary.count; i++) secondary.setXY(i, secondary.getY(i), .15 + secondary.getX(i) * .7);
    geometry.setAttribute('uv1', secondary);
    const transformed = sampledTexture(T); transformed.channel = 1;
    transformed.repeat.set(1.6, .8); transformed.offset.set(.1, .05);
    transformed.center.set(.5, .5); transformed.rotation = .25;
    const generated = sampledTexture(T, 1); generated.channel = 3;
    generated.repeat.set(3, .4); generated.offset.set(.25, .3); generated.rotation = .9;
    const nearest = sampledTexture(T, 2, T.NoColorSpace);
    nearest.minFilter = nearest.magFilter = T.NearestFilter;
    const srgb = sampledTexture(T, 1), alpha = sampledTexture(T, 0, T.SRGBColorSpace, true);
    const materials = Array.from({length: 4}, () => new T.MeshBasicNodeMaterial());
    materials[0].colorNode = texture(transformed).rgb;
    // Explicit generated UVs bypass the channel and default texture matrix.
    materials[1].colorNode = texture(generated, positionLocal.xy.mul(.7).add(.5)).rgb;
    materials[2].colorNode = texture(nearest, uv()).rgb.mul(.45).add(texture(srgb, uv().yx).rgb.mul(.55));
    materials[2].positionNode = positionLocal.add(vec3(0, texture(nearest, uv()).r.sub(.5).mul(.15), 0));
    materials[3].transparent = true; materials[3].depthWrite = false;
    materials[3].fragmentNode = texture(alpha, uv().mul(vec2(1.2, .7)).add(vec2(.05, .1))).level(0);
    const result = {scene: scene(T), camera: camera(T)};
    for (const [i, material] of materials.entries()) {
      const mesh = new T.Mesh(geometry, material);
      mesh.position.set(i % 2 ? .72 : -.72, i < 2 ? .65 : -.65, 0);
      mesh.rotation.z = (i - 1.5) * .04; result.scene.add(mesh);
    }
    return result;
  },
  native_tsl_texture_updates(T, renderer) {
    const {texture, uniform} = T.TSL;
    const maps = [sampledTexture(T), sampledTexture(T, 1), sampledTexture(T, 2)];
    maps[0].repeat.set(1.4, .9); maps[1].offset.set(.25, .1);
    const sampled = texture(maps[0]), gain = uniform(.7);
    const material = new T.MeshBasicNodeMaterial(); material.colorNode = sampled.rgb.mul(gain);
    const writer = scene(T);
    writer.add(new T.Mesh(new T.PlaneGeometry(1.9, 1.35), material));
    const writeCamera = new T.OrthographicCamera(-1.2, 1.2, .9, -.9, .1, 10); writeCamera.position.z = 3;
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
        if (frame === 8) {
          for (const map of maps.slice(0, 2)) {
            for (let i = 0; i < map.image.data.length; i += 4) map.image.data[i] = 255 - map.image.data[i];
            map.needsUpdate = true;
          }
          maps[0].matrixAutoUpdate = false;
          maps[0].matrix.setUvTransform(.1, -.15, .8, 1.2, .35, .5, .5);
          maps[1].offset.set(-.1, .3); maps[1].rotation = -.25;
        }
        for (const [i, target] of targets.entries()) {
          sampled.value = maps[frame === 1 ? i : 1 - i];
          gain.value = frame === 1 ? (i ? .4 : .8) : (i ? .9 : .45);
          renderer.setRenderTarget(target); render(writer, writeCamera);
        }
        renderer.setRenderTarget(null);
        // Both queued texture identities and uniforms precede this later state.
        sampled.value = maps[2]; gain.value = .15;
        for (let i = 2; i < maps[0].image.data.length; i += 4) maps[0].image.data[i] = 255 - maps[0].image.data[i];
        maps[0].needsUpdate = true;
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_target_texture(T, renderer) {
    const {texture, uv, vec3} = T.TSL;
    const target = new T.RenderTarget(96, 72, {generateMipmaps: false, minFilter: T.LinearFilter, magFilter: T.LinearFilter});
    renderer.initRenderTarget(target);
    const writer = scene(T); writer.background = new T.Color(0x29445a);
    const front = new T.Mesh(new T.PlaneGeometry(.9, .45), new T.MeshBasicMaterial({color: 0xe9c848}));
    front.position.set(-.3, .3, .2);
    const back = new T.Mesh(new T.PlaneGeometry(1.3, .7), new T.MeshBasicMaterial({color: 0x4489d6}));
    back.position.set(.15, -.2, 0); writer.add(front, back);
    const writeCamera = new T.OrthographicCamera(-1.2, 1.2, .9, -.9, .1, 10); writeCamera.position.z = 3;
    const result = {scene: scene(T), camera: camera(T, true)};
    for (let i = 0; i < 2; i++) {
      const material = new T.MeshBasicNodeMaterial();
      material.colorNode = i ? texture(target.texture, uv().yx).level(0).rgb.mul(vec3(.75, 1, .85)) : texture(target.texture).rgb;
      const panel = new T.Mesh(new T.PlaneGeometry(1.85, 1.4), material);
      panel.position.x = i ? 1 : -1; result.scene.add(panel);
    }
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => {
      const previous = renderer.getRenderTarget(), previousMip = renderer.getActiveMipmapLevel();
      try { renderer.setRenderTarget(target); await compile(writer, writeCamera); }
      finally { renderer.setRenderTarget(previous, 0, previousMip); }
      return compile(...args);
    };
    let frame = 0;
    renderer.render = (...args) => {
      frame++;
      if (frame === 1 || frame === 8) {
        if (frame === 8) {
          front.material.color.set(0xd64a92); front.position.y = -.35;
          back.material.color.set(0x5bce83); back.position.y = .25;
          target.texture.offset.set(.08, .12); target.texture.repeat.set(.85, .8);
        }
        renderer.setRenderTarget(target); render(writer, writeCamera); renderer.setRenderTarget(null);
      }
      return render(...args);
    };
    return result;
  },
};
