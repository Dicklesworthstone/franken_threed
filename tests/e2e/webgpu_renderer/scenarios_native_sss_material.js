function world(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x172334);
  const camera = new T.PerspectiveCamera(48, 4 / 3, .1, 30);
  camera.position.set(.2, .15, 5); camera.lookAt(0, 0, 0);
  const back = new T.DirectionalLight(0xffd9b3, 1.7); back.position.set(-2, 1.5, -3);
  const front = new T.PointLight(0xb7d8ff, 9, 12, 2); front.position.set(1.7, 1.2, 3);
  scene.add(back, front, new T.HemisphereLight(0xb7d2ef, 0x413328, .25));
  return {scene, camera, back, front};
}
function surface(T, parameters = {}) {
  const N = T.TSL, size = 8, pixels = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
    pixels.set([35 + x * 28, 40 + y * 24, 70 + ((x + y) % 4) * 45, 255], (y * size + x) * 4);
  const texture = new T.DataTexture(pixels, size, size);
  texture.minFilter = texture.magFilter = T.LinearFilter; texture.needsUpdate = true;
  const material = new T.MeshSSSNodeMaterial({color: 0x9f6755, roughness: .48, ...parameters});
  const nodes = [N.uniform(new T.Color(.7, .3, .12)), ...[.15, .22, .65, 2.5, 3.5].map(value => N.uniform(value))];
  const colorGraph = N.texture(texture).rgb.mul(nodes[0]);
  material.thicknessColorNode = colorGraph;
  ['thicknessDistortionNode', 'thicknessAmbientNode', 'thicknessAttenuationNode',
    'thicknessPowerNode', 'thicknessScaleNode'].forEach((name, i) => { material[name] = nodes[i + 1]; });
  return {material, texture, nodes, colorGraph};
}
function changeControls(nodes) {
  nodes[0].value.setRGB(.15, .55, .8);
  [.55, .4, .35, 4, 5.5].forEach((value, i) => { nodes[i + 1].value = value; });
}
function drive(renderer, result, update, verifyToggle = false) {
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  const target = renderer.getRenderTarget(); let frame = 0, first = null, disabled = null;
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] === target) {
      const background = result.scene.background.toArray().map(value => Math.round(value * 255));
      const colors = new Set(); let changed = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
        if (background.some((value, channel) => Math.abs(pixels[i + channel] - value) > 2)) changed++;
      }
      if (changed < 256 || colors.size < 24) throw Error('SSS output must be visible and nonconstant');
      if (frame === 1) first = pixels.slice();
      if (frame === 8 && first && pixels.every((value, i) => value === first[i]))
        throw Error('Live SSS controls did not change the rendered output');
      if (verifyToggle && frame === 12) disabled = pixels.slice();
      if (verifyToggle && frame === 13 && disabled && pixels.every((value, i) => value === disabled[i]))
        throw Error('Restoring thicknessColorNode did not restore visible SSS');
    }
    return pixels;
  };
  renderer.render = (...args) => {
    // Nested source shadow passes share the main frame, while the offscreen
    // loop advances a deterministic epoch for FRAME light callbacks.
    if (args[0] === result.scene && args[1] === result.camera) {
      renderer._nodes.nodeFrame.update(); update(++frame);
    }
    return render(...args);
  };
  return result;
}

export const sssMaterialScenarios = {
  native_sss_thickness(T, renderer) {
    const result = world(T), {material, texture, nodes, colorGraph} = surface(T);
    const sphere = new T.Mesh(new T.SphereGeometry(.76, 32, 24), material);
    sphere.position.x = -.93; result.scene.add(sphere);
    const knot = new T.Mesh(new T.TorusKnotGeometry(.44, .16, 64, 12), material);
    knot.position.x = .95; knot.rotation.set(.2, -.35, .1); result.scene.add(knot);
    return drive(renderer, result, frame => {
      if (frame === 8) changeControls(nodes);
      if (frame === 10) {
        for (let i = 0; i < texture.image.data.length; i += 4) texture.image.data[i] = 255 - texture.image.data[i];
        texture.needsUpdate = true;
      }
      if (frame === 12) { material.thicknessColorNode = null; material.needsUpdate = true; }
      if (frame === 13) { material.thicknessColorNode = colorGraph; material.needsUpdate = true; }
    }, true);
  },
  native_sss_transmission(T, renderer) {
    const result = world(T), N = T.TSL;
    const {material, nodes} = surface(T, {color: 0xb9d5e8, roughness: .16, transmission: .82,
      thickness: .8, ior: 1.35, attenuationDistance: 2.5, attenuationColor: new T.Color(0xf1bf91),
      clearcoat: .85, clearcoatRoughness: .12, dispersion: .15});
    const background = new T.MeshBasicNodeMaterial(), stripe = N.step(.5, N.uv().x.mul(7).fract());
    background.colorNode = N.mix(N.vec3(.08, .3, .65), N.vec3(.8, .25, .08), stripe);
    const plane = new T.Mesh(new T.PlaneGeometry(4.8, 3.6), background); plane.position.z = -1.5;
    const sphere = new T.Mesh(new T.SphereGeometry(.98, 40, 28), material);
    result.scene.add(plane, sphere);
    return drive(renderer, result, frame => {
      if (frame === 8) {
        changeControls(nodes); material.transmission = .5; material.thickness = 1.3;
        material.ior = 1.6; material.clearcoat = .35; material.clearcoatRoughness = .38;
      }
      if (frame === 10) { material.transmission = 0; material.needsUpdate = true; }
      if (frame === 12) { material.transmission = .9; material.needsUpdate = true; }
      if (frame === 13) { material.dispersion = 0; material.roughness = .28; material.needsUpdate = true; }
    });
  },
  native_sss_shadow_environment(T, renderer) {
    const result = world(T), N = T.TSL, {material, nodes} = surface(T);
    result.scene.remove(result.back); result.scene.remove(result.front);
    result.camera.position.set(3.6, 2.9, 5.2); result.camera.lookAt(0, .65, 0);
    result.scene.environmentNode = N.uniform(new T.Color(.025, .04, .075));
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
    const light = new T.SpotLight(0xffddba, 90, 18, Math.PI / 3, .3, 2);
    light.position.set(-2.5, 4, 2.5); light.target.position.set(0, .5, 0); light.castShadow = true;
    light.shadow.mapSize.set(128, 128); light.shadow.camera.near = .2; light.shadow.camera.far = 18;
    light.shadow.bias = -.001; light.shadow.normalBias = .02;
    result.scene.add(light, light.target);
    const sphere = new T.Mesh(new T.SphereGeometry(.8, 32, 24), material);
    sphere.position.y = .85; sphere.castShadow = sphere.receiveShadow = true; result.scene.add(sphere);
    const ground = new T.Mesh(new T.PlaneGeometry(7, 6), new T.MeshStandardNodeMaterial({color: 0x919eaa, roughness: 1}));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; result.scene.add(ground);
    const blocker = new T.Mesh(new T.BoxGeometry(.4, 1.2, .4), new T.MeshBasicNodeMaterial({color: 0x775e55}));
    blocker.position.set(-.75, .8, .75); blocker.castShadow = true; result.scene.add(blocker);
    return drive(renderer, result, frame => {
      if (frame === 8) { changeControls(nodes); blocker.position.x = -.25; light.intensity = 65; }
      if (frame === 10) { material.envNode = N.vec3(.12, .04, .02); material.needsUpdate = true; }
      if (frame === 12) { material.envNode = null; material.needsUpdate = true; }
      if (frame === 13) { light.position.set(1.5, 3.5, 2); blocker.position.z = -.5; }
    });
  },
};
