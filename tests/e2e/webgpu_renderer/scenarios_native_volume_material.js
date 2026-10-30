// The same deterministic 3D density field and source raymarching graph execute
// in both renderers. No elapsed time, random noise or captured candidate pixels.
function volume(T, emissionOnly = false) {
  const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0x172334);
  const camera = new T.PerspectiveCamera(48, 4 / 3, .1, 20);
  camera.position.set(.2, .15, 5); camera.lookAt(0, 0, 0);
  const size = 8, pixels = new Uint8Array(size ** 3);
  for (let z = 0; z < size; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
    pixels[(z * size + y) * size + x] = 16 + x * 16 + y * 8 + z * 4;
  const texture = new T.Data3DTexture(pixels, size, size, size);
  texture.format = T.RedFormat; texture.type = T.UnsignedByteType;
  texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.unpackAlignment = 1; texture.needsUpdate = true;
  const density = N.uniform(3), offset = N.uniform(.2), gain = N.uniform(emissionOnly ? 35 : .5);
  const tint = N.uniform(new T.Color(.9, .3, .1));
  const field = positionRay => positionRay.length().lessThan(1.5)
    .select(N.texture3D(texture, positionRay.div(3).add(.5), 0).r, 0);
  const material = new T.VolumeNodeMaterial({steps: 12, blending: T.AdditiveBlending});
  material.offsetNode = offset;
  material.scatteringNode = N.Fn(({positionRay}) => field(positionRay).mul(density));
  material.scatteringEmissiveNode = N.Fn(({positionRay}) => field(positionRay).mul(gain).mul(tint));
  const mesh = new T.Mesh(new T.BoxGeometry(2.8, 2.8, 2.8), material);
  mesh.frustumCulled = false; scene.add(mesh);
  // The pinned material only enters its lighting model when a light scope is
  // present. A zero-intensity point light admits pure emission without adding
  // scattering; an empty light scope would skip the source raymarch entirely.
  const point = new T.PointLight(0xffa64d, emissionOnly ? 0 : 35, 8, 2);
  point.position.set(-.8, .5, 1.1); scene.add(point);
  let spot = null;
  if (!emissionOnly) {
    spot = new T.SpotLight(0x51aaff, 70, 10, .75, .35, 2);
    spot.position.set(1.6, 1.3, 2.5); spot.target.position.set(-.2, -.2, 0);
    scene.add(spot, spot.target);
  }
  return {scene, camera, material, mesh, point, spot, texture, density, offset, gain, tint};
}

function update(result, frame) {
  const {material, camera, point, spot, texture, density, offset, gain, tint} = result;
  if (frame === 8) {
    material.steps = 21; offset.value = .65; density.value = 5;
    tint.value.setRGB(.15, .65, .9); gain.value *= .75;
    if (spot) { point.position.set(.7, -.3, .9); spot.intensity = 110; }
  }
  if (frame === 10) {
    for (let i = 0; i < texture.image.data.length; i++) texture.image.data[i] = 255 - texture.image.data[i];
    texture.needsUpdate = true;
  }
  if (frame === 12) { material.steps = 7; offset.value = 0; }
  if (frame === 13) {
    material.steps = 16; offset.value = .4; density.value = 2;
    camera.position.set(-.6, .3, 4.3); camera.lookAt(0, 0, 0);
    if (spot) { spot.position.set(-1.5, .8, 2.2); point.color.setHex(0x70d8ff); }
  }
}

function drive(renderer, result, before = () => {}) {
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  const target = renderer.getRenderTarget(); let frame = 0, firstPixels = null;
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] === target) {
      const background = result.scene.background.toArray().map(value => Math.round(value * 255));
      const colors = new Set(); let changed = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
        if (background.some((value, channel) => Math.abs(pixels[i + channel] - value) > 2)) changed++;
      }
      if (changed < 256 || colors.size < 16)
        throw Error('Volume output must contain visible, nonconstant raymarched color');
      if (frame === 1) firstPixels = pixels.slice();
      if (frame === 8 && firstPixels && pixels.every((value, i) => value === firstPixels[i]))
        throw Error('Live volume uniforms did not change the rendered output');
    }
    return pixels;
  };
  renderer.render = (...args) => {
    if (args[0] !== result.scene) return render(...args);
    // This offscreen runner can finish several renders within one browser RAF.
    // Advance the same pinned NodeFrame before each logical animation frame so
    // FRAME light updates do not depend on reference/candidate execution speed.
    renderer._nodes.nodeFrame.update();
    update(result, ++frame); before(frame, render); return render(...args);
  };
  return result;
}

export const volumeMaterialScenarios = {
  native_volume_emission(T, renderer) { return drive(renderer, volume(T, true)); },
  native_volume_scattering(T, renderer) { return drive(renderer, volume(T)); },
  native_volume_depth(T, renderer) {
    const result = volume(T), N = T.TSL;
    result.material.scatteringEmissiveNode = null;
    const depthScene = new T.Scene(), depthTarget = new T.RenderTarget(128, 96);
    depthTarget.depthTexture = new T.DepthTexture(128, 96, T.FloatType);
    const opaqueMaterial = new T.MeshBasicNodeMaterial(); opaqueMaterial.colorNode = N.vec3(.08, .18, .35);
    const blocker = new T.Mesh(new T.BoxGeometry(.9, 1.55, .4), opaqueMaterial);
    blocker.position.set(-.35, .05, .7); depthScene.add(blocker);
    const visibleBlocker = blocker.clone(); result.scene.add(visibleBlocker);
    result.material.depthNode = N.texture(depthTarget.depthTexture, N.screenUV).r;
    const compile = renderer.compileAsync.bind(renderer);
    renderer.compileAsync = async (...args) => {
      if (args[0] !== result.scene) return compile(...args);
      const target = renderer.getRenderTarget();
      try {
        renderer.setRenderTarget(depthTarget); await compile(depthScene, result.camera);
      } finally { renderer.setRenderTarget(target); }
      return compile(...args);
    };
    return drive(renderer, result, (frame, render) => {
      if (frame === 8) blocker.position.set(.55, -.1, 1.3);
      if (frame === 12) blocker.visible = false;
      if (frame === 13) { blocker.visible = true; blocker.position.set(0, .1, -.2); }
      visibleBlocker.position.copy(blocker.position); visibleBlocker.visible = blocker.visible;
      const target = renderer.getRenderTarget();
      try { renderer.setRenderTarget(depthTarget); render(depthScene, result.camera); }
      finally { renderer.setRenderTarget(target); }
    });
  },
};
