import {CSMShadowNode} from '../../../upstream/three.js/examples/jsm/csm/CSMShadowNode.js';

// The unmodified source addon owns every split, camera fit and fade expression.
// Both renderer modes receive the same live graph and public control changes.
function csmScene(T, renderer, {fade, type, cascades, mode, ordinary = false, orthographic = false}) {
  const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0xa4b7c9);
  const camera = orthographic ? new T.OrthographicCamera(-7, 7, 5.25, -5.25, .1, 24)
    : new T.PerspectiveCamera(50, 4 / 3, .1, 24);
  camera.position.set(4.8, 5.2, 8.8); camera.lookAt(0, .5, -1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = type;
  const light = new T.DirectionalLight(0xffe4be, 3);
  light.position.set(-4, 7, 3); light.castShadow = true;
  light.shadow.mapSize.set(64, 64); light.shadow.camera.near = .1; light.shadow.camera.far = 40;
  light.shadow.bias = -.0008; light.shadow.normalBias = .02; light.shadow.radius = 1.5;
  light.shadow.blurSamples = 4;
  scene.add(light, light.target, new T.HemisphereLight(0xb8d8f5, 0x4c4032, .5));
  const csm = new CSMShadowNode(light, {cascades, mode, maxFar: 22, lightMargin: 8,
    customSplitsCallback(count, _near, _far, breaks) {
      for (let i = 1; i <= count; i++) breaks.push(i === count ? 1 : .58 * i);
    }});
  csm.fade = fade; light.shadow.shadowNode = csm;
  const Standard = ordinary ? T.MeshStandardMaterial : T.MeshStandardNodeMaterial;
  const ground = new T.Mesh(new T.PlaneGeometry(16, 24), new Standard({color: 0xb4afa3, roughness: 1}));
  ground.rotation.x = -Math.PI / 2; ground.position.z = -2;
  ground.receiveShadow = true; scene.add(ground);
  const phase = N.uniform(.15), material = new Standard({color: 0xd47c3e, roughness: .8});
  if (!ordinary) material.positionNode = N.positionLocal.add(N.vec3(N.positionLocal.y.mul(3).add(phase).sin().mul(.16), 0, 0));
  const casters = [[-1.1, .9, .3], [.9, .9, -4], [1.2, .9, 4]].map(position => {
    const object = new T.Mesh(new T.BoxGeometry(1.1, 1.8, 1.1, 1, 8, 1), material);
    object.position.set(...position); object.castShadow = true; object.frustumCulled = false;
    scene.add(object); return object;
  });
  const pixels = new Uint8Array(8 * 8 * 4);
  for (let i = 0; i < 64; i++) pixels.set([25, 115, 153, i % 4 < 2 ? 255 : 0], i * 4);
  const alphaMap = new T.DataTexture(pixels, 8, 8); alphaMap.needsUpdate = true;
  alphaMap.minFilter = alphaMap.magFilter = T.NearestFilter;
  const Basic = ordinary ? T.MeshBasicMaterial : T.MeshBasicNodeMaterial;
  const cutout = new Basic({map: alphaMap, side: T.DoubleSide, alphaTest: .5});
  cutout.forceSinglePass = true;
  const card = new T.Mesh(new T.PlaneGeometry(1.7, 2), cutout);
  card.position.set(-2, 1.1, -2); card.castShadow = true; scene.add(card);

  const target = renderer.getRenderTarget(), render = renderer.render.bind(renderer);
  const read = renderer.readRenderTargetPixelsAsync.bind(renderer), dispose = renderer.dispose.bind(renderer);
  let frame = 0, noShadow = null;
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] === target) {
      const colors = new Set(), background = scene.background.toArray().map(value => Math.round(value * 255));
      let geometryPixels = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
        if (background.some((value, channel) => Math.abs(pixels[i + channel] - value) > 2)) geometryPixels++;
      }
      if (colors.size < 3 || geometryPixels < 256) throw Error('CSM output must contain visible shaded geometry');
      if (frame === 12) noShadow = pixels.slice();
      if (frame === 13 && noShadow) {
        const changed = pixels.reduce((count, value, i) => count + (Math.abs(value - noShadow[i]) > 2), 0);
        if (changed < 32) throw Error('Restoring cascade shadow intensity must visibly darken the receiver');
      }
    }
    return pixels;
  };
  renderer.render = (...args) => {
    if (args[0] === scene && args[1] === camera) {
      renderer._nodes.nodeFrame.update(); frame++;
      if (frame === 8) {
        phase.value = 1.4; casters[0].position.x = -.6; camera.position.x += .4;
        camera.lookAt(0, .5, -1); camera.far = 20;
        csm.mode = fade ? 'practical' : 'uniform'; csm.updateFrustums();
      }
      if (frame === 10) {
        csm.mode = fade ? 'custom' : 'logarithmic'; csm.updateFrustums();
      }
      if (frame === 12 || frame === 13)
        for (const cascade of csm.lights) cascade.shadow.intensity = frame === 12 ? 0 : 1;
      if (frame === 16) for (const cascade of csm.lights) cascade.shadow.autoUpdate = false;
      if (frame === 18) { phase.value = 2.5; casters[1].position.x -= .5; }
      if (frame === 22) for (const cascade of csm.lights) cascade.shadow.needsUpdate = true;
    }
    return render(...args);
  };
  renderer.dispose = () => { csm.dispose(); dispose(); };
  return {scene, camera};
}

export const csmShadowScenarios = {
  native_csm_pcf(T, renderer) {
    return csmScene(T, renderer, {fade: false, type: T.PCFShadowMap, cascades: 3, mode: 'practical', ordinary: true});
  },
  native_csm_fade_vsm(T, renderer) {
    return csmScene(T, renderer, {fade: true, type: T.VSMShadowMap, cascades: 2, mode: 'custom'});
  },
  native_csm_orthographic_basic(T, renderer) {
    return csmScene(T, renderer, {fade: false, type: T.BasicShadowMap, cascades: 3, mode: 'uniform', orthographic: true});
  },
};
