import { SunLight } from '../../../upstream/three.js/examples/jsm/lights/SunLight.js';
import { SunLightNode } from '../../../upstream/three.js/examples/jsm/lights/SunLightNode.js';

// Identical authored graphs and shadow updates run on the source and native paths.
function shadowScene(T, renderer, shadowType, kind = 'directional') {
  const N = T.TSL, scene = new T.Scene();
  scene.background = new T.Color(0xbac9d8);
  const camera = new T.PerspectiveCamera(45, 320 / 240, .1, 40);
  camera.position.set(5.8, 4.7, 6.8); camera.lookAt(0, .6, 0);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = shadowType;

  const light = kind === 'sun' ? new SunLight(0xffe6c8, 3)
    : kind === 'spot' ? new T.SpotLight(0xffe6c8, 100, 30, Math.PI / 3, .25, 2)
      : new T.DirectionalLight(0xffe6c8, 3);
  if (kind === 'sun') renderer.library.addLight(SunLightNode, SunLight);
  light.position.set(-3, 6, 4); light.castShadow = true;
  light.shadow.mapSize.set(128, 128);
  Object.assign(light.shadow.camera, {left: -5, right: 5, top: 5, bottom: -5, near: .2, far: 20});
  light.shadow.camera.updateProjectionMatrix();
  light.shadow.bias = -.001; light.shadow.normalBias = .02; light.shadow.radius = 1.5;
  scene.add(light, new T.HemisphereLight(0xcbe1ff, 0x564733, .6));
  if (light.target) scene.add(light.target);

  const ground = new T.Mesh(new T.PlaneGeometry(10, 8),
    kind === 'sun' ? new T.ShadowMaterial({color: 0x5d6568, opacity: .65})
      : new T.MeshStandardNodeMaterial({color: 0xb4b8ae, roughness: 1}));
  ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);

  const phase = N.uniform(.2);
  const casterMaterial = new T.MeshStandardNodeMaterial({color: 0xd98036, roughness: .72});
  casterMaterial.positionNode = N.positionLocal.add(
    N.vec3(N.positionLocal.y.mul(3).add(phase).sin().mul(.28), 0, 0));
  const caster = new T.Mesh(new T.BoxGeometry(1, 1.8, 1, 1, 12, 1), casterMaterial);
  caster.position.set(-1.1, .9, .1); caster.castShadow = true;
  caster.frustumCulled = false; scene.add(caster);
  if (kind === 'sun') {
    camera.position.set(4.5, 3.6, 5.5); camera.lookAt(0, .6, 0);
    // Put a caster nearer the viewer as well as beyond the cascade split.
    const nearCaster = new T.Mesh(caster.geometry, casterMaterial);
    nearCaster.position.set(1.8, .45, 2.2); nearCaster.scale.setScalar(.5);
    nearCaster.castShadow = true; nearCaster.frustumCulled = false; scene.add(nearCaster);
  }

  // Binary holes make accidental opaque shadow casting visible on the receiver.
  const size = 16, pixels = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const filled = ((x >> 2) + (y >> 2)) % 2 === 0;
    pixels.set([55, 151, 179, filled ? 255 : 0], (y * size + x) * 4);
  }
  const map = new T.DataTexture(pixels, size, size);
  map.minFilter = map.magFilter = T.NearestFilter;
  map.colorSpace = T.SRGBColorSpace; map.generateMipmaps = false; map.needsUpdate = true;
  const cardMaterial = new T.MeshBasicNodeMaterial({map, alphaTest: .5, side: T.DoubleSide});
  cardMaterial.forceSinglePass = true;
  const card = new T.Mesh(new T.PlaneGeometry(1.6, 1.8), cardMaterial);
  card.position.set(1.1, 1, -.6); card.rotation.y = -.35; card.castShadow = true; scene.add(card);

  // This visible mesh intentionally does not participate in the shadow pass.
  const noncaster = new T.Mesh(new T.SphereGeometry(.48, 24, 16),
    new T.MeshStandardNodeMaterial({color: 0x6c8fcb, roughness: .5}));
  noncaster.position.set(1.5, .48, 1.25); scene.add(noncaster);

  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    // ShadowNode recursively calls render with the same scene and another camera.
    if (args[0] === scene && args[1] === camera) {
      frame++;
      if (frame === 8) {
        phase.value = 1.1; caster.position.x = -.7; light.position.x = -2;
        light.shadow.autoUpdate = false; light.shadow.needsUpdate = true;
      }
      if (frame === 12) {
        // The visible deformation moves while the last requested shadow stays frozen.
        phase.value = 2.4; caster.position.x = -1.3;
      }
    }
    return render(...args);
  };
  return {scene, camera, light, ground, caster, card, noncaster, phase, map};
}

export const nativeShadowScenarios = {
  native_tsl_shadows(T, renderer) {
    return shadowScene(T, renderer, T.PCFShadowMap);
  },
  native_tsl_shadows_basic(T, renderer) {
    return shadowScene(T, renderer, T.BasicShadowMap);
  },
  native_tsl_shadows_sun(T, renderer) {
    return shadowScene(T, renderer, T.PCFShadowMap, 'sun');
  },
  native_tsl_shadows_spot(T, renderer) {
    return shadowScene(T, renderer, T.PCFShadowMap, 'spot');
  },
};
