import { nativeMorphScenarios } from './scenarios_native_morph.js';

export const nativeFogScenarios = {
  native_tsl_fog_node(T, renderer) {
    const result = nativeFogScenarios.native_tsl_fog(T, renderer), N = T.TSL;
    const color = N.uniform(new T.Color(0xa2bbc8)), near = N.uniform(2.5), far = N.uniform(11);
    const height = N.positionWorld.y.add(1.5).mul(.35).clamp(.15, 1);
    result.scene.fogNode = N.fog(color, N.rangeFogFactor(near, far).mul(height));
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && args[1] === result.camera) {
        frame++;
        if (frame === 8) { color.value.set(0xc8a9b8); near.value = 1.5; }
        if (frame === 18) {
          result.scene.fogNode = N.fog(color, N.rangeFogFactor(near, far).mul(height.oneMinus()));
        }
        if (frame === 22) { color.value.set(0x8fbcc8); far.value = 9; }
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_fog_node_texture(T, renderer) {
    const result = nativeFogScenarios.native_tsl_fog(T, renderer), N = T.TSL;
    const pixels = new Uint8Array([210, 140, 100, 255, 80, 170, 210, 255,
      110, 180, 120, 255, 170, 110, 200, 255]);
    const texture = new T.DataTexture(pixels, 2, 2);
    texture.colorSpace = T.SRGBColorSpace; texture.needsUpdate = true;
    const factor = N.uniform(.65);
    result.scene.fogNode = N.fog(N.texture(texture, N.screenUV).rgb, factor);
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && args[1] === result.camera) {
        frame++;
        if (frame === 8) {
          pixels.set([80, 140, 210, 255, 200, 170, 90, 255,
            190, 100, 170, 255, 90, 190, 160, 255]);
          texture.needsUpdate = true; factor.value = .8;
        }
        if (frame === 18) result.scene.fogNode = null;
        if (frame === 22) result.scene.fogNode = N.fog(N.texture(texture, N.screenUV).rgb, factor);
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_fog(T, renderer) {
    const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0x263748);
    scene.fog = new T.Fog(0x98adba, 3.5, 11);
    const camera = new T.PerspectiveCamera(43, 4 / 3, .1, 30);
    camera.position.set(.3, 2.3, 7); camera.lookAt(0, .3, -1.8);
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
    const light = new T.PointLight(0xffd5b1, 65, 20); light.position.set(-2, 3.8, 2.2); light.castShadow = true;
    light.shadow.mapSize.set(128, 128); light.shadow.camera.near = .1; light.shadow.camera.far = 20;
    light.shadow.camera.updateProjectionMatrix(); light.shadow.normalBias = .02;
    scene.add(light, new T.HemisphereLight(0xb9d8ff, 0x6b4b35, .9));
    const ground = new T.Mesh(new T.PlaneGeometry(11, 13), new T.MeshStandardNodeMaterial({color: 0x8f9d9b, roughness: 1}));
    ground.rotation.x = -Math.PI / 2; ground.position.set(0, -.7, -2.5); ground.receiveShadow = true; scene.add(ground);
    const phase = N.uniform(.2), meshes = [];
    for (const [i, color] of [0xe3924d, 0x52b6b8, 0x9c77cd, 0xc75f8a].entries()) {
      const material = new T.MeshStandardNodeMaterial({roughness: .42});
      material.colorNode = N.uniform(new T.Color(color));
      material.positionNode = N.positionLocal.add(N.vec3(N.positionLocal.y.mul(3).add(phase).sin().mul(.12), 0, 0));
      material.fog = i !== 3;
      const geometry = i % 2 ? new T.SphereGeometry(.62, 32, 20) : new T.BoxGeometry(.95, 1.25, .8, 1, 10, 1);
      const mesh = new T.Mesh(geometry, material);
      mesh.position.set([-1.4, .15, 1.55, -1.25][i], -.04, [.6, -1.3, -2.8, -4.4][i]);
      mesh.rotation.y = i * .3; mesh.scale.setScalar(1 + i * .12);
      mesh.castShadow = mesh.receiveShadow = true; scene.add(mesh); meshes.push(mesh);
    }
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === scene && args[1] === camera) {
        frame++;
        if (frame === 8) {
          scene.fog = new T.FogExp2(0xc39b78, .16); phase.value = 1.1;
          meshes[0].position.z = -.1;
        }
        if (frame === 12) { scene.fog.color.set(0xa6b1cc); scene.fog.density = .22; }
        if (frame === 18) {
          scene.fog = new T.Fog(0x9bb4c3, 4.5, 12.5); phase.value = 2;
          meshes[2].material.fog = false;
        }
        if (frame === 22) { scene.fog.near = 3.8; scene.fog.far = 10.8; scene.fog.color.set(0x9aafbc); }
      }
      return render(...args);
    };
    return {scene, camera, light, meshes, ground, phase};
  },
  native_tsl_fog_pipeline(T, renderer) {
    const result = nativeMorphScenarios.native_tsl_morph_skinning(T, renderer), N = T.TSL;
    const source = result.scene, sourceCamera = result.camera;
    source.fog = new T.FogExp2(0x8aa4ba, .15);
    result.meshes[0].position.z = -.8; result.meshes[1].position.z = .3;
    result.meshes[1].material.fog = false;
    const scenePass = N.pass(source, sourceCamera), pipeline = new T.RenderPipeline(renderer, scenePass);
    renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
    const scene = new T.Scene(), camera = new T.PerspectiveCamera(), render = renderer.render.bind(renderer);
    let frame = 0;
    renderer.render = (...args) => {
      if (args[0] !== scene) return render(...args);
      frame++;
      if (frame === 8) { source.fog.color.set(0xb7a38d); source.fog.density = .24; }
      if (frame === 18) {
        source.fog = new T.FogExp2(0x91b6c1, .19);
        result.meshes[0].morphTargetInfluences[0] = .7;
      }
      if (frame === 22) { source.fog.density = .21; source.fog.color.set(0x9ab0c5); }
      return pipeline.render();
    };
    return {...result, scene, camera, source, sourceCamera, scenePass, pipeline};
  },
};
