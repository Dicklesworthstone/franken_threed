// Ordinary source material environments, including the Phong path used by FBX.
function cube(T, phase = 0) {
  const size = 8, faces = Array.from({length: 6}, (_, face) => {
    const pixels = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      pixels.set([35 + ((face * 39 + x * 17 + phase * 57) % 210),
        30 + ((face * 23 + y * 25 + phase * 83) % 220),
        25 + ((face * 31 + x * 13 + y * 11 + phase * 41) % 225), 255], (y * size + x) * 4);
    }
    const texture = new T.DataTexture(pixels, size, size);
    texture.minFilter = texture.magFilter = T.LinearFilter; texture.needsUpdate = true;
    return texture;
  });
  const texture = new T.CubeTexture(faces);
  texture.colorSpace = T.LinearSRGBColorSpace;
  texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.generateMipmaps = false; texture.needsUpdate = true;
  return texture;
}

function panorama(T, phase = 0) {
  const width = 32, height = 16, pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++)
    pixels.set([30 + ((x * 7 + phase * 63) % 220), 25 + ((y * 14 + phase * 97) % 225),
      35 + (((x ^ y) * 13 + phase * 47) % 210), 255], (y * width + x) * 4);
  const texture = new T.DataTexture(pixels, width, height);
  texture.mapping = T.EquirectangularReflectionMapping; texture.colorSpace = T.LinearSRGBColorSpace;
  texture.minFilter = texture.magFilter = T.LinearFilter; texture.needsUpdate = true;
  return texture;
}

function world(T) {
  const scene = new T.Scene(); scene.background = new T.Color(.035, .055, .09);
  const camera = new T.PerspectiveCamera(43, 4 / 3, .1, 30);
  camera.position.set(.15, .25, 5.4); camera.lookAt(0, 0, 0);
  const light = new T.DirectionalLight(0xffe0c0, 2.1); light.position.set(-2, 3, 4);
  scene.add(light, new T.HemisphereLight(0xc7dcff, 0x635344, .8));
  return {scene, camera};
}

function drive(T, renderer, result, update) {
  const render = renderer.render.bind(renderer), read = renderer.readRenderTargetPixelsAsync.bind(renderer);
  const target = renderer.getRenderTarget(); let frame = 0, previous = null;
  renderer.render = (...args) => {
    if (args[0] === result.scene && args[1] === result.camera) {
      renderer._nodes.nodeFrame.update(); update(++frame);
    }
    return render(...args);
  };
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] === target) {
      const colors = new Set();
      for (let i = 0; i < pixels.length; i += 4) colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
      if (colors.size < 32) throw Error('Environment materials must produce visible, nonconstant output');
      if ([8, 10, 12, 13].includes(frame) && previous && pixels.every((value, i) => value === previous[i]))
        throw Error(`Environment controls failed to change the rendered output at frame ${frame}`);
      previous = pixels.slice();
    }
    return pixels;
  };
  return result;
}

export const basicEnvironmentScenarios = {
  native_basic_environment_cube(T, renderer) {
    const result = world(T), first = cube(T), second = cube(T, 1);
    const materials = [new T.MeshBasicMaterial({color: 0xc5a37f, combine: T.MultiplyOperation}),
      new T.MeshLambertMaterial({color: 0x83bfc3, combine: T.MixOperation}),
      new T.MeshPhongMaterial({color: 0xc79dc9, combine: T.AddOperation, shininess: 35})];
    materials.forEach((material, index) => {
      material.envMap = first; material.reflectivity = .7; material.refractionRatio = .85;
      const mesh = new T.Mesh(new T.SphereGeometry(.61, 28, 20), material);
      mesh.position.x = (index - 1) * 1.27; result.scene.add(mesh);
    });
    return drive(T, renderer, result, frame => {
      if (frame === 8) materials.forEach((material, index) => {
        material.reflectivity = .2 + index * .2; material.envMapRotation.set(.25, -.6, .15);
      });
      if (frame === 10) {
        first.mapping = T.CubeRefractionMapping;
        materials.forEach(material => { material.refractionRatio = .58; material.needsUpdate = true; });
      }
      if (frame === 12) materials.forEach(material => { material.envMap = second; material.needsUpdate = true; });
      if (frame === 13) materials.forEach((material, index) => {
        // Source RenderObject aliases Mix/Add as nonzero cache values. Dispose
        // the previous material programs before selecting a new combine mode.
        material.dispose();
        material.combine = [T.MixOperation, T.AddOperation, T.MultiplyOperation][index]; material.needsUpdate = true;
      });
    });
  },
  native_basic_environment_panorama(T, renderer) {
    const result = world(T), first = panorama(T), second = panorama(T, 1), direct = cube(T, 2);
    const material = new T.MeshLambertMaterial({color: 0xb6a9d2, envMap: first, reflectivity: .85, combine: T.MixOperation});
    const authored = new T.MeshPhongNodeMaterial({color: 0xa9c7b6, reflectivity: .7, combine: T.AddOperation, shininess: 45});
    authored.envNode = T.TSL.texture(first);
    [material, authored].forEach((value, index) => {
      const mesh = new T.Mesh(new T.TorusKnotGeometry(.42, .17, 64, 12), value);
      mesh.position.x = index ? .87 : -.87; mesh.rotation.set(.2, -.3, 0); result.scene.add(mesh);
    });
    return drive(T, renderer, result, frame => {
      if (frame === 8) for (const value of [material, authored]) { value.envMapRotation.y = .9; value.reflectivity = .4; }
      if (frame === 10) {
        first.mapping = T.EquirectangularRefractionMapping;
        // A source TextureNode cache key excludes its texture's mapping.
        authored.envNode = T.TSL.texture(first);
        for (const value of [material, authored]) { value.refractionRatio = .65; value.needsUpdate = true; }
      }
      if (frame === 12) {
        material.envMap = second; authored.envNode = T.TSL.texture(second);
        material.needsUpdate = authored.needsUpdate = true;
      }
      if (frame === 13) {
        material.envMap = direct; authored.envNode = T.TSL.cubeTexture(direct);
        material.needsUpdate = authored.needsUpdate = true;
      }
    });
  },
  native_basic_environment_skinning(T, renderer) {
    const result = world(T), env = cube(T), geometry = new T.CylinderGeometry(.32, .42, 1.9, 24, 18);
    const indices = [], weights = [], position = geometry.attributes.position;
    for (let i = 0; i < position.count; i++) {
      const weight = T.MathUtils.clamp((position.getY(i) + .95) / 1.9, 0, 1);
      indices.push(0, 1, 0, 0); weights.push(1 - weight, weight, 0, 0);
    }
    geometry.setAttribute('skinIndex', new T.Uint16BufferAttribute(indices, 4));
    geometry.setAttribute('skinWeight', new T.Float32BufferAttribute(weights, 4));
    const material = new T.MeshPhongMaterial({color: 0xc79c70, envMap: env, reflectivity: .7,
      combine: T.MixOperation, shininess: 45});
    const mesh = new T.SkinnedMesh(geometry, material), root = new T.Bone(), tip = new T.Bone();
    root.position.y = -.95; tip.position.y = 1.9; root.add(tip); mesh.add(root);
    mesh.bind(new T.Skeleton([root, tip])); tip.rotation.z = .45; result.scene.add(mesh);
    return drive(T, renderer, result, frame => {
      if (frame === 8) { tip.rotation.z = -.55; root.rotation.y = .35; }
      if (frame === 10) { material.envMap = null; material.needsUpdate = true; }
      if (frame === 12) { material.envMap = env; material.needsUpdate = true; }
      if (frame === 13) {
        env.mapping = T.CubeRefractionMapping; material.refractionRatio = .6; material.needsUpdate = true;
      }
    });
  },
};
