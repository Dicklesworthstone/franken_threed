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
function litScene(T, materials = [new T.MeshToonNodeMaterial(), new T.MeshLambertNodeMaterial(), new T.MeshPhongNodeMaterial()]) {
  const result = {scene: scene(T), camera: camera(T)};
  const tints = [0xe29c54, 0x58bd9b, 0x709be2].map(color => T.TSL.uniform(new T.Color(color)));
  const geometry = new T.SphereGeometry(.59, 40, 24);
  for (const [i, material] of materials.entries()) {
    material.colorNode = tints[i]; material.emissive.setRGB(.015, .008, .02);
    const mesh = new T.Mesh(geometry, material);
    mesh.position.set((i - 1) * 1.26, (i % 2) * .16 - .08, 0);
    mesh.rotation.set(.12 * i, .35 * i, -.08 * i); result.scene.add(mesh);
  }
  if (materials[2].isMeshPhongNodeMaterial) {
    materials[2].specular.setRGB(.4, .55, .7); materials[2].shininess = 38;
  }
  const ambient = new T.AmbientLight(0xdde8ff, .18);
  const hemisphere = new T.HemisphereLight(0xa8caff, 0x69452c, .55);
  const directional = new T.DirectionalLight(0xffd1a6, 1.3); directional.position.set(-2.5, 3, 3);
  const point = new T.PointLight(0x72bcff, 12, 8, 2); point.position.set(1.5, .5, 2.3);
  const spot = new T.SpotLight(0xf2a9cc, 16, 10, .65, .3, 2); spot.position.set(-1.8, -2, 3);
  spot.target.position.set(.3, 0, 0);
  result.scene.add(ambient, hemisphere, directional, directional.target, point, spot, spot.target);
  return {...result, materials, tints, lights: {ambient, hemisphere, directional, point, spot}};
}
function pbrScene(T) {
  return litScene(T, [new T.MeshStandardNodeMaterial(), new T.MeshStandardNodeMaterial(), new T.MeshPhysicalNodeMaterial()]);
}
function sampledNormal(T) {
  const size = 16, pixels = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const nx = Math.sin(x / size * Math.PI * 4) * .45, ny = Math.cos(y / size * Math.PI * 4) * .35;
    const nz = Math.sqrt(1 - nx * nx - ny * ny);
    pixels.set([Math.round((nx + 1) * 127.5), Math.round((ny + 1) * 127.5), Math.round((nz + 1) * 127.5), 255], (y * size + x) * 4);
  }
  const result = new T.DataTexture(pixels, size, size);
  result.minFilter = result.magFilter = T.LinearFilter;
  result.wrapS = result.wrapT = T.RepeatWrapping;
  result.colorSpace = T.NoColorSpace; result.generateMipmaps = false; result.needsUpdate = true; return result;
}
function environmentTexture(T, variant) {
  const data = new Uint16Array(64 * 32 * 4);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 64; x++) {
    const u = (x + .5) / 64, v = (y + .5) / 32;
    const du = u - (.18 + .17 * variant), dv = v - (.25 + .12 * variant);
    const spot = Math.exp(-(du * du * 80 + dv * dv * 30));
    const color = [.08 + u * .5 + spot * (2.5 - .4 * variant),
      .1 + v * .35 + spot * (.8 + .35 * variant), .12 + (1 - u) * .45 + spot * (.25 + .5 * variant), 1];
    for (let channel = 0; channel < 4; channel++) data[(y * 64 + x) * 4 + channel] = T.DataUtils.toHalfFloat(color[channel]);
  }
  const texture = new T.DataTexture(data, 64, 32, T.RGBAFormat, T.HalfFloatType);
  texture.mapping = T.EquirectangularReflectionMapping; texture.colorSpace = T.LinearSRGBColorSpace;
  texture.minFilter = texture.magFilter = T.LinearFilter; texture.generateMipmaps = false; texture.needsUpdate = true;
  return texture;
}
export const nativeTSLScenarios = {
  native_tsl_environment(T, renderer) {
    const {uniform, vec3} = T.TSL;
    const standard = new T.MeshStandardNodeMaterial(), physical = new T.MeshPhysicalNodeMaterial();
    standard.colorNode = vec3(.75, .42, .16); standard.roughnessNode = uniform(.28); standard.metalnessNode = uniform(.65);
    physical.colorNode = vec3(.13, .48, .7); physical.roughnessNode = uniform(.36); physical.metalnessNode = uniform(.08);
    physical.clearcoatNode = uniform(.75); physical.clearcoatRoughnessNode = uniform(.12);
    physical.sheenNode = vec3(.3, .09, .22); physical.sheenRoughnessNode = uniform(.45); physical.retroreflectivityNode = uniform(.3);
    const environments = [environmentTexture(T, 0), environmentTexture(T, 1)];
    const result = {scene: new T.Scene(), camera: camera(T), materials: [standard, physical], environments};
    result.scene.environment = environments[0];
    // Keep background null so this compares lighting independently of the
    // separately owned panorama/background pass.
    const geometry = new T.SphereGeometry(.68, 32, 20);
    for (const [index, material] of result.materials.entries()) {
      const mesh = new T.Mesh(geometry, material); mesh.position.x = index === 0 ? -.82 : .82; result.scene.add(mesh);
    }
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      // PMREM filtering calls render() for its own internal meshes.
      if (args[0] !== result.scene) return render(...args);
      frame++;
      if (frame === 8) { result.scene.environmentIntensity = .45; result.scene.environmentRotation.set(.1, .8, .25); }
      if (frame === 13) {
        result.scene.environment = environments[1]; result.scene.environmentIntensity = .9;
        result.scene.environmentRotation.set(0, -.55, .1);
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_storage_structs(T, renderer) {
    const {struct, instancedArray, Fn, instanceIndex, positionLocal, vec3, mat3, uniform} = T.TSL;
    // Exact H8 particle layout: position at byte 0, velocity at 16, and
    // three padded mat3 columns at 32/48/64, for an 80-byte array stride.
    const Particle = struct({position: 'vec3', velocity: 'vec3', C: 'mat3'});
    const particles = instancedArray(new Float32Array(5 * 20), Particle), phase = uniform(.2);
    const initialize = Fn(() => {
      const index = instanceIndex.toFloat(), particle = particles.element(instanceIndex);
      particle.get('position').assign(vec3(index.sub(2).mul(.68), index.mul(1.3).add(phase).sin().mul(.22), 0));
      particle.get('velocity').assign(vec3(index.mul(.12).add(.15), phase.mul(.08).add(.38), index.mul(-.08).add(.85)));
      particle.get('C').assign(mat3(vec3(.8, phase.mul(.08), 0), vec3(-.12, .8, 0), vec3(0, 0, 1)));
    })().compute(5);
    const material = new T.MeshBasicNodeMaterial(), particle = particles.element(instanceIndex);
    // Share the source readWrite node with compute. Native graphics lowers its
    // vertex and fragment bindings to read-only storage, as the source does.
    material.positionNode = particle.get('C').mul(positionLocal).add(particle.get('position'));
    material.colorNode = particle.get('velocity');
    const mesh = new T.Mesh(new T.PlaneGeometry(.55, .7), material); mesh.count = 5; mesh.frustumCulled = false;
    const result = {scene: scene(T), camera: camera(T)}; result.scene.add(mesh);
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => {
      await renderer.computeAsync(initialize);
      // No needsUpdate: these deliberately stale CPU bytes must never overwrite
      // the authoritative GPU-written struct when graphics begins reading it.
      particles.value.array.fill(73);
      return compile(...args);
    };
    let frame = 0;
    renderer.render = (...args) => {
      frame++;
      if (frame === 8) { phase.value = 1.7; renderer.compute(initialize); }
      if (frame === 10) mesh.count = 3;
      if (frame === 12) mesh.count = 0;
      if (frame === 13) mesh.count = 5;
      if (particles.value.array.some(value => value !== 73)) throw new Error('Compute changed source CPU struct storage');
      return render(...args);
    };
    return {...result, mesh, particles, initialize, phase};
  },
  native_tsl_storage_instances(T, renderer) {
    const {storage, Fn, instanceIndex, positionLocal, vec4} = T.TSL;
    const positions = new T.StorageBufferAttribute(9, 4), output = storage(positions, 'vec4', 9);
    const initialize = Fn(() => {
      const index = instanceIndex.toFloat();
      output.element(instanceIndex).assign(vec4(index.mod(3).sub(1).mul(.85), index.div(3).floor().sub(1).mul(.75), 0, 1));
    })().compute(9);
    const material = new T.MeshBasicNodeMaterial(); material.color.setRGB(.15, .55, .8);
    material.positionNode = positionLocal.add(storage(positions, 'vec4', 9).toReadOnly().element(instanceIndex).xyz);
    const mesh = new T.Mesh(new T.PlaneGeometry(.6, .6), material); mesh.count = 9; mesh.frustumCulled = false;
    const result = {scene: scene(T), camera: camera(T)}; result.scene.add(mesh);
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => { await renderer.computeAsync(initialize); return compile(...args); };
    let frame = 0;
    renderer.render = (...args) => {
      frame++;
      if (frame === 8) mesh.count = 4;
      if (frame === 12) mesh.count = 0;
      if (frame === 13) mesh.count = 6;
      return render(...args);
    };
    return result;
  },
  native_tsl_storage(T) {
    const {storage, vertexIndex, uv, positionLocal} = T.TSL;
    const geometry = new T.PlaneGeometry(1.45, 1.4, 12, 12);
    const offsets = new T.StorageBufferAttribute(geometry.attributes.position.count, 4);
    for (let i = 0; i < offsets.count; i++) {
      const p = geometry.attributes.position;
      offsets.setXYZW(i, .04 * Math.cos(p.getY(i) * 7), .12 * Math.sin(p.getX(i) * 6), 0, 0);
    }
    const colors = new T.StorageBufferAttribute(new Float32Array([.8, .15, .08, 1, .08, .5, .85, 1]), 4);
    const material = new T.MeshBasicNodeMaterial();
    material.positionNode = positionLocal.add(storage(offsets, 'vec4', offsets.count).toReadOnly().element(vertexIndex).xyz);
    material.colorNode = storage(colors, 'vec4', colors.count).toReadOnly().element(uv().x.greaterThan(.5).select(1, 0).toUint()).rgb;
    const result = {scene: scene(T), camera: camera(T)};
    const mesh = new T.Mesh(geometry, material); mesh.rotation.set(.12, -.2, .1); result.scene.add(mesh);
    return result;
  },
  native_tsl_storage_compute(T, renderer) {
    const {storage, Fn, instanceIndex, vertexIndex, positionLocal, vec4, uniform} = T.TSL;
    const geometry = new T.PlaneGeometry(1.6, 1.45, 12, 12);
    const offsets = new T.StorageBufferAttribute(geometry.attributes.position.count, 4);
    const colors = new T.StorageBufferAttribute(1, 4), phase = uniform(.2);
    const outputOffsets = storage(offsets, 'vec4', offsets.count), outputColor = storage(colors, 'vec4', 1);
    const deform = Fn(() => {
      outputOffsets.element(instanceIndex).assign(vec4(0, instanceIndex.toFloat().mul(.35).add(phase).sin().mul(.12), 0, 0));
    })().compute(offsets.count);
    const recolor = Fn(() => { outputColor.element(0).assign(vec4(phase.mul(.25), .45, .8, 1)); })().compute(1);
    const material = new T.MeshBasicNodeMaterial();
    material.positionNode = positionLocal.add(storage(offsets, 'vec4', offsets.count).toReadOnly().element(vertexIndex).xyz);
    material.colorNode = storage(colors, 'vec4', 1).toReadOnly().element(0).rgb;
    const result = {scene: scene(T), camera: camera(T)};
    result.scene.add(new T.Mesh(geometry, material));
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => {
      await renderer.computeAsync([deform, recolor]);
      return compile(...args);
    };
    let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) { phase.value = 2.5; renderer.compute([deform, recolor]); }
      // CPU arrays remain their initial zeroes; rendering must consume GPU data.
      if (offsets.array.some(value => value !== 0) || colors.array.some(value => value !== 0))
        throw new Error('Compute unexpectedly changed source CPU storage');
      return render(...args);
    };
    return result;
  },
  native_tsl_storage_targets(T, renderer) {
    const {storage} = T.TSL;
    const colors = new T.StorageBufferAttribute(new Float32Array([.8, .2, .1, 1]), 4);
    const material = new T.MeshBasicNodeMaterial(); material.colorNode = storage(colors, 'vec4', 1).toReadOnly().element(0).rgb;
    const writer = scene(T); writer.add(new T.Mesh(new T.PlaneGeometry(1.8, 1.3), material));
    const writeCamera = new T.OrthographicCamera(-1.2, 1.2, .9, -.9, .1, 10); writeCamera.position.z = 3;
    const targets = Array.from({length: 2}, () => new T.RenderTarget(96, 72,
      {depthBuffer: false, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
    const result = {scene: scene(T), camera: camera(T, true)};
    for (const [i, target] of targets.entries()) {
      renderer.initRenderTarget(target);
      const panel = new T.Mesh(new T.PlaneGeometry(1.85, 1.4), new T.MeshBasicMaterial({map: target.texture}));
      panel.position.x = i ? 1 : -1; result.scene.add(panel);
    }
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 1 || frame === 8) {
        for (const [i, target] of targets.entries()) {
          colors.array.set(frame === 1 ? (i ? [.1, .3, .85, 1] : [.85, .2, .1, 1])
            : (i ? [.6, .15, .8, 1] : [.1, .75, .3, 1]));
          colors.needsUpdate = true;
          renderer.setRenderTarget(target); render(writer, writeCamera);
        }
        renderer.setRenderTarget(null);
        colors.array.fill(0); // No upload request: neither submitted target may change.
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_storage_fifo(T, renderer) {
    const {Fn, If, storage, instanceIndex, vertexIndex, uniform, float, int, uint, vec3, vec4, uv} = T.TSL;
    const geometry = new T.PlaneGeometry(1.7, 1.25, 16, 10), count = geometry.attributes.position.count;
    const initial = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) initial.set([
      geometry.attributes.position.getX(i), geometry.attributes.position.getY(i), 0, 1,
    ], i * 4);
    const baseData = new T.StorageBufferAttribute(initial, 4);
    const positionData = new T.StorageBufferAttribute(new Float32Array(count * 4), 4);
    const colorData = new T.StorageBufferAttribute(new Float32Array(12), 4);
    const lookupData = new T.StorageBufferAttribute(new Uint32Array(3), 1);
    const signedData = new T.StorageBufferAttribute(new Int32Array(3), 1);
    const base = storage(baseData, 'vec4', count).toReadOnly();
    const positions = storage(positionData, 'vec4', count), colors = storage(colorData, 'vec4', 3);
    const lookup = storage(lookupData, 'uint', 3), signed = storage(signedData, 'int', 3);
    const shift = uniform(0), phase = uniform(0), gain = uniform(.8), rotation = uniform(0, 'uint');
    const kernel = Fn(() => {
      const p = base.element(instanceIndex);
      positions.element(instanceIndex).assign(vec4(p.xyz.add(vec3(shift, p.x.mul(4).add(phase).sin().mul(.13), 0)), 1));
      If(instanceIndex.lessThan(3), () => {
        const i = float(instanceIndex);
        colors.element(instanceIndex).assign(vec4(vec3(i.mul(.21).add(.15), i.mul(-.12).add(.65), i.mul(.18).add(.25)).mul(gain), 1));
        lookup.element(instanceIndex).assign(instanceIndex.add(rotation).mod(3));
        signed.element(instanceIndex).assign(int(instanceIndex).sub(1));
      });
    })().compute(count, [64]);
    // Separate source nodes express graphics read access to the exact compute attributes.
    const positionRead = storage(positionData, 'vec4', count).toReadOnly();
    const colorRead = storage(colorData, 'vec4', 3).toReadOnly(), lookupRead = storage(lookupData, 'uint', 3).toReadOnly();
    const signedRead = storage(signedData, 'int', 3).toReadOnly();
    const material = new T.MeshBasicNodeMaterial();
    material.positionNode = positionRead.element(vertexIndex).xyz.add(vec3(0, colorRead.element(0).g.mul(.2), 0));
    const band = uint(uv().x.mul(3).min(2));
    material.colorNode = colorRead.element(lookupRead.element(band)).rgb.add(float(signedRead.element(band)).mul(.045));
    const writer = scene(T), mesh = new T.Mesh(geometry, material);
    mesh.frustumCulled = false; writer.add(mesh);
    const writeCamera = new T.OrthographicCamera(-1.35, 1.35, 1, -1, .1, 10); writeCamera.position.z = 3;
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
      // Compute owns storage allocation; graphics borrows those resident buffers.
      await renderer.compileComputeAsync(kernel);
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
          for (let i = 0; i < count; i++) baseData.array[i * 4 + 1] *= .8;
          baseData.needsUpdate = true;
          // Reset the source versions before a queued compute overwrites their zeros.
          // Graphics must resolve that pending version without uploading CPU zeros again.
          positionData.needsUpdate = colorData.needsUpdate = lookupData.needsUpdate = signedData.needsUpdate = true;
        }
        for (const [i, target] of targets.entries()) {
          shift.value = i ? .22 : -.24; phase.value = (frame === 8 ? 1.2 : .2) + i * 1.4;
          gain.value = frame === 8 ? (i ? .45 : .9) : (i ? .8 : .55);
          rotation.value = frame === 8 ? 2 - i : i;
          renderer.compute(kernel);
          renderer.setRenderTarget(target); render(writer, writeCamera);
        }
        renderer.setRenderTarget(null);
        // Later application values must not replace either accepted dispatch's uniforms.
        shift.value = 4; phase.value = 8; gain.value = .05; rotation.value = 0;
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_storage_instanced_mesh(T, renderer) {
    const {Fn, storage, instanceIndex, positionLocal, uniform, vec3, vec4, mat4} = T.TSL;
    const count = 6, matrices = new T.StorageInstancedBufferAttribute(count, 16);
    // The source uploader pads these three-word colors to vec3's four-word stride.
    const colors = new T.StorageInstancedBufferAttribute(count, 3);
    const offsets = new T.StorageBufferAttribute(count, 4), weights = new T.StorageBufferAttribute(count, 4);
    const matrixWrite = storage(matrices, 'mat4', count), colorWrite = storage(colors, 'vec3', count);
    const offsetWrite = storage(offsets, 'vec4', count), weightWrite = storage(weights, 'vec4', count);
    const phase = uniform(.2), tint = uniform(new T.Color(.95, .82, .72));
    const initialize = Fn(() => {
      const index = instanceIndex.toFloat(), angle = phase.add(index.mul(.27)), tilt = index.mul(.18).add(.25);
      const cz = angle.cos(), sz = angle.sin(), cy = tilt.cos(), sy = tilt.sin();
      const sx = index.mul(.06).add(.7), scaleY = index.mod(2).mul(.2).add(.9), scaleZ = phase.mul(.05).add(.65);
      // Column-major Rz * Ry * scale plus translation: nonuniform scale also
      // requires the source instance normal transform for the lit faces.
      matrixWrite.element(instanceIndex).assign(mat4(
        vec4(cz.mul(cy).mul(sx), sz.mul(cy).mul(sx), sy.negate().mul(sx), 0),
        vec4(sz.negate().mul(scaleY), cz.mul(scaleY), 0, 0),
        vec4(cz.mul(sy).mul(scaleZ), sz.mul(sy).mul(scaleZ), cy.mul(scaleZ), 0),
        vec4(index.mod(3).sub(1).mul(.95), index.div(3).floor().sub(.5).mul(.9), index.mul(.025), 1),
      ));
      colorWrite.element(instanceIndex).assign(vec3(index.mul(.1).add(.25), phase.mul(.08).add(.3), index.mul(-.08).add(.8)));
      offsetWrite.element(instanceIndex).assign(vec4(angle.sin().mul(.07), angle.cos().mul(.04), 0, 0));
      weightWrite.element(instanceIndex).assign(vec4(index.mul(.04).add(.65), phase.mul(-.025).add(.8), .9, 1));
    })().compute(count, [64]);
    const material = new T.MeshPhongNodeMaterial(); material.shininess = 24; material.specular.setRGB(.25, .3, .4);
    // Authored nodes compose with the built-in instance transform/color, and
    // independently address storage by instanceIndex in both shader stages.
    material.positionNode = positionLocal.add(storage(offsets, 'vec4', count).toReadOnly().element(instanceIndex).xyz);
    material.colorNode = tint.mul(storage(weights, 'vec4', count).toReadOnly().element(instanceIndex).rgb);
    const mesh = new T.InstancedMesh(new T.BoxGeometry(.62, .5, .42), material, count);
    mesh.instanceMatrix = matrices; mesh.instanceColor = colors; mesh.frustumCulled = false;
    const writer = scene(T), light = new T.DirectionalLight(0xe5edff, 1.65); light.position.set(-2, 3, 4);
    writer.add(mesh, new T.AmbientLight(0xffd5b0, .35), light, light.target);
    const writeCamera = new T.OrthographicCamera(-1.6, 1.6, 1.2, -1.2, .1, 10); writeCamera.position.z = 4;
    const targets = Array.from({length: 4}, () => new T.RenderTarget(128, 96,
      {generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
    const result = {scene: scene(T), camera: camera(T, true)};
    for (const [i, target] of targets.entries()) {
      renderer.initRenderTarget(target);
      const panel = new T.Mesh(new T.PlaneGeometry(1.9, 1.425), new T.MeshBasicMaterial({map: target.texture}));
      panel.position.set(i % 2 ? 1 : -1, i < 2 ? .76 : -.76, 0); result.scene.add(panel);
    }
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => {
      await renderer.compileComputeAsync(initialize);
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
          // A fresh source version is uploaded once before compute replaces it;
          // instance setup must not upload these zeros over the GPU result.
          for (const attribute of [matrices, colors, offsets, weights]) {
            attribute.array.fill(0); attribute.needsUpdate = true;
          }
        }
        for (const [i, target] of targets.entries()) {
          phase.value = (frame === 8 ? 1.1 : .2) + i * .45;
          tint.value.setRGB(.95 - i * .1, .75 + i * .06, frame === 8 ? .9 : .65);
          renderer.compute(initialize);
          mesh.count = (frame === 8 ? [6, 3, 0, 5] : [6, 4, 0, 2])[i];
          renderer.setRenderTarget(target); render(writer, writeCamera);
        }
        renderer.setRenderTarget(null);
        // Each accepted target retains its count, uniforms, and GPU writes.
        mesh.count = 1; phase.value = 20; tint.value.setRGB(0, 0, 0);
        for (const attribute of [matrices, colors, offsets, weights]) attribute.array.fill(73);
      }
      for (const attribute of [matrices, colors, offsets, weights])
        if (attribute.array.some(value => value !== 73)) throw new Error('Compute changed source CPU instance storage');
      return render(...args);
    };
    return {...result, mesh, initialize, matrices, colors, offsets, weights};
  },
  native_tsl_pbr_materials(T) {
    const {uniform, vec2, vec3} = T.TSL;
    const result = pbrScene(T), [rough, metal, physical] = result.materials;
    rough.roughness = .72; rough.metalness = .05;
    metal.roughnessNode = uniform(.23); metal.metalnessNode = uniform(.88);
    physical.roughnessNode = uniform(.3); physical.metalnessNode = uniform(.32);
    physical.clearcoatNode = uniform(.8); physical.clearcoatRoughnessNode = uniform(.13);
    physical.sheenNode = vec3(.12, .38, .55); physical.sheenRoughnessNode = uniform(.45);
    physical.iridescenceNode = uniform(.65); physical.iridescenceIORNode = uniform(1.4);
    physical.iridescenceThicknessNode = uniform(330); physical.anisotropyNode = vec2(.45, .2);
    physical.retroreflectivityNode = uniform(.3); physical.iorNode = uniform(1.65);
    physical.specularColorNode = vec3(.8, .95, 1); physical.specularIntensityNode = uniform(.8);
    return result;
  },
  native_tsl_pbr_updates(T, renderer) {
    const {uniform} = T.TSL;
    const result = pbrScene(T), [defaults, authored, physical] = result.materials;
    const roughness = uniform(.22), metalness = uniform(.65), emission = uniform(new T.Color(.01, .02, .03));
    defaults.roughness = .8; defaults.metalness = .05;
    authored.roughnessNode = roughness; authored.metalnessNode = metalness; authored.emissiveNode = emission;
    physical.roughness = .55; physical.metalness = .05;
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) {
        defaults.roughness = .18; defaults.metalness = .9;
        roughness.value = .68; metalness.value = .15; emission.value.setRGB(.035, .01, .07);
        result.tints[0].value.setRGB(.8, .42, .12); result.tints[1].value.setRGB(.12, .4, .65);
        physical.roughness = .24; physical.metalness = .35;
        physical.clearcoat = .85; physical.clearcoatRoughness = .12;
        physical.sheen = .8; physical.sheenColor.setRGB(.65, .18, .4); physical.sheenRoughness = .4;
        physical.iridescence = .7; physical.iridescenceIOR = 1.35; physical.iridescenceThicknessRange = [180, 380];
        physical.anisotropy = .55; physical.anisotropyRotation = .6; physical.retroreflectivity = .35;
        physical.ior = 1.7; physical.specularColor.setRGB(.9, .7, .5); physical.specularIntensity = .75;
        // Enabling source physical lobes changes the program as well as its uniforms.
        physical.needsUpdate = true;
        result.lights.point.position.set(-.8, .9, 2.1); result.lights.point.intensity = 16;
        result.lights.directional.position.set(1.8, 2.5, 2.8);
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_pbr_textures(T) {
    const {texture, normalMap, uv, vec2} = T.TSL;
    const result = pbrScene(T), normal = sampledNormal(T), coatNormal = sampledNormal(T);
    normal.repeat.set(1.3, 1.1); coatNormal.repeat.set(.8, 1.7); coatNormal.rotation = .4;
    const properties = sampledTexture(T, 2, T.NoColorSpace), glow = sampledTexture(T, 1);
    for (const [i, material] of result.materials.entries()) {
      const color = sampledTexture(T, i); color.offset.set(.12 * i, .05); color.repeat.set(1.5, .9);
      material.colorNode = texture(color).rgb;
      material.normalNode = normalMap(texture(normal), vec2(.65, .8));
      material.roughnessNode = texture(properties).g.mul(.65).add(.15);
      material.metalnessNode = texture(properties).r.mul(.8);
      material.emissiveNode = texture(glow, uv().yx).rgb.mul(.05);
    }
    const physical = result.materials[2];
    physical.clearcoatNode = texture(properties).b;
    physical.clearcoatRoughnessNode = texture(properties).r.mul(.35).add(.08);
    physical.clearcoatNormalNode = normalMap(texture(coatNormal), vec2(.6, .7));
    physical.sheenNode = texture(glow).rgb.mul(.45); physical.sheenRoughnessNode = texture(properties).g;
    physical.iridescenceNode = texture(properties).r; physical.iridescenceIORNode = texture(properties).g.mul(.3).add(1.2);
    physical.iridescenceThicknessNode = texture(properties).b.mul(220).add(160);
    physical.anisotropyNode = texture(properties).rg.sub(.5).mul(.8);
    physical.retroreflectivityNode = texture(properties).g.mul(.3);
    physical.iorNode = texture(properties).r.mul(.5).add(1.3);
    physical.specularColorNode = texture(glow).rgb; physical.specularIntensityNode = texture(properties).b;
    return result;
  },
  native_tsl_lit_materials(T) {
    return litScene(T);
  },
  native_tsl_lit_updates(T, renderer) {
    const result = litScene(T), {ambient, hemisphere, directional, point, spot} = result.lights;
    const alternate = new T.PointLight(0xe89a51, 10, 9, 2); alternate.position.set(-1, -.5, 2.2);
    alternate.layers.set(1); alternate.visible = false; result.scene.add(alternate);
    result.scene.traverse(object => { if (object !== point && object !== alternate) object.layers.enable(1); });
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) {
        result.camera.layers.set(1); alternate.visible = true;
        ambient.color.set(0xffc9b7); ambient.intensity = .3;
        hemisphere.color.set(0xb1e6bf); hemisphere.groundColor.set(0x493867); hemisphere.intensity = .7;
        directional.position.set(2.2, 1.1, 2.8); directional.target.position.set(-.6, -.3, 0);
        directional.color.set(0xc3d9ff); directional.intensity = .9;
        // The old point is now outside the camera layer; its large intensity must not leak.
        point.intensity = 200;
        alternate.color.set(0xfbc26c); alternate.intensity = 18; alternate.distance = 7; alternate.decay = 1.8;
        spot.position.set(.7, 2.7, 2.4); spot.target.position.set(-.4, -.5, 0);
        spot.color.set(0x9dcfff); spot.intensity = 13; spot.distance = 8; spot.decay = 1.7;
        spot.angle = .52; spot.penumbra = .55;
        for (const [i, tint] of result.tints.entries()) tint.value.setRGB(.2 + i * .2, .55 - i * .12, .35 + i * .2);
        result.materials[1].emissive.setRGB(.04, .01, .025);
        result.materials[2].specular.setRGB(.75, .3, .2); result.materials[2].shininess = 12;
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_lit_textures(T) {
    const {texture, normalMap, uv, vec2} = T.TSL;
    const result = litScene(T), normals = sampledNormal(T);
    normals.repeat.set(1.4, 1.1); normals.offset.set(.1, -.05);
    const emission = sampledTexture(T, 1), gloss = sampledTexture(T, 2, T.NoColorSpace);
    for (const [i, material] of result.materials.entries()) {
      const color = sampledTexture(T, i); color.repeat.set(1.2 + .2 * i, .8); color.offset.set(.08 * i, .1);
      material.colorNode = texture(color).rgb;
      material.normalNode = normalMap(texture(normals), vec2(.7, .85));
      material.emissiveNode = texture(emission, uv().yx).rgb.mul(.08);
    }
    result.materials[2].specularNode = texture(gloss).rgb.mul(.55).add(.08);
    result.materials[2].shininessNode = texture(gloss).g.mul(60).add(8);
    return result;
  },
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
