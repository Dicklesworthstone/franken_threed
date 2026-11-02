// Identical source/candidate compute producers and graphics texel consumers.
// Every authored mip has distinct values; no automatic mip generation is used.
function storageReads(T, renderer, array, integer) {
  const N = T.TSL, scene = new T.Scene(), kernels = [], meshes = [];
  scene.background = new T.Color(0x182435);
  const texture = array ? new T.StorageArrayTexture(4, 4, 2) : new T.StorageTexture(4, 4);
  texture.type = integer ? T.UnsignedIntType : array ? T.FloatType : T.UnsignedByteType;
  texture.format = T.RGBAFormat; texture.colorSpace = T.NoColorSpace;
  texture.minFilter = texture.magFilter = T.NearestFilter;
  texture.generateMipmaps = true; texture.mipmapsAutoUpdate = false;
  const phase = N.uniform(integer ? 64 : .25, integer ? 'uint' : 'float');
  for (let mip = 0; mip < 3; mip++) {
    const size = 4 >> mip;
    kernels.push(N.Fn(() => {
      const layer = N.instanceIndex.div(size * size);
      const coordinates = N.uvec2(N.instanceIndex.mod(size), N.instanceIndex.div(size).mod(size));
      let output = N.storageTexture(texture).toWriteOnly().setMipLevel(mip);
      if (array) output = output.depth(layer);
      const value = integer ? N.uvec4(32 + mip * 64, phase, 192 - mip * 48, 255)
        : N.vec4(.125 + mip * .25, phase, layer.toFloat().mul(.125).add(.25), 1);
      N.textureStore(output, coordinates, value);
    })().compute(size * size * (array ? 2 : 1)));
    let input = N.storageTexture(texture, N.ivec2(0, 0)).setSampler(false).toReadOnly().setMipLevel(mip);
    if (array) input = input.depth(N.int(1));
    const color = integer ? N.vec3(input.rgb).div(255) : input.rgb;
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = color;
    material.positionNode = N.positionLocal.add(N.vec3(color.r.mul(.2), 0, 0));
    const mesh = new T.Mesh(new T.PlaneGeometry(.9, 1.4), material);
    mesh.position.x = (mip - 1) * 1.2; mesh.frustumCulled = false;
    scene.add(mesh); meshes.push(mesh);
  }
  const camera = new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 20);
  camera.position.z = 5;
  const compile = renderer.compileAsync.bind(renderer);
  renderer.compileAsync = async (...args) => { await renderer.computeAsync(kernels); return compile(...args); };
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    frame++;
    if (frame === 8 || frame === 13) {
      phase.value = integer ? frame === 8 ? 192 : 96 : frame === 8 ? .75 : .375;
      renderer.compute(kernels);
    }
    if (frame === 10) meshes[1].visible = false;
    if (frame === 12) meshes[1].visible = true;
    return render(...args);
  };
  return {scene, camera};
}

// Ordinary texture painting follows the pinned partial-update example: each
// copy changes GPU storage while the material and its CPU image remain intact.
// The same public calls and controls execute on the source and candidate.
function sampledCopies(T, renderer, imageDestination, aliases, bundled = false, preserveHistory = false) {
  const scene = new T.Scene(), width = 8, height = 8;
  const group = bundled ? new T.BundleGroup() : scene;
  if (bundled) { group.static = true; scene.add(group); }
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const offset = (y * width + x) * 4;
    pixels.set([20 + x * 23, 30 + y * 21, 190 - x * 13 - y * 7, 255], offset);
  }
  let destination, ready = Promise.resolve();
  if (imageDestination) {
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
    const image = new Image(); image.src = canvas.toDataURL('image/png'); ready = image.decode();
    destination = new T.Texture(image);
  } else destination = new T.DataTexture(pixels, width, height);
  destination.generateMipmaps = false;
  destination.minFilter = destination.magFilter = imageDestination ? T.LinearFilter : T.NearestFilter;
  destination.colorSpace = imageDestination ? T.SRGBColorSpace : T.NoColorSpace;
  destination.needsUpdate = true;
  const alias = aliases ? destination.clone() : null;
  const sourceBytes = Uint8Array.of(240, 20, 40, 255, 20, 230, 60, 255,
    30, 40, 220, 255, 220, 210, 20, 255);
  const source = new T.DataTexture(sourceBytes, 2, 2);
  source.colorSpace = destination.colorSpace; source.needsUpdate = true;
  const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 10); camera.position.z = 3;
  const add = (texture, x, size) => {
    const mesh = new T.Mesh(new T.PlaneGeometry(size, 2.4), new T.MeshBasicMaterial({map: texture}));
    mesh.position.x = x; mesh.frustumCulled = false; group.add(mesh);
  };
  if (alias) { add(destination, -1, 1.8); add(alias, 1, 1.8); }
  else add(destination, 0, 3.2);
  const compile = renderer.compileAsync.bind(renderer);
  renderer.compileAsync = async (...args) => { await ready; return compile(...args); };
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    frame++;
    if (frame === 2 && preserveHistory) {
      // A copy initializes storage only when its Texture version requests it.
      // Keep both an already uploaded alias and changed CPU/source identities
      // GPU-stale until the explicit needsUpdate at frame 10.
      pixels.fill(77);
      destination.source = new T.Source({width, height, data: pixels});
    }
    if (frame === 2 || frame === 7)
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(frame === 2 ? 1 : 4, 1));
    if (frame === 8) {
      sourceBytes.set([240, 120, 20, 255], 8); sourceBytes.set([20, 120, 240, 255], 12);
      source.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination,
        new T.Box2(new T.Vector2(0, 1), new T.Vector2(2, 2)), new T.Vector2(2, 5));
    }
    if (frame === 10) {
      if (!imageDestination) {
        for (let i = 0; i < pixels.length; i += 4) pixels.set([150, 60 + i % 128, 90, 255], i);
        // Native r186 deliberately ignores DataTexture's WebGL row ranges.
        destination.addUpdateRange(0, 4);
      }
      if (bundled) {
        destination.source = new T.Source({width, height, data: pixels});
        destination.minFilter = destination.magFilter = T.LinearFilter;
        destination.flipY = true;
      }
      destination.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(5, 5));
    }
    if (frame === 12) {
      destination.dispose();
      if (bundled) group.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(0, 6));
    }
    if (frame === 13) {
      source.flipY = true; source.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, null, new T.Vector2(5, 0));
    }
    return render(...args);
  };
  return {scene, camera};
}

function sampledMipCopies(T, renderer, generated) {
  const N = T.TSL, scene = new T.Scene();
  const pixels = (size, level, seed) => Uint8Array.from({length: size * size * 4}, (_, offset) => {
    const channel = offset % 4, pixel = Math.floor(offset / 4);
    return channel === 3 ? 255 : (seed + level * 53 + channel * 47 + pixel * (channel + 1) * 7) % 256;
  });
  const chain = (size, seed) => Array.from({length: 1 + Math.log2(size)}, (_, level) => {
    const width = size >> level; return {width, height: width, data: pixels(width, level, seed)};
  });
  const destination = new T.DataTexture(pixels(8, 0, 17), 8, 8);
  destination.generateMipmaps = generated; destination.minFilter = T.NearestMipmapNearestFilter;
  if (!generated) destination.mipmaps = chain(8, 17);
  destination.needsUpdate = true;
  const alias = destination.clone();
  const source = new T.DataTexture(pixels(4, 0, 91), 4, 4);
  source.mipmaps = chain(4, 91); source.minFilter = T.NearestMipmapNearestFilter; source.needsUpdate = true;
  for (const [row, texture] of [destination, alias].entries()) for (let level = 0; level < 4; level++) {
    const size = 8 >> level, material = new T.MeshBasicNodeMaterial();
    material.colorNode = N.textureLoad(texture, N.ivec2(N.uv().mul(size)), N.int(level)).rgb;
    const mesh = new T.Mesh(new T.PlaneGeometry(.95, 1.1), material);
    mesh.position.set((level - 1.5) * 1.15, row === 0 ? .7 : -.7, 0); mesh.frustumCulled = false; scene.add(mesh);
  }
  const camera = new T.OrthographicCamera(-2.5, 2.5, 1.8, -1.8, .1, 10); camera.position.z = 3;
  const region = size => new T.Box2(new T.Vector2(), new T.Vector2(size, size));
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    frame++;
    if (frame === 2) renderer.copyTextureToTexture(source, destination, region(4), new T.Vector2(2, 2));
    if (frame === 7) renderer.copyTextureToTexture(source, destination, region(2), new T.Vector2(1, 1), 1, 1);
    if (frame === 8) {
      destination.generateMipmaps = true;
      source.mipmaps[2].data.set([240, 40, 80, 255]); source.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, region(1), new T.Vector2(0, 0), 2, 2);
    }
    if (frame === 10) {
      destination.generateMipmaps = false;
      destination.image.data.fill(153);
      if (!generated) destination.mipmaps.forEach(mip => mip.data.fill(153));
      renderer.copyTextureToTexture(source, destination, region(4), new T.Vector2(4, 4));
    }
    if (frame === 12) {
      destination.generateMipmaps = true; destination.dispose();
      renderer.copyTextureToTexture(source, destination, region(2), new T.Vector2(0, 0), 1, 1);
    }
    if (frame === 13) {
      destination.needsUpdate = true;
      renderer.copyTextureToTexture(source, destination, region(1), new T.Vector2(0, 0), 2, 0);
    }
    return render(...args);
  };
  return {scene, camera};
}

// GPU-authored source bytes are copied directly into an ordinary material map.
// A compute consumer reads that same map in the next FIFO slot; the two planes
// therefore prove both consumers observe the copy, including warm bundles.
function producedCopies(T, renderer, {target = false, array = false, volume = false, mip = false, mrt = false, samples = 0, bundled = false} = {}) {
  const N = T.TSL, scene = new T.Scene(), width = 8, pixels = new Uint8Array(width * width * 4);
  for (let i = 0; i < pixels.length; i += 4) pixels.set([20 + i % 71, 40 + i % 53, 150, 255], i);
  const destination = new T.DataTexture(pixels, width, width);
  destination.generateMipmaps = false; destination.minFilter = destination.magFilter = T.NearestFilter; destination.needsUpdate = true;
  const consumer = new T.StorageTexture(width, width);
  consumer.generateMipmaps = false; consumer.minFilter = consumer.magFilter = T.NearestFilter;
  const consume = N.Fn(() => {
    const coordinates = N.uvec2(N.instanceIndex.mod(width), N.instanceIndex.div(width));
    N.textureStore(consumer, coordinates, N.textureLoad(destination, coordinates));
  })().compute(width * width);
  const group = bundled ? new T.BundleGroup() : scene;
  if (bundled) { group.static = true; scene.add(group); }
  for (const [index, texture] of [destination, consumer].entries()) {
    const mesh = new T.Mesh(new T.PlaneGeometry(1.8, 2.4), new T.MeshBasicMaterial({map: texture}));
    mesh.position.x = index ? 1 : -1; mesh.frustumCulled = false; group.add(mesh);
  }
  const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 10); camera.position.z = 3;
  const phase = N.uniform(.25), level = mip ? 1 : 0, sourceWidth = 4, size = sourceWidth >> level;
  let source, sourceTarget, producer, sourceScene, sourceCamera, attachment = 0;
  if (target) {
    sourceTarget = new T.RenderTarget(sourceWidth, sourceWidth, {count: mrt ? 2 : 1, depthBuffer: false,
      samples, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
    if (mip) sourceTarget.textures.forEach(texture => { texture.mipmaps = [{}, {}]; });
    attachment = mrt ? 1 : 0; source = sourceTarget.textures[attachment];
    const material = new T.NodeMaterial(); material.depthTest = material.depthWrite = false; material.blending = T.NoBlending;
    const color = N.vec4(N.uv().x.mul(.5).add(.125), N.uv().y.mul(.5).add(.125), phase, 1);
    material.fragmentNode = mrt ? N.outputStruct(N.vec4(phase, .125, .75, 1), color) : color;
    sourceScene = new T.QuadMesh(material); sourceCamera = sourceScene.camera;
  } else {
    source = array ? new T.StorageArrayTexture(sourceWidth, sourceWidth, 4)
      : volume ? new T.Storage3DTexture(sourceWidth, sourceWidth, 4) : new T.StorageTexture(sourceWidth, sourceWidth);
    source.generateMipmaps = mip; source.mipmapsAutoUpdate = false;
    source.minFilter = source.magFilter = T.NearestFilter;
    const layers = array ? 4 : volume ? 4 >> level : 1;
    producer = N.Fn(() => {
      const x = N.instanceIndex.mod(size), y = N.instanceIndex.div(size).mod(size), z = N.instanceIndex.div(size * size);
      let output = N.storageTexture(source).toWriteOnly().setMipLevel(level);
      if (array) output = output.depth(z);
      const coordinates = volume ? N.uvec3(x, y, z) : N.uvec2(x, y);
      N.textureStore(output, coordinates, N.vec4(x.toFloat().mul(.125).add(.125),
        y.toFloat().mul(.125).add(z.toFloat().mul(.125)).add(.125), phase, 1));
    })().compute(size * size * layers);
  }
  const render = renderer.render.bind(renderer), compile = renderer.compileAsync.bind(renderer);
  const selectProducer = () => renderer.setRenderTarget(sourceTarget, 0, level);
  const produce = () => {
    if (!target) { renderer.compute(producer); return; }
    const previous = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), oldMip = renderer.getActiveMipmapLevel();
    selectProducer(); render(sourceScene, sourceCamera); renderer.setRenderTarget(previous, face, oldMip);
  };
  renderer.compileAsync = async (...args) => {
    if (target) {
      const previous = renderer.getRenderTarget(); selectProducer(); await compile(sourceScene, sourceCamera); renderer.setRenderTarget(previous);
    } else await renderer.compileComputeAsync(producer);
    await renderer.computeAsync(consume); return compile(...args);
  };
  const region = array || volume ? new T.Box3(new T.Vector3(0, 0, 1), new T.Vector3(size, size, 2))
    : new T.Box2(new T.Vector2(), new T.Vector2(size, size));
  const copy = (x, y) => renderer.copyTextureToTexture(source, destination, region, new T.Vector2(x, y), level);
  let frame = 0;
  renderer.render = (...args) => {
    if (args[0] !== scene) return render(...args);
    frame++;
    if ([2, 7, 8, 10, 12, 13].includes(frame)) {
      if (frame === 10) {
        pixels.fill(90); for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255;
        destination.needsUpdate = true;
      }
      if (frame === 12) { source.dispose(); destination.dispose(); if (bundled) group.needsUpdate = true; }
      phase.value = frame === 7 ? .25 : frame % 3 * .25 + .125; produce(); copy(frame === 8 ? 4 : 0, frame < 10 ? 0 : 4);
      if (frame === 7) { phase.value = .75; produce(); copy(4, 0); }
    }
    renderer.compute(consume); return render(...args);
  };
  return {scene, camera};
}

export const storageReadScenarios = {
  native_storage_read_mips(T, renderer) { return storageReads(T, renderer, false, false); },
  native_storage_read_array_mips(T, renderer) { return storageReads(T, renderer, true, false); },
  native_storage_read_uint_mips(T, renderer) { return storageReads(T, renderer, false, true); },
  native_sampled_copy_data(T, renderer) { return sampledCopies(T, renderer, false, false); },
  native_sampled_copy_image(T, renderer) { return sampledCopies(T, renderer, true, false); },
  native_sampled_copy_aliases(T, renderer) { return sampledCopies(T, renderer, false, true, false, true); },
  native_sampled_copy_bundle(T, renderer) { return sampledCopies(T, renderer, false, false, true); },
  native_sampled_copy_history(T, renderer) { return sampledCopies(T, renderer, false, false, false, true); },
  native_sampled_copy_authored_mips(T, renderer) { return sampledMipCopies(T, renderer, false); },
  native_sampled_copy_generated_mips(T, renderer) { return sampledMipCopies(T, renderer, true); },
  native_sampled_copy_storage(T, renderer) { return producedCopies(T, renderer); },
  native_sampled_copy_storage_array_mip(T, renderer) { return producedCopies(T, renderer, {array: true, mip: true}); },
  native_sampled_copy_storage_volume_mip(T, renderer) { return producedCopies(T, renderer, {volume: true, mip: true}); },
  native_sampled_copy_target(T, renderer) { return producedCopies(T, renderer, {target: true, samples: 4}); },
  native_sampled_copy_target_mip(T, renderer) { return producedCopies(T, renderer, {target: true, mip: true}); },
  native_sampled_copy_target_mrt_bundle(T, renderer) { return producedCopies(T, renderer, {target: true, mrt: true, bundled: true}); },
};
