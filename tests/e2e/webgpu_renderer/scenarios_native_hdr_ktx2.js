const rawBytes = data => new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
function rotateBlocks(texture) {
  for (const mip of texture.mipmaps) {
    const bytes = rawBytes(mip.data);
    if (bytes.length <= 16) continue;
    const first = bytes.slice(0, 16); bytes.copyWithin(0, 16); bytes.set(first, bytes.length - 16);
  }
  texture.needsUpdate = true;
}

export const hdrKtx2Scenarios = {
  native_hdr_ktx2(T, renderer) {
    const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0x172334);
    const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20); camera.position.z = 5;
    const material = new T.MeshBasicNodeMaterial(), gain = N.uniform(1);
    scene.add(new T.Mesh(new T.PlaneGeometry(2.5, 2.5), material));
    let texture, frame = 0, firstPixels, changedPixels;
    const compile = renderer.compileAsync.bind(renderer);
    renderer.compileAsync = async (...args) => {
      if (args[0] === scene && !texture) {
        if (!renderer.hasFeature('texture-compression-bc')) throw Error('HDR KTX2 browser case requires enabled BC compression');
        const {KTX2Loader} = await import('/upstream/three.js/examples/jsm/loaders/KTX2Loader.js');
        const loader = new KTX2Loader().setTranscoderPath('/upstream/three.js/examples/jsm/libs/basis/')
          .setWorkerLimit(1).detectSupport(renderer);
        try { texture = await loader.loadAsync('/upstream/three.js/examples/textures/ktx2/2d_uastc_hdr4x4.ktx2'); }
        finally { loader.dispose(); }
        if (texture.format !== T.RGB_BPTC_UNSIGNED_Format || texture.type !== T.HalfFloatType ||
            texture.mipmaps.length < 2 || !texture.mipmaps.every(mip => mip.data instanceof Uint16Array))
          throw Error('Retained KTX2Loader must actually transcode HDR to Uint16 BC6H authored mips');
        const color = N.texture(texture).rgb;
        // Keep HDR differences visible in the harness's RGBA8/NoToneMapping target.
        material.colorNode = color.div(color.add(1)).mul(gain); material.needsUpdate = true;
      }
      return compile(...args);
    };
    const render = renderer.render.bind(renderer);
    renderer.render = (...args) => {
      if (args[0] === scene && args[1] === camera) {
        frame++;
        if (frame === 8) { gain.value = .65; rotateBlocks(texture); }
        if (frame === 10) {
          for (const mip of texture.mipmaps) {
            const bytes = rawBytes(mip.data), buffer = new ArrayBuffer(bytes.length + 14);
            new Uint8Array(buffer).fill(239); new Uint8Array(buffer, 6, bytes.length).set(bytes);
            mip.data = new Uint16Array(buffer, 6, bytes.length / 2);
          }
          texture.needsUpdate = true;
        }
        if (frame === 12) rotateBlocks(texture);
        if (frame === 13) { gain.value = 1; rotateBlocks(texture); }
      }
      return render(...args);
    };
    const read = renderer.readRenderTargetPixelsAsync.bind(renderer), target = renderer.getRenderTarget();
    renderer.readRenderTargetPixelsAsync = async (...args) => {
      const pixels = await read(...args); if (args[0] !== target) return pixels;
      const colors = new Set();
      for (let i = 0; i < pixels.length; i += 4) colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
      if (colors.size < 4) throw Error('Decoded HDR KTX2 texture must produce visible nonconstant output');
      if (frame === 1) firstPixels = pixels.slice();
      if (frame === 8) {
        if (pixels.every((value, i) => value === firstPixels[i])) throw Error('HDR block and uniform updates must change rendered pixels');
        changedPixels = pixels.slice();
      }
      if (frame === 10 && pixels.some((value, i) => value !== changedPixels[i]))
        throw Error('Changing only a decoded block view byteOffset must preserve rendered pixels');
      if (frame >= 10 && !texture.mipmaps.every(mip => mip.data instanceof Uint16Array && mip.data.byteOffset === 6))
        throw Error('Native uploads must preserve caller-owned HDR mip views');
      return pixels;
    };
    return {scene, camera};
  },
};
