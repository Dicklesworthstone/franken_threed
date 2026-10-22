// Authored mip rendering uses only source APIs on both native renderers.
function camera(T, bounds = 1) {
  const result = new T.OrthographicCamera(-bounds, bounds, bounds * .75, -bounds * .75, .1, 10);
  result.position.z = 3; return result;
}
function writer(T, level, updated) {
  const scene = new T.Scene();
  scene.background = new T.Color((updated ? [0x193077, 0x773019, 0x297739, 0x775819]
    : [0x773129, 0x19774b, 0x293a77, 0x674177])[level]);
  // Intersecting planes submitted with explicit order exercise private depth.
  const front = new T.Mesh(new T.PlaneGeometry(.9, 1), new T.MeshBasicMaterial({color: updated ? 0x55dddd : 0xeecb44}));
  front.position.set(level % 2 ? .3 : -.3, .1, .3); front.renderOrder = 0;
  const back = new T.Mesh(new T.PlaneGeometry(1.4, .55), new T.MeshBasicMaterial({color: updated ? 0xdd4499 : 0x4488dd}));
  back.position.set(0, -.2, 0); back.renderOrder = 1;
  scene.add(front, back); return scene;
}
function setup(T, renderer, updates) {
  const source = new T.RenderTarget(64, 64, {generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  source.texture.mipmaps = Array.from({length: 4}, () => ({}));
  const outputs = Array.from({length: 4}, (_, level) => new T.RenderTarget(64 >> level, 64 >> level,
    {depthBuffer: false, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
  const copyRegions = outputs.map(target => new T.Box2(new T.Vector2(), new T.Vector2(target.width, target.height)));
  for (const target of outputs) renderer.initRenderTarget(target);
  const writers = [Array.from({length: 4}, (_, level) => writer(T, level, false))];
  if (updates) writers.push(Array.from({length: 4}, (_, level) => writer(T, level, true)));
  const writeCamera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10); writeCamera.position.z = 3;
  const scene = new T.Scene(); scene.background = new T.Color(0x131b25);
  for (const [level, target] of outputs.entries()) {
    const panel = new T.Mesh(new T.PlaneGeometry(1.7, 1.2), new T.MeshBasicMaterial({map: target.texture}));
    panel.position.set(level % 2 ? .95 : -.95, level < 2 ? .7 : -.7, 0); scene.add(panel);
  }
  // Prepare every immutable writer before the harness starts its frame loop.
  const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
  renderer.compileAsync = async (...args) => {
    const previous = renderer.getRenderTarget(), previousMip = renderer.getActiveMipmapLevel();
    try {
      for (const version of writers) for (const [level, sourceScene] of version.entries()) {
        renderer.setRenderTarget(source, 0, level); await compile(sourceScene, writeCamera);
      }
    } finally { renderer.setRenderTarget(previous, 0, previousMip); }
    return compile(...args);
  };
  let frame = 0;
  renderer.render = (...args) => {
    frame++;
    if (frame === 1 || (updates && frame === 8)) {
      const version = writers[frame === 1 ? 0 : 1];
      for (const [level, sourceScene] of version.entries()) {
        renderer.setRenderTarget(source, 0, level); render(sourceScene, writeCamera);
        // The source API has no mip-selector readback; copy into an ordinary
        // target so the final canvas exposes every authored level unchanged.
        // r186's null region defaults to base dimensions even for a mip copy.
        renderer.copyTextureToTexture(source.texture, outputs[level].texture, copyRegions[level], null, level, 0);
      }
      renderer.setRenderTarget(null);
    }
    return render(...args);
  };
  return {scene, camera: camera(T, 2.1)};
}
export const renderMipScenarios = {
  native_render_mips(T, renderer) { return setup(T, renderer, false); },
  native_render_mips_updates(T, renderer) { return setup(T, renderer, true); },
};
