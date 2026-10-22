// Ordinary source billboards, shared unchanged by native reference/candidate.
function camera(T, orthographic = false) {
  const c = orthographic ? new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 100) : new T.PerspectiveCamera(50, 320 / 240, .1, 100);
  c.position.set(1, .6, 5); c.lookAt(0, 0, 0); return c;
}
function spriteTexture(T, alpha = false) {
  const pixels = new Uint8Array(8 * 8 * 4);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    const i = (y * 8 + x) * 4, checker = (x + y) % 2;
    pixels.set(alpha ? [255, checker ? 210 : 100, 0, 255] : [40 + x * 27, 50 + y * 25, checker ? 240 : 80, x === 0 || y === 0 ? 0 : 220], i);
  }
  const texture = new T.DataTexture(pixels, 8, 8);
  if (!alpha) texture.colorSpace = T.SRGBColorSpace;
  texture.wrapS = texture.wrapT = T.RepeatWrapping; texture.needsUpdate = true; return texture;
}
function scene(T, orthographic = false) {
  const scene = new T.Scene(); scene.background = new T.Color(0x182435);
  scene.fog = new T.Fog(0x182435, 4, 11);
  const sprites = [];
  for (let i = 0; i < 6; i++) {
    const C = i === 2 || i === 5 ? T.SpriteNodeMaterial : T.SpriteMaterial;
    const map = spriteTexture(T), alphaMap = spriteTexture(T, true);
    map.repeat.set(1 + (i % 2), 1); map.rotation = i * .12;
    alphaMap.offset.set(.13 * i, .07); alphaMap.rotation = -.2;
    const material = new C({color: i % 2 ? 0xffbb88 : 0x88ddff, map, alphaMap, rotation: i * .27,
      opacity: .8, alphaTest: i === 3 ? .3 : 0, sizeAttenuation: i < 3});
    material.fog = i !== 4;
    const sprite = new T.Sprite(material); sprite.position.set((i % 3 - 1) * 1.4, i < 3 ? .65 : -.65, i === 5 ? -1.5 : 0);
    sprite.scale.set(i < 3 ? .9 : .18, i < 3 ? 1.1 : .2, 1);
    sprite.center.set(.2 + i * .1, .8 - i * .09); sprite.rotation.set(.3, .7, .2);
    if (i === 1) { const group = new T.Group(); group.scale.set(-1, 1.2, .7); group.rotation.set(.25, -.5, .1); group.add(sprite); scene.add(group); }
    else scene.add(sprite);
    if (i === 5) sprite.count = 3;
    sprites.push(sprite);
  }
  return {scene, camera: camera(T, orthographic), sprites};
}
export const spriteScenarios = {
  native_sprites(T) { return scene(T); },
  native_sprites_orthographic(T) { return scene(T, true); },
  native_sprites_updates(T, renderer) {
    const result = scene(T), render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) {
        result.camera.position.set(-1, .3, 4.5); result.camera.lookAt(0, 0, 0);
        for (const [i, sprite] of result.sprites.entries()) {
          sprite.center.set(.8 - i * .08, .2 + i * .09); sprite.material.rotation = -.5 + i * .2;
          sprite.material.map.offset.set(.2, .3); sprite.material.alphaMap.rotation = .5;
          if (i === 4) sprite.visible = false;
        }
      }
      return render(...args);
    };
    return result;
  },
  native_sprites_array_camera(T) {
    const result = scene(T), left = camera(T), right = camera(T);
    left.position.set(-1.2, .4, 5); left.lookAt(0, 0, 0); left.viewport = new T.Vector4(0, 0, 160, 240);
    right.position.set(1.4, -.2, 4.5); right.lookAt(0, 0, 0); right.viewport = new T.Vector4(160, 0, 160, 240);
    left.updateMatrixWorld(); right.updateMatrixWorld();
    const array = new T.ArrayCamera([left, right]); array.position.z = 5;
    result.camera = array; return result;
  },
};
