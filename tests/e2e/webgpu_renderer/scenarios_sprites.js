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

// These use the source SpriteNodeMaterial billboard program, including its
// authored position, scale and rotation inputs. The same graph executes in both
// renderers; no candidate-only image or CPU particle replica is involved.
function authoredScene(T, orthographic = false) {
  const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0x182435);
  const sprites = [], controls = [];
  for (let i = 0; i < 6; i++) {
    const position = N.uniform(new T.Vector3(.06 * i, -.03 * i, 0));
    const scale = N.uniform(new T.Vector2(.7 + .04 * i, 1.1 - .06 * i));
    const rotation = N.uniform(i * .3), opacity = N.uniform(.65 + .04 * i);
    const material = new T.SpriteNodeMaterial({sizeAttenuation: i < 3, depthWrite: false});
    material.positionNode = position; material.scaleNode = scale; material.rotationNode = rotation;
    material.colorNode = N.vec3(N.uv().x.mul(.7).add(.2), N.uv().y.mul(.6).add(.25), i % 2 ? .85 : .3);
    material.opacityNode = N.shapeCircle().mul(opacity);
    const sprite = new T.Sprite(material);
    sprite.position.set((i % 3 - 1) * 1.3, i < 3 ? .65 : -.65, -.1 * i);
    sprite.scale.set(i < 3 ? .9 : .18, i < 3 ? 1.1 : .2, 1);
    sprite.center.set(.15 + .12 * i, .8 - .1 * i);
    if (i === 1) {
      const group = new T.Group(); group.matrixAutoUpdate = false;
      group.matrix.set(-1, .25, .1, 0, .12, 1.15, 0, 0, 0, .15, .8, 0, 0, 0, 0, 1);
      group.add(sprite); scene.add(group);
    } else scene.add(sprite);
    sprites.push(sprite); controls.push({position, scale, rotation, opacity});
  }
  const mesh = new T.Mesh(new T.BoxGeometry(.3, .3, .3), new T.MeshBasicMaterial({color: 0xe0a050}));
  mesh.position.set(0, -.15, -.4); scene.add(mesh);
  return {scene, camera: camera(T, orthographic), sprites, controls};
}
export const spriteScenarios = {
  native_tsl_sprites(T) { return authoredScene(T); },
  native_tsl_sprites_orthographic(T) { return authoredScene(T, true); },
  native_tsl_sprites_updates(T, renderer) {
    const result = authoredScene(T), render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) {
        result.camera.position.set(-1, .4, 4.5); result.camera.lookAt(0, 0, 0);
        for (const [i, sprite] of result.sprites.entries()) {
          sprite.center.set(.8 - .1 * i, .2 + .1 * i);
          result.controls[i].position.value.set(-.06 * i, .04 * i, 0);
          result.controls[i].scale.value.set(.9, .65 + .08 * i);
          result.controls[i].rotation.value = -.5 + .2 * i;
          result.controls[i].opacity.value = .8;
        }
      }
      return render(...args);
    };
    return result;
  },
  native_tsl_sprite_storage(T, renderer) {
    const N = T.TSL, count = 12, positions = N.instancedArray(count, 'vec3'), colors = N.instancedArray(count, 'vec4');
    const phase = N.uniform(.2);
    const initialize = N.Fn(() => {
      const index = N.instanceIndex.toFloat();
      positions.element(N.instanceIndex).assign(N.vec3(index.mod(4).sub(1.5).mul(.85),
        index.div(4).floor().sub(1).mul(.8), phase.mul(index.mul(.5).sin()).mul(.2)));
      colors.element(N.instanceIndex).assign(N.vec4(index.mod(4).mul(.18).add(.2),
        index.div(4).floor().mul(.2).add(.25), phase.mul(.2).add(.4), 1));
    })().compute(count);
    const material = new T.SpriteNodeMaterial({depthWrite: false});
    material.positionNode = positions.toAttribute(); material.colorNode = colors.element(N.instanceIndex).rgb;
    material.scaleNode = N.vec2(.55, .7); material.rotationNode = phase;
    material.opacityNode = N.shapeCircle().mul(.85);
    const sprite = new T.Sprite(material); sprite.count = count; sprite.frustumCulled = false;
    const scene = new T.Scene(); scene.background = new T.Color(0x182435); scene.add(sprite);
    const compile = renderer.compileAsync.bind(renderer), render = renderer.render.bind(renderer);
    renderer.compileAsync = async (...args) => {
      await renderer.computeAsync(initialize);
      // No upload request: graphics must consume the authoritative GPU values.
      positions.value.array.fill(73); colors.value.array.fill(73);
      return compile(...args);
    };
    let frame = 0;
    renderer.render = (...args) => {
      if (++frame === 8) { phase.value = 1.4; renderer.compute(initialize); }
      if (frame === 10) sprite.count = 5;
      if (frame === 12) sprite.count = 0;
      if (frame === 13) sprite.count = count;
      if (positions.value.array.some(value => value !== 73) || colors.value.array.some(value => value !== 73))
        throw new Error('Native sprite compute unexpectedly changed CPU storage');
      return render(...args);
    };
    return {scene, camera: camera(T)};
  },
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
