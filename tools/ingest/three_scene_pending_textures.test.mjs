/** Textures whose source image is still loading bind the r186 default
 * (zero, 1x1) texture; their arrival is a preparation boundary. */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {textureDevice} from './fixtures/gpu_texture_device.mjs';
import {createPMREMGeneratorClass} from './three_renderer.mjs';
const T = await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js'), 'build/three.core.js')));
const camera = () => { const c = new T.PerspectiveCamera(60, 1, 0.1, 10); c.position.z = 3; return c; };

test('a loading image texture renders with the default texture until prepare binds it', async () => {
  const d = textureDevice(), scene = new T.Scene();
  // Like a texture whose loader has not delivered data yet: version 0, no pixels.
  const texture = new T.DataTexture(null, 1, 1);
  const mesh = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial({map: texture}));
  scene.add(mesh);
  const b = await createGpuThreeScene(d, scene, {three: T, renderer: {maxDraws: 4}});
  b.render(camera(), {colorView: {}, depthView: {}});
  const placeholder = d.textures.find(t => t.label === 'f3d-default-texture');
  assert.ok(placeholder, 'zero-initialized default texture bound');
  assert.deepEqual([placeholder.size[0], placeholder.size[1]], [1, 1]);
  assert.equal(d.textureWrites.length, 0, 'no bytes are uploaded for the placeholder');
  // The image arrives (TextureLoader sets image and needsUpdate).
  texture.image.data = new Uint8Array([255, 0, 0, 255]);
  texture.needsUpdate = true;
  assert.throws(() => b.render(camera(), {colorView: {}, depthView: {}}), {code: 'THREE_SCENE_PREPARE'});
  b.dispose();
  assert.equal(placeholder.destroyed, true);
});

test('PMREMGenerator substitute describes captures for new-backend renderers only', () => {
  class Upstream { constructor(r) { this.upstream = r; } }
  const PMREMGenerator = createPMREMGeneratorClass(T, {exactBackend: Upstream});
  assert.ok(new PMREMGenerator({isWebGLRenderer: true}) instanceof Upstream);
  const pmrem = new PMREMGenerator({isF3DRenderer: true});
  const room = new T.Scene();
  const target = pmrem.fromScene(room, 0.04, 0.5, 50, {size: 128});
  assert.equal(target.texture.isF3DSceneEnvironment, true);
  assert.equal(target.texture.mapping, T.CubeUVReflectionMapping);
  assert.deepEqual([target.texture.f3dCapture.scene, target.texture.f3dCapture.size, target.texture.f3dCapture.near], [room, 128, 0.5]);
  const hdr = new T.DataTexture(new Float32Array(8 * 4 * 4), 8, 4, T.RGBAFormat, T.FloatType);
  hdr.needsUpdate = true;
  const equirect = pmrem.fromEquirectangular(hdr).texture;
  assert.notEqual(equirect, hdr); assert.equal(equirect.source, hdr.source);
  assert.equal(equirect.mapping, T.EquirectangularReflectionMapping);
  // fromCubemap describes its source; the cube-UV map is generated at preparation.
  const cube = new T.CubeTexture(), fromCube = pmrem.fromCubemap(cube).texture;
  assert.equal(fromCube.mapping, T.CubeUVReflectionMapping);
  assert.equal(fromCube.f3dPMREMSource, cube);
  assert.throws(() => pmrem.fromCubemap(new T.Texture()), {code: 'F3D_RENDERER_SOURCE'});
});
