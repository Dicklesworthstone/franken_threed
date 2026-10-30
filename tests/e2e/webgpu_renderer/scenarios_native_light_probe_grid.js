import {LightProbeGrid} from '../../../upstream/three.js/examples/jsm/lighting/LightProbeGrid.js';

function probeScene(T, renderer, ranged) {
  const N = T.TSL, scene = new T.Scene(); scene.background = new T.Color(0x101820);
  const camera = new T.PerspectiveCamera(48, 4 / 3, .1, 30);
  camera.position.set(4.4, 3.1, 7); camera.lookAt(0, .7, 0);
  const material = new T.MeshStandardNodeMaterial({roughness: .85});
  material.colorNode = N.color(0xb8b0a4);
  const ball = new T.Mesh(new T.SphereGeometry(.85, 16, 12), material);
  ball.position.set(.35, .9, .2); scene.add(ball);
  const floor = new T.Mesh(new T.PlaneGeometry(6, 6), material);
  floor.rotation.x = -Math.PI / 2; scene.add(floor);
  const wallMaterial = new T.MeshStandardNodeMaterial({side: T.DoubleSide, roughness: 1});
  wallMaterial.colorNode = N.color(0x4488bb);
  const wall = new T.Mesh(new T.PlaneGeometry(6, 3), wallMaterial);
  wall.position.set(0, 1.5, -2); scene.add(wall);
  const emission = N.uniform(new T.Color(2.5, .7, .15));
  const emitterMaterial = new T.MeshBasicNodeMaterial({side: T.BackSide});
  emitterMaterial.colorNode = emission.mul(N.positionWorld.normalize().mul(.35).add(.65));
  const emitter = new T.Mesh(new T.BoxGeometry(24, 24, 24), emitterMaterial); scene.add(emitter);
  scene.add(new T.AmbientLight(0xffffff, .05));
  const grid = new LightProbeGrid(4, 3, 4, 2, 2, 2); grid.position.y = 1.6; grid.falloff = 2;
  scene.add(grid);
  const options = {cubemapSize: 8, sampleCount: 64};
  if (ranged) {
    grid.bake(renderer, scene, {...options, start: 0, count: 3});
    grid.bake(renderer, scene, {...options, start: 3, count: 5});
    grid.bake(renderer, scene, {...options, pass: 1, start: 0, count: 5});
    grid.bake(renderer, scene, {...options, pass: 1, start: 5, count: 3});
  } else grid.bake(renderer, scene, {...options, bounces: 1});

  const target = renderer.getRenderTarget(), render = renderer.render.bind(renderer);
  const read = renderer.readRenderTargetPixelsAsync.bind(renderer), dispose = renderer.dispose.bind(renderer);
  let frame = 0, unlit = null;
  renderer.readRenderTargetPixelsAsync = async (...args) => {
    const pixels = await read(...args);
    if (args[0] === target) {
      const colors = new Set();
      for (let i = 0; i < pixels.length; i += 4) colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
      if (grid.intensity > 0 && colors.size < 64)
        throw Error(`Probe-grid output must contain spatially varying illuminated geometry (frame ${frame}, ${colors.size} colors)`);
      if (frame === 12) unlit = pixels.slice();
      if (frame === 13 && unlit && pixels.reduce((sum, value, i) => sum + (Math.abs(value - unlit[i]) > 2), 0) < 128)
        throw Error('Baked probe-grid irradiance must visibly illuminate the scene');
    }
    return pixels;
  };
  renderer.render = (...args) => {
    if (args[0] === scene && args[1] === camera) {
      renderer._nodes.nodeFrame.update(); frame++;
      if (frame === 8) { grid.intensity = .4; grid.falloff = 1.5; }
      if (frame === 10) { grid.position.x = .5; grid.updateBoundingBox(); }
      if (frame === 12) grid.intensity = 0;
      if (frame === 13) grid.intensity = 1;
      if (frame === 20) {
        emission.value.setRGB(.2, 1.5, 3); grid.position.x = 0;
        if (ranged) {
          grid.bake(renderer, scene, {...options, start: 0, count: 4});
          grid.bake(renderer, scene, {...options, start: 4, count: 4});
        } else grid.bake(renderer, scene, options);
      }
    }
    return render(...args);
  };
  renderer.dispose = () => { grid.dispose(); dispose(); };
  return {scene, camera};
}

export const lightProbeGridScenarios = {
  native_probe_grid_bounce(T, renderer) { return probeScene(T, renderer, false); },
  native_probe_grid_ranged(T, renderer) { return probeScene(T, renderer, true); },
};
