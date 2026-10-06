import { scenarios } from './scenarios.js';
// Renders one scenario (location.hash) in the page's renderer, then publishes
// a pixel grid sampled through a 2D canvas in the same animation frame.
export async function run(THREE, createRenderer) {
  const name = location.hash.slice(1);
  const canvas = document.querySelector('canvas');
  const state = (window.__f3d = { name, frames: 0, error: null, pixels: null });
  try {
    const renderer = createRenderer(canvas);
    renderer.setPixelRatio(1); renderer.setSize(320, 240, false);
    // WebGPU apps initialize before PMREM/compute work; WebGLRenderer has no init().
    if (typeof renderer.init === 'function' && !renderer.isWebGLRenderer) await renderer.init();
    const { scene, camera } = scenarios[name](THREE, renderer);
    await renderer.setAnimationLoop(() => {
      try {
        renderer.render(scene, camera); state.frames++;
        // The new backend may defer its first frames to a preparation boundary;
        // sample only after it has presented for a while.
        const presented = renderer.info.f3d ? renderer.info.f3d.presentedRenders : state.frames;
        if (state.frames >= 30 && presented >= 30 && !state.pixels) state.pixels = sample(canvas);
      } catch (e) { state.error ??= String(e?.message ?? e); }
    });
  } catch (e) { state.error = String(e?.message ?? e); }
}
function sample(canvas) {
  const o = document.createElement('canvas'); o.width = canvas.width; o.height = canvas.height;
  const g = o.getContext('2d'); g.drawImage(canvas, 0, 0);
  return Array.from(g.getImageData(0, 0, o.width, o.height).data);
}
