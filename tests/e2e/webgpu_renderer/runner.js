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
    const { scene, camera } = scenarios[name](THREE, renderer);
    await renderer.setAnimationLoop(() => {
      try {
        renderer.render(scene, camera); state.frames++;
        if (state.frames === 30) state.pixels = sample(canvas);
      } catch (e) { state.error ??= String(e?.message ?? e); }
    });
  } catch (e) { state.error = String(e?.message ?? e); }
}
function sample(canvas) {
  const o = document.createElement('canvas'); o.width = canvas.width; o.height = canvas.height;
  const g = o.getContext('2d'); g.drawImage(canvas, 0, 0);
  return Array.from(g.getImageData(0, 0, o.width, o.height).data);
}
