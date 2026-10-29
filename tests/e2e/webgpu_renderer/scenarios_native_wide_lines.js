import {Line2} from '../../../upstream/three.js/examples/jsm/lines/webgpu/Line2.js';
import {LineSegments2} from '../../../upstream/three.js/examples/jsm/lines/webgpu/LineSegments2.js';
import {LineGeometry} from '../../../upstream/three.js/examples/jsm/lines/LineGeometry.js';
import {LineSegmentsGeometry} from '../../../upstream/three.js/examples/jsm/lines/LineSegmentsGeometry.js';

function view(T) {
  const camera = new T.PerspectiveCamera(50, 4 / 3, .1, 30);
  camera.position.set(.2, .15, 5); camera.lookAt(0, 0, 0); return camera;
}
function world(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x182435); return scene;
}
function animate(renderer, lines, update) {
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    update(++frame);
    const result = render(...args);
    // The actual addon callback updates the public state used by raycasting.
    // A native material path must preserve that callback as well as the pixels.
    for (const line of lines) if (line.visible && line._resolution.x === 0)
      throw Error('Source wide-line onBeforeRender did not update raycast resolution');
    return result;
  };
}

export const wideLineScenarios = {
  native_wide_line_screen(T, renderer) {
    const scene = world(T), camera = view(T), lines = [];
    for (let row = 0; row < 2; row++) {
      const geometry = new LineGeometry().setPositions([
        -1.7, -.22, 0, -1.05, .3, .2, -.25, -.15, -.2,
        .4, .3, .1, 1.15, -.23, -.1, 1.7, .2, 0,
      ]);
      geometry.setColors([1, .2, .1, .8, .7, .1, .2, 1, .35,
        .1, .7, 1, .55, .2, 1, 1, .4, .2]);
      const material = new T.Line2NodeMaterial({linewidth: row ? 9 : 4, vertexColors: true});
      const line = new Line2(geometry, material); line.position.y = row ? -.65 : .65;
      line.frustumCulled = false; scene.add(line); lines.push(line);
    }
    animate(renderer, lines, frame => {
      if (frame === 8) {
        lines[0].material.linewidth = 10; lines[1].material.linewidth = 3;
        renderer.setPixelRatio(2);
      }
      if (frame === 10) lines[0].geometry.instanceCount = 2;
      if (frame === 12) lines[1].visible = false;
      if (frame === 13) {
        lines[0].geometry.instanceCount = 5; lines[1].visible = true;
        lines[1].material.color.setHex(0x40ffff); renderer.setPixelRatio(1);
        camera.position.set(-.4, .2, 4.5); camera.lookAt(0, 0, 0);
      }
    });
    return {scene, camera};
  },
  native_wide_line_world_dashes(T, renderer) {
    const scene = world(T), camera = view(T), lines = [];
    for (let row = 0; row < 2; row++) {
      const geometry = new LineSegmentsGeometry().setPositions([
        -1.6, -.2, .25, -.6, .25, -.3, -.25, -.3, .1,
        .25, .3, -.2, .6, -.25, -.3, 1.6, .2, .3,
      ]);
      geometry.setColors([1, .25, .1, .9, .8, .2, .2, .9, .4,
        .1, .55, .9, .65, .3, 1, 1, .4, .5]);
      const material = new T.Line2NodeMaterial({linewidth: .15 + row * .08, worldUnits: true,
        dashed: row === 1, dashSize: .18, gapSize: .12, vertexColors: true});
      const line = new LineSegments2(geometry, material); line.computeLineDistances();
      line.position.y = row ? -.65 : .65; line.frustumCulled = false;
      scene.add(line); lines.push(line);
    }
    animate(renderer, lines, frame => {
      if (frame === 8) {
        lines[0].material.linewidth = .3; lines[1].material.linewidth = .13;
        lines[1].material.dashOffset = .07; lines[1].material.scale = 1.6;
      }
      if (frame === 10) lines[0].material.dashed = true;
      if (frame === 12) lines[1].geometry.instanceCount = 1;
      if (frame === 13) {
        lines[0].material.dashed = false; lines[1].geometry.instanceCount = 3;
        lines[1].material.dashSize = .35; lines[1].material.gapSize = .08;
        camera.position.set(.7, .3, 4.4); camera.lookAt(0, 0, 0);
      }
    });
    return {scene, camera};
  },
  native_wide_line_transparent(T, renderer) {
    const N = T.TSL, scene = world(T), camera = view(T);
    for (let i = 0; i < 2; i++) {
      const material = new T.MeshBasicNodeMaterial();
      material.colorNode = N.vec3(i ? .65 : .1, .2, i ? .15 : .6);
      const plane = new T.Mesh(new T.PlaneGeometry(1.75, 1.7), material);
      plane.position.set(i ? .88 : -.88, 0, -.4); scene.add(plane);
    }
    const geometry = new LineGeometry().setPositions([
      -1.6, -.6, 0, -1, .6, .2, -.35, -.5, .1,
      .4, .55, 0, 1.1, -.6, .2, 1.6, .5, 0,
    ]);
    const dash = N.uniform(.35), offset = N.uniform(.03), opacity = N.uniform(.6);
    const material = new T.Line2NodeMaterial({linewidth: 12, dashed: true, transparent: true});
    material.colorNode = N.vec3(.2, .9, .45); material.opacityNode = opacity;
    material.dashSizeNode = dash; material.gapSizeNode = N.float(.1); material.offsetNode = offset;
    const line = new Line2(geometry, material); line.computeLineDistances();
    line.frustumCulled = false; scene.add(line);
    animate(renderer, [line], frame => {
      if (frame === 8) { dash.value = .15; offset.value = .08; opacity.value = .85; }
      if (frame === 10) material.linewidth = 6;
      if (frame === 12) geometry.instanceCount = 2;
      if (frame === 13) { geometry.instanceCount = 5; opacity.value = .35; material.linewidth = 15; }
    });
    return {scene, camera};
  },
};
