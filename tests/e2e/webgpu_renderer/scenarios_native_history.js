// Latch two distinct logical frames after the source's real PassNode swaps.
// Plain texture nodes then keep the comparison stable across GPU warmup without
// invoking the PassNode again or replacing its previous image with a CPU copy.
function historyScene(T, renderer, freezeOnResize, accumulation = false) {
  if (accumulation && renderer.samples !== 4) throw new Error('Temporal accumulation requires source constructor antialias');
  const N = T.TSL, source = new T.Scene(), sourceCamera = new T.PerspectiveCamera(43, 4 / 3, .1, 15);
  sourceCamera.position.set(.25, .4, 4.2); sourceCamera.lookAt(0, 0, 0);
  const phase = N.uniform(0), objects = [];
  for (const [i, geometry] of [new T.SphereGeometry(.62, 32, 22), new T.BoxGeometry(.85, 1.1, .7)].entries()) {
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = N.normalLocal.mul(.2).add(N.vec3(i ? .6 : .15, i ? .2 : .55, .35))
      .mul(phase.mul(.35).add(.8));
    const mesh = new T.Mesh(geometry, material); mesh.position.set(i ? .7 : -.7, i ? -.1 : .12, i ? -.2 : .2);
    mesh.rotation.set(.2, i ? .45 : -.2, i ? -.15 : .1); source.add(mesh); objects.push(mesh);
  }
  const scenePass = N.pass(source, sourceCamera);
  scenePass.setMRT(N.mrt({output: N.output, normal: N.vec4(N.normalView.mul(.5).add(.5), 1)}));
  scenePass.getTexture('normal').type = T.UnsignedByteType;
  const current = scenePass.getTextureNode(), previous = scenePass.getPreviousTextureNode();
  const normal = scenePass.getTextureNode('normal'), previousNormal = scenePass.getPreviousTextureNode('normal');
  if (accumulation) {
    // A previous-frame input from this very pass is legal; its current output
    // remains a distinct attachment. The retained NodeFrame prevents recursive
    // before-updates from scheduling a second render of the same logical frame.
    const feedback = new T.MeshBasicNodeMaterial();
    feedback.colorNode = previous.rgb.mul(.72)
      .add(N.vec3(N.uv().x.mul(.6), phase.mul(.2).add(.1), N.uv().y.mul(.6)).mul(.28));
    const backdrop = new T.Mesh(new T.PlaneGeometry(6, 4.5), feedback);
    backdrop.position.z = -.9; source.add(backdrop);
  }
  const combine = (color, oldColor, normals, oldNormals) => {
    const side = N.step(.5, N.uv().x);
    return N.vec4(N.mix(oldColor.rgb, color.rgb, side).mul(.78)
      .add(N.mix(oldNormals.rgb, normals.rgb, side).mul(.22)), 1);
  };
  const pipeline = new T.RenderPipeline(renderer, combine(current, previous, normal, previousNormal));
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(), render = renderer.render.bind(renderer);
  const stop = freezeOnResize ? 18 : 22; let frame = 0;
  renderer.render = (...args) => {
    if (args[0] !== scene) return render(...args);
    frame++;
    if (frame <= stop) {
      const odd = frame % 2;
      source.background = new T.Color(odd ? 0xa34d25 : 0x286c96);
      phase.value = odd ? .1 : 1.2;
      objects[0].position.y = odd ? -.15 : .35;
      objects[1].rotation.y = odd ? -.45 : .8;
      if (frame === 8) {
        renderer.setPixelRatio(1.5); renderer.setSize(240, 180, false);
        scenePass.setResolutionScale(.5);
      }
      if (frame === 18) {
        renderer.setPixelRatio(1); renderer.setSize(320, 240, false);
        scenePass.setResolutionScale(.75);
      }
    }
    const result = pipeline.render();
    if (frame === stop) {
      pipeline.outputNode = combine(N.texture(scenePass.getTexture('output')),
        N.texture(scenePass.getPreviousTexture('output')), N.texture(scenePass.getTexture('normal')),
        N.texture(scenePass.getPreviousTexture('normal')));
      pipeline.needsUpdate = true;
    }
    return result;
  };
  return {scene, camera, source, sourceCamera, scenePass, pipeline, objects, phase};
}

export const nativeHistoryScenarios = {
  native_pipeline_history(T, renderer) { return historyScene(T, renderer, false); },
  native_pipeline_history_resize(T, renderer) { return historyScene(T, renderer, true); },
  native_pipeline_history_accumulation(T, renderer) { return historyScene(T, renderer, false, true); },
};
