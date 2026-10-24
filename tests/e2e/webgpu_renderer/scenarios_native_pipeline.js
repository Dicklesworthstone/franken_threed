// Retained public RenderPipeline/PassNode construction, native GPU execution.
function drive(T, renderer, pipeline, update = () => {}) {
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(), render = renderer.render.bind(renderer);
  let frame = 0;
  renderer.render = (...args) => {
    if (args[0] !== scene) return render(...args);
    update(++frame); return pipeline.render();
  };
  return {scene, camera, pipeline};
}

export const nativePipelineScenarios = {
  native_pipeline_msaa_depth(T, renderer) {
    if (renderer.samples !== 4) throw new Error('The MSAA depth case requires the source antialias constructor');
    // Both source scene passes inherit the renderer's real default4x sample count.
    return nativePipelineScenarios.native_pipeline_passes(T, renderer);
  },
  native_pipeline_output(T, renderer) {
    const N = T.TSL, gain = N.uniform(.8), p = N.uv();
    renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = .9;
    const color = N.vec4(N.vec3(p.x.mul(4).add(.05), p.y.mul(2.5).add(.12),
      p.x.sub(p.y).mul(9).sin().mul(.4).add(.6)).mul(gain), .75);
    const pipeline = new T.RenderPipeline(renderer, color);
    const result = drive(T, renderer, pipeline, frame => {
      if (frame === 8) {
        // Move output conversion into the authored graph, as FXAA/SMAA apps do.
        pipeline.outputColorTransform = false;
        pipeline.outputNode = N.renderOutput(color); pipeline.needsUpdate = true;
      }
      if (frame === 18) { gain.value = 1.25; renderer.toneMappingExposure = 1.1; }
    });
    return {...result, gain};
  },
  native_pipeline_passes(T, renderer) {
    const N = T.TSL, source = new T.Scene(); source.background = new T.Color(0x253c58);
    const camera = new T.PerspectiveCamera(45, 4 / 3, .2, 12); camera.position.set(.2, .4, 4.2); camera.lookAt(0, 0, 0);
    const objects = [];
    for (const [i, geometry] of [new T.SphereGeometry(.62, 32, 20), new T.BoxGeometry(.85, 1.1, .7)].entries()) {
      const material = new T.MeshStandardNodeMaterial({roughness: .35});
      material.colorNode = N.uniform(new T.Color(i ? 0xd8783d : 0x49b1a5));
      const mesh = new T.Mesh(geometry, material); mesh.position.set(i ? .72 : -.68, i ? -.06 : .12, i ? -.3 : .3);
      mesh.rotation.set(.15, i ? .5 : -.2, i ? -.15 : .1); source.add(mesh); objects.push(mesh);
    }
    const light = new T.DirectionalLight(0xffe0c2, 2.5); light.position.set(-2, 3, 4);
    source.add(light, new T.HemisphereLight(0xb9d5ff, 0x3f2e26, .7));
    const scenePass = N.pass(source, camera), color = scenePass.getTextureNode('output');
    const depth = scenePass.getLinearDepthNode(), distance = scenePass.getViewZNode().negate().div(12);
    const gain = N.uniform(.85), material = new T.MeshBasicNodeMaterial();
    material.fragmentNode = N.vec4(color.rgb.mul(depth.oneMinus().mul(.6).add(.4)).mul(gain)
      .add(N.vec3(depth.mul(.12), distance.mul(.08), depth.mul(.03))), 1);
    const intermediate = new T.Scene(), plane = new T.Mesh(new T.PlaneGeometry(2, 2), material);
    const ortho = new T.OrthographicCamera(-1, 1, 1, -1, .1, 2); ortho.position.z = 1;
    intermediate.add(plane);
    const colorPass = N.pass(intermediate, ortho);
    renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
    const pipeline = new T.RenderPipeline(renderer, colorPass);
    const result = drive(T, renderer, pipeline, frame => {
      if (frame === 8) {
        renderer.setPixelRatio(1.5); renderer.setSize(240, 180, false);
        scenePass.setResolutionScale(.75); colorPass.setResolutionScale(.5);
        objects[0].position.z = -.4; objects[1].rotation.y = -.4; gain.value = 1.1;
      }
      if (frame === 18) {
        renderer.setPixelRatio(1); renderer.setSize(320, 240, false);
        scenePass.setResolutionScale(1); colorPass.setResolutionScale(.75);
        objects[0].position.z = .5; objects[1].rotation.y = .9; gain.value = .95;
      }
    });
    return {...result, source, sourceCamera: camera, scenePass, colorPass, objects, gain};
  },
};
