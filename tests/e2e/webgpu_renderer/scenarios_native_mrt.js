// Source MRT graphs write named attachments; all draws use the native backend.
function drive(T, renderer, pipeline, update) {
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(), render = renderer.render.bind(renderer);
  let frame = 0;
  renderer.render = (...args) => {
    if (args[0] !== scene) return render(...args);
    update(++frame); return pipeline.render();
  };
  return {scene, camera, pipeline};
}

function attachments(T, renderer, packedNormals = false) {
  const N = T.TSL, source = new T.Scene(), camera = new T.PerspectiveCamera(45, 4 / 3, .2, 14);
  camera.position.set(.2, .3, 4.3); camera.lookAt(0, 0, 0);
  const phase = N.uniform(.2), objects = [];
  for (const [i, geometry] of [new T.SphereGeometry(.62, 32, 24), new T.BoxGeometry(.85, 1.15, .7)].entries()) {
    const material = new T.MeshStandardNodeMaterial({roughness: .3});
    material.colorNode = N.uniform(new T.Color(i ? 0xd97732 : 0x38a9a1).multiplyScalar(1.4));
    material.positionNode = N.positionLocal.add(N.vec3(N.positionLocal.y.mul(3).add(phase).sin().mul(.08), 0, 0));
    const mesh = new T.Mesh(geometry, material); mesh.position.set(i ? .72 : -.68, i ? -.08 : .1, i ? -.25 : .1);
    mesh.rotation.set(.15, i ? .5 : -.25, i ? -.12 : .08); source.add(mesh); objects.push(mesh);
  }
  const glass = new T.MeshStandardNodeMaterial({roughness: .25, transparent: true, opacity: .45, depthWrite: false});
  glass.colorNode = N.uniform(new T.Color(0x8b67cd));
  const foreground = new T.Mesh(new T.PlaneGeometry(.8, 1.45), glass);
  foreground.position.set(.08, .05, .8); foreground.rotation.z = .22; source.add(foreground);
  const light = new T.DirectionalLight(0xffdfbc, 2.5); light.position.set(-2, 3, 4);
  source.add(light, new T.HemisphereLight(0xbdd7ff, 0x493027, .65));

  // Declaration order differs from attachment order; an unused output stays unallocated.
  const normalValue = packedNormals ? N.normalView.mul(.5).add(.5) : N.normalView;
  const mrt = N.mrt({normal: N.vec4(normalValue, 1), output: N.output, unused: N.float(.4)});
  mrt.setClearColor('output', new T.Color(0x192d43), .8);
  mrt.setClearColor('normal', new T.Color(.12, .28, .72), 0);
  const scenePass = N.pass(source, camera); scenePass.setMRT(mrt);
  const normal = scenePass.getTextureNode('normal'), color = scenePass.getTextureNode('output');
  if (packedNormals) scenePass.getTexture('normal').type = T.UnsignedByteType;
  const depth = scenePass.getLinearDepthNode(), gain = N.uniform(.9);
  const normalColor = packedNormals ? normal.rgb : normal.rgb.mul(.5).add(.5);
  const composite = N.vec4(color.rgb.mul(.65).add(normalColor.mul(.28))
    .mul(depth.oneMinus().mul(.35).add(.65)).mul(normal.a.mul(.15).add(.85)).mul(gain), 1);
  return {source, sourceCamera: camera, scenePass, mrt, color, normal, composite, phase, gain, objects, foreground};
}

export const nativeMRTScenarios = {
  native_pipeline_mrt(T, renderer) {
    const N = T.TSL, result = attachments(T, renderer);
    renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = .95;
    const pipeline = new T.RenderPipeline(renderer, result.composite);
    const driver = drive(T, renderer, pipeline, frame => {
      if (frame === 8) {
        pipeline.outputColorTransform = false;
        pipeline.outputNode = N.renderOutput(result.composite); pipeline.needsUpdate = true;
      }
      if (frame === 18) {
        result.phase.value = 1.1; result.gain.value = 1.15; renderer.toneMappingExposure = 1.1;
        result.objects[0].rotation.y = .4; result.foreground.position.x = -.18;
      }
    });
    return {...result, ...driver};
  },
  native_pipeline_mrt_updates(T, renderer) {
    if (renderer.samples !== 4) throw new Error('The MRT update case requires the source antialias constructor');
    const N = T.TSL, result = attachments(T, renderer, true), overrideNormal = N.uniform(new T.Color(.2, .8, .35));
    result.objects[1].material.mrtNode = N.mrt({normal: N.vec4(overrideNormal, 1)});
    const material = new T.MeshBasicNodeMaterial(); material.fragmentNode = result.composite;
    const intermediate = new T.Scene(), camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 2);
    camera.position.z = 1; intermediate.add(new T.Mesh(new T.PlaneGeometry(2, 2), material));
    const colorPass = N.pass(intermediate, camera);
    renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
    const pipeline = new T.RenderPipeline(renderer, colorPass);
    const driver = drive(T, renderer, pipeline, frame => {
      if (frame === 8) {
        renderer.setPixelRatio(1.5); renderer.setSize(240, 180, false);
        result.scenePass.setResolutionScale(.75); colorPass.setResolutionScale(.5);
        result.mrt.setClearColor('normal', new T.Color(.7, .18, .3), .25);
        result.objects[0].position.z = -.3; result.phase.value = .8; result.gain.value = 1.1;
      }
      if (frame === 18) {
        renderer.setPixelRatio(1); renderer.setSize(320, 240, false);
        result.scenePass.setResolutionScale(.75); colorPass.setResolutionScale(1);
        result.mrt.setClearColor('output', new T.Color(0x382149), .6);
        result.mrt.setClearColor('normal', new T.Color(.14, .55, .38), .4);
        overrideNormal.value.setRGB(.8, .2, .65); result.phase.value = 1.3; result.gain.value = .95;
        result.objects[1].rotation.y = -.35; result.foreground.position.x = .3;
      }
    });
    return {...result, ...driver, colorPass, overrideNormal};
  },
};
