// Authored source attributes, unchanged between the native candidate and r186.
function stage(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x182535);
  const camera = new T.OrthographicCamera(-2.2, 2.2, 1.65, -1.65, .1, 20);
  camera.position.z = 5;
  return {scene, camera};
}

function instances(T, {indexed = true, interleaved = false} = {}) {
  const source = new T.PlaneGeometry(.64, .58);
  const geometry = new T.InstancedBufferGeometry().copy(indexed ? source : source.toNonIndexed());
  const offsets = [], colors = [], records = [];
  for (let i = 0; i < 6; i++) {
    const position = [(i % 3 - 1) * 1.12, i < 3 ? .52 : -.52, .04 * i];
    const color = [.15 + i * .12, .82 - i * .1, .25 + (i % 3) * .2];
    offsets.push(...position); colors.push(...color); records.push(...position, ...color, .8 + i * .08);
  }
  let owner;
  if (interleaved) {
    owner = new T.InstancedInterleavedBuffer(new Float32Array(records), 7, 1);
    geometry.setAttribute('instanceOffset', new T.InterleavedBufferAttribute(owner, 3, 0));
    geometry.setAttribute('instanceTint', new T.InterleavedBufferAttribute(owner, 3, 3));
    geometry.setAttribute('instanceScale', new T.InterleavedBufferAttribute(owner, 1, 6));
  } else {
    geometry.setAttribute('instanceOffset', new T.InstancedBufferAttribute(new Float32Array(offsets), 3));
    // Native r186 uses instance stepMode and ignores the WebGL divisor field.
    owner = new T.InstancedBufferAttribute(new Float32Array(colors), 3, false, 2);
    geometry.setAttribute('instanceTint', owner);
  }
  geometry.instanceCount = 6;
  const material = new T.MeshBasicNodeMaterial(), N = T.TSL;
  material.positionNode = (interleaved ? N.positionLocal.mul(N.attribute('instanceScale', 'float')) : N.positionLocal)
    .add(N.attribute('instanceOffset', 'vec3'));
  material.colorNode = N.attribute('instanceTint', 'vec3').mul(N.uv().y.mul(.3).add(.7));
  const mesh = new T.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.count = 1; // geometry.instanceCount has precedence in the native renderer.
  return {mesh, geometry, material, owner};
}

export const nativeInstancedGeometryScenarios = {
  native_tsl_instanced_geometry_indexed(T) {
    const result = stage(T), active = instances(T), empty = instances(T);
    empty.geometry.instanceCount = 0; empty.mesh.count = 6;
    empty.material.colorNode = T.TSL.color(0xff00ff); empty.mesh.position.z = .5;
    result.scene.add(active.mesh, empty.mesh);
    return result;
  },

  native_tsl_instanced_geometry_interleaved(T, renderer) {
    const result = stage(T), source = instances(T, {indexed: false, interleaved: true});
    result.scene.add(source.mesh);
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && ++frame === 8) {
        const {geometry, owner, mesh} = source;
        geometry.instanceCount = 4; mesh.count = 0;
        // Change two complete instance records with an ordinary source upload.
        owner.array.set([-.45, .38, .1, .1, .4, .95, 1.2], 7);
        owner.array.set([1.05, .75, .2, .85, .25, .7, .75], 14);
        owner.addUpdateRange(7, 14); owner.needsUpdate = true;
        // An unrequested CPU edit must not replace the already uploaded row.
        owner.array[0] = 20;
      }
      return render(...args);
    };
    return result;
  },

  native_tsl_instanced_geometry_targets(T, renderer) {
    const result = stage(T), writer = stage(T), source = instances(T);
    writer.scene.add(source.mesh); source.mesh.count = 0;
    const targets = Array.from({length: 3}, () => new T.RenderTarget(128, 96,
      {depthBuffer: false, generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter}));
    for (const [i, target] of targets.entries()) {
      renderer.initRenderTarget(target);
      const panel = new T.Mesh(new T.PlaneGeometry(1.32, .99), new T.MeshBasicMaterial({map: target.texture}));
      panel.position.x = (i - 1) * 1.43; result.scene.add(panel);
    }
    const render = renderer.render.bind(renderer); let frame = 0;
    renderer.render = (...args) => {
      if (args[0] === result.scene && (++frame === 1 || frame === 8)) {
        if (frame === 8) {
          for (let i = 0; i < 6; i++) source.owner.array.set([.8 - i * .08, .25 + i * .1, .7 - i * .06], i * 3);
          source.owner.needsUpdate = true;
        }
        const previous = renderer.getRenderTarget(), previousMip = renderer.getActiveMipmapLevel();
        try {
          for (const [i, target] of targets.entries()) {
            source.geometry.instanceCount = (frame === 1 ? [2, 5, 0] : [1, 6, 0])[i];
            renderer.setRenderTarget(target); render(writer.scene, writer.camera);
          }
        } finally { renderer.setRenderTarget(previous, 0, previousMip); }
        // The writer was deliberately not precompiled. Cold candidate requests
        // must retain their counts across asynchronous pipeline preparation.
        source.geometry.instanceCount = 3;
      }
      return render(...args);
    };
    return result;
  },
};
