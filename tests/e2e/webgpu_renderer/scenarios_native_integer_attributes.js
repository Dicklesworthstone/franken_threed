// Exact typed IDs and bit fields: every live word is outside Float32's exact
// integer range. Colors and positions depend on its low bits, so a floating
// upload cannot accidentally match the pinned integer vertex interpretation.
function stage(T) {
  const scene = new T.Scene(); scene.background = new T.Color(0x182535);
  const camera = new T.OrthographicCamera(-2.2, 2.2, 1.65, -1.65, .1, 20);
  camera.position.z = 5;
  return {scene, camera};
}

function updates(renderer, result, node, material, edit) {
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => {
    if (args[0] === result.scene) {
      frame++;
      if ([8, 10, 12, 13].includes(frame)) {
        const owner = node.attribute.data;
        edit(owner.array, frame);
        if (frame === 8) owner.addUpdateRange(12, 12);
        if (frame === 12) owner.addUpdateRange(0, 0);
        if (frame === 13) owner.addUpdateRange(0, 12);
        if (frame !== 10) owner.needsUpdate = true;
        if (frame === 13) material.needsUpdate = true;
      }
    }
    return render(...args);
  };
  return result;
}

export const integerAttributeScenarios = {
  native_integer_vertices(T, renderer) {
    const result = stage(T), N = T.TSL, base = -0x70000000;
    const geometry = new T.PlaneGeometry(3.6, 2.2).toNonIndexed();
    const array = new Int32Array(24);
    for (let i = 0; i < 6; i++) array.set((i < 3 ? [59, 113, 229, 128] : [223, 71, 131, 128]).map(x => base + x), i * 4);
    const color = N.bufferAttribute(array, 'ivec3', 4, 0), shift = N.bufferAttribute(array, 'int', 4, 3);
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = N.vec3(color.bitAnd(255)).div(255);
    material.positionNode = N.positionLocal.add(N.vec3(N.float(shift.bitAnd(255)).sub(128).mul(.002), 0, 0));
    const mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; result.scene.add(mesh);
    return updates(renderer, result, color, material, (data, frame) => {
      if (frame === 8) {
        for (let i = 0; i < 6; i++) data.set([base + 107, base + 211, base + 43, base + 160], i * 4);
        // Only the second triangle is uploaded. The first keeps its old GPU
        // colors and position despite those unrequested CPU edits.
      } else if (frame === 10 || frame === 12) data[0] = base + 17;
      else for (let i = 0; i < 3; i++) data.set([base + 83, base + 47, base + 239, base + 96], i * 4);
    });
  },

  native_integer_repeated_vertices(T, renderer) {
    const result = stage(T), N = T.TSL, base = 0xf0000000, array = new Uint32Array(36);
    for (let i = 0; i < 6; i++) array.set([base + 1, base + 47 + i * 29, base + 229 - i * 23,
      base + 59 + (i % 3) * 61, base + 64 + (i % 3) * 64, base + (i < 3 ? 164 : 92)], i * 6);
    // The explicit flag still leaves the raw source owner's step mode as
    // vertex. Repeat the mesh using instanceIndex, so incorrectly promoting
    // this interleaved buffer to per-instance is visibly different.
    const color = N.instancedBufferAttribute(array, 'uvec3', 6, 1).setInstanced(true);
    const location = N.instancedBufferAttribute(array, 'uvec2', 6, 4).setInstanced(true);
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = N.vec3(color.bitAnd(255)).div(255);
    const instanceOffset = N.vec2(N.float(N.instanceIndex.mod(3)).sub(1).mul(1.2), N.select(N.instanceIndex.lessThan(3), .55, -.55));
    material.positionNode = N.positionLocal.add(N.vec3(instanceOffset.add(N.vec2(location.bitAnd(255)).sub(128).mul(.001)), 0));
    const mesh = new T.Mesh(new T.PlaneGeometry(.72, .64), material);
    mesh.count = 6; mesh.frustumCulled = false; result.scene.add(mesh);
    return updates(renderer, result, color, material, (data, frame) => {
      if (frame === 8) {
        for (let i = 0; i < 6; i++) data.set([base + 197 - i * 19, base + 83 + i * 17, base + 113], i * 6 + 1);
        data[16] = base + 180; data[23] = base + 108;
      } else if (frame === 10 || frame === 12) data[1] = base + 11;
      else for (let i = 0; i < 2; i++) data.set([base + 31, base + 157, base + 233], i * 6 + 1);
    });
  },
};
