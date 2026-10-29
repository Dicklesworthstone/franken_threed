// The same source scene, storage, controls and nodes run on both renderers.
function camera(T, orthographic = false) {
  const result = orthographic ? new T.OrthographicCamera(-2.4, 2.4, 1.8, -1.8, .1, 50)
    : new T.PerspectiveCamera(50, 4 / 3, .1, 50);
  result.position.set(.35, .2, 5); result.lookAt(0, 0, 0); return result;
}
function scene(T) {
  const result = new T.Scene(); result.background = new T.Color(0x182435);
  result.fog = new T.Fog(0x182435, 3, 9); return result;
}
function installChanges(renderer, change) {
  const render = renderer.render.bind(renderer); let frame = 0;
  renderer.render = (...args) => { change(++frame); return render(...args); };
}
function indirectVertexAlias(T, renderer, bundled) {
  const N = T.TSL, world = new T.Scene(), geometry = new T.PlaneGeometry(2, 2);
  world.background = new T.Color(0x182435);
  const attribute = new T.StorageBufferAttribute(new Uint32Array([3, 1, 0, 0]), 1);
  const argumentsBuffer = new T.IndirectStorageBufferAttribute(new Uint32Array([3, 1, 0, 0, 0]), 1);
  const data = N.storage(attribute, 'uint', attribute.count);
  const args = N.storage(argumentsBuffer, 'uint', argumentsBuffer.count);
  const publish = N.Fn(() => { data.element(0).addAssign(0); args.element(0).addAssign(0); })().compute(1);
  geometry.setIndirect(argumentsBuffer, 0);
  const material = new T.MeshBasicMaterial({color: 0x58c7fa}), object = new T.Mesh(geometry, material);
  object.frustumCulled = false;
  // Source SHARED refresh and retained bundles keep this compute/vertex alias
  // bound while preceding dispatches publish newer host versions. Indirect
  // arguments are separate because the pinned source cannot combine usages.
  world.fogNode = N.fog(N.vec3(data.toAttribute().toFloat().div(6)), N.float(.5));
  const group = bundled ? new T.BundleGroup() : world;
  if (bundled) world.add(group);
  group.add(object);
  const compile = renderer.compileAsync.bind(renderer);
  renderer.compileAsync = async (...args) => {
    await renderer.computeAsync(publish);
    attribute.array.fill(73); argumentsBuffer.array.fill(73);
    return compile(...args);
  };
  installChanges(renderer, frame => {
    const count = frame === 8 ? 6 : frame === 12 ? 0 : frame === 13 ? 3 : null;
    if (count !== null) {
      attribute.array.set([count, 1, 0, 0]); attribute.needsUpdate = true;
      argumentsBuffer.array.set([count, 1, 0, 0, 0]); argumentsBuffer.needsUpdate = true;
      renderer.compute(publish);
      attribute.array.fill(73); argumentsBuffer.array.fill(73);
    }
    if (attribute.array.some(value => value !== 73) || argumentsBuffer.array.some(value => value !== 73))
      throw new Error('Indirect vertex alias changed compute-owned CPU arguments');
  });
  return {scene: world, camera: camera(T)};
}
function pointSprites(T, renderer, orthographic) {
  const N = T.TSL, world = scene(T), objects = [], sizes = [], rotations = [];
  renderer.setPixelRatio(1.5);
  for (let i = 0; i < 6; i++) {
    const size = N.uniform(new T.Vector2(i < 3 ? 2.5 : 13, i < 3 ? 3.5 : 19));
    const rotation = N.uniform(i * .27);
    const material = new T.PointsNodeMaterial({sizeAttenuation: i < 3, transparent: true, depthWrite: false});
    material.positionNode = N.vec3(N.instanceIndex.toFloat().mul(.55).sub(.275), 0, 0);
    material.sizeNode = size; material.rotationNode = rotation; material.scaleNode = N.vec2(.8, 1.1);
    material.colorNode = N.vec3(N.uv().x.mul(.7).add(.2), N.uv().y.mul(.6).add(.25), i % 2 ? .85 : .3);
    material.opacityNode = N.shapeCircle().mul(.8);
    const object = i === 2 || i === 5 ? new T.Mesh(new T.PlaneGeometry(), material) : new T.Sprite(material);
    object.position.set((i % 3 - 1) * 1.2, i < 3 ? .6 : -.6, -.1 * i);
    object.count = 2; object.frustumCulled = false; if (i === 1) object.scale.set(-2, 1.5, 1);
    world.add(object); objects.push(object); sizes.push(size); rotations.push(rotation);
  }
  const view = camera(T, orthographic);
  installChanges(renderer, frame => {
    if (frame === 8) {
      renderer.setPixelRatio(2);
      view.position.set(-.5, .3, 4.5); view.lookAt(0, 0, 0);
      sizes.forEach((size, i) => size.value.multiplyScalar(i % 2 ? .75 : 1.25));
      rotations.forEach((rotation, i) => { rotation.value = -.3 + i * .2; });
    }
    if (frame === 10) objects[0].count = 1;
    if (frame === 12) objects[0].count = 0;
    if (frame === 13) objects[0].count = 2;
  });
  return {scene: world, camera: view};
}
export const nativePrimitiveScenarios = {
  native_indirect_vertex_alias(T, renderer) { return indirectVertexAlias(T, renderer, false); },
  native_indirect_vertex_alias_bundle(T, renderer) { return indirectVertexAlias(T, renderer, true); },
  native_indirect_compute(T, renderer) {
    const N = T.TSL, world = new T.Scene(), vertices = N.uniform(3, 'uint'), objects = [], buffers = [], kernels = [];
    world.background = new T.Color(0x182435);
    for (let i = 0; i < 3; i++) {
      const source = new T.PlaneGeometry(.65, .65);
      let geometry = i === 1 ? source.toNonIndexed() : source;
      if (i === 2) {
        geometry = new T.InstancedBufferGeometry();
        geometry.attributes = source.attributes; geometry.setIndex(source.index);
      }
      const words = geometry.index ? 5 : 4, data = new Uint32Array(words * 2);
      data.set(words === 5 ? [0, 1, 0, 0, 0, 3, 1, 3, 0, 0] : [0, 1, 0, 0, 3, 1, 3, 0]);
      const buffer = new T.IndirectStorageBufferAttribute(data, 1), storage = N.storage(buffer, 'uint', data.length);
      const kernel = N.Fn(() => { storage.element(0).assign(vertices); storage.element(1).assign(i === 2 ? 2 : 1); })().compute(1);
      const material = new T.MeshBasicNodeMaterial({color: [0xff6945, 0x58c7fa, 0xa7e74b][i]});
      if (i === 2) material.positionNode = N.positionLocal.add(N.vec3(N.instanceIndex.toFloat().mul(.75).sub(.375), 0, 0));
      const object = new T.Mesh(geometry, material); object.frustumCulled = false;
      object.position.set(i === 0 ? -.85 : i === 1 ? .85 : 0, i === 2 ? -.65 : .55, 0);
      geometry.setIndirect(buffer, 0); world.add(object);
      objects.push(object); buffers.push(buffer); kernels.push(kernel);
    }
    const compile = renderer.compileAsync.bind(renderer);
    renderer.compileAsync = async (...args) => {
      await renderer.computeAsync(kernels); buffers.forEach(buffer => buffer.array.fill(73));
      return compile(...args);
    };
    installChanges(renderer, frame => {
      if (frame === 8) { vertices.value = 6; renderer.compute(kernels); }
      if (frame === 10) {
        objects[0].geometry.indirectOffset = [20, 0]; objects[1].geometry.indirectOffset = [16];
        objects[2].geometry.setDrawRange(0, 0); // Only GPU arguments choose this draw's extent.
      }
      if (frame === 12) { objects[0].count = 0; objects[1].geometry.indirectOffset = []; }
      if (frame === 13) {
        objects[0].count = 1; objects[1].geometry.indirectOffset = 0;
        vertices.value = 3; renderer.compute(kernels);
      }
      if (buffers.some(buffer => buffer.array.some(value => value !== 73)))
        throw new Error('Indirect drawing changed compute-owned CPU arguments');
    });
    return {scene: world, camera: camera(T)};
  },
  native_indirect_host_upload(T, renderer) {
    const world = new T.Scene(); world.background = new T.Color(0x182435);
    const geometry = new T.PlaneGeometry(2, 2), data = new Uint32Array([3, 1, 0, 0, 0]);
    const argumentsBuffer = new T.BufferAttribute(data, 1);
    geometry.setIndirect(argumentsBuffer);
    const material = new T.MeshBasicMaterial({color: 0x58c7fa}), object = new T.Mesh(geometry, material);
    object.frustumCulled = false; world.add(object);
    installChanges(renderer, frame => {
      if (frame === 8) { data[0] = 6; argumentsBuffer.needsUpdate = true; }
      if (frame === 10) material.color.setHex(0xa7e74b); // Source observer now chooses FULL refresh.
      if (frame === 12) { data[0] = 0; argumentsBuffer.needsUpdate = true; material.color.setHex(0xff6945); }
      if (frame === 13) { data[0] = 3; argumentsBuffer.needsUpdate = true; material.color.setHex(0x58c7fa); }
    });
    return {scene: world, camera: camera(T)};
  },
  native_tsl_lines(T, renderer) {
    const N = T.TSL, world = scene(T), shift = N.uniform(.1), dash = N.uniform(.23), objects = [];
    for (let i = 0; i < 4; i++) {
      const geometry = new T.BufferGeometry().setAttribute('position', new T.Float32BufferAttribute([
        -1.7, -.2, 0, -1.2, .25, 0, -.55, -.15, .1, .05, .2, 0, .65, -.2, -.1, 1.25, .25, 0, 1.75, -.1, 0,
      ], 3));
      if (i === 1) geometry.setIndex(new T.BufferAttribute(new Uint16Array([0, 2, 1, 3, 5, 4, 6]), 1));
      if (i === 2) geometry.setIndex(new T.BufferAttribute(new Uint32Array([0, 1, 3, 2, 4, 6, 5]), 1));
      const material = i === 3 ? new T.LineDashedNodeMaterial({dashSize: .23, gapSize: .12})
        : i === 1 ? new T.LineBasicMaterial() : new T.LineBasicNodeMaterial();
      material.positionNode = N.positionLocal.add(N.vec3(0, shift.mul(N.positionLocal.x.sin()), 0));
      material.colorNode = N.vec3(.25 + i * .17, .85 - i * .15, .3 + i * .14);
      if (i === 3) material.dashSizeNode = dash;
      const object = i === 3 ? new T.LineSegments(geometry, material) : new T.Line(geometry, material);
      if (i === 3) object.computeLineDistances();
      object.position.y = .95 - i * .62; object.frustumCulled = false;
      world.add(object); objects.push(object);
    }
    installChanges(renderer, frame => {
      if (frame === 8) { shift.value = .35; dash.value = .1; objects[1].geometry.setDrawRange(1, 5); }
      if (frame === 10) objects[2].visible = false;
      if (frame === 12) objects[0].geometry.setDrawRange(0, 0);
      if (frame === 13) { objects[0].geometry.setDrawRange(0, 7); objects[2].visible = true; }
    });
    return {scene: world, camera: camera(T)};
  },
  native_tsl_point_storage(T, renderer) {
    const N = T.TSL, count = 80, positions = N.instancedArray(count, 'vec3'), colors = N.instancedArray(count, 'vec4');
    const phase = N.uniform(.2), world = scene(T);
    const kernel = N.Fn(() => {
      const index = N.instanceIndex.toFloat();
      positions.element(N.instanceIndex).assign(N.vec3(index.mod(10).sub(4.5).mul(.34),
        index.div(10).floor().sub(3.5).mul(.3), phase.mul(index.mul(.5).sin()).mul(.2)));
      colors.element(N.instanceIndex).assign(N.vec4(index.mod(10).mul(.06).add(.3),
        index.div(10).floor().mul(.08).add(.25), phase.mul(.2).add(.5), 1));
    })().compute(count);
    const geometry = new T.BufferGeometry().setAttribute('position', new T.Float32BufferAttribute(new Float32Array(count * 3), 3));
    const material = new T.PointsNodeMaterial(); material.positionNode = positions.element(N.vertexIndex);
    material.colorNode = N.varying(colors.element(N.vertexIndex).rgb); material.sizeNode = N.float(19);
    const points = new T.Points(geometry, material); points.frustumCulled = false; world.add(points);
    const compile = renderer.compileAsync.bind(renderer);
    renderer.compileAsync = async (...args) => {
      await renderer.computeAsync(kernel); positions.value.array.fill(73); colors.value.array.fill(73);
      return compile(...args);
    };
    installChanges(renderer, frame => {
      if (frame === 8) { phase.value = 1.4; renderer.compute(kernel); }
      if (frame === 10) geometry.setDrawRange(10, 25);
      if (frame === 12) geometry.setDrawRange(0, 0);
      if (frame === 13) geometry.setDrawRange(0, count);
      if (positions.value.array.some(value => value !== 73) || colors.value.array.some(value => value !== 73))
        throw new Error('Point rendering changed compute-owned CPU storage');
    });
    return {scene: world, camera: camera(T)};
  },
  native_tsl_point_sprites(T, renderer) { return pointSprites(T, renderer, false); },
  native_tsl_point_sprites_orthographic(T, renderer) { return pointSprites(T, renderer, true); },
};
