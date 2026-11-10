import {createWebGLInstanceHistory} from './webgl_instance_history.mjs';

/** Callback-free rigid InstancedMesh packets for deferred WebGL scene passes.
 * Source objects stay untouched while preparation awaits. Stable private views
 * reuse the ordinary geometry/program owners; each packet installs only its
 * accepted streams, count, transform and projection selection. */
export function createThreePassInstances(three, {maxBytes = 128 * 1024 * 1024, maxMeshes = 256} = {}) {
  const history = createWebGLInstanceHistory({maxBytes, maxAttributes: maxMeshes * 2});
  const sources = new Map(); let attributes = new WeakMap();
  let pendingBytes = 0;
  const fail = message => { throw Object.assign(new Error(`F3D_RENDERER_UNSUPPORTED: ${message}`), {code: 'F3D_RENDERER_UNSUPPORTED'}); };
  function privateAttribute(source) {
    let attribute = attributes.get(source);
    if (!attribute) {
      attribute = new three.InstancedBufferAttribute(new Float32Array(0), source.itemSize);
      history.markPrivate(attribute); attributes.set(source, attribute);
    }
    return attribute;
  }
  function privateObject(source) {
    let state = sources.get(source);
    if (!state) {
      const view = Object.create(source, Object.fromEntries(Object.entries({matrixWorld: new three.Matrix4(),
        modelViewMatrix: new three.Matrix4(), normalMatrix: new three.Matrix3(),
        _listeners: {}, frustumCulled: false, isInstancedMesh: true, isSkinnedMesh: false, isBatchedMesh: false, morphTexture: null,
        onBeforeRender: three.Object3D.prototype.onBeforeRender, onAfterRender: three.Object3D.prototype.onAfterRender,
        intersectsFrustum: three.Mesh.prototype.intersectsFrustum,
        computeBoundingSphere: three.InstancedMesh.prototype.computeBoundingSphere,
        getMatrixAt: three.InstancedMesh.prototype.getMatrixAt,
      }).map(([key, value]) => [key, {value, writable: true, configurable: true, enumerable: true}])));
      state = {view, attributes: new Set(), dispose: null};
      state.dispose = () => {
        for (const attribute of state.attributes) { history.forget(attribute); attributes.delete(attribute); }
        view.dispatchEvent({type: 'dispose'});
        source.removeEventListener('dispose', state.dispose); sources.delete(source);
      };
      source.addEventListener('dispose', state.dispose); sources.set(source, state);
    }
    return state;
  }
  function capture(root, camera, frame, {maxValues = 1048576, sortObjects = true, required = true} = {}) {
    if (!frame?.instances?.size) return null;
    const refuse = message => { if (required) fail(message); return null; };
    if (sources.size + [...frame.instances].filter(object => !sources.has(object)).length > maxMeshes)
      return refuse('Pass instance identity capacity exceeded');
    const hierarchy = new Map(), visibility = new Map(), layers = new Map(), orders = new Map(), selected = new Set();
    const stack = [{object: root, visible: true}], objects = new Map(), packets = [], streams = new Set();
    const vp = new three.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const frustum = new three.Frustum().setFromProjectionMatrix(vp, three.WebGLCoordinateSystem, camera.reversedDepth);
    let bytes = 0;
    // Validate budgets and projection eligibility before acknowledging a source
    // upload or clearing any update range. Material eligibility was established
    // by captureThreeMaterialState without invoking application accessors.
    while (stack.length) {
      const {object, visible: parentVisible} = stack.pop();
      if (object.isLOD && object.autoUpdate) return refuse('Instanced pass capture requires an explicit LOD selection');
      const visible = parentVisible && object.visible !== false && !frame.excludedObjects?.has(object);
      hierarchy.set(object, object.children.slice()); visibility.set(object, visible);
      layers.set(object, object.layers.mask); orders.set(object, object.renderOrder);
      for (let i = object.children.length - 1; i >= 0; i--) stack.push({object: object.children[i], visible});
      if (!frame.instances.has(object)) continue;
      for (const attribute of [object.instanceMatrix, object.instanceColor]) if (attribute) bytes += attribute.array.byteLength;
      if (visible && object.layers.test(camera.layers) && (!object.frustumCulled || object.intersectsFrustum(frustum))) {
        selected.add(object);
        for (const attribute of [object.instanceMatrix, object.instanceColor]) if (attribute) streams.add(attribute);
        if (sortObjects && object.boundingSphere === null) object.computeBoundingSphere();
      }
    }
    const values = hierarchy.size * 4 + frame.instances.size * 40;
    if (values > maxValues || bytes > maxBytes - pendingBytes) return refuse('Pending instanced pass snapshot capacity exceeded');
    try { history.validate(streams); }
    catch (error) { if (error.code !== 'GEOMETRY_GPU_HISTORY') throw error; return refuse(error.message); }
    const accepted = new Map([...streams].map(attribute => [attribute, history.capture(attribute)]));
    for (const source of frame.instances) {
      const state = privateObject(source), view = state.view, active = selected.has(source), captured = [];
      for (const key of ['instanceMatrix', 'instanceColor']) {
        const attribute = source[key];
        if (!attribute) { captured.push([key, null, null]); continue; }
        state.attributes.add(attribute);
        if (!accepted.has(attribute)) accepted.set(attribute, attribute.array.slice());
        captured.push([key, privateAttribute(attribute), accepted.get(attribute)]);
      }
      objects.set(source, view);
      hierarchy.set(view, hierarchy.get(source)); visibility.set(view, visibility.get(source));
      layers.set(view, layers.get(source)); orders.set(view, orders.get(source));
      frame.objects.set(view, active ? frame.objects.get(source) : null);
      packets.push({state, captured, count: source.count, geometry: source.geometry,
        matrixWorld: new three.Matrix4().copy(source.matrixWorld), boundingSphere: source.boundingSphere
          ? new three.Sphere(new three.Vector3().copy(source.boundingSphere.center), source.boundingSphere.radius) : null});
    }
    pendingBytes += bytes;
    let released = false;
    return {objects, hierarchy, visibility, layers, orders, values, bytes,
      install() {
        if (released) fail('An instanced pass snapshot was released before submission');
        for (const packet of packets) {
          const view = packet.state.view;
          view.count = packet.count; view.geometry = packet.geometry;
          view.matrixWorld.copy(packet.matrixWorld); view.boundingSphere = packet.boundingSphere;
          for (const [key, attribute, array] of packet.captured) {
            view[key] = attribute;
            if (attribute && attribute.array !== array) {
              attribute.array = array; attribute.count = array.length / attribute.itemSize; attribute.needsUpdate = true;
            }
          }
        }
      },
      release() { if (!released) { released = true; pendingBytes -= bytes; } },
    };
  }
  return {history, capture,
    dispose() { for (const [source, state] of sources) source.removeEventListener('dispose', state.dispose); sources.clear(); attributes = new WeakMap(); history.dispose(); },
    get pendingBytes() { return pendingBytes; },
  };
}
