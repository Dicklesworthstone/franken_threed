import {createWebGLInstanceHistory} from './webgl_instance_history.mjs';
import {createThreePassGeometry} from './three_pass_geometry.mjs';
import {createThreePassSkeletons} from './three_pass_deformation.mjs';

const applyLodUpdate = Reflect.apply;

/** Callback-free object packets for deferred WebGL scene passes.
 * Source objects stay untouched while preparation awaits. Stable private views
 * reuse the ordinary geometry/program owners; each packet installs only its
 * accepted geometry selector, transform and projection selection. Ordinary
 * geometry and instance streams share their source upload history with live
 * draws, preserving needsUpdate, partial ranges and disposal generations. */
export function createThreePassInstances(three, {maxBytes = 128 * 1024 * 1024, maxMeshes = 256} = {}) {
  const history = createWebGLInstanceHistory({maxBytes, maxAttributes: maxMeshes * 16, geometry: true});
  const geometry = createThreePassGeometry(three, history, {maxGeometries: maxMeshes * 4, maxBytes});
  const skeletons = createThreePassSkeletons(three);
  history.geometryView = geometry.live;
  history.wireframeGeometry = geometry.wireframeGeometry;
  const sources = new Map(); let attributes = new WeakMap(), rigidSources = new WeakMap();
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
  function privateObject(source, instanced, skinned) {
    const owners = instanced ? sources : rigidSources;
    let state = owners.get(source);
    if (!state) {
      const view = Object.create(source, Object.fromEntries(Object.entries({matrixWorld: new three.Matrix4(),
        modelViewMatrix: new three.Matrix4(), normalMatrix: new three.Matrix3(),
        _listeners: {}, frustumCulled: false, isInstancedMesh: instanced, isSkinnedMesh: skinned, isBatchedMesh: false, morphTexture: null,
        onBeforeRender: three.Object3D.prototype.onBeforeRender, onAfterRender: three.Object3D.prototype.onAfterRender,
        ...(skinned ? {bindMatrix: new three.Matrix4(), bindMatrixInverse: new three.Matrix4()} : {}),
        ...(instanced ? {intersectsFrustum: three.Mesh.prototype.intersectsFrustum,
          computeBoundingSphere: three.InstancedMesh.prototype.computeBoundingSphere,
          getMatrixAt: three.InstancedMesh.prototype.getMatrixAt} : {}),
      }).map(([key, value]) => [key, {value, writable: true, configurable: true, enumerable: true}])));
      state = {view, attributes: new Set(), dispose: null};
      if (instanced) {
        state.dispose = () => {
          for (const attribute of state.attributes) { history.forget(attribute); attributes.delete(attribute); }
          view.dispatchEvent({type: 'dispose'});
          source.removeEventListener('dispose', state.dispose); sources.delete(source);
        };
        source.addEventListener('dispose', state.dispose);
      }
      // Ordinary Mesh/Line/Points have no disposal contract of their own. Weak
      // identities avoid retaining every object ever drawn by this renderer.
      owners.set(source, state);
    }
    return state;
  }
  function capture(root, camera, frame, {maxValues = 1048576, sortObjects = true, required = true} = {}) {
    if (!frame?.objects?.size) return null;
    const candidates = new Map(), deformationObjects = new Set(), skeletonSources = new Set();
    for (const source of frame.objects.keys()) {
      const instanced = frame.instances.has(source);
      const morph = Object.values(source.geometry?.morphAttributes ?? {}).some(values => values.length);
      const skinned = source.isSkinnedMesh === true;
      if (instanced || (source.isMesh || source.isLine || source.isPoints) && (frame.allowDeformations || !morph && !skinned)) {
        candidates.set(source, {instanced, skinned, morph, geometry: source.geometry});
        if (morph || skinned) deformationObjects.add(source);
        if (skinned) skeletonSources.add(source.skeleton);
      }
    }
    if (!candidates.size) return null;
    const refuse = message => { if (required) fail(message); return null; };
    if (frame.lods?.size && !frame.cameraSafe()) return refuse('Automatic LOD source state changed before pass capture');
    // Attribute uploads precede Skeleton.update in WebGLObjects. Shared palette
    // storage would couple otherwise independent captured inputs to that order.
    const paletteBuffers = new Map(), paletteTextures = new Set();
    for (const skeleton of skeletonSources) {
      const buffer = skeleton.boneMatrices.buffer;
      if (paletteBuffers.has(buffer) && paletteBuffers.get(buffer) !== skeleton)
        return refuse('Queued skeletons require independent palettes or one shared Skeleton');
      paletteBuffers.set(buffer, skeleton);
      if (skeleton.boneTexture) paletteTextures.add(skeleton.boneTexture);
    }
    if (paletteBuffers.size) {
      for (const [object, candidate] of candidates) {
        const g = candidate.geometry;
        for (const attribute of [...Object.values(g.attributes), g.index, ...Object.values(g.morphAttributes).flat(),
          object.instanceMatrix, object.instanceColor]) if (attribute) {
          const owner = attribute.isInterleavedBufferAttribute ? attribute.data : attribute;
          if (paletteBuffers.has(owner.array?.buffer)) return refuse('Queued skeleton palettes cannot also serve as geometry or morph inputs');
        }
      }
      const pending = [], seen = new Set();
      for (const material of frame.materials.keys()) for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(material)))
        if (Object.hasOwn(descriptor, 'value') && (descriptor.value instanceof three.Texture || material.isShaderMaterial && key === 'uniforms'))
          pending.push(descriptor.value);
      const typedBuffer = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(Uint8Array.prototype), 'buffer').get;
      const viewBuffer = Object.getOwnPropertyDescriptor(DataView.prototype, 'buffer').get;
      while (pending.length) {
        const value = pending.pop();
        if (!value || typeof value !== 'object' || seen.has(value)) continue;
        if (paletteTextures.has(value) || ArrayBuffer.isView(value) &&
            paletteBuffers.has((value instanceof DataView ? viewBuffer : typedBuffer).call(value)))
          return refuse('Queued skeleton palettes cannot also serve as material inputs');
        if (ArrayBuffer.isView(value)) continue;
        if (seen.size >= maxValues) return refuse('Queued deformation material input capacity exceeded');
        seen.add(value);
        const prototype = Object.getPrototypeOf(value);
        if (value instanceof three.Texture) {
          const source = Object.getOwnPropertyDescriptor(value, 'source')?.value;
          if (source) pending.push(Object.getOwnPropertyDescriptor(source, 'data')?.value);
        } else if (prototype === Object.prototype || prototype === null || prototype === three.Uniform?.prototype || Array.isArray(value))
          for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(value)))
            if (Object.hasOwn(descriptor, 'value')) pending.push(descriptor.value);
      }
    }
    if (sources.size + [...frame.instances].filter(object => !sources.has(object)).length > maxMeshes)
      return refuse('Pass instance identity capacity exceeded');
    const hierarchy = new Map(), visibility = new Map(), layers = new Map(), orders = new Map(), selected = new Set(), geometries = new Map();
    const lodViews = new Map(), levelViews = new Map();
    const stack = [{object: root, visible: true}], objects = new Map(), packets = [], streams = new Set(), shapes = new Map();
    const vp = new three.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const frustum = new three.Frustum().setFromProjectionMatrix(vp, three.WebGLCoordinateSystem, camera.reversedDepth);
    let bytes = 0, geometryValues = 0, deformationValues = 0;
    for (const skeleton of skeletonSources) {
      const size = Math.max(4, Math.ceil(Math.sqrt(skeleton.bones.length * 4) / 4) * 4);
      bytes += size * size * 16; deformationValues += skeleton.bones.length * 32;
    }
    for (const object of deformationObjects) deformationValues +=
      (candidates.get(object).morph ? object.morphTargetInfluences.length : 0) + (object.isSkinnedMesh ? 32 : 0);
    const selectedSkeletons = new Set(), skeletonTextures = new Set();
    const wireframes = new Map([...frame.materials.keys()].map(material => [material, material.wireframe === true]));
    // Validate budgets and projection eligibility before acknowledging a source
    // upload or clearing any update range. Material eligibility was established
    // by captureThreeMaterialState without invoking application accessors.
    while (stack.length) {
      const {object, visible: parentVisible} = stack.pop();
      if (object.isLOD && object.autoUpdate && !frame.lods?.has(object)) return frame.instances.size || deformationObjects.size
        ? refuse('Instanced and deformed pass capture requires an explicit LOD selection') : null;
      const sourceVisible = (frame.excludedRoots ?? frame.excludedObjects)?.has(object) ? false : object.visible;
      const visible = parentVisible && (levelViews.get(object)?.visible ?? sourceVisible) !== false;
      // Match projectObject: hidden ancestors suppress updates; a layer-masked
      // LOD skips its own update but still traverses independently layered
      // children. Run the retained method on private levels before capturing
      // those children. A later budget/refusal must not publish source effects.
      if (visible && object.layers.test(camera.layers) && !object.isGroup && object.isLOD && object.autoUpdate) {
        const levels = object.levels.map(level => {
          let view = levelViews.get(level.object);
          if (!view) {
            view = {visible: frame.excludedRoots?.has(level.object) ? false : level.object.visible};
            levelViews.set(level.object, view);
          }
          return {distance: level.distance, hysteresis: level.hysteresis, object: view};
        });
        const view = Object.create(object, {
          levels: {value: levels}, _currentLevel: {value: object._currentLevel, writable: true},
        });
        applyLodUpdate(frame.lodUpdate, view, [camera]); lodViews.set(object, view);
      }
      hierarchy.set(object, object.children.slice()); visibility.set(object, visible);
      layers.set(object, object.layers.mask); orders.set(object, object.renderOrder);
      for (let i = object.children.length - 1; i >= 0; i--) stack.push({object: object.children[i], visible});
      const candidate = candidates.get(object);
      if (!candidate) continue;
      geometries.set(object, candidate.geometry);
      if (candidate.instanced)
        for (const attribute of [object.instanceMatrix, object.instanceColor]) if (attribute) bytes += attribute.array.byteLength;
      if (visible && object.layers.test(camera.layers) && (!object.frustumCulled || object.intersectsFrustum(frustum))) {
        selected.add(object);
        if (candidate.skinned) selectedSkeletons.add(object.skeleton);
        if (candidate.instanced)
          for (const attribute of [object.instanceMatrix, object.instanceColor]) if (attribute) streams.add(attribute);
        const bounds = candidate.instanced || candidate.skinned ? object : candidate.geometry;
        if (sortObjects && bounds.boundingSphere === null) bounds.computeBoundingSphere();
      }
    }
    try {
      for (const source of selected) {
        const selectedGeometry = candidates.get(source).geometry;
        if (!shapes.has(selectedGeometry)) {
          const shape = geometry.inspect(selectedGeometry, maxValues - geometryValues);
          shape.uploads = new Set(shape.attributes.map(([, entry]) => entry.owner));
          shapes.set(selectedGeometry, shape); geometryValues += shape.values;
        }
        const shape = shapes.get(selectedGeometry), materials = frame.objects.get(source);
        const selections = Array.isArray(materials)
          ? shape.groups.map(group => ({material: materials[group.materialIndex], group})) : [{material: materials, group: null}];
        for (const {material, group} of selections) if (material?.visible) {
          shape.morphSelected = true;
          if (source.isSkinnedMesh) skeletonTextures.add(source.skeleton);
          const effective = frame.overrideMaterial && material.allowOverride === true ? frame.overrideMaterial : material;
          if (source.isMesh && wireframes.get(effective)) shape.edgePlan ??= geometry.edgePlan(selectedGeometry);
          else if (shape.index) {
            const start = Math.max(shape.drawRange.start, group?.start ?? 0);
            const end = Math.min(shape.drawRange.start + shape.drawRange.count,
              group ? group.start + group.count : Infinity, shape.index.count);
            if (end >= start && Number.isFinite(end - start)) shape.uploads.add(shape.index.owner);
          }
        }
      }
      geometry.validate(shapes);
    } catch (error) {
      if (error.code !== 'GEOMETRY_GPU_HISTORY') throw error;
      if (required) return refuse(error.message);
      // An exotic geometry keeps its existing live upload path. Its ordinary
      // object's already-supported pose and membership capture still applies.
      shapes.clear(); geometryValues = 0;
    }
    // Count each shared geometry storage once per accepted use. Instance
    // packets above also reserve hidden streams, matching their prior contract.
    const geometryStreams = new Set([...shapes.values()].flatMap(shape => [...shape.uploads]));
    for (const owner of geometryStreams) { bytes += owner.array.byteLength; streams.add(owner); }
    for (const shape of shapes.values()) bytes += (shape.edgePlan?.bytes ?? 0) + (shape.morphSelected ? shape.morphBytes : 0);
    const values = hierarchy.size * 4 + candidates.size * 40 + geometryValues + deformationValues;
    if (values > maxValues || bytes > maxBytes - pendingBytes) return refuse('Pending instanced pass snapshot capacity exceeded');
    try { history.validate(streams); }
    catch (error) { if (error.code !== 'GEOMETRY_GPU_HISTORY') throw error; return refuse(error.message); }
    const accepted = new Map([...streams].map(attribute => [attribute, history.capture(attribute)]));
    // Preserve the source's public update and lazy texture creation effects at
    // acceptance. Private views recompute only from frozen bones during draws.
    for (const skeleton of selectedSkeletons) skeleton.update();
    for (const skeleton of skeletonTextures) if (skeleton.boneTexture === null) skeleton.computeBoneTexture();
    const skeletonPackets = new Map([...skeletonSources].map(source => [source, skeletons.capture(source)]));
    const geometryPackets = new Map([...shapes].map(([source, shape]) => [source, geometry.capture(shape, accepted)]));
    const deformations = new Set();
    for (const [source, candidate] of candidates) {
      const state = privateObject(source, candidate.instanced, candidate.skinned), view = state.view, active = selected.has(source), captured = [];
      for (const key of candidate.instanced ? ['instanceMatrix', 'instanceColor'] : []) {
        const attribute = source[key];
        if (!attribute) { captured.push([key, null, null]); continue; }
        state.attributes.add(attribute);
        if (!accepted.has(attribute)) accepted.set(attribute, attribute.array.slice());
        captured.push([key, privateAttribute(attribute), accepted.get(attribute)]);
      }
      objects.set(source, view);
      hierarchy.set(view, hierarchy.get(source)); visibility.set(view, visibility.get(source));
      layers.set(view, layers.get(source)); orders.set(view, orders.get(source));
      let materials = frame.objects.get(source);
      if (deformationObjects.has(source)) {
        // WebGL projects a skinned mesh (updating its skeleton) before it
        // selects visible materials. An unselected material must not prepare
        // an index/atlas that the source never consumed for this request.
        materials = Array.isArray(materials) ? materials.map(material => material?.visible ? material : null)
          : materials?.visible ? materials : null;
      }
      frame.objects.set(view, active ? materials : null);
      const bounds = candidate.instanced || candidate.skinned ? source : candidate.geometry;
      if (deformationObjects.has(source)) deformations.add(view);
      packets.push({state, captured, count: source.count, geometry: geometryPackets.get(candidate.geometry)?.view ?? candidate.geometry,
        influences: candidate.morph ? source.morphTargetInfluences.slice() : undefined, skin: candidate.skinned ? {
          skeleton: skeletonPackets.get(source.skeleton).view, bindMode: source.bindMode,
          bindMatrix: new three.Matrix4().copy(source.bindMatrix), bindMatrixInverse: new three.Matrix4().copy(source.bindMatrixInverse)} : null,
        matrixWorld: new three.Matrix4().copy(source.matrixWorld), boundingSphere: bounds.boundingSphere
          ? new three.Sphere(new three.Vector3().copy(bounds.boundingSphere.center), bounds.boundingSphere.radius) : null,
        castShadow: source.castShadow, receiveShadow: source.receiveShadow});
    }
    // Passes initially hide their exclusions, but a source LOD update can
    // reveal a level again. Keep only exclusions that remain effectively
    // hidden after the captured preorder, including inherited visibility.
    if (frame.lods?.size && frame.excludedObjects)
      frame.excludedObjects = new Set([...frame.excludedObjects].filter(object => visibility.get(object) !== true));
    pendingBytes += bytes;
    let released = false, publishedLod = false;
    return {objects, hierarchy, visibility, layers, orders, geometries, values, bytes, wireframes, deformations, deformationObjects, lods: frame.lods,
      geometryView: source => geometryPackets.get(source)?.view,
      geometrySafe: () => [...geometryPackets.values()].every(packet => packet.safe()),
      deformationSafe: () => [...skeletonPackets.values()].every(packet => packet.safe()),
      publishLod() {
        if (released || publishedLod) return;
        // The renderer calls this only once all request captures and queue
        // limits succeed. These admitted writable data properties have no
        // application callbacks, and submission never replays their effects.
        for (const [source, view] of levelViews) source.visible = view.visible;
        for (const [source, view] of lodViews) source._currentLevel = view._currentLevel;
        publishedLod = true;
      },
      install() {
        if (released) fail('An instanced pass snapshot was released before submission');
        for (const packet of geometryPackets.values()) packet.install();
        for (const packet of skeletonPackets.values()) packet.install();
        for (const packet of packets) {
          const view = packet.state.view;
          view.count = packet.count; view.geometry = packet.geometry;
          view.morphTargetInfluences = packet.influences;
          if (packet.skin) {
            view.skeleton = packet.skin.skeleton; view.bindMode = packet.skin.bindMode;
            view.bindMatrix.copy(packet.skin.bindMatrix); view.bindMatrixInverse.copy(packet.skin.bindMatrixInverse);
          }
          view.matrixWorld.copy(packet.matrixWorld); view.boundingSphere = packet.boundingSphere;
          view.castShadow = packet.castShadow; view.receiveShadow = packet.receiveShadow;
          for (const [key, attribute, array] of packet.captured) {
            view[key] = attribute;
            if (attribute && attribute.array !== array) {
              attribute.array = array; attribute.count = array.length / attribute.itemSize; attribute.needsUpdate = true;
            }
          }
        }
      },
      release() { if (!released) {
        released = true; pendingBytes -= bytes;
        for (const packet of geometryPackets.values()) packet.release();
        for (const packet of skeletonPackets.values()) packet.release();
      } },
    };
  }
  return {history, capture,
    dispose() { for (const [source, state] of sources) source.removeEventListener('dispose', state.dispose); sources.clear(); attributes = new WeakMap(); rigidSources = new WeakMap(); skeletons.dispose(); geometry.dispose(); history.dispose(); },
    get pendingBytes() { return pendingBytes; },
  };
}
