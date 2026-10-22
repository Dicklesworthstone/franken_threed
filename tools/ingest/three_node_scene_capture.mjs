/** Synchronous selection and uniform capture for the native TSL graphics route.
 * Source object/material identities reach the retained builder and NodeFrame.
 * Pipeline preparation consumes these packets later and never repeats updates.
 * Geometry bytes still follow the bridge's existing upload/version contract.
 */
import {hasThreeNodeProgram, ThreeNodeProgramError} from './three_node_program.mjs';

const fail = (code, message) => { throw new ThreeNodeProgramError(code, message); };
const integer = value => Number.isSafeInteger(value) && value >= 0;
const transparent = material => material.transparent === true || material.transmission > 0 ||
  material.transmissionNode?.isNode === true || material.backdropNode?.isNode === true;

export function captureThreeNodeScene(three, root, camera, {getPrograms, renderer,
  overrideMaterial = null, maxNodes = 16384, maxDraws = 16384, uniforms = true,
  maxUniformBytes = 64 * 1024 * 1024, raster, getTextures, destination = null} = {}) {
  if (!integer(maxNodes) || maxNodes < 1 || !integer(maxDraws) || maxDraws < 1 || !integer(maxUniformBytes))
    fail('LIMIT', 'Invalid native scene capture capacity');
  const hierarchy = new Map(), objects = new Set(), bindings = [], pairs = new Map(), stack = [root];
  // Traverse hidden objects too: preparation must admit every program before a
  // visible logical draw can run an application node callback.
  while (stack.length) {
    const object = stack.pop();
    if (!(object instanceof three.Object3D) || hierarchy.has(object) || !Array.isArray(object.children))
      fail('GRAPH', 'Expected an acyclic source Object3D hierarchy');
    if (hierarchy.size + stack.length + object.children.length >= maxNodes)
      fail('LIMIT', 'Native scene traversal exceeds its node budget');
    const children = object.children.slice(); hierarchy.set(object, children);
    for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]);
    if (!object.material) continue;
    const originals = Array.isArray(object.material) ? object.material.slice() : [object.material];
    if (originals.length > maxNodes) fail('LIMIT', 'Native scene material array exceeds its budget');
    const effective = originals.map(original => original?.allowOverride === true && overrideMaterial !== null ? overrideMaterial : original);
    for (let i = 0; i < originals.length; i++) {
      const original = originals[i], material = effective[i];
      if (original === material || (!hasThreeNodeProgram(original) && !hasThreeNodeProgram(material))) continue;
      // r186 renderObject inherits these values before applying an override.
      // The first profile admits the already-equivalent state without mutating
      // a shared override material or changing its callback identity.
      if ((original.positionNode?.isNode && original.positionNode !== material.positionNode) ||
          ['alphaTest', 'alphaMap', 'displacementMap', 'displacementScale', 'displacementBias'].some(key => original[key] !== material[key]) ||
          transparent(original) !== material.transparent)
        fail('EFFECT', 'Native node material overrides require matching inherited position, alpha, displacement and transparency state');
    }
    if (!effective.some(hasThreeNodeProgram)) continue;
    if (effective.some(material => material && !hasThreeNodeProgram(material)))
      fail('MATERIAL', 'A native node mesh cannot mix authored node and fixed materials');
    if (!(object instanceof three.Mesh) || object.intersectsFrustum !== three.Mesh.prototype.intersectsFrustum ||
        (object.count !== undefined && object.count !== 1))
      fail('OBJECT', 'Native node scenes require ordinary meshes with the source frustum implementation');
    if (object.onBeforeRender !== three.Object3D.prototype.onBeforeRender || object.onAfterRender !== three.Object3D.prototype.onAfterRender)
      fail('HOOK', 'Native node object render hooks are not admitted');
    const geometry = object.geometry;
    if (!(geometry instanceof three.BufferGeometry) || !Array.isArray(geometry.groups) || geometry.groups.length > maxDraws)
      fail('GEOMETRY', 'Native node meshes require bounded source BufferGeometry groups');
    const owners = new Set(Object.values(geometry.attributes).map(attribute => attribute.isInterleavedBufferAttribute ? attribute.data : attribute));
    if (geometry.index) owners.add(geometry.index);
    for (const owner of owners) {
      const prototype = owner.isInterleavedBuffer ? three.InterleavedBuffer.prototype : three.BufferAttribute.prototype;
      if (owner.onUploadCallback !== prototype.onUploadCallback)
        fail('HOOK', 'Native node attributes require the source upload callback');
    }
    const byMaterial = new Map();
    for (const material of effective) if (material && !byMaterial.has(material)) {
      if (bindings.length >= maxDraws) fail('LIMIT', 'Native scene program binding capacity exceeded');
      const binding = {object, geometry, material, nodeProgram: null, raster: null};
      byMaterial.set(material, binding); bindings.push(binding);
    }
    pairs.set(object, {originals, effective, byMaterial, array: Array.isArray(object.material), geometry});
    objects.add(object);
  }
  if (!objects.size) return null;
  if (!(camera instanceof three.Camera) || camera.isArrayCamera || camera.isMultiViewCamera || camera.reversedDepth || renderer?.reversedDepthBuffer)
    fail('CAMERA', 'Native node scenes require an ordinary non-reversed camera');
  if (typeof getPrograms !== 'function' || typeof raster !== 'function' || !renderer?._nodes?.nodeFrame)
    fail('SOURCE', 'Native scene capture needs its source compiler, renderer NodeFrame and raster adapter');
  if (renderer.xr?.isPresenting) fail('CAMERA', 'Native node scenes do not yet capture XR camera views');
  const programs = getPrograms(), context = {scene: root, camera};
  // Admission is completed before camera updates, graph construction or node
  // updates. In particular, a hidden unsupported graph cannot cause partial
  // visible update callbacks followed by a preparation refusal.
  for (const binding of bindings) programs.inspect(binding.material, binding.object, context);
  if (root.matrixWorldAutoUpdate === true) root.updateMatrixWorld();
  if (camera.coordinateSystem !== three.WebGPUCoordinateSystem) {
    if (typeof camera.updateProjectionMatrix !== 'function') fail('CAMERA', 'The source camera cannot update its WebGPU projection');
    camera.coordinateSystem = three.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  }
  if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld();
  const opaque = [], translucent = [], draws = [];
  if (uniforms) {
    const projection = new three.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const frustum = new three.Frustum().setFromProjectionMatrix(projection, camera.coordinateSystem, camera.reversedDepth);
    const center = new three.Vector4(), sortObjects = renderer.sortObjects === true;
    const visible = [{object: root, groupOrder: 0}];
    const append = (object, state, original, material, group, groupOrder, z) => {
      if (!original?.visible || !material) return;
      if (opaque.length + translucent.length >= maxDraws) fail('LIMIT', 'Native source draw list exceeds its capacity');
      const {geometry, byMaterial} = state, range = geometry.drawRange;
      const extent = geometry.index ? geometry.index.count : geometry.attributes.position?.count;
      if (!integer(extent) || !integer(range?.start) || !(range.count === Infinity || integer(range.count)) ||
          (group && (!integer(group.start) || !(group.count === Infinity || integer(group.count)))))
        fail('GEOMETRY', 'Native node draw ranges require nonnegative integer starts and counts');
      const first = Math.min(extent, Math.max(range.start, group?.start ?? 0));
      const end = Math.min(extent, range.start + range.count, group ? group.start + group.count : Infinity);
      const listMaterial = Object.freeze({id: original.id, transparent: transparent(original), transmission: original.transmission ?? 0});
      const packet = {binding: byMaterial.get(material), object, geometry, material, listMaterial,
        group: group ? Object.freeze({...group}) : null, groupOrder, z, renderOrder: object.renderOrder,
        first, count: Math.max(0, end - first), frontFaceCW: object.matrixWorld.determinant() < 0,
        nodeRangeCaptured: true, nodeUniforms: null};
      (listMaterial.transparent ? translucent : opaque).push(packet);
    };
    while (visible.length) {
      const item = visible.pop(), object = item.object; let {groupOrder} = item;
      if (object.visible === false) continue;
      if (object.layers.test(camera.layers)) {
        if (object.isGroup) groupOrder = object.renderOrder;
        else if (object.isLOD) { if (object.autoUpdate === true) object.update(camera); }
        else if (objects.has(object) && (!object.frustumCulled || object.intersectsFrustum(frustum))) {
          const state = pairs.get(object), {geometry} = state;
          let z = 0;
          if (sortObjects) {
            if (geometry.boundingSphere === null) geometry.computeBoundingSphere();
            z = center.copy(geometry.boundingSphere.center).applyMatrix4(object.matrixWorld).applyMatrix4(projection).z;
          }
          if (state.array) for (const group of geometry.groups)
            append(object, state, state.originals[group.materialIndex], state.effective[group.materialIndex], group, groupOrder, z);
          else append(object, state, state.originals[0], state.effective[0], null, groupOrder, z);
        }
      }
      const children = hierarchy.get(object);
      for (let i = children.length - 1; i >= 0; i--) visible.push({object: children[i], groupOrder});
    }
    if (sortObjects) {
      const order = (a, b) => a.groupOrder - b.groupOrder || a.renderOrder - b.renderOrder;
      // Source WebGPU opaque sorting deliberately has no material-id key.
      opaque.sort((a, b) => order(a, b) || a.z - b.z || a.object.id - b.object.id);
      translucent.sort((a, b) => order(a, b) || b.z - a.z || a.object.id - b.object.id);
    }
    draws.push(...opaque, ...translucent);
  }
  let textureOwner = null;
  for (const binding of bindings) {
    binding.nodeProgram = programs.compile(binding.material, binding.object, context);
    binding.raster = raster(binding.material); Object.freeze(binding);
    if (programs.textures(binding.nodeProgram).length) {
      if (typeof getTextures !== 'function') fail('SOURCE', 'Native sampled graphs require the renderer texture owner');
      textureOwner ??= getTextures();
    }
  }
  let bytes = 0;
  for (const draw of draws) for (const group of draw.binding.nodeProgram.groups) {
    bytes += group.byteLength;
    if (!Number.isSafeInteger(bytes) || bytes > maxUniformBytes)
      fail('LIMIT', 'Native scene capture exceeds its pending uniform byte budget');
  }
  const frame = renderer._nodes.nodeFrame;
  const frameContext = {...context, frameId: frame.frameId, time: frame.time, deltaTime: frame.deltaTime,
    renderId: renderer.info.calls};
  const textureUses = [];
  const capture = () => {
    for (const draw of draws) {
      draw.object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, draw.object.matrixWorld);
      draw.object.normalMatrix.getNormalMatrix(draw.object.modelViewMatrix);
      const updateContext = {...frameContext, object: draw.object, material: draw.material};
      const captured = programs.captureDraw(draw.binding.nodeProgram, updateContext);
      draw.nodeUniforms = captured.uniforms;
      // Lease each post-update texture before after() or the next logical draw
      // can edit it. The shared owner snapshots CPU upload/sampler histories;
      // target leases hold generation metadata and observe contents in the FIFO.
      draw.nodeTextures = Object.freeze(captured.textures.map(binding => {
        if (destination && binding.texture.renderTarget === destination)
          fail('FEEDBACK', 'A native draw cannot sample its own render target');
        const lease = textureOwner.captureBinding(binding);
        textureUses.push(lease);
        return Object.freeze({group: binding.group, binding: binding.binding, lease});
      }));
      programs.after(draw.binding.nodeProgram, updateContext);
      Object.freeze(draw.nodeUniforms); Object.freeze(draw);
    }
  };
  try {
    if (textureOwner) textureOwner.captureDispatch(capture);
    else capture();
  } catch (error) {
    for (const lease of textureUses) textureOwner.release(lease);
    throw error;
  }
  return Object.freeze({objects, bindings: Object.freeze(bindings), draws: Object.freeze(draws), bytes,
    textureOwner, textureUses: Object.freeze(textureUses)});
}
