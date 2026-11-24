/** Native ClippingGroup projection. The source NodeMaterial/ClippingNode owns
 * all shader equations; this adapter supplies the projected plane arrays that
 * the private r186 ClippingContext normally records during scene traversal.
 * Project before draw callbacks, and retain those values for an accepted render.
 */
const contextOwners = new WeakMap(), snapshotSources = new WeakMap();
let contextSerial = 0;
const contextState = () => ({id: contextSerial++, version: 0, cacheKey: '', parentVersion: null,
  clipIntersection: null, unionPlanes: [], intersectionPlanes: []});

/** Renderer disposal clears source RenderContexts. Already captured snapshots
 * and builders retain their arrays; later ownership starts fresh contexts. */
export function releaseThreeNodeClipping(contextOwner) {
  contextOwners.delete(contextOwner);
}

export function captureThreeNodeClipping(three, {root, camera, hierarchy, visibility,
  shadowPass = false, maxPlanes = 8192, backgroundMesh = null, contextOwner = root,
  renderContext = 'default', fail} = {}) {
  let scopes = contextOwners.get(contextOwner);
  if (!scopes) contextOwners.set(contextOwner, scopes = new Map());
  let scope = scopes.get(renderContext);
  if (!scope) scopes.set(renderContext, scope = {root: contextState(), groups: new WeakMap()});
  const contexts = new Map(), snapshots = new Map();
  // Source ClippingContext owns stable group IDs and plane arrays per render
  // context. Snapshot those arrays for this accepted call; later projection
  // must not change either queued coefficients or the captured cache key.
  const snapshot = context => {
    let value = snapshots.get(context);
    if (!value) {
      snapshots.set(context, value = {cacheKey: context.cacheKey,
        unionPlanes: context.unionPlanes.map(plane => plane.clone()),
        intersectionPlanes: context.intersectionPlanes.map(plane => plane.clone())});
      snapshotSources.set(value, context);
    }
    return value;
  };
  const normalMatrix = new three.Matrix3().getNormalMatrix(camera.matrixWorldInverse);
  const plane = new three.Plane(), stack = [{object: root, context: scope.root}];
  let planeCount = 0;
  if (backgroundMesh) contexts.set(backgroundMesh, snapshot(scope.root));
  while (stack.length) {
    const item = stack.pop(), object = item.object;
    let context = item.context;
    if (!visibility.get(object)) continue;
    // A group's layer mask controls its own clipping contribution, not whether
    // its descendants are traversed. Group transforms do not transform planes:
    // ClippingGroup planes are world-space source values.
    if (object.layers.test(camera.layers) && object.isGroup && object.isClippingGroup && object.enabled &&
        (!shadowPass || object.clipShadows)) {
      if (!(object instanceof three.ClippingGroup) || !Array.isArray(object.clippingPlanes) ||
          object.clippingPlanes.some(value => !(value instanceof three.Plane)))
        fail('CLIPPING', 'Native clipping requires source ClippingGroup Plane arrays');
      planeCount += object.clippingPlanes.length;
      if (planeCount > maxPlanes) fail('LIMIT', 'Native clipping exceeds its projected plane budget');
      const parent = context;
      context = scope.groups.get(object);
      if (!context) scope.groups.set(object, context = contextState());
      if (context.parentVersion !== parent.version) {
        context.unionPlanes = Array.from(parent.unionPlanes);
        context.intersectionPlanes = Array.from(parent.intersectionPlanes);
        context.parentVersion = parent.version;
      }
      if (context.clipIntersection !== object.clipIntersection) {
        context.clipIntersection = object.clipIntersection;
        const other = object.clipIntersection ? 'unionPlanes' : 'intersectionPlanes';
        context[other].length = parent[other].length;
      }
      const key = object.clipIntersection ? 'intersectionPlanes' : 'unionPlanes';
      const values = context[key], offset = parent[key].length, count = object.clippingPlanes.length;
      const changed = values.length !== offset + count;
      if (changed) {
        values.length = offset + count;
        for (let i = 0; i < count; i++) values[offset + i] = new three.Vector4();
      }
      for (let i = 0; i < count; i++) {
        plane.copy(object.clippingPlanes[i]).applyMatrix4(camera.matrixWorldInverse, normalMatrix);
        values[offset + i].set(-plane.normal.x, -plane.normal.y, -plane.normal.z, plane.constant);
      }
      if (changed) {
        context.version++;
        context.cacheKey = `${context.id}:${context.intersectionPlanes.length}:${context.unionPlanes.length}`;
      }
    }
    contexts.set(object, snapshot(context));
    const children = hierarchy.get(object);
    for (let i = children.length - 1; i >= 0; i--) stack.push({object: children[i], context});
  }
  return contexts;
}

/** Shared source builders close over one clipping context's uniform arrays.
 * Counts alone cannot distinguish equally-sized groups with different planes.
 * Public scene projection supplies this key; explicit callers can pass the
 * pinned ClippingContext directly, without evaluating application getters.
 */
export function threeNodeClippingCacheKey(context, fail) {
  const clipping = context.clippingContext;
  if (clipping == null) return '';
  const descriptor = Object.getOwnPropertyDescriptor(clipping, 'cacheKey');
  if (!descriptor && !clipping.unionPlanes.length && !clipping.intersectionPlanes.length) return '';
  if (typeof descriptor?.value !== 'string')
    fail('CLIPPING', 'Shared native clipping requires the captured source context cache key');
  return descriptor.value;
}

/** Source UniformArrayNode closes over the arrays selected at construction,
 * even if a later ClippingContext update replaces those arrays. Keep that
 * relationship private while public captures retain detached plane values. */
export function threeNodeClippingSource(context) {
  const clipping = context.clippingContext;
  const source = snapshotSources.get(clipping) ?? clipping;
  return {unionPlanes: source?.unionPlanes ?? [], intersectionPlanes: source?.intersectionPlanes ?? []};
}

export function inspectThreeNodeClipping(three, context, maxPlanes, fail) {
  const clipping = context.clippingContext;
  if (clipping == null) return '0/0';
  if (!Array.isArray(clipping.unionPlanes) || !Array.isArray(clipping.intersectionPlanes) ||
      clipping.unionPlanes.length + clipping.intersectionPlanes.length > maxPlanes ||
      [...clipping.unionPlanes, ...clipping.intersectionPlanes].some(plane => !(plane instanceof three.Vector4)))
    fail('CLIPPING', 'Native clipping requires bounded projected Vector4 plane arrays');
  return `${clipping.intersectionPlanes.length}/${clipping.unionPlanes.length}`;
}

/** A compiled source UniformArrayNode retains its input array. Copy the next
 * projection into those stable values before its source RENDER update, rather
 * than replacing the array or rebuilding shaders when only planes move.
 */
export function updateThreeNodeClipping(clipping, context, retained = null) {
  for (const key of ['unionPlanes', 'intersectionPlanes'])
    // A source material version comparison can retain its old builder for one
    // draw while the selected clipping context changes. UniformArrayNode keeps
    // its allocation in that case: longer arrays truncate and shorter arrays
    // leave the previously uploaded tail unchanged.
    for (let i = 0, length = retained ? Math.min(clipping[key].length, retained[key].length) : clipping[key].length; i < length; i++)
      clipping[key][i].copy((retained ?? context.clippingContext)[key][i]);
}
