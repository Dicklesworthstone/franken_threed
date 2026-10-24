/** Native ClippingGroup projection. The source NodeMaterial/ClippingNode owns
 * all shader equations; this adapter supplies the projected plane arrays that
 * the private r186 ClippingContext normally records during scene traversal.
 * Project before draw callbacks, and retain those values for an accepted render.
 */
export function captureThreeNodeClipping(three, {root, camera, hierarchy, visibility,
  shadowPass = false, maxPlanes = 8192, backgroundMesh = null, fail} = {}) {
  const contexts = new Map(), empty = {unionPlanes: [], intersectionPlanes: []};
  const normalMatrix = new three.Matrix3().getNormalMatrix(camera.matrixWorldInverse);
  const plane = new three.Plane(), stack = [{object: root, context: empty}];
  let planeCount = 0;
  if (backgroundMesh) contexts.set(backgroundMesh, empty);
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
      const projected = object.clippingPlanes.map(value => {
        plane.copy(value).applyMatrix4(camera.matrixWorldInverse, normalMatrix);
        return new three.Vector4(-plane.normal.x, -plane.normal.y, -plane.normal.z, plane.constant);
      });
      context = object.clipIntersection
        ? {unionPlanes: context.unionPlanes, intersectionPlanes: [...context.intersectionPlanes, ...projected]}
        : {unionPlanes: [...context.unionPlanes, ...projected], intersectionPlanes: context.intersectionPlanes};
    }
    contexts.set(object, context);
    const children = hierarchy.get(object);
    for (let i = children.length - 1; i >= 0; i--) stack.push({object: children[i], context});
  }
  return contexts;
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
export function updateThreeNodeClipping(clipping, context) {
  for (const key of ['unionPlanes', 'intersectionPlanes'])
    for (let i = 0; i < clipping[key].length; i++) clipping[key][i].copy(context.clippingContext[key][i]);
}
