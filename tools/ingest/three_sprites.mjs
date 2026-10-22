/** Pinned r186 SpriteNodeMaterial's default camera-facing position transform.
 * Shader construction remains source-owned; this bounded native path lowers
 * the default position/rotation/scale into the existing unlit draw matrix.
 * Source geometry, object transforms and public matrix caches remain untouched.
 */
export class ThreeSpriteError extends Error {
  constructor(message) {
    super(`THREE_SPRITE_VALUE: ${message}`);
    this.name = 'ThreeSpriteError'; this.code = 'THREE_SPRITE_VALUE';
  }
}
const finite = (n, name) => {
  if (typeof n !== 'number' || !Number.isFinite(n)) throw new ThreeSpriteError(`Expected finite ${name}`);
  return n;
};

/** Source RenderObject clamps negative sprite counts to zero. The explicit
 * native bridge requires an integer count and charges every repeated draw to
 * its ordinary draw budget, preserving overlapping transparent instances.
 */
export function threeSpriteCount(object) {
  const count = object.count ?? 1;
  if (!Number.isSafeInteger(count)) throw new ThreeSpriteError('Sprite count must be an integer');
  return Math.max(0, count);
}

/** World-space matrix whose projection equals SpriteNodeMaterial.setupPositionView
 * with null node overrides. World column lengths retain parent scale/shear;
 * source object rotation does not turn the billboard away from its camera.
 * The zero third column deliberately ignores positionGeometry.z, as r186 does.
 */
export function threeSpriteWorldMatrix(T, object, material, camera, {perspective = camera.isPerspectiveCamera === true} = {}) {
  const rotation = finite(material.rotation, 'sprite rotation');
  if (typeof material.sizeAttenuation !== 'boolean') throw new ThreeSpriteError('Sprite sizeAttenuation must be boolean');
  const world = object.matrixWorld.elements;
  const modelView = new T.Matrix4().multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
  const mv = modelView.elements;
  let sx = Math.hypot(world[0], world[1], world[2]), sy = Math.hypot(world[4], world[5], world[6]);
  if (perspective && material.sizeAttenuation === false) { sx *= -mv[14]; sy *= -mv[14]; }
  const center = object.center?.isVector2 === true ? object.center : {x: .5, y: .5};
  const dx = .5 - finite(center.x, 'sprite center x'), dy = .5 - finite(center.y, 'sprite center y');
  const c = Math.cos(rotation), s = Math.sin(rotation);
  const x = [c * sx, s * sx], y = [-s * sy, c * sy];
  const view = new T.Matrix4().set(
    x[0], y[0], 0, mv[12] + dx * x[0] + dy * y[0],
    x[1], y[1], 0, mv[13] + dx * x[1] + dy * y[1],
    0, 0, 0, mv[14],
    0, 0, 0, 1,
  );
  // Invert the actual view matrix: Camera.matrixWorld can retain scale that
  // the native renderer intentionally removed from matrixWorldInverse.
  const inverseView = new T.Matrix4().copy(camera.matrixWorldInverse).invert();
  const result = inverseView.multiply(view).elements;
  if (result.some(n => !Number.isFinite(Math.fround(n)))) throw new ThreeSpriteError('Sprite transform must fit finite f32');
  // Inverting an affine camera can round its homogeneous 1 to 1 + epsilon.
  // Keep the generated draw matrix exactly affine without relaxing core checks.
  if (result[3] !== 0 || result[7] !== 0 || result[11] !== 0 || Math.abs(result[15] - 1) > 1e-12)
    throw new ThreeSpriteError('Sprite camera must have an affine view matrix');
  result[3] = result[7] = result[11] = 0; result[15] = 1;
  return result;
}
