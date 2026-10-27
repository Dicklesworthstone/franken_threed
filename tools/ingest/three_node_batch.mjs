/** Native BatchedMesh identity and draw snapshots. The pinned source callback
 * owns selection, culling and sorting; its shader owns indirect texture lookup.
 * WebGPU's firstInstance carries the source draw ordinal, not an instance ID.
 */
export function inspectThreeNativeBatch(three, object, fail) {
  if (object.isBatchedMesh !== true) return null;
  if (!(object instanceof three.BatchedMesh) || Object.getPrototypeOf(object) !== three.BatchedMesh.prototype ||
      object.onBeforeRender !== three.BatchedMesh.prototype.onBeforeRender ||
      object.onBeforeShadow !== three.BatchedMesh.prototype.onBeforeShadow ||
      object.onAfterRender !== three.Object3D.prototype.onAfterRender ||
      object.onAfterShadow !== three.Object3D.prototype.onAfterShadow ||
      !(object.geometry instanceof three.BufferGeometry) ||
      object.geometry.isInstancedBufferGeometry || object.isInstancedMesh ||
      object.customSort !== null && typeof object.customSort !== 'function')
    fail('OBJECT', 'Native batches require the pinned BatchedMesh lifecycle and ordinary geometry');
  const textures = [object._matricesTexture, object._indirectTexture, object._colorsTexture];
  for (const [index, texture] of textures.entries()) {
    if (index === 2 && texture === null) continue;
    const integral = index === 1, ArrayType = integral ? Uint32Array : Float32Array;
    if (!(texture instanceof three.DataTexture) || Object.getPrototypeOf(texture) !== three.DataTexture.prototype ||
        !(texture.image?.data instanceof ArrayType) ||
        texture.type !== (integral ? three.UnsignedIntType : three.FloatType) ||
        texture.format !== (integral ? three.RedIntegerFormat : three.RGBAFormat))
      fail('OBJECT', 'Native batches require their source matrix, indirect-index and optional color textures');
  }
  // Batch.js closes over these allocations. Resizing the batch or introducing
  // its first color must rebuild the graph before any source node updates.
  return textures.map(texture => texture?.id ?? null).join('/');
}

export function captureThreeNativeBatch(object, remainingDraws, fail, instanceCount, {wireframe = false, indexCount} = {}) {
  if (object.isBatchedMesh !== true) return null;
  if (remainingDraws < 1) fail('LIMIT', 'Native batch subdraws exceed the captured scene draw budget');
  // RenderObject gates a batch on object.count, then the backend issues one
  // instance per subdraw. It does not multiply the batch by object.count.
  if (instanceCount === 0) return Object.freeze([]);
  const count = object._multiDrawCount, starts = object._multiDrawStarts, counts = object._multiDrawCounts;
  if (!(starts instanceof Int32Array) || !(counts instanceof Int32Array) ||
      !Number.isSafeInteger(count) || count < 0 || count > starts.length || count > counts.length)
    fail('OBJECT', 'Native batch draw arrays require bounded source subdraw counts');
  if (Math.max(1, count) > remainingDraws) fail('LIMIT', 'Native batch subdraws exceed the captured scene draw budget');
  const index = object.geometry.index, indexed = wireframe || index !== null;
  const extent = wireframe ? indexCount : index?.count ?? object.geometry.attributes.position?.count;
  // Source selection already doubled wireframe ranges. Even nonindexed batches
  // now draw the derived edge index; preserve the source callback's byte stride.
  const divisor = indexed ? object._multiDrawBytesPerElement : 1;
  if (!Number.isSafeInteger(extent) || extent < 0 || indexed && !(wireframe ? [1, 2, 4] : [2, 4]).includes(divisor))
    fail('GEOMETRY', 'Native batch ranges require a valid source vertex/index extent and index stride');
  const draws = [];
  for (let ordinal = 0; ordinal < count; ordinal++) {
    const first = starts[ordinal] / divisor, size = counts[ordinal];
    if (!Number.isSafeInteger(first) || first < 0 || size < 0 || first + size > extent)
      fail('GEOMETRY', 'Native batch subdraw range is outside its captured geometry');
    draws.push(Object.freeze({first, count: size, instanceCount: 1, firstInstance: ordinal}));
  }
  return Object.freeze(draws);
}
