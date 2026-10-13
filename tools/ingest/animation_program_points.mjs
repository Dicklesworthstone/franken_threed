/** Indexed source points on the existing instanced-quad shader ABI.
 * Native firstInstance selects BOTH the instance-step vertex record and the
 * shader's source gl_VertexID. Merge ascending runs only: never sort, dedupe,
 * gather CPU vertex arrays, or replace the authored point-size/clip shader.
 * Plans own immutable numbers from resident index history, not live source data.
 */
export function snapshotProgramPointRuns(geometry, input, maxRuns, fail) {
  const invalid = message => fail('ANIMATION_RENDER_GEOMETRY', message);
  const extent = geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount;
  const start = input.first ?? 0, count = input.count ?? extent;
  if (!Number.isSafeInteger(start) || start < 0 || start > 0xffffffff ||
      !Number.isSafeInteger(count) || count < 0 || count > 0xffffffff)
    invalid('Point draw ranges require nonnegative 32-bit integers');
  const instances = input.instanceCount ?? 1;
  if (instances !== 0 && instances !== 1)
    invalid('Instanced point sprites require a separate source-instance shader path');
  const first = Math.min(extent, Math.max(start, geometry.drawRange.first));
  const end = Math.min(extent, start + count, geometry.drawRange.first + geometry.drawRange.count);
  const runs = [];
  let points = 0;
  const append = (vertex, length) => {
    if (runs.length / 2 >= maxRuns)
      fail('ANIMATION_RENDER_LIMIT', 'Point draw expansion exceeds the per-frame native draw budget');
    runs.push(vertex, length);
  };
  if (instances === 0 || end <= first) return Object.freeze({runs: Object.freeze(runs), count: 0});
  if (!geometry.indexBuffer) {
    append(first, end - first); points = end - first;
  } else {
    if (typeof geometry.readIndex !== 'function' || !['uint16', 'uint32'].includes(geometry.indexFormat))
      invalid('Indexed point sprites require resident index history');
    const restart = geometry.indexFormat === 'uint16' ? 0xffff : 0xffffffff;
    // Attribute-free programs may calculate positions from any gl_VertexID.
    const bounded = geometry.layouts.some(layout => layout.arrayStride > 0);
    let previous = -2;
    for (let i = first; i < end; i++) {
      const vertex = geometry.readIndex(i);
      // WebGL2's fixed-index restart is always enabled, also for points.
      if (vertex === restart) { previous = -2; continue; }
      if (!Number.isSafeInteger(vertex) || vertex < 0 || vertex >= restart ||
          (bounded && vertex >= geometry.vertexCount))
        invalid('Point index exceeds resident vertex storage');
      if (vertex === previous + 1 && runs.length) runs[runs.length - 1]++;
      else append(vertex, 1);
      previous = vertex; points++;
    }
  }
  return Object.freeze({runs: Object.freeze(runs), count: points});
}
