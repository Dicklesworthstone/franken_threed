/**
 * Closed numeric body of the pinned r186 MarchingCubes.update triangulation.
 * The build-time compiler lowers this entire body to one import-free Wasm call.
 *
 * Arrays are independent, fixed, unshared storage. All floating arrays,
 * INCLUDING the three edge lists, are Float32Array; tables are Int32Array.
 * Number arithmetic and every intermediate Float32 store retain source order.
 * Loop order is z/y/x, edges 0..11, then the upstream triangle-table order.
 * The caller supplies the ORIGINAL edge lists and normal cache, not substitutes:
 * they retain stale entries across calls, including after public table edits.
 *
 * This body does not own public objects, materials, upload versions or callbacks.
 * Flags are numeric 0/1, and flatShading is a proven ordinary data property.
 * The guarded adapter must retain original JS for aliases/accessors/unsupported
 * storage. Capacity/bounds/fuel failures abort private execution before publish.
 * No mesh welding, normal normalization, new topology or speedup is implied.
 *
 * Algorithm reference: Three.js r186 148ef33ecb6d2502ff796d4554abd1549c95d519,
 * examples/jsm/objects/MarchingCubes.js (MIT, see MARCHING_CUBES.md).
 */
export function triangulateMarchingCubes(
  field, normalCache, palette, position, normal, uv, color,
  vlist, nlist, clist, edgeTable, triTable,
  size, size2, halfsize, delta, yd, zd, isolation,
  flatShading, enableUvs, enableColors,
) {
  let count = 0;
  const end = size - 2;
  for (let z = 1; z < end; z++) {
    const zOffset = size2 * z;
    const fz = (z - halfsize) / halfsize;
    for (let y = 1; y < end; y++) {
      const yOffset = zOffset + size * y;
      const fy = (y - halfsize) / halfsize;
      for (let x = 1; x < end; x++) {
        const fx = (x - halfsize) / halfsize;
        const q = yOffset + x;
        let cube = 0;
        if (field[q] < isolation) cube |= 1;
        if (field[q + 1] < isolation) cube |= 2;
        if (field[q + yd] < isolation) cube |= 8;
        if (field[q + 1 + yd] < isolation) cube |= 4;
        if (field[q + zd] < isolation) cube |= 16;
        if (field[q + 1 + zd] < isolation) cube |= 32;
        if (field[q + yd + zd] < isolation) cube |= 128;
        if (field[q + 1 + yd + zd] < isolation) cube |= 64;
        const bits = edgeTable[cube];
        if (bits === 0) continue;
        const fx2 = fx + delta, fy2 = fy + delta, fz2 = fz + delta;
        for (let edge = 0; edge < 12; edge++) {
          if ((bits & (1 << edge)) === 0) continue;
          // Orient every edge in the same positive-axis direction as VIntX/Y/Z.
          // Reversing an edge is algebraically equivalent but can round differently.
          let cx = 0, cy = 0, cz = 0, axis = 0;
          if (edge === 1 || edge === 5 || edge === 9 || edge === 10) cx = 1;
          if (edge === 2 || edge === 6 || edge === 10 || edge === 11) cy = 1;
          if (edge >= 4 && edge <= 7) cz = 1;
          if (edge === 1 || edge === 3 || edge === 5 || edge === 7) axis = 1;
          if (edge >= 8) axis = 2;
          const a = q + cx + cy * yd + cz * zd;
          const b = a + (axis === 0 ? 1 : axis === 1 ? yd : zd);
          for (let endpoint = 0; endpoint < 2; endpoint++) {
            const point = endpoint === 0 ? a : b;
            const at = point * 3;
            // Upstream's sentinel is ONLY the X component. Do not clear the
            // cache per update or reinterpret a zero gradient as a cached bit.
            if (normalCache[at] === 0) {
              normalCache[at] = field[point - 1] - field[point + 1];
              normalCache[at + 1] = field[point - yd] - field[point + yd];
              normalCache[at + 2] = field[point - zd] - field[point + zd];
            }
          }
          const mu = (isolation - field[a]) / (field[b] - field[a]);
          const at = edge * 3, a3 = a * 3, b3 = b * 3;
          const px = cx === 0 ? fx : fx2;
          const py = cy === 0 ? fy : fy2;
          const pz = cz === 0 ? fz : fz2;
          vlist[at] = axis === 0 ? px + mu * delta : px;
          vlist[at + 1] = axis === 1 ? py + mu * delta : py;
          vlist[at + 2] = axis === 2 ? pz + mu * delta : pz;
          for (let component = 0; component < 3; component++) {
            nlist[at + component] = normalCache[a3 + component] +
              (normalCache[b3 + component] - normalCache[a3 + component]) * mu;
          }
          // The source updates clist even when public vertex colors are off.
          for (let component = 0; component < 3; component++) {
            clist[at + component] = palette[a3 + component] +
              (palette[b3 + component] - palette[a3 + component]) * mu;
          }
        }
        const row = cube << 4;
        for (let entry = 0; triTable[row + entry] !== -1; entry += 3) {
          const o1 = 3 * triTable[row + entry];
          const o2 = 3 * triTable[row + entry + 1];
          const o3 = 3 * triTable[row + entry + 2];
          const at = count * 3;
          for (let component = 0; component < 3; component++) {
            position[at + component] = vlist[o1 + component];
            position[at + 3 + component] = vlist[o2 + component];
            position[at + 6 + component] = vlist[o3 + component];
            if (flatShading === 1) {
              const mean = (nlist[o1 + component] + nlist[o2 + component] + nlist[o3 + component]) / 3;
              normal[at + component] = mean;
              normal[at + 3 + component] = mean;
              normal[at + 6 + component] = mean;
            } else {
              normal[at + component] = nlist[o1 + component];
              normal[at + 3 + component] = nlist[o2 + component];
              normal[at + 6 + component] = nlist[o3 + component];
            }
            if (enableColors === 1) {
              color[at + component] = clist[o1 + component];
              color[at + 3 + component] = clist[o2 + component];
              color[at + 6 + component] = clist[o3 + component];
            }
          }
          if (enableUvs === 1) {
            const atUv = count * 2;
            uv[atUv] = vlist[o1];
            uv[atUv + 1] = vlist[o1 + 2];
            uv[atUv + 2] = vlist[o2];
            uv[atUv + 3] = vlist[o2 + 2];
            uv[atUv + 4] = vlist[o3];
            uv[atUv + 5] = vlist[o3 + 2];
          }
          count += 3;
        }
      }
    }
  }
  return count;
}
