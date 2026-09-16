/** A complete indexed-mesh update, kept as ordinary JavaScript for differential tests. */
export function recomputeNormals(positions, indices, normals) {
  let twiceArea = 0;
  for (let i = 0; i < normals.length; i++) normals[i] = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ax = positions[a], ay = positions[a + 1], az = positions[a + 2];
    const ux = positions[b] - ax, uy = positions[b + 1] - ay, uz = positions[b + 2] - az;
    const vx = positions[c] - ax, vy = positions[c + 1] - ay, vz = positions[c + 2] - az;
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    twiceArea += Math.sqrt(nx * nx + ny * ny + nz * nz);
    normals[a] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
    normals[b] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
    normals[c] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const x = normals[i], y = normals[i + 1], z = normals[i + 2];
    const length = Math.sqrt(x * x + y * y + z * z);
    if (length > 0) {
      normals[i] = x / length; normals[i + 1] = y / length; normals[i + 2] = z / length;
    }
  }
  return twiceArea / 2;
}
