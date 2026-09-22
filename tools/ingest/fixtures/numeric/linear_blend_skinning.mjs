/**
 * Ordinary CPU linear-blend skinning of bind-space XYZ positions. Each vertex
 * has four joint indices and four weights. The caller supplies final joint
 * matrices (column-major, 16 numbers each); skeleton evaluation, inverse-bind
 * composition, events and rendering remain the application's responsibility.
 */
export function skinPositions(positions, joints, weights, matrices, output) {
  for (let i = 0; i < positions.length; i += 3) {
    const influenceBase = (i / 3) * 4;
    const x = positions[i],
      y = positions[i + 1],
      z = positions[i + 2];
    let resultX = 0,
      resultY = 0,
      resultZ = 0;
    for (let influence = 0; influence < 4; influence++) {
      const weight = weights[influenceBase + influence];
      if (weight !== 0) {
        const base = joints[influenceBase + influence] * 16;
        resultX +=
          weight *
          (matrices[base] * x +
            matrices[base + 4] * y +
            matrices[base + 8] * z +
            matrices[base + 12]);
        resultY +=
          weight *
          (matrices[base + 1] * x +
            matrices[base + 5] * y +
            matrices[base + 9] * z +
            matrices[base + 13]);
        resultZ +=
          weight *
          (matrices[base + 2] * x +
            matrices[base + 6] * y +
            matrices[base + 10] * z +
            matrices[base + 14]);
      }
    }
    output[i] = resultX;
    output[i + 1] = resultY;
    output[i + 2] = resultZ;
  }
}
