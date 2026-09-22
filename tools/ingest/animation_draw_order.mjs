/**
 * Scene-level alpha ordering and optional conservative frustum rejection, not
 * triangle sorting or a Three.js RenderList. Explicit draw lists bypass this.
 * Opaque/masked draws retain source order; blended draws sort by projected node
 * origin, with source-order ties. Culling never reorders surviving draws.
 */
import { animationBoundsVisible, createAnimationBounds } from "./animation_bounds.mjs";
export class AnimationDrawOrderError extends Error {
  constructor(message) {
    super(`ANIMATION_SCENE_SORT: ${message}`);
    this.name = "AnimationDrawOrderError";
    this.code = "ANIMATION_SCENE_SORT";
  }
}
const fail = (message) => {
  throw new AnimationDrawOrderError(message);
};
function matrix(value) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== 16) {
    fail("Expected a 16-component matrix");
  }
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable)
      fail("Matrices require fixed unshared storage");
    try {
      new Uint8Array(value.buffer, 0, 0);
    } catch {
      fail("Detached matrix storage");
    }
  }
  const result = new Float64Array(16);
  for (let i = 0; i < 16; i++) {
    const item = value[i];
    if (typeof item !== "number" || !Number.isFinite(item))
      fail("Matrix values must be finite numbers");
    result[i] = item;
  }
  return result;
}

/** Entries borrow mesh/deformer identity; alpha mode and insertion order freeze.
 * With frustumCulling, entries also supply source geometry during construction.
 * Only compact bounds summaries survive construction. Bounds and draw resources
 * must describe the SAME uploaded pose. The caller owns meshes/deformers/pose.
 */
export function createAnimationDrawOrder(
  entries,
  {
    pose = null,
    sortObjects = true,
    frustumCulling = false,
    maxBoundsBytes = 16 * 1024 * 1024,
    maxBoundsComponents = 16777216,
  } = {},
) {
  if (!Array.isArray(entries) || entries.length > 4096) fail("Invalid scene draw count");
  if (typeof sortObjects !== "boolean" || typeof frustumCulling !== "boolean")
    fail("Sorting and culling options must be boolean");
  if (
    frustumCulling &&
    (!pose ||
      !Number.isSafeInteger(maxBoundsBytes) ||
      maxBoundsBytes < 1 ||
      !Number.isSafeInteger(maxBoundsComponents) ||
      maxBoundsComponents < 1)
  )
    fail("Invalid bounds limits or pose");
  let bytes = 0,
    components = 0,
    disposed = false,
    busy = false,
    stats = null,
    cameraSnapshot = null;
  const records = [];
  try {
    for (const [index, entry] of entries.entries()) {
      if (
        !entry?.mesh ||
        !entry.deformer ||
        !["OPAQUE", "MASK", "BLEND"].includes(entry.alphaMode)
      ) {
        fail("Expected a mesh, deformer and alpha mode");
      }
      let bounds = null;
      if (frustumCulling) {
        if (bytes >= maxBoundsBytes || components >= maxBoundsComponents)
          fail("Aggregate bounds limit exceeded");
        bounds = createAnimationBounds(pose, entry.geometry, {
          maxBytes: maxBoundsBytes - bytes,
          maxComponents: maxBoundsComponents - components,
        });
        bytes += bounds.byteLength;
        components += bounds.sourceComponents;
      }
      records.push({
        mesh: entry.mesh,
        deformer: entry.deformer,
        alphaMode: entry.alphaMode,
        index,
        bounds,
      });
    }
  } catch (error) {
    for (const r of records) r.bounds?.dispose();
    throw error;
  }
  function live() {
    if (disposed) fail("Draw ordering has been disposed");
  }
  function synchronized(record) {
    if (
      pose.disposed ||
      record.deformer.disposed ||
      record.deformer.failed ||
      record.deformer.poseVersion !== pose.version ||
      record.bounds.snapshot.poseVersion !== pose.version
    )
      fail("Bounds and geometry need the same uploaded pose");
  }
  return Object.freeze({
    get boundsBytes() {
      return disposed ? 0 : bytes;
    },
    get lastCulling() {
      return stats;
    },
    get viewProjection() {
      return cameraSnapshot;
    },
    updateBounds() {
      live();
      if (busy) fail("Cannot update bounds during ordering");
      for (const record of records)
        if (record.bounds) {
          record.bounds.update();
          synchronized(record);
        }
    },
    // drawIndices selects source entries before culling/sorting, never sort order.
    // Inactive LODs cannot affect camera admission, bounds tests or alpha order.
    order(viewProjection, drawIndices = null) {
      live();
      if (busy) fail("Draw ordering cannot be reentered");
      busy = true;
      try {
        let active = records;
        if (drawIndices !== null) {
          if (!Array.isArray(drawIndices) || drawIndices.length > records.length)
            fail("Invalid source draw selection");
          const selected = new Set();
          for (const index of drawIndices) {
            if (
              !Number.isSafeInteger(index) ||
              index < 0 ||
              index >= records.length ||
              selected.has(index)
            )
              fail("Invalid or duplicate source draw index");
            selected.add(index);
          }
          active = records.filter((record) => selected.has(record.index));
        }
        const version = frustumCulling ? pose.version : null;
        const camera =
          frustumCulling || (sortObjects && active.some((r) => r.alphaMode === "BLEND"))
            ? matrix(viewProjection)
            : null;
        const visible = frustumCulling
          ? active.filter((record) => {
              synchronized(record);
              return animationBoundsVisible(
                record.bounds.snapshot,
                camera,
                record.deformer.worldMatrix,
              );
            })
          : active;
        let result;
        if (!sortObjects) result = visible.map((record) => record.mesh);
        else {
          const opaque = visible
            .filter((record) => record.alphaMode !== "BLEND")
            .map((record) => record.mesh);
          const sorted = visible
            .filter((record) => record.alphaMode === "BLEND")
            .map((record) => {
              const world = matrix(record.deformer.worldMatrix);
              if (world[3] !== 0 || world[7] !== 0 || world[11] !== 0 || world[15] !== 1)
                fail("World matrix must be affine");
              const x = world[12],
                y = world[13],
                z = world[14];
              const clipZ = camera[2] * x + camera[6] * y + camera[10] * z + camera[14];
              const clipW = camera[3] * x + camera[7] * y + camera[11] * z + camera[15];
              if (!Number.isFinite(clipZ) || !Number.isFinite(clipW))
                fail("Projected sort position overflowed");
              const depth = clipW === 0 ? Math.sign(clipZ) * Infinity : clipZ / clipW;
              return { ...record, depth: Number.isNaN(depth) ? 0 : depth };
            });
          sorted.sort((a, b) =>
            a.depth === b.depth ? a.index - b.index : a.depth > b.depth ? -1 : 1,
          );
          result = opaque.concat(sorted.map((record) => record.mesh));
        }
        if (frustumCulling) {
          if (pose.version !== version) fail("Pose changed during culling");
          cameraSnapshot = camera;
          stats = Object.freeze({
            poseVersion: version,
            testedMeshes: active.length,
            culledMeshes: active.length - visible.length,
            submittedDraws: visible.length,
          });
        }
        return result;
      } finally {
        busy = false;
      }
    },
    dispose() {
      if (busy) fail("Cannot dispose during ordering");
      if (!disposed) {
        for (const record of records) record.bounds?.dispose();
        disposed = true;
        cameraSnapshot = null;
        stats = null;
      }
    },
  });
}
