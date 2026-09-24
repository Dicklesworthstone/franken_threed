/**
 * Resolve one KHR_materials_variants choice before existing glTF decoding.
 * This is a copy-on-write projection, not a renderer or an asset loader. The
 * authored document is never changed. Geometry, node, material, texture and
 * buffer indices remain stable; only primitive material references change.
 *
 * selectGltfMaterialVariant(json, null) selects the authored/default materials.
 * An integer selects a variant index; a string selects an exact, unique name.
 * Always select from the authored document, not a previous projection. Returned
 * JSON shares untouched source objects: keep both stable during consumption.
 * Selection consumes only KHR_materials_variants. Other extensions/requirements
 * remain intact for the existing decoder to accept or reject normally.
 *
 * https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_variants
 */
export const GLTF_MATERIAL_VARIANTS = "KHR_materials_variants";

export class GltfMaterialVariantError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "GltfMaterialVariantError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new GltfMaterialVariantError("GLTF_VARIANT_" + code, message);
};
const object = (value, label) => {
  if (!value || typeof value !== "object" || Array.isArray(value))
    fail("SHAPE", `Expected ${label} object`);
  return value;
};
const array = (value, label, min = 0) => {
  if (!Array.isArray(value) || value.length < min) fail("SHAPE", `Invalid ${label} array`);
  return value;
};
const index = (value, length, label) => {
  if (!Number.isSafeInteger(value) || value < 0 || value >= length)
    fail("INDEX", `Invalid ${label} index`);
  return value;
};
function fields(value, allowed, label) {
  object(value, label);
  for (const key of Object.keys(value))
    if (!allowed.includes(key) && key !== "extras" && key !== "extensions")
      fail("UNSUPPORTED", `Unsupported ${label} field: ${key}`);
  if (value.extensions !== undefined && Object.keys(object(value.extensions, "extensions")).length)
    fail("UNSUPPORTED", `Extended ${label} requires the source route`);
}
const EMPTY = Object.freeze([]);

/** Bounded, synchronous selection; validates ALL variant mappings before a
 * result can reach a buffer provider, image resolver or GPU constructor.
 * Unselected materials are not decoded and their images need not be loaded.
 */
export function selectGltfMaterialVariant(json, selection = null, options = {}) {
  object(json, "glTF");
  object(options, "variant limits");
  const limits = {
    maxVariants: 4096,
    maxMappings: 65536,
    maxAssignments: 1048576,
    maxPrimitives: 65536,
    maxMeshes: 65536,
    ...options,
  };
  for (const [key, value] of Object.entries(limits)) {
    if (!["maxVariants", "maxMappings", "maxAssignments", "maxPrimitives", "maxMeshes"].includes(key))
      fail("LIMIT", `Unknown variant limit: ${key}`);
    if (!Number.isSafeInteger(value) || value < 1)
      fail("LIMIT", `Invalid variant limit: ${key}`);
  }
  if (selection !== null && typeof selection !== "string" &&
      (!Number.isSafeInteger(selection) || selection < 0))
    fail("SELECTION", "Select null, a nonnegative variant index, or an exact name");

  const extension = GLTF_MATERIAL_VARIANTS;
  const rootExtensions = json.extensions === undefined ? {} : object(json.extensions, "extensions");
  const hasRoot = Object.hasOwn(rootExtensions, extension);
  const declarations = {};
  for (const key of ["extensionsUsed", "extensionsRequired"]) {
    if (json[key] !== undefined) {
      const values = array(json[key], key);
      if (Array.from(values).some(value => typeof value !== "string")) fail("SHAPE", `Invalid ${key}`);
      declarations[key] = values;
      if (!hasRoot && values.includes(extension)) fail("ROOT", "Declared variants need a root definition");
    }
  }
  let variants = EMPTY;
  if (hasRoot) {
    const root = rootExtensions[extension];
    fields(root, ["variants"], "material variants");
    const entries = array(root.variants, "variants", 1);
    if (entries.length > limits.maxVariants) fail("LIMIT", "Too many material variants");
    variants = Object.freeze(Array.from(entries, (entry, i) => {
      fields(entry, ["name"], "variant");
      if (typeof entry.name !== "string") fail("SHAPE", "Variant name must be a string");
      return Object.freeze({ index: i, name: entry.name });
    }));
  }
  let variant = null;
  if (typeof selection === "number") variant = index(selection, variants.length, "variant");
  else if (typeof selection === "string") {
    for (const entry of variants) {
      if (entry.name !== selection) continue;
      if (variant !== null) fail("AMBIGUOUS", "Variant name is ambiguous; select by index");
      variant = entry.index;
    }
    if (variant === null) fail("SELECTION", "Unknown material variant name");
  }

  const meshes = json.meshes === undefined ? [] : array(json.meshes, "meshes");
  if (meshes.length > limits.maxMeshes) fail("LIMIT", "Too many meshes for variant selection");
  let projectedMeshes = meshes, primitiveCount = 0, mappingCount = 0, assignmentCount = 0;
  for (let m = 0; m < meshes.length; m++) {
    const mesh = object(meshes[m], "mesh");
    const primitives = array(mesh.primitives, "primitives", 1);
    if (primitives.length > limits.maxPrimitives - primitiveCount)
      fail("LIMIT", "Too many primitives for variant selection");
    primitiveCount += primitives.length;
    let projectedPrimitives = primitives;
    for (let p = 0; p < primitives.length; p++) {
      const primitive = object(primitives[p], "primitive");
      const extensions = primitive.extensions === undefined ? {} : object(primitive.extensions, "primitive extensions");
      if (!Object.hasOwn(extensions, extension)) continue;
      if (!hasRoot) fail("ROOT", "Primitive variants need a root definition");
      const definition = extensions[extension];
      fields(definition, ["mappings"], "primitive variants");
      const mappings = array(definition.mappings, "variant mappings", 1);
      if (mappings.length > limits.maxMappings - mappingCount) fail("LIMIT", "Too many variant mappings");
      mappingCount += mappings.length;
      const materials = array(json.materials, "materials");
      if (primitive.material !== undefined)
        object(materials[index(primitive.material, materials.length, "default material")], "material");
      const assigned = new Set();
      let selectedMaterial = primitive.material;
      for (const mapping of mappings) {
        fields(mapping, ["material", "variants"], "variant mapping");
        const material = index(mapping.material, materials.length, "mapped material");
        object(materials[material], "material");
        const choices = array(mapping.variants, "mapped variants", 1);
        if (choices.length > limits.maxAssignments - assignmentCount)
          fail("LIMIT", "Too many variant assignments");
        assignmentCount += choices.length;
        for (const choice of choices) {
          index(choice, variants.length, "mapped variant");
          if (assigned.has(choice)) fail("DUPLICATE", "A primitive maps a variant more than once");
          assigned.add(choice);
          if (choice === variant) selectedMaterial = material;
        }
      }
      const copy = { ...primitive, extensions: { ...extensions } };
      delete copy.extensions[extension];
      if (!Object.keys(copy.extensions).length) delete copy.extensions;
      // Missing default material means glTF's implicit default, not material 0.
      if (selectedMaterial !== undefined) copy.material = selectedMaterial;
      if (projectedPrimitives === primitives) projectedPrimitives = primitives.slice();
      projectedPrimitives[p] = copy;
    }
    if (projectedPrimitives !== primitives) {
      if (projectedMeshes === meshes) projectedMeshes = meshes.slice();
      projectedMeshes[m] = { ...mesh, primitives: projectedPrimitives };
    }
  }
  if (!hasRoot) return Object.freeze({ json, variants, variant });
  const projected = { ...json, extensions: { ...rootExtensions } };
  delete projected.extensions[extension];
  if (!Object.keys(projected.extensions).length) delete projected.extensions;
  if (projectedMeshes !== meshes) projected.meshes = projectedMeshes;
  for (const [key, values] of Object.entries(declarations)) {
    const remaining = values.filter(value => value !== extension);
    if (remaining.length) projected[key] = remaining;
    else delete projected[key];
  }
  return Object.freeze({ json: projected, variants, variant });
}
