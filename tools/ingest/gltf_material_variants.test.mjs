import assert from "node:assert/strict";
import test from "node:test";
import {
  GLTF_MATERIAL_VARIANTS as EXT,
  GltfMaterialVariantError,
  selectGltfMaterialVariant as select,
} from "./gltf_material_variants.mjs";

function asset() {
  return {
    asset: { version: "2.0" },
    extensionsUsed: [EXT, "KHR_materials_unlit"],
    extensionsRequired: [EXT],
    extensions: { [EXT]: { variants: [{ name: "red" }, { name: "blue" }, { name: "plain" }] } },
    materials: [{ name: "default" }, { name: "red paint" }, { name: "blue paint" }],
    meshes: [{ name: "body", primitives: [
      { attributes: { POSITION: 0 }, material: 0, extensions: { [EXT]: { mappings: [
        { material: 1, variants: [0] }, { material: 2, variants: [1] },
      ] } } },
      { attributes: { POSITION: 1 }, extensions: { [EXT]: { mappings: [
        { material: 2, variants: [0, 1] },
      ] } } },
      { attributes: { POSITION: 2 }, material: 1 },
    ] }, { primitives: [{ attributes: { POSITION: 0 }, material: 2 }] }],
    nodes: [{ mesh: 0 }, { mesh: 0 }, { mesh: 1 }],
    scenes: [{ nodes: [0, 1] }, { nodes: [2] }],
    buffers: [{ byteLength: 36, uri: "mesh.bin" }],
    bufferViews: [{ buffer: 0, byteLength: 36 }],
    accessors: [{ bufferView: 0, count: 3, type: "VEC3", componentType: 5126 }],
    images: [{ uri: "red.png" }, { uri: "blue.png" }],
    textures: [{ source: 0 }, { source: 1 }],
  };
}
const primitive = doc => doc.meshes[0].primitives[0];
const mappings = doc => primitive(doc).extensions[EXT].mappings;
const materials = result => result.json.meshes[0].primitives.map(p => p.material);
const code = expected => error => error instanceof GltfMaterialVariantError && error.code === `GLTF_VARIANT_${expected}`;
function deepFreeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const entry of Object.values(value)) deepFreeze(entry);
  }
  return value;
}

test("select by index or exact name with per-primitive fallback", () => {
  const doc = asset();
  assert.deepEqual(materials(select(doc, 0)), [1, 2, 1]);
  assert.deepEqual(materials(select(doc, "blue")), [2, 2, 1]);
  assert.deepEqual(materials(select(doc, "plain")), [0, undefined, 1]);
  assert.deepEqual(materials(select(doc)), [0, undefined, 1]);
  assert.equal(select(doc, 0).variant, 0);
  assert.equal(select(doc, null).variant, null);
});
test("selection never edits frozen authored data or renumbers shared resources", () => {
  const doc = deepFreeze(asset()), before = JSON.stringify(doc);
  const result = select(doc, "blue"), projected = result.json;
  assert.equal(JSON.stringify(doc), before);
  for (const key of ["asset", "materials", "nodes", "scenes", "buffers", "bufferViews", "accessors", "textures", "images"])
    assert.strictEqual(projected[key], doc[key], key);
  assert.notStrictEqual(projected.meshes, doc.meshes);
  assert.notStrictEqual(projected.meshes[0], doc.meshes[0]);
  assert.strictEqual(projected.meshes[1], doc.meshes[1]);
  assert.strictEqual(projected.meshes[0].primitives[2], doc.meshes[0].primitives[2]);
  assert.strictEqual(primitive(projected).attributes, primitive(doc).attributes);
  assert.deepEqual(materials(select(doc, null)), [0, undefined, 1]);
});
test("consume only the resolved extension and preserve unknown requirements", () => {
  const doc = asset();
  doc.extensions.OTHER = { untouched: true };
  doc.extensionsRequired.push("OTHER");
  primitive(doc).extensions.OTHER = { bufferView: 0 };
  const projected = select(doc, 1).json;
  assert.deepEqual(projected.extensions, { OTHER: { untouched: true } });
  assert.deepEqual(projected.extensionsRequired, ["OTHER"]);
  assert.deepEqual(projected.extensionsUsed, ["KHR_materials_unlit"]);
  assert.deepEqual(primitive(projected).extensions, { OTHER: { bufferView: 0 } });
  assert.equal(projected.meshes[0].primitives[1].extensions, undefined);
});
test("remove empty declarations and retain implicit default material", () => {
  const doc = asset(); doc.extensionsUsed = [EXT];
  const projected = select(doc, null).json;
  assert.equal(projected.extensions, undefined);
  assert.equal(projected.extensionsRequired, undefined);
  assert.equal(projected.extensionsUsed, undefined);
  assert.equal(Object.hasOwn(projected.meshes[0].primitives[1], "material"), false);
});
test("plain asset is identity preserving; variants can be defined without meshes", () => {
  const plain = { asset: { version: "2.0" } };
  assert.strictEqual(select(plain).json, plain);
  assert.deepEqual(select(plain).variants, []);
  const doc = asset(); delete doc.meshes;
  assert.equal(select(doc, "red").variant, 0);
});
test("variant list is a frozen snapshot including names that are object property names", () => {
  const doc = asset();
  const entries = doc.extensions[EXT].variants;
  entries[0].name = "__proto__"; entries[1].name = "constructor";
  const result = select(doc, "__proto__");
  entries[0].name = "changed";
  assert.deepEqual(result.variants[0], { index: 0, name: "__proto__" });
  assert.ok(Object.isFrozen(result) && Object.isFrozen(result.variants) && Object.isFrozen(result.variants[0]));
  assert.equal(select(doc, "constructor").variant, 1);
});
test("duplicate names require an index, never select an arbitrary material", () => {
  const doc = asset(); doc.extensions[EXT].variants[1].name = "red";
  assert.throws(() => select(doc, "red"), code("AMBIGUOUS"));
  assert.equal(select(doc, 1).variant, 1);
  assert.equal(select(doc, null).variant, null);
});
test("empty names are legal and selection is case sensitive", () => {
  const doc = asset(); doc.extensions[EXT].variants[0].name = "";
  assert.equal(select(doc, "").variant, 0);
  assert.throws(() => select(doc, "Blue"), code("SELECTION"));
});
test("same variant may map independently on different primitives and meshes", () => {
  const doc = asset();
  doc.meshes[1].primitives[0].extensions = { [EXT]: { mappings: [{ material: 0, variants: [1] }] } };
  const result = select(doc, 1);
  assert.equal(result.json.meshes[1].primitives[0].material, 0);
  assert.deepEqual(materials(result), [2, 2, 1]);
});

for (const [name, mutate, expected] of [
  ["empty variants", d => { d.extensions[EXT].variants = []; }, "SHAPE"],
  ["non-object root", d => { d.extensions[EXT] = null; }, "SHAPE"],
  ["missing name", d => { delete d.extensions[EXT].variants[0].name; }, "SHAPE"],
  ["sparse variants", d => { delete d.extensions[EXT].variants[0]; }, "SHAPE"],
  ["non-string name", d => { d.extensions[EXT].variants[0].name = 1; }, "SHAPE"],
  ["empty mappings", d => { primitive(d).extensions[EXT].mappings = []; }, "SHAPE"],
  ["null mapping", d => { mappings(d)[0] = null; }, "SHAPE"],
  ["empty mapped variants", d => { mappings(d)[0].variants = []; }, "SHAPE"],
  ["duplicate within mapping", d => { mappings(d)[0].variants = [0, 0]; }, "DUPLICATE"],
  ["duplicate across mappings", d => { mappings(d)[1].variants = [0]; }, "DUPLICATE"],
  ["out of range unselected variant", d => { mappings(d)[1].variants = [3]; }, "INDEX"],
  ["noninteger mapped variant", d => { mappings(d)[1].variants = [0.5]; }, "INDEX"],
  ["negative mapped variant", d => { mappings(d)[1].variants = [-1]; }, "INDEX"],
  ["out of range unselected material", d => { mappings(d)[1].material = 3; }, "INDEX"],
  ["missing mapped material", d => { delete mappings(d)[1].material; }, "INDEX"],
  ["null mapped material", d => { d.materials[2] = null; }, "SHAPE"],
  ["invalid authored default", d => { primitive(d).material = 10; }, "INDEX"],
  ["unknown mapping field", d => { mappings(d)[0].unexpected = true; }, "UNSUPPORTED"],
  ["unknown variant semantics", d => { d.extensions[EXT].variants[0].extensions = { OTHER: {} }; }, "UNSUPPORTED"],
  ["orphan mappings", d => { delete d.extensions; delete d.extensionsUsed; delete d.extensionsRequired; }, "ROOT"],
  ["orphan declaration", d => { delete d.extensions; delete d.meshes; }, "ROOT"],
  ["invalid declarations", d => { d.extensionsUsed = [EXT, 1]; }, "SHAPE"],
  ["null primitive", d => { d.meshes[0].primitives[1] = null; }, "SHAPE"],
]) {
  test(`reject ${name} before publishing even when selecting defaults`, () => {
    const doc = asset(); mutate(doc); const before = JSON.stringify(doc);
    assert.throws(() => select(doc, null), code(expected));
    assert.equal(JSON.stringify(doc), before);
  });
}
for (const selection of [-1, 0.5, Infinity, NaN, true, {}, [], 9007199254740992]) {
  test(`reject invalid selection ${String(selection)}`, () => {
    assert.throws(() => select(asset(), selection), code("SELECTION"));
  });
}
test("reject unknown names and out of range indices, including assets with no variants", () => {
  assert.throws(() => select(asset(), "missing"), code("SELECTION"));
  assert.throws(() => select(asset(), 3), code("INDEX"));
  assert.throws(() => select({}, 0), code("INDEX"));
  assert.throws(() => select({}, "red"), code("SELECTION"));
});
for (const [key, value] of [["maxVariants", 2], ["maxMeshes", 1], ["maxPrimitives", 3], ["maxMappings", 2], ["maxAssignments", 3]]) {
  test(`enforce aggregate ${key} before returning a projection`, () => {
    assert.throws(() => select(asset(), 0, { [key]: value }), code("LIMIT"));
  });
}
test("exact budgets succeed and malformed limits fail explicitly", () => {
  assert.equal(select(asset(), 0, { maxVariants: 3, maxMeshes: 2, maxPrimitives: 4, maxMappings: 3, maxAssignments: 4 }).variant, 0);
  for (const value of [0, -1, NaN, Infinity, 1.5, "10", null])
    assert.throws(() => select(asset(), 0, { maxVariants: value }), code("LIMIT"));
  assert.throws(() => select(asset(), 0, { maxVarants: 5 }), code("LIMIT"));
});
