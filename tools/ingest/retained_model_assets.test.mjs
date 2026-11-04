import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { buildApplication } from "./build_application.mjs";
import { packHtml } from "./pack_html.mjs";

const html = '<!doctype html><script>globalThis.loadModels = () => import("./scripts/models.mjs");</script>';

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-retained-models-"));
  const source = path.join(root, "source");
  for (const [name, bytes] of Object.entries({ "index.html": html, ...files })) {
    const file = path.join(source, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, bytes);
  }
  return { root, source, entry: path.join(source, "index.html"), out: path.join(root, "dist") };
}

function glb(model, bin) {
  const json = Buffer.from(JSON.stringify(model));
  const jsonLength = Math.ceil(json.length / 4) * 4;
  const bytes = Buffer.alloc(20 + jsonLength + 8 + bin.length);
  bytes.writeUInt32LE(0x46546c67, 0);
  bytes.writeUInt32LE(2, 4);
  bytes.writeUInt32LE(bytes.length, 8);
  bytes.writeUInt32LE(jsonLength, 12);
  bytes.writeUInt32LE(0x4e4f534a, 16);
  bytes.fill(0x20, 20, 20 + jsonLength);
  json.copy(bytes, 20);
  bytes.writeUInt32LE(bin.length, 20 + jsonLength);
  bytes.writeUInt32LE(0x004e4942, 24 + jsonLength);
  bin.copy(bytes, 28 + jsonLength);
  return bytes;
}

function modelFiles() {
  const positions = Buffer.from(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer);
  const image = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1sAAAAASUVORK5CYII=", "base64");
  const binary = Buffer.from([4, 3, 2, 1]);
  const model = {
    asset: { version: "2.0" },
    buffers: [{ uri: "../shared/geometry.bin?version=1#positions", byteLength: positions.length }],
    bufferViews: [{ buffer: 0, byteLength: positions.length }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
    images: [{ uri: "../textures/base%20%23.png?version=1#pixels" }],
    textures: [{ source: 0 }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }], scene: 0,
    extras: { label: "Retained source model" },
  };
  return {
    "scripts/models.mjs": `const json = '../models/scene.gltf?variant=json#mesh';
      const binary = '../models/scene.glb?variant=binary#mesh';
      export const select = isBinary => new URL(isBinary ? binary : json, import.meta.url);`,
    "models/scene.gltf": JSON.stringify(model, null, 2) + "\n",
    "models/scene.glb": glb({
      asset: { version: "2.0" },
      buffers: [{ byteLength: binary.length }, { uri: "../shared/external.bin?version=2#data", byteLength: 3 }],
      images: [{ uri: "../textures/base%20%23.png?version=1#pixels" }],
    }, binary),
    "shared/geometry.bin": positions,
    "shared/external.bin": Buffer.from([5, 6, 7]),
    "textures/base #.png": image,
  };
}

async function relocate(f) {
  const built = await buildApplication(f.entry, f.out);
  assert.deepEqual(built.entryFiles, [], "A classic dynamic import retains the original module");
  const deployed = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "f3d-retained-models-deployed-")), "app");
  fs.renameSync(f.out, deployed);
  fs.renameSync(f.source, path.join(f.root, "source-unavailable"));
  return { built, deployed };
}

test("retained glTF and GLB imports carry model-relative buffers and images after relocation", async () => {
  const files = modelFiles();
  const f = fixture(files);
  const { built, deployed } = await relocate(f);
  for (const [name, bytes] of Object.entries(files)) {
    assert.ok(built.emittedFiles.includes(name), `${name} is emitted`);
    assert.deepEqual(fs.readFileSync(path.join(deployed, name)), Buffer.from(bytes), `${name} stays byte-identical`);
  }
  const module = await import(pathToFileURL(path.join(deployed, "scripts/models.mjs")));
  for (const binary of [false, true]) {
    const url = module.select(binary);
    assert.ok(url instanceof URL);
    assert.equal(url.search, binary ? "?variant=binary" : "?variant=json");
    assert.equal(url.hash, "#mesh");
    const bytes = fs.readFileSync(url);
    const model = binary
      ? JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString("utf8"))
      : JSON.parse(bytes.toString("utf8"));
    const buffer = model.buffers.find(resource => resource.uri);
    assert.deepEqual(fs.readFileSync(new URL(buffer.uri, url)), files[binary ? "shared/external.bin" : "shared/geometry.bin"]);
    assert.deepEqual(fs.readFileSync(new URL(model.images[0].uri, url)), files["textures/base #.png"]);
  }
});

test("standalone packing closes relocated retained models with no original source or build directory", async () => {
  const files = modelFiles();
  const f = fixture(files);
  const { deployed } = await relocate(f);
  const packedFile = path.join(f.root, "models.html");
  packHtml(path.join(deployed, "index.html"), packedFile);
  const packed = fs.readFileSync(packedFile, "utf8");
  fs.renameSync(deployed, deployed + "-unavailable");
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(packed)[1]).imports;
  assert.equal(Object.keys(imports).length, 1);
  const module = await import(Object.values(imports)[0]);
  const readData = async uri => {
    assert.ok(uri.startsWith("data:"), "Packed dependencies cannot use an external source");
    return Buffer.from(await (await fetch(uri)).arrayBuffer());
  };
  for (const binary of [false, true]) {
    const bytes = await readData(module.select(binary).href);
    const jsonEnd = binary ? 20 + bytes.readUInt32LE(12) : bytes.length;
    const model = JSON.parse(bytes.subarray(binary ? 20 : 0, jsonEnd).toString("utf8"));
    const buffer = model.buffers.find(resource => resource.uri);
    assert.deepEqual(await readData(buffer.uri), files[binary ? "shared/external.bin" : "shared/geometry.bin"]);
    assert.deepEqual(await readData(model.images[0].uri), files["textures/base #.png"]);
    if (binary) {
      const original = files["models/scene.glb"];
      assert.equal(bytes.readUInt32LE(8), bytes.length, "Repacked GLB length remains valid");
      assert.deepEqual(bytes.subarray(jsonEnd), original.subarray(20 + original.readUInt32LE(12)), "BIN chunk stays exact");
    } else {
      const original = JSON.parse(files["models/scene.gltf"]);
      for (const key of ["bufferViews", "accessors", "meshes", "nodes", "scenes", "extras"])
        assert.deepEqual(model[key], original[key], key);
    }
  }
});

test("model sidecars remain opaque bytes even when their names have source or model extensions", async () => {
  const files = {
    "scripts/models.mjs": "export const url = new URL('../model.gltf', import.meta.url);",
    "model.gltf": JSON.stringify({asset: {version: "2.0"}, buffers: [
      {uri: "payload.js", byteLength: 3}, {uri: "payload.gltf", byteLength: 2},
    ], images: [{uri: "image.css"}]}),
    "payload.js": Buffer.from([0xff, 0, 0xfe]),
    "payload.gltf": Buffer.from([0, 0xff]),
    "image.css": "url(must-not-be-read.bin)",
  };
  const f = fixture(files);
  await buildApplication(f.entry, f.out);
  for (const [name, bytes] of Object.entries(files))
    assert.deepEqual(fs.readFileSync(path.join(f.out, name)), Buffer.from(bytes), name);
});

test("retained model resources preserve authored live and embedded URI routes", async () => {
  const model = JSON.stringify({asset: {version: "2.0"}, buffers: [
    {uri: "data:application/octet-stream;base64,AQID", byteLength: 3},
    {uri: "https://example.invalid/live.bin", byteLength: 4},
    {uri: "//example.invalid/shared.bin", byteLength: 4},
    {uri: "/shared/root.bin", byteLength: 4},
  ]});
  const f = fixture({
    "scripts/models.mjs": "export const url = new URL('../model.gltf', import.meta.url);",
    "model.gltf": model,
  });
  const result = await buildApplication(f.entry, f.out);
  assert.deepEqual(new Set(result.emittedFiles), new Set(["index.html", "scripts/models.mjs", "model.gltf"]));
  assert.equal(fs.readFileSync(path.join(f.out, "model.gltf"), "utf8"), model);
});

test("invalid models, missing sidecars and root escapes fail before publishing retained applications", async () => {
  const cases = [
    ["missing buffer", {asset: {version: "2.0"}, buffers: [{uri: "missing.bin", byteLength: 4}]}, /Unresolved relative resource.*missing\.bin/],
    ["missing image", {asset: {version: "2.0"}, images: [{uri: "missing.png"}]}, /Unresolved relative resource.*missing\.png/],
    ["escaping sidecar", {asset: {version: "2.0"}, buffers: [{uri: "../outside.bin", byteLength: 4}]}, /escapes application root directory/],
    ["invalid URI", {asset: {version: "2.0"}, images: [{uri: 7}]}, /GLTF_RESOURCE.*URI/],
    ["invalid JSON", "{", /GLTF_JSON/],
  ];
  for (const [label, model, error] of cases) {
    const f = fixture({
      "scripts/models.mjs": "export const url = new URL('../model.gltf', import.meta.url);",
      "model.gltf": typeof model === "string" ? model : JSON.stringify(model),
    });
    await assert.rejects(buildApplication(f.entry, f.out), error, label);
    assert.equal(fs.existsSync(f.out), false, `${label}: no partial application`);
  }
  const f = fixture({
    "scripts/models.mjs": "export const url = new URL('../model.glb', import.meta.url);",
    "model.glb": Buffer.from([1, 2, 3]),
  });
  await assert.rejects(buildApplication(f.entry, f.out), /GLB_HEADER/);
  assert.equal(fs.existsSync(f.out), false);
});
