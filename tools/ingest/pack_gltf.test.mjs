import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { embedGltfAssets } from "./pack_gltf.mjs";
import { packHtml } from "./pack_html.mjs";

const fail = (code, message) => {
  throw Object.assign(new Error(message), { code });
};
const embed = (uri) =>
  "data:application/octet-stream;base64," + Buffer.from(uri).toString("base64");
function glb(model, bin = Buffer.from([0, 1, 2, 3])) {
  const json = Buffer.from(JSON.stringify(model)),
    pad = (4 - (json.length % 4)) % 4;
  const header = Buffer.alloc(20);
  header.writeUInt32LE(0x46546c67);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(20 + json.length + pad + 8 + bin.length, 8);
  header.writeUInt32LE(json.length + pad, 12);
  header.writeUInt32LE(0x4e4f534a, 16);
  const binHeader = Buffer.alloc(8);
  binHeader.writeUInt32LE(bin.length);
  binHeader.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([header, json, Buffer.alloc(pad, 32), binHeader, bin]);
}

test("embeds core glTF buffers and images without changing geometry or metadata", () => {
  const model = {
    asset: { version: "2.0" },
    buffers: [{ uri: "data.bin", byteLength: 12 }],
    images: [{ uri: "texture.png" }],
    bufferViews: [{ buffer: 0, byteOffset: 4, byteLength: 8 }],
    accessors: [{ bufferView: 0, count: 2, componentType: 5126, type: "SCALAR" }],
    extras: { uri: "application metadata" },
  };
  const result = JSON.parse(
    embedGltfAssets(Buffer.from(JSON.stringify(model)), false, embed, fail),
  );
  assert.equal(result.buffers[0].uri, embed("data.bin"));
  assert.equal(result.images[0].uri, embed("texture.png"));
  assert.deepEqual(result.bufferViews, model.bufferViews);
  assert.deepEqual(result.accessors, model.accessors);
  assert.deepEqual(result.extras, model.extras);
});

test("GLB preserves exact binary chunk bytes while rebuilding its JSON chunk and total length", () => {
  const model = {
    asset: { version: "2.0" },
    buffers: [{ byteLength: 4 }],
    images: [{ uri: "external.png" }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 4 }],
  };
  const before = glb(model),
    after = embedGltfAssets(before, true, embed, fail);
  assert.equal(after.readUInt32LE(8), after.length);
  assert.equal(after.readUInt32LE(12) % 4, 0);
  assert.deepEqual(
    after.subarray(20 + after.readUInt32LE(12)),
    before.subarray(20 + before.readUInt32LE(12)),
  );
  const json = JSON.parse(after.subarray(20, 20 + after.readUInt32LE(12)).toString());
  assert.equal(json.buffers[0].uri, undefined);
  assert.equal(json.images[0].uri, embed("external.png"));
  assert.deepEqual(json.bufferViews, model.bufferViews);
});

test("already-closed JSON and GLB assets remain byte-identical", () => {
  const model = {
    asset: { version: "2.0" },
    buffers: [{ byteLength: 4 }],
    images: [{ bufferView: 0, mimeType: "image/png" }],
  };
  for (const [bytes, binary] of [
    [Buffer.from(JSON.stringify(model, null, 2)), false],
    [glb(model), true],
  ]) {
    assert.equal(
      embedGltfAssets(
        bytes,
        binary,
        () => {
          throw Error("not needed");
        },
        fail,
      ),
      bytes,
    );
  }
});

test("rejects truncated GLB or unsupported extension URIs instead of silently shipping missing assets", () => {
  const good = glb({ asset: { version: "2.0" } });
  assert.throws(() => embedGltfAssets(good.subarray(0, good.length - 1), true, embed, fail), {
    code: "GLB_HEADER",
  });
  const bad = Buffer.from(good);
  bad.writeUInt32LE(1, 12);
  assert.throws(() => embedGltfAssets(bad, true, embed, fail), { code: "GLB_CHUNK" });
  const extension = Buffer.from(
    JSON.stringify({ asset: { version: "2.0" }, extensions: { CUSTOM: { uri: "sidecar.bin" } } }),
  );
  assert.throws(() => embedGltfAssets(extension, false, embed, fail), {
    code: "GLTF_EXTENSION_RESOURCE",
  });
});

test("single HTML packs a model relative to the model file, including binary and texture bytes", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-model-html-"));
  fs.mkdirSync(path.join(root, "models"));
  fs.writeFileSync(path.join(root, "index.html"), '<script type="module" src="app.mjs"></script>');
  fs.writeFileSync(
    path.join(root, "app.mjs"),
    `export const model=new URL('./models/scene.gltf',import.meta.url);`,
  );
  fs.writeFileSync(
    path.join(root, "models", "scene.gltf"),
    JSON.stringify({
      asset: { version: "2.0" },
      buffers: [{ uri: "geometry.bin", byteLength: 3 }],
      images: [{ uri: "texture.png" }],
    }),
  );
  fs.writeFileSync(path.join(root, "models", "geometry.bin"), Buffer.from([1, 2, 3]));
  fs.writeFileSync(path.join(root, "models", "texture.png"), Buffer.from([4, 5, 6]));
  const out = path.join(root, "single.html"),
    result = packHtml(path.join(root, "index.html"), out);
  assert.equal(result.assetCount, 3);
  const html = fs.readFileSync(out, "utf8"),
    map = JSON.parse(/<script type="importmap">(.*?)<\/script>/.exec(html)[1]);
  const js = Buffer.from(
    Object.values(map.imports)[0].split(",")[1].split("#")[0],
    "base64",
  ).toString();
  const url = /new URL\("([^"]+)"\)/.exec(js)[1],
    model = JSON.parse(Buffer.from(url.split(",")[1], "base64").toString());
  assert.equal(model.buffers[0].uri, "data:application/octet-stream;base64,AQID");
  assert.match(model.images[0].uri, /data:image\/png;f3d-resource=[0-9a-f]+;base64,BAUG/);
});

test("repackaging preserves signed zero in transforms instead of JSON-stringifying it to positive zero", () => {
  const original = Buffer.from(
    '{"asset":{"version":"2.0"},"nodes":[{"translation":[-0,0,1]}],"buffers":[{"uri":"a.bin","byteLength":4}]}',
  );
  const model = JSON.parse(embedGltfAssets(original, false, embed, fail));
  assert.ok(Object.is(model.nodes[0].translation[0], -0));
});
