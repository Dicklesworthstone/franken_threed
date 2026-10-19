import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildApplication } from "./build_application.mjs";
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

function buildFixture(files, module) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-build-model-"));
  for (const [filename, bytes] of Object.entries({
    "package.json": '{"type":"module"}',
    "index.html": '<script type="module" src="app.mjs"></script>',
    "app.mjs": module,
    ...files,
  })) {
    fs.mkdirSync(path.dirname(path.join(root, filename)), { recursive: true });
    fs.writeFileSync(path.join(root, filename), bytes);
  }
  return root;
}

async function buildAndImport(root, destination = "dist") {
  const built = await buildApplication(path.join(root, "index.html"), path.join(root, destination));
  const module = await import(pathToFileURL(path.join(built.outDir, built.entryFiles[0])).href);
  return { built, module };
}

test("application build preserves glTF resource bases, encoded filenames and URL suffixes", async () => {
  const source = Buffer.from('{"asset":{"version":"2.0"},"nodes":[{"translation":[-0,0,1]}],' +
    '"buffers":[{"uri":"../buffers/geometry%20%231.bin?v=2#mesh","byteLength":4}],' +
    '"images":[{"uri":"../../textures/wood%20%25%23.png?filter=nearest#sample"}]}');
  const buffer = Buffer.from([1, 2, 3, 4]), image = Buffer.from([9, 8, 7]);
  const root = buildFixture({
    "models/nested/scene #1.gltf": source,
    "models/buffers/geometry #1.bin": buffer,
    "textures/wood %#.png": image,
  }, 'export const model = new URL("./models/nested/scene%20%231.gltf?quality=full#scene", import.meta.url);');
  const { built, module } = await buildAndImport(root);
  assert.equal(module.model.search, "?quality=full");
  assert.equal(module.model.hash, "#scene");
  assert.ok(fileURLToPath(module.model).startsWith(built.outDir + path.sep));
  assert.deepEqual(fs.readFileSync(module.model), source, "Networked model JSON stays byte-identical");
  const model = JSON.parse(fs.readFileSync(module.model));
  assert.ok(Object.is(model.nodes[0].translation[0], -0));
  for (const [uri, expected] of [[model.buffers[0].uri, buffer], [model.images[0].uri, image]]) {
    const emittedUrl = new URL(uri, module.model);
    assert.ok(fileURLToPath(emittedUrl).startsWith(built.outDir + path.sep));
    assert.deepEqual(fs.readFileSync(emittedUrl), expected, `Load actual emitted dependency ${uri}`);
  }
});

test("built glTF and GLB applications close through the standalone packer with their real resources", async () => {
  for (const binary of [false, true]) {
    const extension = binary ? "glb" : "gltf";
    const model = { asset: { version: "2.0" },
      buffers: [binary ? { byteLength: 4 } : { uri: "../geometry.bin", byteLength: 4 }],
      images: [{ uri: "images/pixel.png?v=1#image" }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 4 }],
    };
    const buffer = Buffer.from([1, 2, 3, 4]), image = Buffer.from([9, 8, 7]);
    const source = binary ? glb(model, buffer) : Buffer.from(JSON.stringify(model));
    const root = buildFixture({
      ["models/scene." + extension]: source,
      "geometry.bin": buffer,
      "models/images/pixel.png": image,
    }, `export const model = new URL("./models/scene.${extension}", import.meta.url);`);
    const { built, module } = await buildAndImport(root);
    assert.deepEqual(fs.readFileSync(module.model), source, "Original model and GLB chunks are retained exactly");
    assert.deepEqual(fs.readFileSync(new URL(model.images[0].uri, module.model)), image);
    if (!binary) assert.deepEqual(fs.readFileSync(new URL(model.buffers[0].uri, module.model)), buffer);

    const packedPath = path.join(root, "single.html");
    packHtml(path.join(built.outDir, built.htmlFile), packedPath);
    const html = fs.readFileSync(packedPath, "utf8");
    const map = JSON.parse(/<script type="importmap">(.*?)<\/script>/.exec(html)[1]);
    const packed = await import(Object.values(map.imports)[0]);
    const embeddedBytes = Buffer.from(await (await fetch(packed.model)).arrayBuffer());
    const embedded = JSON.parse(binary
      ? embeddedBytes.subarray(20, 20 + embeddedBytes.readUInt32LE(12)) : embeddedBytes);
    assert.deepEqual(Buffer.from(await (await fetch(embedded.images[0].uri)).arrayBuffer()), image);
    if (binary) {
      assert.deepEqual(embeddedBytes.subarray(20 + embeddedBytes.readUInt32LE(12)),
        source.subarray(20 + source.readUInt32LE(12)), "Packing preserves the original GLB BIN chunk");
    } else assert.deepEqual(Buffer.from(await (await fetch(embedded.buffers[0].uri)).arrayBuffer()), buffer);
  }
});

test("identical glTF JSON in separate directories retains the right sidecars and changed content changes its URL", async () => {
  const source = JSON.stringify({ asset: { version: "2.0" }, buffers: [{ uri: "geometry.bin", byteLength: 4 }] });
  const first = Buffer.from([1, 2, 3, 4]), second = Buffer.from([5, 6, 7, 8]);
  const root = buildFixture({
    "first/scene.gltf": source, "first/geometry.bin": first,
    "second/scene.gltf": source, "second/geometry.bin": second,
  }, 'export const first = new URL("./first/scene.gltf", import.meta.url);\n' +
    'export const second = new URL("./second/scene.gltf", import.meta.url);');
  const { built, module } = await buildAndImport(root);
  assert.notEqual(module.first.href, module.second.href);
  assert.deepEqual(fs.readFileSync(new URL("geometry.bin", module.first)), first);
  assert.deepEqual(fs.readFileSync(new URL("geometry.bin", module.second)), second);
  const originalPath = path.relative(built.outDir, fileURLToPath(module.first));
  fs.writeFileSync(path.join(root, "first/geometry.bin"), second);
  const rebuilt = await buildAndImport(root, "changed");
  const changedPath = path.relative(rebuilt.built.outDir, fileURLToPath(rebuilt.module.first));
  assert.notEqual(changedPath, originalPath, "Sidecar bytes are part of the model's output URL hash");
  assert.deepEqual(fs.readFileSync(new URL("geometry.bin", rebuilt.module.first)), second);
});

test("networked model builds retain external, data, root-relative and extension URI routes", async () => {
  const source = Buffer.from(JSON.stringify({ asset: { version: "2.0" },
    buffers: [{ uri: "https://cdn.example.invalid/geometry.bin", byteLength: 4 }],
    images: [{ uri: "//cdn.example.invalid/image.png" }, { uri: "/shared/image.png" },
      { uri: "data:image/png;base64,AQID" }],
    extensions: { CUSTOM: { uri: "custom-runtime-resource" } },
  }));
  const root = buildFixture({ "scene.gltf": source },
    'export const model = new URL("./scene.gltf", import.meta.url);');
  const { module } = await buildAndImport(root);
  assert.deepEqual(fs.readFileSync(module.model), source, "No remote fetching, metadata rewriting or fake local files");
});

test("missing glTF sidecars fail before publishing an incomplete application", async () => {
  const root = buildFixture({ "models/scene.gltf": JSON.stringify({ asset: { version: "2.0" },
    buffers: [{ uri: "missing.bin", byteLength: 4 }] }) },
    'export const model = new URL("./models/scene.gltf", import.meta.url);');
  await assert.rejects(buildApplication(path.join(root, "index.html"), path.join(root, "dist")),
    (error) => error.pluginCode === "GLTF_RESOURCE_MISSING" && error.message.includes("missing.bin"));
  assert.equal(fs.existsSync(path.join(root, "dist")), false);
});
