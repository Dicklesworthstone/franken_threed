import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { prepareUsdAssetUris } from "./pack_usd.mjs";
import { usdcBytes, usdcImageSpecs } from "./fixtures/usdc_resources.mjs";
import { USDCParser } from "../../upstream/three.js/examples/jsm/loaders/usd/USDCParser.js";

const requests = [{loader: "USDLoader"}];
const bytes = text => Buffer.from(text);
const root = new URL("../../upstream/three.js/", import.meta.url);
const imageNodes = "#usda 1.0\ndef Material \"Paint\" {\n  # asset inputs:file = @comment.png@\n  string documentation = \"asset inputs:file = @string.png@\"\n  asset inputs:reference = @not-an-image.png@\n  def Shader \"Image\" {\n    uniform token info:id = \"UsdUVTexture\"\n    asset inputs:file = @discarded.png@\n    asset inputs:file = @\"./images/paint%20%23.png?v=1#surface\"@\n  }\n  def Shader \"Arnold\" {\n    uniform token info:id = \"arnold:image\"\n    string inputs:filename = \"@./images/arnold.png@\"\n  }\n  def \"MaterialX\" {\n    uniform token info:id = \"ND_image_color3\"\n    asset inputs:file = @./images/mx.png@\n  }\n  def Shader \"Noise\" {\n    uniform token info:id = \"ND_noise3d_color3\"\n    asset inputs:file = @not-loaded.png@\n  }\n}\ndef Xform \"Other\" (\n  prepend references = @./not-loaded.usda@</Geometry>\n)\n{\n}";
const connectedNodes = "#usda 1.0\ndef Material \"Paint\" {\n  asset inputs:shared = @./shared/linked.png@\n  def Shader \"Image\" {\n    uniform token info:id = \"UsdUVTexture\"\n    asset inputs:file.connect = </Paint.inputs:shared>\n    asset inputs:file = @default.png@\n  }\n  def Shader \"Animated\" {\n    uniform token info:id = \"ND_image_color3\"\n    asset inputs:file.timeSamples = {\n      1: @first.png@,\n      0: @zero.png@\n    }\n  }\n  def Shader \"Cycle\" {\n    uniform token info:id = \"UsdUVTexture\"\n    asset inputs:file.connect = </Paint/Cycle.inputs:other>\n    asset inputs:other.connect = </Paint/Cycle.inputs:file>\n    asset inputs:file = @cycle-fallback.png@\n  }\n}";

test("pinned USDA discovery reads only image-node fields with native duplicate and string rules", async () => {
  const extract = await prepareUsdAssetUris(requests, {strict: true});
  assert.deepEqual(extract(bytes(imageNodes)), [
    "images/paint%20%23.png?v=1#surface", "images/arnold.png", "images/mx.png",
  ]);
  assert.deepEqual(extract(bytes('#usda 1.0\n/* def Shader "False" { */\ndef Shader "Wrong" {\n token info:id = "UsdPreviewSurface"\n asset inputs:file = @false.png@\n}')), []);
});

test("USDA image attributes follow connections, cycle fallback and rest-pose time samples", async () => {
  const extract = await prepareUsdAssetUris(requests, {strict: true});
  assert.deepEqual(extract(bytes(connectedNodes)), ["shared/linked.png", "zero.png", "cycle-fallback.png"]);
});

test("USDA discovery preserves malformed-input ownership and enforces build-realm work bounds", async () => {
  const extract = await prepareUsdAssetUris(requests, {strict: true});
  assert.deepEqual(extract(bytes('def Shader "Bad"\n}\nasset inputs:file = @bad.png@')), []);
  assert.throws(() => extract({length: 64 * 1024 * 1024 + 1}), {code: "USD_BUDGET"});
  assert.throws(() => extract(bytes('#usda 1.0\n' + 'def Xform "Nested" {\n'.repeat(129) + '}\n'.repeat(129))),
    {code: "USD_BUDGET"});
  const before = Object.getOwnPropertyDescriptors(Object);
  assert.throws(() => extract(bytes('#usda 1.0\nconstructor {\n probe = 1\n}')),
    {code: "USD_SOURCE_UNSUPPORTED"});
  assert.deepEqual(Object.getOwnPropertyDescriptors(Object), before);
});

test("non-USD and archive payloads do not require local USD parsers", async () => {
  const packageRootUrl = pathToFileURL(fs.mkdtempSync(path.join(os.tmpdir(), "f3d-usd-unavailable-"))).href;
  assert.equal(await prepareUsdAssetUris([{loader: "FileLoader"}], {packageRootUrl, strict: true}), null);
  const extract = await prepareUsdAssetUris(requests, {packageRootUrl, strict: true});
  assert.deepEqual(extract(Buffer.from([0x50, 0x4b])), []);
  assert.throws(() => extract(Buffer.from("PXR-USDC")), {code: "USD_SOURCE_UNSUPPORTED"});
  assert.throws(() => extract(bytes(imageNodes)), {code: "USD_SOURCE_UNSUPPORTED"});
  const networked = await prepareUsdAssetUris(requests, {packageRootUrl});
  assert.deepEqual(networked(bytes(imageNodes)), []);
  assert.deepEqual(networked(Buffer.from("PXR-USDC")), []);
});

test("changed USD parser source is never executed as build tooling", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-usd-modified-"));
  for (const name of ["USDLoader.js", "usd/USDComposer.js", "usd/USDAParser.js"]) {
    const filename = path.join(directory, "examples/jsm/loaders", name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    const source = fs.readFileSync(new URL("examples/jsm/loaders/" + name, root), "utf8");
    fs.writeFileSync(filename, source + (name.endsWith("USDAParser.js") ? "\nglobalThis.__f3dUnverifiedUsdParserRan = true;\n" : ""));
  }
  fs.writeFileSync(path.join(directory, "package.json"), '{"type":"module"}');
  const before = globalThis.__f3dUnverifiedUsdParserRan;
  const extract = await prepareUsdAssetUris(requests, {packageRootUrl: directory, strict: true});
  assert.throws(() => extract(bytes(imageNodes)), {code: "USD_SOURCE_UNSUPPORTED"});
  assert.equal(globalThis.__f3dUnverifiedUsdParserRan, before);
});

test("compressed and legacy USDC discovery uses native image values, connections and string-table identity", async () => {
  const specs = usdcImageSpecs();
  specs["/Materials/Paint/Image.inputs:file"].fields.default = {type: "asset", value: "unused-fallback.png"};
  specs["/Materials/Paint/MaterialX"] = {specType: 6, fields: {}};
  specs["/Materials/Paint/MaterialX.info:id"] = {specType: 1, fields: {default: {type: "string", value: "ND_image_color3"}}};
  specs["/Materials/Paint/MaterialX.inputs:file"] = {specType: 1, fields: {default: {type: "string", value: "@./images/mx.png@"}}};
  specs["/Materials/Paint/Arnold"] = {specType: 6, fields: {"info:id": {type: "token", value: "arnold:image"}}};
  specs["/Materials/Paint/Arnold.inputs:filename"] = {specType: 1, fields: {default: {type: "asset", value: "./images/arnold.png"}}};
  const extract = await prepareUsdAssetUris(requests, {strict: true});
  for (const compressed of [false, true]) {
    const encoded = usdcBytes(specs, {compressed}), before = Buffer.from(encoded);
    const data = new USDCParser().parseData(Uint8Array.from(encoded).buffer).specsByPath;
    assert.equal(data["/Materials/Paint/MaterialX.info:id"].fields.default, "ND_image_color3");
    assert.deepEqual(data["/Materials/Paint/Image.inputs:file"].fields.connectionPaths, ["/Materials/Paint.inputs:image"]);
    assert.deepEqual(extract(encoded), ["paint%20%23.png?edition=2#surface", "images/mx.png", "images/arnold.png"]);
    assert.deepEqual(encoded, before, "metadata discovery preserves every source byte");
  }
});

test("USDC rest-pose image samples and cyclic connection fallback match pinned values", async () => {
  const specs = usdcImageSpecs();
  specs["/Materials/Paint.inputs:image"].fields = {timeSamples: {type: "times", value: {
    times: [1, 0], values: [{type: "asset", value: "first.png"}, {type: "asset", value: "zero.png"}],
  }}};
  const extract = await prepareUsdAssetUris(requests, {strict: true});
  for (const compressed of [false, true]) {
    const encoded = usdcBytes(specs, {compressed});
    const data = new USDCParser().parseData(Uint8Array.from(encoded).buffer).specsByPath;
    assert.deepEqual(data["/Materials/Paint.inputs:image"].fields.timeSamples, {times: [1, 0], values: ["first.png", "zero.png"]});
    assert.deepEqual(extract(encoded), ["zero.png"]);
  }
  specs["/Materials/Paint.inputs:image"].fields = {
    connectionPaths: {type: "paths", value: ["/Materials/Paint/Image.inputs:file"]},
    default: {type: "asset", value: "cycle.png"},
  };
  assert.deepEqual(extract(usdcBytes(specs)), ["cycle.png"]);
});

test("USDC metadata bounds precede decoded allocations and recursive binary paths", async () => {
  const extract = await prepareUsdAssetUris(requests, {strict: true});
  const source = usdcBytes(usdcImageSpecs());
  const section = (bytes, name) => {
    const toc = Number(bytes.readBigUInt64LE(16)), count = Number(bytes.readBigUInt64LE(toc));
    for (let i = 0; i < count; i++) {
      const start = toc + 8 + i * 32;
      if (bytes.toString("utf8", start, start + 16).split("\0")[0] === name) return Number(bytes.readBigUInt64LE(start + 16));
    }
    throw new Error("Missing fixture section " + name);
  };
  for (const name of ["TOKENS", "STRINGS", "FIELDS", "FIELDSETS", "PATHS", "SPECS"]) {
    const malformed = Buffer.from(source);
    malformed.writeBigUInt64LE(1000000000n, section(malformed, name));
    assert.throws(() => extract(malformed), {code: "USD_BUDGET"}, name);
  }
  const expanded = Buffer.from(source);
  expanded.writeBigUInt64LE(64n * 1024n * 1024n, section(expanded, "TOKENS") + 8);
  assert.throws(() => extract(expanded), {code: "USD_BUDGET"});
  const cyclic = usdcBytes(usdcImageSpecs(), {compressed: false});
  const box = section(cyclic, "PATHS") + 8 + 9;
  cyclic.writeBigUInt64LE(BigInt(box), box + 9);
  assert.throws(() => extract(cyclic), {code: "USD_BUDGET"});
  const sparse = usdcBytes(usdcImageSpecs(), {compressed: false});
  sparse.writeUInt32LE(1000000, section(sparse, "PATHS") + 8);
  const sourceAccepted = new USDCParser().parseData(Uint8Array.from(sparse).buffer).specsByPath;
  assert.equal(sourceAccepted["/Materials/Paint.inputs:image"].fields.default, "./paint%20%23.png?edition=2#surface",
    "the native parser accepts growing a sparse path array; refusal must not be reported as a parse error");
  assert.throws(() => extract(sparse), {code: "USD_SOURCE_UNSUPPORTED"});
  const networked = await prepareUsdAssetUris(requests);
  assert.deepEqual(networked(sparse), [], "unadmitted metadata remains on its declared host path");
  assert.deepEqual(extract(Buffer.from("PXR-USDC")), [], "malformed input errors remain runtime-owned");
  assert.deepEqual(extract(source), ["paint%20%23.png?edition=2#surface"], "a refusal does not poison later discovery");
});

test("unverified USDC source never executes as build tooling or disables valid USDA discovery", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-usdc-modified-"));
  for (const name of ["USDLoader.js", "usd/USDComposer.js", "usd/USDAParser.js", "usd/USDCParser.js"]) {
    const filename = path.join(directory, "examples/jsm/loaders", name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    const source = fs.readFileSync(new URL("examples/jsm/loaders/" + name, root), "utf8");
    fs.writeFileSync(filename, source + (name.endsWith("USDCParser.js") ? "\nglobalThis.__f3dUnverifiedUsdcParserRan = true;\n" : ""));
  }
  fs.writeFileSync(path.join(directory, "package.json"), '{"type":"module"}');
  const before = globalThis.__f3dUnverifiedUsdcParserRan;
  const extract = await prepareUsdAssetUris(requests, {packageRootUrl: directory, strict: true});
  assert.throws(() => extract(usdcBytes(usdcImageSpecs())), {code: "USD_SOURCE_UNSUPPORTED"});
  assert.deepEqual(extract(bytes(imageNodes)), ["images/paint%20%23.png?v=1#surface", "images/arnold.png", "images/mx.png"]);
  assert.equal(globalThis.__f3dUnverifiedUsdcParserRan, before);
});
