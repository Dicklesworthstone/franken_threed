import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { prepareUsdAssetUris } from "./pack_usd.mjs";

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

test("non-USD and archive/binary payloads do not require a local USDA parser", async () => {
  const packageRootUrl = pathToFileURL(fs.mkdtempSync(path.join(os.tmpdir(), "f3d-usd-unavailable-"))).href;
  assert.equal(await prepareUsdAssetUris([{loader: "FileLoader"}], {packageRootUrl, strict: true}), null);
  const extract = await prepareUsdAssetUris(requests, {packageRootUrl, strict: true});
  assert.deepEqual(extract(Buffer.from([0x50, 0x4b])), []);
  assert.deepEqual(extract(Buffer.from("PXR-USDC")), []);
  assert.throws(() => extract(bytes(imageNodes)), {code: "USD_SOURCE_UNSUPPORTED"});
  const networked = await prepareUsdAssetUris(requests, {packageRootUrl});
  assert.deepEqual(networked(bytes(imageNodes)), []);
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
