/**
 * Build-only image discovery for standalone USDA layers. The retained USDLoader
 * and USDComposer still parse, compose, load images and publish source callbacks.
 * External USD references are not fetched by r186's standalone loader. USDZ and
 * USDC keep their existing paths; neither is decoded by this metadata pass.
 */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";

const PINNED_ROOT = new URL("../../upstream/three.js/", import.meta.url);
const PINNED_SOURCES = new Map([
  ["examples/jsm/loaders/USDLoader.js", "6b1ff7ccf76789c543bc039e0d5fe0ec50d5dc08"],
  ["examples/jsm/loaders/usd/USDComposer.js", "be4ca92d1a110ab6babcfec979e50a6ea9d13bb3"],
  ["examples/jsm/loaders/usd/USDAParser.js", "fbe60dff840c204c40178c1e5f621c1c6fe54549"],
]);
const error = (code, message) => { throw Object.assign(new Error(code + ": " + message), {code}); };
const isUsd = request => request.loader === "USDLoader" || request.loader === "USDZLoader";
const MAX_BYTES = 64 * 1024 * 1024;

/**
 * Import the audited, import-free source parser only for actual USD consumers.
 * Rollup awaits this preparation; the collector and per-file extraction stay
 * synchronous. Dynamic import preserves the tools' Node 20.18 minimum.
 *
 * A modified/missing source keeps the ordinary networked host path. A requested
 * standalone build fails explicitly for loose text instead of claiming closure.
 * Archive/binary payloads do not acquire a new parser-identity requirement.
 */
export async function prepareUsdAssetUris(requests, {packageRootUrl, strict = false} = {}) {
  if (!requests.some(isUsd)) return null;
  let Parser = null, unavailable = "Pinned USDA image discovery is unavailable";
  try {
    const root = packageRootUrl
      ? (String(packageRootUrl).startsWith("file:") ? new URL(packageRootUrl) : pathToFileURL(path.resolve(packageRootUrl)))
      : new URL(PINNED_ROOT);
    root.search = ""; root.hash = "";
    if (!root.pathname.endsWith("/")) root.pathname += "/";
    for (const [name, expected] of PINNED_SOURCES) {
      const bytes = fs.readFileSync(new URL(name, root));
      const actual = createHash("sha1").update("blob " + bytes.length + "\0").update(bytes).digest("hex");
      if (actual !== expected) throw new Error("Modified USD source: " + name);
    }
    const parserUrl = new URL("examples/jsm/loaders/usd/USDAParser.js", root);
    ({USDAParser: Parser} = await import(parserUrl.href));
    if (typeof Parser !== "function") throw new Error("Missing USDAParser export");
  } catch (cause) {
    Parser = null;
    unavailable = cause.message;
  }
  return bytes => {
    // These are the same content signatures USDLoader checks before USDA text.
    if (bytes[0] === 0x50 && bytes[1] === 0x4b ||
        bytes.length >= 8 && [0x50,0x58,0x52,0x2d,0x55,0x53,0x44,0x43].every((byte, i) => bytes[i] === byte)) return [];
    if (!Parser) {
      if (strict) error("USD_SOURCE_UNSUPPORTED", unavailable);
      return [];
    }
    if (bytes.length > MAX_BYTES) error("USD_BUDGET", "USDA image discovery exceeds 64 MiB");
    const text = new TextDecoder().decode(bytes);
    const parser = new Parser();
    // The upstream parser reuses plain-object dictionary entries. Prevent an
    // input dictionary from selecting an inherited object in the build realm.
    // Preprocessing is source-owned and has no dictionary or code evaluation.
    const inherited = new Set(Object.getOwnPropertyNames(Object.prototype));
    let depth = 0;
    for (const line of parser._preprocess(text).split("\n")) {
      const key = line.endsWith("{") ? line.slice(0, -1).trim() : line;
      if (inherited.has(key)) {
        if (strict) error("USD_SOURCE_UNSUPPORTED", "USDA dictionary key cannot be inspected in the build realm");
        return [];
      }
      if (line.endsWith("{") || line.endsWith("(")) depth++;
      else if (line.endsWith("}") || line.endsWith(")")) depth--;
      if (depth > 128) error("USD_BUDGET", "USDA metadata exceeds the depth work bound");
    }
    let data;
    try { data = parser.parseData(text); }
    catch { return []; } // Keep malformed bytes and the retained parser's error.
    // USDComposer follows connection paths before a default and uses time zero
    // (or the first sample) for a time-sampled rest-pose attribute.
    const resolve = (name, seen = new Set()) => {
      const key = String(name).replace(/[<>]/g, "");
      if (seen.has(key)) return undefined;
      const fields = data.specsByPath[key]?.fields;
      if (!fields) return undefined;
      if (seen.size >= 128) error("USD_BUDGET", "USDA image attribute chain exceeds the depth work bound");
      const next = new Set(seen).add(key);
      for (const connection of fields.connectionPaths ?? []) {
        const value = resolve(connection, next);
        if (value !== undefined) return value;
      }
      if (fields.default !== undefined) return fields.default;
      const {times, values} = fields.timeSamples ?? {};
      if (times?.length && values) {
        const index = times.indexOf(0);
        return values[index >= 0 ? index : 0];
      }
      return undefined;
    };
    const uris = new Set();
    for (const [name, spec] of Object.entries(data.specsByPath)) {
      if (spec.specType !== 6) continue;
      const attribute = key => resolve(name + "." + key);
      const kind = attribute("info:id") ?? spec.fields["info:id"];
      const key = kind === "arnold:image" ? "inputs:filename"
        : kind === "UsdUVTexture" || typeof kind === "string" && kind.startsWith("ND_image_") ? "inputs:file" : null;
      let uri = key && attribute(key);
      if (typeof uri !== "string" || !uri) continue;
      // _loadTexture strips one surrounding @ pair even for string inputs;
      // _resolveFilePath then removes one leading "./" before LoaderUtils.
      if (uri.startsWith("@")) uri = uri.slice(1);
      if (uri.endsWith("@")) uri = uri.slice(0, -1);
      if (uri.startsWith("./")) uri = uri.slice(2);
      if (uri) uris.add(uri);
    }
    return [...uris];
  };
}
