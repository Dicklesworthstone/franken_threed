import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { resolvePackageExports, resolveExportTargetValue } from "../../tools/ingest/resolver.mjs";
import { IngestionResolutionError } from "../../tools/ingest/types.mjs";

const span = { line: 7, column: 4, offset: 92 };
function resolve(exports, subpath = ".") {
  return resolvePackageExports(subpath, exports, "/fixture/package.json", "three" + subpath.slice(1), "file:///app/main.mjs", span);
}
const cases = [
  ["main string shorthand", "./index.mjs", ".", "./index.mjs"],
  ["main array shorthand", [{ require: "./index.cjs" }, "./index.mjs"], ".", "./index.mjs"],
  ["main conditional shorthand", { import: "./index.mjs", default: "./fallback.mjs" }, ".", "./index.mjs"],
  ["nested conditions fall through when unmatched", { import: { development: "./dev.mjs" }, default: "./index.mjs" }, ".", "./index.mjs"],
  ["condition order is declaration order", { default: "./first.mjs", import: "./later.mjs" }, ".", "./first.mjs"],
  ["array skips invalid targets", { ".": [42, "../outside.mjs", "./index.mjs"] }, ".", "./index.mjs"],
  ["array skips null and unmatched conditions", { ".": [null, { require: "./index.cjs" }, "./index.mjs"] }, ".", "./index.mjs"],
  ["nested fallback arrays", { ".": [[{ require: "./index.cjs" }], ["./index.mjs"]] }, ".", "./index.mjs"],
  ["exact match beats a pattern", { "./*": "./wild/*.mjs", "./entry": "./exact.mjs" }, "./entry", "./exact.mjs"],
  ["longer trailer wins equal prefixes", { "./*": "./broad/*.mjs", "./*.js": "./specific/*.mjs" }, "./entry.js", "./specific/entry.mjs"],
  ["longer prefix wins before trailer", { "./*long.js": "./broad/*.mjs", "./feature/*": "./specific/*.mjs" }, "./feature/long.js", "./specific/long.js.mjs"],
  ["every target wildcard is replaced", { "./*": "./out/*/copy-*.mjs" }, "./entry", "./out/entry/copy-entry.mjs"],
  ["replacement metacharacters are literal", { "./*": "./out/*/copy-*.mjs" }, "./$&", "./out/$&/copy-$&.mjs"],
  ["wildcards span directories", { "./addons/*": "./examples/jsm/*" }, "./addons/controls/OrbitControls.js", "./examples/jsm/controls/OrbitControls.js"],
  ["target query and fragment survive", { ".": "./index.mjs?profile=web#one" }, ".", "./index.mjs?profile=web#one"],
  ["invalid multi-wildcard keys are not expanded", { "./*/*": "./wrong.mjs", "./*": "./right/*.mjs" }, "./a/b", "./right/a/b.mjs"],
];
for (const [name, exports, subpath, expected] of cases) {
  test(name, () => assert.equal(resolve(exports, subpath), expected));
}

const rejected = [
  ["shorthand never exposes subpaths", "./index.mjs", "./hidden"],
  ["exact null blocks wider pattern", { "./*": "./out/*.mjs", "./hidden": null }, "./hidden"],
  ["specific null pattern blocks wider pattern", { "./*": "./out/*.mjs", "./private/*": null }, "./private/secret"],
  ["unmatched specific pattern cannot fall back to a wider export", { "./*": "./out/*.mjs", "./private/*": { require: "./private/*.cjs" } }, "./private/secret"],
  ["empty capture does not match", { "./*.js": "./index.mjs" }, "./.js"],
  ["overlapping prefix and suffix do not match", { "./feature*feature": "./index.mjs" }, "./feature"],
  ["mixed subpath and condition keys are invalid", { ".": "./index.mjs", import: "./other.mjs" }, "."],
  ["numeric conditions are invalid", { ".": { 0: "./wrong.mjs", default: "./index.mjs" } }, "."],
  ["empty fallback array blocks", { ".": [] }, "."],
  ["blocked export root", null, "."],
  ["invalid scalar export", true, "."],
  ["absolute URL targets are invalid", { ".": "https://example.org/escape.mjs" }, "."],
  ["parent traversal targets are invalid", { ".": "./lib/../../escape.mjs" }, "."],
  ["encoded traversal targets are invalid", { ".": "./lib/%2e%2e/escape.mjs" }, "."],
  ["nested node_modules targets are invalid", { ".": "./node_modules/other/index.mjs" }, "."],
  ["wildcard captures cannot traverse", { "./*": "./out/*.mjs" }, "./../outside"],
  ["wildcard captures cannot enter node_modules", { "./*": "./out/*.mjs" }, "./node_modules/other"],
];
for (const [name, exports, subpath] of rejected) {
  test(name, () => assert.throws(() => resolve(exports, subpath), error => {
    assert.ok(error instanceof IngestionResolutionError);
    assert.equal(error.specifier, "three" + subpath.slice(1));
    assert.equal(error.referrerUrl, "file:///app/main.mjs");
    assert.equal(error.span, span);
    assert.match(error.message, /package\.json/);
    return true;
  }));
}
test("target helper preserves unmatched versus blocked and custom conditions", () => {
  assert.equal(resolveExportTargetValue({ require: "./index.cjs" }), undefined);
  assert.equal(resolveExportTargetValue(null), null);
  assert.equal(resolveExportTargetValue({ browser: "./browser.mjs", default: "./index.mjs" }, ["browser"]), "./browser.mjs");
  assert.equal(resolveExportTargetValue([{ require: "./index.cjs" }, "./index.mjs"]), "./index.mjs");
});

test("supported export choices agree with the executing Node ESM resolver", t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-export-oracle-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const expectations = [];
  for (const [index, [, exports, subpath, expected]] of cases.entries()) {
    const name = `fixture-${index}`;
    const packageRoot = path.join(directory, "node_modules", name);
    fs.mkdirSync(packageRoot, { recursive: true });
    fs.writeFileSync(path.join(packageRoot, "package.json"), JSON.stringify({ name, type: "module", exports }));
    expectations.push([name + subpath.slice(1), new URL(expected, pathToFileURL(packageRoot + path.sep)).href]);
  }
  const script = `const requests = ${JSON.stringify(expectations)}; console.log(JSON.stringify(requests.map(([name]) => import.meta.resolve(name))));`;
  const child = spawnSync(process.execPath, ["--input-type=module", "--eval", script], { cwd: directory, encoding: "utf8" });
  assert.equal(child.status, 0, child.stderr);
  assert.deepEqual(JSON.parse(child.stdout), expectations.map(([, url]) => url));
});
