import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { packHtml } from "./pack_html.mjs";

const cli = fileURLToPath(new URL("./cli.mjs", import.meta.url));
const run = (args) => spawnSync(process.execPath, [cli, ...args], { encoding: "utf8" });
function fixture(files = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-pack-cli-"));
  for (const [name, data] of Object.entries({
    "index.html": '<script type="module" src="entry.mjs"></script>',
    "entry.mjs": "export const value=42;",
    ...files,
  })) {
    const file = path.join(root, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, data);
  }
  return { root, entry: path.join(root, "index.html"), out: path.join(root, "single.html") };
}
const sourceModules = (html) =>
  Object.values(
    JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports,
  ).map((url) => Buffer.from(url.split(",")[1].split("#")[0], "base64").toString("utf8"));

test("CLI exports a standalone HTML entry without importing Rollup and writes the requested report", () => {
  const f = fixture(),
    report = path.join(f.root, "report.json");
  const result = run(["--entry", f.entry, "--pack-html", f.out, "--output", report]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Single HTML exported/);
  assert.match(result.stdout, /application-created networking/);
  assert.equal(JSON.parse(fs.readFileSync(report)).moduleCount, 1);
  assert.ok(fs.readFileSync(f.out, "utf8").includes("data:text/javascript;base64,"));
  assert.ok(fs.existsSync(f.entry));
  assert.ok(fs.existsSync(path.join(f.root, "entry.mjs")));
});

test("CLI exposes the export mode and rejects incompatible modes before building", () => {
  assert.match(run(["--help"]).stdout, /--pack-html/);
  const f = fixture();
  for (const args of [
    ["--entry", f.entry, "--pack-html"],
    ["--entry", path.join(f.root, "entry.mjs"), "--pack-html", f.out],
    [
      "--entry",
      f.entry,
      "--pack-html",
      f.out,
      "--build-kernel",
      path.join(f.root, "kernel"),
      "--parameter-types",
      "f64[]",
    ],
    ["--entry", f.entry, "--pack-html", f.out, "--package-root", "https://example.com/"],
  ]) {
    assert.equal(run(args).status, 1);
    assert.equal(fs.existsSync(f.out), false);
  }
});

test("CLI preflight rejects existing files, symlinks and output collisions without partial output", () => {
  const f = fixture();
  assert.equal(run(["--entry", f.entry, "--pack-html", f.entry]).status, 1);
  assert.equal(run(["--entry", f.entry, "--pack-html", f.out, "--output", f.out]).status, 1);
  fs.symlinkSync(f.entry, f.out);
  assert.equal(run(["--entry", f.entry, "--pack-html", f.out]).status, 1);
  assert.equal(fs.readFileSync(f.entry, "utf8"), '<script type="module" src="entry.mjs"></script>');
  const build = path.join(f.root, "build");
  assert.equal(
    run(["--entry", f.entry, "--build-app", build, "--pack-html", path.join(build, "index.html")])
      .status,
    1,
  );
  assert.equal(fs.existsSync(build), false);
});

test("CLI refuses an unclosed graph but retains the source directory", () => {
  const f = fixture({ "entry.mjs": `export const load=value=>import(value);` });
  const result = run(["--entry", f.entry, "--pack-html", f.out]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /DYNAMIC_IMPORT_OPEN/);
  assert.equal(fs.existsSync(f.out), false);
  assert.ok(fs.existsSync(f.entry));
});

test("packs the actual finite-import mapper expression emitted by the existing bundler", () => {
  const source = `export const load = flag => import((s => s === './left.mjs' ? './left-ABC.js' : s === './right.mjs' ? './right-DEF.js' : s)(flag ? './left.mjs' : './right.mjs'));`;
  const f = fixture({
    "entry.mjs": source,
    "left-ABC.js": `export const value='left';`,
    "right-DEF.js": `export const value='right';`,
  });
  const result = packHtml(f.entry, f.out);
  assert.equal(result.moduleCount, 3);
  const sources = sourceModules(fs.readFileSync(f.out, "utf8"));
  assert.ok(sources.some((s) => s.includes(`s === './left.mjs' ? './left-ABC.js'`)));
  assert.ok(sources.some((s) => s.includes('v === "./left-ABC.js" ? "f3d-packed/')));
});

test("nested imports in dynamic selectors retain disjoint source edits", () => {
  const source = `export const load = async () => import((await import('./selector.mjs'), './lazy.mjs'));`;
  const f = fixture({
    "entry.mjs": source,
    "selector.mjs": "export const value=true;",
    "lazy.mjs": "export const value=3;",
  });
  assert.equal(packHtml(f.entry, f.out).moduleCount, 3);
  const sources = sourceModules(fs.readFileSync(f.out, "utf8"));
  assert.ok(sources.some((s) => s.includes('await import("f3d-packed/')));
});

test("JSON modules remain JSON while import attributes remain in the importing source", () => {
  const f = fixture({
    "entry.mjs": `import config from './config.json' with {type:'json'}; export {config};`,
    "config.json": '{"answer":42}',
  });
  assert.equal(packHtml(f.entry, f.out).moduleCount, 2);
  const output = fs.readFileSync(f.out, "utf8");
  assert.ok(output.includes("data:application/json;base64,"));
  assert.ok(sourceModules(output).some((s) => s.includes("with {type:'json'}")));
});

test("responsive srcset keeps descriptors and data-URL commas without losing alternatives", () => {
  const f = fixture({
    "index.html":
      '<picture><source srcset="small.png 400w, large.png 800w" sizes="50vw"><img src="small.png" srcset="small.png 1x, large.png 2x"></picture>',
    "small.png": Buffer.from([1, 2, 3]),
    "large.png": Buffer.from([4, 5, 6]),
  });
  assert.equal(packHtml(f.entry, f.out).assetCount, 2);
  const output = fs.readFileSync(f.out, "utf8");
  assert.match(
    output,
    /srcset="data:image\/png;f3d-resource=[0-9a-f]+;base64,AQID 400w, data:image\/png;f3d-resource=[0-9a-f]+;base64,BAUG 800w"/,
  );
  assert.match(
    output,
    /srcset="data:image\/png;f3d-resource=[0-9a-f]+;base64,AQID 1x, data:image\/png;f3d-resource=[0-9a-f]+;base64,BAUG 2x"/,
  );
  assert.ok(output.includes('sizes="50vw"'));
});

test("legacy CLI option validation remains unchanged when packing is not requested", () => {
  const f = fixture();
  assert.match(run(["--entry", f.entry, "--specialize-numeric"]).stderr, /requires --build-app/);
  assert.match(
    run(["--entry", f.entry, "--max-memory-pages", "4"]).stderr,
    /requires --build-kernel or --specialize-numeric/,
  );
  assert.match(
    run(["--entry", f.entry, "--parameter-types", "f64[]"]).stderr,
    /--parameter-types requires --build-kernel/,
  );
});
