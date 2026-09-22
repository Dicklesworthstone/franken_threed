/** Real offline archive/install tests: a working output folder is not enough. */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { emitPortablePackage } from "./portable-package.mjs";

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
function execute(command, args, cwd, env = process.env) {
  const result = spawnSync(command, args, {
    cwd,
    env,
    encoding: "utf8",
    timeout: 30_000,
    maxBuffer: 4 * 1024 * 1024,
  });
  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result.stdout;
}

for (const packageName of ["@franken/three", "three"]) {
  test(`offline npm archive installs as ${packageName} with all runtime assets`, (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-pack-"));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const source = path.join(root, "source");
    const out = path.join(root, "output");
    const consumer = path.join(root, "consumer");
    const cache = path.join(root, "cache");
    const archives = path.join(root, "archives");
    for (const dir of [source, consumer, archives]) fs.mkdirSync(dir);
    const put = (relative, content) => {
      const target = path.join(source, relative);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, content);
    };
    const manifest = {
      name: "three",
      version: "1.0.0",
      type: "module",
      exports: {
        ".": { import: "./build/root.js", require: "./build/root.cjs" },
        "./webgpu": "./build/root.js",
        "./addons/*": "./examples/jsm/*",
        "./codec.wasm": "./assets/codec.wasm",
        "./private/*": null,
      },
      // These allowlists and ignore files intentionally conflict with the asset
      // inventory. The generated archive must not lose decoder/worker files.
      files: ["build"],
      sideEffects: ["./src/nodes/**/*"],
      scripts: { postinstall: 'node -e "process.exit(99)"' },
    };
    put("package.json", JSON.stringify(manifest));
    put("build/root.js", "export class Mesh {}");
    put("build/root.cjs", "module.exports = { ready: true };");
    put(
      "examples/jsm/Loader.js",
      `
      import { Mesh } from 'three';
      export { Mesh };
      export const workerURL = new URL('../workers/decoder.js', import.meta.url);
    `,
    );
    put(
      "examples/workers/decoder.js",
      `
      import { parentPort } from 'node:worker_threads';
      import fs from 'node:fs';
      import { Mesh } from 'three';
      const bytes = fs.readFileSync(new URL('../../assets/codec.wasm', import.meta.url));
      await WebAssembly.instantiate(bytes);
      parentPort.postMessage({ bytes: [...bytes], mesh: new Mesh().constructor.name });
    `,
    );
    put("assets/codec.wasm", Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
    put("assets/fonts/fixture.typeface.json", '{"glyphs":{}}');
    put("src/nodes/register.js", "globalThis.nodeRegistered = true;");
    put(".npmignore", "assets/\nexamples/\n");
    put("assets/.npmignore", "*.wasm\nfonts/\n");
    put("examples/.gitignore", "workers/\n");
    put(".npmrc", "registry=https://invalid.example/\n");
    put("LICENSE", "MIT fixture license");
    const originalManifest = fs.readFileSync(path.join(source, "package.json"), "utf8");
    const emitted = emitPortablePackage(out, { packageDir: source, packageName });
    const config = path.join(root, "empty-npmrc");
    fs.writeFileSync(config, "");
    const env = {
      ...process.env,
      npm_config_userconfig: config,
      npm_config_globalconfig: path.join(root, "global-npmrc"),
    };
    const flags = ["--offline", "--ignore-scripts", "--no-audit", "--no-fund", "--cache", cache];
    const packed = JSON.parse(
      execute(npm, ["pack", "--json", "--pack-destination", archives, ...flags], out, env),
    )[0];
    const packedPaths = new Set(packed.files.map((entry) => entry.path));
    for (const relative of emitted.files)
      assert.ok(packedPaths.has(relative), `npm pack silently omitted ${relative}`);
    assert.ok(packedPaths.has("LICENSE"));
    assert.equal(
      [...packedPaths].some((name) => /(?:^|\/)\.(?:npmrc|npmignore|gitignore)$/.test(name)),
      false,
    );
    // Neither a checkout nor a generated folder is available during installation.
    fs.rmSync(source, { recursive: true });
    fs.rmSync(out, { recursive: true });
    fs.writeFileSync(
      path.join(consumer, "package.json"),
      '{"name":"consumer","version":"1.0.0","private":true}',
    );
    execute(
      npm,
      ["install", "--package-lock=false", path.join(archives, packed.filename), ...flags],
      consumer,
      env,
    );
    const installed = path.join(consumer, "node_modules", ...packageName.split("/"));
    assert.equal(
      fs.readFileSync(path.join(installed, "_retained/package.json"), "utf8"),
      originalManifest,
    );
    assert.deepEqual(
      JSON.parse(fs.readFileSync(path.join(installed, "package.json"))).sideEffects,
      ["./_retained/src/nodes/**/*"],
    );
    const program = path.join(consumer, "exercise.mjs");
    fs.writeFileSync(
      program,
      `
      import assert from 'node:assert/strict';
      import { createRequire } from 'node:module';
      import { once } from 'node:events';
      import { Worker } from 'node:worker_threads';
      import fs from 'node:fs';
      import { Mesh } from ${JSON.stringify(packageName)};
      import { Mesh as GPU } from ${JSON.stringify(`${packageName}/webgpu`)};
      import { Mesh as Addon, workerURL } from ${JSON.stringify(`${packageName}/addons/Loader.js`)};
      const require = createRequire(import.meta.url);
      assert.equal(Mesh, GPU); assert.equal(Mesh, Addon);
      assert.deepEqual(require(${JSON.stringify(packageName)}), { ready: true });
      assert.throws(() => require.resolve(${JSON.stringify(`${packageName}/private/hidden.js`)}), { code: 'ERR_PACKAGE_PATH_NOT_EXPORTED' });
      const worker = new Worker(workerURL);
      const exit = once(worker, 'exit');
      const [message] = await once(worker, 'message');
      assert.deepEqual(message, { bytes: [0,97,115,109,1,0,0,0], mesh: 'Mesh' });
      assert.deepEqual(await exit, [0]);
      assert.deepEqual(JSON.parse(fs.readFileSync(new URL('../../assets/fonts/fixture.typeface.json', workerURL))), { glyphs: {} });
    `,
    );
    execute(process.execPath, [program], consumer);
  });
}
