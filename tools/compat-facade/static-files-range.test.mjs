import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { serveStaticFile } from "./static-files.mjs";

async function withServer(run) {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "f3d-byte-range-"));
  const root = path.join(parent, "public");
  fs.mkdirSync(root);
  const data = Buffer.from(Array.from({ length: 180013 }, (_, i) => (31 * i) & 255));
  fs.writeFileSync(path.join(root, "model.glb"), data);
  fs.writeFileSync(path.join(root, "empty.glb"), "");
  fs.writeFileSync(path.join(parent, "secret.glb"), "outside");
  fs.symlinkSync(path.join(parent, "secret.glb"), path.join(root, "escape.glb"));
  const server = http.createServer((request, response) => {
    const requested = path.join(root, decodeURIComponent(new URL(request.url, "http://local").pathname));
    if (!serveStaticFile(request, response, requested, root, root, { ".glb": "model/gltf-binary" })) {
      response.writeHead(404);
      response.end();
    }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const request = (pathname, headers = {}, method = "GET") => new Promise((resolve, reject) => {
    const req = http.request({ host: "127.0.0.1", port: server.address().port, path: pathname, headers,
      method, agent: false }, res => {
      const chunks = [];
      res.on("data", chunk => chunks.push(chunk));
      res.once("error", reject);
      res.once("end", () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
    });
    req.once("error", reject);
    req.end();
  });
  try { await run({ request, data }); }
  finally {
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(parent, { recursive: true, force: true });
  }
}

test("GLB and media byte ranges preserve exact binary content across stream boundaries", async () => {
  await withServer(async ({ request, data }) => {
    const full = await request("/model.glb");
    assert.equal(full.status, 200);
    assert.equal(full.headers["accept-ranges"], "bytes");
    assert.equal(full.headers["content-length"], String(data.length));
    assert.equal(full.headers["content-type"], "model/gltf-binary");
    assert.deepEqual(full.body, data);
    for (const [range, begin, end] of [
      ["bytes=0-15", 0, 15],
      ["bytes=65530-65540", 65530, 65540],
      ["bytes=180000-", 180000, data.length - 1],
      ["bytes=-16", data.length - 16, data.length - 1],
      ["bytes=-999999", 0, data.length - 1],
      ["bytes=180010-999999", 180010, data.length - 1],
    ]) {
      const response = await request("/model.glb", { Range: range });
      assert.equal(response.status, 206, range);
      assert.equal(response.headers["content-range"], "bytes " + begin + "-" + end + "/" + data.length);
      assert.equal(response.headers["content-length"], String(end - begin + 1));
      assert.deepEqual(response.body, data.subarray(begin, end + 1), range);
    }
  });
});

test("HEAD mirrors ranged GET and invalid single ranges report 416", async () => {
  await withServer(async ({ request, data }) => {
    for (const range of [null, "bytes=10-20", "bytes=999999-", "bytes=-0"]) {
      const headers = range ? { Range: range } : {};
      const get = await request("/model.glb", headers);
      const head = await request("/model.glb", headers, "HEAD");
      assert.equal(head.status, get.status);
      assert.equal(head.body.length, 0);
      for (const name of ["etag", "last-modified", "content-length", "content-range", "accept-ranges"])
        assert.equal(head.headers[name], get.headers[name], String(range) + " " + name);
    }
    for (const range of ["bytes=999999-", "bytes=19-3", "bytes=-0"]) {
      const response = await request("/model.glb", { Range: range });
      assert.equal(response.status, 416, range);
      assert.equal(response.headers["content-range"], "bytes */" + data.length);
      assert.equal(response.body.length, 0);
    }
    assert.equal((await request("/empty.glb")).status, 200);
    const empty = await request("/empty.glb", { Range: "bytes=0-0" });
    assert.equal(empty.status, 416);
    assert.equal(empty.headers["content-range"], "bytes */0");
  });
});

test("If-Range supports strong ETags and HTTP dates without stale partial responses", async () => {
  await withServer(async ({ request, data }) => {
    const full = await request("/model.glb");
    for (const validator of [full.headers.etag, full.headers["last-modified"]]) {
      const response = await request("/model.glb", { Range: "bytes=5-9", "If-Range": validator });
      assert.equal(response.status, 206, validator);
      assert.deepEqual(response.body, data.subarray(5, 10));
    }
    for (const validator of ['"stale"', "W/" + full.headers.etag, "invalid", "Thu, 01 Jan 1970 00:00:00 GMT"]) {
      const response = await request("/model.glb", { Range: "bytes=5-9", "If-Range": validator });
      assert.equal(response.status, 200, validator);
      assert.deepEqual(response.body, data);
    }
  });
});

test("unsupported ranges fall back to full assets without escaping filesystem containment", async () => {
  await withServer(async ({ request, data }) => {
    for (const range of ["bytes=0-3,6-8", "items=0-3", "bytes=nope", "bytes=1.5-2", "bytes=--1"]) {
      const response = await request("/model.glb", { Range: range });
      assert.equal(response.status, 200, range);
      assert.deepEqual(response.body, data, range);
    }
    assert.equal((await request("/escape.glb", { Range: "bytes=0-3" })).status, 403);
    assert.equal((await request("/missing.glb", { Range: "bytes=0-3" })).status, 404);
    assert.equal((await request("/model.glb", { Range: "bytes=0-3" })).status, 206);
  });
});
