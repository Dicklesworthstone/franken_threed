import assert from "node:assert/strict";
import { test } from "node:test";
import { routeImportMap } from "./import-map-routing.mjs";
import { createDevImportMap, transformHtmlImportMap } from "./index.mjs";

const readMap = (html) => JSON.parse(/<script\b[^>]*>([\s\S]*?)<\/script>/i.exec(html)[1]);
const wrap = (map, attributes = 'type="importmap"') =>
  `<script ${attributes}>${JSON.stringify(map)}</script>`;

test("scoped routes preserve aliases, per-scope selection and non-Three dependencies", () => {
  const source = {
    imports: { three: "../build/three.module.js", utility: "/utility.mjs", blocked: null },
    scopes: {
      "/gpu/": {
        three: "../build/three.webgpu.js",
        "three/webgpu": "../build/three.webgpu.js",
        utility: "/gpu/utility.mjs",
      },
      "/legacy/": { three: "../build/three.module.js", blocked: null },
    },
    custom: { keep: true },
  };
  const snapshot = structuredClone(source);
  const result = readMap(transformHtmlImportMap(wrap(source)));
  assert.deepEqual(source, snapshot);
  assert.equal(result.imports.three, "/compat-facade/three.js");
  assert.equal(result.scopes["/gpu/"].three, "/compat-facade/webgpu.js");
  assert.equal(result.scopes["/gpu/"].three, result.scopes["/gpu/"]["three/webgpu"]);
  assert.equal(result.scopes["/legacy/"].three, "/compat-facade/three.js");
  assert.equal(result.scopes["/gpu/"].utility, "/gpu/utility.mjs");
  assert.equal(result.scopes["/legacy/"].blocked, null);
  assert.deepEqual(result.custom, source.custom);
});

test("programmatic development maps preserve custom entries, scopes and explicit blocks", () => {
  const source = {
    imports: { three: null, "three/webgpu": null, utility: "/utility.mjs" },
    scopes: { "/private/": { three: null } },
  };
  const result = createDevImportMap({ sourceImportMap: source });
  assert.equal(result.imports.three, null);
  assert.equal(result.imports["three/webgpu"], null);
  assert.equal(result.imports.utility, "/utility.mjs");
  assert.deepEqual(result.scopes, source.scopes);
  assert.equal(createDevImportMap().imports.three, "/compat-facade/three.js");
  assert.equal(
    createDevImportMap({ routeThreeToWebgpu: true }).imports.three,
    "/compat-facade/webgpu.js",
  );
});

test("HTML routing never replaces an explicit null with an unblocked default", () => {
  const map = readMap(
    transformHtmlImportMap(wrap({ imports: { three: null, "three/webgpu": null } })),
  );
  assert.equal(map.imports.three, null);
  assert.equal(map.imports["three/webgpu"], null);
});

test("WebGPU aliases keep identical query and fragment module identity", () => {
  const source = {
    imports: {
      three: "../build/three.webgpu.js?revision=186#entry",
      alias: "../build/three.webgpu.js?revision=186#entry",
    },
  };
  const result = createDevImportMap({
    baseUrl: "https://local.test/dev/",
    sourceImportMap: source,
  });
  assert.equal(
    result.imports.three,
    "https://local.test/dev/compat-facade/webgpu.js?revision=186#entry",
  );
  assert.equal(result.imports.three, result.imports.alias);
  assert.equal(result.imports.three, result.imports["three/webgpu"]);
  const htmlMap = readMap(transformHtmlImportMap(wrap(source)));
  assert.equal(htmlMap.imports.three, htmlMap.imports["three/webgpu"]);
});

test("bundle-like query text and unrelated jsm directories are not rerouted", () => {
  const source = {
    imports: {
      utility: "/tools/inspect.js?file=three.webgpu.js",
      other: "/vendor/jsm/",
      fake: "/not-three.module.js",
    },
  };
  assert.deepEqual(routeImportMap(source), source);
});

test("exact addon targets and prefix aliases are routed by their selected files", () => {
  const source = {
    imports: {
      controls: "../examples/jsm/controls/OrbitControls.js",
      alias: "./jsm/loaders/GLTFLoader.js?rev=1",
      "three/addons/": "../examples/jsm/",
      "../selected/": "../examples/jsm/",
    },
  };
  assert.deepEqual(routeImportMap(source).imports, {
    controls: "/compat-facade/addons/controls/OrbitControls.js",
    alias: "/compat-facade/addons/loaders/GLTFLoader.js?rev=1",
    "three/addons/": "/compat-facade/addons/",
    "../selected/": "/compat-facade/addons/",
  });
});

test("attributes, CSP nonces, application code and outer HTML are unchanged", () => {
  const opening = `<ScRiPt data-note='a > b' nonce="csp-123" TYPE = 'importmap'>`;
  const prefix = "<!doctype html><style>body {margin:0}</style>";
  const suffix = '<script type="module">console.log("application unchanged");</script></body>';
  const html =
    prefix + opening + '{"imports":{"three":"../build/three.module.js"}}' + "</ScRiPt>" + suffix;
  const result = transformHtmlImportMap(html);
  assert.ok(result.startsWith(prefix + opening));
  assert.ok(result.endsWith("</ScRiPt>" + suffix));
  assert.ok(result.includes("/compat-facade/three.js"));
});

test("unquoted type attributes and whitespace around equals are accepted", () => {
  const result = transformHtmlImportMap(wrap({ imports: {} }, 'nonce="n" type = importmap'));
  assert.ok(result.startsWith('<script nonce="n" type = importmap>'));
});

test("comments, raw text, inert templates and data-type attributes are not import maps", () => {
  const fake = wrap({ imports: { three: "/fake/three.module.js" } });
  const prefix = `<!--${fake}--><textarea>${fake}</textarea><template><template>${fake}</template></template><script data-type="importmap">{"imports":{}}</script>`;
  const result = transformHtmlImportMap(
    prefix + wrap({ imports: { three: "../build/three.module.js" } }),
  );
  assert.ok(result.startsWith(prefix));
  assert.throws(() => transformHtmlImportMap(prefix), /No .*importmap/);
});

test("multiple maps are rewritten independently without shadowing later explicit mappings", () => {
  const html =
    wrap({ imports: { utility: "/utility.mjs" } }) +
    "\n" +
    wrap({ imports: { three: "../build/three.webgpu.js" } });
  const result = transformHtmlImportMap(html);
  const maps = [...result.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map((m) =>
    JSON.parse(m[1]),
  );
  assert.deepEqual(maps[0], { imports: { utility: "/utility.mjs" } });
  assert.deepEqual(maps[1], { imports: { three: "/compat-facade/webgpu.js" } });
});

test("malformed map JSON and invalid structures fail rather than fabricate a new map", () => {
  assert.throws(
    () => transformHtmlImportMap('<script type="importmap">{bad}</script>'),
    /Invalid import map JSON/,
  );
  for (const value of [
    null,
    [],
    1,
    { imports: [] },
    { scopes: null },
    { scopes: { "/": [] } },
    { integrity: [] },
  ]) {
    assert.throws(() => transformHtmlImportMap(wrap(value)), /must be an object/);
  }
  assert.throws(() => transformHtmlImportMap('<script type="importmap">{}'), /Unclosed/);
  assert.throws(
    () => transformHtmlImportMap('<script type="importmap" src="map.json">{}</script>'),
    /must be inline/,
  );
});

test("integrity metadata is never silently discarded or copied onto different bytes", () => {
  const untouched = {
    imports: { utility: "/utility.mjs" },
    integrity: { "/utility.mjs": "sha384-existing" },
  };
  assert.deepEqual(routeImportMap(untouched), untouched);
  assert.throws(
    () => routeImportMap({ ...untouched, imports: { three: "../build/three.module.js" } }),
    /integrity.*regenerated/,
  );
  assert.throws(
    () =>
      routeImportMap({ ...untouched, scopes: { "/gpu/": { three: "../build/three.webgpu.js" } } }),
    /integrity.*regenerated/,
  );
});

test("JSON text cannot close the rewritten script or trigger string replacement substitutions", () => {
  const dangerous = "</script><script>alert(1)</script>$&$`";
  const json = JSON.stringify({ imports: { utility: dangerous } }).replace(/</g, "\\u003c");
  const result = transformHtmlImportMap(`<script type="importmap">${json}</script>`);
  assert.equal((result.match(/<script/g) ?? []).length, 1);
  assert.equal((result.match(/<\/script>/g) ?? []).length, 1);
  assert.equal(readMap(result).imports.utility, dangerous);
});

test("prototype-shaped import keys remain inert own entries", () => {
  const source = JSON.parse(
    '{"imports":{"__proto__":"/prototype.mjs","constructor":"/ctor.mjs"},"scopes":{"__proto__":{"three":null}}}',
  );
  const result = createDevImportMap({ sourceImportMap: source });
  assert.equal(Object.getPrototypeOf(result.imports), Object.prototype);
  assert.equal(Object.hasOwn(result.imports, "__proto__"), true);
  assert.equal(result.imports.__proto__, "/prototype.mjs");
  assert.equal(Object.hasOwn(result.scopes, "__proto__"), true);
});

test("routing is idempotent and does not mutate the input map", () => {
  const source = {
    imports: { three: "../build/three.webgpu.js" },
    scopes: { "/gpu/": { tsl: "../build/three.tsl.js" } },
  };
  const snapshot = structuredClone(source);
  const first = routeImportMap(source);
  assert.deepEqual(routeImportMap(first), first);
  assert.deepEqual(source, snapshot);
  const html = transformHtmlImportMap(wrap(source));
  assert.equal(transformHtmlImportMap(html), html);
});

test("hostnames, non-network protocols and normalized traversal are not mistaken for bundle paths", () => {
  const source = {
    imports: {
      hostname: "https://three.js",
      data: 'data:text/javascript,export default "/three.module.js',
      traversed: "../examples/jsm/../other.mjs",
      encoded: "../examples/jsm/%2e%2e/other.mjs",
    },
  };
  assert.deepEqual(routeImportMap(source), source);
});

test("forcing WebGPU cannot bypass an existing root integrity mapping", () => {
  assert.throws(
    () =>
      createDevImportMap({
        routeThreeToWebgpu: true,
        sourceImportMap: {
          imports: { three: "/custom.mjs" },
          integrity: { "/custom.mjs": "sha384-existing" },
        },
      }),
    /integrity.*regenerated/,
  );
});

test("noscript fallback content is not treated as an active import map", () => {
  const fallback =
    "<noscript>" + wrap({ imports: { three: "/fallback/three.js" } }) + "</noscript>";
  assert.throws(() => transformHtmlImportMap(fallback), /No .*importmap/);
  assert.ok(transformHtmlImportMap(fallback + wrap({ imports: {} })).startsWith(fallback));
});
