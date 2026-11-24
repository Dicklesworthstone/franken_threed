import assert from "node:assert/strict";
import test from "node:test";
import { cssResources } from "./css_resources.mjs";

test("CSS resource scanner finds each responsive image candidate without consuming its descriptors", () => {
  const source = `.one { background: image-set("small.png" 1x type("image/png"), url(large.png) type('image/png') 2x); }
    .two { background: -webkit-image-set('legacy.png' 1x, "retina.png" 192dpi); }
    .three { background: IMAGE-SET("modern.avif" type("image/avif"), "fallback.webp" type("image/webp")); }`;
  const resources = cssResources(source);
  assert.deepEqual(resources.map(({ url }) => url),
    ["small.png", "large.png", "legacy.png", "retina.png", "modern.avif", "fallback.webp"]);
  assert.deepEqual(resources.map(({ start, end }) => source.slice(start, end)),
    ['"small.png"', "url(large.png)", "'legacy.png'", '"retina.png"', '"modern.avif"', '"fallback.webp"']);
});

test("CSS resource scanner isolates comments, ordinary strings and nested non-image descriptors", () => {
  const resources = cssResources(`
    /* image-set("comment.png" 1x, "comment2.png" 2x) */
    .label::before { content: 'image-set("text.png" 1x), url(text2.png)'; }
    .scene { background: image-set(linear-gradient(red, blue) 1x,
      var(--image, "variable-string.png") 2x, "real.png" type("fake.png") 3x); }
    .other { value: my-url("not-a-url.png"); background: url(other.png); }
  `);
  assert.deepEqual(resources.map(({ url }) => url), ["real.png", "other.png"]);
});

test("CSS resource scanner decodes escapes and all CSS line continuations identically", () => {
  for (const newline of ["\n", "\r", "\r\n", "\f"]) {
    const source = `@import './the\\${newline}me.css' layer(theme) supports(display:grid) screen;
      .a {background: image-set('./im\\${newline}age.png' 1x, url(./te\\73${newline}t.png) 2x);}`;
    const resources = cssResources(source);
    assert.deepEqual(resources.map(({ url, kind }) => [url, kind]),
      [["./theme.css", "import"], ["./image.png", "image-set"], ["./test.png", "url"]]);
    assert.equal(source.slice(resources[0].end, resources[0].end + 13), " layer(theme)");
    assert.deepEqual(cssResources(`.a {background:url(./im\\${newline}age.png)}`), [],
      "unquoted URLs cannot contain escaped newlines");
  }
  assert.deepEqual(cssResources(String.raw`.a {background:image-set('./space\20 name.png' 1x,
    './\01f680.png' 2x, './\0.png' 3x, './\d800.png' 4x, './\110000.png' 5x);}`)
    .map(({ url }) => url), ["./space name.png", "./🚀.png", "./�.png", "./�.png", "./�.png"]);
});

test("CSS source offsets remain stable across nested image sets and separate scans", () => {
  const source = `.a{background:image-set(image-set("inner.png" 1x) 1x, "outer.png" 2x)}
    @import url("theme.css") layer(theme); .b{background:url(#local)}`;
  const first = cssResources(source);
  assert.deepEqual(first.map(({ url }) => url), ["inner.png", "outer.png", "theme.css", "#local"]);
  assert.deepEqual(cssResources(source), first);
  assert.deepEqual(cssResources(""), []);
  assert.deepEqual(cssResources(null), []);
});
