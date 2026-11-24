/**
 * Shared static CSS resource discovery for normal and standalone application
 * builds. Source offsets cover only a resource token, preserving image-set
 * density/type descriptors and @import layer/supports/media conditions.
 */

// Comments and ordinary strings must be consumed along with URL tokens: a
// shader preview or CSS content string containing url(...) is not a dependency.
export const CSS_RESOURCE_REGEX =
  /(\/\*[\s\S]*?\*\/)|(@import\s+(?:url\(\s*)?(?:"([^"\\]*(?:\\[\s\S][^"\\]*)*)"|'([^'\\]*(?:\\[\s\S][^'\\]*)*)'|((?:\\(?:[0-9a-f]{1,6}(?:\r\n|[ \t\r\n\f])?|[^\r\n\f0-9a-f])|[^\\'"\s();])+))\s*\)?)|(\burl\(\s*(?:"([^"\\]*(?:\\[\s\S][^"\\]*)*)"|'([^'\\]*(?:\\[\s\S][^'\\]*)*)'|((?:\\(?:[0-9a-f]{1,6}(?:\r\n|[ \t\r\n\f])?|[^\r\n\f0-9a-f])|[^\\'"()\s])+))\s*\))|("(?:[^"\\]|\\[\s\S])*")|('(?:[^'\\]|\\[\s\S])*')/gi;

// Also consume complete function names so a suffix such as my-url(...) never
// becomes a url(...) token. Parentheses isolate image-set candidate strings
// from type(), gradients and other nested functions.
const TOKENS = new RegExp(CSS_RESOURCE_REGEX.source +
  "|([a-z_\\u0080-\\uffff-][a-z0-9_\\u0080-\\uffff-]*\\()|([(),])", "gi");

function decodeUrl(raw) {
  return raw.replace(
    /\\(?:([0-9a-f]{1,6})(?:\r\n|[ \t\r\n\f])?|(\r\n|[\r\n\f])|([\s\S]))/gi,
    (_escape, hex, continuation, character) => {
      if (continuation) return "";
      if (!hex) return character;
      const codePoint = Number.parseInt(hex, 16);
      return codePoint === 0 || codePoint > 0x10ffff || (codePoint >= 0xd800 && codePoint <= 0xdfff)
        ? "\uFFFD" : String.fromCodePoint(codePoint);
    },
  );
}

/** Enumerate URL/import tokens and image-set string candidates in source order.
 * CSS remains browser-owned; this is resource discovery, not CSS evaluation.
 */
export function cssResources(source) {
  if (!source || typeof source !== "string") return [];
  const resources = [], frames = [];
  const tokens = new RegExp(TOKENS.source, TOKENS.flags);
  for (const match of source.matchAll(tokens)) {
    const frame = frames.at(-1);
    if (match[1]) continue;
    if (match[2] || match[6]) {
      const raw = match[3] ?? match[4] ?? match[5] ?? match[7] ?? match[8] ?? match[9];
      // A quoted @import may be followed by layer/supports/media tokens. Leave
      // its separating whitespace outside the rewritten resource span.
      const length = match[2] ? match[0].trimEnd().length : match[0].length;
      resources.push({ start: match.index, end: match.index + length,
        url: decodeUrl(raw), kind: match[2] ? "import" : "url" });
      if (frame) frame.candidate = false;
    } else if (match[10] || match[11]) {
      if (frame?.imageSet && frame.candidate) {
        resources.push({ start: match.index, end: match.index + match[0].length,
          url: decodeUrl(match[0].slice(1, -1)), kind: "image-set" });
        frame.candidate = false;
      }
    } else if (match[12]) {
      if (frame) frame.candidate = false;
      const name = match[12].slice(0, -1).toLowerCase();
      frames.push({ imageSet: name === "image-set" || name === "-webkit-image-set", candidate: true });
    } else if (match[13] === "(") {
      if (frame) frame.candidate = false;
      frames.push({ imageSet: false, candidate: false });
    } else if (match[13] === ")") {
      frames.pop();
    } else if (match[13] === "," && frame?.imageSet) {
      frame.candidate = true;
    }
  }
  return resources;
}
