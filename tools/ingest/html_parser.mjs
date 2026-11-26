/**
 * HTML entry point parser.
 * Extracts import maps, inline and external <script type="module"> elements,
 * and <link rel="modulepreload"> hints while preserving source line offsets.
 *
 * Robustly ignores HTML comments (<!-- ... -->) without altering line/col numbers,
 * handles whitespace around attribute '=' signs, unquoted attribute values,
 * and ignores prefixed attributes like data-type / data-src.
 */

import { IngestionParseError } from "./types.mjs";
import { normalizeDocumentRoot, resolveDocumentUrl } from "./document_urls.mjs";
import { normalizeSpecifierKey } from "./resolver.mjs";

/** Source-preserving HTML text boundaries shared by discovery and emission.
 * A quote-aware ordinary tag is consumed as a whole, so a displayed tag in an
 * attribute cannot start a second element. RCDATA/raw-text contents belong to
 * their element until its matching end tag (or EOF), never to the module graph.
 * Scripting is enabled for these application builds, including <noscript>.
 * This is not a DOM tree builder: template activation and foreign-content tree
 * construction are not inferred from these lexical text boundaries.
 */
export function* htmlContextTags(html) {
  const tags = /<!--[\s\S]*?(?:-->|$)|<![^>]*>|<\/?([a-z][^\t\n\f\r />]*)(?=[\t\n\f\r />])((?:[^"'<>]|"[^"]*"|'[^']*')*)>/gi;
  const textNames = new Set(["script", "style", "textarea", "title", "xmp", "iframe", "noembed", "noframes", "noscript", "plaintext"]);
  let match;
  while ((match = tags.exec(html)) !== null) {
    const rawName = match[1], name = rawName?.toLowerCase() ?? "";
    const token = {name, rawName, attributes: match[2] ?? "", start: match.index,
      end: tags.lastIndex, bodyStart: tags.lastIndex, body: undefined,
      closing: match[0].startsWith("</"), comment: match[0].startsWith("<!--"), closed: true};
    if (!token.closing && textNames.has(name)) {
      const endTag = new RegExp(`</${name}(?=[\\t\\n\\f\\r />])(?:[^"'<>]|"[^"]*"|'[^']*')*>`, "gi");
      endTag.lastIndex = tags.lastIndex;
      const end = name === "plaintext" ? null : endTag.exec(html);
      token.body = html.slice(token.bodyStart, end?.index ?? html.length);
      token.end = end ? endTag.lastIndex : html.length;
      token.closed = end !== null;
      tags.lastIndex = token.end;
    }
    token.text = html.slice(token.start, token.end);
    yield token;
  }
}

/** Rewrite only actual source tags, preserving every intervening byte. */
export function rewriteHtmlContextTags(html, rewrite) {
  const parts = [];
  let end = 0;
  for (const tag of htmlContextTags(html)) {
    parts.push(html.slice(end, tag.start), rewrite(tag));
    end = tag.end;
  }
  parts.push(html.slice(end));
  return parts.join("");
}

/**
 * Strips HTML comments while preserving layout coordinates and byte offsets.
 * Comment markers inside RCDATA and raw text remain source content.
 *
 * @param {string} html
 * @returns {string}
 */
export function stripHtmlComments(html) {
  return rewriteHtmlContextTags(html, tag => tag.comment ? tag.text.replace(/[^\r\n]/g, " ") : tag.text);
}

/**
 * Masks comments and text-element bodies with spaces, preserving opening/closing
 * tags and exact source offsets. Discovery uses htmlContextTags directly so
 * quoted attribute values cannot introduce a second apparent tag.
 *
 * @param {string} html
 * @returns {string}
 */
export function stripScriptAndStyleBodies(html) {
  return rewriteHtmlContextTags(html, tag => {
    if (tag.comment) return tag.text.replace(/[^\r\n]/g, " ");
    if (tag.body === undefined) return tag.text;
    const openEnd = tag.bodyStart - tag.start;
    return tag.text.slice(0, openEnd) + tag.body.replace(/[^\r\n]/g, " ") +
      tag.text.slice(openEnd + tag.body.length);
  });
}

/**
 * Robustly parses HTML tag attributes into a lowercase key-value dictionary.
 * Supports quoted and unquoted values, whitespace around '=', and exact attribute names.
 * @param {string} attrString
 * @returns {Record<string, string>}
 */
export function parseTagAttributes(attrString) {
  const attrs = Object.create(null);
  const attrRegex = /(?:^|\s+)([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let match;
  while ((match = attrRegex.exec(attrString)) !== null) {
    const name = match[1].toLowerCase();
    if (attrs[name] === undefined) {
      const val =
        match[2] !== undefined
          ? match[2]
          : match[3] !== undefined
            ? match[3]
            : match[4] !== undefined
              ? match[4]
              : "";
      attrs[name] = val;
    }
  }
  return attrs;
}

// Normalize identities before merging separate import maps. Keep original keys
// and addresses for the resolver's prefix diagnostics and document-root checks.
// Within one map, later equivalent keys still win; across maps, existing rules
// (including blocked addresses) cannot be replaced by a later registration.
function mergeSpecifierMaps(previous, incoming, baseUrl, documentRoot) {
  if (Object.keys(previous).length === 0) return {...incoming};
  const defined = new Set(Object.keys(previous).map(key =>
    normalizeSpecifierKey(key, baseUrl, documentRoot)).filter(key => key !== null));
  const additions = Object.entries(incoming).filter(([key]) => {
    const normalized = normalizeSpecifierKey(key, baseUrl, documentRoot);
    return normalized !== null && !defined.has(normalized);
  });
  return {...previous, ...Object.fromEntries(additions)};
}

function mergeScopeMaps(previous, incoming, baseUrl, documentRoot) {
  // The resolver already normalizes a single map. Preserve its source shape,
  // including the last scope when two spellings normalize to the same URL.
  if (Object.keys(previous).length === 0) return {...incoming};
  const normalizedScopes = scopes => {
    const result = new Map();
    for (const [prefix, entries] of Object.entries(scopes)) {
      let url;
      try { url = resolveDocumentUrl(prefix, baseUrl, documentRoot); }
      catch (error) {
        if (error.code?.startsWith("F3D_DOCUMENT_")) throw error;
        continue;
      }
      result.set(url, {prefix, entries});
    }
    return result;
  };
  const existing = normalizedScopes(previous), result = {...previous};
  for (const [url, {prefix, entries}] of normalizedScopes(incoming)) {
    const old = existing.get(url);
    // Define an own data property even for a scope literally named __proto__.
    Object.defineProperty(result, old?.prefix ?? prefix, {
      value: old ? mergeSpecifierMaps(old.entries, entries, baseUrl, documentRoot) : entries,
      enumerable: true, configurable: true, writable: true,
    });
  }
  return result;
}

/**
 * @typedef {Object} ModuleScriptEntry
 * @property {string} id - Canonical module URL or synthetic URL for inline scripts
 * @property {string | null} src - External src attribute if present
 * @property {string | null} inlineContent - Inline code if present
 * @property {number} startLine - 1-based start line of code within HTML
 * @property {number} startColumn - 1-based start column within HTML
 * @property {number} startOffset - 0-based character offset
 */

/**
 * Parse an HTML document and extract module scripts and import maps.
 * @param {string} rawHtmlContent
 * @param {string} documentUrl - Canonical URL of the HTML document
 * @returns {{ importMap: { imports: Record<string, string | null>, scopes: Record<string, Record<string, string | null>> }, moduleScripts: ModuleScriptEntry[], preloads: string[] }}
 */
export function parseHtmlEntries(rawHtmlContent, documentUrl, options = {}) {
  const documentRoot = normalizeDocumentRoot(options.documentRoot);
  const importMap = {
    imports: {},
    scopes: {},
  };
  /** @type {ModuleScriptEntry[]} */
  const moduleScripts = [];
  /** @type {string[]} */
  const preloads = [];

  const tags = [...htmlContextTags(rawHtmlContent)].filter(tag => !tag.closing);

  // Match <base href="..."> if present to determine effective base URL in DOM
  let baseHref = null;
  for (const tag of tags.filter(tag => tag.name === "base")) {
    const attrs = parseTagAttributes(tag.attributes);
    if (baseHref === null && Object.hasOwn(attrs, "href")) {
      baseHref = attrs.href;
    }
  }
  const effectiveBaseUrl = baseHref !== null ? resolveDocumentUrl(baseHref, documentUrl, documentRoot) : documentUrl;

  // Match <link rel="modulepreload" ...> using quote-aware attribute scanner on DOM
  for (const tag of tags.filter(tag => tag.name === "link")) {
    const attrs = parseTagAttributes(tag.attributes);
    if (attrs.rel && attrs.rel.toLowerCase() === "modulepreload" && attrs.href) {
      preloads.push(attrs.href);
    }
  }

  // Match all <script>...</script> tags with attributes using quote-aware scanner
  let inlineModuleIndex = 0;

  for (const tag of tags.filter(tag => tag.name === "script" && tag.closed)) {
    const attrString = tag.attributes;
    const scriptBody = tag.body;

    const attrs = parseTagAttributes(attrString);
    const scriptType = (attrs.type || "text/javascript").toLowerCase();

    // Calculate line and column of script body start using matched attribute length
    // rather than indexOf('>') to prevent corruption when attributes contain quoted '>'
    const openingTagEndIndex = tag.bodyStart;
    const prefixBeforeBody = rawHtmlContent.slice(0, openingTagEndIndex);
    const lines = prefixBeforeBody.split("\n");
    const startLine = lines.length;
    const startColumn = lines[lines.length - 1].length + 1;

    // Read script body from ORIGINAL rawHtmlContent using matched offsets,
    // not comment-stripped text (which replaced e.g. "<!-- keep me -->" with spaces)
    const rawScriptBody = rawHtmlContent.slice(
      openingTagEndIndex,
      openingTagEndIndex + scriptBody.length,
    );

    if (scriptType === "importmap") {
      try {
        const parsed = JSON.parse(rawScriptBody.trim());
        if (parsed.imports && typeof parsed.imports === "object") {
          importMap.imports = mergeSpecifierMaps(importMap.imports, parsed.imports, effectiveBaseUrl, documentRoot);
        }
        if (parsed.scopes && typeof parsed.scopes === "object") {
          importMap.scopes = mergeScopeMaps(importMap.scopes, parsed.scopes, effectiveBaseUrl, documentRoot);
        }
      } catch (err) {
        throw new IngestionParseError(
          `Failed to parse importmap JSON in ${documentUrl}: ${err.message}`,
          documentUrl,
          { line: startLine, column: startColumn, offset: openingTagEndIndex },
        );
      }
    } else if (scriptType === "module") {
      if (Object.hasOwn(attrs, "src")) {
        const externalSrc = attrs.src;
        moduleScripts.push({
          id: resolveDocumentUrl(externalSrc, effectiveBaseUrl, documentRoot),
          src: externalSrc,
          inlineContent: null,
          startLine,
          startColumn,
          startOffset: openingTagEndIndex,
        });
      } else {
        inlineModuleIndex++;
        const syntheticUrl = `${effectiveBaseUrl}#inline-module-${inlineModuleIndex}`;
        moduleScripts.push({
          id: syntheticUrl,
          src: null,
          inlineContent: rawScriptBody,
          startLine,
          startColumn,
          startOffset: openingTagEndIndex,
        });
      }
    }
  }

  return { importMap, moduleScripts, preloads, baseUrl: effectiveBaseUrl, baseHref };
}

/**
 * Checks if a character is an ASCII whitespace character per WHATWG HTML spec:
 * U+0009 TAB, U+000A LF, U+000C FF, U+000D CR, U+0020 SPACE.
 * (Note: non-breaking space U+00A0 is NOT ASCII whitespace).
 *
 * @param {string} ch
 * @returns {boolean}
 */
function isAsciiWhitespace(ch) {
  return ch === " " || ch === "\t" || ch === "\n" || ch === "\r" || ch === "\f";
}

/**
 * Parses a `srcset` attribute value into an array of candidate image URLs,
 * adhering strictly to the WHATWG HTML specification (Section 4.8.4.3.8 "Parse a srcset attribute"):
 * - Collects candidate URLs by advancing until ASCII whitespace (preserving internal commas
 *   in filenames, query params, and data: URLs without any ad-hoc special cases).
 * - Strips trailing commas from the URL token if present; if stripped, candidate has no descriptors.
 * - Parses and validates descriptors (w, x, h) and skips invalid candidates so browser-ignored
 *   candidates do not trigger phantom missing-file failures.
 * - Extracts candidate URLs in document order without modifying the original attribute string.
 *
 * @param {string} srcsetString - Raw value of the srcset attribute
 * @returns {string[]} Candidate image URLs extracted in document order
 */
export function parseSrcsetUrls(srcsetString) {
  if (!srcsetString || typeof srcsetString !== "string") {
    return [];
  }

  const urls = [];
  const input = srcsetString;
  const len = input.length;
  let pos = 0;

  while (pos < len) {
    // 1. Skip leading ASCII whitespace and commas
    while (pos < len && (isAsciiWhitespace(input[pos]) || input[pos] === ",")) {
      pos++;
    }
    if (pos >= len) break;

    // 2. Collect sequence of characters that are NOT ASCII whitespace
    const urlStart = pos;
    while (pos < len && !isAsciiWhitespace(input[pos])) {
      pos++;
    }
    let url = input.slice(urlStart, pos);

    let error = false;
    const descriptors = [];

    // 3. If url ends with U+002C COMMA (,), strip all trailing commas
    if (url.endsWith(",")) {
      let end = url.length;
      while (end > 0 && url[end - 1] === ",") {
        end--;
      }
      url = url.slice(0, end);
      // Trailing comma indicates candidate separator without descriptors
    } else {
      // 4. Descriptor tokenizer: tokenize descriptors separated by ASCII whitespace
      // until a comma outside parentheses (or EOF) is encountered
      let currentDescriptor = "";
      let state = "in_descriptor";

      while (pos < len) {
        const c = input[pos];
        if (state === "in_descriptor") {
          if (isAsciiWhitespace(c)) {
            if (currentDescriptor.length > 0) {
              descriptors.push(currentDescriptor);
              currentDescriptor = "";
            }
            state = "after_descriptor";
          } else if (c === ",") {
            pos++; // consume candidate separator comma
            if (currentDescriptor.length > 0) {
              descriptors.push(currentDescriptor);
              currentDescriptor = "";
            }
            break;
          } else if (c === "(") {
            currentDescriptor += c;
            state = "in_parens";
          } else {
            currentDescriptor += c;
          }
        } else if (state === "in_parens") {
          if (c === ")") {
            currentDescriptor += c;
            state = "in_descriptor";
          } else {
            currentDescriptor += c;
          }
        } else if (state === "after_descriptor") {
          if (isAsciiWhitespace(c)) {
            // stay in after_descriptor
          } else if (c === ",") {
            pos++; // consume candidate separator comma
            break;
          } else if (c === "(") {
            currentDescriptor += c;
            state = "in_parens";
          } else {
            state = "in_descriptor";
            currentDescriptor += c;
          }
        }
        pos++;
      }

      if (currentDescriptor.length > 0) {
        descriptors.push(currentDescriptor);
      }

      // 5. Descriptor parser / validator per WHATWG
      let width = null;
      let density = null;
      let height = null;

      for (const desc of descriptors) {
        // Valid non-negative integer followed by lowercase 'w'
        if (/^[0-9]+w$/.test(desc)) {
          if (width !== null || density !== null) {
            error = true;
            break;
          }
          const wVal = parseInt(desc.slice(0, -1), 10);
          if (wVal === 0) {
            error = true;
            break;
          }
          width = wVal;
        }
        // Valid floating-point number (including leading minus e.g. -0x) followed by lowercase 'x'
        // WHATWG: density must not be less than 0 (rejects negative nonzero, accepts 0 and -0)
        else if (/^-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?x$/.test(desc)) {
          if (width !== null || density !== null || height !== null) {
            error = true;
            break;
          }
          const dVal = parseFloat(desc.slice(0, -1));
          if (isNaN(dVal) || dVal < 0) {
            error = true;
            break;
          }
          density = dVal;
        }
        // Valid non-negative integer followed by lowercase 'h'
        else if (/^[0-9]+h$/.test(desc)) {
          if (height !== null || density !== null) {
            error = true;
            break;
          }
          const hVal = parseInt(desc.slice(0, -1), 10);
          if (hVal === 0) {
            error = true;
            break;
          }
          height = hVal;
        } else {
          // Unknown or malformed descriptor (including uppercase W/X/H)
          error = true;
          break;
        }
      }

      // If future-compat-h is present, width must also be present
      if (height !== null && width === null) {
        error = true;
      }
    }

    // Only append valid candidate if no parse error and URL is non-empty
    if (!error && url.length > 0) {
      urls.push(url);
    }
  }

  return urls;
}
