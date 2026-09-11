/**
 * HTML entry point parser.
 * Extracts import maps, inline and external <script type="module"> elements,
 * and <link rel="modulepreload"> hints while preserving source line offsets.
 *
 * Robustly ignores HTML comments (<!-- ... -->) without altering line/col numbers,
 * handles whitespace around attribute '=' signs, unquoted attribute values,
 * and ignores prefixed attributes like data-type / data-src.
 */

import { IngestionParseError } from './types.mjs';

/**
 * Strips HTML comments while preserving characters and newlines
 * so that line numbers, column numbers, and byte offsets remain exact.
 * @param {string} html
 * @returns {string}
 */
/**
 * Shared contextual scanner regex for HTML documents.
 * Matches HTML comments, <script>...</script>, and <style>...</style> tokens contextually
 * so that comment markers (<!--) inside script or style text are never confused with HTML comments.
 */
export const HTML_CONTEXT_REGEX = /(<!--[\s\S]*?-->)|(<script\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/script\s*>)|(<style\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/style\s*>)/gi;

/**
 * Strips HTML comments while preserving layout coordinates and byte offsets.
 * Recognizes complete script and style rawtext tokens so that comment markers
 * (e.g. const marker = "<!--";) inside script/style bodies are never treated as HTML comments.
 *
 * @param {string} html
 * @returns {string}
 */
export function stripHtmlComments(html) {
  return html.replace(
    HTML_CONTEXT_REGEX,
    (match, comment) => {
      if (comment) {
        return comment.replace(/[^\r\n]/g, ' ');
      }
      return match;
    }
  );
}

/**
 * Masks comments as well as script and style bodies with spaces, preserving opening/closing
 * tags and exact byte offsets. Used for DOM-level asset discovery and base tag inspection
 * to prevent false discoveries inside JS/CSS strings.
 *
 * @param {string} html
 * @returns {string}
 */
export function stripScriptAndStyleBodies(html) {
  return html.replace(
    HTML_CONTEXT_REGEX,
    (match, comment, scriptBlock, scriptAttrs, scriptBody, styleBlock, styleAttrs, styleBody) => {
      if (comment) {
        return comment.replace(/[^\r\n]/g, ' ');
      }
      if (scriptBlock) {
        const openTagLen = 7 + scriptAttrs.length + 1;
        const openTag = scriptBlock.slice(0, openTagLen);
        const closeTag = scriptBlock.slice(openTagLen + scriptBody.length);
        return openTag + scriptBody.replace(/[^\r\n]/g, ' ') + closeTag;
      }
      if (styleBlock) {
        const openTagLen = 6 + styleAttrs.length + 1;
        const openTag = styleBlock.slice(0, openTagLen);
        const closeTag = styleBlock.slice(openTagLen + styleBody.length);
        return openTag + styleBody.replace(/[^\r\n]/g, ' ') + closeTag;
      }
      return match;
    }
  );
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
    const val = match[2] !== undefined
      ? match[2]
      : (match[3] !== undefined ? match[3] : (match[4] !== undefined ? match[4] : ''));
    attrs[name] = val;
  }
  return attrs;
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
export function parseHtmlEntries(rawHtmlContent, documentUrl) {
  const importMap = {
    imports: {},
    scopes: {}
  };
  /** @type {ModuleScriptEntry[]} */
  const moduleScripts = [];
  /** @type {string[]} */
  const preloads = [];

  // Strip comments while preserving layout coordinates
  const sanitizedHtml = stripHtmlComments(rawHtmlContent);

  // Strip script and style bodies for DOM/base/preload discovery
  const domHtml = stripScriptAndStyleBodies(sanitizedHtml);

  // Match <base href="..."> if present to determine effective base URL in DOM
  let baseHref = null;
  const baseRegex = /<base\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi;
  let baseMatch;
  while ((baseMatch = baseRegex.exec(domHtml)) !== null) {
    const attrs = parseTagAttributes(baseMatch[1]);
    if (attrs.href && !baseHref) {
      baseHref = attrs.href;
    }
  }
  const effectiveBaseUrl = baseHref ? new URL(baseHref, documentUrl).href : documentUrl;

  // Match <link rel="modulepreload" ...> using quote-aware attribute scanner on DOM
  const linkRegex = /<link\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi;
  let linkMatch;
  while ((linkMatch = linkRegex.exec(domHtml)) !== null) {
    const attrs = parseTagAttributes(linkMatch[1]);
    if (attrs.rel && attrs.rel.toLowerCase() === 'modulepreload' && attrs.href) {
      preloads.push(attrs.href);
    }
  }

  // Match all <script>...</script> tags with attributes using quote-aware scanner
  const scriptRegex = /<script\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>([\s\S]*?)<\/script\s*>/gi;
  let scriptMatch;
  let inlineModuleIndex = 0;

  while ((scriptMatch = scriptRegex.exec(sanitizedHtml)) !== null) {
    const fullTag = scriptMatch[0];
    const attrString = scriptMatch[1];
    const scriptBody = scriptMatch[2];
    const matchOffset = scriptMatch.index;

    const attrs = parseTagAttributes(attrString);
    const scriptType = (attrs.type || 'text/javascript').toLowerCase();

    // Calculate line and column of script body start using matched attribute length
    // rather than indexOf('>') to prevent corruption when attributes contain quoted '>'
    const openingTagEndIndex = matchOffset + 7 + attrString.length + 1;
    const prefixBeforeBody = rawHtmlContent.slice(0, openingTagEndIndex);
    const lines = prefixBeforeBody.split('\n');
    const startLine = lines.length;
    const startColumn = lines[lines.length - 1].length + 1;

    // Read script body from ORIGINAL rawHtmlContent using matched offsets,
    // not comment-stripped text (which replaced e.g. "<!-- keep me -->" with spaces)
    const rawScriptBody = rawHtmlContent.slice(
      openingTagEndIndex,
      openingTagEndIndex + scriptBody.length
    );

    if (scriptType === 'importmap') {
      try {
        const parsed = JSON.parse(rawScriptBody.trim());
        if (parsed.imports && typeof parsed.imports === 'object') {
          Object.assign(importMap.imports, parsed.imports);
        }
        if (parsed.scopes && typeof parsed.scopes === 'object') {
          Object.assign(importMap.scopes, parsed.scopes);
        }
      } catch (err) {
        throw new IngestionParseError(
          `Failed to parse importmap JSON in ${documentUrl}: ${err.message}`,
          documentUrl,
          { line: startLine, column: startColumn, offset: openingTagEndIndex }
        );
      }
    } else if (scriptType === 'module') {
      if (attrs.src) {
        const externalSrc = attrs.src;
        moduleScripts.push({
          id: new URL(externalSrc, effectiveBaseUrl).href,
          src: externalSrc,
          inlineContent: null,
          startLine,
          startColumn,
          startOffset: openingTagEndIndex
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
          startOffset: openingTagEndIndex
        });
      }
    }
  }

  return { importMap, moduleScripts, preloads, baseUrl: effectiveBaseUrl, baseHref };
}
