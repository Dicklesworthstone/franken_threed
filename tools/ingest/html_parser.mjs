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
export function stripHtmlComments(html) {
  return html.replace(/<!--([\s\S]*?)-->/g, (match) => {
    return match.replace(/[^\r\n]/g, ' ');
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

  // Match <link rel="modulepreload" ...>
  const linkRegex = /<link\b([^>]*)>/gi;
  let linkMatch;
  while ((linkMatch = linkRegex.exec(sanitizedHtml)) !== null) {
    const attrs = parseTagAttributes(linkMatch[1]);
    if (attrs.rel && attrs.rel.toLowerCase() === 'modulepreload' && attrs.href) {
      preloads.push(attrs.href);
    }
  }

  // Match all <script>...</script> tags with attributes
  const scriptRegex = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let scriptMatch;
  let inlineModuleIndex = 0;

  while ((scriptMatch = scriptRegex.exec(sanitizedHtml)) !== null) {
    const fullTag = scriptMatch[0];
    const attrString = scriptMatch[1];
    const scriptBody = scriptMatch[2];
    const matchOffset = scriptMatch.index;

    const attrs = parseTagAttributes(attrString);
    const scriptType = (attrs.type || 'text/javascript').toLowerCase();

    // Calculate line and column of script body start
    const openingTagEndIndex = matchOffset + fullTag.indexOf('>') + 1;
    const prefixBeforeBody = rawHtmlContent.slice(0, openingTagEndIndex);
    const lines = prefixBeforeBody.split('\n');
    const startLine = lines.length;
    const startColumn = lines[lines.length - 1].length + 1;

    if (scriptType === 'importmap') {
      try {
        const parsed = JSON.parse(scriptBody.trim());
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
          id: new URL(externalSrc, documentUrl).href,
          src: externalSrc,
          inlineContent: null,
          startLine,
          startColumn,
          startOffset: openingTagEndIndex
        });
      } else {
        inlineModuleIndex++;
        const syntheticUrl = `${documentUrl}#inline-module-${inlineModuleIndex}`;
        moduleScripts.push({
          id: syntheticUrl,
          src: null,
          inlineContent: scriptBody,
          startLine,
          startColumn,
          startOffset: openingTagEndIndex
        });
      }
    }
  }

  return { importMap, moduleScripts, preloads };
}
