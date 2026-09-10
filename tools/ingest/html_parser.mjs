/**
 * HTML entry point parser.
 * Extracts import maps, inline and external <script type="module"> elements,
 * and <link rel="modulepreload"> hints while preserving source line offsets.
 */

import { IngestionParseError } from './types.mjs';

/**
 * @typedef {Object} ImportMapEntry
 * @property {Record<string, string>} imports
 * @property {Record<string, Record<string, string>>} [scopes]
 * @property {number} startLine
 * @property {number} startColumn
 */

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
 * @param {string} htmlContent
 * @param {string} documentUrl - Canonical URL of the HTML document
 * @returns {{ importMap: { imports: Record<string, string>, scopes: Record<string, Record<string, string>> }, moduleScripts: ModuleScriptEntry[], preloads: string[] }}
 */
export function parseHtmlEntries(htmlContent, documentUrl) {
  const importMap = {
    imports: {},
    scopes: {}
  };
  /** @type {ModuleScriptEntry[]} */
  const moduleScripts = [];
  /** @type {string[]} */
  const preloads = [];

  // Match <link rel="modulepreload" ...>
  const linkRegex = /<link\s+[^>]*rel=["']?modulepreload["']?[^>]*>/gi;
  let linkMatch;
  while ((linkMatch = linkRegex.exec(htmlContent)) !== null) {
    const tag = linkMatch[0];
    const hrefMatch = /href=["']([^"']+)["']/i.exec(tag);
    if (hrefMatch) {
      preloads.push(hrefMatch[1]);
    }
  }

  // Match all <script>...</script> tags with attributes
  const scriptRegex = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let scriptMatch;
  let inlineModuleIndex = 0;

  while ((scriptMatch = scriptRegex.exec(htmlContent)) !== null) {
    const fullTag = scriptMatch[0];
    const attrString = scriptMatch[1];
    const scriptBody = scriptMatch[2];
    const matchOffset = scriptMatch.index;

    // Determine type
    const typeMatch = /type=["']?([^"'\s>]+)["']?/i.exec(attrString);
    const scriptType = typeMatch ? typeMatch[1].toLowerCase() : 'text/javascript';

    // Calculate line and column of script body start
    const openingTagEndIndex = matchOffset + fullTag.indexOf('>') + 1;
    const prefixBeforeBody = htmlContent.slice(0, openingTagEndIndex);
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
      const srcMatch = /src=["']([^"']+)["']/i.exec(attrString);
      if (srcMatch) {
        const externalSrc = srcMatch[1];
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
