/**
 * Schema types and version constants for FrankenThreeD module ingestion (f3d-04).
 */

export const SCHEMA_VERSION = '1.0.0';

export class IngestionResolutionError extends Error {
  /**
   * @param {string} message
   * @param {string} specifier
   * @param {string} referrerUrl
   * @param {{ line: number, column: number, offset: number } | null} [span]
   */
  constructor(message, specifier, referrerUrl, span = null) {
    super(message);
    this.name = 'IngestionResolutionError';
    this.specifier = specifier;
    this.referrerUrl = referrerUrl;
    this.span = span;
  }
}

export class IngestionParseError extends Error {
  /**
   * @param {string} message
   * @param {string} url
   * @param {{ line: number, column: number, offset: number } | null} [span]
   */
  constructor(message, url, span = null) {
    super(message);
    this.name = 'IngestionParseError';
    this.url = url;
    this.span = span;
  }
}
