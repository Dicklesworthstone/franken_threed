/** One build-local snapshot of the project's pre-bundle JavaScript transforms.
 * Files keep their original URL/base and instance identity. This is an input
 * seam for an existing synchronous frontend, not a TypeScript/JSX compiler or
 * a virtual filesystem. JSON and binary resources never pass through it.
 */
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

const hash = source => 'sha256:' + createHash('sha256').update(source).digest('hex');

export function createModuleSources(transformModule) {
  if (transformModule !== undefined && typeof transformModule !== 'function')
    throw new TypeError('transformModule must be a synchronous function');
  const records = new Map();
  return Object.freeze({
    read(url, {sourceType = 'module', source} = {}) {
      url = new URL(url).href;
      if (!['module', 'script'].includes(sourceType))
        throw new TypeError('Only JavaScript modules and classic scripts may be normalized');
      const key = JSON.stringify([url, sourceType]);
      if (records.has(key)) return records.get(key);
      const input = source ?? fs.readFileSync(fileURLToPath(url));
      const original = typeof input === 'string' ? input : input.toString('utf8');
      let code = original;
      if (transformModule) {
        try {
          const result = transformModule(original, Object.freeze({url, sourceType}));
          if (result !== null && result !== undefined) {
            if (typeof result?.then === 'function') {
              // Reject the asynchronous handoff without leaving its eventual
              // failure as an unrelated unhandled rejection in the build host.
              Promise.resolve(result).catch(() => {});
              throw new TypeError('Asynchronous transforms must finish before this synchronous source handoff');
            }
            code = typeof result === 'string' ? result : result.code;
            if (typeof code !== 'string') throw new TypeError('Expected JavaScript code, {code}, or null');
          }
        } catch (cause) {
          const error = new Error(`Source transform failed for ${url}: ${cause?.message ?? String(cause)}`, {cause});
          error.code = 'F3D_SOURCE_TRANSFORM'; error.url = url;
          throw error;
        }
      }
      const record = Object.freeze({code, changed: code !== original,
        bytes: code === original && Buffer.isBuffer(input) ? input : Buffer.from(code),
        originalHash: hash(input), contentHash: hash(code)});
      records.set(key, record);
      return record;
    },
    changed(url) { return records.get(JSON.stringify([new URL(url).href, 'module']))?.changed === true; },
  });
}

// Internal propagation keeps analysis, route proofs and emission on exactly the
// same transformed input. Public standalone graph/bundler calls start a snapshot.
export const moduleSourcesFor = options => options._moduleSources ?? createModuleSources(options.transformModule);
