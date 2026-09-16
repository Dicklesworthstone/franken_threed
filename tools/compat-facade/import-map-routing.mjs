/** Target-preserving development routing; no application code is evaluated. */
function record(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value;
}

function routeTarget(specifier, target, base) {
  if (typeof target !== 'string') return target;
  const split = target.search(/[?#]/);
  let url;
  try { url = new URL(target, 'https://f3d.invalid/'); } catch { return target; }
  if (!['http:', 'https:', 'file:'].includes(url.protocol)) return target;
  const pathname = url.pathname;
  const suffix = split < 0 ? '' : target.slice(split);
  const file = pathname.slice(pathname.lastIndexOf('/') + 1);
  const bundles = new Map([
    ['three.webgpu.js', 'webgpu.js'], ['three.tsl.js', 'tsl.js'],
    ['three.module.js', 'three.js'], ['three.js', 'three.js'],
  ]);
  if (bundles.has(file)) return `${base}/compat-facade/${bundles.get(file)}${suffix}`;

  // Preserve the selected addon, not merely its import-map alias. Avoid
  // rerouting an unrelated package just because its directory is named jsm.
  const addon = /(?:^|\/)examples\/jsm\/(.*)$/.exec(pathname)
    || (/^(?:\.\/)?jsm\//.test(target) && /^\/jsm\/(.*)$/.exec(pathname));
  if (addon && !addon[1].split('/').some((part) => /^(?:\.|%2e){1,2}$/i.test(part))) {
    return `${base}/compat-facade/addons/${addon[1]}${suffix}`;
  }
  if (specifier === 'three/addons/' && pathname.endsWith('/')) {
    return `${base}/compat-facade/addons/${suffix}`;
  }
  return target;
}

/**
 * Route imports and every scope without changing keys, blocked entries, or
 * unrelated metadata. Integrity-pinned rewrites fail closed: a retained-file
 * hash cannot authenticate the different bytes of a generated facade.
 */
export function routeImportMap(source, { baseUrl = '' } = {}) {
  record(source, 'Import map');
  const base = baseUrl.replace(/\/+$/, '');
  const integrity = source.integrity === undefined ? null : record(source.integrity, 'Import map integrity');
  const hasIntegrity = integrity !== null && Object.keys(integrity).length > 0;
  const routeEntries = (entries, label) => Object.fromEntries(
    Object.entries(record(entries, label)).map(([specifier, target]) => {
      const routed = routeTarget(specifier, target, base);
      if (hasIntegrity && routed !== target) {
        throw new Error('Cannot reroute an integrity-pinned import map without regenerated facade integrity hashes');
      }
      return [specifier, routed];
    }),
  );
  const result = { ...source };
  if (source.imports !== undefined) result.imports = routeEntries(source.imports, 'Import map imports');
  if (source.scopes !== undefined) {
    result.scopes = Object.fromEntries(Object.entries(record(source.scopes, 'Import map scopes'))
      .map(([scope, imports]) => [scope, routeEntries(imports, `Import map scope ${JSON.stringify(scope)}`)]));
  }
  return result;
}

export function createDevImportMap({ baseUrl = '', sourceImportMap = null, routeThreeToWebgpu = false } = {}) {
  const base = baseUrl.replace(/\/+$/, '');
  const routed = routeImportMap(sourceImportMap ?? {}, { baseUrl });
  const imports = {
    three: `${base}/compat-facade/${routeThreeToWebgpu ? 'webgpu' : 'three'}.js`,
    'three/webgpu': `${base}/compat-facade/webgpu.js`,
    'three/tsl': `${base}/compat-facade/tsl.js`,
    'three/addons/': `${base}/compat-facade/addons/`,
    ...routed.imports,
  };
  // An explicit null remains blocked. Preserve query/fragment module identity
  // when a WebGPU-first source supplies only the root alias.
  if (routeThreeToWebgpu && imports.three !== null) {
    if (routed.integrity && Object.keys(routed.integrity).length &&
        Object.hasOwn(routed.imports ?? {}, 'three') && imports.three !== imports['three/webgpu']) {
      throw new Error('Cannot reroute an integrity-pinned import map without regenerated facade integrity hashes');
    }
    imports.three = imports['three/webgpu'];
  } else if (!Object.hasOwn(routed.imports ?? {}, 'three/webgpu') &&
      typeof imports.three === 'string' && imports.three.split(/[?#]/)[0] === `${base}/compat-facade/webgpu.js`) {
    imports['three/webgpu'] = imports.three;
  }
  return { ...routed, imports };
}

const RAW_TEXT = new Set(['script', 'style', 'textarea', 'title', 'xmp', 'iframe', 'noembed', 'noframes', 'noscript']);

// Scan HTML tokens instead of searching for a script-shaped substring. Quoted
// attribute values, comments, raw-text elements and inert templates are skipped.
function inlineImportMaps(html) {
  const tags = /<!--[\s\S]*?(?:-->|$)|<![^>]*>|<\/?([A-Za-z][\w:-]*)\b(?:[^"'<>]|"[^"]*"|'[^']*')*>/g;
  const maps = [];
  let templates = 0;
  for (let tag; (tag = tags.exec(html));) {
    if (!tag[1]) continue;
    const name = tag[1].toLowerCase();
    const closing = tag[0].startsWith('</');
    if (name === 'template') {
      templates = Math.max(0, templates + (closing ? -1 : 1));
      continue;
    }
    if (closing) continue;
    if (name === 'plaintext') break;
    if (!RAW_TEXT.has(name)) continue;
    const bodyStart = tags.lastIndex;
    const close = new RegExp(`</${name}\\s*>`, 'gi');
    close.lastIndex = bodyStart;
    const end = close.exec(html);
    tags.lastIndex = end ? close.lastIndex : html.length;
    if (name !== 'script' || templates) continue;
    const attrs = new Map();
    const attributes = tag[0].slice(1 + tag[1].length, -1);
    const attrPattern = /([^\s"'<>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
    for (const attr of attributes.matchAll(attrPattern)) {
      const key = attr[1].toLowerCase();
      if (!attrs.has(key)) attrs.set(key, attr[2] ?? attr[3] ?? attr[4] ?? '');
    }
    if (attrs.get('type')?.toLowerCase() !== 'importmap') continue;
    if (!end) throw new SyntaxError('Unclosed import-map script');
    if (attrs.has('src')) throw new TypeError('Import maps must be inline, not script src references');
    maps.push({start: bodyStart, end: end.index});
  }
  return maps;
}

/**
 * Rewrite only inline import-map bodies. Opening tags (including CSP nonces),
 * scopes, application code and the rest of the HTML remain intact. With several
 * maps, do not inject defaults that could shadow a later map's selected targets.
 */
export function transformHtmlImportMap(html, { baseUrl = '' } = {}) {
  const maps = inlineImportMaps(html);
  if (!maps.length) throw new Error('No <script type="importmap"> tag found in HTML document');
  const base = baseUrl.replace(/\/+$/, '');
  let output = '';
  let cursor = 0;
  for (const map of maps) {
    let source;
    try {
      source = JSON.parse(html.slice(map.start, map.end));
    } catch (cause) {
      throw new SyntaxError('Invalid import map JSON', { cause });
    }
    const routed = routeImportMap(source, { baseUrl });
    if (maps.length === 1) {
      const imports = routed.imports ?? (routed.imports = {});
      if (!Object.hasOwn(imports, 'three/webgpu')) {
        imports['three/webgpu'] = typeof imports.three === 'string' &&
          imports.three.split(/[?#]/)[0] === `${base}/compat-facade/webgpu.js`
          ? imports.three : `${base}/compat-facade/webgpu.js`;
      }
      if (!Object.hasOwn(imports, 'three')) imports.three = imports['three/webgpu'];
    }
    // Literal '<' can terminate an HTML script even inside a JSON string.
    const json = JSON.stringify(routed, null, 2).replace(/</g, '\\u003c');
    output += html.slice(cursor, map.start) + '\n' + json + '\n';
    cursor = map.end;
  }
  return output + html.slice(cursor);
}
