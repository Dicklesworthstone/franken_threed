/** Finite library-part discovery for the pinned r186 LDrawLoader. Only file
 * references and MPD section names are read. Geometry, materials, cache keys,
 * asynchronous search and errors remain in the original retained loader.
 */
const error = (code, message) => { throw Object.assign(new Error(`${code}: ${message}`), {code}); };
const MAX_BYTES = 64 * 1024 * 1024;
const inheritedKeys = new Set(Object.getOwnPropertyNames(Object.prototype));

export function ldrawPartReferences(bytes, fileMaps = [{}], fail = error) {
  if (bytes.length > MAX_BYTES) fail('LDRAW_BUDGET', 'LDraw resource discovery exceeds 64 MiB');
  const lines = bytes.toString('utf8').replace(/\r\n/g, '\n').split('\n');
  const references = new Set(), embedded = new Set();
  let work = 0;
  const parse = (start, end, depth) => {
    if (depth > 64) fail('LDRAW_BUDGET', 'LDraw embedded-file nesting exceeds the work bound');
    let embeddedName = null, embeddedStart = end;
    const saveEmbedded = stop => {
      embedded.add(embeddedName.toLowerCase());
      parse(embeddedStart, stop, depth + 1);
    };
    for (let index = start; index < end; index++) {
      if (++work > 1000000) fail('LDRAW_BUDGET', 'LDraw resource discovery exceeds the line work bound');
      const line = lines[index];
      if (embeddedName !== null) {
        // Once MPD parsing starts, source recognizes only this literal prefix.
        if (line.startsWith('0 FILE ')) {
          saveEmbedded(index); embeddedName = line.substring(7); embeddedStart = index + 1;
        }
        continue;
      }
      let offset = 0;
      const skip = () => { while (line[offset] === ' ' || line[offset] === '\t') offset++; };
      const token = () => {
        const begin = offset++;
        while (offset < line.length && line[offset] !== ' ' && line[offset] !== '\t') offset++;
        const value = line.substring(begin, offset); skip(); return value;
      };
      skip(); if (offset >= line.length) continue;
      const type = token();
      if (type === '0') {
        if (token() === 'FILE' && index > start) {
          embeddedName = line.substring(offset); embeddedStart = index + 1;
        }
      } else if (type === '1') {
        for (let field = 0; field < 13; field++) token(); // color, position and 3x3 transform
        const name = line.substring(offset).trim().replace(/\\/g, '/');
        for (const map of fileMaps) {
          if (!Object.hasOwn(map, name) && inheritedKeys.has(name)) continue;
          for (const mapped of map[name] ?? ['']) {
            if (++work > 1000000) fail('LDRAW_BUDGET', 'LDraw resource discovery exceeds the reference work bound');
            const reference = mapped || (name.startsWith('s/') ? 'parts/' + name : name.startsWith('48/') ? 'p/' + name : name);
            if (reference) references.add(reference);
            if (references.size > 100000) fail('LDRAW_BUDGET', 'LDraw resource discovery exceeds the reference count bound');
          }
        }
      }
    }
    if (embeddedName !== null) saveEmbedded(end);
  };
  parse(0, lines.length, 0);
  return [...references].filter(name => !embedded.has(name.toLowerCase()));
}

/** Exact fetchData attempt order, including the lowercase as-is attempt before
 * restarting the search and the repeated final absolute attempt. */
export function ldrawPartCandidates(name) {
  const search = value => ['parts/' + value, 'p/' + value, 'models/' + value,
    value, value.substring(0, value.lastIndexOf('/') + 1) + value];
  const lower = name.toLowerCase();
  return [...search(name), lower, ...search(lower), lower];
}

export function collectLDrawPartAssets(asset, request, add, fail = error) {
  const pending = [asset], seen = new Set();
  let bytes = 0, attempts = 0;
  for (let index = 0; index < pending.length; index++) {
    const current = pending[index];
    if (seen.has(current.originalFileName)) continue;
    seen.add(current.originalFileName); bytes += current.source.length;
    if (seen.size > 4096 || bytes > MAX_BYTES) fail('LDRAW_BUDGET', 'LDraw part closure exceeds the file or byte work bound');
    for (const name of ldrawPartReferences(current.source, request.partFileMaps, fail)) {
      for (const candidate of ldrawPartCandidates(name)) for (const prefix of request.partPaths) {
        if (++attempts > 100000) fail('LDRAW_BUDGET', 'LDraw part search exceeds the request work bound');
        // Copy every existing finite candidate. Do not choose a successful
        // route here: a host response or live path change can alter which one
        // the original asynchronous loader reaches at runtime.
        const part = add(prefix + candidate);
        if (part && !seen.has(part.originalFileName)) pending.push(part);
      }
    }
  }
}
