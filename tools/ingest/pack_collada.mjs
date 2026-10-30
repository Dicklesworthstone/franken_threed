/** Bounded Collada image metadata discovery. The original document, geometry,
 * animation, kinematics and native XML parser remain in the retained loader.
 * r186 reads the first direct library_images and first direct init_from of each
 * image, retaining the last duplicate image ID. Preserve textContent literally;
 * in particular, neither trim paths nor treat every init_from as an image URL.
 */
const error = (code, message) => { throw Object.assign(new Error(`${code}: ${message}`), {code}); };
const NAME = '[A-Za-z_:][A-Za-z0-9_.:-]*';
const opening = new RegExp(`^(${NAME})([\\s\\S]*)$`);
const attribute = new RegExp(`\\s+(${NAME})\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, 'y');
const predefined = {amp: '&', lt: '<', gt: '>', quot: '"', apos: "'"};

export function colladaAssetUris(bytes, fail = error) {
  if (bytes.length > 64 * 1024 * 1024) fail('COLLADA_BUDGET', 'Collada resource discovery exceeds 64 MiB');
  const text = new TextDecoder().decode(bytes).replace(/\r\n?/g, '\n');
  const images = new Map(), stack = [];
  let offset = 0, nodes = 0, librarySeen = false, libraryDepth = -1, image = null, contentDepth = -1;
  const invalid = message => fail('COLLADA_XML', message);
  function decode(value) {
    return value.replace(/&([^;]*);|&/g, (match, name) => {
      if (Object.hasOwn(predefined, name)) return predefined[name];
      const number = /^#x[\da-f]+$/i.test(name) ? parseInt(name.slice(2), 16)
        : /^#\d+$/.test(name) ? Number(name.slice(1)) : NaN;
      if (!(number === 9 || number === 10 || number === 13 ||
          number >= 0x20 && number <= 0xd7ff || number >= 0xe000 && number <= 0xfffd ||
          number >= 0x10000 && number <= 0x10ffff)) invalid('Unresolved or invalid XML entity in image metadata');
      return String.fromCodePoint(number);
    });
  }
  function close(name) {
    if (stack.at(-1) !== name) invalid('Mismatched XML element');
    if (stack.length === contentDepth) contentDepth = -1;
    if (image && stack.length === libraryDepth + 1 && name === 'image') {
      if (image.value !== null) images.set(image.id, image.value);
      image = null;
    }
    if (stack.length === libraryDepth) libraryDepth = -1;
    stack.pop();
  }
  while (offset < text.length) {
    if (text[offset] !== '<') {
      const next = text.indexOf('<', offset), end = next < 0 ? text.length : next;
      if (contentDepth >= 0) image.value += decode(text.slice(offset, end));
      offset = end; continue;
    }
    if (text.startsWith('<!--', offset) || text.startsWith('<?', offset) || text.startsWith('<![CDATA[', offset)) {
      const cdata = text.startsWith('<![CDATA[', offset);
      const prefix = cdata ? 9 : text.startsWith('<!--', offset) ? 4 : 2;
      const suffix = cdata ? ']]>' : prefix === 4 ? '-->' : '?>';
      const end = text.indexOf(suffix, offset + prefix);
      if (end < 0) invalid('Unterminated XML comment, instruction or CDATA');
      if (cdata && contentDepth >= 0) image.value += text.slice(offset + prefix, end);
      offset = end + suffix.length; continue;
    }
    // A valid DTD can change image textContent through custom entities. This is
    // not malformed source: stop closure explicitly, rather than silently
    // publishing only the model while leaving a required image on the network.
    if (/^<!DOCTYPE(?:\s|>)/.test(text.slice(offset)))
      fail('COLLADA_XML_UNSUPPORTED', 'DTD-based Collada image closure requires the original networked deployment');
    if (text.startsWith('<!', offset)) invalid('Invalid XML declaration');
    let end = offset + 1, quote = null;
    for (; end < text.length; end++) {
      const char = text[end];
      if (quote) { if (char === quote) quote = null; }
      else if (char === '"' || char === "'") quote = char;
      else if (char === '>') break;
    }
    if (end === text.length) invalid('Unterminated XML element');
    let body = text.slice(offset + 1, end); offset = end + 1;
    if (body.startsWith('/')) { close(body.slice(1).trim()); continue; }
    const empty = body.endsWith('/');
    if (empty) body = body.slice(0, -1);
    const match = opening.exec(body);
    if (!match) invalid('Invalid XML element name');
    const name = match[1], attrs = new Map();
    let cursor = name.length;
    while (cursor < body.length && body.slice(cursor).trim()) {
      attribute.lastIndex = cursor;
      const part = attribute.exec(body);
      if (!part || attrs.has(part[1])) invalid('Invalid or duplicate XML attribute');
      // XML normalizes literal whitespace before expanding character entities.
      attrs.set(part[1], decode((part[2] ?? part[3]).replace(/[\t\n]/g, ' ')));
      cursor = attribute.lastIndex;
    }
    stack.push(name);
    if (++nodes > 1000000 || stack.length > 128) fail('COLLADA_BUDGET', 'Collada XML exceeds the node/depth work bound');
    if (!librarySeen && stack.length === 2 && stack[0] === 'COLLADA' && name === 'library_images') {
      librarySeen = true; libraryDepth = 2;
    } else if (libraryDepth === 2 && stack.length === 3 && name === 'image') {
      image = {id: attrs.get('id') ?? null, value: null};
    } else if (image && stack.length === 4 && name === 'init_from' && image.value === null) {
      image.value = ''; contentDepth = 4;
    }
    if (empty) close(name);
  }
  if (stack.length) invalid('Unclosed XML element');
  return [...new Set([...images.values()].filter(Boolean))];
}
