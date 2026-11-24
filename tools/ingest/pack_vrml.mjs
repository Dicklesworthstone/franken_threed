/** Bounded resource metadata for the pinned r186 VRMLLoader. Only the source
 * syntax and texture-reachable node edges are read; geometry, materials,
 * cameras, DEF/USE object cloning and parse errors remain in the native loader.
 */
const error = (code, message) => { throw Object.assign(new Error(`${code}: ${message}`), {code}); };
const NODE_NAMES = new Set(('Anchor Billboard Collision Group Transform Inline LOD Switch PerspectiveCamera OrthographicCamera ' +
  'AudioClip DirectionalLight PointLight Script Shape Sound SpotLight WorldInfo CylinderSensor PlaneSensor ProximitySensor ' +
  'SphereSensor TimeSensor TouchSensor VisibilitySensor Box Cone Cylinder ElevationGrid Extrusion IndexedFaceSet IndexedLineSet ' +
  'PointSet Sphere Color Coordinate Normal TextureCoordinate Appearance FontStyle ImageTexture Material MovieTexture PixelTexture ' +
  'TextureTransform ColorInterpolator CoordinateInterpolator NormalInterpolator OrientationInterpolator PositionInterpolator ' +
  'ScalarInterpolator Background Fog NavigationInfo Viewpoint Text').split(' '));
const IDENTIFIER = /[^\x30-\x39\0-\x20\x22\x27\x23\x2b\x2c\x2d\x2e\x5b\x5d\x5c\x7b\x7d]([^\0-\x20\x22\x27\x23\x2b\x2c\x2e\x5b\x5d\x5c\x7b\x7d])*/y;
const ROUTE = /[^\x30-\x39\0-\x20\x22\x27\x23\x2b\x2c\x2d\x2e\x5b\x5d\x5c\x7b\x7d][^\0-\x20\x22\x27\x23\x2b\x2c\x2d\x2e\x5b\x5d\x5c\x7b\x7d]*[.][^\x30-\x39\0-\x20\x22\x27\x23\x2b\x2c\x2d\x2e\x5b\x5d\x5c\x7b\x7d][^\0-\x20\x22\x27\x23\x2b\x2c\x2d\x2e\x5b\x5d\x5c\x7b\x7d]*/y;
const STRING = /"(?:[^\\"]|\\[bfnrtv"\\/]|\\u[0-9a-fA-F]{4})*"/y;
const NUMBER = /[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/y;
const HEX = /0[xX][0-9a-fA-F]+/y;
const GROUPS = new Set(['Anchor', 'Group', 'Transform', 'Collision']);
const CHILD_FIELDS = new Map([
  ['Shape', new Set(['appearance', 'geometry'])],
  ['IndexedFaceSet', new Set(['color', 'coord', 'normal', 'texCoord'])],
  ['IndexedLineSet', new Set(['color', 'coord'])],
  ['PointSet', new Set(['color', 'coord'])],
  ['ElevationGrid', new Set(['color', 'normal', 'texCoord'])],
]);

export function vrmlAssetUris(bytes, fail = error) {
  if (bytes.length > 64 * 1024 * 1024) fail('VRML_BUDGET', 'VRML resource discovery exceeds 64 MiB');
  const text = new TextDecoder().decode(bytes), tokens = [];
  const invalid = message => fail('VRML_PARSE', message);
  if (!text.includes('#VRML V2.0')) return [];
  let offset = 0;
  const match = regex => {regex.lastIndex = offset; return regex.exec(text)?.[0];};
  const emit = (type, value, length) => {
    if (tokens.length >= 1000000) fail('VRML_BUDGET', 'VRML metadata exceeds the token work bound');
    tokens.push({type, value}); offset += length;
  };
  while (offset < text.length) {
    const char = text[offset];
    if (/[ ,\s]/.test(char)) {offset++; continue;}
    if (char === '#') {
      const line = text.slice(offset).match(/^[^\r\n\u2028\u2029]*/)[0];
      if (line.startsWith('#VRML')) emit('version', line, line.length); else offset += line.length;
      continue;
    }
    if ('[]{}'.includes(char)) {emit(char, char, 1); continue;}
    if (char === '"') {
      const raw = match(STRING);
      if (!raw) invalid('Unterminated or invalid VRML string');
      // The pinned visitor removes ALL quote characters without unescaping.
      emit('string', raw.replace(/'|"/g, ''), raw.length); continue;
    }
    const identifier = match(IDENTIFIER);
    if (identifier) {
      const route = match(ROUTE);
      if (route && route.length > identifier.length) {emit('route', route, route.length); continue;}
      const literal = ['TRUE', 'FALSE', 'NULL'].find(value => identifier.startsWith(value));
      if (literal) {emit(literal, literal === 'NULL' ? null : literal === 'TRUE', literal.length); continue;}
      emit(NODE_NAMES.has(identifier) ? 'node' : ['DEF', 'USE', 'ROUTE', 'TO'].includes(identifier) ? identifier : 'id', identifier, identifier.length);
      continue;
    }
    const hex = match(HEX);
    if (hex) {emit('hex', hex, hex.length); continue;}
    const number = match(NUMBER);
    if (number) {emit('number', parseFloat(number), number.length); continue;}
    invalid('Invalid VRML token');
  }
  let cursor = 0, nodes = 0;
  const take = type => {
    const token = tokens[cursor++];
    if (token?.type !== type) invalid('Unexpected VRML token');
    return token.value;
  };
  const identifier = () => tokens[cursor]?.type === 'node' ? take('node') : take('id');
  const readNode = depth => {
    if (depth > 128 || ++nodes > 100000) fail('VRML_BUDGET', 'VRML metadata exceeds the node/depth work bound');
    let def = null;
    if (tokens[cursor]?.type === 'DEF') {take('DEF'); def = identifier();}
    const node = {name: take('node'), def, fields: []}; take('{');
    while (tokens[cursor]?.type === 'id') {
      const field = {name: take('id'), type: null, values: []}, grouped = new Map();
      const array = tokens[cursor]?.type === '[';
      if (array) take('[');
      while (true) {
        let type = tokens[cursor]?.type, value;
        if (type === 'node' || type === 'DEF') {value = readNode(depth + 1); type = 'node';}
        else if (type === 'USE') {take('USE'); value = {use: identifier()}; type = 'use';}
        else if (['string', 'number', 'hex', 'NULL', ...(array ? [] : ['TRUE', 'FALSE'])].includes(type)) value = take(type);
        else break;
        if (!grouped.has(type)) grouped.set(type, []);
        grouped.get(type).push(value);
      }
      if (array) take(']'); else if (!grouped.size) invalid('Missing VRML field value');
      // The source CST visitor groups mixed field kinds in this order, rather
      // than preserving their interleaving. Its final type controls DEF walks.
      for (const type of ['node', 'use', 'string', 'number', 'hex', 'TRUE', 'FALSE', 'NULL']) {
        if (!grouped.has(type)) continue;
        field.type = ['TRUE', 'FALSE'].includes(type) ? 'boolean' : type === 'NULL' ? 'null' : type;
        for (const value of grouped.get(type)) field.values.push(value);
      }
      node.fields.push(field);
    }
    take('}'); return node;
  };
  take('version');
  const roots = [];
  while (['node', 'DEF'].includes(tokens[cursor]?.type)) roots.push(readNode(0));
  if (!roots.length) invalid('Missing VRML scene node');
  while (tokens[cursor]?.type === 'ROUTE') {take('ROUTE'); take('route'); take('TO'); take('route');}
  if (cursor !== tokens.length) invalid('Unconsumed VRML metadata');

  const definitions = new Map(), visited = new Set(), uris = new Set();
  const define = node => {
    if (node.def) definitions.set(node.def, node);
    for (const field of node.fields) if (field.type === 'node') for (const child of field.values) define(child);
  };
  for (const node of roots) define(node);
  const visit = (node, depth = 0) => {
    if (node?.use) node = definitions.get(node.use);
    if (!node?.fields || visited.has(node)) return;
    if (depth > 128) fail('VRML_BUDGET', 'VRML resource references exceed the depth work bound');
    visited.add(node);
    for (const field of node.fields) {
      const first = field.values[0];
      if (node.name === 'ImageTexture' && field.name === 'url') {
        if (first) uris.add(String(first)); // Only the FIRST URL is loaded; later url fields still execute.
      } else if (GROUPS.has(node.name) && field.name === 'children') {
        for (const child of field.values) visit(child, depth + 1);
      } else if (CHILD_FIELDS.get(node.name)?.has(field.name)) visit(first, depth + 1);
      else if (node.name === 'Appearance') {
        if (['material', 'textureTransform'].includes(field.name) || field.name === 'texture' &&
            ['ImageTexture', 'PixelTexture'].includes(first?.name)) visit(first, depth + 1);
      }
    }
  };
  for (const node of roots) visit(node);
  return [...uris];
}
