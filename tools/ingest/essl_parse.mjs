/** GLSL ES 3.00 parser producing a small AST for essl_wgsl.mjs.
 *
 * Declarations: precision, struct, variables (storage/interpolation/layout/
 * invariant qualifiers), uniform interface blocks, function prototypes and
 * definitions. Statements and expressions follow ESSL 3.00 chapter 5/6 grammar
 * and precedence, including the comma operator, ?:, compound assignment and
 * array constructors. Type checking happens in essl_wgsl.mjs.
 */
import {essError} from './essl_preprocess.mjs';

export const BASIC_TYPES = new Set(['void', 'bool', 'int', 'uint', 'float',
  'vec2', 'vec3', 'vec4', 'bvec2', 'bvec3', 'bvec4', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4',
  'mat2', 'mat3', 'mat4', 'mat2x2', 'mat2x3', 'mat2x4', 'mat3x2', 'mat3x3', 'mat3x4', 'mat4x2', 'mat4x3', 'mat4x4',
  'sampler2D', 'sampler3D', 'samplerCube', 'sampler2DShadow', 'samplerCubeShadow', 'sampler2DArray', 'sampler2DArrayShadow',
  'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray', 'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray']);
const STORAGE = new Set(['const', 'uniform', 'in', 'out', 'inout', 'attribute', 'varying', 'centroid']);
const INTERP = new Set(['smooth', 'flat']);
const PRECISION = new Set(['highp', 'mediump', 'lowp']);
const ASSIGN = new Set(['=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=']);
const BINARY = [['||'], ['^^'], ['&&'], ['|'], ['^'], ['&'], ['==', '!='], ['<', '>', '<=', '>='], ['<<', '>>'], ['+', '-'], ['*', '/', '%']];

export function parse(tokens) {
  let p = 0;
  const structs = new Set();
  const peek = (k = 0) => tokens[p + k];
  const at = (v, k = 0) => peek(k)?.v === v && peek(k).t !== 'num';
  const line = () => peek()?.line ?? tokens.at(-1)?.line;
  const next = () => { const t = tokens[p++]; if (!t) essError('Unexpected end of shader', tokens.at(-1)?.line); return t; };
  const expect = v => { const t = next(); if (t.v !== v || t.t === 'num') essError(`Expected '${v}' but found '${t.v}'`, t.line); return t; };
  const ident = () => { const t = next(); if (t.t !== 'id') essError(`Expected identifier, found '${t.v}'`, t.line); return t.v; };
  const isTypeName = t => t?.t === 'id' && (BASIC_TYPES.has(t.v) || structs.has(t.v));

  function qualifiers() {
    const q = {storage: null, interp: null, layout: null, invariant: false, precision: null};
    for (;;) {
      const t = peek(); if (t?.t !== 'id') break;
      if (STORAGE.has(t.v)) {
        next();
        if (t.v === 'centroid') continue;
        const storage = t.v === 'attribute' ? 'in' : t.v === 'varying' ? null : t.v;
        // `varying` resolves from the stage (three defines it as in/out anyway).
        q.storage = t.v === 'varying' ? 'varying' : storage;
      } else if (INTERP.has(t.v)) { next(); q.interp = t.v; }
      else if (PRECISION.has(t.v)) { next(); q.precision = t.v; }
      else if (t.v === 'invariant') { next(); q.invariant = true; }
      else if (t.v === 'layout') {
        next(); expect('('); q.layout = {};
        for (;;) {
          const name = ident();
          if (at('=')) { next(); const v = next(); q.layout[name] = Number(v.v.replace(/[uU]$/, '')); } else q.layout[name] = true;
          if (at(',')) { next(); continue; }
          expect(')'); break;
        }
      } else break;
    }
    return q;
  }
  function typeSpecifier() {
    if (at('struct')) return structSpecifier();
    const t = next();
    if (!isTypeName(t)) essError(`Expected a type, found '${t.v}'`, t.line);
    const type = {name: t.v, array: null};
    if (at('[')) { next(); type.array = at(']') ? 'unsized' : expression(); expect(']'); }
    return type;
  }
  function structSpecifier() {
    const l = line(); expect('struct');
    const name = peek()?.t === 'id' && !at('{') ? ident() : `f3d_anon_${l}_${p}`;
    structs.add(name);
    expect('{'); const fields = [];
    while (!at('}')) {
      const q = qualifiers(); if (q.storage || q.interp) essError('Qualifiers are not allowed on struct members', line());
      const type = typeSpecifier();
      for (;;) {
        const field = {type, name: ident(), array: null, line: line()};
        if (at('[')) { next(); field.array = expression(); expect(']'); }
        fields.push(field);
        if (at(',')) { next(); continue; }
        break;
      }
      expect(';');
    }
    expect('}');
    return {name, array: null, struct: {name, fields, line: l}};
  }
  function declarators(q, type, l) {
    const list = [];
    for (;;) {
      const d = {name: ident(), array: null, init: null, line: line()};
      if (at('[')) { next(); d.array = at(']') ? 'unsized' : expression(); expect(']'); }
      if (at('=')) { next(); d.init = assignment(); }
      list.push(d);
      if (at(',')) { next(); continue; }
      break;
    }
    return {k: 'var', q, type, list, line: l};
  }

  // ---- expressions --------------------------------------------------------
  function expression() {
    const first = assignment();
    if (!at(',')) return first;
    const list = [first];
    while (at(',')) { next(); list.push(assignment()); }
    return {k: 'seq', list, line: first.line};
  }
  function assignment() {
    const l = line(), left = conditional();
    if (peek()?.t === 'op' && ASSIGN.has(peek().v)) {
      const op = next().v;
      return {k: 'assign', op, target: left, value: assignment(), line: l};
    }
    return left;
  }
  function conditional() {
    const l = line(), c = binary(0);
    if (!at('?')) return c;
    next(); const a = expression(); expect(':'); const b = assignment();
    return {k: 'cond', c, a, b, line: l};
  }
  function binary(level) {
    if (level === BINARY.length) return unary();
    let left = binary(level + 1);
    while (peek()?.t === 'op' && BINARY[level].includes(peek().v)) {
      const l = line(), op = next().v;
      left = {k: 'bin', op, a: left, b: binary(level + 1), line: l};
    }
    return left;
  }
  function unary() {
    const t = peek();
    if (t?.t === 'op' && ['+', '-', '!', '~', '++', '--'].includes(t.v)) {
      next();
      const operand = unary();
      if (t.v === '++' || t.v === '--') return {k: 'preinc', op: t.v, a: operand, line: t.line};
      return {k: 'unary', op: t.v, a: operand, line: t.line};
    }
    return postfix(primary());
  }
  function args() {
    expect('(');
    const list = [];
    if (at('void') && at(')', 1)) { next(); }
    if (!at(')')) for (;;) { list.push(assignment()); if (at(',')) { next(); continue; } break; }
    expect(')');
    return list;
  }
  function primary() {
    const t = peek(), l = t?.line;
    if (!t) essError('Unexpected end of expression', line());
    if (t.t === 'num') { next(); return {k: 'num', v: t.v, line: l}; }
    if (t.t === 'id' && (t.v === 'true' || t.v === 'false')) { next(); return {k: 'bool', v: t.v === 'true', line: l}; }
    if (at('(')) { next(); const e = expression(); expect(')'); return {k: 'paren', e, line: l}; }
    if (isTypeName(t)) {
      next();
      let array = null;
      if (at('[')) { next(); array = at(']') ? 'unsized' : expression(); expect(']'); }
      return {k: 'ctor', type: {name: t.v, array}, args: args(), line: l};
    }
    if (t.t === 'id') {
      next();
      if (at('(')) return {k: 'call', name: t.v, args: args(), line: l};
      return {k: 'id', name: t.v, line: l};
    }
    essError(`Unexpected '${t.v}'`, l);
  }
  function postfix(e) {
    for (;;) {
      const l = line();
      if (at('[')) { next(); const index = expression(); expect(']'); e = {k: 'index', a: e, index, line: l}; continue; }
      if (at('.')) {
        next(); const name = ident();
        if (at('(')) { e = {k: 'method', a: e, name, args: args(), line: l}; continue; }
        e = {k: 'field', a: e, name, line: l}; continue;
      }
      if (at('++') || at('--')) { e = {k: 'postinc', op: next().v, a: e, line: l}; continue; }
      return e;
    }
  }

  // ---- statements ---------------------------------------------------------
  function startsDeclaration() {
    const t = peek();
    if (!t || t.t !== 'id') return false;
    if (STORAGE.has(t.v) || INTERP.has(t.v) || PRECISION.has(t.v) || t.v === 'struct' || t.v === 'invariant' || t.v === 'layout') return true;
    if (!isTypeName(t)) return false;
    // `vec3 a`, `float[3] a`, `S a` versus constructor calls `vec3(...)`.
    if (peek(1)?.t === 'id') return true;
    if (at('[', 1)) {
      let depth = 0, k = 1;
      for (; peek(k); k++) { if (at('[', k)) depth++; else if (at(']', k)) { depth--; if (!depth) break; } }
      return peek(k + 1)?.t === 'id';
    }
    return false;
  }
  function statement() {
    const t = peek(), l = t?.line;
    if (at('{')) return block();
    if (at(';')) { next(); return {k: 'empty', line: l}; }
    if (t.t === 'id') switch (t.v) {
      case 'if': {
        next(); expect('('); const c = expression(); expect(')');
        const a = statement(); let b = null;
        if (at('else')) { next(); b = statement(); }
        return {k: 'if', c, a, b, line: l};
      }
      case 'for': {
        next(); expect('(');
        const init = at(';') ? (next(), null) : simpleStatement();
        const cond = at(';') ? null : conditionOrDeclaration(); expect(';');
        const step = at(')') ? null : expression(); expect(')');
        return {k: 'for', init, cond, step, body: statement(), line: l};
      }
      case 'while': { next(); expect('('); const c = conditionOrDeclaration(); expect(')'); return {k: 'while', c, body: statement(), line: l}; }
      case 'do': { next(); const body = statement(); expect('while'); expect('('); const c = expression(); expect(')'); expect(';'); return {k: 'do', c, body, line: l}; }
      case 'switch': {
        next(); expect('('); const e = expression(); expect(')'); expect('{');
        const cases = [];
        while (!at('}')) {
          if (at('case')) { next(); const v = expression(); expect(':'); cases.push({value: v, body: []}); continue; }
          if (at('default')) { next(); expect(':'); cases.push({value: null, body: []}); continue; }
          if (!cases.length) essError('Statement before the first case', line());
          cases.at(-1).body.push(statement());
        }
        expect('}');
        return {k: 'switch', e, cases, line: l};
      }
      case 'break': next(); expect(';'); return {k: 'break', line: l};
      case 'continue': next(); expect(';'); return {k: 'continue', line: l};
      case 'discard': next(); expect(';'); return {k: 'discard', line: l};
      case 'return': { next(); const e = at(';') ? null : expression(); expect(';'); return {k: 'return', e, line: l}; }
    }
    return simpleStatement();
  }
  function conditionOrDeclaration() {
    if (startsDeclaration()) {
      const l = line(), q = qualifiers(), type = typeSpecifier(), name = ident(); expect('=');
      return {k: 'condDecl', q, type, name, init: assignment(), line: l};
    }
    return expression();
  }
  function simpleStatement() {
    const l = line();
    if (startsDeclaration()) {
      const q = qualifiers();
      if (at(';')) { next(); return {k: 'empty', line: l}; }
      const type = typeSpecifier();
      if (at(';')) { next(); return type.struct ? {k: 'struct', struct: type.struct, line: l} : {k: 'empty', line: l}; }
      const d = declarators(q, type, l); expect(';');
      return type.struct ? {k: 'block', list: [{k: 'struct', struct: type.struct, line: l}, d], scoped: false, line: l} : d;
    }
    const e = expression(); expect(';');
    return {k: 'expr', e, line: l};
  }
  function block() {
    const l = line(); expect('{'); const list = [];
    while (!at('}')) list.push(statement());
    expect('}');
    return {k: 'block', list, scoped: true, line: l};
  }

  // ---- external declarations ---------------------------------------------
  const decls = [];
  while (p < tokens.length) {
    const l = line();
    if (at(';')) { next(); continue; }
    if (at('precision')) { next(); qualifiers(); typeSpecifier(); expect(';'); continue; }
    if (at('invariant') && peek(1)?.t === 'id' && at(';', 2)) { next(); next(); next(); continue; }
    const q = qualifiers();
    // Uniform interface block: uniform Name { ... } [instance];
    if (q.storage === 'uniform' && peek()?.t === 'id' && !isTypeName(peek()) && at('{', 1)) {
      const blockName = ident(); expect('{'); const fields = [];
      while (!at('}')) {
        const fq = qualifiers(); void fq;
        const type = typeSpecifier();
        for (;;) {
          const f = {type, name: ident(), array: null, line: line()};
          if (at('[')) { next(); f.array = expression(); expect(']'); }
          fields.push(f); if (at(',')) { next(); continue; } break;
        }
        expect(';');
      }
      expect('}');
      const instance = peek()?.t === 'id' ? ident() : null;
      if (instance && at('[')) essError('Arrays of uniform blocks are not supported', line());
      expect(';');
      decls.push({k: 'ublock', name: blockName, instance, fields, line: l});
      continue;
    }
    if (at(';')) { next(); continue; }
    const type = typeSpecifier();
    if (at(';')) { next(); if (type.struct) decls.push({k: 'struct', struct: type.struct, line: l}); continue; }
    if (type.struct) decls.push({k: 'struct', struct: type.struct, line: l});
    if (peek()?.t === 'id' && at('(', 1)) {
      const name = ident(); expect('(');
      const params = [];
      if (at('void') && at(')', 1)) next();
      if (!at(')')) for (;;) {
        const pq = qualifiers(), ptype = typeSpecifier();
        const param = {q: pq.storage ?? 'in', type: ptype, name: null, array: null, line: line()};
        if (pq.storage === 'const') param.q = 'in';
        if (peek()?.t === 'id') param.name = ident();
        if (at('[')) { next(); param.array = expression(); expect(']'); }
        params.push(param);
        if (at(',')) { next(); continue; } break;
      }
      expect(')');
      if (at(';')) { next(); decls.push({k: 'proto', ret: type, name, params, line: l}); continue; }
      decls.push({k: 'func', ret: type, name, params, body: block(), line: l});
      continue;
    }
    const d = declarators(q, type, l); expect(';');
    decls.push(d);
  }
  return decls;
}
