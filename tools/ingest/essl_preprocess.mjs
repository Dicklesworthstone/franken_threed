/** GLSL ES 3.00 preprocessing and tokenization for the ESSL -> WGSL compiler.
 *
 * Implements the ESSL 3.00 section 3.4 directives this compiler needs before
 * parsing: comments, line continuation, object/function-like #define (with
 * argument pre-expansion, # stringizing is not part of ESSL, ## pasting is
 * rejected), #undef, #if/#ifdef/#ifndef/#elif/#else/#endif with defined() and
 * integer expressions, #error. #version/#extension/#pragma/#line are recorded
 * and dropped. Undefined identifiers in #if evaluate to 0 (ESSL rule).
 * Errors throw EsslError with a line number; nothing is silently skipped.
 */
export class EsslError extends Error {
  constructor(message, line) { super(line ? `ESSL line ${line}: ${message}` : `ESSL: ${message}`); this.name = 'EsslError'; this.code = 'ESSL_COMPILE'; this.line = line; }
}
export const essError = (message, line) => { throw new EsslError(message, line); };

const PUNCT = ['<<=', '>>=', '++', '--', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '^^', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
  '##', '(', ')', '[', ']', '{', '}', '.', ',', ';', ':', '?', '+', '-', '*', '/', '%', '<', '>', '=', '!', '~', '&', '|', '^', '#'];
const ID_START = /[A-Za-z_]/, ID_PART = /[A-Za-z0-9_]/, DIGIT = /[0-9]/;

/** Tokenize one already comment-free text. Tokens: {t:'id'|'num'|'op', v, line, ws}
 * where ws marks preceding whitespace (needed to tell `#define F(` from `#define F (`). */
export function tokenize(text, line = 1) {
  const out = [];
  let i = 0, ws = false;
  while (i < text.length) {
    const c = text[i];
    if (c === '\n') { line++; i++; ws = true; continue; }
    if (c === ' ' || c === '\t' || c === '\r' || c === '\f' || c === '\v') { i++; ws = true; continue; }
    if (ID_START.test(c)) {
      let j = i + 1; while (j < text.length && ID_PART.test(text[j])) j++;
      out.push({t: 'id', v: text.slice(i, j), line, ws}); i = j; ws = false; continue;
    }
    if (DIGIT.test(c) || (c === '.' && DIGIT.test(text[i + 1] ?? ''))) {
      let j = i;
      if (c === '0' && /[xX]/.test(text[i + 1] ?? '')) { j += 2; while (j < text.length && /[0-9a-fA-F]/.test(text[j])) j++; }
      else {
        while (j < text.length && DIGIT.test(text[j])) j++;
        if (text[j] === '.') { j++; while (j < text.length && DIGIT.test(text[j])) j++; }
        if (/[eE]/.test(text[j] ?? '') && /[0-9+-]/.test(text[j + 1] ?? '')) {
          j++; if (/[+-]/.test(text[j])) j++; while (j < text.length && DIGIT.test(text[j])) j++;
        }
      }
      if (/[uUfF]/.test(text[j] ?? '')) j++;
      out.push({t: 'num', v: text.slice(i, j), line, ws}); i = j; ws = false; continue;
    }
    const p = PUNCT.find(q => text.startsWith(q, i));
    if (!p) essError(`Unexpected character '${c}'`, line);
    out.push({t: 'op', v: p, line, ws}); i += p.length; ws = false;
  }
  return out;
}

/** Remove comments and splice continuation lines, preserving line numbers. */
function stripComments(source) {
  let out = '', i = 0;
  source = source.replace(/\\\r?\n/g, '');
  while (i < source.length) {
    if (source.startsWith('//', i)) { while (i < source.length && source[i] !== '\n') i++; out += ' '; continue; }
    if (source.startsWith('/*', i)) {
      const end = source.indexOf('*/', i + 2);
      if (end < 0) essError('Unterminated block comment');
      out += ' ' + source.slice(i, end + 2).replace(/[^\n]/g, ''); i = end + 2; continue;
    }
    out += source[i++];
  }
  return out;
}

function expand(tokens, macros, disabled = new Set()) {
  const out = [];
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const macro = tok.t === 'id' && !disabled.has(tok.v) ? macros.get(tok.v) : undefined;
    if (!macro) { out.push(tok); continue; }
    if (macro.params === null) {
      const inner = new Set(disabled); inner.add(tok.v);
      for (const b of expand(macro.body.map(x => ({...x, line: tok.line})), macros, inner)) out.push(b);
      continue;
    }
    // Function-like: only an invocation when followed by '('.
    if (tokens[i + 1]?.v !== '(') { out.push(tok); continue; }
    const args = []; let depth = 0, j = i + 2, current = [];
    for (; j < tokens.length; j++) {
      const v = tokens[j].v;
      if (tokens[j].t === 'op' && (v === '(' || v === '[')) depth++;
      if (tokens[j].t === 'op' && (v === ')' || v === ']')) { if (depth === 0 && v === ')') break; depth--; }
      if (tokens[j].t === 'op' && v === ',' && depth === 0) { args.push(current); current = []; continue; }
      current.push(tokens[j]);
    }
    if (j >= tokens.length) essError(`Unterminated invocation of macro ${tok.v}`, tok.line);
    if (current.length || args.length || macro.params.length) args.push(current);
    if (args.length === 1 && args[0].length === 0 && macro.params.length === 0) args.length = 0;
    if (args.length !== macro.params.length) essError(`Macro ${tok.v} expects ${macro.params.length} arguments`, tok.line);
    const expanded = args.map(a => expand(a, macros, disabled));
    const body = [];
    for (const b of macro.body) {
      const k = b.t === 'id' ? macro.params.indexOf(b.v) : -1;
      if (k >= 0) body.push(...expanded[k]); else body.push({...b, line: tok.line});
    }
    const inner = new Set(disabled); inner.add(tok.v);
    out.push(...expand(body, macros, inner));
    i = j;
  }
  return out;
}

/** Integer #if expression evaluation (ESSL 3.4). */
function evaluate(tokens, macros, line) {
  // defined X / defined(X) before expansion.
  const pre = [];
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].t === 'id' && tokens[i].v === 'defined') {
      let name, j = i + 1;
      if (tokens[j]?.v === '(') { name = tokens[j + 1]?.v; if (tokens[j + 2]?.v !== ')') essError('Malformed defined()', line); i = j + 2; }
      else { name = tokens[j]?.v; i = j; }
      if (!name) essError('Malformed defined', line);
      pre.push({t: 'num', v: macros.has(name) ? '1' : '0', line});
    } else pre.push(tokens[i]);
  }
  const toks = expand(pre, macros).map(t => t.t === 'id' ? {t: 'num', v: '0', line} : t);
  let p = 0;
  const peek = () => toks[p]?.v, next = () => toks[p++];
  const num = t => { const v = t.v.replace(/[uU]$/, ''); return /^0[xX]/.test(v) ? parseInt(v, 16) : /^0[0-7]+$/.test(v) ? parseInt(v, 8) : parseInt(v, 10); };
  const BIN = [['||'], ['&&'], ['|'], ['^'], ['&'], ['==', '!='], ['<', '>', '<=', '>='], ['<<', '>>'], ['+', '-'], ['*', '/', '%']];
  function unary() {
    const t = next(); if (!t) essError('Incomplete #if expression', line);
    if (t.v === '(') { const v = ternary(); if (next()?.v !== ')') essError('Missing ) in #if', line); return v; }
    if (t.v === '!') return unary() ? 0 : 1;
    if (t.v === '-') return -unary();
    if (t.v === '+') return unary();
    if (t.v === '~') return ~unary();
    if (t.t === 'num') return num(t);
    essError(`Unexpected '${t.v}' in #if`, line);
  }
  function binary(level) {
    if (level === BIN.length) return unary();
    let left = binary(level + 1);
    while (BIN[level].includes(peek())) {
      const op = next().v, right = binary(level + 1);
      switch (op) {
        case '||': left = left || right ? 1 : 0; break; case '&&': left = left && right ? 1 : 0; break;
        case '|': left |= right; break; case '^': left ^= right; break; case '&': left &= right; break;
        case '==': left = left === right ? 1 : 0; break; case '!=': left = left !== right ? 1 : 0; break;
        case '<': left = left < right ? 1 : 0; break; case '>': left = left > right ? 1 : 0; break;
        case '<=': left = left <= right ? 1 : 0; break; case '>=': left = left >= right ? 1 : 0; break;
        case '<<': left <<= right; break; case '>>': left >>= right; break;
        case '+': left += right; break; case '-': left -= right; break; case '*': left *= right; break;
        case '/': if (!right) essError('Division by zero in #if', line); left = Math.trunc(left / right); break;
        case '%': if (!right) essError('Division by zero in #if', line); left %= right; break;
      }
    }
    return left;
  }
  function ternary() {
    const c = binary(0);
    if (peek() !== '?') return c;
    next(); const a = ternary(); if (next()?.v !== ':') essError('Malformed ?: in #if', line); const b = ternary();
    return c ? a : b;
  }
  const v = ternary();
  if (p !== toks.length) essError('Trailing tokens in #if', line);
  return v;
}

/** Preprocess one shader stage. Returns {tokens, version, extensions, pragmas}. */
export function preprocess(source, {defines = {}} = {}) {
  const text = stripComments(source), lines = text.split('\n');
  const macros = new Map([['GL_ES', {params: null, body: [{t: 'num', v: '1'}]}], ['__VERSION__', {params: null, body: [{t: 'num', v: '300'}]}],
    ['GL_FRAGMENT_PRECISION_HIGH', {params: null, body: [{t: 'num', v: '1'}]}]]);
  for (const [k, v] of Object.entries(defines)) macros.set(k, {params: null, body: tokenize(String(v))});
  const stack = [], out = [], extensions = [], pragmas = [];
  let version = null, pending = [], pendingStart = 1;
  const active = () => stack.every(s => s.active);
  const flush = () => {
    if (!pending.length) return;
    out.push(...expand(tokenize(pending.join('\n'), pendingStart), macros));
    pending = [];
  };
  for (let n = 0; n < lines.length; n++) {
    const raw = lines[n], line = n + 1, m = /^\s*#\s*([A-Za-z_]*)(.*)$/.exec(raw);
    if (!m) { if (active()) { if (!pending.length) pendingStart = line; pending.push(raw); } else if (pending.length) pending.push(''); continue; }
    const [, directive, rest] = m;
    const tokens = () => tokenize(rest, line);
    if (['if', 'ifdef', 'ifndef'].includes(directive)) {
      const parentActive = active();
      let value = false;
      if (parentActive) {
        if (directive === 'if') value = evaluate(tokens(), macros, line) !== 0;
        else { const name = tokens()[0]?.v; if (!name) essError(`#${directive} needs a name`, line); value = macros.has(name) === (directive === 'ifdef'); }
      }
      flush(); stack.push({active: parentActive && value, taken: value, parentActive, sawElse: false});
      continue;
    }
    if (directive === 'elif' || directive === 'else') {
      const top = stack.at(-1); if (!top || top.sawElse) essError(`Unmatched #${directive}`, line);
      flush();
      if (directive === 'else') { top.sawElse = true; top.active = top.parentActive && !top.taken; top.taken = true; }
      else {
        const value = top.parentActive && !top.taken && evaluate(tokens(), macros, line) !== 0;
        top.active = value; if (value) top.taken = true;
      }
      continue;
    }
    if (directive === 'endif') { if (!stack.length) essError('Unmatched #endif', line); flush(); stack.pop(); continue; }
    if (!active()) continue;
    flush();
    if (directive === '') continue;
    if (directive === 'define') {
      const m2 = /^\s*([A-Za-z_][A-Za-z0-9_]*)(\(([^)]*)\))?(.*)$/.exec(rest);
      if (!m2) essError('Malformed #define', line);
      const [, name, paren, list, body] = m2;
      if (name.startsWith('GL_') || name.includes('__')) {
        // ESSL reserves these; three never defines them. Accept but warn-free.
      }
      const params = paren === undefined ? null : list.trim() === '' ? [] : list.split(',').map(s => s.trim());
      const bodyTokens = tokenize(body, line);
      if (bodyTokens.some(t => t.v === '##' || t.v === '#')) essError('Token pasting/stringizing is not ESSL', line);
      macros.set(name, {params, body: bodyTokens});
      continue;
    }
    if (directive === 'undef') { macros.delete(tokens()[0]?.v); continue; }
    if (directive === 'version') { version = rest.trim(); continue; }
    if (directive === 'extension') { extensions.push(rest.trim()); continue; }
    if (directive === 'pragma') { pragmas.push(rest.trim()); continue; }
    if (directive === 'line') continue;
    if (directive === 'error') essError(`#error ${rest.trim()}`, line);
    essError(`Unknown directive #${directive}`, line);
  }
  flush();
  if (stack.length) essError('Unterminated #if');
  return {tokens: out, version, extensions, pragmas};
}
