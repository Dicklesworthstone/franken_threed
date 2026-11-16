/** Closed ESSL expressions used by array extents, resource selectors and offsets.
 * ESSL 3.00 §4.3.3 permits aggregate constructors/selections and builtin calls,
 * but never user functions, assignments or sequence expressions. This evaluator
 * does not execute shader code or replace runtime calculations. Float operands
 * round at f32 boundaries; transcendental/implementation-dependent operations
 * remain unevaluated rather than choosing a host Math result for a resource index.
 */
export function evaluateEsslConstant(expression, {lookup, basicType, parseNumber, same, structs, isUserFunction}) {
  let steps = 0, components = 0;
  const scalar = s => ({k: 's', s}), vector = (s, n) => ({k: 'v', s, n});
  const integer = t => t?.k === 's' && ['int', 'uint'].includes(t.s);
  const numeric = t => ['s', 'v'].includes(t?.k) && ['float', 'int', 'uint'].includes(t.s);
  const round = (s, value) => {
    if (s === 'bool') return typeof value === 'boolean' ? value : undefined;
    if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
    const result = s === 'uint' ? value >>> 0 : s === 'int' ? value | 0 : Math.fround(value);
    return Number.isFinite(result) ? result : undefined;
  };
  const result = (type, value) => value === undefined ? undefined : ({type, value});
  const shaped = (type, values) => {
    if (type.k === 's') return result(type, round(type.s, values[0]));
    if (type.k !== 'v' || values.length !== type.n || (components += values.length) > 4096) return undefined;
    const value = values.map(x => round(type.s, x));
    return value.includes(undefined) ? undefined : result(type, value);
  };
  const lanes = a => a.type.k === 's' ? [a.value] : a.type.k === 'v' ? a.value : undefined;
  const allEqual = (a, b) => Array.isArray(a) && Array.isArray(b)
    ? a.length === b.length && a.every((value, i) => allEqual(value, b[i])) : a === b;
  const componentwise = (type, args, operation) => {
    const values = args.map(lanes);
    if (values.some(a => !a)) return undefined;
    const count = type.k === 's' ? 1 : type.n;
    if (values.some(a => a.length !== 1 && a.length !== count)) return undefined;
    return shaped(type, Array.from({length: count}, (_, i) => operation(...values.map(a => a[a.length === 1 ? 0 : i]))));
  };
  // length() can inspect an array's declared type without treating its mutable
  // contents as constant. Resolve only structural selection, never callbacks.
  const typeOf = (e, depth) => {
    if (!e || depth > 128 || ++steps > 8192) return undefined;
    if (e.k === 'paren') return typeOf(e.e, depth + 1);
    if (e.k === 'id') return lookup(e.name)?.type;
    if (e.k === 'index') {
      const t = typeOf(e.a, depth + 1);
      return t?.k === 'a' ? t.of : t?.k === 'm' ? vector('float', t.r) : t?.k === 'v' ? scalar(t.s) : undefined;
    }
    if (e.k === 'field') {
      const t = typeOf(e.a, depth + 1);
      return t?.k === 'st' ? structs.get(t.name)?.fields.find(f => f.name === e.name)?.type : undefined;
    }
    return evaluate(e, depth + 1)?.type;
  };
  const construct = (e, args, depth) => {
    let type = basicType(e.type.name);
    if (!type && structs.has(e.type.name)) type = {k: 'st', name: e.type.name};
    if (!type) return undefined;
    if (e.type.array !== null && e.type.array !== undefined) {
      const size = e.type.array === 'unsized' ? result(scalar('int'), args.length) : evaluate(e.type.array, depth + 1);
      if (!integer(size?.type) || size.value < 1 || size.value > 4096 || size.value !== args.length || args.some(a => !same(a.type, type))) return undefined;
      components += args.length;
      return components > 4096 ? undefined : result({k: 'a', of: type, n: size.value}, args.map(a => a.value));
    }
    if (type.k === 'st') {
      const fields = structs.get(type.name).fields;
      if (fields.length !== args.length || args.some((a, i) => !same(a.type, fields[i].type))) return undefined;
      return result(type, args.map(a => a.value));
    }
    if (!args.length || args.some(a => !['s', 'v', 'm'].includes(a.type.k))) return undefined;
    const convert = (value, source) => {
      if (type.s === 'bool') return Boolean(value);
      if (source === 'float' && ['int', 'uint'].includes(type.s) &&
          (Math.trunc(value) < (type.s === 'uint' ? 0 : -2147483648) || Math.trunc(value) > (type.s === 'uint' ? 4294967295 : 2147483647))) return undefined;
      return round(type.s ?? 'float', Number(value));
    };
    const values = args.flatMap(a => a.type.k === 'm' ? a.value.flat().map(v => convert(v, 'float'))
      : (a.type.k === 's' ? [a.value] : a.value).map(v => convert(v, a.type.s)));
    if (values.includes(undefined)) return undefined;
    if (type.k === 's') return args.length === 1 ? result(type, values[0]) : undefined;
    if (type.k === 'v') {
      if (args.length === 1 && args[0].type.k === 's') return shaped(type, Array(type.n).fill(values[0]));
      // Excess components may only come from the final supplied argument.
      const last = args.at(-1).type, lastCount = last.k === 's' ? 1 : last.k === 'v' ? last.n : last.c * last.r;
      return values.length >= type.n && values.length - lastCount < type.n ? shaped(type, values.slice(0, type.n)) : undefined;
    }
    if (type.k !== 'm') return undefined;
    if ((components += type.c * type.r) > 4096) return undefined;
    if (args.length === 1 && args[0].type.k === 's')
      return result(type, Array.from({length: type.c}, (_, c) => Array.from({length: type.r}, (_, r) => c === r ? values[0] : 0)));
    if (args.length === 1 && args[0].type.k === 'm')
      return result(type, Array.from({length: type.c}, (_, c) => Array.from({length: type.r}, (_, r) => args[0].value[c]?.[r] ?? (c === r ? 1 : 0))));
    if (values.length !== type.c * type.r) return undefined;
    return result(type, Array.from({length: type.c}, (_, c) => values.slice(c * type.r, (c + 1) * type.r)));
  };
  const binary = (op, a, b) => {
    if (['==', '!='].includes(op)) return same(a.type, b.type)
      ? result(scalar('bool'), allEqual(a.value, b.value) === (op === '==')) : undefined;
    if (['&&', '||', '^^'].includes(op)) return a.type.k === 's' && a.type.s === 'bool' && same(a.type, b.type)
      ? result(a.type, op === '&&' ? a.value && b.value : op === '||' ? a.value || b.value : a.value !== b.value) : undefined;
    if (['<', '<=', '>', '>='].includes(op)) {
      if (!numeric(a.type) || a.type.k !== 's' || !same(a.type, b.type)) return undefined;
      return result(scalar('bool'), op === '<' ? a.value < b.value : op === '<=' ? a.value <= b.value : op === '>' ? a.value > b.value : a.value >= b.value);
    }
    if (!numeric(a.type) || !numeric(b.type)) return undefined;
    const shift = op === '<<' || op === '>>';
    if (shift ? a.type.s === 'float' || b.type.s === 'float' || b.type.k === 'v' && !same({...a.type, s: b.type.s}, b.type) : a.type.s !== b.type.s) return undefined;
    const type = shift || a.type.k === 'v' ? a.type : b.type;
    if (a.type.k === 'v' && b.type.k === 'v' && a.type.n !== b.type.n) return undefined;
    if (['%', '&', '|', '^'].includes(op) && type.s === 'float') return undefined;
    return componentwise(type, [a, b], (x, y) => {
      switch (op) {
        case '+': return x + y; case '-': return x - y;
        case '*': return type.s === 'float' ? x * y : Math.imul(x, y);
        case '/': return y === 0 ? undefined : x / y;
        case '%': return y === 0 ? undefined : x % y;
        case '&': return x & y; case '|': return x | y; case '^': return x ^ y;
        case '<<': return y < 0 || y >= 32 ? undefined : x << y;
        case '>>': return y < 0 || y >= 32 ? undefined : type.s === 'uint' ? x >>> y : x >> y;
        default: return undefined;
      }
    });
  };
  const builtin = (name, args) => {
    const first = args[0]; if (!first) return undefined;
    const t = first.type, count = args.length;
    if (['any', 'all', 'not'].includes(name)) {
      if (count !== 1 || t.k !== 'v' || t.s !== 'bool') return undefined;
      return name === 'not' ? shaped(t, first.value.map(x => !x))
        : result(scalar('bool'), name === 'any' ? first.value.some(Boolean) : first.value.every(Boolean));
    }
    const comparisons = {lessThan: '<', lessThanEqual: '<=', greaterThan: '>', greaterThanEqual: '>='};
    if (name in comparisons || name === 'equal' || name === 'notEqual') {
      if (count !== 2 || t.k !== 'v' || !same(t, args[1].type) || name in comparisons && !numeric(t)) return undefined;
      return shaped(vector('bool', t.n), first.value.map((v, i) => {
        const x = result(scalar(t.s), v), y = result(scalar(t.s), args[1].value[i]);
        return binary(comparisons[name] ?? (name === 'equal' ? '==' : '!='), x, y)?.value;
      }));
    }
    if (name === 'mix' && count === 3 && ['s', 'v'].includes(t.k) && t.s === 'float' && same(t, args[1].type) &&
        same({...t, s: 'bool'}, args[2].type))
      return componentwise(t, args, (x, y, select) => select ? y : x);
    if (!numeric(t)) return undefined;
    if (['abs', 'sign', 'floor', 'ceil', 'trunc', 'fract', 'roundEven'].includes(name)) {
      if (count !== 1 || (name === 'abs' || name === 'sign' ? t.s === 'uint' : t.s !== 'float')) return undefined;
      const operation = {abs: Math.abs, sign: Math.sign, floor: Math.floor, ceil: Math.ceil, trunc: Math.trunc,
        fract: x => x - Math.floor(x), roundEven: x => {const lo = Math.floor(x), part = x - lo; return part < .5 ? lo : part > .5 ? lo + 1 : lo % 2 === 0 ? lo : lo + 1;}}[name];
      return componentwise(t, args, operation);
    }
    if (name === 'min' || name === 'max' || name === 'clamp') {
      if (count !== (name === 'clamp' ? 3 : 2) || args.slice(1).some(a => !same(a.type, t) && !(t.k === 'v' && same(a.type, scalar(t.s)))) ||
          name === 'clamp' && !same(args[1].type, args[2].type)) return undefined;
      return componentwise(t, args, name === 'min' ? Math.min : name === 'max' ? Math.max
        : (x, lo, hi) => lo > hi ? undefined : Math.min(Math.max(x, lo), hi));
    }
    if (name === 'mod' && count === 2 && t.s === 'float' && (same(t, args[1].type) || t.k === 'v' && same(args[1].type, scalar('float'))))
      return componentwise(t, args, (x, y) => y === 0 ? undefined : Math.fround(x - Math.fround(y * Math.floor(Math.fround(x / y)))));
    if (name === 'step' && count === 2 && t.s === 'float' && args[1].type.s === 'float' &&
        (same(t, args[1].type) || t.k === 's' && args[1].type.k === 'v'))
      return componentwise(args[1].type, args, (edge, x) => x < edge ? 0 : 1);
    return undefined;
  };
  const evaluate = (e, depth = 0) => {
    if (!e || depth > 128 || ++steps > 8192) return undefined;
    const next = value => evaluate(value, depth + 1);
    switch (e.k) {
      case 'num': {const n = parseNumber(e.v); return result(n.type, round(n.type.s, n.value));}
      case 'bool': return result(scalar('bool'), e.v);
      case 'paren': return next(e.e);
      case 'id': {const s = lookup(e.name); return s && (s.isConst || s.readonly) ? result(s.type, s.constValue) : undefined;}
      case 'ctor': {const args = e.args.map(next); return args.some(a => !a) ? undefined : construct(e, args, depth);}
      case 'unary': {
        const a = next(e.a); if (!a) return undefined;
        if (e.op === '!') return a.type.k === 's' && a.type.s === 'bool' ? result(a.type, !a.value) : undefined;
        if (!numeric(a.type) || e.op === '~' && a.type.s === 'float') return undefined;
        return componentwise(a.type, [a], x => e.op === '+' ? x : e.op === '-' ? -x : e.op === '~' ? ~x : undefined);
      }
      case 'bin': {const a = next(e.a), b = next(e.b); return a && b ? binary(e.op, a, b) : undefined;}
      case 'cond': {
        const c = next(e.c), a = next(e.a), b = next(e.b);
        // Even unselected conditional/short-circuit operands must be constant.
        return c?.type.k === 's' && c.type.s === 'bool' && a && b && same(a.type, b.type) ? c.value ? a : b : undefined;
      }
      case 'call': {
        if (isUserFunction(e.name)) return undefined;
        const args = e.args.map(next); return args.some(a => !a) ? undefined : builtin(e.name, args);
      }
      case 'index': {
        const a = next(e.a), index = next(e.index);
        if (!a || !integer(index?.type) || !Array.isArray(a.value) || index.value < 0 || index.value >= a.value.length) return undefined;
        const type = a.type.k === 'a' ? a.type.of : a.type.k === 'm' ? vector('float', a.type.r) : a.type.k === 'v' ? scalar(a.type.s) : undefined;
        return type ? result(type, a.value[index.value]) : undefined;
      }
      case 'field': {
        const a = next(e.a); if (!a) return undefined;
        if (a.type.k === 'st') {
          const fields = structs.get(a.type.name).fields, i = fields.findIndex(f => f.name === e.name);
          return i < 0 ? undefined : result(fields[i].type, a.value[i]);
        }
        if (a.type.k !== 'v' || !e.name.length || e.name.length > 4) return undefined;
        const set = ['xyzw', 'rgba', 'stpq'].find(s => [...e.name].every(c => s.includes(c)));
        if (!set) return undefined;
        const indices = [...e.name].map(c => set.indexOf(c));
        if (indices.some(i => i >= a.type.n)) return undefined;
        return shaped(indices.length === 1 ? scalar(a.type.s) : vector(a.type.s, indices.length), indices.map(i => a.value[i]));
      }
      case 'method': {
        if (e.name !== 'length' || e.args.length) return undefined;
        let receiver = e.a; while (receiver.k === 'paren') receiver = receiver.e;
        const symbol = receiver.k === 'id' ? lookup(receiver.name) : null, t = typeOf(e.a, depth + 1);
        const size = symbol?.elements?.length ?? (t?.k === 'a' ? t.n : undefined);
        return Number.isInteger(size) ? result(scalar('int'), size) : undefined;
      }
      default: return undefined;
    }
  };
  return evaluate(expression);
}
