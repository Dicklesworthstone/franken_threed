import test from 'node:test';
import assert from 'node:assert/strict';
import { emitToUint32 } from './numeric_integer.mjs';
import { createMathIntrinsicCompiler } from './numeric_intrinsics.mjs';

function u32(value) {
  const bytes = [];
  do { const byte = value & 127; value >>>= 7; bytes.push(byte | (value ? 128 : 0)); } while (value);
  return bytes;
}
const section = (id, bytes) => [id, ...u32(bytes.length), ...bytes];
function compile(arity, emit) {
  let locals = 0;
  const expression = emit(() => arity + locals++);
  const body = [...(locals ? [1, ...u32(locals), 0x7c] : [0]), ...expression, 0x0b];
  const bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0,
    ...section(1, [1, 0x60, arity, ...Array(arity).fill(0x7c), 1, 0x7c]),
    ...section(3, [1, 0]), ...section(7, [1, 3, 114, 117, 110, 0, 0]),
    ...section(10, [1, ...u32(body.length), ...body])]);
  assert.equal(WebAssembly.validate(bytes), true);
  const module = new WebAssembly.Module(bytes);
  assert.deepEqual(WebAssembly.Module.imports(module), []);
  return new WebAssembly.Instance(module).exports.run;
}
function intrinsic(name, arity) {
  const compiler = createMathIntrinsicCompiler(true, message => { throw new Error(message); });
  const node = {type:'CallExpression', callee:{type:'MemberExpression', computed:false,
    object:{type:'Identifier', name:'Math'}, property:{type:'Identifier', name}},
    arguments:Array.from({length:arity}, (_, index) => ({index}))};
  const run = compile(arity, allocate => compiler.call(node, arg => [0x20, arg.index], false, allocate));
  assert.deepEqual(compiler.requirements(), [name]);
  return run;
}
function inputs() {
  const values = [0, -0, NaN, Infinity, -Infinity, Number.MIN_VALUE, -Number.MIN_VALUE,
    Number.MAX_VALUE, -Number.MAX_VALUE, 0.5, -0.5, 1.9, -1.9];
  // Both neighbours of every binary64 binade, including the 2^84 cutoff.
  const bytes = new DataView(new ArrayBuffer(8));
  for (let exponent = -1074; exponent <= 1023; exponent++) {
    const value = 2 ** exponent;
    bytes.setFloat64(0, value);
    const bits = bytes.getBigUint64(0);
    for (const delta of [-1n, 0n, 1n]) {
      bytes.setBigUint64(0, bits + delta);
      const neighbour = bytes.getFloat64(0);
      values.push(neighbour, -neighbour);
    }
  }
  // Deterministic raw bit patterns exercise mantissas and NaN payloads too.
  let seed = 0x51f15e;
  const next = () => {seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed >>> 0;};
  for (let i = 0; i < 20000; i++) {
    bytes.setUint32(0, next()); bytes.setUint32(4, next());
    values.push(bytes.getFloat64(0));
  }
  return values;
}
const values = inputs();

test('Number coercion matches both JS integer interpretations across binary64', () => {
  const unsigned = compile(1, allocate => [...emitToUint32([0x20, 0], allocate), 0xb8]);
  const signed = compile(1, allocate => [...emitToUint32([0x20, 0], allocate), 0xb7]);
  for (const value of values) {
    assert.equal(unsigned(value), value >>> 0, `ToUint32(${value})`);
    assert.equal(signed(value), value | 0, `ToInt32(${value})`);
  }
});

test('Math.imul wraps its product and Math.clz32 counts the converted low bits', () => {
  const imul = intrinsic('imul', 2), clz32 = intrinsic('clz32', 1);
  const factors = [0, -0, 1, -1, 0x7fffffff, 0x80000000, 0xffffffff, 2 ** 32 + 3,
    -(2 ** 32) - 3, 1.9, -1.9, Infinity, NaN, Number.MAX_VALUE];
  for (let i = 0; i < values.length; i++) {
    const a = values[i], b = values[(i * 97 + 23) % values.length];
    assert.equal(clz32(a), Math.clz32(a), `clz32(${a})`);
    assert.equal(imul(a, b), Math.imul(a, b), `imul(${a}, ${b})`);
    for (const factor of factors) assert.equal(imul(a, factor), Math.imul(a, factor));
  }
});

test('integer intrinsics retain lexical, arity, optional-call and spread refusals', () => {
  for (const [name, arity] of [['imul', 2], ['clz32', 1]]) {
    const node = {type:'CallExpression', callee:{type:'MemberExpression', computed:false,
      object:{type:'Identifier', name:'Math'}, property:{type:'Identifier', name}},
      arguments:Array.from({length:arity}, () => ({type:'Literal', value:1}))};
    const fail = message => {throw new Error(message);};
    const emitter = () => {throw new Error('must refuse before evaluating arguments');};
    const check = (candidate, enabled=true, shadowed=false) => assert.throws(
      () => createMathIntrinsicCompiler(enabled, fail).call(candidate, emitter, shadowed, () => 0),
      error => !error.message.includes('must refuse'));
    check(node, false); check(node, true, true);
    check({...node, optional:true}); check({...node, callee:{...node.callee, optional:true}});
    check({...node, callee:{...node.callee, computed:true}});
    check({...node, arguments:[]}); check({...node, arguments:[...node.arguments, node.arguments[0]]});
    check({...node, arguments:[{type:'SpreadElement'}]});
  }
});
