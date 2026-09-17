import test from 'node:test';
import assert from 'node:assert/strict';
import { expandNumericFixedLoops } from './numeric_fixed_loops.mjs';

// Evaluating generated fixture code is test-only; expansion never evaluates it.
const evaluate = source => Function(`'use strict'; return (${source});`)();
function compare(source, args = []) {
  const result = expandNumericFixedLoops(source);
  assert.equal(result.changed, true);
  const copy = values => structuredClone(values);
  const before = copy(args), after = copy(args);
  assert.deepEqual(evaluate(result.source)(...after), evaluate(source)(...before));
  assert.deepEqual(after, before);
  return result;
}

for (const control of ['let j=0;j<4;j++', 'let j=1;j<=4;++j', 'let j=6;j>0;j-=2',
  'let j=3;j>=0;j--', 'let j=-3;j<4;j+=2', 'let j=0;j<=6;j-= -2']) {
  test('preserves fixed iteration values and source order: ' + control, () => {
    const result = compare(`function f(){const values=[];for(${control})values.push(j);return values;}`);
    assert.equal(result.loops.length, 1);
    assert.ok(result.loops[0].iterations <= 4);
  });
}

test('distinct per-iteration bindings and the original body scope survive expansion', () => {
  compare(`function f(){const callbacks=[];for(let j=0;j<4;j++){
    const value=j*2;callbacks.push(()=>[j,value]);
  }return callbacks.map(fn=>fn());}`);
  compare(`function f(){const result=[];for(let j=0;j<4;j++){let j=9;result.push(j);}return result;}`);
});

test('numeric negative-zero initializer retains its sign', () => {
  const result = compare('function f(){let value=1;for(let j=-0;j<1;j++)value=1/j;return value;}');
  assert.ok(result.source.includes('const j = -0;'));
  assert.equal(evaluate(result.source)(), -Infinity);
});

test('a runtime-sized outer loop remains intact around expanded influence loops', () => {
  const source = `function f(input, output){for(let i=0;i<input.length;i++){
    let value=0;for(let influence=0;influence<4;influence++)value+=input[i]*(influence+1);
    output[i]=value;
  }}`;
  const result = compare(source, [new Float32Array([0, -0, 1/3, Infinity, NaN]), new Float32Array(5)]);
  assert.equal(result.loops.length, 1);
  assert.equal(result.expandedIterations, 4);
  assert.ok(result.source.includes('for(let i=0;i<input.length;i++)'));
});

test('nested fixed row/column loops preserve store rounding and account multiplicative work', () => {
  const result = compare(`function f(a){for(let row=0;row<4;row++){
    for(let col=0;col<4;col++){a[row*4+col]+=1/3;a[row*4+col]*=0.7;}
  }} `, [new Float32Array(16).fill(1/7)]);
  assert.equal(result.loops.length, 2);
  assert.equal(result.expandedIterations, 4 + 16);
});

test('array stores and compound assignments retain their exact source ordering', () => {
  compare('function f(a){for(let j=0;j<3;j++){a[0]+=a[j];a[j]=a[0];}return a[0];}',
    [new Float32Array([16777216, 1, -16777216])]);
});

test('zero iterations do not execute the body; early returns remain returns', () => {
  const result = compare('function f(){let a=7;for(let j=4;j<0;j++)a=unreachable();return a;}');
  assert.equal(result.expandedIterations, 0);
  compare('function f(){for(let j=0;j<4;j++){if(j===2)return j;}return -1;}');
});

for (const body of ['break;', 'continue;', 'outer: { break outer; }', 'j++;', 'j=8;', '[j]=[8];',
  '({x:j}={x:8});', 'for(j of [1,2]){}', 'eval("j=8");',
  'var a;', 'function callback(){return j;}', 'class C {}']) {
  test('retains loop with unproved control or function-scope semantics: ' + body, () => {
    const source = `function f(){for(let j=0;j<4;j++){${body}}}`;
    const result = expandNumericFixedLoops(source);
    assert.equal(result.changed, false);
    assert.equal(result.source, source);
  });
}

for (const control of ['var j=0;j<4;j++','let j=0;j<4;j+=0','let j=0;j<4;j--',
  'let j=0;j<1000;j++','let j=0;j<count;j++','let j=start;j<4;j++',
  'let j=0.5;j<4;j++','let j=0;j<4;j*=2','let j=9007199254740991;j<=9007199254740991;j++']) {
  test('retains runtime/unbounded/non-integer control: ' + control, () => {
    const source = `function f(){for(${control}){}}`;
    assert.equal(expandNumericFixedLoops(source).source, source);
  });
}

test('expansion budgets retain the entire source without a partial prefix', () => {
  const source = 'function f(){for(let x=0;x<16;x++){for(let y=0;y<16;y++){for(let z=0;z<16;z++){use(x,y,z);}}}}';
  const result = expandNumericFixedLoops(source);
  assert.equal(result.changed, false);
  assert.equal(result.source, source);
  assert.equal(result.reason, 'EXPANSION_BUDGET');
  const longBody = 'function f(){for(let j=0;j<16;j++){const message="' + 'a'.repeat(200) + '";use(message);}}';
  assert.equal(expandNumericFixedLoops(longBody, {maxSourceLength: 1000}).source, longBody);
});

test('keeps surrounding source byte-for-byte and reports original spans', () => {
  const source = 'export function f(){/*before*/for(let j=0;j<2;j++){use(j);/*inside*/}/*after*/}';
  const a = expandNumericFixedLoops(source), b = expandNumericFixedLoops(source);
  assert.deepEqual(a, b);
  const span = a.loops[0].sourceSpan;
  assert.equal(source.slice(span.start, span.end), 'for(let j=0;j<2;j++){use(j);/*inside*/}');
  assert.ok(a.source.startsWith('export function f(){/*before*/'));
  assert.ok(a.source.endsWith('/*after*/}'));
  assert.equal(a.source.split('/*inside*/').length - 1, 2);
});

test('invalid inputs/configuration and parse-unsupported code never produce a partial transform', () => {
  assert.throws(() => expandNumericFixedLoops(null), TypeError);
  for (const options of [{maxIterations:0},{maxIterations:65},{maxExpandedIterations:NaN},{maxSourceLength:0}]) {
    assert.throws(() => expandNumericFixedLoops('', options), RangeError);
  }
  assert.equal(expandNumericFixedLoops('let x: number = 1;').reason, 'PARSE_UNSUPPORTED');
  assert.equal(expandNumericFixedLoops('function f(){}').reason, 'NO_FIXED_LOOPS');
});
