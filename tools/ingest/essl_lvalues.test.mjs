import assert from 'node:assert/strict';
import test from 'node:test';
import {compileEsslProgram, compileEsslStage} from './essl_wgsl.mjs';
import {lvalueCases, lvalueSources} from './fixtures/essl_lvalues.mjs';

const version = source => '#version 300 es\n' + source;
const compile = name => {
  const fixture = lvalueCases.find(fixture => fixture.name === name);
  const {vertex, fragment} = lvalueSources(fixture, 'fragment');
  return compileEsslProgram(version(vertex), version(fragment)).fragment;
};
const main = wgsl => /fn main_\(\) \{\n([\s\S]*?)^\}/m.exec(wgsl)[1];

test('dynamic lvalue semantic fixtures lower in both stages with linked interfaces', () => {
  for (const fixture of lvalueCases) for (const stage of ['vertex', 'fragment']) {
    const {vertex, fragment} = lvalueSources(fixture, stage);
    const result = compileEsslProgram(version(vertex), version(fragment));
    assert.doesNotMatch(result[stage], /\bnull\b/, `${fixture.name} ${stage}`);
    assert.match(result[stage], /fn main_\(\)/, `${fixture.name} ${stage}`);
  }
});

test('assignment and increment results reuse one captured target address', () => {
  for (const expression of ['(values[destination()] = 71.)', '++values[destination()]', 'values[destination()]++']) {
    const wgsl = compileEsslStage('fragment', version(`precision highp float;precision highp int;out vec4 color;
      int calls = 0; int destination() { calls++; return 0; }
      void main(){vec2 values=vec2(10.,20.);float result=${expression};color=vec4(values,result,float(calls));}`));
    const body = main(wgsl), captures = [...body.matchAll(/let (f3d_t\d+) = destination\(\);/g)];
    assert.equal(captures.length, 1, expression);
    assert.ok(body.includes(`values[${captures[0][1]}] =`), expression);
    assert.equal(body.match(/destination\(\)/g)?.length, 1, expression);
    assert.match(body, /let f3d_t\d+ = values\[f3d_t\d+\];/);
  }
  const body = main(compile('compound-assignment-address'));
  assert.ok(body.indexOf('= cursor;') < body.indexOf('cursor = (cursor + 1);'));
  assert.match(body, /values\[(f3d_t\d+)\] = \(values\[\1\] \+ f32\(f3d_t\d+\)\);/);
});

test('out addresses are captured once without reading old output values', () => {
  const body = main(compile('out-address-before-body'));
  assert.equal(body.match(/cursor = \(cursor \+ 1\);/g)?.length, 1);
  const temporary = /var (f3d_t\d+): f32;\s+store\(&\1\);\s+values\[(f3d_t\d+)\] = \1;/.exec(body);
  assert.ok(temporary, 'out temporary is uninitialized and writes back to the saved destination');
  assert.ok(body.indexOf('= cursor;') < body.indexOf('store(&'));
  const paired = main(compile('multiple-output-addresses'));
  assert.equal(paired.match(/cursor = \(cursor \+ 1\);/g)?.length, 2);
});

test('inout copies its value before later argument mutations and reuses its address', () => {
  const value = main(compile('inout-value-before-later-argument'));
  assert.ok(value.indexOf(': f32 = values.x;') < value.indexOf('= mutate();'));
  assert.match(value, /adjust\(&f3d_t\d+, f3d_t\d+\);\s+values\.x = f3d_t\d+;/);
  const address = main(compile('inout-address-and-result'));
  assert.equal(address.match(/cursor = \(cursor \+ 1\);/g)?.length, 2);
  const copy = /var (f3d_t\d+): f32 = values\[(f3d_t\d+)\];/.exec(address);
  assert.ok(copy);
  assert.ok(address.includes(`values[${copy[2]}] = ${copy[1]};`));
});

test('nested swizzle stores map dynamic and field destinations back to the original vector', () => {
  const body = main(compile('nested-swizzle-stores'));
  assert.match(body, /value\[array<i32, 3>\(3, 1, 0\)\[f3d_t\d+\]\] = 75\.0;/);
  assert.match(body, /value\.y = f3d_t\d+\.x;\s+value\.z = f3d_t\d+\.y;/);
  assert.match(body, /value\.x = 21\.0;/);
  const output = main(compile('out-nested-swizzle-address'));
  assert.match(output, /value\[array<i32, 3>\(2, 1, 0\)\[f3d_t\d+\]\] = f3d_t\d+;/);
});

test('modf evaluates its input before the output address and stores to that address once', () => {
  const body = main(compile('modf-input-before-output-address'));
  assert.ok(body.indexOf('= value;') < body.indexOf('= destination();'));
  assert.equal(body.match(/destination\(\)/g)?.length, 1);
  const index = /let (f3d_t\d+) = destination\(\);/.exec(body)[1];
  assert.ok(body.includes(`whole[${index}] =`));
  assert.match(body, /let f3d_t\d+ = modf\(f3d_t\d+\);/);
});

test('argument sequences retain earlier values and preserve multiple producing calls in source order', () => {
  const user = main(compile('user-argument-value-order'));
  assert.ok(user.indexOf('= value;') < user.indexOf('value = (value + 1.0);'));
  for (const name of ['builtin-argument-effect-order', 'constructor-argument-effect-order']) {
    const body = main(compile(name));
    assert.ok(body.indexOf('= first();') < body.indexOf('= second();'), name);
    assert.ok(body.indexOf('= second();') < body.indexOf(' + 1);'), name);
    assert.equal(body.match(/first\(\)/g)?.length, 1); assert.equal(body.match(/second\(\)/g)?.length, 1);
  }
});

test('index evaluation reads stored aggregates afterward but captures produced aggregate values beforehand', () => {
  const stored = main(compile('index-side-effect-before-value-read'));
  assert.doesNotMatch(stored, /= values;/);
  assert.ok(stored.indexOf('= destination();') < stored.indexOf('= values['));
  const produced = main(compile('aggregate-producer-before-index'));
  assert.ok(produced.indexOf('= produce();') < produced.indexOf('= destination();'));
});

test('constant aggregate indices remain constant in offset arguments and switch labels', () => {
  const switchBody = main(compile('constant-aggregate-index'));
  assert.match(switchBody, /const key: i32 = vec2<i32>\(1, 2\)\[0\];/);
  assert.match(switchBody, /case vec2<i32>\(1, 2\)\[1\]/);
  const texture = compileEsslStage('fragment', version(`precision highp float;uniform sampler2D image;out vec4 color;
    void main(){color=textureOffset(image,vec2(.5),ivec2(ivec2(1,2)[0]));}`));
  assert.match(texture, /textureSample\(image_texture, image_sampler, f3d_t\d+, vec2<i32>\(vec2<i32>\(1, 2\)\[0\]\)\)/);
});

test('out destinations still reject constants, repeated swizzles and non-lvalues', () => {
  for (const [body, error] of [
    ['const float value=1.;store(value);', /Cannot assign to const/],
    ['vec2 value=vec2(1.);store2(value.xx);', /Repeated swizzle/],
    ['store(1.+2.);', /not assignable/],
  ]) assert.throws(() => compileEsslStage('fragment', version(`precision highp float;out vec4 color;
    void store(out float value){value=2.;}void store2(out vec2 value){value=vec2(2.);}
    void main(){${body}color=vec4(1.);}`)), error);
});
