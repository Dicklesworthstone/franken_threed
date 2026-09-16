import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { specializeNumericModule } from './numeric_specialization.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';

const dispatchURL = new URL('./numeric_dispatch.mjs', import.meta.url).href;
const digest = value => createHash('sha256').update(value).digest('hex');
const url = (root, name) => pathToFileURL(path.join(root, name)).href;
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-scalar-graph-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}
async function application(t, source) {
  const root = fixture(t);
  // Observe the real dispatch tokens without replacing the compiler, runtime,
  // Wasm engine or retained application implementation.
  fs.writeFileSync(path.join(root, 'observer.mjs'), `
    import { createNumericDispatch as create, dispatchNumericCall, numericDispatchDiagnostics } from ${JSON.stringify(dispatchURL)};
    export { dispatchNumericCall };
    const tokens = [];
    export function createNumericDispatch(...args) { const token = create(...args); tokens.push(token); return token; }
    export function diagnostics() { return tokens.map(numericDispatchDiagnostics); }
  `);
  const result = specializeNumericModule(source, { sourceName: 'application.mjs', runtimeModule: url(root, 'observer.mjs') });
  fs.writeFileSync(path.join(root, 'application.mjs'), result.code);
  fs.writeFileSync(path.join(root, 'reference.mjs'), source);
  return { result, module: await import(url(root, 'application.mjs')), reference: await import(url(root, 'reference.mjs')),
    diagnostics: (await import(url(root, 'observer.mjs'))).diagnostics };
}
const scene = `
export function clamp(x, low, high) { if(x<low)return low; if(x>high)return high; return x; }
function smooth(x) { const t=clamp(x,0,1); return t*t*(3-2*t); }
export function lerp(a,b,t) { return a+(b-a)*t; }
export function update(x,target,dt) {
  const t=smooth(dt);
  for(let i=0;i<x.length;i++) x[i]=lerp(x[i],target[i],t);
}
export const originalUpdate=update, originalLerp=lerp;
export function frame(x,target,dt) { return update(x,target,dt); }
`;

test('automatically closes scalar dependencies of ordinary exported update calls', async t => {
  const app = await application(t, scene);
  assert.equal(app.result.report.compiledKernels, 1);
  assert.equal(app.result.report.rewrittenCalls, 1);
  const helpers = app.result.report.candidates[0].scalarHelpers;
  assert.deepEqual(helpers.map(item => item.name), ['smooth', 'clamp', 'lerp']);
  for (const helper of helpers) assert.ok(scene.slice(helper.sourceSpan.start, helper.sourceSpan.end).startsWith(`function ${helper.name}`));
  assert.equal(app.diagnostics()[0].initialized, false);
  assert.deepEqual(Object.keys(app.module), Object.keys(app.reference));
  assert.equal(app.module.originalUpdate, app.module.update);
  assert.equal(app.module.originalLerp, app.module.lerp);
  assert.equal(app.module.update.toString(), app.reference.update.toString());
  assert.equal(app.module.lerp.toString(), app.reference.lerp.toString());
  for (const ArrayType of [Float32Array, Float64Array]) {
    const x = ArrayType.from({ length: 10000 }, (_, i) => i / 19);
    const target = ArrayType.from({ length: 10000 }, (_, i) => (10000 - i) / 13);
    const expected = x.slice();
    for (let frame = 0; frame < 120; frame++) {
      const dt = (frame + 1) / 180;
      assert.equal(app.module.frame(x, target, dt), app.reference.frame(expected, target, dt));
      assert.deepEqual(x, expected);
    }
    assert.equal(app.diagnostics()[0].kernel.wasmCalls, 120);
    assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 0);
  }
  assert.equal(app.result.report.accelerated, false);
});

test('returns reductions across ordinary calls while preserving helper function identity', async t => {
  const source = `function squared(x){return x*x;}
    function energy(x){let sum=0;for(let i=0;i<x.length;i++)sum+=squared(x[i]);return sum;}
    export function normSquared(x){return energy(x);}`;
  const app = await application(t, source);
  for (const count of [0, 1, 100000]) {
    const x = Float64Array.from({ length: count }, (_, i) => (i % 37) / 11);
    assert.ok(Object.is(app.module.normSquared(x), app.reference.normSquared(x)));
  }
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 3);
  assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 0);
});

test('selects all four streamed/uniform storage layouts for helper-based geometry', async t => {
  const source = `function transform(x,y,z,a,b,c,d){return a*x+b*y+c*z+d;}
    function update(x,m){for(let i=0;i<x.length;i+=3){const a=x[i],b=x[i+1],c=x[i+2];
      x[i]=transform(a,b,c,m[0],m[4],m[8],m[12]);
      x[i+1]=transform(a,b,c,m[1],m[5],m[9],m[13]);
      x[i+2]=transform(a,b,c,m[2],m[6],m[10],m[14]);}}
    export function frame(x,m){update(x,m);}`;
  const app = await application(t, source);
  assert.equal(app.result.report.candidates[0].variants.length, 4);
  for (const VertexType of [Float32Array, Float64Array]) for (const MatrixType of [Float32Array, Float64Array]) {
    const actual = new VertexType([1, 2, 3, 4, 5, 6]), expected = actual.slice();
    const matrix = new MatrixType([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.125, -0.25, 0.5, 1]);
    app.module.frame(actual, matrix); app.reference.frame(expected, matrix);
    assert.deepEqual(actual, expected);
    assert.equal(app.diagnostics()[0].kernel.wasmCalls, 1);
    assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 0);
  }
});

test('ordinary arguments, source exceptions and shadowed root callees retain their semantics', async t => {
  const source = scene + `
    export function shadow(update,x){return update(x);}
    export function evaluated(nextX,nextTarget,nextDelta){return update(nextX(),nextTarget(),nextDelta());}`;
  const app = await application(t, source);
  const order = [];
  const x = new Float64Array([1, 2]), target = new Float64Array([3, 4]);
  app.module.evaluated(() => { order.push('x'); return x; }, () => { order.push('target'); return target; }, () => { order.push('dt'); return 0.5; });
  assert.deepEqual(order, ['x', 'target', 'dt']);
  assert.deepEqual([...x], [2, 3]);
  const marker = {};
  assert.equal(app.module.shadow(value => { assert.equal(value, x); return marker; }, x), marker);
  assert.equal(app.diagnostics()[0].identityMisses, 1);
  const sentinel = new Error('application argument failed');
  assert.throws(() => app.module.evaluated(() => { throw sentinel; }, () => target, () => 1), error => error === sentinel);
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 1);
});

test('runtime guard refusal preserves source coercions through helper calls', async t => {
  const app = await application(t, scene);
  let coercions = 0;
  const dt = { valueOf() { coercions++; return 0.5; } };
  const x = [1, 2], target = [3, 4];
  app.module.frame(x, target, dt);
  assert.deepEqual(x, [2, 3]);
  assert.equal(coercions, 5); // clamp comparisons plus the original t*t*(3-2*t).
  assert.equal(app.diagnostics()[0].kernel.fallbackCalls, 1);
  assert.equal(app.diagnostics()[0].kernel.wasmCalls, 0);
});

test('unavailable Wasm retains the entire original helper graph', async t => {
  const app = await application(t, scene);
  const prior = Object.getOwnPropertyDescriptor(globalThis, 'WebAssembly');
  Object.defineProperty(globalThis, 'WebAssembly', { configurable: true, value: undefined });
  try {
    const x = new Float64Array([1, 2]); app.module.frame(x, new Float64Array([3, 4]), 0.5);
    assert.deepEqual([...x], [2, 3]);
    assert.equal(app.diagnostics()[0].retainedCalls, 1);
    assert.equal(app.diagnostics()[0].kernel, null);
  } finally { Object.defineProperty(globalThis, 'WebAssembly', prior); }
});

test('mutable transitive dependencies are retained, including rebinding during argument evaluation', async t => {
  const source = `function leaf(x){return x*2;} function scalar(x){return leaf(x);}
    function update(x){for(let i=0;i<x.length;i++)x[i]=scalar(x[i]);}
    export function frame(x){return update((leaf=v=>v*3,x));}`;
  const app = await application(t, source);
  assert.equal(app.result.changed, false);
  assert.match(app.result.report.candidates[0].detail, /leaf/);
  const x = new Float64Array([1, 2]); app.module.frame(x);
  assert.deepEqual([...x], [3, 6]);
  assert.deepEqual(app.diagnostics(), []);
});

test('captured state, helper closures and direct eval never silently become constants', async t => {
  for (const helper of [
    'function scalar(x){return x+gain;}',
    'const scalar=(x)=>x+1;',
    'function scalar(x){return this.gain+x;}',
    'function scalar(x){return x[0];}',
    'function scalar(x){return scalar(x);}',
  ]) {
    const source = `let gain=2; ${helper} function update(x){for(let i=0;i<x.length;i++)x[i]=scalar(x[i]);} export function frame(x){update(x);}`;
    const result = specializeNumericModule(source);
    assert.equal(result.changed, false, helper);
    assert.equal(result.code, source);
  }
  const result = specializeNumericModule(scene + '\nexport function dynamic(s){return eval(s);}');
  assert.equal(result.changed, false);
  assert.equal(result.report.refusal.code, 'DIRECT_EVAL');
});

test('unresolved imports remain JavaScript rather than guessing cross-chunk bindings', async t => {
  const root = fixture(t);
  fs.writeFileSync(path.join(root, 'helper.mjs'), 'export function scalar(x){return x*3;}');
  const source = `import {scalar} from ${JSON.stringify(url(root, 'helper.mjs'))};
    function update(x){for(let i=0;i<x.length;i++)x[i]=scalar(x[i]);}export function frame(x){update(x);}`;
  const app = await application(t, source);
  assert.equal(app.result.changed, false);
  const x = new Float64Array([2, 3]); app.module.frame(x); assert.deepEqual([...x], [6, 9]);
});

test('unrelated unclosed declarations do not block a reachable pure helper graph', async t => {
  const app = await application(t, scene + '\nexport function externalOnly(){return unknownAPI();}');
  assert.equal(app.result.report.compiledKernels, 1);
  assert.equal(app.result.report.candidates[0].scalarHelpers.length, 3);
});

const packageSource = `export function square(x){return x*x;}
export function energy(x){let sum=0;for(let i=0;i<x.length;i++)sum+=square(x[i]);return sum;}`;
function packageFixture(t, source = packageSource) {
  const root = fixture(t), entry = path.join(root, 'source.mjs'), out = path.join(root, 'package');
  fs.writeFileSync(entry, source);
  return { root, entry, out };
}

test('standalone packages select the loop entry and retain every helper without runtime compiler dependencies', async t => {
  const { root, entry, out } = packageFixture(t);
  const result = buildNumericKernel(entry, out, { parameterTypes: ['f64[]'] });
  assert.equal(result.selectedFunction.name, 'energy');
  assert.deepEqual(result.selectedFunction.scalarHelpers.map(item => item.name), ['square']);
  assert.equal(result.sourceSha256, digest(packageSource));
  assert.equal(result.wasmSha256, digest(fs.readFileSync(path.join(out, 'kernel.wasm'))));
  assert.equal(fs.readFileSync(path.join(out, 'retained.mjs'), 'utf8'), `${packageSource}\nexport default energy;\n`);
  assert.ok(!fs.readFileSync(path.join(out, 'kernel.mjs'), 'utf8').includes(root));
  const moved = path.join(root, 'relocated'); fs.renameSync(out, moved);
  const mod = await import(url(moved, 'kernel.mjs'));
  const retained = await import(url(moved, 'retained.mjs'));
  assert.equal(mod.retained, retained.energy);
  assert.equal(retained.square(3), 9);
  const engine = mod.createKernel();
  assert.equal(engine.run(new Float64Array([1, 2, 3])), 14);
  assert.equal(engine.diagnostics.wasmCalls, 1);
  assert.equal(engine.run([1, 2, 3]), 14);
  assert.equal(engine.diagnostics.fallbackCalls, 1);
  assert.deepEqual(fs.readdirSync(moved).sort(), result.emittedFiles.slice().sort());
});

test('standalone helper graphs work through Wasm-unavailable fallback and disposal', async t => {
  const { entry, out } = packageFixture(t);
  buildNumericKernel(entry, out, { parameterTypes: ['f64[]'] });
  const mod = await import(url(out, 'kernel.mjs'));
  const prior = Object.getOwnPropertyDescriptor(globalThis, 'WebAssembly');
  Object.defineProperty(globalThis, 'WebAssembly', { configurable: true, value: undefined });
  try {
    const engine = mod.createKernel();
    assert.equal(engine.run(new Float64Array([3, 4])), 25);
    assert.equal(engine.diagnostics.fallbackCalls, 1);
    assert.equal(engine.diagnostics.wasmCalls, 0);
    engine.dispose(); assert.throws(() => engine.run([]), /KERNEL_DISPOSED/);
  } finally { Object.defineProperty(globalThis, 'WebAssembly', prior); }
});

test('ambiguous packages require an explicit function selection instead of choosing the wrong loop', async t => {
  const source = packageSource + '\nfunction other(x){for(let i=0;i<x.length;i++)x[i]*=2;}';
  const { entry, out } = packageFixture(t, source);
  assert.throws(() => buildNumericKernel(entry, out, { parameterTypes: ['f64[]'] }), /functionName/);
  assert.equal(fs.existsSync(out), false);
  buildNumericKernel(entry, out, { parameterTypes: ['f64[]'], functionName: 'energy' });
  const { createKernel } = await import(url(out, 'kernel.mjs'));
  assert.equal(createKernel().run(new Float64Array([3, 4])), 25);
});

test('package effects, rebinding and unclosed helper sources fail before any output is created', t => {
  for (const source of [
    packageSource + '\nsquare=x=>x;', packageSource + '\nthrow new Error("not evaluated");',
    packageSource + '\nexport default energy;',
    packageSource + '\nexport function replace(){square=x=>x;}',
    packageSource + '\nexport function replace(){[square]=[x=>x];}',
    packageSource + '\nexport function replace(){eval("square=x=>x");}',
    packageSource.replace('return x*x;', 'return captured+x;'),
    packageSource.replace('return x*x;', 'return square(x);'),
    'throw new Error("not evaluated");',
  ]) {
    const { entry, out } = packageFixture(t, source);
    assert.throws(() => buildNumericKernel(entry, out, { parameterTypes: ['f64[]'] }), /KERNEL_NOT_CLOSED/);
    assert.equal(fs.existsSync(out), false);
  }
});

test('standalone helper packages are deterministic and never overwrite existing output', t => {
  const { root, entry, out } = packageFixture(t);
  const result = buildNumericKernel(entry, out, { parameterTypes: ['f64[]'] });
  const second = path.join(root, 'second'); buildNumericKernel(entry, second, { parameterTypes: ['f64[]'] });
  for (const name of result.emittedFiles) assert.deepEqual(fs.readFileSync(path.join(out, name)), fs.readFileSync(path.join(second, name)));
  assert.throws(() => buildNumericKernel(entry, out, { parameterTypes: ['f64[]'] }), { code: 'EEXIST' });
});


test('single-function packages keep scalar parameter mutation when it shadows the function name', async t => {
  const { entry, out } = packageFixture(t, 'function sum(sum,x){for(let i=0;i<x.length;i++){sum+=x[i];x[i]=sum;}return sum;}');
  const result = buildNumericKernel(entry, out, { parameterTypes: ['f64', 'f64[]'] });
  assert.equal(result.selectedFunction, undefined);
  const { createKernel } = await import(url(out, 'kernel.mjs'));
  const engine = createKernel(), x = new Float64Array([1, 2, 3]);
  assert.equal(engine.run(4, x), 10);
  assert.deepEqual([...x], [5, 7, 10]);
  assert.equal(engine.diagnostics.wasmCalls, 1);
});
