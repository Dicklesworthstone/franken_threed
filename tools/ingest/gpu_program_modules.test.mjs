import test from 'node:test';
import assert from 'node:assert/strict';
import {gpuProgramModule} from './gpu_program_modules.mjs';

function device() {
  let lose;
  return {modules: [], scopes: [], lost: new Promise(resolve => { lose = resolve; }), lose: () => lose(),
    pushErrorScope(kind) { this.scopes.push(kind); },
    popErrorScope() { assert.ok(this.scopes.pop()); return Promise.resolve(this.error ?? null); },
    createShaderModule(descriptor) {
      const info = this.info ?? {messages: []};
      const module = {...descriptor, getCompilationInfo: () => Promise.resolve(info)};
      this.modules.push(module); return module;
    }};
}

test('exact WGSL shares a validated module within a device, independently of labels', async () => {
  const a = device(), b = device(), first = gpuProgramModule(a, {code: 'one', label: 'warm'});
  assert.equal(gpuProgramModule(a, {code: 'one', label: 'draw'}), first);
  assert.notEqual(gpuProgramModule(b, {code: 'one'}).module, first.module);
  assert.notEqual(gpuProgramModule(a, {code: 'two'}).module, first.module);
  assert.deepEqual(a.scopes, [], 'the shared device stack closes synchronously');
  assert.equal(await first.ready, first.module); assert.equal(a.modules.length, 2);
});

test('compilation messages and error scopes reject admission without poisoning retries', async () => {
  const d = device(); d.info = {messages: [{type: 'error', message: 'invalid source'}]};
  const first = gpuProgramModule(d, {code: 'retry'});
  await assert.rejects(first.ready, /invalid source/);
  d.info = {messages: [{type: 'warning', message: 'advisory'}]};
  const second = gpuProgramModule(d, {code: 'retry'});
  assert.notEqual(first.module, second.module); await second.ready;
  d.error = {message: 'allocation rejected'};
  await assert.rejects(gpuProgramModule(d, {code: 'scoped'}).ready, /allocation rejected/);
  d.error = null; await gpuProgramModule(d, {code: 'scoped'}).ready;
  assert.deepEqual(d.scopes, []);
});

test('synchronous device failure closes every scope and permits another attempt', async () => {
  const d = device(), create = d.createShaderModule;
  d.createShaderModule = () => { throw new Error('synchronous allocation failure'); };
  assert.throws(() => gpuProgramModule(d, {code: 'retry'}), /synchronous allocation failure/);
  assert.deepEqual(d.scopes, []); d.createShaderModule = create;
  await gpuProgramModule(d, {code: 'retry'}).ready;
});

test('module residency is bounded and released on device loss', async () => {
  const d = device(), first = gpuProgramModule(d, {code: 'first'});
  await Promise.all(Array.from({length: 256}, (_, index) => gpuProgramModule(d, {code: String(index)}).ready));
  const next = gpuProgramModule(d, {code: 'first'});
  assert.notEqual(next.module, first.module, 'the least recently used entry retires');
  await next.ready; d.lose(); await Promise.resolve();
  assert.notEqual(gpuProgramModule(d, {code: 'first'}).module, next.module);
  const large = 'x'.repeat(8 * 1024 * 1024 + 1);
  const oversized = gpuProgramModule(d, {code: large}); await oversized.ready;
  assert.notEqual(gpuProgramModule(d, {code: large}).module, oversized.module, 'large programs compile without retaining oversized cache keys');
});
