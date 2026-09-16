import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const cli = fileURLToPath(new URL('./cli.mjs', import.meta.url));
const run = args => spawnSync(process.execPath, [cli, ...args], { encoding: 'utf8' });

test('help exposes normal-build numeric specialization', () => {
  const result = run(['--help']); assert.equal(result.status, 0); assert.match(result.stdout, /--specialize-numeric/);
});
test('numeric specialization needs an application build', () => {
  for (const args of [[], ['--build-kernel', '/not-created', '--parameter-types', 'f64[]']]) {
    const result = run(['--entry', cli, '--specialize-numeric', ...args]);
    assert.equal(result.status, 1); assert.match(result.stderr, /requires --build-app/);
  }
});
test('parameter ABI stays exclusive to standalone kernel packages', () => {
  const result = run(['--entry', cli, '--specialize-numeric', '--build-app', '/not-created', '--parameter-types', 'f64[]']);
  assert.equal(result.status, 1); assert.match(result.stderr, /--parameter-types requires --build-kernel/);
});
test('memory limits without numeric compilation are rejected', () => {
  const result = run(['--entry', cli, '--max-memory-pages', '4']);
  assert.equal(result.status, 1); assert.match(result.stderr, /--max-memory-pages requires/);
});
