// Ranged and reusable readback of real GPU reductions. This compares pinned
// source semantics and native copy sizes, not hardware throughput.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = JSON.parse(await readFile(path.join(repo, 'upstream/pin.json'), 'utf8')).oracle;
assert.equal(pin.source_commit, '148ef33ecb6d2502ff796d4554abd1549c95d519');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
for (const file of ['build/three.core.js', 'build/three.webgpu.js']) {
  const source = execFileSync('git', ['show', `${pin.source_commit}:${file}`],
    {cwd: path.join(repo, pin.target_checkout_dir), maxBuffer: 16 * 1024 * 1024});
  assert.equal(digest(await readFile(path.join(repo, pin.target_checkout_dir, file))), digest(source));
}
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) { if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning); }});
let rendererSource;
try { rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code; }
finally { await bundle.close(); }
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); return; }
    if (pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
    response.setHeader('Content-Type', 'text/javascript');
    if (pathname === '/renderer.mjs') { response.end(rendererSource); return; }
    const filename = path.resolve(repo, '.' + decodeURIComponent(pathname));
    if (!filename.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.end(await readFile(filename));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode) {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      const Renderer = mode === 'source' ? T.WebGPURenderer : (await import('/renderer.mjs')).createWebGPURendererClass(T);
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, device}), errors = [], copies = [], allocations = [], observations = [];
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const createBuffer = device.createBuffer.bind(device);
      device.createBuffer = descriptor => {
        if (descriptor.usage & GPUBufferUsage.MAP_READ) allocations.push(descriptor.size);
        return createBuffer(descriptor);
      };
      const copyBuffer = GPUCommandEncoder.prototype.copyBufferToBuffer;
      GPUCommandEncoder.prototype.copyBufferToBuffer = function (source, offset, target, targetOffset, size) {
        if (target.usage & GPUBufferUsage.MAP_READ) copies.push({offset, size, capacity: target.size});
        return copyBuffer.call(this, source, offset, target, targetOffset, size);
      };
      const size = 32768, groupSize = 64, groups = size / groupSize, summaryOffset = (size - 2) * 4;
      const values = N.instancedArray(new Uint32Array(size), 'uint'), partials = N.instancedArray(groups, 'uint');
      const phase = N.uniform(3, 'uint'), work = N.workgroupArray('uint', groupSize);
      const fill = N.Fn(() => { values.element(N.instanceIndex).assign(N.instanceIndex.mul(3).add(phase)); })().compute(size, [groupSize]);
      const reduceWorkgroup = () => {
        N.workgroupBarrier();
        for (let stride = groupSize / 2; stride >= 1; stride /= 2) {
          N.If(N.localId.x.lessThan(stride), () => work.element(N.localId.x).addAssign(work.element(N.localId.x.add(stride))));
          N.workgroupBarrier();
        }
      };
      const reduce = N.Fn(() => {
        work.element(N.localId.x).assign(values.element(N.instanceIndex));
        reduceWorkgroup();
        N.If(N.localId.x.equal(0), () => partials.element(N.workgroupId.x).assign(work.element(N.uint(0))));
      })().compute(size, [groupSize]);
      const finish = N.Fn(() => {
        const sum = N.uint(0).toVar();
        for (let block = 0; block < groups / groupSize; block++) sum.addAssign(partials.element(N.localId.x.add(block * groupSize)));
        work.element(N.localId.x).assign(sum); reduceWorkgroup();
        N.If(N.localId.x.equal(0), () => {
          values.element(size - 2).assign(work.element(N.uint(0))); values.element(size - 1).assign(phase);
        });
      })().compute(groupSize, [groupSize]);
      const kernels = [fill, reduce, finish], handle = new T.ReadbackBuffer(32);
      const words = buffer => Array.from(new Uint32Array(buffer));
      const expected = value => [(3 * size * (size - 1) / 2 + value * size) >>> 0, value];
      const check = (condition, message) => { if (!condition) throw new Error(message); };
      const equal = (a, b, message) => check(JSON.stringify(a) === JSON.stringify(b), `${message}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
      let failure = null;
      device.pushErrorScope('validation');
      try {
        await renderer.init(); await renderer.compileComputeAsync(kernels);
        const caller = new Uint32Array([0, 0, 0x12345678, 0x87654321]).buffer;
        renderer.compute(kernels);
        const first = renderer.getArrayBufferAsync(values.value, caller, summaryOffset, 8);
        phase.value = 7; renderer.compute(kernels);
        const second = renderer.getArrayBufferAsync(values.value, handle, summaryOffset, 8);
        check(handle._mapped && handle.buffer === null, 'handle must be reserved before mapping completes');
        const busy = renderer.getArrayBufferAsync(values.value, handle, summaryOffset, 8).then(() => false, () => true);
        check(await first === caller, 'caller ArrayBuffer identity');
        equal(words(caller), [...expected(3), 0x12345678, 0x87654321], 'earlier reduction and caller tail');
        check(await second === handle && await busy, 'handle identity and busy refusal');
        equal(words(handle.buffer), expected(7), 'later reduction');
        observations.push(words(caller), words(handle.buffer));
        check(renderer.info.memory.readbackBuffers === 1 && renderer.info.memory.readbackBuffersSize === 32, 'retained staging memory accounting');
        const detached = handle.buffer; handle.release();
        check(detached.byteLength === 0 && handle.buffer === null && !handle._mapped, 'release detaches mapped range and resets handle');
        handle.maxByteLength = 4; // The source retains its initial native capacity until disposal.
        phase.value = 11; renderer.compute(kernels);
        await renderer.getArrayBufferAsync(values.value, handle, summaryOffset, 8);
        equal(words(handle.buffer), expected(11), 'reused handle reduction'); observations.push(words(handle.buffer));
        handle.release();
        const tail = await renderer.getArrayBufferAsync(values.value, null, summaryOffset);
        equal(words(tail), expected(11), 'default remaining byte count'); observations.push(words(tail));
        const empty = await renderer.getArrayBufferAsync(values.value, null, size * 4, 0);
        check(empty.byteLength === 0, 'zero byte range');
        await renderer.getArrayBufferAsync(values.value, handle, size * 4, 0);
        check(handle._mapped && handle.buffer.byteLength === 0, 'zero byte mapped handle'); handle.release();
        handle.dispose();
        check(renderer.info.memory.readbackBuffers === 0 && renderer.info.memory.readbackBuffersSize === 0, 'disposed staging accounting');
        handle.maxByteLength = 32;
        await renderer.getArrayBufferAsync(values.value, handle, 0, 4);
        equal(words(handle.buffer), [11], 'disposed source handle can allocate again'); observations.push(words(handle.buffer));
        handle.dispose();
        check(values.value.array.every(value => value === 0), 'GPU reductions must not mutate their CPU input shadow');
        if (mode === 'candidate') await renderer.waitForGPU();
        await device.queue.onSubmittedWorkDone();
      } catch (error) { failure = error.message; }
      finally {
        const invalid = await device.popErrorScope(); if (invalid) errors.push(invalid.message);
        handle.dispose(); for (const kernel of kernels) kernel.dispose(); renderer.dispose(); canvas.remove(); device.destroy();
      }
      if (failure) throw new Error(`${mode}: ${failure}; GPU errors: ${errors.join('; ')}`);
      return {observations, copies, allocations, errors, sourceBytes: size * 4,
        route: renderer.info.f3d?.route, nativeBackend: renderer.backend?.isF3DBackend};
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}

try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.equal(candidate.nativeBackend, true);
  assert.deepEqual(candidate.observations, source.observations);
  assert.deepEqual(candidate.copies, source.copies); assert.deepEqual(candidate.allocations, source.allocations);
  assert.deepEqual(candidate.allocations, [8, 32, 8, 0, 32]);
  assert.equal(candidate.copies.length, 7); assert.equal(candidate.copies.reduce((sum, copy) => sum + copy.size, 0), 36);
  assert.ok(candidate.copies.every(copy => copy.size <= 8));
  console.log(JSON.stringify({case: 'native_compute_readback', status: 'pass', exactComparison: true,
    sourceBytes: candidate.sourceBytes, bytesCopied: 36, readbacks: candidate.copies.length,
    reusableCapacity: 32, reductions: 3, gpuErrors: 0, route: candidate.route,
    browserVersion: candidate.browserVersion, sourcePin: pin.source_commit}));
} finally { await new Promise(resolve => server.close(resolve)); }
