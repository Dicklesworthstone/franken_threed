/** Native-call boundary only. Does not execute WGSL or certify GPU output. */
import assert from 'node:assert/strict';
export function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
export function gpuPostprocessFixture() {
  const lost = deferred();
  const calls = { textures: [], buffers: [], pipelines: [], modules: [], groups: [], writes: [], submissions: [], scopes: [], popped: [], deviceDestroyed: 0 };
  const controls = { textureFailure: 0, pipelineFailure: null, completion: null, scopeError: null, compile: null };
  let textureAttempt = 0;
  function texture(width = 16, height = 8, format = 'rgba16float', usage = 4 | 16, owned = false) {
    const result = { width, height, format, usage, dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, mipLevelCount: 1, destroyed: 0,
      createView(descriptor = {}) { assert.equal(result.destroyed, 0); return { texture: result, descriptor }; },
      destroy() { result.destroyed++; },
    };
    if (owned) calls.textures.push(result);
    return result;
  }
  const device = {
    limits: { maxTextureDimension2D: 8192, maxSampledTexturesPerShaderStage: 16 }, lost: lost.promise,
    createTexture(descriptor) {
      textureAttempt++;
      if (controls.textureFailure === textureAttempt) throw new Error('injected texture allocation failure');
      const size = descriptor.size;
      return texture(size.width ?? size[0], size.height ?? size[1], descriptor.format, descriptor.usage, true);
    },
    createBuffer(descriptor) { const value = { descriptor, destroyed: 0, destroy() { this.destroyed++; } }; calls.buffers.push(value); return value; },
    createShaderModule(descriptor) { calls.modules.push(descriptor); return descriptor; },
    createBindGroupLayout(descriptor) { return descriptor; },
    createPipelineLayout(descriptor) { return descriptor; },
    async createRenderPipelineAsync(descriptor) {
      if (controls.compile) await controls.compile.promise;
      if (controls.pipelineFailure) throw controls.pipelineFailure;
      calls.pipelines.push(descriptor); return descriptor;
    },
    createBindGroup(descriptor) { calls.groups.push(descriptor); return descriptor; },
    pushErrorScope(scope) { calls.scopes.push(scope); },
    popErrorScope() { calls.popped.push(calls.scopes.pop()); const error = controls.scopeError; controls.scopeError = null; return Promise.resolve(error); },
    createCommandEncoder() {
      const passes = [];
      return {
        beginRenderPass(descriptor) {
          const pass = { descriptor, bindings: null, pipeline: null, vertices: null, ended: false }; passes.push(pass);
          return {
            setPipeline(value) { pass.pipeline = value; },
            setBindGroup(index, value) {
              assert.equal(index, 0); pass.bindings = value;
              const output = descriptor.colorAttachments[0].view.texture;
              for (const entry of value.entries) if (entry.resource.texture) assert.notEqual(entry.resource.texture, output, 'read/write texture feedback');
            },
            draw(vertices) { pass.vertices = vertices; }, end() { pass.ended = true; },
          };
        },
        finish() { for (const pass of passes) assert.equal(pass.ended, true); return { passes }; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, bytes) { assert.equal(buffer.destroyed, 0); calls.writes.push({ buffer, offset, bytes: new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength).slice() }); },
      submit(commands) { calls.submissions.push(commands); },
      onSubmittedWorkDone() { return controls.completion?.promise ?? Promise.resolve(); },
    },
    destroy() { calls.deviceDestroyed++; },
  };
  return { device, calls, controls, texture, lost };
}
