/** Explicit WebGPU call recorder, not a WGSL executor or a pixel oracle. */
import assert from 'node:assert/strict';
export function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
export function gpuTargetMipmapFixture() {
  const textures = [], pipelines = [], samplers = [], submissions = [], groups = [], scopes = [];
  const lost = deferred(), controls = {completion: null, validation: null, pipeline: null, submit: null};
  const device = {
    limits: {maxTextureDimension2D: 8192}, lost: lost.promise,
    pushErrorScope(type) { scopes.push(type); },
    popErrorScope() { assert.ok(scopes.pop()); const error = controls.validation; controls.validation = null; return Promise.resolve(error); },
    createTexture(descriptor) {
      const texture = {descriptor, ...descriptor, ...descriptor.size, destroyed: 0,
        createView(settings = {}) {
          assert.equal(this.destroyed, 0);
          const baseMipLevel = settings.baseMipLevel ?? 0;
          const mipLevelCount = settings.mipLevelCount ?? descriptor.mipLevelCount - baseMipLevel;
          assert.ok(baseMipLevel >= 0 && mipLevelCount > 0 && baseMipLevel + mipLevelCount <= descriptor.mipLevelCount);
          return {texture: this, descriptor: {...settings, baseMipLevel, mipLevelCount}};
        },
        destroy() { this.destroyed++; },
      };
      textures.push(texture); return texture;
    },
    createShaderModule(descriptor) { return descriptor; },
    createRenderPipeline(descriptor) {
      if (controls.pipeline) throw controls.pipeline;
      const pipeline = {descriptor, getBindGroupLayout() { return {}; }};
      pipelines.push(pipeline); return pipeline;
    },
    createSampler(descriptor) { const sampler = {descriptor}; samplers.push(sampler); return sampler; },
    createBindGroup(descriptor) { groups.push(descriptor); return descriptor; },
    createCommandEncoder() {
      const passes = [];
      return {
        beginRenderPass(descriptor) {
          const output = descriptor.colorAttachments[0].view;
          assert.equal(output.descriptor.mipLevelCount, 1, 'a render attachment has one mip');
          assert.equal(output.texture.sampleCount, 1);
          const pass = {descriptor, ended: false, pipeline: null, group: null}; passes.push(pass);
          return {
            setPipeline(pipeline) { pass.pipeline = pipeline; },
            setBindGroup(index, group) {
              assert.equal(index, 0); pass.group = group;
              for (const {resource: input} of group.entries) {
                if (!input.texture) continue;
                assert.equal(input.texture.destroyed, 0);
                if (input.texture === output.texture) {
                  const a = input.descriptor, b = output.descriptor;
                  assert.ok(a.baseMipLevel + a.mipLevelCount <= b.baseMipLevel ||
                    b.baseMipLevel + b.mipLevelCount <= a.baseMipLevel, 'sample/attachment mip ranges do not overlap');
                }
              }
            },
            draw(vertices) { pass.vertices = vertices; },
            end() { pass.ended = true; },
          };
        },
        finish() { for (const pass of passes) assert.ok(pass.ended); return {passes}; },
      };
    },
    queue: {
      submit(commands) { if (controls.submit) throw controls.submit; submissions.push(commands); },
      onSubmittedWorkDone() { return controls.completion?.promise ?? Promise.resolve(); },
    },
    destroy() { assert.fail('caller-owned device'); },
  };
  return {device, textures, pipelines, samplers, submissions, groups, scopes, controls, lost};
}
