/** Native command entry points observed by the retained r186 Inspector Timeline.
 * Every public operation executes its GPU command; resource ownership, admission,
 * validation and scheduling remain with the existing renderer/compute owners.
 */
import {encodeAnimationDrawCommand} from './animation_render_bundles.mjs';

const CONTEXT = ['currentFrame', 'currentContext', 'currentRender', 'currentCompute'];

class ThreeBackendOperations {
  constructor(renderer) { this._renderer = renderer; }

  _capture(context) {
    const backend = this, renderer = this._renderer, frame = renderer.info.frame;
    const inspector = renderer.inspector;
    const captured = Object.fromEntries(CONTEXT.filter(key => Object.hasOwn(inspector, key)).map(key => [key, inspector[key]]));
    return Object.freeze({backend, context, frame,
      run(operation) {
        // GPU preparation may finish in a later animation frame. Enter only the
        // accepted diagnostic context, without replaying Inspector/source hooks.
        // A replacement Inspector did not accept this work. Never copy the old
        // Inspector's records into it or temporarily manufacture its own fields.
        if (renderer.inspector !== inspector) return operation();
        const previousFrame = renderer.info.frame;
        const previous = Object.keys(captured).map(key => [key, Object.getOwnPropertyDescriptor(inspector, key)]);
        renderer.info.frame = frame;
        Object.assign(inspector, captured);
        try { return operation(); }
        finally {
          for (const [key, descriptor] of previous) {
            if (descriptor) Object.defineProperty(inspector, key, descriptor);
            else delete inspector[key];
          }
          renderer.info.frame = previousFrame;
        }
      },
    });
  }

  _drawObject(commands, index, instances, {instancing} = {}) {
    // Unobserved H1 draws allocate no diagnostic objects. Timeline installs an
    // own wrapper on draw(); custom backend instrumentation takes the same path.
    if (this.draw === nativeDraw) return null;
    const command = commands[index], source = command.source;
    const parameters = Object.freeze({vertexCount: command.count, instanceCount: command.instanceCount ?? instances,
      firstVertex: command.first, firstInstance: command.nodeProgram || command.program ? command.firstInstance ?? 0 :
        instancing && command.instanceCount == null ? index : 0});
    return {object: source.object, material: source.material, geometry: source.sourceGeometry ?? source.geometry,
      getDrawParameters: () => parameters,
      ...(instances > 1 ? {sourceDraws: commands.slice(index, index + instances).map(({source}) =>
        ({object: source.object, material: source.material, geometry: source.sourceGeometry ?? source.geometry}))} : {})};
  }

  beginRender(renderContext, encoder, descriptor) { return encoder.beginRenderPass(descriptor); }
  finishRender(renderContext, pass) { pass.end(); }
  draw(renderObject, encoder, command, parameters, index, instances) {
    return encodeAnimationDrawCommand(encoder, command, parameters, index, instances);
  }

  beginCompute(computeGroup, encoder, descriptor) { return encoder.beginComputePass(descriptor); }
  compute(computeGroup, computeNode, bindings, pipeline, dispatchSize, pass, workgroups) {
    pass.setPipeline(pipeline);
    for (const {index, group} of bindings) pass.setBindGroup(index, group);
    if (Array.isArray(workgroups)) pass.dispatchWorkgroups(...workgroups);
    else pass.dispatchWorkgroupsIndirect(workgroups.buffer, workgroups.offset ?? 0);
  }
  finishCompute(computeGroup, pass) { pass.end(); }

  beginBundle(renderContext, device, descriptor) { return device.createRenderBundleEncoder(descriptor); }
  finishBundle(renderContext, encoder, descriptor) { return encoder.finish(descriptor); }
  addBundle(renderContext, bundle, pass) { pass.executeBundles([bundle]); }
}

const nativeDraw = ThreeBackendOperations.prototype.draw;
export function createThreeBackendOperations(renderer) { return new ThreeBackendOperations(renderer); }
