/** Native queue/pixel regressions for CPU-authored geometry. The small ordinary
 * attribute records below isolate the GPU boundary; the Node suite separately
 * runs real pinned BufferAttributes and WebGLAttributes. No missing-GPU skip.
 */
import {createGpuBufferGeometry, bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const check = (ok, message) => { if (!ok) throw new Error(message); };
function attribute(array, itemSize) {
  return {array, itemSize, count: array.length/itemSize, version: 0, updateRanges: [],
    normalized: false, onUploadCallback() {}, clearUpdateRanges() { this.updateRanges.length=0; }};
}
export async function runBufferGeometryRenderChecks(device, {renderBundles = false} = {}) {
  const results=[], resources=[], errors=[];
  const own = x => { resources.push(x); return x; };
  const onError = event => errors.push(event.error?.message ?? String(event));
  device.addEventListener('uncapturederror', onError);
  let gpu, renderer;
  try {
    const color=own(device.createTexture({size:[64,64],format:'rgba8unorm',usage:17}));
    const depth=own(device.createTexture({size:[64,64],format:'depth32float',usage:16}));
    const colorView=color.createView(), depthView=depth.createView();
    const readback=own(device.createBuffer({size:64*256,usage:9}));
    async function image() {
      const encoder=device.createCommandEncoder();
      encoder.copyTextureToBuffer({texture:color},{buffer:readback,bytesPerRow:256},[64,64]);
      device.queue.submit([encoder.finish()]);await readback.mapAsync(1);
      try { return new Uint8Array(readback.getMappedRange()).slice(); } finally { readback.unmap(); }
    }
    function pixel(bytes, x, y, expected, label) {
      const offset=(y*64+x)*4;
      check(expected.every((v,i)=>Math.abs(bytes[offset+i]-v)<=1),`${label}: ${bytes.slice(offset,offset+4)}`);
    }
    for (const instancing of [false,true]) {
      const position=attribute(new Float32Array([-.3,-.4,.5, .3,-.4,.5, 0,.4,.5]),3);
      const colors=attribute(new Float32Array([1,0,0,1,0,0,1,0,0]),3);
      const index=attribute(new Uint16Array([0,1,2]),1);
      const geometry={attributes:{position,color:colors},index,drawRange:{start:0,count:3}};
      gpu=createGpuBufferGeometry(device,geometry);
      renderer=await createGpuAnimationRenderer(device,{instancing,renderBundles,depthFormat:'depth32float'});
      const mesh=await renderer.addMesh(gpu), frame=(draws)=>({colorView,depthView,viewProjection:identity(),draws});
      renderer.render(frame([mesh]));await renderer.whenIdle();
      pixel(await image(),32,34,[255,0,0,255],'initial RGB source stream');
      colors.array.set([0,0,1,0,0,1,0,0,1]);gpu.update();
      renderer.render(frame([mesh]));await renderer.whenIdle();
      pixel(await image(),32,34,[255,0,0,255],'CPU edits without an upload stay stale');
      colors.version++;gpu.update();renderer.render(frame([mesh]));await renderer.whenIdle();
      pixel(await image(),32,34,[0,0,255,255],'requested color upload');
      results.push(`source-requested RGB uploads (instancing=${instancing})`);

      // Submit version A, then update and submit version B WITHOUT awaiting
      // completion in between. Queue order, not encoding-time snapshots, owns
      // these histories. Uniforms and all borrowed streams must remain correct.
      colors.array.set([1,0,0,1,0,0,1,0,0]);colors.version++;gpu.update();
      const left=identity();left[12]=-.5;
      renderer.render(frame([{mesh,worldMatrix:left}]));
      colors.array.set([0,0,1,0,0,1,0,0,1]);colors.version++;gpu.update();
      const right=identity();right[12]=.5;
      renderer.render({...frame([{mesh,worldMatrix:right}]),loadOp:'load',depthLoadOp:'load'});
      await renderer.whenIdle();const both=await image();
      pixel(both,16,34,[255,0,0,255],'earlier submitted geometry version');
      pixel(both,48,34,[0,0,255,255],'later submitted geometry version');
      results.push(`two geometry versions in flight (instancing=${instancing})`);

      // An odd Uint16 write must not publish the neighboring CPU edit. Verify
      // physical bytes, including padding, through an actual mapped GPU copy.
      index.array.set([2,0,1]);index.updateRanges.push({start:1,count:1});index.version++;gpu.update();
      const buffer=bufferGeometrySnapshot(gpu,device).indexBuffer;
      const bytes=own(device.createBuffer({size:8,usage:9}));
      const copy=device.createCommandEncoder();copy.copyBufferToBuffer(buffer,0,bytes,0,8);
      device.queue.submit([copy.finish()]);await bytes.mapAsync(1);
      try { check([...new Uint16Array(bytes.getMappedRange())].join(',')==='0,0,2,0','subword upload exposed CPU neighbors'); }
      finally { bytes.unmap(); }
      results.push(`aligned index shadow readback (instancing=${instancing})`);
      geometry.drawRange.count=0;renderer.render(frame([mesh]));await renderer.whenIdle();
      pixel(await image(),32,34,[0,0,0,0],'zero draw range');
      results.push(`live draw range (instancing=${instancing})`);
      if (renderBundles) {
        check(renderer.bundleDiagnostics.builds===2,'data-only frames re-recorded bundles');
        check(renderer.bundleDiagnostics.reuses===4,'stable frames did not reuse their bundle');
        results.push(`persistent bundle reuse (instancing=${instancing})`);
      }
      check(gpu.diagnostics.allocations===3,'stable geometry allocated replacement buffers');
      renderer.dispose();renderer=null;gpu.dispose();gpu=null;
    }
    check(errors.length===0,errors.join('\n'));
    return {status:'passed',renderBundles,checks:results,execution:'actual WebGPU pixel and buffer readback',performanceClaim:false};
  } finally {
    renderer?.dispose();gpu?.dispose();
    for (const resource of resources) resource.destroy();
    device.removeEventListener('uncapturederror',onError);
  }
}
